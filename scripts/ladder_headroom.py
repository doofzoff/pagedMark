"""How much recall is the coarse scale ladder costing, and what would a denser one cost?

THE FINDING THIS MEASURES
  `_ladder_best` sweeps three rungs -- (0.8, 1.0, 1.25) -- and the `binary` front-end
  sweeps none at all. Measured on stamped marks over controlled backgrounds
  (`scripts/detector_response.py`), the response is a COMB: doubao scores 0.99 exactly at
  each rung and collapses to 0.37-0.48 between them, against a 0.50 gate. So a mark whose
  rendered size lands mid-gap is missed at FULL contrast. Jimeng (binary, one nominal
  size) holds a single 0.90-1.20 lobe; samsung only 0.95-1.05.

WHY A SYNTHETIC SWEEP IS NOT ENOUGH TO JUSTIFY A FIX
  Dead zones only cost recall if real marks land in them. The fractions were calibrated on
  real captures, so it is entirely possible that real marks cluster at ratio 1.0 and the
  gaps are never visited. That question is not answerable by stamping -- it needs the
  corpus.

THE MEASUREMENT
  Positives are images carrying INDEPENDENT vendor provenance (TC260 / C2PA metadata
  naming the vendor), which is evidence that does not come from the pixel detector we are
  grading -- the same discriminator that settled the pill gate. For every such image the
  detector currently MISSES, rescore at a dense ladder and ask whether it would now cross
  the gate, and at which scale.

  Negatives are images with NO metadata signal at all. That is deliberately NOT "and no
  mark was detected": defining the set by the detector's own verdict would make its
  false-fire rate 0 by construction and the comparison vacuous. The cost is that a
  metadata-stripped screenshot of a marked image sits in the negative set and its correct
  detection is counted as a false fire -- but that impurity is identical under both
  ladders, so the DELTA between them, which is what the fix is judged on, stays sound.

  Two more impurities to keep in view when reading the positive arm. Provenance says the
  vendor produced the file, not that a visible mark is on it. And the TC260 label names no
  specific vendor, so `visible_provenance` maps it to BOTH doubao and jimeng -- a "doubao
  positive" may carry a jimeng mark or none. Both inflate the miss count, so the recovery
  percentage is an upper bound on what a denser ladder buys.

DATA SAFETY
  Treat input datasets as sensitive and read-only, and keep output gitignored.

    uv run python scripts/ladder_headroom.py --mark doubao --n 4000
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / ".local-eval" / "originals"
OUT = REPO / ".local-eval" / "ladder-headroom.jsonl"

# The rungs the product ships today, and the dense ladder under evaluation. The dense one
# is geometric with a ~6% step, chosen from the measured half-width of a rung's lobe
# (doubao holds >=0.5 for about +/-6% around each peak), so no gap is left uncovered.
SHIPPED = (0.8, 1.0, 1.25)
LADDERS: dict[str, tuple[float, ...]] = {
    # A geometric ~6% step across the whole plausible range. Measures the CEILING of what
    # any ladder change can buy, and the worst case of what it costs in false fires.
    "dense": tuple(round(0.70 * (1.06**i), 4) for i in range(13)),  # 0.70 .. ~1.49
    # The targeted alternative. On the full run, 26 of 28 recoveries came from ONE rung
    # (~1.116) and 22 of 28 were LANDSCAPE frames -- that is a geometry gap, not a density
    # gap, so the honest comparison is one extra rung against thirteen.
    "plus_one": (0.8, 1.0, 1.1157, 1.25),
    "shipped": SHIPPED,
}
DENSE = LADDERS["dense"]
# Score every rung any candidate ladder might use and store them all, so a new candidate
# is evaluated by re-reading the JSONL instead of re-running the hour-long sweep.
PROBE_SCALES: tuple[float, ...] = tuple(sorted({s for rungs in LADDERS.values() for s in rungs}))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - hw), 100 * (c + hw))


def score_at_scales(engine: Any, image: np.ndarray, scales: tuple[float, ...]) -> dict[float, float]:
    """`_ladder_best`'s inner loop, opened up so the ladder is a parameter.

    A DELIBERATE second copy of the sweep: it exists to measure ladders the shipped
    engine does not have, so it must not be folded back into `_ladder_best`.

    Deliberately reaches into the engine (`tophat_response`, `_glyph_silhouette`): a
    measurement script may, product code may not. Kept a faithful copy of the shipped
    scoring -- if it drifts, the numbers below stop describing the product.
    """
    c = engine.config
    loc = engine.locate(image)
    resp = engine.tophat_response(image, loc)
    sil = engine._glyph_silhouette()
    if resp is None or sil is None:
        return {}
    base = engine.scale_base(image)
    out: dict[float, float] = {}
    for scale in scales:
        gw = max(c.min_gw, int(c.alpha_width_frac * base * scale))
        gh = max(4, int(c.alpha_height_frac * base * scale))
        if gw >= resp.shape[1] or gh >= resp.shape[0]:
            continue
        tmpl = cv2.resize(sil, (gw, gh), interpolation=cv2.INTER_AREA).astype(np.float32)
        if c.template_blur > 0:
            tmpl = cv2.GaussianBlur(tmpl, (0, 0), sigmaX=c.template_blur, sigmaY=c.template_blur)
        out[scale] = float(cv2.matchTemplate(resp, tmpl.astype(np.uint8), cv2.TM_CCOEFF_NORMED).max())
    return out


def _one(args: tuple[str, str]) -> dict[str, Any] | None:
    path_str, mark_key = args
    from pagedmark.api import visible_provenance
    from pagedmark.identify import identify
    from pagedmark.image_io import imread
    from pagedmark.watermark_registry import get_mark

    path = Path(path_str)
    try:
        prov = visible_provenance(path)
    except Exception:
        return None
    positive = mark_key in prov

    if not positive:
        # A negative is a frame with NO metadata signal at all -- evidence independent of
        # the detector being graded. Do NOT also require that no mark was detected: that
        # defines the negative set by the very verdict under test, and "false fire today"
        # would then be 0 by construction on every run.
        try:
            if identify(path, check_visible=False).signals:
                return None
        except Exception:
            return None

    img = imread(path_str)
    if img is None or min(img.shape[:2]) < 200:
        return None

    mark = get_mark(mark_key)
    det = mark.detect(img)

    import importlib

    mod = importlib.import_module(f"pagedmark.{mark_key}_engine")
    cls = next(
        o for n, o in vars(mod).items() if isinstance(o, type) and n.endswith("Engine") and n != "TextMarkEngine"
    )
    engine = cls()
    scores = score_at_scales(engine, img, PROBE_SCALES)
    if not scores:
        return None
    gate = engine.config.detect_ncc_threshold
    dense = {s: v for s, v in scores.items() if s in LADDERS["dense"]}
    best_scale = max(dense, key=lambda s: dense[s]) if dense else 0.0
    h, w = img.shape[:2]
    return {
        "src": path.name,
        "arm": "positive" if positive else "negative",
        "detected_now": bool(det.detected),
        "conf_now": round(float(det.confidence), 4),
        "gate": gate,
        # Every rung, so a new candidate ladder is evaluated by re-reading this file.
        "scores": {str(s): round(v, 4) for s, v in scores.items()},
        "dense_best": round(dense[best_scale], 4) if dense else 0.0,
        "dense_best_scale": best_scale,
        "dense_crosses": bool(dense) and dense[best_scale] >= gate,
        "aspect": "portrait" if w / h < 0.9 else ("landscape" if w / h > 1.1 else "square"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    # doubao only, and that is not laziness. `score_at_scales` reproduces the TOPHAT
    # front-end, which is the only one with a ladder to widen; scoring it for jimeng or
    # samsung would measure a code path their detectors never run. Doubao is also the one
    # mark that declares NO rival, so crossing its NCC gate really is the whole verdict --
    # for a mark with rivals, `dense_crosses` would ignore the competitive margin and read
    # optimistically. Widening the binary front-end is a separate experiment.
    ap.add_argument("--mark", default="doubao", choices=["doubao"])
    ap.add_argument("--n", type=int, default=4000, help="local files to scan")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()

    out = a.out.with_name(f"{a.out.stem}_{a.mark}{a.out.suffix}")
    if a.report_only:
        report([json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()], a.mark)
        return

    pool = glob.glob(str(CORPUS / "*" / "*"))
    random.Random(19).shuffle(pool)  # noqa: S311 -- deterministic sampling, not crypto
    pool = pool[: a.n]
    print(f"mark={a.mark}  scanning {len(pool)} corpus files  workers={a.workers}")
    print(f"shipped ladder {SHIPPED}\ndense ladder   {DENSE}\n", flush=True)

    rows: list[dict[str, Any]] = []
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh, ProcessPoolExecutor(max_workers=a.workers) as ex:
        futures = [ex.submit(_one, (p, a.mark)) for p in pool]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                rec = fut.result()
            except Exception:  # noqa: S112 -- one bad file must not kill the sweep
                continue
            if rec is None:
                continue
            fh.write(json.dumps(rec) + "\n")
            rows.append(rec)
            if i % 500 == 0:
                fh.flush()
                pos = sum(1 for r in rows if r["arm"] == "positive")
                print(f"  {i}/{len(pool)}  usable={len(rows)} (pos={pos})", flush=True)
    report(rows, a.mark)


def report(rows: list[dict[str, Any]], mark: str) -> None:
    pos = [r for r in rows if r["arm"] == "positive"]
    neg = [r for r in rows if r["arm"] == "negative"]
    print(f"\n{'=' * 82}\nLADDER HEADROOM  mark={mark}   positives={len(pos)}  verified-clean negatives={len(neg)}")
    print(f"{'=' * 82}")
    if not pos and not neg:
        print("no usable rows")
        return

    if pos:
        miss = [r for r in pos if not r["detected_now"]]
        rec = [r for r in miss if r["dense_crosses"]]
        now = len(pos) - len(miss)
        lo, hi = wilson(len(rec), len(miss)) if miss else (0.0, 0.0)
        print("\nPOSITIVE ARM  (images whose METADATA names this vendor -- an upper bound,")
        print("provenance says the vendor produced the file, not that a mark is visible on it)\n")
        print(f"  detected today            {now:5d} / {len(pos)}  ({100 * now / len(pos):.1f}%)")
        print(f"  of the {len(miss)} misses, the dense ladder crosses the gate on {len(rec)}")
        if miss:
            print(f"                            ({100 * len(rec) / len(miss):.1f}%, 95% CI {lo:.1f}-{hi:.1f})")
            after = now + len(rec)
            print(f"  detection would go        {100 * now / len(pos):.1f}%  ->  {100 * after / len(pos):.1f}%")
        if rec:
            by_scale = collections.Counter(r["dense_best_scale"] for r in rec)
            print("\n  which rung recovers them (a rung near a SHIPPED one recovering many means")
            print("  the gain is the finer STEP, not the wider RANGE):")
            for s, n in sorted(by_scale.items()):
                near = "  <- shipped" if any(abs(s - x) < 0.02 for x in SHIPPED) else ""
                print(f"    scale {s:<6} {n:4d}{near}")
            by_asp = collections.Counter(r["aspect"] for r in rec)
            print(f"\n  by aspect: {dict(by_asp)}")

    if neg:
        fires_now = sum(1 for r in neg if r["detected_now"])
        fires_dense = sum(1 for r in neg if r["dense_crosses"])
        lo0, hi0 = wilson(fires_now, len(neg))
        lo1, hi1 = wilson(fires_dense, len(neg))
        print("\n\nNEGATIVE ARM  (no metadata signal; NOT filtered on the detector's own verdict)\n")
        p0, p1 = 100 * fires_now / len(neg), 100 * fires_dense / len(neg)
        print(f"  false fire today          {fires_now:4d} / {len(neg)}  {p0:.2f}%  (CI {lo0:.2f}-{hi0:.2f})")
        print(f"  false fire, dense ladder  {fires_dense:4d} / {len(neg)}  {p1:.2f}%  (CI {lo1:.2f}-{hi1:.2f})")
        print("\n  This is the price. A denser ladder gives a spurious blob more chances to")
        print("  match at SOME scale, so the gain above is only real if this line barely moves.")

    ladder_table(pos, neg)


def _crosses(row: dict[str, Any], rungs: tuple[float, ...]) -> bool:
    """Would this frame cross its gate on the given ladder, from the stored per-rung scores."""
    sc = row.get("scores") or {}
    gate = row["gate"]
    return any(v >= gate for k, v in sc.items() if float(k) in rungs)


def ladder_table(pos: list[dict[str, Any]], neg: list[dict[str, Any]]) -> None:
    """Every candidate ladder side by side: what it recovers against what it costs."""
    if not (pos and neg) or "scores" not in pos[0]:
        return  # older run without per-rung scores
    miss = [r for r in pos if not r["detected_now"]]
    base_fire = sum(1 for r in neg if r["detected_now"])
    print("\n\nLADDER CANDIDATES -- recovered marks against the false fires they cost\n")
    print("A candidate is only worth shipping if the recovered column beats the added-fires")
    print("column by enough to survive the base rates: clean frames vastly outnumber marked")
    print("ones in real traffic, so a small percentage on the negative arm is a large count.\n")
    hdr = f"{'ladder':10s} {'rungs':>6s} {'recovered':>10s} {'of misses':>10s}"
    print(f"{hdr} {'false fire':>11s} {'added':>7s} {'ratio':>7s}")
    for name, rungs in LADDERS.items():
        rec = sum(1 for r in miss if _crosses(r, rungs))
        fire = sum(1 for r in neg if _crosses(r, rungs) or r["detected_now"])
        added = fire - base_fire
        ratio = f"{rec / added:.1f}:1" if added > 0 else ("inf" if rec else "-")
        pct = 100 * rec / len(miss) if miss else 0.0
        print(f"{name:10s} {len(rungs):6d} {rec:10d} {pct:9.1f}% {100 * fire / len(neg):10.2f}% {added:7d} {ratio:>7s}")


if __name__ == "__main__":
    main()
