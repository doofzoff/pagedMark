"""Tier B2: detector response curves -- recall vs mark size, contrast, background and aspect.

WHY THIS EXISTS
  Every recall number the project quotes rests on found positives: doubao 89% on n=240,
  jimeng 71% on n=14, the pill 50% on n=6. Those tell you how the detectors do on the
  marks a corpus happened to contain, and nothing about WHERE they fall over. Two bugs
  have now shipped in exactly that blind spot:

    * `scale_basis` -- doubao detected 0 of 435 LANDSCAPE marks, a 100% miss, because
      every fraction was calibrated on portrait captures where width == short side.
    * the front-end mismatch -- detection moved to the continuous `tophat` response while
      the removal mask still came from the BINARIZED blob, so ~8% of doubao detections
      produced an empty mask and removal was a silent no-op.

  Both are geometry/plumbing failures at the edge of the operating range, and both were
  invisible to a corpus sweep because the corpus is concentrated in the middle of it.
  This harness constructs the edges instead of waiting for them to turn up.

THE AGGREGATE HERE IS NOT RECALL
  The grid deliberately visits sizes and opacities the engines were never calibrated for,
  so a mark counted as missed at size=0.6 is not evidence of a weak detector -- it is the
  point of the sweep. Quote the NOMINAL cell (size=1.0, alpha=1.0) as the reference, and
  read the per-axis tables for shape. Production recall comes from an unbiased corpus
  sample (`scripts/visible_recall_sample.py`), never from here.

WHAT IT MEASURES -- TWO NUMBERS, NOT ONE
  `detected`  the detector fires.
  `maskable`  the same call path then yields a NON-EMPTY removal mask.

  The second is the one that has no natural home anywhere else. A mark that is detected
  but not maskable is reported by `identify` and silently skipped by `visible` -- the user
  sees a mark the tool says it removed. Anything that measures only detection scores that
  bug as a success. They are reported side by side and their GAP is the headline.

THE CONSTRUCTION
  Take a verified-clean corpus image, stamp a real mark onto it with the forward model the
  reverse-alpha work established (`stamped = (1-a)*bg + a*white`), and sweep two knobs the
  engine's own geometry normally pins:

    size    multiplies the glyph box the engine would use (a re-rendered or rescaled mark)
    alpha   multiplies the captured opacity (bold stamp -> faint translucent overlay)

  Background texture and frame aspect are not swept -- they come from the corpus and are
  recorded, so the grid is crossed with real backgrounds rather than synthetic flats.

WHY EVERY SOURCE ALSO RUNS UNSTAMPED
  A recall curve with no false-fire baseline is uninterpretable in the same way a fill
  PSNR with no damage baseline is: "detected on 70% of faint marks" is a triumph or a
  scandal depending on whether the detector also fires on 70% of the clean frames those
  marks were stamped onto. The control runs the identical call path on the identical
  image with no stamp.

READING `mask_hit`
  Fraction of the stamped glyph box the removal mask covers -- real localization accuracy
  for the text marks and gemini, because the mask has to be placed from a detection.
  For `jimeng_pill` it is CONSTRUCTED and means nothing: the pill's footprint is a fixed
  top-left geometry box, so it covers the stamp by definition. Reported, and flagged.

DATA SAFETY
  Treat input datasets as sensitive and read-only. Keep generated reports under
  .local-eval/. Reports record source filenames and measurements, never image content.

    uv run python scripts/detector_response.py --n 12          # trial, measures throughput
    uv run python scripts/detector_response.py --n 150         # the real run, resumable
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from fill_quality import SLOT_STAMPABLE, STAMPABLE, clean_sources, stamp_any, texture_of

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / ".local-eval" / "detector-response.jsonl"

# 1.0 = the geometry/opacity the engine's own constants assume. The sweep reaches below
# it (a mark rendered smaller, or a faint translucent overlay -- the class the tophat
# front-end was introduced for) and above it (a larger re-render, or an upscaled upload).
SIZES = (0.6, 0.8, 1.0, 1.3)
ALPHAS = (0.3, 0.5, 0.75, 1.0)
MARKS = (*STAMPABLE, *SLOT_STAMPABLE)


def aspect_of(image: np.ndarray) -> str:
    h, w = image.shape[:2]
    r = w / max(1, h)
    return "portrait" if r < 0.9 else ("landscape" if r > 1.1 else "square")


def _probe(image: np.ndarray, mark_key: str, box: tuple[int, int, int, int] | None) -> dict[str, Any]:
    """Run the product detect -> localize path once and score it.

    ``box`` is the stamped glyph box, or None for the unstamped control.
    """
    from pagedmark.watermark_registry import get_mark

    mark = get_mark(mark_key)
    loc = mark.localize(image, force=False)
    mask_px = 0 if loc.mask is None else int(np.count_nonzero(loc.mask))
    hit = None
    if box is not None and loc.mask is not None:
        x, y, w, h = box
        inside = loc.mask[y : y + h, x : x + w]
        hit = round(float(np.count_nonzero(inside)) / max(1, w * h), 3)
    return {
        "detected": bool(loc.detected),
        "confidence": round(float(loc.confidence), 4),
        "maskable": mask_px > 0,
        "mask_px": mask_px,
        "mask_hit": hit,
    }


def _one_source(path_str: str) -> list[dict[str, Any]]:
    """Every (mark, size, alpha) cell for one clean background, plus its controls."""
    from pagedmark.image_io import imread

    base = imread(path_str)
    if base is None:
        return []
    name = Path(path_str).name
    asp = aspect_of(base)
    h, w = base.shape[:2]
    rows: list[dict[str, Any]] = []

    for key in MARKS:
        common = {"src": name, "mark": key, "aspect": asp, "px": w * h}
        # Control first: the same call path on the same frame with nothing stamped.
        rows.append({**common, "cell": "control", "size": None, "alpha": None, **_probe(base, key, None)})
        for size in SIZES:
            for alpha in ALPHAS:
                st = stamp_any(base, key, size_mult=size, alpha_mult=alpha)
                if st is None:
                    continue
                stamped, box = st
                # Texture of the region the mark sits on, from the CLEAN frame -- the
                # same median-Sobel proxy the pill's flatness gate uses.
                x, y, gw, gh = box
                rows.append(
                    {
                        **common,
                        "cell": "stamped",
                        "size": size,
                        "alpha": alpha,
                        "texture": round(texture_of(base[y : y + gh, x : x + gw]), 2),
                        **_probe(stamped, key, box),
                    }
                )
    return rows


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - hw), 100 * (c + hw))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="verified-clean source images")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--report-only", action="store_true", help="re-read --out and print the report")
    a = ap.parse_args()

    if a.report_only:
        rows = [json.loads(line) for line in a.out.read_text(encoding="utf-8").splitlines() if line.strip()]
        report(rows)
        return

    if a.restart and a.out.exists():
        a.out.unlink()
    done: set[str] = set()
    if a.out.exists():
        with open(a.out, encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["src"])
                except Exception:  # noqa: S112 -- tolerate a torn last line
                    continue

    print(f"selecting {a.n} verified-clean sources (already measured: {len(done)})...", flush=True)
    sources = [p for p in clean_sources(a.n) if p.name not in done]
    print(f"to do {len(sources)}  workers {a.workers}  grid {len(MARKS)}x{len(SIZES)}x{len(ALPHAS)}\n", flush=True)

    rows: list[dict[str, Any]] = []
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fh, ProcessPoolExecutor(max_workers=a.workers) as ex:
        futures = {ex.submit(_one_source, str(p)): p for p in sources}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                got = fut.result()
            except Exception as e:  # a crashed source must not kill the sweep
                print(f"  source failed: {type(e).__name__}: {e}", flush=True)
                continue
            for r in got:
                fh.write(json.dumps(r) + "\n")
            rows += got
            if i % 10 == 0:
                fh.flush()
                print(f"  {i}/{len(sources)}", flush=True)

    if done:  # merge the resumed rows so the report covers the whole file
        rows = [json.loads(line) for line in a.out.read_text(encoding="utf-8").splitlines() if line.strip()]
    report(rows)


def _rate(rs: list[dict[str, Any]], field: str) -> tuple[float, float, float, int]:
    k, n = sum(bool(r[field]) for r in rs), len(rs)
    lo, hi = wilson(k, n)
    return (100 * k / n if n else 0.0, lo, hi, n)


def report(rows: list[dict[str, Any]]) -> None:
    stamped = [r for r in rows if r["cell"] == "stamped"]
    control = [r for r in rows if r["cell"] == "control"]
    if not stamped:
        print("no measurements")
        return
    srcs = len({r["src"] for r in rows})
    print(f"\n{'=' * 84}\nDETECTOR RESPONSE  {len(stamped)} stamped cells over {srcs} clean backgrounds\n{'=' * 84}")

    # Lead with the nominal cell. The grid deliberately includes sizes and opacities the
    # engines were never calibrated for, so an aggregate over the whole grid is an
    # adversarial score, NOT production recall -- quoting it as recall would understate
    # the detectors badly. The nominal cell is the honest reference point: the mark at the
    # exact geometry and opacity the engine assumes, on a real clean background.
    print("\nNOMINAL CELL (size=1.0, alpha=1.0) -- a mark exactly as the engine expects it.")
    print("Read every other number against THIS, not as production recall: the grid is")
    print("adversarial by design and its aggregate is not a recall figure.\n")
    print(f"{'mark':13s} {'n':>5s} {'detected':>10s} {'maskable':>10s}")
    for key in MARKS:
        rs = [r for r in stamped if r["mark"] == key and r["size"] == 1.0 and r["alpha"] == 1.0]
        if rs:
            print(f"{key:13s} {len(rs):5d} {_rate(rs, 'detected')[0]:9.1f}% {_rate(rs, 'maskable')[0]:9.1f}%")

    print("\n\nFALSE FIRE on the unstamped controls (the baseline every recall below is read against)\n")
    print(f"{'mark':13s} {'n':>5s} {'fires':>8s} {'95% CI':>16s}")
    for key in MARKS:
        rs = [r for r in control if r["mark"] == key]
        if rs:
            pct, lo, hi, n = _rate(rs, "detected")
            print(f"{key:13s} {n:5d} {pct:7.1f}% {f'{lo:.1f}-{hi:.1f}':>16s}")

    print("\n\nDETECTED vs MASKABLE by mark. The GAP is the silent-no-op rate:")
    print("a mark counted here as detected-but-not-maskable is reported by `identify` and")
    print("skipped by `visible`, so the user is told it was removed when it was not.\n")
    print(f"{'mark':13s} {'n':>6s} {'detected':>10s} {'maskable':>10s} {'gap':>7s} {'mask_hit':>9s}")
    for key in MARKS:
        rs = [r for r in stamped if r["mark"] == key]
        if not rs:
            continue
        det = [r for r in rs if r["detected"]]
        d, m = _rate(rs, "detected")[0], _rate(rs, "maskable")[0]
        gap = (100 * sum(1 for r in det if not r["maskable"]) / len(det)) if det else 0.0
        hits = [r["mask_hit"] for r in det if r["mask_hit"] is not None]
        note = "  (constructed)" if key == "jimeng_pill" else ""
        hv = f"{float(np.median(hits)):.2f}" if hits else "-"
        print(f"{key:13s} {len(rs):6d} {d:9.1f}% {m:9.1f}% {gap:6.1f}% {hv:>9s}{note}")

    for axis, label in (("alpha", "CONTRAST (alpha multiplier)"), ("size", "SIZE (glyph box multiplier)")):
        print(f"\n\nRECALL BY {label}  -- 1.0 is the geometry/opacity the engine assumes\n")
        vals = sorted({r[axis] for r in stamped})
        print(f"{'mark':13s}" + "".join(f"{v:>12}" for v in vals))
        for key in MARKS:
            cells = []
            for v in vals:
                rs = [r for r in stamped if r["mark"] == key and r[axis] == v]
                cells.append(f"{_rate(rs, 'detected')[0]:10.0f}%" if rs else f"{'-':>11s}")
            print(f"{key:13s}" + "".join(f"{c:>12}" for c in cells))

    print("\n\nRECALL BY FRAME ASPECT  -- the axis the `scale_basis` bug lived on\n")
    asps = ["portrait", "square", "landscape"]
    print(f"{'mark':13s}" + "".join(f"{x:>14s}" for x in asps))
    for key in MARKS:
        cells = []
        for asp in asps:
            rs = [r for r in stamped if r["mark"] == key and r["aspect"] == asp]
            cells.append(f"{_rate(rs, 'detected')[0]:.0f}% (n={len(rs)})" if rs else "-")
        print(f"{key:13s}" + "".join(f"{c:>14s}" for c in cells))

    tex = sorted(r["texture"] for r in stamped if "texture" in r)
    if tex:
        t1, t2 = tex[len(tex) // 3], tex[2 * len(tex) // 3]
        print(f"\n\nRECALL BY BACKGROUND TEXTURE  (terciles at {t1:.1f} / {t2:.1f} median-Sobel)\n")
        print(f"{'mark':13s}" + "".join(f"{x:>14s}" for x in ("flat", "mid", "textured")))
        for key in MARKS:
            cells = []
            for lo, hi in ((-1, t1), (t1, t2), (t2, 1e9)):
                rs = [r for r in stamped if r["mark"] == key and lo < r.get("texture", -2) <= hi]
                cells.append(f"{_rate(rs, 'detected')[0]:.0f}% (n={len(rs)})" if rs else "-")
            print(f"{key:13s}" + "".join(f"{c:>14s}" for c in cells))

    worst = collections.Counter((r["mark"], r["size"], r["alpha"]) for r in stamped if not r["detected"]).most_common(8)
    if worst:
        print("\n\nWORST CELLS (most misses)\n")
        for (mark, size, alpha), n in worst:
            print(f"  {mark:13s} size={size}  alpha={alpha}   {n} misses")


if __name__ == "__main__":
    main()
