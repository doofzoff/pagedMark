"""Can one GENERIC template cover the CJK AI labels no per-vendor detector fires on?

THE OPPORTUNITY
  Corpus inspection of doubao-provenance misses turned up `千问AI生成` (Alibaba Qwen) and
  `百度 AI生成` (Baidu) sitting in the same bottom-right corner as the marks we do cover,
  bold and plainly legible, with no detector able to fire on either. `docs/...landscape`
  puts uncovered vendors at ~6% of sampled images -- larger than any tuning gain left in
  the covered ones (a dense scale ladder recovers 1 miss in 26; see ladder_headroom.py).

WHY A SHARED TEMPLATE IS EVEN PLAUSIBLE
  China's GB 45438-2025 mandates the label, and every compliant vendor ends it with the
  same run: `AI生成`. The vendor PREFIX varies (豆包 / 千问 / 百度 / 星绘) and is what makes
  per-vendor silhouettes expensive; the TAIL is guaranteed. So the tail is the part worth
  detecting, and vendor attribution -- which the shared tail cannot provide anyway
  (measured 千问-vs-doubao AUC ~0.5) -- moves to metadata, where it is reliable.

WHY IT IS WORTH RETRYING NOW
  `render_vendor_silhouettes.py` records 千问 as ruled out, and its reasoning was sound:
  the then-current front-end binarized the glyph, and a thin translucent overlay shatters
  into specks that no template can match. It closes by naming what would be needed --
  "a detection front-end that does not depend on binarizing the glyph (grayscale/edge
  correlation on the raw top-hat)". That front-end now EXISTS: `detect_frontend="tophat"`
  was built for doubao and correlates a soft template against the continuous response.
  The blocker was removed by unrelated work, so the ruling deserves re-measurement rather
  than inheritance.

THE MEASUREMENT
  Three arms, one scorer (the tail template on the tophat response, swept over a dense
  size ladder because an unregistered vendor's glyph size is genuinely unknown -- the one
  place a dense ladder earns its cost):

    covered    a registered mark already fires -- a sanity arm; the tail is inside those
               marks too, so a tail template that cannot score THESE is simply broken
    uncovered  TC260 provenance but no detector fires -- the target population
    clean      no metadata signal at all -- the precision arm

  The decision rests on whether `uncovered` carries a high-scoring subpopulation that
  `clean` does not. It does NOT rest on the raw rate: TC260 provenance does not imply a
  visible mark, so a modest fire rate on that arm is expected even if the detector is
  perfect. Any threshold this suggests must then be confirmed by eyeballing the crops it
  selects -- the script writes a contact sheet for exactly that.

DATA SAFETY
  Treat input datasets as sensitive and read-only, and keep output gitignored. The
  template is font-rendered synthetic, never cut from an input image.

    uv run python scripts/cjk_tail_probe.py --n 6000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / ".local-eval" / "originals"
OUT = REPO / ".local-eval" / "cjk-tail-probe.jsonl"
# Cached under the gitignored data dir, not in scripts/: this is a probe artifact, not a
# product asset. If the tail mark is ever registered, `render_vendor_silhouettes.py` is
# what writes the committed silhouette into src/.../assets/.
TAIL_PNG = REPO / ".local-eval" / "cjk-tail-silhouette.png"

# The tail is a fraction of a full vendor mark's width (`豆包AI生成` is ~5 CJK widths,
# `AI生成` ~3), and the prefix length differs per vendor, so the size is genuinely
# unknown here -- unlike a registered mark, whose fraction is calibrated. Hence a wide,
# dense ladder: this is the case a dense ladder is actually for.
TAIL_SCALES = tuple(round(0.30 * (1.08**i), 4) for i in range(18))  # ~0.30 .. ~1.15


def build_tail_silhouette() -> np.ndarray:
    """Font-rendered `AI生成` silhouette, cached next to this script (data-safe)."""
    if TAIL_PNG.exists():
        img = cv2.imread(str(TAIL_PNG), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
    from render_vendor_silhouettes import render

    sil = render("AI生成", width=200)
    cv2.imwrite(str(TAIL_PNG), sil)
    return sil


def tail_score(engine: Any, image: np.ndarray, sil: np.ndarray) -> tuple[float, float]:
    """Best tail correlation over the size ladder; returns (score, winning_scale)."""
    loc = engine.locate(image)
    resp = engine.tophat_response(image, loc)
    if resp is None:
        return (0.0, 0.0)
    c = engine.config
    base = engine.scale_base(image)
    ar = sil.shape[0] / max(1, sil.shape[1])
    best, best_s = 0.0, 0.0
    for s in TAIL_SCALES:
        gw = max(12, int(c.alpha_width_frac * base * s))
        gh = max(6, int(gw * ar))
        if gw >= resp.shape[1] or gh >= resp.shape[0]:
            continue
        t = cv2.resize(sil, (gw, gh), interpolation=cv2.INTER_AREA)
        v = float(cv2.matchTemplate(resp, t, cv2.TM_CCOEFF_NORMED).max())
        if v > best:
            best, best_s = v, s
    return (best, best_s)


def _one(path_str: str) -> dict[str, Any] | None:
    from pagedmark.api import visible_provenance
    from pagedmark.doubao_engine import DoubaoEngine
    from pagedmark.identify import identify
    from pagedmark.image_io import imread
    from pagedmark.watermark_registry import detect_marks

    path = Path(path_str)
    try:
        prov = visible_provenance(path)
    except Exception:
        return None
    tc260 = bool({"doubao", "jimeng"} & prov)

    if not tc260:
        try:
            if identify(path, check_visible=False).signals:
                return None  # some other provenance: neither target nor clean control
        except Exception:
            return None

    img = imread(path_str)
    if img is None or min(img.shape[:2]) < 200:
        return None
    try:
        fired = [d.key for d in detect_marks(img) if d.detected]
    except Exception:
        return None

    if fired:
        arm = "covered"
    elif tc260:
        arm = "uncovered"
    else:
        arm = "clean"

    score, scale = tail_score(DoubaoEngine(), img, build_tail_silhouette())
    return {
        "src": path.name,
        "arm": arm,
        "fired": fired,
        "tail_score": round(score, 4),
        "tail_scale": scale,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument(
        "--sheet-at", type=float, default=0.0, help="write a contact sheet of `uncovered` crops scoring >= this"
    )
    a = ap.parse_args()

    if a.report_only:
        rows = [json.loads(x) for x in a.out.read_text(encoding="utf-8").splitlines() if x.strip()]
        report(rows)
        if a.sheet_at:
            contact_sheet(rows, a.sheet_at)
        return

    build_tail_silhouette()
    pool = glob.glob(str(CORPUS / "*" / "*"))
    random.Random(19).shuffle(pool)  # noqa: S311 -- deterministic sampling, not crypto
    pool = pool[: a.n]
    print(f"scanning {len(pool)} corpus files  workers={a.workers}")
    print(f"tail ladder {TAIL_SCALES[0]} .. {TAIL_SCALES[-1]} ({len(TAIL_SCALES)} rungs)\n", flush=True)

    rows: list[dict[str, Any]] = []
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh, ProcessPoolExecutor(max_workers=a.workers) as ex:
        futures = [ex.submit(_one, p) for p in pool]
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
                print(f"  {i}/{len(pool)}  usable={len(rows)}", flush=True)
    report(rows)
    if a.sheet_at:
        contact_sheet(rows, a.sheet_at)


def report(rows: list[dict[str, Any]]) -> None:
    arms = {k: [r for r in rows if r["arm"] == k] for k in ("covered", "uncovered", "clean")}
    print(f"\n{'=' * 80}\nGENERIC CJK TAIL PROBE  ({', '.join(f'{k}={len(v)}' for k, v in arms.items())})\n{'=' * 80}")
    print("\nScore distribution of the shared `AI生成` tail on the tophat response\n")
    print(f"{'arm':11s} {'n':>6s} {'p50':>7s} {'p75':>7s} {'p90':>7s} {'p95':>7s} {'p99':>7s} {'max':>7s}")
    for k, v in arms.items():
        if not v:
            continue
        s = np.array([r["tail_score"] for r in v])
        qs = [np.percentile(s, q) for q in (50, 75, 90, 95, 99)]
        print(f"{k:11s} {len(v):6d} " + " ".join(f"{q:7.3f}" for q in qs) + f" {s.max():7.3f}")

    if arms["clean"] and arms["uncovered"]:
        clean = np.array([r["tail_score"] for r in arms["clean"]])
        unc = np.array([r["tail_score"] for r in arms["uncovered"]])
        print("\n\nOPERATING POINTS -- threshold set on the CLEAN arm, yield read on `uncovered`")
        print("`clean` fires are the cost; `uncovered` fires are candidate recoveries that")
        print("still need eyeballing, since TC260 provenance does not imply a visible mark.\n")
        print(f"{'threshold':>10s} {'clean fire':>12s} {'uncovered fire':>16s} {'candidates':>12s}")
        for q in (95, 97.5, 99, 99.5, 99.9):
            t = float(np.percentile(clean, q))
            cf = float((clean >= t).mean())
            uf = float((unc >= t).mean())
            print(f"{t:10.3f} {100 * cf:11.2f}% {100 * uf:15.1f}% {int((unc >= t).sum()):12d}")

    cov = arms["covered"]
    if cov:
        s = np.array([r["tail_score"] for r in cov])
        print(f"\n\nSANITY: on frames where a registered mark fires, the tail scores median {np.median(s):.3f}.")
        print("A tail template that cannot score these is broken, whatever it does elsewhere.")


def contact_sheet(rows: list[dict[str, Any]], thresh: float, limit: int = 30) -> None:
    """Crop the corner of the top-scoring `uncovered` frames so the fires can be judged."""
    from pagedmark.doubao_engine import DoubaoEngine
    from pagedmark.image_io import imread

    eng = DoubaoEngine()
    picks = sorted(
        (r for r in rows if r["arm"] == "uncovered" and r["tail_score"] >= thresh),
        key=lambda r: -r["tail_score"],
    )[:limit]
    tiles = []
    for r in picks:
        hits = glob.glob(str(CORPUS / "*" / r["src"]))
        if not hits:
            continue
        img = imread(hits[0])
        if img is None:
            continue
        loc = eng.locate(img)
        crop = img[loc.y : loc.y + loc.h, loc.x : loc.x + loc.w]
        if crop.size:
            tiles.append(cv2.resize(crop, (320, 96), interpolation=cv2.INTER_AREA))
    if tiles:
        dest = REPO / ".local-eval" / "cjk-tail-sheet.png"
        cv2.imwrite(str(dest), np.vstack(tiles))
        print(f"\ncontact sheet ({len(tiles)} crops, score >= {thresh:.3f}) -> {dest}")
        print("scores: " + ", ".join(f"{r['tail_score']:.2f}" for r in picks[: len(tiles)]))


if __name__ == "__main__":
    main()
