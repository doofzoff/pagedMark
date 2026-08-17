"""Benchmark harness for the visible-mark detectors.

Run this before AND after any detector change. It re-runs perception over the
hand-labelled ground truth and reports, per mark, how often a fire is correct --
with Wilson intervals, so a change inside the noise is visible as such.

    uv run python scripts/visible_eval.py                 # score current code
    uv run python scripts/visible_eval.py --save baseline # snapshot for comparison
    uv run python scripts/visible_eval.py --vs baseline   # diff against a snapshot

WHAT THIS SET CAN AND CANNOT MEASURE -- read before quoting a number:

  * PRECISION: sound. Every labelled crop is centred on the region a detector
    pointed at, so "the detector fired mark K here, was K actually there" is
    exactly the question the labels answer.
  * RECALL: NOT measurable here, and the harness refuses to print it. The labelled
    images were SAMPLED WHERE DETECTORS FIRED (relaxation additions plus controls),
    so images carrying a mark that every detector missed are absent by construction.
    Computing recall on this set would divide by a denominator that excludes exactly
    the failures recall is meant to expose, and would report a flattering number.
    Recall needs a RANDOM corpus sample laballed exhaustively -- a separate round.

  * `other_ai_label` (千问 / 百度 / 星绘 / 抖音) counts as a FALSE fire for any
    registered mark, because it is a different vendor's label. It is tracked
    separately in the confusion output since it is the dominant jimeng failure.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pagedmark import watermark_registry as wr
from pagedmark.image_io import imread

GT = Path(".local-eval/textmark-relaxation/groundtruth.jsonl")
SNAP = Path(".local-eval/textmark-relaxation/snapshots")
MARKS = ("gemini", "doubao", "jimeng", "samsung", "jimeng_pill")


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def provenance_for(rec: dict) -> frozenset[str]:
    """The provenance production would have: read from the file's own METADATA.

    Never derive this from the labels. A relaxation arm only fires when provenance
    names the vendor, so label-derived provenance silently tells the detector the
    answer and every arm scores near-perfect (observed: gemini 99% instead of 34%).
    """
    return frozenset(rec.get("provenance", []))


def score(sensitivity: str = "auto") -> dict:
    recs = [json.loads(line) for line in GT.open()]
    per: dict[str, Counter] = {m: Counter() for m in MARKS}
    confusion: dict[str, Counter] = {m: Counter() for m in MARKS}
    missing = 0
    for rec in recs:
        img = imread(rec["path"])
        if img is None:
            missing += 1
            continue
        cands = wr._build_candidates(img)
        ctx = wr.Context(sensitivity=sensitivity, provenance=provenance_for(rec))
        fired = {d.candidate.key for d in wr.decide(cands, ctx)}
        for m in MARKS:
            # Only score a mark on images whose shown crop could rule on it. Scoring
            # outside that scope books real detections as false fires -- see the
            # adjudication note in visible_groundtruth.py.
            if m not in rec.get("adjudicated", []):
                continue
            if m not in fired:
                per[m]["fn"] += 1 if m in rec["present"] else 0
                continue
            if m in rec["present"]:
                per[m]["tp"] += 1
            else:
                per[m]["fp"] += 1
                for s in rec["seen"]:
                    confusion[m][s] += 1
    scope = {m: sum(1 for r in recs if m in r.get("adjudicated", [])) for m in MARKS}
    return {
        "per": {m: dict(c) for m, c in per.items()},
        "scope": scope,
        "confusion": {m: dict(c) for m, c in confusion.items()},
        "missing": missing,
        "n": len(recs),
        "sensitivity": sensitivity,
    }


def report(res: dict, prev: dict | None = None) -> None:
    print(f"\nground truth: {res['n']} images ({res['missing']} unreadable)   sensitivity={res['sensitivity']}")
    print("=" * 78)
    print(
        f"{'mark':12s} {'scope':>6s} {'fires':>6s} {'correct':>8s} "
        f"{'precision':>11s} {'95% CI':>13s}  {'missed':>7s}  delta"
    )
    print("-" * 78)
    for m in MARKS:
        c = res["per"][m]
        tp, fp, fn = c.get("tp", 0), c.get("fp", 0), c.get("fn", 0)
        n = tp + fp
        scope = res["scope"].get(m, 0)
        if n == 0:
            print(f"{m:12s} {scope:6d} {0:6d} {'-':>8s} {'-':>11s} {'-':>13s} {fn:7d}")
            continue
        lo, hi = wilson(tp, n)
        delta = ""
        if prev:
            pc = prev["per"][m]
            pn = pc.get("tp", 0) + pc.get("fp", 0)
            if pn:
                d = tp / n - pc.get("tp", 0) / pn
                delta = f"{d:+.1%} (fires {pn}->{n})"
        print(f"{m:12s} {scope:6d} {n:6d} {tp:8d} {tp / n:10.0%} {lo:5.0%}-{hi:<6.0%} {fn:7d}  {delta}")
    print("\nwhat the FALSE fires actually were:")
    for m in MARKS:
        conf = {k: v for k, v in res["confusion"][m].items() if k != m}
        if conf:
            print(f"  {m:12s} {dict(sorted(conf.items(), key=lambda kv: -kv[1]))}")
    print("\n'scope' = images whose crop could rule on that mark; 'missed' = labelled marks it did not fire on.")
    print("NOTE: 'missed' is NOT recall -- this set was sampled where detectors fired, so images")
    print("      every detector missed are absent by construction. Use it only to catch a change")
    print("      LOSING marks it used to find; an unbiased random sample is needed for true recall.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", metavar="NAME")
    ap.add_argument("--vs", metavar="NAME")
    ap.add_argument("--sensitivity", default="auto")
    a = ap.parse_args()
    res = score(a.sensitivity)
    prev = None
    if a.vs:
        f = SNAP / f"{a.vs}.json"
        if f.exists():
            prev = json.loads(f.read_text())
        else:
            print(f"(no snapshot {f}; showing absolute numbers)")
    report(res, prev)
    if a.save:
        SNAP.mkdir(parents=True, exist_ok=True)
        (SNAP / f"{a.save}.json").write_text(json.dumps(res, indent=1))
        print(f"\nsaved snapshot -> {SNAP / f'{a.save}.json'}")


if __name__ == "__main__":
    main()
