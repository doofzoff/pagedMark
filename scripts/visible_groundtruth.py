"""Consolidate the hand-labelled contact-sheet rounds into ONE ground-truth file.

Ground truth is `uid -> the set of visible marks actually present`, hand-labelled
blind against contact sheets with a two-sided control in every round. Rounds so far:

  2026-07-18 text-mark/pill round : 423 cells (doubao / jimeng / jimeng_pill arms)
  2026-07-18 gemini round         : 356 cells (gemini relaxation additions)

DATA SAFETY: treat the input dataset as sensitive. This script reads a gitignored
dataset and writes a gitignored ground-truth file. Neither the images nor this
output may be committed; only the harness is. See the repo CLAUDE.md.

The labels record what the LABELLER SAW in the crop, one of:
  doubao | jimeng | pill | sparkle | other_ai_label | none | uncertain
`other_ai_label` is a real visible AI label from a vendor we do NOT have a mark for
(千问 / 百度 / 星绘 / 抖音); it is NOT a positive for any registered mark, but it is
also not "clean" -- it is exactly what the relaxed jimeng detector confuses.
`uncertain` rows are EXCLUDED from scoring rather than coerced, so a labeller's
honest doubt never becomes a fabricated data point.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

SEEN_TO_MARK = {
    "doubao": "doubao",
    "jimeng": "jimeng",
    "pill": "jimeng_pill",
    "sparkle": "gemini",
}
# Which marks a crop centred on `key` lets the labeller rule on (same corner = visible
# in the same crop). Doubao and Jimeng share the bottom-right corner.
_ADJUDICATES = {
    "doubao": ("doubao", "jimeng"),
    "jimeng": ("doubao", "jimeng"),
    "jimeng_pill": ("jimeng_pill",),
    "gemini": ("gemini",),
    "samsung": ("samsung",),
}

ROUNDS = [
    ("textmark", "labels.csv", "manifest.csv"),
    ("gemini", "gemini_labels.csv", "gemini_manifest.csv"),
]


def metadata_provenance(path: str) -> list[str]:
    """The vendor keys LOCAL METADATA confirms -- what cli._visible_provenance reads.

    Must come from the file's own metadata, never from the labels: deriving it from
    the ground truth would hand the detector the answer it is being scored on (a
    relaxation only fires when provenance names the vendor, so label-derived
    provenance makes every arm look near-perfect). Read from the corpus `identify`
    sidecar, which is the same signal production computes.
    """
    p = Path(path)
    sidecar = Path(str(p.parent).replace("/originals/", "/identify/")) / (p.name.split("_src")[0] + ".json")
    if not sidecar.exists():
        return []
    try:
        with sidecar.open() as fh:
            wm = " | ".join(json.load(fh).get("watermarks", []))
    except Exception:
        return []
    keys: list[str] = []
    if "China AIGC label" in wm:
        keys += ["doubao", "jimeng"]
    if "C2PA Content Credentials (Google LLC" in wm:
        keys.append("gemini")
    if "Samsung Galaxy AI" in wm:
        keys.append("samsung")
    return keys


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".local-eval/textmark-relaxation")
    out = root / "groundtruth.jsonl"
    rows: dict[str, dict] = {}
    stats: dict[str, int] = {}
    for round_name, labels_file, manifest_file in ROUNDS:
        with (root / labels_file).open() as fh:
            labels = {int(r["idx"]): r["seen"] for r in csv.DictReader(fh)}
        with (root / manifest_file).open() as fh:
            manifest_rows = list(csv.DictReader(fh))
        for m in manifest_rows:
            seen = labels.get(int(m["idx"]))
            if seen is None:
                continue
            stats[seen] = stats.get(seen, 0) + 1
            if seen == "uncertain":
                continue  # excluded by design -- never coerce a doubt into a label
            rec = rows.setdefault(
                m["uid"],
                {"uid": m["uid"], "path": m["path"], "present": [], "seen": [], "rounds": [], "adjudicated": []},
            )
            # ADJUDICATION SCOPE -- load-bearing. A crop centred on one mark only lets
            # the labeller rule on marks visible IN THAT CROP. A pill crop (top-left)
            # says nothing about a bottom-right wordmark, so scoring jimeng against a
            # pill-round image would book real detections as false fires (~61% of pills
            # carry a wordmark). Bottom-right marks co-adjudicate each other: one crop
            # of that corner shows whichever of Doubao/Jimeng is there.
            for k in _ADJUDICATES.get(m["key"], (m["key"],)):
                if k not in rec["adjudicated"]:
                    rec["adjudicated"].append(k)
            mark = SEEN_TO_MARK.get(seen)
            if mark and mark not in rec["present"]:
                rec["present"].append(mark)
            if seen not in rec["seen"]:
                rec["seen"].append(seen)
            if round_name not in rec["rounds"]:
                rec["rounds"].append(round_name)

    for r in rows.values():
        r["provenance"] = metadata_provenance(r["path"])
    with out.open("w") as fh:
        for r in rows.values():
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {out}  images={len(rows)}")
    print("label distribution across all rounds:", dict(sorted(stats.items(), key=lambda kv: -kv[1])))
    n_pos = sum(1 for r in rows.values() if r["present"])
    print(f"images with at least one registered mark: {n_pos}; clean-of-registered-marks: {len(rows) - n_pos}")
    adj: dict[str, int] = {}
    for r in rows.values():
        for k in r["adjudicated"]:
            adj[k] = adj.get(k, 0) + 1
    print("images each mark can be SCORED on (adjudication scope):", dict(sorted(adj.items())))


if __name__ == "__main__":
    main()
