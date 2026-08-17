"""Audit the record path against the file path over a whole dataset.

WHY THIS EXISTS

Two seams reach the same provenance verdict:

    identify(path, check_visible=False, check_invisible=False)

    identify_metadata_record(collect_metadata_record(path), path=path)

Their equality is the record's entire contract, and it can break from either side --
a region the collector stops walking, or a placement the file path learns to read and
the record does not. ``tests/test_metadata_record.py`` pins it over the tracked
fixtures; those cover the signal families we already know about. This covers the ones
we do not: every real placement in a real corpus, which is where all three defects
found so far actually came from.

The record is round-tripped through ``json.dumps``/``loads`` before it is judged, so
a value that only survives in memory fails here rather than at a customer.

WHAT IT REPORTS

One JSONL row per image: both verdicts, whether they agree, the record size, and any
exception from either side. The summary counts disagreements by field and by signal,
so "the record lost samsung_genai on 13 files" reads directly off the output instead
of being reconstructed.

Pass ``--baseline`` with an earlier run to also diff against it. That answers the
other question a detection change raises: which files changed verdict, and are they
exactly the ones that were meant to.

DATA SAFETY

Read-only over a local dataset. Writes only the given output path, which belongs
outside the repository. Resumable: rerunning skips files already recorded.

    uv run python scripts/record_parity_audit.py data/spaces/originals .local-eval/parity.jsonl
    uv run python scripts/record_parity_audit.py <dataset> <out> --baseline .local-eval/previous.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

# The package's OWN tree, not the repository root: from a worktree, an editable
# install resolves `pagedmark` to the MAIN checkout, so a script measuring
# this tree would silently import a different one.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pagedmark.identify import identify, identify_metadata_record
from pagedmark.metadata_record import collect_metadata_record

log = logging.getLogger(__name__)

SUPPORTED = frozenset({".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif", ".avif"})
# Every field of the verdict a caller can act on. `path` is excluded: it is extraction
# context, and the two paths are handed the same one by construction.
COMPARED = ("is_ai_generated", "platform", "confidence", "ai_source_kind", "ai_from_metadata")


def _verdict(report: Any) -> dict[str, Any]:
    return {
        **{field: getattr(report, field) for field in COMPARED},
        "signals": sorted(signal.name for signal in report.signals),
        "watermarks": sorted(report.watermarks),
    }


def _audit(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {"path": str(path)}
    try:
        row["bytes"] = path.stat().st_size
    except OSError as exc:
        return {**row, "error": f"stat: {exc}"}

    try:
        started = time.perf_counter()
        record = json.loads(json.dumps(collect_metadata_record(path)))
        row["collect_ms"] = (time.perf_counter() - started) * 1000
        row["record_bytes"] = len(json.dumps(record))
        row["container"] = record.get("container")
        via_record = _verdict(identify_metadata_record(record, path=path))
    except Exception as exc:
        return {**row, "error": f"record path: {type(exc).__name__}: {exc}"}

    try:
        via_file = _verdict(identify(path, check_visible=False, check_invisible=False))
    except Exception as exc:
        return {**row, "error": f"file path: {type(exc).__name__}: {exc}"}

    row["record"] = via_record
    row["file"] = via_file
    row["agree"] = via_record == via_file
    return row


def _iter_images(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            yield path


def _done(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done: set[str] = set()
    with out_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                done.add(json.loads(line)["path"])
            except (ValueError, KeyError):
                continue
    return done


def _summarize(rows: list[dict[str, Any]], baseline: Path | None) -> None:
    failed = [r for r in rows if "error" in r]
    usable = [r for r in rows if "error" not in r]
    disagreed = [r for r in usable if not r["agree"]]

    print(f"\nimages: {len(rows)}   errors: {len(failed)}   compared: {len(usable)}")
    print(f"record path disagrees with file path: {len(disagreed)}")
    for row in failed[:10]:
        print(f"   ERROR {Path(row['path']).name}: {row['error']}")

    fields: collections.Counter[str] = collections.Counter()
    for row in disagreed:
        fields.update(field for field in COMPARED if row["record"][field] != row["file"][field])
        for name in set(row["file"]["signals"]) - set(row["record"]["signals"]):
            fields[f"signal missing from record: {name}"] += 1
        for name in set(row["record"]["signals"]) - set(row["file"]["signals"]):
            fields[f"signal only in record: {name}"] += 1
    for label, count in fields.most_common():
        print(f"   {label}: {count}")
    for row in disagreed[:10]:
        print(f"   {Path(row['path']).name}\n      record: {row['record']}\n      file:   {row['file']}")

    if baseline is None:
        return
    previous = {}
    with baseline.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if "error" not in item:
                previous[Path(item["path"]).name] = item

    changed = []
    for row in usable:
        was = previous.get(Path(row["path"]).name)
        if was is None:
            continue
        # The WHOLE verdict, not a chosen subset. A first version compared confidence
        # and signals only and reported "0 changed" for a run whose single intended
        # correction was a watermark line -- the change it existed to show.
        before = was.get("file") or {}
        if before and before != row["file"]:
            changed.append((row["path"], before, row["file"]))
    print(f"\nverdicts changed against the baseline: {len(changed)}")
    moved: collections.Counter[str] = collections.Counter()
    for _, before, after in changed:
        for name in set(after["signals"]) - set(before.get("signals") or []):
            moved[f"gained signal {name}"] += 1
        for name in set(before.get("signals") or []) - set(after["signals"]):
            moved[f"LOST signal {name}"] += 1
        for field in (*COMPARED, "watermarks"):
            if before.get(field) != after.get(field):
                moved[f"{field} changed"] += 1
    for label, count in moved.most_common():
        print(f"   {label}: {count}")
    for path, before, after in changed[:10]:
        print(f"   {Path(path).name}\n      before: {before}\n      after:  {after}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--baseline", type=Path, default=None, help="an earlier run to diff verdicts against")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=2000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = _done(args.out)
    if done:
        log.info("resuming: %d images already audited", len(done))

    processed = 0
    started = time.monotonic()
    with args.out.open("a", encoding="utf-8") as handle:
        for path in _iter_images(args.dataset):
            if str(path) in done:
                continue
            handle.write(json.dumps(_audit(path), ensure_ascii=False, default=str) + "\n")
            handle.flush()
            processed += 1
            if processed % args.progress_every == 0:
                log.info("%d images, %.1f/s", processed, processed / (time.monotonic() - started))
            if args.limit and processed >= args.limit:
                break

    log.info("audited %d images in %.1f s", processed, time.monotonic() - started)
    with args.out.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    _summarize(rows, args.baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
