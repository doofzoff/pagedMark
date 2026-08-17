"""Aggregate ``detection_timing.py`` records into per-method, per-segment tables.

WHAT IT PRODUCES
  Reading it back: the per-file JSONL is one row per image with a millisecond field
  per method. This collapses it into percentiles per method, then repeats that per
  segment -- container, megapixels, file size, C2PA presence, verdict confidence --
  because a single median hides a path whose cost is carried entirely by one
  container or by the files that actually have a manifest.

  Writes ``<prefix>_summary.csv`` (long form: segment_kind, segment, method, n, p50,
  p90, p99, mean) and prints a markdown report.

  Percentiles are computed by nearest-rank on the sorted sample, not interpolated:
  every reported number is a time some real file actually took.

DATA SAFETY
  Reads and writes only the given prefix, which belongs outside the repository.

    uv run python scripts/detection_timing_report.py .local-eval/timing/full
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detection_timing import COMPONENTS as _TIMED

# Taken from the script that WROTE the rows, in its order, so a probe added, removed
# or reordered there cannot silently leave a column missing or mislabelled here.
COMPONENTS = tuple(name for name, _ in _TIMED)
METHODS = (
    ("cold_extract_evidence_ms", "extract_provenance_evidence (cold)"),
    ("extract_evidence_ms", "extract_provenance_evidence (warm)"),
    *((f"meta_{name}_ms", f"  {name}") for name in COMPONENTS),
    ("meta_components_sum_ms", "  (sum of components)"),
    ("verdict_from_evidence_ms", "identify_from_evidence (verdict)"),
    ("identify_metadata_only_ms", "identify(metadata only), end to end"),
)
# Median-only columns in the per-segment tables; the full percentile set goes to the CSV.
HEADLINE_LABELS = (
    ("extract_evidence_ms", "extract p50"),
    ("verdict_from_evidence_ms", "verdict p50"),
    ("identify_metadata_only_ms", "end-to-end p50"),
)
HEADLINE = tuple(key for key, _ in HEADLINE_LABELS)


def _bucket(value: float | None, edges: tuple[float, ...], labels: tuple[str, ...]) -> str:
    if value is None:
        return "unknown"
    for edge, label in zip(edges, labels, strict=False):
        if value < edge:
            return label
    return labels[-1]


SEGMENTS: tuple[tuple[str, Callable[[dict[str, Any]], str]], ...] = (
    ("container", lambda r: str(r.get("format") or "unknown")),
    (
        "megapixels",
        lambda r: _bucket(r.get("megapixels"), (1, 4, 12), ("<1 MP", "1-4 MP", "4-12 MP", ">12 MP")),
    ),
    (
        "file size",
        lambda r: _bucket(
            (r["bytes"] / 1e6) if r.get("bytes") is not None else None,
            (1, 5, 20),
            ("<1 MB", "1-5 MB", "5-20 MB", ">20 MB"),
        ),
    ),
    ("c2pa", lambda r: "with C2PA" if r.get("has_c2pa") else "no C2PA"),
    ("verdict", lambda r: f"confidence={r.get('confidence') or 'n/a'}"),
)


def _numeric(records: list[dict[str, Any]], key: str) -> list[float]:
    """The numeric values of ``key``, skipping rows where it is absent or non-numeric.

    One definition of "counts as a measurement", so the printed table and the CSV can
    never disagree about which rows a method was measured on.
    """
    return [float(r[key]) for r in records if isinstance(r.get(key), (int, float))]


def _percentile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank percentile: the reported value is one a real file produced."""
    index = max(0, math.ceil(q * len(sorted_values)) - 1)
    return sorted_values[index]


def _stats(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "p99": _percentile(ordered, 0.99),
        "mean": statistics.fmean(ordered),
    }


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                yield json.loads(line)
            except ValueError:  # a truncated tail while the run is still writing
                continue


def _table(rows: list[dict[str, Any]], methods: tuple[tuple[str, str], ...]) -> list[str]:
    out = ["| method | n | p50 | p90 | p99 | mean |", "|---|---:|---:|---:|---:|---:|"]
    for key, label in methods:
        values = _numeric(rows, key)
        if not values:
            continue
        s = _stats(values)
        out.append(f"| {label} | {s['n']} | {s['p50']:.3f} | {s['p90']:.3f} | {s['p99']:.3f} | {s['mean']:.3f} |")
    return out


def _correlation(rows: list[dict[str, Any]], x_key: str, y_key: str) -> float | None:
    both = [r for r in rows if isinstance(r.get(x_key), (int, float)) and isinstance(r.get(y_key), (int, float))]
    if len(both) < 3:
        return None
    xs = _numeric(both, x_key)
    ys = _numeric(both, y_key)
    try:
        return statistics.correlation(xs, ys)
    except statistics.StatisticsError:  # a constant column has no correlation
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prefix", type=Path, help="prefix given to detection_timing.py")
    args = parser.parse_args()

    jsonl = args.prefix.with_suffix(".jsonl")
    all_rows = list(_rows(jsonl))
    failed = [r for r in all_rows if "error" in r]
    rows = [r for r in all_rows if "error" not in r]
    if not rows:
        print(f"no usable records in {jsonl}")
        return 1

    disagreed = [r for r in rows if r.get("verdict_agrees") is False]

    lines = [
        f"# Detection timing over {len(rows)} images",
        "",
        f"Source: `{jsonl}`. Failed rows: {len(failed)}. Verdict-path mismatches: {len(disagreed)}.",
        "",
        "All times in milliseconds. Percentiles are nearest-rank.",
        "",
        "## Whole corpus",
        "",
        *_table(rows, METHODS),
        "",
    ]

    csv_rows: list[dict[str, Any]] = []
    for key, label in METHODS:
        values = _numeric(rows, key)
        if values:
            csv_rows.append({"segment_kind": "all", "segment": "all", "method": label.strip(), **_stats(values)})

    for kind, classify in SEGMENTS:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[classify(row)].append(row)
        lines += [f"## By {kind}", ""]
        header = ["| segment | n | " + " | ".join(label for _, label in HEADLINE_LABELS) + " |"]
        header.append("|---|---:|" + "---:|" * len(HEADLINE))
        lines += header
        for segment, group in sorted(groups.items(), key=lambda item: -len(item[1])):
            cells = []
            for key in HEADLINE:
                values = _numeric(group, key)
                cells.append(f"{_percentile(sorted(values), 0.5):.2f}" if values else "-")
            lines.append(f"| {segment} | {len(group)} | " + " | ".join(cells) + " |")
            for key, label in METHODS:
                values = _numeric(group, key)
                if values:
                    csv_rows.append(
                        {"segment_kind": kind, "segment": segment, "method": label.strip(), **_stats(values)}
                    )
        lines.append("")

    corr = _correlation(rows, "scan_bytes", "verdict_from_evidence_ms")
    corr_size = _correlation(rows, "bytes", "extract_evidence_ms")
    lines += [
        "## Scaling",
        "",
        f"- verdict time vs scan-buffer size: r = {corr:.3f}" if corr is not None else "- verdict correlation: n/a",
        (
            f"- extraction time vs file size: r = {corr_size:.3f}"
            if corr_size is not None
            else "- extraction correlation: n/a"
        ),
        "",
    ]

    summary_csv = args.prefix.with_name(args.prefix.name + "_summary.csv")
    columns = ["segment_kind", "segment", "method", "n", "p50", "p90", "p99", "mean"]
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(csv_rows)

    report = "\n".join(lines)
    args.prefix.with_name(args.prefix.name + "_report.md").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nwrote {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
