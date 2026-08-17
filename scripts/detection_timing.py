"""Per-method wall time for metadata extraction and the verdict built on it.

WHY THIS EXISTS
  The detection path is documented by CAPABILITY -- which signals it reads, in what
  order, with what confidence -- but not by COST. Batch throughput and the question
  "which probe dominates, and does it depend on container or on file size" have never
  been measured.

WHAT IT MEASURES
  Only the file-backed half: metadata extraction, then the verdict evaluated on the
  extracted evidence. The pixel-backed detectors are deliberately out of scope.

  COLD pass: one measurement on a file the process has never touched --
  ``extract_provenance_evidence``. Only the first read of a file is genuinely cold,
  so it buys exactly one number, and that is the one a single-shot run pays.

  WARM pass, with the filesystem cache now hot:
    1. ``extract_provenance_evidence`` as a whole.
    2. Its nine components, timed IN THE ORDER the dataclass constructs them and with
       the per-file caches cleared once beforehand -- so ``extract_c2pa_info`` carries
       the Rust manifest reader and the later ``get_ai_metadata`` sees the same warm
       cache it sees in production. Timing them in any other order moves that cost to
       a different row and flatters whichever ran second.
    3. ``identify_from_evidence``: pure verdict logic, the source is never reopened.
    4. ``identify(check_visible=False, check_invisible=False)`` -- extraction plus
       verdict as one call, the cross-check that 1 + 3 is the whole metadata path.

  Every ``@lru_cache`` in the metadata and C2PA modules is cleared before each timed
  unit. Without that the second measurement of a file answers from the memo and
  reports a cost of zero -- the caches are keyed on (path, mtime, size) and this
  script reads each file several times.

READING IT
  Component times do NOT sum to the ``extract_provenance_evidence`` total for free:
  they come from a separate cache-cleared run, so the sum is a cross-check. A gap
  means a component is missing from the list. Both numbers are written.

DATA SAFETY
  Read-only over a local dataset. Writes only to the given output prefix, which
  belongs outside the repository.

    uv run python scripts/detection_timing.py <dataset> <prefix> --limit 200
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

# The package's OWN tree, not the repository root: from a worktree, an editable
# install resolves `pagedmark` to the MAIN checkout, so a script measuring
# this tree would silently import a different one.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pagedmark import identify as identify_mod
from pagedmark import metadata as metadata_mod
from pagedmark._internal import c2pa as c2pa_mod

log = logging.getLogger(__name__)

SUPPORTED = frozenset({".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif", ".avif"})

# Timed in the order ``ProvenanceEvidence`` is constructed in ``identify.py``. The
# report script imports this to label its columns, so the order is the contract.
COMPONENTS: tuple[tuple[str, Callable[[Path], Any]], ...] = (
    ("c2pa_info", c2pa_mod.extract_c2pa_info),
    ("ai_metadata", metadata_mod.get_ai_metadata),
    ("scan_head", lambda p: metadata_mod.scan_head(p, identify_mod._SCAN_BYTES)),
    ("iptc_ai_system", metadata_mod.iptc_ai_system),
    ("aigc_label", metadata_mod.aigc_label),
    ("exif_generator", metadata_mod.exif_generator),
    ("xai_signature", metadata_mod.xai_signature),
    ("huggingface_job", metadata_mod.huggingface_job),
    ("samsung_genai", metadata_mod.samsung_genai),
)


def _cache_clearers() -> tuple[Callable[[], None], ...]:
    """Every per-file memo in the metadata path, found by attribute, not by name list.

    A hand-written list silently goes stale the next time a probe gains a cache, and a
    stale entry shows up as a suspiciously fast row rather than as an error.
    """
    found: list[Callable[[], None]] = []
    for module in (metadata_mod, c2pa_mod):
        for name in dir(module):
            clear = getattr(getattr(module, name, None), "cache_clear", None)
            if callable(clear):
                found.append(clear)
    return tuple(found)


_CLEARERS = _cache_clearers()


def _clear() -> None:
    for clear in _CLEARERS:
        clear()


def _ms(fn: Callable[[], Any]) -> tuple[float, Any]:
    """Wall time in milliseconds plus the call's result."""
    start = time.perf_counter_ns()
    value = fn()
    return (time.perf_counter_ns() - start) / 1e6, value


def _pixel_geometry(path: Path) -> tuple[str | None, int | None, int | None]:
    """Container format and pixel dimensions from the header alone."""
    try:
        from PIL import Image

        with Image.open(path) as img:
            return img.format, img.width, img.height
    except Exception:  # unreadable or an unsupported container
        return None, None, None


def _warm_breakdown(path: Path, row: dict[str, Any]) -> None:
    """Fill ``row`` with the warm-cache per-method breakdown."""
    _clear()
    row["extract_evidence_ms"], evidence = _ms(lambda: identify_mod.extract_provenance_evidence(path))

    _clear()
    component_sum = 0.0
    for name, fn in COMPONENTS:
        elapsed, _ = _ms(lambda fn=fn: fn(path))
        row[f"meta_{name}_ms"] = elapsed
        component_sum += elapsed
    row["meta_components_sum_ms"] = component_sum

    row["verdict_from_evidence_ms"], report = _ms(lambda: identify_mod.identify_from_evidence(evidence))

    _clear()
    row["identify_metadata_only_ms"], full = _ms(
        lambda: identify_mod.identify(path, check_visible=False, check_invisible=False)
    )

    row["scan_bytes"] = len(evidence.scan)
    row["has_c2pa"] = bool(evidence.c2pa_info)
    row["has_ai_metadata"] = bool(evidence.ai_metadata)
    row["is_ai_generated"] = full.is_ai_generated
    row["confidence"] = full.confidence
    row["platform"] = full.platform
    row["signals"] = [signal.name for signal in full.signals]
    # The two verdict paths must agree; a mismatch means the breakdown timed a
    # different code path than the end-to-end call and the rows are not comparable.
    row["verdict_agrees"] = (report.confidence, report.platform) == (full.confidence, full.platform)


def _measure(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {"path": str(path), "ext": path.suffix.lower()}
    try:
        row["bytes"] = path.stat().st_size
    except OSError as exc:
        return {**row, "error": f"stat: {exc}"}

    # COLD first: this is the only moment the file is untouched by this process.
    _clear()
    try:
        row["cold_extract_evidence_ms"], _ = _ms(lambda: identify_mod.extract_provenance_evidence(path))
    except Exception as exc:
        return {**row, "error": f"cold extract: {type(exc).__name__}: {exc}"}

    row["format"], row["width"], row["height"] = _pixel_geometry(path)
    width, height = row["width"], row["height"]
    row["megapixels"] = round(width * height / 1e6, 3) if width and height else None

    try:
        _warm_breakdown(path, row)
    except Exception as exc:
        row["error"] = f"warm pass: {type(exc).__name__}: {exc}"
    return row


def _iter_images(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            yield path


def _done_paths(out_path: Path) -> set[str]:
    """Paths already recorded, so a long run resumes instead of restarting."""
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, help="directory of images, scanned recursively")
    parser.add_argument("out_prefix", type=Path, help="output prefix; writes <prefix>.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="stop after N files (0 = all)")
    parser.add_argument("--progress-every", type=int, default=200)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out_path = args.out_prefix.with_suffix(".jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_paths(out_path)
    if done:
        log.info("resuming: %d files already recorded", len(done))

    processed = 0
    started = time.monotonic()
    with out_path.open("a", encoding="utf-8") as handle:
        for path in _iter_images(args.dataset):
            if str(path) in done:
                continue
            row = _measure(path)
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            processed += 1
            if processed % args.progress_every == 0:
                log.info("%d files, %.2f files/s", processed, processed / (time.monotonic() - started))
            if args.limit and processed >= args.limit:
                break

    elapsed = time.monotonic() - started
    rate = processed / max(elapsed, 1e-9)
    log.info("done: %d files in %.1f s (%.2f files/s) -> %s", processed, elapsed, rate, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
