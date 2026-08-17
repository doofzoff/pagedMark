"""Audit visible-watermark removal over a local image corpus.

For every image the registry detects a known visible mark in, run that mark's
removal and re-detect on the output, recording before/after confidence and
whether the detector still fires. Also bucket the detected-positive originals
into per-mark dataset dirs so the visible-mark corpora are reproducible.

Detector-clean after removal is necessary but, for the Doubao/Jimeng text marks,
NOT sufficient (their NCC detector is fooled by a thin residual outline -- see
CLAUDE.md). Treat a detector-clean Doubao/Jimeng as "detector passes"; visual
residual is a separate check.

Backend: for a REALISTIC quality audit pass ``--backend migan`` (the production
fill), not ``cv2``. Removal SUCCESS (detector-clean) is backend-independent, so cv2
is fine for a fast pass/fail sweep, but only migan/lama reflect the recovered-region
quality a user actually gets. Run migan when validating the visible pipeline for real.

Operates on gitignored local data only; writes nothing tracked.

    uv run python scripts/visible_removal_audit.py \
        --corpus .local-eval/originals --out .local-eval/visible-audit.csv \
        --dataset-root .local-eval/visible-datasets
"""

from __future__ import annotations

import csv
import logging
import shutil
from pathlib import Path

import click

from pagedmark import image_io
from pagedmark.watermark_registry import detect_marks, get_mark

log = logging.getLogger(__name__)

_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".avif", ".heic"}


def _rel(p: Path, corpus: Path) -> str:
    try:
        return str(p.relative_to(corpus))
    except ValueError:
        return p.name


@click.command()
@click.option(
    "--corpus",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path(".local-eval/originals"),
)
@click.option("--out", type=click.Path(path_type=Path), default=Path(".local-eval/visible-audit.csv"))
@click.option(
    "--dataset-root",
    type=click.Path(path_type=Path),
    default=Path(".local-eval/visible-datasets"),
)
@click.option(
    "--paths-file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Audit only these paths (one per line), skipping the full rglob.",
)
@click.option("--limit", type=int, default=0, help="Scan at most N files (0 = all).")
@click.option(
    "--backend",
    type=click.Choice(["auto", "cv2", "migan", "lama"]),
    default="auto",
    help="Fill backend for removal. Removal SUCCESS (detector-clean) is backend-independent; "
    "cv2 is fastest for a bulk audit, migan/lama only change recovered-region quality.",
)
def main(corpus: Path, out: Path, dataset_root: Path, paths_file: Path | None, limit: int, backend: str) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    if paths_file is not None:
        files = [Path(s) for line in paths_file.read_text().splitlines() if (s := line.strip()) and Path(s).is_file()]
    else:
        files = sorted(p for p in corpus.rglob("*") if p.is_file() and p.suffix.lower() in _EXTS)
    if limit:
        files = files[:limit]
    click.echo(f"Scanning {len(files)} files under {corpus} ...")

    rows: list[dict[str, str]] = []
    n_detected = 0
    n_clean_after = 0
    fails: list[tuple[str, str, float]] = []

    with click.progressbar(files, label="audit") as bar:
        for p in bar:
            img = image_io.imread(p)
            if img is None:
                continue
            for det in detect_marks(img, include_explicit=False):
                if not det.detected:
                    continue
                n_detected += 1
                mark = get_mark(det.key)
                # Bucket the positive original into the per-mark dataset.
                ddir = dataset_root / det.key
                ddir.mkdir(parents=True, exist_ok=True)
                if not (ddir / p.name).exists():
                    shutil.copy2(p, ddir / p.name)
                # Remove, then re-detect with the SAME mark's detector.
                try:
                    cleaned, _ = mark.remove(img, backend=backend)
                    after = mark.detect(cleaned)
                except Exception as exc:
                    log.warning("remove failed on %s (%s): %s", p.name, det.key, exc)
                    rows.append(
                        {
                            "path": _rel(p, corpus),
                            "mark": det.key,
                            "conf_before": f"{det.confidence:.3f}",
                            "conf_after": "",
                            "removed": "error",
                        }
                    )
                    continue
                removed = not after.detected
                n_clean_after += int(removed)
                if not removed:
                    fails.append((_rel(p, corpus), det.key, after.confidence))
                rows.append(
                    {
                        "path": _rel(p, corpus),
                        "mark": det.key,
                        "conf_before": f"{det.confidence:.3f}",
                        "conf_after": f"{after.confidence:.3f}",
                        "removed": str(removed),
                    }
                )

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "mark", "conf_before", "conf_after", "removed"])
        w.writeheader()
        w.writerows(rows)

    by_mark: dict[str, list[bool]] = {}
    for r in rows:
        if r["removed"] in ("True", "False"):
            by_mark.setdefault(r["mark"], []).append(r["removed"] == "True")
    click.echo(f"\nDetected positives: {n_detected}; detector-clean after removal: {n_clean_after}")
    for k, v in sorted(by_mark.items()):
        click.echo(f"  {k:8} removed {sum(v)}/{len(v)} ({100 * sum(v) // max(1, len(v))}%)")
    if fails:
        click.echo(f"\nDetector still fires after removal ({len(fails)}):")
        for path, key, conf in fails[:30]:
            click.echo(f"  {key:8} {conf:.3f}  {path}")
    click.echo(f"\nReport: {out}  |  Datasets: {dataset_root}/<mark>/")


if __name__ == "__main__":
    main()
