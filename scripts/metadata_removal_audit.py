"""Audit AI-metadata REMOVAL over a local image corpus (detection<->removal parity).

`corpus_gap_scan.py` proves the DETECTOR sees a marker; this proves the STRIPPER
reaches it. For every file that carries an AI-metadata signal, run
``remove_ai_metadata`` and re-scan the output with the SAME oracle
(``get_ai_metadata``). Any signal that survives is a real parity bug: a re-served
file still reads as AI. Also assert the strip is lossless -- decoded pixels
(and alpha) bit-identical before/after -- since the removal must only touch
metadata, never the coded image.

A no-op control set (clean images with no AI metadata) verifies the stripper
neither ADDS a signal nor corrupts pixels on files it should leave alone.

Operates on gitignored local data only; writes nothing tracked.

    uv run python scripts/metadata_removal_audit.py \
        --corpus .local-eval/originals --identify .local-eval/identify \
        --out .local-eval/metadata-removal-audit.csv --jobs 8
"""

from __future__ import annotations

import csv
import json
import logging
import random
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
import numpy as np

from pagedmark._internal.constants import SUPPORTED_FORMATS

log = logging.getLogger(__name__)

# identify-JSON watermark substrings that imply a METADATA-borne signal (as
# opposed to a purely visual sparkle/text mark). Used only to pick the candidate
# population fast; get_ai_metadata is the per-file ground truth.
_META_HINTS = (
    "C2PA",
    "Content Credentials",
    "IPTC",
    "Made with AI",
    "AIGC",
    "TC260",
    "EXIF",
    "Signature",
    "SynthID",
    "hf-job",
    "HuggingFace",
    "Samsung",
    "soft-binding",
    "metadata",
)


def _pixels(path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Decoded BGR + alpha, for the lossless-strip integrity check."""
    from pagedmark import image_io

    return image_io.read_bgr_and_alpha(path)


def _same_pixels(a: Path, b: Path) -> bool | None:
    """True/False if both decode; None if either is undecodable (can't compare)."""
    try:
        bgr_a, al_a = _pixels(a)
        bgr_b, al_b = _pixels(b)
    except Exception:
        return None
    if bgr_a is None or bgr_b is None:
        return None
    if bgr_a.shape != bgr_b.shape or not np.array_equal(bgr_a, bgr_b):
        return False
    if (al_a is None) != (al_b is None):
        return False
    return al_a is None or al_b is None or np.array_equal(al_a, al_b)


def _audit_one(path_str: str) -> dict[str, object]:
    """Worker: detect -> strip -> re-detect + pixel-integrity for one file."""
    from pagedmark.metadata import get_ai_metadata, remove_ai_metadata

    path = Path(path_str)
    row: dict[str, object] = {
        "path": path.name,
        "ext": path.suffix.lower(),
        "carrier": False,
        "before": "",
        "after": "",
        "parity_ok": "",
        "pixels_identical": "",
        "status": "ok",
    }
    try:
        before = get_ai_metadata(path)
    except Exception as exc:
        row["status"] = f"scan_error:{type(exc).__name__}"
        return row
    row["before"] = "|".join(sorted(before))
    row["carrier"] = bool(before)

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / f"clean{path.suffix.lower()}"
        try:
            remove_ai_metadata(path, out)
        except Exception as exc:
            row["status"] = f"strip_error:{type(exc).__name__}"
            return row
        if not out.exists():
            row["status"] = "no_output"
            return row
        try:
            after = get_ai_metadata(out)
        except Exception as exc:
            row["status"] = f"rescan_error:{type(exc).__name__}"
            return row
        row["after"] = "|".join(sorted(after))
        row["parity_ok"] = not after  # every AI signal must be gone
        same = _same_pixels(path, out)
        row["pixels_identical"] = "" if same is None else same
    return row


def _candidate_paths(corpus: Path, identify: Path | None, clean_sample: int) -> tuple[list[Path], list[Path]]:
    """Return (carriers, clean_controls). Uses identify JSONs when present to pick
    metadata carriers fast; falls back to scanning every file."""
    if identify is None or not identify.exists():
        files = sorted(p for p in corpus.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_FORMATS)
        return files, []

    carriers: list[Path] = []
    clean: list[Path] = []
    for jf in identify.rglob("*.json"):
        try:
            d = json.loads(jf.read_text())
        except Exception:  # noqa: S112 -- skip an unreadable identify JSON, not security-relevant
            continue
        src = d.get("src")
        if not src:
            continue
        img = corpus / jf.parent.name / src
        if not img.exists() or img.suffix.lower() not in SUPPORTED_FORMATS:
            continue
        wm = " | ".join(d.get("watermarks") or [])
        if any(h in wm for h in _META_HINTS):
            carriers.append(img)
        elif not d.get("is_ai_generated"):
            clean.append(img)
    rng = random.Random(0)  # noqa: S311 -- deterministic sampling seed, not cryptographic
    rng.shuffle(clean)
    return carriers, clean[:clean_sample]


@click.command()
@click.option(
    "--corpus",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path(".local-eval/originals"),
)
@click.option(
    "--identify",
    type=click.Path(path_type=Path),
    default=Path(".local-eval/identify"),
    help="identify-JSON dir to pick carriers (skip = scan all).",
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=Path(".local-eval/metadata-removal-audit.csv"),
)
@click.option(
    "--clean-sample", type=int, default=1500, help="No-op control: N clean images to prove the strip is a no-op."
)
@click.option("--limit", type=int, default=0, help="Cap carriers scanned (0 = all).")
@click.option("--jobs", type=int, default=8)
def main(corpus: Path, identify: Path | None, out: Path, clean_sample: int, limit: int, jobs: int) -> None:
    logging.basicConfig(level=logging.ERROR, format="%(message)s")
    carriers, clean = _candidate_paths(corpus, identify, clean_sample)
    if limit:
        carriers = carriers[:limit]
    tasks = [(p, "carrier") for p in carriers] + [(p, "clean") for p in clean]
    click.echo(f"Carriers: {len(carriers)} | clean controls: {len(clean)} | jobs {jobs}")

    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(_audit_one, str(p)): k for p, k in tasks}
        for done, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            row["kind"] = futs[fut]
            rows.append(row)
            if done % 500 == 0:
                click.echo(f"  {done}/{len(tasks)}")

    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["kind", "path", "ext", "carrier", "before", "after", "parity_ok", "pixels_identical", "status"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    # ---- Summary ----
    car = [r for r in rows if r["kind"] == "carrier"]
    ctl = [r for r in rows if r["kind"] == "clean"]
    real_carriers = [r for r in car if r["carrier"] and r["status"] == "ok"]
    parity_fail = [r for r in real_carriers if r["parity_ok"] is False]
    pixel_fail = [r for r in real_carriers if r["pixels_identical"] is False]
    errors = [r for r in rows if r["status"] != "ok"]

    click.echo("\n===== METADATA REMOVAL PARITY =====")
    click.echo(f"Carrier candidates:      {len(car)}")
    click.echo(f"Confirmed carriers:      {len(real_carriers)} (get_ai_metadata non-empty)")
    click.echo(f"Parity FAILS (signal survives strip): {len(parity_fail)}")
    click.echo(f"Pixel-integrity FAILS (strip altered pixels): {len(pixel_fail)}")
    click.echo(f"Errors (scan/strip/decode): {len(errors)}")

    # Per-signal parity breakdown.
    sig_total: Counter[str] = Counter()
    sig_fail: Counter[str] = Counter()
    for r in real_carriers:
        for s in str(r["before"]).split("|"):
            if s:
                sig_total[s] += 1
        if r["parity_ok"] is False:
            for s in str(r["after"]).split("|"):
                if s:
                    sig_fail[s] += 1
    click.echo("\nPer-signal (carriers / surviving-after-strip):")
    for s, n in sig_total.most_common():
        click.echo(f"  {s:24} {n:6}  survived: {sig_fail.get(s, 0)}")

    click.echo("\n===== NO-OP CONTROL (clean images) =====")
    ctl_ok = [r for r in ctl if r["status"] == "ok"]
    added = [r for r in ctl_ok if r["after"]]  # strip must not ADD a signal
    corrupted = [r for r in ctl_ok if r["pixels_identical"] is False]
    click.echo(f"Clean controls scanned:  {len(ctl_ok)}")
    click.echo(f"Strip ADDED a signal:    {len(added)}")
    click.echo(f"Strip corrupted pixels:  {len(corrupted)}")

    if parity_fail:
        click.echo("\n--- Parity failures (first 30) ---")
        for r in parity_fail[:30]:
            click.echo(f"  {r['ext']:6} survived=[{r['after']}]  {r['path']}")
    if pixel_fail:
        click.echo("\n--- Pixel-integrity failures (first 30) ---")
        for r in pixel_fail[:30]:
            click.echo(f"  {r['ext']:6} {r['path']}")
    if errors:
        ec: Counter[str] = Counter(str(r["status"]) for r in errors)
        click.echo("\n--- Errors by kind ---")
        for k, n in ec.most_common():
            click.echo(f"  {n:5}  {k}")

    click.echo(f"\nReport: {out}")


if __name__ == "__main__":
    main()
