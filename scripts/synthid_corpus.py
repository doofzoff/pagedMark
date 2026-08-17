"""Ingest and inspect the local SynthID reference corpus.

Copies images into ``data/synthid/originals/`` and records one row
per image in ``manifest.csv`` (sha256, resolution, format, C2PA issuer, and the
external verification level). Dogfoods the project's own C2PA detector so the
recorded metadata matches what the library sees.

See ``data/synthid/README.md`` for the collection protocol.

Usage:
    uv run python scripts/synthid_corpus.py ingest IMAGES... --label pos \\
        --source "Gemini app" --model gemini-3-pro --verified-via gemini-app
    uv run python scripts/synthid_corpus.py status
"""

from __future__ import annotations

import csv
import hashlib
import logging
import shutil
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import click
from _plain_console import Console, Table
from PIL import Image

from pagedmark._internal.c2pa import extract_c2pa_info

log = logging.getLogger(__name__)
console = Console()

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "data" / "synthid"
LABELS = ("pos", "neg")
VERIFIED_VIA = ("gemini-app", "openai-verify", "synthid-portal", "c2pa-metadata", "third-party", "none")
FIELDNAMES = [
    "sha256",
    "filename",
    "label",
    "source",
    "model",
    "width",
    "height",
    "format",
    "c2pa_issuer",
    "synthid_metadata",
    "verified_via",
    "added",
    "notes",
]


def _manifest_path(root: Path) -> Path:
    return root / "manifest.csv"


def _read_manifest(root: Path) -> list[dict[str, str]]:
    path = _manifest_path(root)
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dims_via_sips(path: Path) -> tuple[int, int]:
    """Read pixel dimensions via macOS ``sips`` (handles HEIC, which PIL can't
    open without pillow-heif). Returns (0, 0) off macOS or on failure."""
    sips = shutil.which("sips")
    if sys.platform != "darwin" or not sips:
        return 0, 0
    try:
        # Fixed command; the only variable is a local file path passed to ingest.
        out = subprocess.run(  # noqa: S603
            [sips, "-g", "pixelWidth", "-g", "pixelHeight", str(path)],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
        dims = {"pixelWidth": 0, "pixelHeight": 0}
        for line in out.splitlines():
            for key in dims:
                if f"{key}:" in line:
                    dims[key] = int(line.split(":")[1].strip())
        return dims["pixelWidth"], dims["pixelHeight"]
    except Exception:  # sips missing/odd output; dims stay unknown
        return 0, 0


def _probe(path: Path) -> tuple[int, int, str, str, bool]:
    """Return (width, height, format, c2pa_issuer, synthid_metadata)."""
    width = height = 0
    fmt = path.suffix.lstrip(".").lower()
    try:
        with Image.open(path) as img:
            width, height = img.size
            fmt = (img.format or fmt).lower()
    except Exception as exc:  # unknown/user formats can raise non-OSError; see CLAUDE.md
        log.debug("PIL could not open %s: %s", path, exc)

    if width == 0:  # e.g. HEIC from iPhone — fall back to macOS sips
        width, height = _dims_via_sips(path)

    info = extract_c2pa_info(path)
    issuer = str(info.get("issuer", ""))
    synthid = "synthid_watermark" in info
    return width, height, fmt, issuer, synthid


@click.group()
def cli() -> None:
    """Manage the local SynthID reference corpus."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@cli.command()
@click.argument("images", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--label", required=True, type=click.Choice(LABELS), help="SynthID label.")
@click.option("--source", default="", help="Where the image came from (free text).")
@click.option("--model", default="", help="Generating model, e.g. gemini-3-pro.")
@click.option(
    "--verified-via", "verified_via", default="none", type=click.Choice(VERIFIED_VIA), help="Ground-truth oracle."
)
@click.option("--notes", default="", help="Free-text notes (e.g. resolution batch).")
@click.option("--root", type=click.Path(path_type=Path), default=DEFAULT_ROOT, help="Corpus root.")
def ingest(
    images: tuple[Path, ...],
    label: str,
    source: str,
    model: str,
    verified_via: str,
    notes: str,
    root: Path,
) -> None:
    """Copy IMAGES into the corpus and append rows to the manifest."""
    dest_dir = root / "originals"
    dest_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_manifest(root)
    seen = {r["sha256"] for r in rows}
    added = 0
    new_rows: list[dict[str, str]] = []

    for src in images:
        digest = _sha256(src)
        if digest in seen:
            console.print(f"  [dim]skip (duplicate sha256): {src.name}[/]")
            continue
        seen.add(digest)

        width, height, fmt, issuer, synthid = _probe(src)
        stored_name = src.name
        destination = dest_dir / stored_name
        if destination.exists() and _sha256(destination) != digest:
            raise click.ClickException(f"filename already exists with different content: {stored_name}")
        if not destination.exists():
            shutil.copy2(src, destination)

        new_rows.append(
            {
                "sha256": digest,
                "filename": stored_name,
                "label": label,
                "source": source,
                "model": model,
                "width": str(width),
                "height": str(height),
                "format": fmt,
                "c2pa_issuer": issuer,
                "synthid_metadata": "yes" if synthid else "",
                "verified_via": verified_via,
                "added": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "notes": notes,
            }
        )
        added += 1
        flag = " [yellow](C2PA-SynthID)[/]" if synthid else ""
        console.print(f"  [green]+[/] {label}/{stored_name}  {width}x{height} {fmt}{flag}")

    if new_rows:
        path = _manifest_path(root)
        write_header = not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerows(new_rows)

    console.print(f"\n  Ingested [bold]{added}[/] image(s); skipped {len(images) - added} duplicate(s).")


@cli.command()
@click.option("--root", type=click.Path(path_type=Path), default=DEFAULT_ROOT, help="Corpus root.")
def status(root: Path) -> None:
    """Print corpus counts by label, resolution, and verification level."""
    rows = _read_manifest(root)
    if not rows:
        console.print("  [dim]Corpus is empty.[/]")
        return

    by_label = Counter(r["label"] for r in rows)
    by_verified = Counter(r["verified_via"] for r in rows)
    by_res = Counter(f"{r['label']}  {r['width']}x{r['height']}" for r in rows)

    console.print(f"\n  [bold]{len(rows)}[/] image(s) in {root}")

    t = Table(title="By label", show_header=True, header_style="bold")
    t.add_column("Label")
    t.add_column("Count", justify="right")
    for k in LABELS:
        if by_label.get(k):
            t.add_row(k, str(by_label[k]))
    console.print(t)

    t = Table(title="By label x resolution", show_header=True, header_style="bold")
    t.add_column("Label / resolution")
    t.add_column("Count", justify="right")
    for k, v in sorted(by_res.items()):
        t.add_row(k, str(v))
    console.print(t)

    t = Table(title="By verification", show_header=True, header_style="bold")
    t.add_column("verified_via")
    t.add_column("Count", justify="right")
    for k, v in by_verified.most_common():
        t.add_row(k, str(v))
    console.print(t)


if __name__ == "__main__":
    cli()
