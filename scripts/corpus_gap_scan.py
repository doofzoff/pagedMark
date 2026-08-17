"""Audit a local image corpus against the library's own ``identify`` detector.

Two jobs in one pass:

1. **Report** -- run ``identify`` over every image and write one CSV row per file
   (verdict, platform, confidence, watermarks, signals, integrity clashes).
2. **Gap audit** -- for every ``unknown``-verdict file, scan only its *metadata
   region* (PNG text/eXIf chunks, JPEG APPn segments before SOS, or the file
   head for other containers) for known provenance markers. A marker found there
   on a file the detector calls ``unknown`` is a concrete lib gap: a serialization
   or generator we do not yet parse. Scanning the metadata region -- not the whole
   file -- is deliberate: short tokens collide randomly inside compressed PNG
   ``IDAT`` / JPEG scan data, which produced false "xAI/Flux/AIGC" hits when the
   first audit naively scanned the first megabyte.

This is how new detector gaps get found (it is what surfaced the JPEG-EXIF
``{"AIGC":{...}}`` form). Re-run after collecting a fresh evaluation batch.

Usage:
    uv run python scripts/corpus_gap_scan.py --corpus .local-eval/originals
    uv run python scripts/corpus_gap_scan.py --corpus .local-eval/originals \\
        --report .local-eval/detector-report.csv
"""

from __future__ import annotations

import csv
import logging
from collections import Counter
from pathlib import Path

import click
from _plain_console import Console, Table

from pagedmark.identify import identify
from pagedmark.metadata import _png_late_metadata

log = logging.getLogger(__name__)
console = Console()

# Distinctive, multi-byte provenance markers worth flagging when they appear in a
# file the detector calls `unknown`. Kept long enough that a random collision in a
# (non-scanned) compressed stream is implausible; the metadata-region restriction
# below is the primary guard, this list is the second. Group: C2PA/JUMBF infra,
# AI source-type / labeling schemes, and distinctive generator name strings.
MARKERS: tuple[bytes, ...] = (
    # C2PA / JUMBF infrastructure and AI source-type / labeling schemes.
    b"c2pa",
    b"jumbf",
    b"contentauth",
    b"trainedAlgorithmicMedia",
    b"digitalSourceType",
    b'"AIGC"',
    b"<TC260:AIGC>",
    b"TC260:AIGC",
    b"tc260.org.cn",
    b"AISystemUsed",
    b"SynthID",
    b"hf-job-id",
    b"genAIType",
    b"PhotoEditor_Re_Edit",
    b"Signature:",
    # Distinctive multi-word generator strings only. Bare single words (Luma,
    # Gemini, Sora, ...) are omitted: they collide with unrelated metadata prose
    # (e.g. "Luma" in Lightroom's EnhanceDenoiseLumaAmount), defeating precision.
    b"Midjourney",
    b"Stable Diffusion",
    b"StableDiffusion",
    b"ComfyUI",
    b"Automatic1111",
    b"DALL-E",
    b"Ideogram AI",
    b"Adobe Firefly",
    b"Black Forest",
    b"volcengine",
    b"Doubao",
    b"\xe8\xb1\x86\xe5\x8c\x85",
    b"Nano Banana",
    b"Stability AI",
    b"Samsung Galaxy",
)


def _metadata_region(path: Path) -> bytes:
    """Return only the bytes where provenance metadata can live, never the
    compressed pixel stream (which produces random short-token collisions)."""
    try:
        head = path.read_bytes()
    except OSError:
        return b""
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        # All ancillary metadata chunks (window=0), via the library's own walker.
        return _png_late_metadata(path, 0)
    if head[:2] == b"\xff\xd8":  # JPEG: APPn segments up to Start-Of-Scan
        out = bytearray()
        p = 2
        n = len(head)
        while p + 4 <= n and head[p] == 0xFF:
            marker = head[p + 1]
            if marker == 0xDA:  # SOS -> compressed scan data follows
                break
            seg_len = (head[p + 2] << 8) | head[p + 3]
            out += head[p + 4 : p + 2 + seg_len]
            p += 2 + seg_len
        return bytes(out)
    return head[:65536]  # webp/avif/heif/jxl: metadata sits near the head


def _row(rep) -> dict[str, str]:  # noqa: ANN001 (ProvenanceReport)
    return {
        "path": "",  # filled by caller (relative)
        "is_ai": str(rep.is_ai_generated),
        "platform": rep.platform or "",
        "confidence": rep.confidence,
        "watermarks": "|".join(rep.watermarks),
        "signals": "|".join(s.name for s in rep.signals),
        "integrity_clashes": "|".join(rep.integrity_clashes),
    }


@click.command()
@click.option(
    "--corpus",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path(".local-eval/originals"),
    show_default=True,
    help="Directory of images to scan (recursively).",
)
@click.option(
    "--report",
    type=click.Path(path_type=Path),
    default=None,
    help="Write the per-file CSV here (default: <corpus>/../detector_report.csv).",
)
@click.option("--limit", type=int, default=0, help="Scan at most N files (0 = all).")
def main(corpus: Path, report: Path | None, limit: int) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    report = report or corpus.parent / "detector_report.csv"

    files = sorted(p for p in corpus.rglob("*") if p.is_file())
    if limit:
        files = files[:limit]
    console.print(f"Scanning [bold]{len(files)}[/bold] files under {corpus} ...")

    verdicts: Counter[str] = Counter()
    platforms: Counter[str] = Counter()
    gap_tokens: Counter[str] = Counter()
    gaps: list[tuple[str, list[str]]] = []
    rows: list[dict[str, str]] = []
    errors = 0

    with click.progressbar(files, label="identify") as bar:
        for p in bar:
            rel = str(p.relative_to(corpus))
            try:
                rep = identify(p)
            except Exception as exc:
                log.warning("identify failed on %s: %s", rel, exc)
                errors += 1
                continue
            row = _row(rep)
            row["path"] = rel
            rows.append(row)
            if rep.is_ai_generated:
                verdicts["ai"] += 1
                platforms[rep.platform or "?"] += 1
                continue
            verdicts["unknown"] += 1
            # A gap candidate is a file identify is *blind* to (no signal at all)
            # yet whose metadata carries a known marker. A file that produced a
            # signal but no AI verdict (e.g. an ASUS Gallery C2PA signer, which we
            # attribute but do not call AI) is handled correctly -- not a gap.
            if rep.signals:
                continue
            region = _metadata_region(p)
            hits = sorted({m.decode("latin-1", "replace") for m in MARKERS if m in region})
            if hits:
                gaps.append((rel, hits))
                gap_tokens.update(hits)

    with report.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["path", "is_ai", "platform", "confidence", "watermarks", "signals", "integrity_clashes"],
        )
        writer.writeheader()
        writer.writerows(rows)
    console.print(f"\nWrote [bold]{len(rows)}[/bold] rows -> {report}")

    console.print(f"\n[bold]Verdicts:[/bold] AI {verdicts['ai']} | unknown {verdicts['unknown']} | errors {errors}")
    plat = Table(title="AI platforms", show_header=False)
    for name, n in platforms.most_common():
        plat.add_row(str(n), name)
    console.print(plat)

    if gaps:
        console.print(
            f"\n[bold red]Gap candidates[/bold red]: {len(gaps)} unknown files carry a known "
            f"marker in their metadata region (potential undetected serialization/generator):"
        )
        tok = Table(title="markers seen in unknown files")
        tok.add_column("count", justify="right")
        tok.add_column("marker")
        for name, n in gap_tokens.most_common():
            tok.add_row(str(n), name)
        console.print(tok)
        for rel, hits in gaps:
            console.print(f"  {rel}  ->  {', '.join(hits)}")
    else:
        console.print("\n[green]No gap candidates: every unknown file is metadata-free.[/green]")


if __name__ == "__main__":
    main()
