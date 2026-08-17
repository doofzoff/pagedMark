# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click",
#   "huggingface-hub>=0.20.0",
#   "numpy",
#   "onnxruntime>=1.24.0",
#   "opencv-python-headless<5",
#   "paddleocr>=3.3.3",
#   "paddlepaddle",
#   "pillow",
# ]
# ///
"""Evaluation-only text restoration over a scrubbed image.

``vae-glyphs`` composites only thresholded glyph-core pixels from a separately
generated VAE reconstruction over a fresh silhouette edge. ``source-glyphs``
can preserve typeface, layout, color, and antialiasing when
SOURCE is itself a regenerated layer. It must not be treated as safe when SOURCE
is the watermarked original: a provider oracle detected SynthID after that exact
paste-back experiment. ``source-silhouette`` instead transfers only a
thresholded glyph shape, then synthesizes fresh flat-color pixels and
antialiasing. ``rerender`` retains the system-font negative control. This is
not a production stage and does not add PaddleOCR to the package graph.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import sys
import unicodedata
from dataclasses import asdict
from pathlib import Path
from typing import Any

import click
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pagedmark import region_eraser  # noqa: E402
from pagedmark._internal.text_restoration import VerifiedTextLine as TextLine  # noqa: E402
from pagedmark._internal.text_restoration import (  # noqa: E402
    composite_fresh_text_edges,
    composite_reconstructed_glyphs,
    group_text_lines,
    residual_glyph_mask,
    source_silhouette_mask,
)
from scripts._text_eval import normalize_text, normalized_edit_distance  # noqa: E402

if ROOT not in Path(region_eraser.__file__).resolve().parents:
    raise RuntimeError("selective_text_restoration imported outside the current worktree")

REGULAR_FONT = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
BOLD_FONT = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
CJK_FONT = Path("/System/Library/Fonts/STHeiti Medium.ttc")


def should_preserve_line(
    expected: str,
    source_text: str,
    source_score: float,
    candidate_text: str,
    candidate_score: float,
) -> bool:
    if min(source_score, candidate_score) < 0.75:
        return False
    if normalized_edit_distance(expected, source_text) > 0.25:
        return False
    return normalize_text(source_text) == normalize_text(candidate_text)


def composite_source_glyphs(
    source_rgb: np.ndarray,
    background_rgb: np.ndarray,
    glyph_mask: np.ndarray,
    *,
    feather: float = 0.7,
) -> np.ndarray:
    """Composite exact source pixels inside a glyph mask with an outer feather."""
    if source_rgb.shape != background_rgb.shape or source_rgb.shape[:2] != glyph_mask.shape:
        raise ValueError("source, background, and glyph mask dimensions must match")
    blurred = cv2.GaussianBlur(glyph_mask, (0, 0), feather) if feather > 0 else glyph_mask
    alpha = np.maximum(glyph_mask, blurred).astype(np.float32) / 255.0
    alpha = alpha[..., None]
    combined = source_rgb.astype(np.float32) * alpha + background_rgb.astype(np.float32) * (1.0 - alpha)
    return np.clip(combined, 0, 255).astype(np.uint8)


def source_box_mask(
    shape: tuple[int, int],
    boxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    """Build a padded text-line mask for aligned regenerated layer compositing."""
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        pad = max(8, (y2 - y1) // 4)
        x1, y1, x2, y2 = _clip_box((x1, y1, x2, y2), width, height, pad=pad)
        mask[y1:y2, x1:x2] = 255
    return mask


def _clip_box(box: tuple[int, int, int, int], width: int, height: int, pad: int = 0) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return max(0, x1 - pad), max(0, y1 - pad), min(width, x2 + pad), min(height, y2 + pad)


def foreground_mask(source_rgb: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    height, width = source_rgb.shape[:2]
    line_height = box[3] - box[1]
    x1, y1, x2, y2 = _clip_box(box, width, height, pad=max(6, int(line_height * 0.12)))
    gray = cv2.cvtColor(source_rgb[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
    ring_pad = max(8, min(24, (y2 - y1) // 5))
    rx1, ry1, rx2, ry2 = _clip_box((x1, y1, x2, y2), width, height, pad=ring_pad)
    context = cv2.cvtColor(source_rgb[ry1:ry2, rx1:rx2], cv2.COLOR_RGB2GRAY)
    ring = np.ones(context.shape, dtype=bool)
    ring[y1 - ry1 : y2 - ry1, x1 - rx1 : x2 - rx1] = False
    background_luma = float(np.median(context[ring])) if ring.any() else float(np.median(gray))
    low, high = float(np.percentile(gray, 4)), float(np.percentile(gray, 96))
    dark_contrast, light_contrast = background_luma - low, high - background_luma
    contrast = max(light_contrast, dark_contrast)
    threshold = max(24.0, min(72.0, contrast * 0.32))
    if light_contrast > dark_contrast:
        mask = (gray.astype(np.float32) >= background_luma + threshold).astype(np.uint8) * 255
    else:
        mask = (gray.astype(np.float32) <= background_luma - threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    dilation = 5 if line_height >= 48 else 3
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation + 1,) * 2))
    result = np.zeros((height, width), dtype=np.uint8)
    result[y1:y2, x1:x2] = mask
    return result


def _line_language(line: TextLine) -> str:
    if line.script == "cjk":
        return "ch"
    return "ru" if any("CYRILLIC" in unicodedata.name(character, "") for character in line.text) else "en"


def _recognition_box(
    line: TextLine,
    width: int,
    height: int,
    vertical_pad_ratio: float | None = None,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = line.box
    line_height = y2 - y1
    if line.script == "cjk":
        left_pad = max(16, round(line_height * 0.2))
        right_pad = max(16, round(line_height * 0.6))
        return max(0, x1 - left_pad), y1, min(width, x2 + right_pad), y2
    pad_x = max(16, line_height)
    pad_y = max(8, line_height // 3) if vertical_pad_ratio is None else max(8, round(line_height * vertical_pad_ratio))
    return max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y)


def _recognize(
    engine: Any,
    image: np.ndarray,
    line: TextLine,
    vertical_pad_ratio: float | None = None,
) -> tuple[str, float]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = _recognition_box(line, width, height, vertical_pad_ratio)
    crop = image[y1:y2, x1:x2]
    if crop.shape[0] < 64:
        scale = 64 / crop.shape[0]
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    result = next(iter(engine.predict(crop)))
    return str(result.get("rec_text", "")), float(result.get("rec_score", 0.0))


def _sample_text_color(
    source_rgb: np.ndarray,
    mask: np.ndarray,
    box: tuple[int, int, int, int],
) -> tuple[int, int, int]:
    height, width = source_rgb.shape[:2]
    x1, y1, x2, y2 = _clip_box(box, width, height, pad=2)
    crop = source_rgb[y1:y2, x1:x2]
    active = mask[y1:y2, x1:x2] > 0
    pixels = crop[active]
    luma = pixels.mean(axis=1)
    background_luma = float(crop[[0, -1], :, :].reshape(-1, 3).mean(axis=1).mean())
    if background_luma >= 128:
        selected = pixels[luma <= np.percentile(luma, 20)]
    else:
        selected = pixels[luma >= np.percentile(luma, 80)]
    return tuple(int(value) for value in np.median(selected, axis=0))


def _render_line(image: Image.Image, line: TextLine, color: tuple[int, int, int]) -> None:
    font_path = CJK_FONT if line.script == "cjk" else (BOLD_FONT if line.box[3] - line.box[1] >= 55 else REGULAR_FONT)
    target_width, target_height = line.box[2] - line.box[0], line.box[3] - line.box[1]
    draw = ImageDraw.Draw(image)
    low, high = 4, max(8, target_height * 2)
    font = ImageFont.truetype(str(font_path), low)
    while low <= high:
        size = (low + high) // 2
        candidate = ImageFont.truetype(str(font_path), size)
        bounds = draw.textbbox((0, 0), line.text, font=candidate)
        if bounds[2] - bounds[0] <= target_width * 1.03 and bounds[3] - bounds[1] <= target_height * 1.08:
            font, low = candidate, size + 1
        else:
            high = size - 1
    bounds = draw.textbbox((0, 0), line.text, font=font)
    y = line.box[1] + math.floor((target_height - (bounds[3] - bounds[1])) / 2) - bounds[1]
    draw.text((line.box[0], y), line.text, fill=color, font=font)


def _write_manifest(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _vertical_overlap_ratio(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    overlap = max(0, min(left[3], right[3]) - max(left[1], right[1]))
    return overlap / max(1, min(left[3] - left[1], right[3] - right[1]))


def group_word_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    groups: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda item: ((item[1] + item[3]) / 2, item[0])):
        matches = []
        for index, group in enumerate(groups):
            if _vertical_overlap_ratio(box, group) < 0.45:
                continue
            horizontal_gap = max(0, max(box[0], group[0]) - min(box[2], group[2]))
            line_height = min(box[3] - box[1], group[3] - group[1])
            if horizontal_gap <= max(24, line_height * 3):
                matches.append(index)
        if not matches:
            groups.append(box)
            continue
        index = max(matches, key=lambda item: _vertical_overlap_ratio(box, groups[item]))
        x1, y1, x2, y2 = groups[index]
        groups[index] = min(x1, box[0]), min(y1, box[1]), max(x2, box[2]), max(y2, box[3])
    return sorted(groups, key=lambda item: ((item[1] + item[3]) / 2, item[0]))


def detect_line_boxes(
    engine: Any,
    source_rgb: np.ndarray,
    expected_count: int | None = None,
) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for page in engine.predict(source_rgb):
        detected = page.get("rec_boxes", None)
        if detected is None or len(detected) == 0:
            detected = page.get("rec_polys", [])
        for score, raw_box in zip(page.get("rec_scores", []), detected, strict=False):
            if float(score) < 0.5:
                continue
            points = np.asarray(raw_box, dtype=np.float32).reshape(-1)
            if points.size == 4:
                x1, y1, x2, y2 = points
            else:
                points = points.reshape(-1, 2)
                x1, y1 = points.min(axis=0)
                x2, y2 = points.max(axis=0)
            boxes.append((round(float(x1)), round(float(y1)), round(float(x2)), round(float(y2))))
    lines = group_word_boxes(boxes)
    if expected_count is not None and len(lines) != expected_count:
        raise click.ClickException(f"detected {len(lines)} source lines; expected exactly {expected_count}")
    return lines


def _load_lines(path: Path, key: str) -> list[TextLine]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        TextLine(tuple(item["box"]), item["text"], item["script"], float(item.get("angle", 0.0)))
        for item in payload[key]
    ]


@click.command()
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("candidate", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--lines-json",
    default=ROOT / "data/evaluations/fidelity/text-lines.json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--source-key", help="Key in lines JSON; defaults to source basename.")
@click.option(
    "--detect-boxes",
    is_flag=True,
    help="Detect source boxes; fail unless their count matches verified lines.",
)
@click.option(
    "--restoration",
    type=click.Choice(("vae-glyphs", "source-glyphs", "source-silhouette", "rerender")),
    required=True,
    help="Choose VAE glyph cores, regenerated pixels, fresh source shapes, or the system-font control.",
)
@click.option(
    "--glyph-donor",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="VAE reconstruction used only by --restoration vae-glyphs.",
)
@click.option(
    "--glyph-feather",
    type=click.FloatRange(min=0.0),
    default=0.5,
    show_default=True,
    help="Outer donor-edge feather used only by --restoration vae-glyphs.",
)
@click.option(
    "--selection",
    type=click.Choice(("all", "changed")),
    default="all",
    show_default=True,
    help="Restore every verified line or only OCR-confirmed changes.",
)
@click.option(
    "--erase-background/--keep-background",
    default=True,
    show_default=True,
    help="Erase candidate glyphs before compositing, or directly blend an aligned regenerated glyph layer.",
)
@click.option(
    "--composite-mask",
    type=click.Choice(("glyphs", "boxes")),
    default="glyphs",
    show_default=True,
    help="Composite isolated glyphs or complete aligned text-line boxes.",
)
@click.option("--manifest", type=click.Path(dir_okay=False, path_type=Path))
def main(
    source: Path,
    candidate: Path,
    output: Path,
    lines_json: Path,
    source_key: str | None,
    detect_boxes: bool,
    restoration: str,
    glyph_donor: Path | None,
    glyph_feather: float,
    selection: str,
    erase_background: bool,
    composite_mask: str,
    manifest: Path | None,
) -> None:
    """Restore SOURCE text over the scrubbed CANDIDATE."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if restoration == "rerender":
        for font in (REGULAR_FONT, BOLD_FONT, CJK_FONT):
            if not font.exists():
                raise click.ClickException(f"required evaluation font is unavailable: {font}")
    if restoration == "vae-glyphs" and glyph_donor is None:
        raise click.ClickException("--glyph-donor is required for --restoration vae-glyphs")
    if restoration != "vae-glyphs" and glyph_donor is not None:
        raise click.ClickException("--glyph-donor is only valid with --restoration vae-glyphs")
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    source_rgb = np.asarray(Image.open(source).convert("RGB"))
    candidate_rgb = np.asarray(Image.open(candidate).convert("RGB"))
    if candidate_rgb.shape != source_rgb.shape:
        raise click.ClickException("source and candidate dimensions must match")
    donor_rgb = np.asarray(Image.open(glyph_donor).convert("RGB")) if glyph_donor else None
    if donor_rgb is not None and donor_rgb.shape != source_rgb.shape:
        raise click.ClickException("source and glyph donor dimensions must match")
    lines = _load_lines(lines_json, source_key or source.name)
    annotation_boxes = [line.box for line in lines]
    if detect_boxes:
        from paddleocr import PaddleOCR

        page_engine = PaddleOCR(
            lang="ch",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        boxes = detect_line_boxes(page_engine, source_rgb, len(lines))
        lines = [TextLine(box, line.text, line.script) for line, box in zip(lines, boxes, strict=True)]
    vertical_pad_ratio = 0.1 if detect_boxes else None
    decisions = []
    selected = list(lines)
    if selection == "all":
        decisions = [{"line": asdict(line), "selected": True, "reason": "all-lines"} for line in lines]
    else:
        from paddleocr import TextRecognition

        engines = {
            "en": TextRecognition(model_name="en_PP-OCRv5_mobile_rec"),
            "ru": TextRecognition(model_name="eslav_PP-OCRv5_mobile_rec"),
            "ch": TextRecognition(model_name="PP-OCRv5_server_rec"),
        }
        selected = []
        for line in lines:
            language = _line_language(line)
            source_text, source_score = _recognize(engines[language], source_rgb, line, vertical_pad_ratio)
            candidate_text, candidate_score = _recognize(engines[language], candidate_rgb, line, vertical_pad_ratio)
            preserve = should_preserve_line(line.text, source_text, source_score, candidate_text, candidate_score)
            decisions.append(
                {
                    "line": asdict(line),
                    "source_text": source_text,
                    "source_score": source_score,
                    "candidate_text": candidate_text,
                    "candidate_score": candidate_score,
                    "preserve": preserve,
                    "selected": not preserve,
                }
            )
            if not preserve:
                selected.append(line)
    output.parent.mkdir(parents=True, exist_ok=True)
    mask_path = output.with_name(output.stem + "_mask.png")
    manifest_common = {
        "source": source.name,
        "candidate": candidate.name,
        "output": output.name,
        "mask": mask_path.name,
        "glyph_donor": glyph_donor.name if glyph_donor else None,
        "glyph_feather": glyph_feather if restoration == "vae-glyphs" else None,
        "restoration": restoration,
        "selection": selection,
        "erase_background": erase_background,
        "composite_mask": composite_mask,
        "box_source": "detector" if detect_boxes else "verified_annotations",
        "annotation_boxes": annotation_boxes,
        "decisions": decisions,
    }
    if not selected:
        combined = np.zeros(source_rgb.shape[:2], dtype=np.uint8)
        shutil.copyfile(candidate, output)
        Image.fromarray(combined).save(mask_path)
        payload = {
            **manifest_common,
            "mask_fraction": 0.0,
            "source_glyph_fraction": 0.0,
            "source_layer_fraction": 0.0,
        }
        _write_manifest(manifest, payload)
        log.info("Copied %s unchanged because every line passed", output)
        return
    if restoration in {"source-silhouette", "vae-glyphs"}:
        source_masks = [source_silhouette_mask(source_rgb, line.box, line.angle) for line in selected]
        candidate_masks = [source_silhouette_mask(candidate_rgb, line.box, line.angle) for line in selected]
        line_masks = []
        for line, source_mask, candidate_mask in zip(selected, source_masks, candidate_masks, strict=True):
            radius = 5 if line.box[3] - line.box[1] >= 48 else 3
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
            line_masks.append(cv2.dilate(np.maximum(source_mask, candidate_mask), kernel))
        masks = line_masks
    else:
        source_masks = [foreground_mask(source_rgb, line.box) for line in selected]
        candidate_masks = [foreground_mask(candidate_rgb, line.box) for line in selected]
        masks = [np.maximum(left, right) for left, right in zip(source_masks, candidate_masks, strict=True)]
    del candidate_masks
    groups = group_text_lines(selected)
    if erase_background:
        background = cv2.cvtColor(candidate_rgb, cv2.COLOR_RGB2BGR)
        for group in groups:
            background = region_eraser.erase_lama(
                background,
                np.maximum.reduce([masks[index] for index in group]),
            )
        background_rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
        residual_masks = [
            residual_glyph_mask(background_rgb, mask, line.box) for line, mask in zip(selected, masks, strict=True)
        ]
        for group in groups:
            residual = np.maximum.reduce([residual_masks[index] for index in group])
            if np.any(residual):
                background = region_eraser.erase_lama(background, residual)
        background_rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    else:
        background_rgb = candidate_rgb
        residual_masks = []
    source_glyph_mask = np.maximum.reduce(source_masks)
    source_layer_mask = source_glyph_mask
    if restoration == "source-glyphs":
        source_layer_mask = (
            source_box_mask(source_rgb.shape[:2], [line.box for line in selected])
            if composite_mask == "boxes"
            else source_glyph_mask
        )
        restored = composite_source_glyphs(source_rgb, background_rgb, source_layer_mask, feather=3.0)
        Image.fromarray(restored).save(output)
    elif restoration in {"source-silhouette", "vae-glyphs"}:
        restored = composite_fresh_text_edges(source_rgb, background_rgb, selected, source_masks)
        if restoration == "vae-glyphs":
            if donor_rgb is None:
                raise RuntimeError("VAE glyph restoration requires a loaded donor")
            restored = composite_reconstructed_glyphs(
                donor_rgb,
                restored,
                source_layer_mask,
                feather=glyph_feather,
            )
        Image.fromarray(restored).save(output)
    else:
        rendered = Image.fromarray(background_rgb)
        for line, source_mask in zip(selected, source_masks, strict=True):
            _render_line(rendered, line, _sample_text_color(source_rgb, source_mask, line.box))
        rendered.save(output)
    combined = np.maximum.reduce([*masks, *residual_masks])
    if restoration == "source-glyphs" and not erase_background:
        combined = source_layer_mask
    Image.fromarray(combined).save(mask_path)
    payload = {
        **manifest_common,
        "mask_fraction": float((combined > 0).mean()),
        "source_glyph_fraction": float((source_glyph_mask > 0).mean()),
        "source_layer_fraction": float((source_layer_mask > 0).mean()),
    }
    _write_manifest(manifest, payload)
    log.info("Wrote %s with %.4f edited fraction", output, payload["mask_fraction"])


if __name__ == "__main__":
    main()
