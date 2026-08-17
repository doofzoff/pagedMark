"""Read image metadata without changing the source container."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import piexif
from PIL import Image

from pagedmark._internal.c2pa import extract_c2pa_chunk, extract_c2pa_info, has_c2pa_metadata
from pagedmark._internal.constants import AI_KEYWORDS, AI_METADATA_KEYS

if TYPE_CHECKING:
    from pathlib import Path

_EXIF_KEY = "exif"
_AI_KEYS_CASEFOLD = frozenset(key.casefold() for key in AI_METADATA_KEYS)


def _read_pillow_info(source_path: Path) -> dict[str, Any]:
    with Image.open(source_path) as image:
        return {key: value for key, value in image.info.items() if isinstance(key, str)}


def _decode_exif(raw: object) -> tuple[str, object]:
    if not isinstance(raw, bytes):
        return "exif_raw", raw
    try:
        return _EXIF_KEY, piexif.load(raw)
    except Exception:
        return "exif_raw", raw


def _is_ai_field(key: str) -> bool:
    folded = key.casefold()
    return folded in _AI_KEYS_CASEFOLD or any(token in folded for token in AI_KEYWORDS)


def _attach_c2pa(source_path: Path, metadata: dict[str, Any]) -> None:
    if not has_c2pa_metadata(source_path):
        return
    metadata["c2pa"] = extract_c2pa_info(source_path)
    payload = extract_c2pa_chunk(source_path)
    if payload is not None:
        metadata["c2pa_chunk"] = payload


def extract_metadata(source_path: Path) -> dict[str, Any]:
    """Return every Pillow-visible field plus decoded EXIF and C2PA data."""
    raw_info = _read_pillow_info(source_path)
    metadata = dict(raw_info)
    if _EXIF_KEY in raw_info:
        metadata.pop(_EXIF_KEY, None)
        decoded_key, decoded_value = _decode_exif(raw_info[_EXIF_KEY])
        metadata[decoded_key] = decoded_value

    _attach_c2pa(source_path, metadata)
    return metadata


def extract_ai_metadata(source_path: Path) -> dict[str, Any]:
    """Return only metadata keys recognized as AI provenance or generation data."""
    metadata = {key: value for key, value in _read_pillow_info(source_path).items() if _is_ai_field(key)}
    _attach_c2pa(source_path, metadata)
    return metadata


def has_ai_metadata(image_path: Path) -> bool:
    """Return whether a supported metadata signal is present."""
    if any(_is_ai_field(key) for key in _read_pillow_info(image_path)):
        return True
    return has_c2pa_metadata(image_path)


def _summary_value(value: object) -> str:
    if isinstance(value, bytes):
        return f"<binary data ({len(value)} bytes)>"
    text = str(value)
    return text if len(text) <= 100 else f"{text[:100]}..."


def get_ai_metadata_summary(source_path: Path) -> str:
    """Format the AI-only metadata view for the command-line report."""
    metadata = extract_ai_metadata(source_path)
    if not metadata:
        return "No AI metadata found."

    lines = ["AI Image Metadata:", "-" * 40]
    for key, value in metadata.items():
        if key == "c2pa_chunk":
            continue
        if key == "c2pa" and isinstance(value, dict):
            lines.append("C2PA Metadata:")
            c2pa_fields = cast("dict[str, object]", value)
            lines.extend(f"  {name}: {_summary_value(item)}" for name, item in c2pa_fields.items())
            continue
        lines.append(f"{key}: {_summary_value(value)}")
    return "\n".join(lines)
