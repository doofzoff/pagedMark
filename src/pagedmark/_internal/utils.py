"""Small path helpers shared by optional image pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pagedmark._internal.constants import SUPPORTED_FORMATS

if TYPE_CHECKING:
    from pathlib import Path

_PIL_FORMAT_BY_SUFFIX = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
}


def is_supported_format(file_path: Path) -> bool:
    """Return whether ``file_path`` has a supported raster suffix."""
    return file_path.suffix.casefold() in SUPPORTED_FORMATS


def get_image_format(file_path: Path) -> str:
    """Return the Pillow save format used by the legacy metadata API.

    The metadata writer only has specialized PNG and JPEG paths. Other accepted
    inputs therefore use its PNG fallback, matching the established API contract.
    """
    return _PIL_FORMAT_BY_SUFFIX.get(file_path.suffix.casefold(), "PNG")
