"""Shared fixtures for pagedmark test suite."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo


@pytest.fixture
def clean_photo(tmp_path: Path) -> Path:
    """Create a deterministic image with no provenance metadata.

    The assertions using this fixture exercise metadata and provenance behavior,
    so a generated fixture is sufficient and avoids committing personal photos.
    """
    path = tmp_path / "clean-control.png"
    Image.new("RGB", (128, 96), color=(90, 140, 190)).save(path)
    return path


@pytest.fixture
def tmp_image_path(tmp_path: Path) -> Path:
    """Create a minimal 200x200 test PNG image and return its path."""
    img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    path = tmp_path / "test_image.png"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def tmp_large_image_path(tmp_path: Path) -> Path:
    """Create a 1200x1200 test PNG image (triggers large watermark branch)."""
    img = np.random.randint(0, 255, (1200, 1200, 3), dtype=np.uint8)
    path = tmp_path / "test_large.png"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def tmp_jpeg_path(tmp_path: Path) -> Path:
    """Create a minimal JPEG test image."""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def tmp_png_with_ai_metadata(tmp_path: Path) -> Path:
    """Create a PNG with AI-related metadata keys."""
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    pnginfo = PngInfo()
    pnginfo.add_text("parameters", "Steps: 20, Sampler: Euler, CFG scale: 7")
    pnginfo.add_text("prompt", "a beautiful landscape")
    pnginfo.add_text("Author", "Test Author")
    path = tmp_path / "ai_metadata.png"
    img.save(path, pnginfo=pnginfo)
    return path


@pytest.fixture
def tmp_clean_png(tmp_path: Path) -> Path:
    """Create a PNG with no AI metadata."""
    img = Image.new("RGB", (64, 64), color=(200, 100, 50))
    pnginfo = PngInfo()
    pnginfo.add_text("Author", "Human Artist")
    pnginfo.add_text("Title", "Test Artwork")
    path = tmp_path / "clean.png"
    img.save(path, pnginfo=pnginfo)
    return path


@pytest.fixture
def tampered_chatgpt_png(tmp_path: Path) -> Path:
    """Add valid PNG metadata after signing so the C2PA asset hash no longer matches."""
    source = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "provenance" / "chatgpt-1.png"
    data = source.read_bytes()
    iend = data.rfind(b"\x00\x00\x00\x00IEND")
    assert iend >= 0

    kind = b"tEXt"
    payload = b"c2pa-test\x00benign post-signing metadata mutation"
    chunk = (
        struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )
    target = tmp_path / "tampered-chatgpt.png"
    target.write_bytes(data[:iend] + chunk + data[iend:])
    assert target.read_bytes() != data
    return target
