"""Regression guards for malformed-length DoS and removal-truncation bugs.

Three verified bugs are locked in here:

1. PNG C2PA parsers (``c2pa.has_c2pa_metadata`` / ``extract_c2pa_info`` and
   ``metadata._png_late_metadata`` via ``scan_head``) used the raw 32-bit chunk
   ``length`` field directly in ``f.read(length)``. A crafted file can declare
   ``length = 0x7FFFFFFF`` (~2 GiB) on a 60-byte file, forcing a multi-GB
   allocation. The fix clamps ``length`` to the bytes actually remaining.

2. ISOBMFF ``strip_c2pa_boxes`` truncated the file from a malformed box to EOF
   (the box walker returns early), so ``remove_ai_metadata`` could emit a
   shorter file and report success. The fix returns the input unchanged when the
   walk does not reach EOF.
"""

from __future__ import annotations

import struct
import tracemalloc
from typing import TYPE_CHECKING

from pagedmark import metadata
from pagedmark._internal import c2pa, isobmff

if TYPE_CHECKING:
    from pathlib import Path

PNG_SIG = b"\x89PNG\r\n\x1a\n"
_HUGE = 0x7FFFFFFF  # ~2 GiB declared length on a tiny file


def _png_with_huge_c2pa_chunk() -> bytes:
    """A ~60-byte 'PNG' whose caBX chunk header lies about its length."""
    header = struct.pack(">I", _HUGE) + c2pa.C2PA_CHUNK_TYPE
    body = b"jumbc2pa-not-really"  # far shorter than the declared length
    return PNG_SIG + header + body


class TestPngLengthClampNoAlloc:
    """Clamping makes the parsers read only the real bytes, not the lie."""

    def test_has_c2pa_metadata_is_bounded(self, tmp_path):
        path = tmp_path / "evil.png"
        path.write_bytes(_png_with_huge_c2pa_chunk())

        tracemalloc.start()
        try:
            # Must return quickly without allocating gigabytes and without raising.
            c2pa.has_c2pa_metadata(path)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 50 * 1024 * 1024  # < 50 MB locks in the clamp

    def test_extract_c2pa_info_is_bounded(self, tmp_path):
        path = tmp_path / "evil.png"
        path.write_bytes(_png_with_huge_c2pa_chunk())

        tracemalloc.start()
        try:
            c2pa.extract_c2pa_info(path)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 50 * 1024 * 1024

    def test_extract_c2pa_chunk_is_bounded(self, tmp_path):
        path = tmp_path / "evil.png"
        path.write_bytes(_png_with_huge_c2pa_chunk())

        tracemalloc.start()
        try:
            c2pa.extract_c2pa_chunk(path)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 50 * 1024 * 1024

    def test_png_late_metadata_scan_is_bounded(self, tmp_path):
        # A PNG with a real IDAT pushing the late-scan window past 1 MB, then a
        # tEXt chunk lying about its length. scan_head() -> _png_late_metadata().
        idat = b"\x00" * (1024 * 1024 + 16)
        text_header = struct.pack(">I", _HUGE) + b"tEXt"
        blob = (
            PNG_SIG
            + struct.pack(">I", len(idat))
            + b"IDAT"
            + idat
            + b"\x00\x00\x00\x00"  # fake CRC
            + text_header
            + b"AIGC short"
        )
        path = tmp_path / "evil_late.png"
        path.write_bytes(blob)

        tracemalloc.start()
        try:
            metadata.scan_head(path)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        # head itself is ~1 MB; the clamp keeps the late read tiny. Generous cap.
        assert peak < 50 * 1024 * 1024


def _box(box_type: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + box_type + payload


class TestIsobmffStripFailSafe:
    def test_well_formed_file_still_strips_uuid(self):
        ftyp = _box(b"ftyp", b"isom\x00\x00\x00\x00mp42")
        c2pa_box = _box(b"uuid", isobmff.C2PA_UUID + b"manifest-bytes")
        mdat = _box(b"mdat", b"\x00" * 32)
        data = ftyp + c2pa_box + mdat

        cleaned, stripped = isobmff.strip_c2pa_boxes(data)
        assert stripped == 1
        assert len(cleaned) == len(data) - len(c2pa_box)
        assert isobmff.C2PA_UUID not in cleaned

    def test_malformed_box_does_not_truncate_tail(self):
        ftyp = _box(b"ftyp", b"isom\x00\x00\x00\x00mp42")
        c2pa_box = _box(b"uuid", isobmff.C2PA_UUID + b"manifest-bytes")
        # A box claiming ~2 GiB before EOF: the walker returns early here.
        bad_box = struct.pack(">I", _HUGE) + b"free" + b"\x00" * 16
        data = ftyp + c2pa_box + bad_box

        cleaned, stripped = isobmff.strip_c2pa_boxes(data)
        # Fail-safe: input returned unchanged, nothing stripped, no truncation.
        assert stripped == 0
        assert cleaned == data
        assert len(cleaned) == len(data)


class TestRiffLateMetadataIsBounded:
    """A metadata scan must not become a near-full-file read because one chunk lied
    about its length. This runs on the memoized verdict path, over images from
    arbitrary sources."""

    def _webp_with_declared_length(self, path: Path, declared: int, payload: bytes) -> Path:
        chunk = b"XMP " + declared.to_bytes(4, "little") + payload
        body = b"WEBP" + b"VP8 " + (4).to_bytes(4, "little") + b"\x00\x00\x00\x00" + chunk
        path.write_bytes(b"RIFF" + len(body).to_bytes(4, "little") + body)
        return path

    def test_a_chunk_claiming_the_whole_file_is_clamped_to_what_remains(self, tmp_path: Path):
        target = self._webp_with_declared_length(tmp_path / "liar.webp", 1 << 30, b"AI" * 64)

        collected = metadata._riff_late_metadata(target, 12)

        assert collected == b"AI" * 64

    def test_the_total_is_capped(self, tmp_path: Path):
        payload = b"x" * 4096
        target = self._webp_with_declared_length(tmp_path / "big.webp", len(payload), payload)

        assert len(metadata._riff_late_metadata(target, 12, max_total=512)) == 512
