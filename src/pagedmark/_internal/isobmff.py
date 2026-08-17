"""Minimal ISOBMFF box walker for AI provenance in AVIF / HEIF / MP4 / JPEG-XL.

The ISO Base Media File Format wraps content in nested ``[size:4][type:4][...]``
boxes. C2PA stores its manifest in a top-level ``uuid`` box keyed by the
C2PA UUID; JPEG-XL uses a ``jumb`` box (JUMBF) instead. To strip provenance
without re-encoding, the image path drops matching boxes and emits the rest
verbatim. The streaming MP4/MOV/M4A path instead preserves all offsets by
retyping matching boxes as ``free`` and blanking their payloads in place. The
codestream (``mdat`` for ISOBMFF, ``jxlc`` / ``jxlp`` for JPEG-XL) is untouched,
so pixel, video, and audio data is preserved bit-for-bit.

TC260-PG-20257A video metadata is nested instead:
``moov.udta.meta.keys/ilst``. Its detector seeks through those boxes without
reading media payloads, and its stripper blanks the validated key/value in
place so fast-start media offsets remain valid.

This file intentionally avoids dependencies on format-specific libraries
(pillow-heif, pillow-jxl, pymp4) so it works on systems where they aren't
installed.

Reference: ISO/IEC 14496-12 (ISOBMFF) and C2PA 2.1 spec §11.
"""

from __future__ import annotations

import io
import logging
import os
import re
import shutil
import struct
from typing import TYPE_CHECKING, Any, BinaryIO

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

from pagedmark.metadata import (
    AIGC_MARKERS,
    C2PA_UUID,
    IPTC_AI_FIELD_MARKERS,
    IPTC_AI_MARKERS,
    MAX_TC260_VALUE_BYTES,
    parse_tc260_aigc_json,
)

logger = logging.getLogger(__name__)

# Top-level box types that may carry AI provenance. ``uuid`` boxes are checked
# against ``C2PA_UUID`` / AI-label markers before being stripped; ``jumb`` boxes
# are always stripped (JPEG-XL uses them exclusively for JUMBF).
C2PA_BOX_TYPES: frozenset[bytes] = frozenset({b"uuid", b"jumb"})

# AI-label byte markers (TC260 AIGC, IPTC "Made with AI", IPTC 2025.1 AI fields)
# whose presence inside an XMP ``uuid`` box means the box carries an AI label.
# Matching the payload rather than a fixed XMP UUID avoids the XMP-box UUID
# byte-order ambiguity and stays surgical: only AI-bearing XMP is dropped, plain
# XMP (copyright, camera info) is kept.
_AI_LABEL_MARKERS: tuple[bytes, ...] = AIGC_MARKERS + IPTC_AI_MARKERS + IPTC_AI_FIELD_MARKERS

# Adobe XMP packet delimiters (XMP spec part 3). In HEIF/AVIF the XMP packet
# sits inside a ``meta``-box ``mime`` item whose bytes live in ``mdat`` / ``idat``,
# out of reach of the top-level box stripper, so an AI-label packet there is
# blanked in place (see ``blank_ai_xmp_packets``).
_XMP_PACKET_RE = re.compile(rb"<\?xpacket begin=.*?<\?xpacket end=[^>]*?\?>", re.DOTALL)
_STREAM_COPY_BYTES = 1024 * 1024
STREAM_SCAN_BYTES = 4 * 1024 * 1024


# TC260-PG-20257A stores an MP4/MOV label as an ``AIGC`` key in
# ``moov.udta.meta.keys`` and its JSON value in the corresponding
# ``moov.udta.meta.ilst`` item. The value is intentionally bounded before it is
# read: the normative object is tiny, and a corrupt size must not allocate an
# arbitrary amount of memory during an inspection.
def _iter_top_level_boxes(data: bytes) -> Iterator[tuple[int, int, bytes, int]]:
    """Yield ``(start, end, type, payload_offset)`` for each top-level box.

    Handles all three ISOBMFF box-size encodings:
    - ``size > 1``: 32-bit size field is the total box length.
    - ``size == 1``: 64-bit ``largesize`` follows after the type field.
    - ``size == 0``: box runs to end of file.
    """
    pos = 0
    n = len(data)
    while pos + 8 <= n:
        size32 = struct.unpack_from(">I", data, pos)[0]
        box_type = data[pos + 4 : pos + 8]
        if size32 == 1:
            if pos + 16 > n:
                return
            size = struct.unpack_from(">Q", data, pos + 8)[0]
            payload_off = pos + 16
        elif size32 == 0:
            size = n - pos
            payload_off = pos + 8
        else:
            size = size32
            payload_off = pos + 8
        if size < (payload_off - pos) or pos + size > n:
            return
        yield pos, pos + size, box_type, payload_off
        pos += size


def _read_box_header(
    stream: BinaryIO,
    pos: int,
    limit: int,
) -> tuple[int, bytes, int] | None:
    """Return ``(end, type, payload_offset)`` for one box inside ``limit``."""
    if pos < 0 or pos + 8 > limit:
        return None
    stream.seek(pos)
    header = stream.read(8)
    if len(header) != 8:
        return None
    size32 = struct.unpack(">I", header[:4])[0]
    box_type = header[4:8]
    payload_off = pos + 8
    if size32 == 1:
        extended = stream.read(8)
        if len(extended) != 8:
            return None
        size = struct.unpack(">Q", extended)[0]
        payload_off = pos + 16
    elif size32 == 0:
        size = limit - pos
    else:
        size = size32
    end = pos + size
    if size < payload_off - pos or end > limit:
        return None
    return end, box_type, payload_off


def iter_file_boxes(
    stream: BinaryIO,
    start: int,
    end: int,
) -> Iterator[tuple[int, int, bytes, int]]:
    """Yield valid boxes from one bounded container region."""
    pos = start
    while pos + 8 <= end:
        header = _read_box_header(stream, pos, end)
        if header is None:
            return
        box_end, box_type, payload_off = header
        yield pos, box_end, box_type, payload_off
        pos = box_end


def _tc260_key_indices(
    stream: BinaryIO,
    payload_off: int,
    box_end: int,
) -> dict[int, tuple[int, int]]:
    """Map every exact ``AIGC`` key index to its byte span."""
    if payload_off + 8 > box_end:
        return {}
    stream.seek(payload_off)
    prefix = stream.read(8)
    if len(prefix) != 8:
        return {}
    entry_count = struct.unpack(">I", prefix[4:8])[0]
    pos = payload_off + 8
    found: dict[int, tuple[int, int]] = {}
    for index in range(1, entry_count + 1):
        if pos + 8 > box_end:
            return {}
        stream.seek(pos)
        header = stream.read(8)
        if len(header) != 8:
            return {}
        entry_size = struct.unpack(">I", header[:4])[0]
        entry_end = pos + entry_size
        if entry_size < 8 or entry_end > box_end:
            return {}
        name_start = pos + 8
        if entry_end - name_start == 4:
            stream.seek(name_start)
            if stream.read(4) == b"AIGC":
                found[index] = (name_start, entry_end)
        pos = entry_end
    return found


def _tc260_aigc_regions(
    stream: BinaryIO,
    file_size: int,
) -> list[tuple[int, int, int, int, bytes]]:
    """Locate validated native TC260 entries without reading media payloads.

    Each tuple is ``(key_start, key_end, value_start, value_end, value)``.
    """
    regions: list[tuple[int, int, int, int, bytes]] = []
    for _moov_start, moov_end, moov_type, moov_payload in iter_file_boxes(stream, 0, file_size):
        if moov_type != b"moov":
            continue
        for _udta_start, udta_end, udta_type, udta_payload in iter_file_boxes(
            stream,
            moov_payload,
            moov_end,
        ):
            if udta_type != b"udta":
                continue
            for _meta_start, meta_end, meta_type, meta_payload in iter_file_boxes(
                stream,
                udta_payload,
                udta_end,
            ):
                if meta_type != b"meta" or meta_payload + 4 > meta_end:
                    continue
                keys: dict[int, tuple[int, int]] = {}
                ilst_boxes: list[tuple[int, int]] = []
                for _child_start, child_end, child_type, child_payload in iter_file_boxes(
                    stream,
                    meta_payload + 4,
                    meta_end,
                ):
                    if child_type == b"keys":
                        keys.update(_tc260_key_indices(stream, child_payload, child_end))
                    elif child_type == b"ilst":
                        ilst_boxes.append((child_payload, child_end))
                if not keys:
                    continue
                for ilst_payload, ilst_end in ilst_boxes:
                    for _item_start, item_end, item_type, item_payload in iter_file_boxes(
                        stream,
                        ilst_payload,
                        ilst_end,
                    ):
                        index = int.from_bytes(item_type, "big")
                        key_span = keys.get(index)
                        if key_span is None:
                            continue
                        for _data_start, data_end, data_type, data_payload in iter_file_boxes(
                            stream,
                            item_payload,
                            item_end,
                        ):
                            value_start = data_payload + 8
                            value_size = data_end - value_start
                            if data_type != b"data" or value_size < 0 or value_size > MAX_TC260_VALUE_BYTES:
                                continue
                            stream.seek(value_start)
                            value = stream.read(value_size)
                            if len(value) == value_size and parse_tc260_aigc_json(value) is not None:
                                regions.append((*key_span, value_start, data_end, value))
    return regions


def tc260_aigc_payloads(path: str | Path) -> tuple[bytes, ...]:
    """Read native TC260 ``AIGC`` JSON values from an MP4/MOV container."""
    try:
        with open(path, "rb") as stream:
            if not is_isobmff(stream.read(8)):
                return ()
            stream.seek(0, 2)
            file_size = stream.tell()
            return tuple(region[4] for region in _tc260_aigc_regions(stream, file_size))
    except OSError:
        return ()


def blank_tc260_aigc_tags(data: bytes) -> tuple[bytes, int]:
    """Blank native TC260 values in place while preserving every box offset.

    Removing a nested ``ilst`` item would shift ``mdat`` in a fast-start MP4 and
    invalidate its chunk offsets. Replacing the four-byte key with ``free`` and
    the JSON value with spaces keeps every box size and media offset unchanged.
    """
    if not is_isobmff(data):
        return data, 0
    regions = _tc260_aigc_regions(io.BytesIO(data), len(data))
    if not regions:
        return data, 0
    out = bytearray(data)
    key_spans: set[tuple[int, int]] = set()
    for key_start, key_end, value_start, value_end, _value in regions:
        key_spans.add((key_start, key_end))
        out[key_start:key_end] = b"free"
        out[value_start:value_end] = b" " * (value_end - value_start)
    return bytes(out), len(key_spans)


def is_isobmff(data: bytes) -> bool:
    """Cheap sniff: ISOBMFF files start with an ``ftyp`` box."""
    return len(data) >= 8 and data[4:8] == b"ftyp"


def scan_c2pa_region(path: str | Path, *, max_total: int = 4 * 1024 * 1024) -> bytes:
    """Concatenated payloads of top-level ``uuid`` / ``jumb`` boxes in an ISOBMFF
    file, found by seeking past other boxes (``mdat`` etc.) by size.

    C2PA manifests and XMP packets (incl. AI labels) live in top-level ``uuid``
    boxes; JPEG-XL uses ``jumb``. In a streaming / non-faststart MP4 the manifest
    sits AFTER a multi-megabyte ``mdat``, so a fixed first-MB read misses it. This
    walks box headers (8-16 bytes each) and seeks past payloads it does not need,
    so it never loads ``mdat`` into memory and works on multi-GB files. Returns
    the relevant box payloads (capped at ``max_total``), or ``b""`` for a
    non-ISOBMFF file or on any read error.
    """
    collected = bytearray()
    try:
        with open(path, "rb") as f:
            sniff = f.read(8)
            if len(sniff) < 8 or sniff[4:8] != b"ftyp":
                return b""
            f.seek(0, 2)
            file_size = f.tell()
            pos = 0
            while pos + 8 <= file_size and len(collected) < max_total:
                f.seek(pos)
                header = f.read(8)
                if len(header) < 8:
                    break
                size32 = struct.unpack(">I", header[:4])[0]
                box_type = header[4:8]
                payload_off = pos + 8
                if size32 == 1:
                    ext = f.read(8)
                    if len(ext) < 8:
                        break
                    size = struct.unpack(">Q", ext)[0]
                    payload_off = pos + 16
                elif size32 == 0:
                    size = file_size - pos
                else:
                    size = size32
                if size < (payload_off - pos) or pos + size > file_size:
                    # Detection-only: a malformed box halts the walk, so a manifest
                    # placed after it is missed (best-effort scan; no resync).
                    break
                if box_type in C2PA_BOX_TYPES:
                    f.seek(payload_off)
                    to_read = min(pos + size - payload_off, max_total - len(collected))
                    if to_read > 0:
                        collected += f.read(to_read)
                pos += size
    except OSError:
        return b""
    return bytes(collected)


def _payload_has_ai_label(
    stream: BinaryIO,
    start: int,
    end: int,
    *,
    max_scan: int,
) -> bool:
    """Scan a bounded prefix of one metadata payload for an AI-label marker."""
    longest_marker = max(len(marker) for marker in _AI_LABEL_MARKERS)
    remaining = min(end - start, max_scan)
    overlap = b""
    stream.seek(start)
    while remaining > 0:
        chunk = stream.read(min(_STREAM_COPY_BYTES, remaining))
        if not chunk:
            return False
        searchable = overlap + chunk
        if any(marker in searchable for marker in _AI_LABEL_MARKERS):
            return True
        overlap = searchable[-(longest_marker - 1) :]
        remaining -= len(chunk)
    return False


def _streaming_provenance_boxes(
    stream: BinaryIO,
    file_size: int,
    *,
    max_scan: int,
) -> list[tuple[int, int, int]] | None:
    """Return top-level provenance boxes, or ``None`` for a malformed walk.

    Each result is ``(box_start, payload_start, box_end)``. The walk reads only
    headers and bounded metadata prefixes, seeking over ``mdat`` payloads.
    """
    stream.seek(0)
    if not is_isobmff(stream.read(8)):
        return None
    targets: list[tuple[int, int, int]] = []
    pos = 0
    while pos < file_size:
        header = _read_box_header(stream, pos, file_size)
        if header is None:
            return None
        box_end, box_type, payload_off = header
        if box_type == b"uuid":
            stream.seek(payload_off)
            is_c2pa = payload_off + 16 <= box_end and stream.read(16) == C2PA_UUID
            has_ai_label = not is_c2pa and _payload_has_ai_label(
                stream,
                payload_off,
                box_end,
                max_scan=max_scan,
            )
            if is_c2pa or has_ai_label:
                targets.append((pos, payload_off, box_end))
        elif box_type == b"jumb":
            targets.append((pos, payload_off, box_end))
        pos = box_end
    return targets


def _overwrite_range(
    stream: BinaryIO,
    start: int,
    end: int,
    *,
    byte: bytes,
) -> None:
    """Overwrite one byte range with bounded allocations."""
    stream.seek(start)
    remaining = end - start
    block = byte * min(_STREAM_COPY_BYTES, max(remaining, 1))
    while remaining > 0:
        size = min(len(block), remaining)
        stream.write(block[:size])
        remaining -= size


def strip_isobmff_media_file(
    source: str | Path,
    output: str | Path,
    *,
    max_box_scan: int = STREAM_SCAN_BYTES,
) -> tuple[int, int]:
    """Stream-copy an MP4/MOV/M4A while removing supported AI metadata.

    The output retains every box size and byte offset. A top-level C2PA/JUMBF or
    AI-label box is converted to a ``free`` box and its payload is zeroed; native
    TC260 key/value spans are blanked in place. Keeping the original lengths is
    required because removing a pre-``mdat`` box would invalidate absolute media
    offsets in an existing sample table.

    The source is copied in bounded chunks to a sibling temporary file and
    atomically published only after all patches succeed. A malformed top-level
    walk is fail-safe: the input is copied unchanged.

    Returns ``(provenance_boxes_blanked, native_tc260_keys_blanked)``.
    """
    from pathlib import Path as _Path

    from pagedmark.video_encoding import atomic_video_output

    source_path = _Path(source)
    output_path = _Path(output)
    with source_path.open("rb") as stream:
        stream.seek(0, 2)
        file_size = stream.tell()
        targets = _streaming_provenance_boxes(
            stream,
            file_size,
            max_scan=max_box_scan,
        )
        tc260_regions = _tc260_aigc_regions(stream, file_size) if targets is not None else []
        tc260_key_spans = {(region[0], region[1]) for region in tc260_regions}

    with atomic_video_output(output_path) as temporary_path:
        with source_path.open("rb") as source_stream, temporary_path.open("r+b") as temporary:
            shutil.copyfileobj(source_stream, temporary, length=_STREAM_COPY_BYTES)
            if targets is not None:
                for box_start, payload_start, box_end in targets:
                    temporary.seek(box_start + 4)
                    temporary.write(b"free")
                    _overwrite_range(temporary, payload_start, box_end, byte=b"\x00")
                for key_start, _key_end, value_start, value_end, _value in tc260_regions:
                    temporary.seek(key_start)
                    temporary.write(b"free")
                    _overwrite_range(temporary, value_start, value_end, byte=b" ")
            temporary.flush()
            os.fsync(temporary.fileno())
        shutil.copymode(source_path, temporary_path)

    if targets is None:
        logger.warning(
            "ISOBMFF box walk failed for %s; copied input unchanged to avoid corrupting media offsets",
            source_path,
        )
        return 0, 0
    return len(targets), len(tc260_key_spans)


def strip_c2pa_boxes(data: bytes) -> tuple[bytes, int]:
    """Return ``(cleaned_bytes, stripped_count)`` with AI-provenance boxes removed.

    Walks top-level boxes and drops:
    - any ``uuid`` box whose UUID equals ``C2PA_UUID`` (a C2PA manifest);
    - any ``uuid`` box whose payload carries an AI-label marker (an XMP packet
      with a TC260 / IPTC / IPTC-2025.1 AI field -- caught by content, not by the
      XMP UUID, so it works regardless of the UUID's byte order, and leaves plain
      non-AI XMP intact);
    - any ``jumb`` box (JPEG-XL JUMBF container).

    All other boxes (incl. ``mdat`` / codestream) are emitted verbatim, so pixel
    and audio data is preserved bit-for-bit. Non-ISOBMFF input is returned
    unchanged. Despite the name this also covers MP4/MOV/M4A video and audio
    (all ISOBMFF). NOTE: this drops only top-level boxes. AI metadata stored as an
    *item inside the ``meta`` box* (typical for AVIF/HEIF) is handled separately and
    in place (same length, no offset rewrite): AI-label XMP by
    :func:`blank_ai_xmp_packets`, and AI-generator tokens in an ``Exif`` item by
    :func:`blank_ai_exif_tokens`.
    """
    if not is_isobmff(data):
        return data, 0

    out = bytearray()
    stripped = 0
    consumed = 0
    for start, end, box_type, payload_off in _iter_top_level_boxes(data):
        consumed = end
        if box_type == b"uuid":
            # uuid boxes carry the 16-byte UUID immediately after the type.
            is_c2pa = payload_off + 16 <= end and data[payload_off : payload_off + 16] == C2PA_UUID
            has_ai_label = any(marker in data[payload_off:end] for marker in _AI_LABEL_MARKERS)
            if is_c2pa or has_ai_label:
                stripped += 1
                continue
        elif box_type == b"jumb":
            stripped += 1
            continue
        out.extend(data[start:end])

    # Fail-safe: the walker returns early on a malformed box (bad size, or a box
    # that runs past EOF), so anything after it was never visited. Emitting `out`
    # would silently truncate the file from the bad box to EOF -- worse than not
    # stripping. If the walk did not consume the whole input, return it unchanged.
    if consumed != len(data):
        logger.warning(
            "ISOBMFF box walk stopped at offset %d of %d (malformed box); "
            "returning input unchanged to avoid truncation",
            consumed,
            len(data),
        )
        return data, 0

    return bytes(out), stripped


def blank_ai_xmp_packets(data: bytes) -> tuple[bytes, int]:
    """Overwrite (with spaces, in place) any XMP packet carrying an AI-label
    marker; return ``(data, blanked_count)``.

    HEIF/AVIF store XMP as a ``meta``-box ``mime`` item whose bytes live in
    ``mdat`` / ``idat``, which ``strip_c2pa_boxes`` cannot remove without
    meta-box surgery (``iinf`` / ``iloc`` rewrite). Instead, the XMP packet is
    located by its ``<?xpacket begin ... end?>`` delimiters and, when it carries
    an AI-label marker (TC260 AIGC / IPTC / IPTC-2025.1), overwritten with spaces.
    Because the replacement is the **same length**, every box size and ``iloc``
    offset stays valid and the coded image data is untouched -- only the AI label
    content is destroyed. Packets without an AI marker (plain copyright / camera
    XMP) are left intact, mirroring the top-level XMP-``uuid`` content match.
    """
    blanked = 0

    def _scrub(match: re.Match[bytes]) -> bytes:
        nonlocal blanked
        packet = match.group()
        if any(marker in packet for marker in _AI_LABEL_MARKERS):
            blanked += 1
            return b" " * len(packet)
        return packet

    return _XMP_PACKET_RE.sub(_scrub, data), blanked


# EXIF TIFF byte-order headers: little-endian (II 0x2a 0x00) and big-endian
# (MM 0x00 0x2a). A HEIF/AVIF ``Exif`` meta-box item stores its TIFF block in
# ``mdat`` / ``idat``, so the block (and these headers) appear in the raw bytes.
_TIFF_HEADERS: tuple[bytes, ...] = (b"II\x2a\x00", b"MM\x00\x2a")
# How far past a TIFF header an EXIF block plausibly extends; bounds the slice we
# hand to piexif and search within (EXIF blocks are small kilobyte-scale).
_EXIF_WINDOW = 256 * 1024


def blank_ai_exif_tokens(data: bytes) -> tuple[bytes, int]:
    """Overwrite (with spaces, in place) any AI-generator token in an EXIF block
    stored as an ISOBMFF ``meta``-box ``Exif`` item; return ``(data, blanked_count)``.

    HEIF/AVIF can carry EXIF as a ``meta``-box ``Exif`` item whose TIFF bytes live
    in ``mdat`` / ``idat`` -- out of reach of the top-level box stripper, and (when
    no pillow-heif plugin is installed) of the PIL EXIF reader too, so an AI
    ``Software`` / ``Make`` / ``Artist`` / ``ImageDescription`` tag there survived
    ``remove_ai_metadata`` (a documented gap). This locates EXIF TIFF blocks by
    their byte-order header, **validates each with piexif** (so a coincidental
    II/MM run in pixel data is ignored -- it will not parse as a TIFF IFD), and
    overwrites any AI value with spaces of the SAME length. Because the replacement
    is same-length, every box size and ``iloc`` offset stays valid and the coded image
    is untouched -- only the AI tag content is destroyed; camera/editor EXIF without an
    AI token is left intact. This mirrors ``metadata._scrub_ai_exif`` in what it removes
    -- generator tokens (``Software``/``Make``/``Artist``/``ImageDescription``), the
    China TC260 ``{"AIGC":{...}}`` block (``ImageDescription``/``UserComment``), and the
    xAI/Grok ``Signature:`` + UUID-``Artist`` pair -- since on the ISOBMFF path this is
    the ONLY EXIF scrubber (``_scrub_ai_exif`` never runs there), so without parity a
    HEIC/AVIF AIGC/xAI tag is detected but not removed.
    """
    import piexif

    # The AI-EXIF rule set is defined ONCE in metadata._ai_exif_targets and shared by both
    # EXIF scrubbers (the JPEG _scrub_ai_exif pops the tag; here we blank the value bytes),
    # so their coverage cannot drift. Imported lazily to avoid import-order coupling with
    # metadata (which imports this module); a deliberate cross-module use, not an API leak.
    from pagedmark.metadata import _ai_exif_targets  # pyright: ignore[reportPrivateUsage]

    out = bytearray(data)
    blanked = 0
    for header in _TIFF_HEADERS:
        pos = data.find(header)
        while pos != -1:
            window = bytes(out[pos : pos + _EXIF_WINDOW])
            try:
                loaded: dict[str, Any] = piexif.load(window)
            except Exception:
                loaded = {}
            for _ifd_key, _tag, value, _name in _ai_exif_targets(loaded):
                # Blank the value bytes in place, within this EXIF block only.
                vpos = out.find(value, pos, pos + _EXIF_WINDOW)
                if vpos != -1:
                    out[vpos : vpos + len(value)] = b" " * len(value)
                    blanked += 1
            pos = data.find(header, pos + len(header))
    return bytes(out), blanked
