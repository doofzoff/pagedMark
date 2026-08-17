"""Bounded Matroska/WebM metadata reader for native TC260 AIGC labels.

TC260-PG-20257A stores the label as a Matroska ``SimpleTag`` whose
``TagName`` is ``AIGC`` and whose ``TagString`` is the normative JSON object.
The walker seeks over unrelated elements such as clusters instead of reading
their payloads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, BinaryIO

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

from pagedmark.metadata import MAX_TC260_VALUE_BYTES, parse_tc260_aigc_json

_EBML_MAGIC = b"\x1aE\xdf\xa3"
_SEGMENT_ID = 0x18538067
_TAGS_ID = 0x1254C367
_TAG_ID = 0x7373
_SIMPLE_TAG_ID = 0x67C8
_TAG_NAME_ID = 0x45A3
_TAG_STRING_ID = 0x4487
_MAX_TAG_NAME_BYTES = 256


def _vint_length(first: int, *, maximum: int) -> int | None:
    """Return an EBML variable-integer length from its first byte."""
    mask = 0x80
    for length in range(1, maximum + 1):
        if first & mask:
            return length
        mask >>= 1
    return None


def _read_element_header(
    stream: BinaryIO,
    pos: int,
    limit: int,
) -> tuple[int, int, int] | None:
    """Return ``(element_id, payload_start, element_end)`` inside ``limit``."""
    if pos < 0 or pos >= limit:
        return None
    stream.seek(pos)
    first_raw = stream.read(1)
    if not first_raw:
        return None
    id_length = _vint_length(first_raw[0], maximum=4)
    if id_length is None or pos + id_length >= limit:
        return None
    element_id_raw = first_raw + stream.read(id_length - 1)
    if len(element_id_raw) != id_length:
        return None
    element_id = int.from_bytes(element_id_raw, "big")

    size_first_raw = stream.read(1)
    if not size_first_raw:
        return None
    size_length = _vint_length(size_first_raw[0], maximum=8)
    if size_length is None:
        return None
    size_rest = stream.read(size_length - 1)
    if len(size_rest) != size_length - 1:
        return None
    marker = 1 << (8 - size_length)
    size_value = int.from_bytes(bytes([size_first_raw[0] & (marker - 1)]) + size_rest, "big")
    payload_start = pos + id_length + size_length
    if payload_start > limit:
        return None
    unknown_size = size_value == (1 << (7 * size_length)) - 1
    element_end = limit if unknown_size else payload_start + size_value
    if element_end > limit:
        return None
    return element_id, payload_start, element_end


def _iter_elements(
    stream: BinaryIO,
    start: int,
    end: int,
) -> Iterator[tuple[int, int, int]]:
    """Yield valid direct children from one bounded EBML region."""
    pos = start
    while pos < end:
        header = _read_element_header(stream, pos, end)
        if header is None:
            return
        element_id, payload_start, element_end = header
        yield element_id, payload_start, element_end
        if element_end <= pos:
            return
        pos = element_end


def _read_bounded(
    stream: BinaryIO,
    start: int,
    end: int,
    maximum: int,
) -> bytes | None:
    size = end - start
    if size < 0 or size > maximum:
        return None
    stream.seek(start)
    value = stream.read(size)
    return value if len(value) == size else None


def _simple_tag_payloads(
    stream: BinaryIO,
    start: int,
    end: int,
) -> tuple[bytes, ...]:
    name: bytes | None = None
    values: list[bytes] = []
    for element_id, payload_start, element_end in _iter_elements(stream, start, end):
        if element_id == _TAG_NAME_ID:
            name = _read_bounded(stream, payload_start, element_end, _MAX_TAG_NAME_BYTES)
        elif element_id == _TAG_STRING_ID:
            value = _read_bounded(stream, payload_start, element_end, MAX_TC260_VALUE_BYTES)
            if value is not None:
                values.append(value)
    if name != b"AIGC":
        return ()
    return tuple(value for value in values if parse_tc260_aigc_json(value) is not None)


def tc260_aigc_payloads(path: str | Path) -> tuple[bytes, ...]:
    """Read validated TC260 values from Matroska/WebM ``SimpleTag`` entries."""
    found: list[bytes] = []
    try:
        with open(path, "rb") as stream:
            if stream.read(4) != _EBML_MAGIC:
                return ()
            stream.seek(0, 2)
            file_size = stream.tell()
            for element_id, payload_start, element_end in _iter_elements(stream, 0, file_size):
                if element_id != _SEGMENT_ID:
                    continue
                for child_id, child_start, child_end in _iter_elements(stream, payload_start, element_end):
                    if child_id != _TAGS_ID:
                        continue
                    for tag_id, tag_start, tag_end in _iter_elements(stream, child_start, child_end):
                        if tag_id != _TAG_ID:
                            continue
                        for simple_id, simple_start, simple_end in _iter_elements(stream, tag_start, tag_end):
                            if simple_id == _SIMPLE_TAG_ID:
                                found.extend(_simple_tag_payloads(stream, simple_start, simple_end))
    except OSError:
        return ()
    return tuple(found)
