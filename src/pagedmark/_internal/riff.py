"""Bounded AVI metadata reader for native TC260 AIGC labels.

TC260-PG-20257A stores the label in an AVI ``LIST/INFO`` chunk whose child
chunk ID is ``AIGC`` and whose value is the normative JSON object. The walker
seeks over media chunks and reads only bounded metadata values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, BinaryIO

if TYPE_CHECKING:
    from pathlib import Path

from pagedmark.metadata import MAX_TC260_VALUE_BYTES, parse_tc260_aigc_json


def _info_payloads(
    stream: BinaryIO,
    start: int,
    end: int,
) -> tuple[bytes, ...]:
    found: list[bytes] = []
    position = start
    while position + 8 <= end:
        stream.seek(position)
        chunk_id = stream.read(4)
        size_raw = stream.read(4)
        if len(chunk_id) != 4 or len(size_raw) != 4:
            break
        size = int.from_bytes(size_raw, "little")
        payload_start = position + 8
        payload_end = payload_start + size
        if payload_end > end:
            break
        if chunk_id == b"AIGC" and size <= MAX_TC260_VALUE_BYTES:
            value = stream.read(size)
            if len(value) == size and parse_tc260_aigc_json(value) is not None:
                found.append(value.rstrip(b"\x00 "))
        position = payload_end + (size & 1)
    return tuple(found)


def tc260_aigc_payloads(path: str | Path) -> tuple[bytes, ...]:
    """Read validated TC260 values from an AVI ``LIST/INFO/AIGC`` chunk."""
    found: list[bytes] = []
    try:
        with open(path, "rb") as stream:
            header = stream.read(12)
            if len(header) != 12 or header[:4] != b"RIFF" or header[8:12] != b"AVI ":
                return ()
            stream.seek(0, 2)
            file_size = stream.tell()
            declared_end = min(8 + int.from_bytes(header[4:8], "little"), file_size)
            position = 12
            while position + 8 <= declared_end:
                stream.seek(position)
                chunk_id = stream.read(4)
                size_raw = stream.read(4)
                if len(chunk_id) != 4 or len(size_raw) != 4:
                    break
                size = int.from_bytes(size_raw, "little")
                payload_start = position + 8
                payload_end = payload_start + size
                if payload_end > declared_end:
                    break
                if chunk_id == b"LIST" and size >= 4:
                    list_type = stream.read(4)
                    if list_type == b"INFO":
                        found.extend(_info_payloads(stream, payload_start + 4, payload_end))
                position = payload_end + (size & 1)
    except OSError:
        return ()
    return tuple(found)
