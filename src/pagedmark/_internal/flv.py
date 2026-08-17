"""Bounded FLV metadata reader for native TC260 AIGC labels.

TC260-PG-20257A stores the label in the ``onMetaData`` script tag as an AMF0
property named ``AIGC`` whose string value is the normative JSON object. Media
tag payloads are skipped without being loaded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from pagedmark.metadata import MAX_TC260_VALUE_BYTES, parse_tc260_aigc_json

_SCRIPT_TAG = 18
_MAX_SCRIPT_BYTES = 4 * 1024 * 1024


def _u24(value: bytes) -> int:
    return int.from_bytes(value, "big")


def _amf0_string(data: bytes, position: int, *, long: bool = False) -> tuple[bytes, int] | None:
    length_size = 4 if long else 2
    if position + length_size > len(data):
        return None
    length = int.from_bytes(data[position : position + length_size], "big")
    start = position + length_size
    end = start + length
    if end > len(data):
        return None
    return data[start:end], end


def _skip_amf0(data: bytes, position: int, depth: int = 0) -> int | None:
    if position >= len(data) or depth > 8:
        return None
    value_type = data[position]
    position += 1
    if value_type == 0:
        return position + 8 if position + 8 <= len(data) else None
    if value_type == 1:
        return position + 1 if position + 1 <= len(data) else None
    if value_type == 2:
        parsed = _amf0_string(data, position)
        return parsed[1] if parsed is not None else None
    if value_type in {5, 6}:
        return position
    if value_type == 7:
        return position + 2 if position + 2 <= len(data) else None
    if value_type == 11:
        return position + 10 if position + 10 <= len(data) else None
    if value_type == 12:
        parsed = _amf0_string(data, position, long=True)
        return parsed[1] if parsed is not None else None
    if value_type == 10:
        if position + 4 > len(data):
            return None
        count = int.from_bytes(data[position : position + 4], "big")
        position += 4
        for _ in range(count):
            next_position = _skip_amf0(data, position, depth + 1)
            if next_position is None:
                return None
            position = next_position
        return position
    if value_type in {3, 8}:
        if value_type == 8:
            if position + 4 > len(data):
                return None
            position += 4
        while position + 3 <= len(data):
            name_length = int.from_bytes(data[position : position + 2], "big")
            position += 2
            if name_length == 0 and data[position] == 9:
                return position + 1
            position += name_length
            if position > len(data):
                return None
            next_position = _skip_amf0(data, position, depth + 1)
            if next_position is None:
                return None
            position = next_position
        return None
    return None


def _script_payloads(data: bytes) -> tuple[bytes, ...]:
    first = _amf0_string(data, 1) if data[:1] == b"\x02" else None
    if first is None or first[0] != b"onMetaData":
        return ()
    position = first[1]
    if position >= len(data) or data[position] not in {3, 8}:
        return ()
    if data[position] == 8:
        position += 5
    else:
        position += 1
    found: list[bytes] = []
    while position + 3 <= len(data):
        name_length = int.from_bytes(data[position : position + 2], "big")
        position += 2
        if name_length == 0 and data[position] == 9:
            break
        name_end = position + name_length
        if name_end > len(data):
            break
        name = data[position:name_end]
        position = name_end
        if name == b"AIGC" and position < len(data) and data[position] in {2, 12}:
            long = data[position] == 12
            parsed = _amf0_string(data, position + 1, long=long)
            if parsed is None:
                break
            value, position = parsed
            if len(value) <= MAX_TC260_VALUE_BYTES and parse_tc260_aigc_json(value) is not None:
                found.append(value)
            continue
        next_position = _skip_amf0(data, position)
        if next_position is None:
            break
        position = next_position
    return tuple(found)


def tc260_aigc_payloads(path: str | Path) -> tuple[bytes, ...]:
    """Read validated TC260 values from FLV ``script.onMetaData.AIGC``."""
    found: list[bytes] = []
    try:
        with open(path, "rb") as stream:
            header = stream.read(9)
            if len(header) != 9 or header[:3] != b"FLV":
                return ()
            data_offset = int.from_bytes(header[5:9], "big")
            stream.seek(0, 2)
            file_size = stream.tell()
            position = data_offset + 4
            while position + 11 <= file_size:
                stream.seek(position)
                tag_header = stream.read(11)
                if len(tag_header) != 11:
                    break
                tag_type = tag_header[0] & 0x1F
                data_size = _u24(tag_header[1:4])
                payload_start = position + 11
                payload_end = payload_start + data_size
                if payload_end + 4 > file_size:
                    break
                if tag_type == _SCRIPT_TAG and data_size <= _MAX_SCRIPT_BYTES:
                    payload = stream.read(data_size)
                    if len(payload) == data_size:
                        found.extend(_script_payloads(payload))
                        if found:
                            return tuple(found)
                position = payload_end + 4
    except OSError:
        return ()
    return tuple(found)
