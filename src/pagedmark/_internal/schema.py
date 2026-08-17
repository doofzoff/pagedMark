"""Shared validation for versioned JSON transport contracts."""

from collections.abc import Collection


def require_schema_version(
    value: object,
    *,
    contract: str,
    supported: Collection[int],
) -> int:
    """Return an explicitly supported integer schema version or raise."""
    if type(value) is not int or value not in supported:
        versions = ", ".join(str(version) for version in sorted(supported))
        raise ValueError(f"Unsupported {contract} schema: {value!r}; supported versions: {versions}")
    return value
