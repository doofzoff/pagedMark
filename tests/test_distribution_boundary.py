"""Distribution-boundary regression tests."""

from __future__ import annotations

import re
from pathlib import Path


def _array_values(section: str, key: str) -> set[str]:
    match = re.search(rf"(?ms)^{key}\s*=\s*\[(.*?)^\]", section)
    assert match is not None
    return set(re.findall(r'"([^"]+)"', match.group(1)))


def test_sdist_has_explicit_public_boundary() -> None:
    """Hatchling must publish only package source and required metadata."""
    root = Path(__file__).resolve().parents[1]
    config = (root / "pyproject.toml").read_text(encoding="utf-8")
    sdist_config = config.split("[tool.hatch.build.targets.sdist]", maxsplit=1)[1].split("\n[", maxsplit=1)[0]

    assert _array_values(sdist_config, "include") == {
        "/src",
        "/LICENSE",
        "/NOTICE",
        "/README.md",
        "/pyproject.toml",
    }
    assert {"/data", "/tmp", "/.sc"} <= _array_values(sdist_config, "exclude")
