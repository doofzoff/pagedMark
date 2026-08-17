"""Regression tests for Dependabot compatibility constraints."""

from pathlib import Path


def test_dependabot_blocks_opencv_releases_that_require_numpy_2() -> None:
    config = Path(".github/dependabot.yml").read_text()

    opencv_ignore = config.split('dependency-name: "opencv-python-headless"', maxsplit=1)[1]
    assert '"<4.12"' not in opencv_ignore
    assert '- ">=4.12"' in opencv_ignore
