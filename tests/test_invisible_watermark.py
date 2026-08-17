"""Tests for open DWT-DCT watermark detection.

The upstream encoder supplies known watermarks, while the in-tree decoder must
both identify them and match the upstream decoder bit for bit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from pagedmark.invisible_watermark import (
    _BITS_48,
    _SD1_STRING,
    _bits_match,
    _bytes_match_frac,
    detect_invisible_watermark,
    is_available,
)

pytestmark = pytest.mark.skipif(not is_available(), reason="detect extra not installed")


def _base_image() -> np.ndarray:
    # imwatermark needs enough DWT coefficients; use a 512x512 textured image.
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)


def _write_bits_watermark(tmp_path: Path, message: int) -> Path:
    from imwatermark import WatermarkEncoder

    bits = [int(b) for b in format(message, "048b")]
    enc = WatermarkEncoder()
    enc.set_watermark("bits", bits)
    wm = enc.encode(_base_image(), "dwtDct")
    path = tmp_path / "wm.png"
    cv2.imwrite(str(path), wm)
    return path


class TestHelpers:
    def test_bits_match_exact(self):
        assert _bits_match(0b1010, 0b1010, width=4) == 4

    def test_bits_match_one_off(self):
        assert _bits_match(0b1010, 0b1011, width=4) == 3

    def test_bytes_match_identical(self):
        assert _bytes_match_frac(_SD1_STRING, _SD1_STRING) == 1.0

    def test_bytes_match_length_mismatch_is_zero(self):
        assert _bytes_match_frac(b"abc", b"abcd") == 0.0


class TestRaveledHaarPass:
    """The precondition that makes the decoder's flat Haar pass legitimate.

    `_approximation` replaces `pywt.dwt(x, "haar", axis=1)[0]` with one
    `downcoef` call over `x.ravel()`. That is exact only while the last axis is
    even. Neither half of this is checked anywhere else: the equivalence is a
    property of pywt's implementation that an upgrade could take away, and an
    odd width produces wrong bits with no exception, since the reshape still
    succeeds whenever the total length is even.
    """

    @pytest.mark.parametrize("shape", [(64, 64), (7, 128), (129, 2), (2, 2), (33, 400)])
    def test_matches_pywt_dwt_on_even_widths(self, shape: tuple[int, int]):
        import pywt

        from pagedmark.dwt_dct import _approximation

        rng = np.random.default_rng(0)
        # uint8 is what the FIRST production pass receives -- `decode` builds its
        # plane with cvtColor, and extractChannel and transpose preserve the
        # dtype -- so a divergence in how downcoef coerces integers would be
        # invisible to a float-only parametrization.
        for array in (
            rng.random(shape),
            rng.integers(0, 256, shape).astype(np.float64),
            rng.integers(0, 256, shape).astype(np.uint8),
        ):
            expected = pywt.dwt(array, "haar", axis=1)[0]
            got = _approximation(array)
            assert got.shape == expected.shape
            assert np.array_equal(got, expected), "downcoef diverged from dwt -- a pywt upgrade may have changed it"

    def test_odd_width_raises_instead_of_returning_wrong_bits(self):
        from pagedmark.dwt_dct import _approximation

        # 4x6 ravels to 24, an even total, so the reshape would happily produce
        # a 4x3 array of numbers that pair across row boundaries.
        with pytest.raises(RuntimeError, match="odd"):
            _approximation(np.zeros((4, 6))[:, :5])


class TestDetect:
    def test_in_tree_decoder_matches_upstream(self, tmp_path: Path):
        from imwatermark import WatermarkDecoder

        from pagedmark.dwt_dct import decode_dwt_dct
        from pagedmark.image_io import imread

        path = _write_bits_watermark(tmp_path, _BITS_48["Stable Diffusion XL"])
        image = imread(path)
        assert image is not None

        upstream = np.asarray(WatermarkDecoder("bits", 48).decode(image, "dwtDct"), dtype=bool)
        ours = np.asarray(decode_dwt_dct(image, wm_len=48), dtype=bool)
        assert np.array_equal(ours, upstream)

    def test_detects_sdxl(self, tmp_path: Path):
        path = _write_bits_watermark(tmp_path, _BITS_48["Stable Diffusion XL"])
        assert detect_invisible_watermark(path) == "Stable Diffusion XL"

    def test_detects_flux(self, tmp_path: Path):
        path = _write_bits_watermark(tmp_path, _BITS_48["FLUX.2 (Black Forest Labs)"])
        assert detect_invisible_watermark(path) == "FLUX.2 (Black Forest Labs)"

    def test_detects_sd1_string(self, tmp_path: Path):
        from imwatermark import WatermarkEncoder

        enc = WatermarkEncoder()
        enc.set_watermark("bytes", _SD1_STRING)
        wm = enc.encode(_base_image(), "dwtDct")
        path = tmp_path / "sd1.png"
        cv2.imwrite(str(path), wm)
        assert detect_invisible_watermark(path) == "Stable Diffusion 1.x / 2.x"

    def test_clean_image_is_none(self, tmp_path: Path):
        path = tmp_path / "clean.png"
        cv2.imwrite(str(path), _base_image())
        assert detect_invisible_watermark(path) is None

    def test_unreadable_file_is_none(self, tmp_path: Path):
        path = tmp_path / "not_image.png"
        path.write_bytes(b"not a png")
        assert detect_invisible_watermark(path) is None
