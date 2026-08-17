"""DWT-DCT decoder compatible with invisible-watermark's ``dwtDct`` path.

Derived from ShieldMnt/invisible-watermark ``imwatermark/maxDct.py`` (MIT),
trimmed to the matrix path used by Stable Diffusion, SDXL, and FLUX. The block
scan is vectorized rather than transcribed, so the file no longer reads line by
line against upstream; what it preserves is the output, bit for bit. See
[`docs/module-internals.md`](../../docs/module-internals.md) for the
measurements and for why a faster hand-rolled transform is not available.

Copyright (c) 2021 ShieldMnt

The complete upstream license is distributed in
``licenses/invisible-watermark-MIT.txt``.
"""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportMissingTypeStubs=false

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import pywt

if TYPE_CHECKING:
    from numpy.typing import NDArray

_DEFAULT_SCALES = (0, 36, 36)
# Fixed by the format being decoded, and the 4-way fold chains in `_frame_bits`
# are written for it. It was a constructor parameter while the block scan was a
# generic Python loop; nothing ever passed another value, and a knob that now
# silently returns wrong bits is worse than no knob.
_BLOCK = 4

# Block-rows of the approximation band handled per strip. Not a tuned value:
# every height measured landed inside the others' noise. What matters is strips
# at all rather than a full-plane intermediate, not this number.
_STRIP = 16


def _approximation(rows: NDArray[Any]) -> NDArray[Any]:
    """One Haar pass along the last axis, approximation band only.

    ``pywt.dwt(x, "haar", axis=1)[0]`` computes and allocates the detail band as
    well, and dispatches per row. Flattening lets one ``downcoef`` call do the
    whole plane, and it is the same numbers in the same order **only while the
    last axis is even**: Haar's filter is length 2, so an even row length makes
    every pair fall inside its own row with no boundary extension. On an odd
    width the pairs walk across row boundaries and the reshape below still
    succeeds whenever the total is even -- wrong bits, no exception, and the
    upstream-parity test is `skipif`-gated. Hence the explicit check.

    The transform itself stays inside pywt however slow that looks. Its C
    convolution contracts into an FMA that no numpy expression reproduces, and
    the caller's threshold is ``peak % 36 > 18.0`` against values that are exact
    multiples of 0.5, so a 1-ulp difference flips real bits.
    """
    height, width = rows.shape
    if width % 2:
        raise RuntimeError(f"row length {width} is odd; the raveled Haar pass requires an even last axis")
    return pywt.downcoef("a", rows.ravel(), "haar").reshape(height, width // 2)


class _DecodeMaxDct:
    """Extract frequency-domain bits using the upstream matrix algorithm."""

    def __init__(
        self,
        wm_lengths: tuple[int, ...],
        scales: tuple[int, int, int] = _DEFAULT_SCALES,
    ) -> None:
        self._wm_lengths = wm_lengths
        self._scales = scales

    def decode(self, bgr: NDArray[Any]) -> dict[int, NDArray[Any]]:
        row, col, _channels = bgr.shape
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
        trimmed = yuv[: row // 4 * 4, : col // 4 * 4]

        per_channel = [
            self._plane_bits(trimmed, channel, self._scales[channel])
            for channel in range(2)
            if self._scales[channel] > 0
        ]
        # Each channel restarts the bit index at 0, so the buckets come from a
        # per-channel arange rather than one running counter.
        index = np.concatenate([np.arange(bits.size) for bits in per_channel] or [np.zeros(0, dtype=np.int64)])
        weights = np.concatenate(per_channel or [np.zeros(0)])

        decoded: dict[int, NDArray[Any]] = {}
        for wm_len in self._wm_lengths:
            bucket = index % wm_len
            sums = np.bincount(bucket, weights=weights, minlength=wm_len)
            counts = np.bincount(bucket, minlength=wm_len)
            decoded[wm_len] = sums * 255 > counts * 127
        return decoded

    def _plane_bits(self, trimmed: NDArray[Any], channel: int, scale: int) -> NDArray[Any]:
        """Block bits for one colour plane, a strip of block-rows at a time.

        ``dwt2`` is ``dwtn``: it transforms along axis 0, then along axis 1 over
        both halves, and three of the four bands it returns are discarded here.
        Only the approximation is ever asked for, and transposing between the
        two passes lets pywt walk a contiguous axis instead of a column -- that
        access pattern, not the arithmetic saved, is where the time goes.

        Strips mean no full-plane float64 intermediate is ever materialized, and
        they are seam-free for the reason ``_approximation`` documents: output
        row ``k`` reads input rows ``2k`` and ``2k + 1`` only. Strips start on
        multiples of ``2 * _BLOCK``, so neither a pair nor a 4x4 block straddles
        one.
        """
        if trimmed.shape[0] == 0 or trimmed.shape[1] == 0:
            # Reachable: a 1x65536 image clears the caller's area check and
            # trims to an empty plane. Left to dwt2 so the exception stays the
            # one this module has always raised -- returning empty bits here
            # instead would silently turn a raise into an all-false verdict.
            return pywt.dwt2(trimmed[:, :, channel], "haar")[0]
        rows = trimmed.shape[0] // (2 * _BLOCK)
        cols = trimmed.shape[1] // (2 * _BLOCK)
        if rows == 0 or cols == 0:
            return np.zeros(0, dtype=np.float64)

        width = cols * 2 * _BLOCK
        pieces: list[NDArray[Any]] = []
        for start in range(0, rows, _STRIP):
            stop = min(start + _STRIP, rows)
            strip = trimmed[start * 2 * _BLOCK : stop * 2 * _BLOCK, :width]
            columns = cv2.transpose(cv2.extractChannel(strip, channel))
            band = _approximation(cv2.transpose(_approximation(columns)))
            pieces.append(self._frame_bits(band, scale))
        return np.concatenate(pieces)

    def _frame_bits(self, frame: NDArray[Any], scale: int) -> NDArray[Any]:
        """One bit per 4x4 block, in row-major block order.

        Upstream's per-block loop, said to numpy once instead of to the
        interpreter ~135k times per image. Zeroing the DC term in the absolute
        band says "ignore index 0" without materializing a ``(nblocks, 16)``
        copy, and the 4-way ``np.maximum`` chains reduce over contiguous rows.
        """
        rows = frame.shape[0] // _BLOCK
        cols = frame.shape[1] // _BLOCK
        band = np.abs(frame[: rows * _BLOCK, : cols * _BLOCK])
        band[::_BLOCK, ::_BLOCK] = 0.0
        folded = np.maximum(np.maximum(band[0::4], band[1::4]), np.maximum(band[2::4], band[3::4]))
        folded = folded.reshape(rows * cols, _BLOCK)
        peak = np.maximum(np.maximum(folded[:, 0], folded[:, 1]), np.maximum(folded[:, 2], folded[:, 3]))
        return ((peak % scale) > 0.5 * scale).astype(np.float64)


def decode_dwt_dct(bgr: NDArray[Any], wm_len: int) -> NDArray[Any]:
    """Extract ``wm_len`` watermark bits from a BGR image."""
    return decode_dwt_dct_lengths(bgr, (wm_len,))[wm_len]


def decode_dwt_dct_lengths(bgr: NDArray[Any], wm_lengths: tuple[int, ...]) -> dict[int, NDArray[Any]]:
    """Extract several watermark lengths with one DWT and block scan."""
    if bgr.size == 0 or min(bgr.shape[:2]) * max(bgr.shape[:2]) < 256 * 256:
        raise RuntimeError("image too small, should be larger than 256x256")
    if not wm_lengths or any(wm_len <= 0 for wm_len in wm_lengths):
        raise ValueError("watermark lengths must be positive")
    return _DecodeMaxDct(wm_lengths=tuple(dict.fromkeys(wm_lengths))).decode(bgr)
