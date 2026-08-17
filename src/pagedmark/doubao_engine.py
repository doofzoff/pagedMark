"""Doubao visible watermark detector/localizer.

Doubao (ByteDance) stamps every generated image with a visible "豆包AI生成"
(Doubao AI generated) text strip in the bottom-right corner -- the explicit AIGC
label mandated by China's TC260 standard, a near-white semi-transparent overlay.

Detection matches the bundled glyph silhouette against the corner candidate; removal
is the shared **localize -> fill** (the glyph-bbox :meth:`footprint_mask` feeds
``region_eraser``), NOT reverse-alpha. This module shares
:class:`pagedmark._text_mark_engine.TextMarkEngine` and
supplies only Doubao's tuned :class:`TextMarkConfig` (bottom-right corner,
``assets/doubao_alpha.png`` -- the detection silhouette, rebuilt by
``scripts/visible_alpha_solve.py``). Arbitrary-region inpainting still lives in
``region_eraser`` / the ``erase`` command.
"""
# The module-level _alpha_template / _glyph_silhouette / _template_match_score below
# are thin test-facing shims (imported by tests/), so pyright's src-only pass sees them
# as unused; the use is cross-module.
# pyright: reportUnusedFunction=false

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pagedmark import _text_mark_engine
from pagedmark._text_mark_engine import TextMarkConfig, TextMarkEngine

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Locate geometry as a fraction of image WIDTH (the mark scales with width, anchored
# bottom-right). The box is GENEROUSLY wider than the mark and reaches close to the
# corner so a per-image re-rasterization shift stays inside the NCC alignment search.
WM_WIDTH_FRAC = 0.22
WM_HEIGHT_FRAC = 0.075
MARGIN_RIGHT_FRAC = 0.004
MARGIN_BOTTOM_FRAC = 0.004

# Glyph appearance: a light, low-saturation gray rendered brighter than the local
# background (white top-hat), so a white-paper document is left untouched.
MAX_SATURATION = 55  # max channel spread to count a pixel as "grayish"
LOGO_MIN_LUMA = 150  # glyphs are at least this bright in absolute terms
TOPHAT_DELTA = 12  # glyph must exceed the local background by this many levels

# Shape-consistent detection: match the bundled alpha glyph silhouette against the
# corner candidate via TM_CCOEFF_NORMED (keys on glyph SHAPE, not coverage; #23).
DETECT_MIN_COVERAGE = 0.04
# NOTE: this gate is FRONT-END SPECIFIC. The continuous top-hat front-end scores higher
# overall than the binary one (mean 0.809 vs 0.723 on the same 90 positives), so the
# binary-era 0.40 left the provenance-relaxed gate (x0.7) far too low and admitted false
# fires. Calibrated on the 240-image unbiased recall sample, full auto path:
#
#   gate   relaxed   recall   precision   true   false
#   0.40     0.280      96%         91%     86       8
#   0.45     0.315      94%         93%     85       6
#   0.50     0.350      92%         99%     83       1   <- chosen
#   0.60     0.420      87%         99%     78       1
#
# 0.50 beats the binary front-end on recall (92% vs 89%) at identical precision (99%),
# which is the only reason the front-end switch is worth it. Do not port this number to
# a binary-front-end mark; re-calibrate per front-end.
DETECT_NCC_THRESHOLD = 0.50

# Detection-silhouette geometry, emitted by scripts/visible_alpha_solve.py at the
# captured width. Sizes the glyph silhouette for the TM_CCOEFF_NORMED detection match
# (removal is the template-free glyph-bbox footprint mask, not this template).
_ALPHA_NATIVE_WIDTH = 2048
_ALPHA_WIDTH_FRAC = 0.1636  # asset width / image width -- sizes the detection silhouette
_ALPHA_HEIGHT_FRAC = 0.0405

_CONFIG = TextMarkConfig(
    name="Doubao",
    asset_name="doubao_alpha.png",
    corner="br",
    margin_floor=4,
    width_frac=WM_WIDTH_FRAC,
    height_frac=WM_HEIGHT_FRAC,
    margin_x_frac=MARGIN_RIGHT_FRAC,
    margin_bottom_frac=MARGIN_BOTTOM_FRAC,
    max_saturation=MAX_SATURATION,
    logo_min_luma=LOGO_MIN_LUMA,
    tophat_delta=TOPHAT_DELTA,
    morph_open_size=5,
    detect_min_coverage=DETECT_MIN_COVERAGE,
    detect_ncc_threshold=DETECT_NCC_THRESHOLD,
    detect_frontend="tophat",
    scale_basis="short",  # measured: recovers 56% of landscape misses (see scale_base)
    # No rival margin: measured 2026-07-18, the symmetric gate cost Doubao 7 genuine
    # detections to prevent 5 false ones (1.4:1 against). Doubao's absolute detector
    # is already 86% precise, so it has nothing to buy; Jimeng's is 38% and gains 25pp
    # for free. The confusion is asymmetric, so the remedy is too.
    alpha_width_frac=_ALPHA_WIDTH_FRAC,
    alpha_height_frac=_ALPHA_HEIGHT_FRAC,
    min_gw=8,
)


def _alpha_template() -> NDArray[Any] | None:
    """The bundled Doubao alpha template (float [0,1]), or None."""
    return _text_mark_engine.load_alpha_template(_CONFIG.asset_name)


def _glyph_silhouette() -> NDArray[Any] | None:
    """Binary "豆包AI生成" silhouette (255 = glyph) from the alpha map, or None."""
    return _text_mark_engine.glyph_silhouette(_CONFIG.asset_name)


def _template_match_score(box_mask: NDArray[Any], scale_base: int) -> float:
    """TM_CCOEFF_NORMED of the Doubao glyph silhouette against ``box_mask``."""
    return _text_mark_engine.template_match_score(box_mask, scale_base, _CONFIG)


class DoubaoEngine(TextMarkEngine):
    """Detect/localize the visible Doubao "豆包AI生成" watermark (locate -> mask; mask feeds the fill)."""

    def __init__(self) -> None:
        super().__init__(_CONFIG)
