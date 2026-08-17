"""Samsung Galaxy AI visible watermark detector/localizer.

Samsung's on-device Generative AI photo edits burn a visible "✦ Contenuti generati
dall'AI" wordmark into the bottom-LEFT corner (the Italian locale variant calibrated
here; the string is locale-specific -- DETECTION only matches this locale's silhouette,
so other locales are not yet detected, though the fill mask itself is locale-agnostic).
It is a faint, near-white semi-transparent overlay, the same overlay class as the
Doubao/Jimeng marks but bottom-left.

Detection matches the bundled glyph silhouette against the corner; removal is the
shared **localize -> fill** (the glyph-bbox :meth:`footprint_mask` feeds
``region_eraser``), NOT reverse-alpha. This module shares
:class:`pagedmark._text_mark_engine.TextMarkEngine` and
supplies only Samsung's tuned :class:`TextMarkConfig` (bottom-LEFT corner, a lower glyph
luma since the mark is faint, ``assets/samsung_alpha.png`` -- the detection silhouette,
solved from the flat captures by ``scripts/visible_alpha_solve.py``). Samsung Galaxy AI
edits are also caught by C2PA + the ``genAIType`` marker, so this is the visible-mark
*removal* path; it also feeds ``identify`` as the medium-confidence ``visible_samsung``
signal via the registry.
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

# Locate geometry as a fraction of image WIDTH (mark scales with width, bottom-LEFT).
WM_WIDTH_FRAC = 0.40
WM_HEIGHT_FRAC = 0.060
MARGIN_LEFT_FRAC = 0.004
MARGIN_BOTTOM_FRAC = 0.002

# Glyph appearance: a light, low-saturation gray. LOGO_MIN_LUMA is lower than Jimeng's
# because the mark is faint (peak alpha ~0.38), so on a mid/dark background its glyph
# luma is lower; a white-paper document is still left untouched.
MAX_SATURATION = 55
LOGO_MIN_LUMA = 110
TOPHAT_DELTA = 8

# Shape-consistent detection. Threshold 0.40; real marks ~0.79, and Doubao/Jimeng score
# 0.0 here (and Samsung 0.0 on theirs) -- no cross-fire (the corner also differs).
DETECT_MIN_COVERAGE = 0.01
DETECT_NCC_THRESHOLD = 0.40

# Detection-silhouette geometry, solved by scripts/visible_alpha_solve.py from the flat
# gray capture (native width 1086). Real photos are ~2958 wide, so the captured glyph is
# upscaled; width-scale + NCC-align sizes the silhouette for the detection match (removal
# is the template-free glyph-bbox footprint mask).
_ALPHA_NATIVE_WIDTH = 1086
_ALPHA_WIDTH_FRAC = 0.3195  # asset width / image width -- sizes the detection silhouette
_ALPHA_HEIGHT_FRAC = 0.0378

_CONFIG = TextMarkConfig(
    name="Samsung Galaxy AI",
    asset_name="samsung_alpha.png",
    corner="bl",
    margin_floor=2,
    width_frac=WM_WIDTH_FRAC,
    height_frac=WM_HEIGHT_FRAC,
    margin_x_frac=MARGIN_LEFT_FRAC,
    margin_bottom_frac=MARGIN_BOTTOM_FRAC,
    max_saturation=MAX_SATURATION,
    logo_min_luma=LOGO_MIN_LUMA,
    tophat_delta=TOPHAT_DELTA,
    morph_open_size=3,
    detect_min_coverage=DETECT_MIN_COVERAGE,
    detect_ncc_threshold=DETECT_NCC_THRESHOLD,
    alpha_width_frac=_ALPHA_WIDTH_FRAC,
    alpha_height_frac=_ALPHA_HEIGHT_FRAC,
    min_gw=16,
)


def _alpha_template() -> NDArray[Any] | None:
    """The bundled Samsung alpha template (float [0,1]), or None."""
    return _text_mark_engine.load_alpha_template(_CONFIG.asset_name)


def _glyph_silhouette() -> NDArray[Any] | None:
    """Binary "Contenuti generati dall'AI" silhouette (255 = glyph), or None."""
    return _text_mark_engine.glyph_silhouette(_CONFIG.asset_name)


def _template_match_score(box_mask: NDArray[Any], scale_base: int) -> float:
    """TM_CCOEFF_NORMED of the Samsung glyph silhouette against ``box_mask``."""
    return _text_mark_engine.template_match_score(box_mask, scale_base, _CONFIG)


class SamsungEngine(TextMarkEngine):
    """Detect/localize the visible Samsung Galaxy AI text mark (locate -> mask; mask feeds the fill)."""

    def __init__(self) -> None:
        super().__init__(_CONFIG)
