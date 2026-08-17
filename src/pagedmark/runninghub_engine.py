"""RunningHub visible watermark detector/localizer.

RunningHub (a hosted ComfyUI platform, USCC 91340100MAEB4N8H76) stamps its
generations with a faint light-gray "RunningHub AI生成" text mark in the
**top-left** corner -- the China TC260 explicit AIGC label, but placed top-left
(unlike the GB 45438-2025 house style bottom-right of Doubao/Qwen/Kling) and
rendered in a mid-gray that the white top-hat front-end suppresses to clean-arm
levels.

Detection therefore uses the ``gray`` front-end (raw-grayscale silhouette NCC,
see ``TextMarkConfig.detect_frontend``); removal is the shared **localize ->
fill** (the detector's best-match box feeds :meth:`footprint_mask` ->
``region_eraser``). This module supplies only RunningHub's tuned
:class:`TextMarkConfig` (``assets/runninghub_alpha.png`` -- a font-rendered
synthetic silhouette from ``scripts/render_vendor_silhouettes.py``, never cut
from an upload).

EVERY tuned number below was measured on the vendor cohort (73 TC260 carriers
whose producer USCC names the entity, harvested 2026-07-22 by
``scripts/vendor_cohort_harvest.py``), NOT inherited from Doubao:

  * Only ~4 of the 73 cohort frames carry a visible mark (the rest are
    metadata-only TC260 carriers -- the platform labels frames it does not
    stamp), so recall of visible marks is 4/4 but the cohort fire rate is not
    a recall estimate. Positions/geometry are consistent across the positives.
  * The mark's width is ~0.32 of the frame WIDTH (0.319 measured on 832/1080/
    1536-wide frames) at ~0.008/0.006 x/y margins; the locate box below covers
    it with NCC slack.
  * ``alpha_height_frac`` comes from the silhouette aspect (0.128) at the
    measured width (0.27 * 1.25 rung ~= 0.3375 >= 0.32), per the standing rule
    that it is measured, not inherited.
  * STRICT ONLY (``provenance_ncc_factor`` 1.0): raw gray NCC is
    contrast-DEPENDENT and the sub-gate band of a corner-anchored gray match is
    unmeasured beyond the clean arm, so no provenance relaxation exists.
  * Gate 0.34: on 283 hand-labelled clean frames (cohort-contamination-guarded)
    corner-anchored gray NCC p99 is 0.264 / max 0.304, while the 4 positives
    score 0.38-0.54. 0.34 sits above the clean max with a small margin; the
    positives are few, so the margin is deliberately thin on the recall side.
"""
# The module-level _alpha_template / _glyph_silhouette / _template_match_score below
# are thin test-facing shims (imported by tests/), so pyright's src-only pass sees them
# as unused; the use is cross-module.
# pyright: reportUnusedFunction=false

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pagedmark import _text_mark_engine
from pagedmark._text_mark_engine import (
    TextMarkConfig,
    TextMarkDetection,
    TextMarkEngine,
    TextMarkScan,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Locate geometry as a fraction of the image WIDTH (the measured basis: every
# positive is portrait, where width == short side). The mark hugs the top-left
# corner (~0.008 of width off the left edge, ~0.006 of height off the top).
WM_WIDTH_FRAC = 0.45
WM_HEIGHT_FRAC = 0.10
MARGIN_LEFT_FRAC = 0.002
MARGIN_TOP_FRAC = 0.002

# Glyph appearance fields are unused by the gray front-end (it never binarizes)
# and kept only for config parity with the other text marks.
MAX_SATURATION = 55
LOGO_MIN_LUMA = 150
TOPHAT_DELTA = 12

DETECT_MIN_COVERAGE = 0.04  # unused by the gray front-end (kept for config parity)
# Calibrated 2026-07-22 on the vendor cohort vs 283 hand-labelled clean frames:
# corner-anchored gray NCC, clean p99 0.264 / max 0.304; positives 0.38-0.54.
DETECT_NCC_THRESHOLD = 0.34

# Detection-silhouette geometry (fraction of the image width), measured on the
# positives: mark width is ~0.320 of width on all three frame sizes (266px at 832,
# 345px at 1080, 491px at 1536), and the NCC is razor-sharp in size (0.537 on-size,
# 0.223 at +5.6% -- the same comb behaviour Qwen measured), so the nominal sits
# exactly on the measured size with a TIGHT ladder around it, not the shared 3 rungs
# (whose nearest rung landed 5.6% off and collapsed the match to 0.22).
_ALPHA_WIDTH_FRAC = 0.32
_ALPHA_HEIGHT_FRAC = 0.04
_LADDER = (0.95, 1.0, 1.05)

_CONFIG = TextMarkConfig(
    name="RunningHub",
    asset_name="runninghub_alpha.png",
    corner="tl",
    margin_floor=4,
    width_frac=WM_WIDTH_FRAC,
    height_frac=WM_HEIGHT_FRAC,
    margin_x_frac=MARGIN_LEFT_FRAC,
    margin_bottom_frac=MARGIN_TOP_FRAC,  # top margin for corner="tl"
    max_saturation=MAX_SATURATION,
    logo_min_luma=LOGO_MIN_LUMA,
    tophat_delta=TOPHAT_DELTA,
    morph_open_size=5,
    detect_min_coverage=DETECT_MIN_COVERAGE,
    detect_ncc_threshold=DETECT_NCC_THRESHOLD,
    detect_frontend="gray",
    scale_basis="width",  # measured: mark width tracks the frame width (0.32)
    ladder=_LADDER,
    alpha_width_frac=_ALPHA_WIDTH_FRAC,
    alpha_height_frac=_ALPHA_HEIGHT_FRAC,
    min_gw=8,
    # STRICT ONLY: contrast-dependent gray NCC; the relaxed band is unmeasured.
    provenance_ncc_factor=1.0,
)


def _alpha_template() -> NDArray[Any] | None:
    """The bundled RunningHub alpha template (float [0,1]), or None."""
    return _text_mark_engine.load_alpha_template(_CONFIG.asset_name)


class RunningHubEngine(TextMarkEngine):
    """Detect/localize the visible RunningHub "RunningHub AI生成" mark (top-left; localize -> fill)."""

    def __init__(self) -> None:
        super().__init__(_CONFIG)

    # Anchor window for the match position, as a fraction of the frame. The true
    # mark hugs the corner, while common false matches sit farther from the anchor.
    _ANCHOR_MAX_X = 0.025
    _ANCHOR_MAX_Y = 0.015

    def _post_gate(self, det: TextMarkDetection, scan: TextMarkScan) -> TextMarkDetection:
        """Demote a match that does not hug the top-left corner.

        A shared post-gate rather than a ``detect`` override, so the single-pass
        perception path (``detect_both``) cannot skip it.
        """
        if not det.detected or scan.loc is None:
            return det
        box = det.match_box  # the sweep the scan already ran on this same loc
        if box is None:
            det.detected = False
            return det
        h, w = scan.frame
        ax = (scan.loc.x + box[0]) / w
        ay = (scan.loc.y + box[1]) / h
        if ax > self._ANCHOR_MAX_X or ay > self._ANCHOR_MAX_Y:
            logger.debug(
                "RunningHub detect: score %.3f but match off-anchor (x=%.3f y=%.3f); demoting.",
                det.confidence,
                ax,
                ay,
            )
            det.detected = False
        return det
