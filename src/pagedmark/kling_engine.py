"""Kling (可灵, Kuaishou) visible watermark detector/localizer.

Kling stamps its generations with a thin, light-gray "可灵AI 3.0" text strip in the
bottom-right corner, preceded by the vendor's spiral logo (not part of the detection
silhouette -- logos vary between releases, the text run is what discriminates).
Known variants: an "Omni" suffix release, a latin "KlingAI 3.0" release, and a
version-less "可灵AI" -- the silhouette targets the common "可灵AI 3.0" core, so the
suffix variants are only caught when the core run is bold enough (measured below).

Detection matches the bundled glyph silhouette against the corner; removal is the
shared **localize -> fill** (the glyph-bbox :meth:`footprint_mask` feeds
``region_eraser``), NOT reverse-alpha. This module supplies only Kling's tuned
:class:`TextMarkConfig` (``assets/kling_alpha.png`` -- a font-rendered synthetic
silhouette from ``scripts/render_vendor_silhouettes.py``, never cut from an
upload). It also feeds ``identify`` as the medium-confidence ``visible_kling``
signal via the registry.

EVERY tuned number below was measured on the vendor cohort (30 TC260 carriers whose
producer USCC 91110108335469089C names the entity, 2026-07-21; harness
``scripts/vendor_mark_calibrate.py``), NOT inherited from Doubao:

  * The mark scales with the SHORT side at ~0.12 of it (mark_w/short measured
    0.118-0.122 across portrait AND landscape carriers -- unimodal, so the shipped
    3-rung ladder covers it) and sits ~0.03 off the right/bottom edges; the locate
    box fractions below are fitted from the measured absolute mark rects.
  * ``alpha_height_frac`` comes from the silhouette aspect (0.239) at the fitted
    width, matching the aspect the fit converged on (0.25).
  * Gate 0.35, one step above the clean arm's max: on the cohort-vs-clean run
    (cohort-contamination-guarded, 286 hand-labelled clean frames) the clean arm
    scored p99 0.304 / max 0.320, and every cohort frame >= 0.35 carries a visible
    可灵AI 3.0 mark (9 of ~19 eyeballed visible marks fire = ~47% recall of visible
    marks; the misses are the faint "Omni"-suffix release, the latin "KlingAI"
    release and the version-less "可灵AI", which score 0.17-0.25 and cannot be
    reached without engulfing the clean arm).
  * STRICT ONLY (``provenance_ncc_factor`` 1.0): the sub-gate band holds real Kling
    variants AND the clean arm's top (clean p90 0.220 vs variant marks at 0.17-0.25
    -- they overlap), so a provenance-relaxed arm cannot separate them. No
    provenance relaxation exists for this mark.
  * No rival margin: at the shipped gate the template fires on 1 of 400
    Doubao-marked frames (0.2%, a 豆包 frame sitting INSIDE the Kling cohort, still
    below the gate), 0 of 298 Jimeng-marked frames and 0 of 286 hand-labelled clean
    frames, and a 0.10 rival margin costs zero genuine Kling detections -- so it is
    simply unnecessary (same conclusion shape as Qwen).
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

# Locate geometry as a fraction of the image SHORT side (measured basis -- see
# scale_base). The box is fitted to the measured mark rects: the mark's right
# margin is ~0.034 of the short side and its bottom margin ~0.027; width/height
# cover the mark plus NCC slack.
WM_WIDTH_FRAC = 0.19
WM_HEIGHT_FRAC = 0.05
MARGIN_RIGHT_FRAC = 0.03
MARGIN_BOTTOM_FRAC = 0.023

# Glyph appearance: a light, low-saturation gray rendered brighter than the local
# background (white top-hat), same overlay class as Doubao -- inherited, and
# harmless because the tophat front-end turns these gates into weights.
MAX_SATURATION = 55
LOGO_MIN_LUMA = 150
TOPHAT_DELTA = 12

DETECT_MIN_COVERAGE = 0.04  # unused by the tophat front-end (kept for config parity)
# Calibrated 2026-07-21 on the vendor cohort vs 286 hand-labelled clean frames
# (cohort-contamination-guarded): clean p99 0.304 / max 0.320, and every cohort
# frame scoring >= 0.35 carries a visible 可灵AI 3.0 mark. 0.35 was picked over
# 0.33 (also zero clean fires) for margin against unseen clean content at a cost
# of zero measured cohort detections.
DETECT_NCC_THRESHOLD = 0.35

# Detection-silhouette geometry (fraction of the short side), fitted on the
# cohort: the mark's width (0.12, unimodal) and the silhouette aspect (0.239).
_ALPHA_WIDTH_FRAC = 0.12
_ALPHA_HEIGHT_FRAC = 0.0287

_CONFIG = TextMarkConfig(
    name="Kling",
    asset_name="kling_alpha.png",
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
    scale_basis="short",  # measured: mark_w/short 0.118-0.122 across orientations
    alpha_width_frac=_ALPHA_WIDTH_FRAC,
    alpha_height_frac=_ALPHA_HEIGHT_FRAC,
    min_gw=8,
    # STRICT ONLY: the sub-gate band (real Kling variants at 0.17-0.25) overlaps
    # the clean arm's top (p90 0.220), so provenance relaxation is disabled
    # outright (factor 1.0 = never relaxed).
    provenance_ncc_factor=1.0,
)


def _alpha_template() -> NDArray[Any] | None:
    """The bundled Kling alpha template (float [0,1]), or None."""
    return _text_mark_engine.load_alpha_template(_CONFIG.asset_name)


def _glyph_silhouette() -> NDArray[Any] | None:
    """Binary "可灵AI 3.0" silhouette (255 = glyph) from the alpha map, or None."""
    return _text_mark_engine.glyph_silhouette(_CONFIG.asset_name)


class KlingEngine(TextMarkEngine):
    """Detect/localize the visible Kling "可灵AI 3.0" watermark (locate -> mask; mask feeds the fill)."""

    def __init__(self) -> None:
        super().__init__(_CONFIG)
