"""Tests for the LibLibAI ("LibLibAI" wordmark) visible-watermark engine.

Every tuned constant in ``liblib_engine`` was measured on the 15-frame vendor
cohort (2026-07-22); these tests pin the load-bearing ones: the bottom-CENTER
anchor, the strict-only gate, and the match-box footprint (the blob bbox both
bled into background structure and did not own the triangle logo).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pagedmark import watermark_registry as registry
from pagedmark.liblib_engine import (
    _ALPHA_HEIGHT_FRAC,
    _ALPHA_WIDTH_FRAC,
    LibLibEngine,
    _alpha_template,
)

_MARK_FRAC = 0.10  # measured wordmark width, fraction of the frame WIDTH


def _compose(w: int, h: int, bg: float = 100.0):
    """Composite a triangle logo + the LibLibAI wordmark, bottom-center."""
    img = np.full((h, w, 3), bg, np.float32)
    at = _alpha_template()
    gw = int(_MARK_FRAC * w)
    gh = max(4, int(_MARK_FRAC * (_ALPHA_HEIGHT_FRAC / _ALPHA_WIDTH_FRAC) * w))
    ax = (w - gw) // 2
    ay = int(0.94 * h) - gh
    amap = np.zeros((h, w), np.float32)
    amap[ay : ay + gh, ax : ax + gw] = cv2.resize(at, (gw, gh))
    # the triangle logo, its own height to the LEFT of the wordmark
    lx1 = ax - int(0.3 * gh)
    lx0 = lx1 - gh
    cv2.fillPoly(amap, [np.array([(lx0, ay + gh), (lx1, ay + gh), (lx1, ay)])], 1.0)
    a3 = amap[:, :, None]
    wm = (a3 * 255.0 + (1 - a3) * img).clip(0, 255).astype(np.uint8)
    return wm, (ax, ay, gw, gh, lx0)


class TestLocate:
    def test_box_horizontally_centered(self):
        eng = LibLibEngine()
        img = np.zeros((2048, 1536, 3), np.uint8)
        loc = eng.locate(img)
        assert (1536 - loc.w) // 2 == pytest.approx(loc.x, abs=2)  # corner="bc"
        assert 2048 - (loc.y + loc.h) > 0  # bottom-anchored

    def test_box_scales_with_width(self):
        eng = LibLibEngine()
        narrow = eng.locate(np.zeros((2048, 1024, 3), np.uint8))
        wide = eng.locate(np.zeros((2048, 2048, 3), np.uint8))
        assert wide.w == pytest.approx(narrow.w * 2, rel=0.05)


class TestConfig:
    def test_tophat_frontend(self):
        assert LibLibEngine().config.detect_frontend == "tophat"

    def test_strict_only_no_provenance_relaxation(self):
        assert LibLibEngine().config.provenance_ncc_factor == 1.0

    def test_gate_above_clean_arm_max(self):
        # The Arial silhouette separates the wordmark from generic Latin UI text.
        assert LibLibEngine().config.detect_ncc_threshold >= 0.42

    def test_small_image_size_floor(self):
        # Small generic icons can resemble the wordmark, so the engine rejects them.
        eng = LibLibEngine()
        assert not eng.detect(np.full((200, 200, 3), 100, np.uint8)).detected
        wm, _ = _compose(200, 200)
        assert not eng.detect(wm).detected  # even a composed mark under the floor

    def test_registry_row(self):
        mark = registry.get_mark("liblib")
        assert mark.location == "bottom-center"
        assert mark.in_auto


class TestDetectAndMask:
    def test_detects_composed_mark(self):
        eng = LibLibEngine()
        wm, _ = _compose(1792, 2400)
        det = eng.detect(wm)
        assert det.detected, f"composed mark missed (conf={det.confidence:.3f})"

    def test_clean_frame_stays_quiet(self):
        eng = LibLibEngine()
        img = np.full((2400, 1792, 3), 100, np.uint8)
        assert not eng.detect(img).detected

    def test_mask_covers_logo_and_wordmark(self):
        """The footprint must cover the triangle logo LEFT of the wordmark while
        staying bounded by the match box vertically (the blob bbox bled into
        background structure and ate real content, 2026-07-22)."""
        eng = LibLibEngine()
        wm, (ax, ay, gw, gh, lx0) = _compose(1792, 2400)
        mask = eng.footprint_mask(wm)
        assert mask is not None
        ys, xs = np.where(mask > 0)
        assert xs.min() <= lx0 + gh // 2  # covers the logo
        assert xs.max() >= ax + gw - int(0.05 * gw)  # covers the wordmark's right edge
        assert ys.min() >= ay - gh  # does not bleed far above the mark

    def test_no_mask_on_clean_frame(self):
        eng = LibLibEngine()
        img = np.full((2400, 1792, 3), 100, np.uint8)
        assert eng.footprint_mask(img) is None

    def test_confident_liblib_detection_suppresses_the_jimeng_pill(self):
        # A LibLibAI image is TC260 too but is not Jimeng-basic: like Doubao/Qwen/
        # Kling/RunningHub/Baidu, a confident LibLibAI detection must veto the pill.
        # It was the one mark the hand-written veto list in ``_keep_pill`` missed.
        from pagedmark.watermark_registry import _keep_pill

        assert not _keep_pill({"liblib"}, provenance=frozenset({"jimeng"}), footprint_flat=1.0)

    def test_force_masks_the_whole_locate_box_on_a_clean_frame(self):
        """``force`` takes priority over detection for this mark, unlike the base
        policy: a --no-detect caller named the mark, so the honest footprint is the
        whole geometry box even though nothing was detected."""
        eng = LibLibEngine()
        img = np.full((2400, 1792, 3), 100, np.uint8)
        mask = eng.footprint_mask(img, force=True)
        assert mask is not None
        bx, by, bw, bh = eng.locate(img).bbox
        ys, xs = np.where(mask > 0)
        assert xs.min() <= bx
        assert xs.max() >= bx + bw - 1
        assert ys.min() <= by
        assert ys.max() >= by + bh - 1
