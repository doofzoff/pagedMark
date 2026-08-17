"""Tests for the Tencent Yuanbao (元宝 / AI生成) visible-watermark engine."""

from __future__ import annotations

import cv2
import numpy as np

from pagedmark import watermark_registry as registry
from pagedmark.yuanbao_engine import (
    _ALPHA_HEIGHT_FRAC,
    _ALPHA_WIDTH_FRAC,
    YuanbaoEngine,
    _alpha_template,
)

_MARK_WIDTH_FRAC = 0.08
_RIGHT_MARGIN_FRAC = 0.028
_BOTTOM_MARGIN_FRAC = 0.031


def _compose(w: int, h: int, *, bg: float, foreground: float):
    """Composite the synthetic Yuanbao silhouette at its measured geometry."""
    image = np.full((h, w, 3), bg, np.float32)
    alpha = _alpha_template()
    assert alpha is not None
    short = min(w, h)
    gw = int(_MARK_WIDTH_FRAC * short)
    gh = max(6, int((_ALPHA_HEIGHT_FRAC / _ALPHA_WIDTH_FRAC) * gw))
    ax = w - int(_RIGHT_MARGIN_FRAC * short) - gw
    ay = h - int(_BOTTOM_MARGIN_FRAC * short) - gh
    mark_alpha = np.zeros((h, w), np.float32)
    mark_alpha[ay : ay + gh, ax : ax + gw] = cv2.resize(alpha, (gw, gh))
    a3 = mark_alpha[:, :, None]
    composed = (a3 * foreground + (1 - a3) * image).clip(0, 255).astype(np.uint8)
    return composed, (ax, ay, gw, gh)


class TestConfig:
    def test_uses_two_polarity_contrast_frontend(self):
        assert YuanbaoEngine().config.detect_frontend == "contrast"

    def test_strict_only(self):
        assert YuanbaoEngine().config.provenance_ncc_factor == 1.0

    def test_registry_row(self):
        mark = registry.get_mark("yuanbao")
        assert mark.location == "bottom-right"
        assert "元宝" in mark.label
        assert mark.in_auto


class TestDetectAndMask:
    def test_detects_light_mark_on_dark_background(self):
        watermark, _ = _compose(1024, 1024, bg=60, foreground=230)
        detection = YuanbaoEngine().detect(watermark)
        assert detection.detected
        assert detection.confidence >= 0.80

    def test_detects_dark_mark_on_light_background(self):
        """Yuanbao switches mark polarity with the background.

        A white top-hat alone misses the dark-gray stamp used on pale scenes.
        """
        watermark, _ = _compose(1024, 1024, bg=230, foreground=110)
        detection = YuanbaoEngine().detect(watermark)
        assert detection.detected
        assert detection.confidence >= 0.80

    def test_clean_gradient_stays_quiet(self):
        ramp = np.tile(np.linspace(40, 220, 1024, dtype=np.uint8), (1024, 1))
        image = cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)
        assert not YuanbaoEngine().detect(image).detected

    def test_match_must_hug_bottom_right_anchor(self):
        watermark, (ax, ay, gw, gh) = _compose(1024, 1024, bg=60, foreground=230)
        assert YuanbaoEngine().detect(watermark).detected
        shifted = np.full_like(watermark, 60)
        shifted[600 : 600 + gh, 600 : 600 + gw] = watermark[ay : ay + gh, ax : ax + gw]
        assert not YuanbaoEngine().detect(shifted).detected

    def test_mask_uses_detector_box_for_dark_mark(self):
        watermark, (ax, ay, gw, gh) = _compose(1024, 1024, bg=230, foreground=110)
        mask = YuanbaoEngine().footprint_mask(watermark)
        assert mask is not None
        ys, xs = np.where(mask > 0)
        assert xs.min() <= ax + int(0.05 * gw)
        assert xs.max() >= ax + gw - int(0.05 * gw)
        assert ys.min() <= ay + gh // 2 <= ys.max()

    def test_remove_clears_detector(self):
        watermark, _ = _compose(1024, 1024, bg=60, foreground=230)
        output, region = registry.get_mark("yuanbao").remove(watermark, backend="cv2")
        assert region is not None
        assert not YuanbaoEngine().detect(output).detected
