"""Visible removal via localize -> fill: backend resolution, footprint masks, dispatch.

Every known mark is removed by LOCALIZING it to a full-frame footprint mask and
handing that mask to ONE shared fill backend (MI-GAN when the ``migan`` extra is
installed, else cv2). These tests avoid any ONNX model download by pinning the
backend to cv2; only pure cv2/numpy paths run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from pagedmark import watermark_registry as registry
from pagedmark._text_mark_engine import load_alpha_template
from pagedmark.doubao_engine import DoubaoEngine
from pagedmark.gemini_engine import GeminiEngine

if TYPE_CHECKING:
    import pytest


def _compose_textmark(engine, bg: float = 120.0, w: int = 1024, h: int = 1024):
    """Composite the engine's captured mark onto a flat ``bg`` at full opacity so
    the mark is detectable. Returns ``(watermarked_uint8, (ax, ay, gw, gh))``."""
    c = engine.config
    at = load_alpha_template(c.asset_name)
    gw = max(c.min_gw, int(c.alpha_width_frac * w))
    gh = max(4, int(c.alpha_height_frac * w))
    margin = int(0.015 * w)
    ax = (w - margin - gw) if c.corner == "br" else margin
    ay = h - margin - gh
    block = cv2.resize(at, (gw, gh))
    img = np.full((h, w, 3), float(bg), np.float32)
    a = np.clip(block, 0.0, 0.99)[:, :, None]
    img[ay : ay + gh, ax : ax + gw] = img[ay : ay + gh, ax : ax + gw] * (1 - a) + 255.0 * a
    return np.clip(img, 0, 255).astype(np.uint8), (ax, ay, gw, gh)


class TestResolveBackend:
    def test_auto_resolves_to_available_backend(self) -> None:
        # auto picks the best available model (LaMa > MI-GAN) or cv2; any is fine.
        assert registry.resolve_backend("auto") in {"cv2", "migan", "lama"}

    def test_cv2_passthrough(self) -> None:
        assert registry.resolve_backend("cv2") == "cv2"

    def test_lama_passthrough(self) -> None:
        assert registry.resolve_backend("lama") == "lama"


class TestFootprintMask:
    def test_textmark_footprint_geometry(self) -> None:
        # A clean flat corner has no glyph, so force=True yields the geometry box.
        mask = DoubaoEngine().footprint_mask(np.full((1024, 1024, 3), 120, np.uint8), force=True)
        assert mask is not None
        assert mask.shape == (1024, 1024)
        assert mask.dtype == np.uint8
        assert mask.any()
        # Doubao sits bottom-right: the mask mass is in the bottom-right quadrant.
        ys, xs = np.where(mask > 0)
        assert ys.mean() > 512
        assert xs.mean() > 512

    def test_textmark_small_image_returns_none(self) -> None:
        assert DoubaoEngine().footprint_mask(np.full((20, 20, 3), 120, np.uint8)) is None

    def test_gemini_footprint_needs_detection_or_force(self) -> None:
        eng = GeminiEngine()
        clean = np.full((1024, 1024, 3), 128, np.uint8)
        assert eng.footprint_mask(clean) is None  # nothing detected -> no mask
        forced = eng.footprint_mask(clean, force=True)  # default sparkle slot
        assert forced is not None
        assert forced.any()


class TestFillDispatch:
    """Force the cv2 backend so no ONNX model downloads; the dispatch/gating logic
    is backend-agnostic."""

    def test_clean_image_is_untouched(self) -> None:
        img = np.full((1024, 1024, 3), 120, np.uint8)
        out, region = registry.get_mark("doubao").remove(img, backend="cv2")
        assert region is None
        assert np.array_equal(out, img)  # not detected, not forced -> no-op

    def test_forced_fill_edits_only_footprint(self) -> None:
        img, (ax, ay, gw, gh) = _compose_textmark(DoubaoEngine())
        out, _ = registry.get_mark("doubao").remove(img, backend="cv2", force=True)
        assert not np.array_equal(out[ay : ay + gh, ax : ax + gw], img[ay : ay + gh, ax : ax + gw])
        assert np.array_equal(out[:200, :200], img[:200, :200])  # far corner untouched

    def test_detected_fill_lowers_confidence(self) -> None:
        mark = registry.get_mark("doubao")
        img, _ = _compose_textmark(DoubaoEngine())
        before = mark.detect(img)
        assert before.detected  # the composed mark is detectable
        out, region = mark.remove(img, backend="cv2")
        assert region is not None
        assert mark.detect(out).confidence < before.confidence


class TestBackendSelection:
    """auto resolves to the best available inpaint backend: LaMa > MI-GAN > cv2.
    cv2 is the floor when no learned ONNX model is present (and warns once)."""

    def test_prefers_lama_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pagedmark import region_eraser

        monkeypatch.setattr(region_eraser, "lama_available", lambda: True)
        monkeypatch.setattr(region_eraser, "migan_available", lambda: True)
        assert registry.preferred_inpaint_backend() == "lama"

    def test_migan_when_only_migan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pagedmark import region_eraser

        monkeypatch.setattr(region_eraser, "lama_available", lambda: False)
        monkeypatch.setattr(region_eraser, "migan_available", lambda: True)
        assert registry.preferred_inpaint_backend() == "migan"

    def test_cv2_when_no_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pagedmark import region_eraser

        monkeypatch.setattr(region_eraser, "lama_available", lambda: False)
        monkeypatch.setattr(region_eraser, "migan_available", lambda: False)
        monkeypatch.setattr(registry, "_warned_cv2_fallback", True)
        assert registry.preferred_inpaint_backend() == "cv2"

    def test_inpaint_model_available_reflects_either(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pagedmark import region_eraser

        monkeypatch.setattr(region_eraser, "migan_available", lambda: False)
        monkeypatch.setattr(region_eraser, "lama_available", lambda: False)
        assert not registry.inpaint_model_available()
        monkeypatch.setattr(region_eraser, "lama_available", lambda: True)
        assert registry.inpaint_model_available()
