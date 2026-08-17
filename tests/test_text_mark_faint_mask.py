"""A mark the `tophat` front-end DETECTS must also be MASKABLE.

Doubao's detection moved to the continuous `tophat` front-end, which does not
binarize, but the
removal mask still came from the BINARIZED glyph blob. A mark faint enough to be found
only by the continuous response therefore produced an empty binary blob, `localize`
returned mask=None, and `remove()` was a silent no-op: `identify` reported
`visible_doubao` while `visible` said "no visible mark" on the same file.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pagedmark.doubao_engine import DoubaoEngine


def _faint_mark_image(w: int = 900, h: int = 1200, alpha: float = 0.06, textured: bool = False) -> np.ndarray:
    """A frame carrying the REAL Doubao glyph shape at very low opacity.

    The shape has to be genuine or the NCC detector will not fire and the test would be
    exercising nothing; the low ``alpha`` is what keeps the binarizing path from finding
    a blob. Composited with the same forward model the marks use:
    ``stamped = (1-a)*bg + a*white``.

    ``textured`` adds fine luminance noise to the background. It is not decoration: on a
    FLAT frame the top-hat response is non-zero only on the glyph, so every threshold
    yields the same bounding box and a test built on a flat fixture cannot see a wrong
    threshold at all -- mutating the constant to an absurd value left the flat tests green.
    Texture puts response outside the glyph, which is the condition under which the mask's
    sizing actually matters, and is what real corner backgrounds look like.
    """
    from pagedmark._text_mark_engine import load_alpha_template

    tmpl = load_alpha_template("doubao_alpha.png")
    if tmpl is None:
        pytest.skip("doubao alpha asset unavailable")
    img = np.full((h, w, 3), 120, np.uint8)
    if textured:
        rng = np.random.default_rng(7)
        noise = rng.normal(0, 14, (h, w, 1)).repeat(3, axis=2)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=1.2)
    eng = DoubaoEngine()
    loc = eng.locate(img)
    base = eng.scale_base(img)
    gw = max(eng.config.min_gw, int(eng.config.alpha_width_frac * base))
    gh = max(4, int(eng.config.alpha_height_frac * base))
    a = cv2.resize(tmpl, (gw, gh), interpolation=cv2.INTER_AREA).astype(np.float32) * alpha
    x = loc.x + (loc.w - gw) // 2
    y = loc.y + (loc.h - gh) // 2
    roi = img[y : y + gh, x : x + gw].astype(np.float32)
    a3 = a[..., None]
    img[y : y + gh, x : x + gw] = np.clip(roi * (1 - a3) + 255.0 * a3, 0, 255).astype(np.uint8)
    return img


class TestFaintMarkIsMaskable:
    def test_binary_glyph_blob_is_empty_on_a_faint_mark(self):
        """The premise: this is the input class the binarizing path cannot segment."""
        eng = DoubaoEngine()
        img = _faint_mark_image()
        loc = eng.locate(img)
        glyph = eng.extract_mask(img, loc)
        assert int((glyph > 0).sum()) < eng._MIN_GLYPH_PIXELS

    def test_footprint_mask_is_not_empty_when_the_continuous_response_has_signal(self):
        """The fix: a faint mark must still yield a removal mask, without --no-detect.

        Without it `localize` returns None and removal silently does nothing while
        `identify` keeps reporting the mark.
        """
        eng = DoubaoEngine()
        img = _faint_mark_image()
        mask = eng.footprint_mask(img, force=False)
        assert mask is not None, "a detectable faint mark produced no removal mask"
        assert int((mask > 0).sum()) > 0

    def test_a_clean_frame_still_produces_no_mask(self):
        """The guard: the fallback must not turn every flat corner into a fill."""
        eng = DoubaoEngine()
        clean = cv2.GaussianBlur(np.full((1200, 900, 3), 120, np.uint8), (5, 5), 0)
        assert eng.footprint_mask(clean, force=False) is None

    @pytest.mark.parametrize("alpha", [0.5, 0.9])
    def test_a_bold_mark_is_unaffected(self, alpha: float):
        """A mark the binary path already segments must keep its tight glyph box."""
        eng = DoubaoEngine()
        img = _faint_mark_image(alpha=alpha)
        mask = eng.footprint_mask(img, force=False)
        assert mask is not None
        assert int((mask > 0).sum()) > 0


class TestFaintMaskStaysTight:
    """The faint fallback must cover the mark WITHOUT filling the whole corner.

    The first version of the fallback thresholded the continuous
    response at 0.5, but `tophat_response` returns uint8 0..255 -- so it selected every
    non-zero pixel, not "half the peak" as its comment claimed. On compatibility frames
    taking this path the resulting box covered the corner ROI, so removal inpainted the
    entire corner instead of the glyph. Parity could not see it: parity asks whether
    the detector is clean afterwards, and a mask that fills everything passes trivially.
    The cost, not the outcome, was the defect.
    """

    def test_mask_does_not_swallow_the_whole_corner_on_a_textured_frame(self):
        eng = DoubaoEngine()
        img = _faint_mark_image(alpha=0.10, textured=True)
        # Asserted, not skipped: a skip here would silently stop guarding the moment the
        # detector changed, which is exactly when this needs to be guarding.
        assert eng.detect(img).detected, "fixture must reach the faint path to test it"
        mask = eng.footprint_mask(img, force=False)
        assert mask is not None, "a detected faint mark must still produce a mask"
        area = int((mask > 0).sum())
        loc = eng.locate(img)
        roi = loc.w * loc.h
        # The mark's own glyph box is ~40% of the corner ROI and the mask pads it, so a
        # correct mask lands near 60%. The pre-fix behaviour measured 120.9% (the whole
        # ROI plus padding), which this bound excludes.
        assert area < 0.85 * roi, f"mask covers {100 * area / roi:.0f}% of the corner box"

    def test_mask_still_covers_the_stamped_glyph(self):
        """Tightness is only a virtue if the mark is still inside. Guards the other way."""
        eng = DoubaoEngine()
        img = _faint_mark_image(alpha=0.10, textured=True)
        assert eng.detect(img).detected, "fixture must reach the faint path to test it"
        mask = eng.footprint_mask(img, force=False)
        assert mask is not None
        loc = eng.locate(img)
        base = eng.scale_base(img)
        gw = max(eng.config.min_gw, int(eng.config.alpha_width_frac * base))
        gh = max(4, int(eng.config.alpha_height_frac * base))
        x = loc.x + (loc.w - gw) // 2
        y = loc.y + (loc.h - gh) // 2
        covered = int(np.count_nonzero(mask[y : y + gh, x : x + gw])) / max(1, gw * gh)
        assert covered > 0.6, f"mask covers only {100 * covered:.0f}% of the stamped glyph"
