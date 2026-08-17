"""Memory guard for the text-mark engine ``extract_mask``.

``extract_mask`` used to build a full ``(h, w)`` uint8 mask that every caller
cropped to the located box; it now returns the box-sized mask directly. This test
locks in the O(footprint) memory characteristic so a regression back to a
full-frame allocation fails loudly.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import pagedmark.doubao_engine as D
import pagedmark.jimeng_engine as J
import pagedmark.samsung_engine as S
from pagedmark.doubao_engine import DoubaoEngine
from pagedmark.jimeng_engine import JimengEngine
from pagedmark.samsung_engine import SamsungEngine

# (engine factory, engine module) for each reverse-alpha text mark.
ENGINES = [
    pytest.param(DoubaoEngine, D, id="doubao"),
    pytest.param(JimengEngine, J, id="jimeng"),
    pytest.param(SamsungEngine, S, id="samsung"),
]


def _watermarked(engine, module) -> np.ndarray:
    """Composite the engine's real alpha glyph (white) onto a flat mid-gray field at
    the captured native width, anchored in the mark's configured corner."""
    cfg = engine.config
    nw = module._ALPHA_NATIVE_WIDTH
    at = module._alpha_template()
    gw, gh = int(cfg.alpha_width_frac * nw), int(cfg.alpha_height_frac * nw)
    margin = int(0.015 * nw)
    ax = (nw - margin - gw) if cfg.corner == "br" else margin
    ay = nw - margin - gh
    amap = np.zeros((nw, nw), np.float32)
    amap[ay : ay + gh, ax : ax + gw] = cv2.resize(at, (gw, gh))
    a3 = amap[:, :, None]
    img = np.full((nw, nw, 3), 100.0, np.float32)
    return (a3 * 255.0 + (1 - a3) * img).clip(0, 255).astype(np.uint8)


@pytest.mark.parametrize(("factory", "module"), ENGINES)
class TestExtractMaskFootprint:
    def test_returns_box_sized_mask(self, factory, module):
        eng = factory()
        img = _watermarked(eng, module)
        loc = eng.locate(img)
        box = eng.extract_mask(img, loc)
        assert box.dtype == np.uint8
        # Shape == loc.bbox, i.e. the old full-frame mask's [y:y+bh, x:x+bw] crop.
        assert box.shape == (loc.h, loc.w)
        # Footprint, not full frame: the box is a tiny fraction of the image.
        assert box.size * 4 < img.shape[0] * img.shape[1]
