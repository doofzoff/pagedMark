"""Jimeng-basic 'AI生成' pill: a CAPTURE-LESS visible mark (issue #54).

The Jimeng free-tier TC260 label is a rounded pill with 'AI生成' in the TOP-LEFT
corner -- distinct from the ``jimeng`` "★ 即梦AI" wordmark (bottom-right). It has no
captured alpha map, so unlike the other marks it is detected purely by a synthetic
silhouette; like every mark it is then removed by the shared localize -> fill:

  * Detect: edge-NCC of a font-rendered SILHOUETTE (``assets/jimeng_pill.png``,
    synthetic, data-safe -- see ``scripts/render_pill_silhouette.py``) against the
    top-left ROI, at the pill's known width fraction. The calibrated
    ``_DETECT_THRESHOLD`` is 0.22.
  * Remove: place the pill footprint at the matched location and inpaint it
    (MI-GAN / cv2 via the registry). Quality comes from the inpaint backend, so the
    silhouette need not be pixel-accurate -- which is why a synthetic render is
    sufficient and no source-derived asset is committed.

Geometry uses width ~0.161*W,
height ~0.091*W, top-left, margins ~0.02-0.05.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import cv2
import numpy as np

from pagedmark import image_io

if TYPE_CHECKING:
    from numpy.typing import NDArray

# cv2/numpy boundary: cv2 ships no usable type info, so strict pyright cannot know
# its array element types. Relax the unknown-type rules for this file only; the
# public signatures are still annotated with NDArray[Any].
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportAttributeAccessIssue=false, reportUnnecessaryComparison=false

_ASSET = Path(__file__).parent / "assets" / "jimeng_pill.png"

# Geometry (fractions of image WIDTH unless noted); top-left corner.
_WIDTH_FRAC = 0.161
_ROI_W_FRAC = 0.34  # search window width (of W)
_ROI_H_FRAC = 0.14  # search window height (of H)
_DETECT_THRESHOLD = 0.22  # calibrated edge-NCC gate
# Inpaint mask GEOMETRY (fractions of W unless noted): a generous fixed top-left box
# covering the pill (measured ~0.167*W wide, ~0.09*W tall, margin ~0.02-0.05) plus
# margin. The mask uses stable geometry, NOT the NCC match position -- the synthetic
# silhouette localizes only approximately, and the corner is negative space, so
# over-covering is harmless while a match-positioned box leaves outline residue.
_MASK_X0, _MASK_Y0 = 0.012, 0.006  # x0 of W, y0 of H
_MASK_W, _MASK_H = 0.205, 0.115  # width of W, height of W

# Background-flatness gate for the metadata-only pill arm (see remove_auto_marks).
# The pill detector is weak (~7% raw false-fire); metadata confirms the platform,
# not pill presence, so its false fires are real Jimeng-class content WITHOUT a pill.
# Those false fires cluster on TEXTURED top-left corners (ceiling fixtures, structure)
# where inpaint visibly SMEARS, while real pills and harmless false fires sit on FLAT
# corners (sky / wall / solid) where inpaint is invisible. So the metadata-only arm
# removes the pill only when the footprint background is flat enough for a safe,
# invisible inpaint. Threshold = median Sobel magnitude over the footprint box at a
# normalized width. The reliable bottom-right wordmark arm is NOT texture-gated:
# a wordmark-confirmed pill is removed regardless.
#
# Measured through the PRODUCT path (the `_keep_pill` gate), not the raw detector, by
# ``scripts/pill_gate_audit.py`` -- the raw path bypasses the gate and reads as a
# disaster that the shipped behaviour does not have. Re-run it when the gate changes.
_FLAT_TEXTURE_MAX = 6.0

_silhouette: NDArray[Any] | None = None


class PillDetection(NamedTuple):
    detected: bool
    confidence: float
    region: tuple[int, int, int, int]  # x, y, w, h of the matched pill


def _load_silhouette() -> NDArray[Any] | None:
    global _silhouette
    if _silhouette is None:
        if not _ASSET.exists():
            return None
        _silhouette = image_io.imread(str(_ASSET), cv2.IMREAD_GRAYSCALE)
    return _silhouette


def _grad(gray: NDArray[Any]) -> NDArray[Any]:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.normalize(cv2.magnitude(gx, gy), None, 0, 255, cv2.NORM_MINMAX)


class PillEngine:
    """Detect + build the removal mask for the top-left 'AI生成' pill (edge-NCC of a synthetic silhouette)."""

    def _match(self, image: NDArray[Any]) -> tuple[float, tuple[int, int, int, int]] | None:
        sil = _load_silhouette()
        if sil is None or image is None or image.size == 0:
            return None
        h, w = image.shape[:2]
        if h < 64 or w < 64:
            return None
        gray = cv2.cvtColor(image_io.to_bgr(image), cv2.COLOR_BGR2GRAY)
        rh, rw = int(h * _ROI_H_FRAC), int(w * _ROI_W_FRAC)
        roi = gray[0:rh, 0:rw]
        tw = max(24, int(_WIDTH_FRAC * w))
        th = max(12, int(tw * sil.shape[0] / sil.shape[1]))
        if th >= rh or tw >= rw:
            return None
        tmpl = cv2.resize(sil, (tw, th))
        res = cv2.matchTemplate(_grad(roi.astype(np.float32)), _grad(tmpl.astype(np.float32)), cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        return float(score), (int(loc[0]), int(loc[1]), tw, th)

    def detect(self, image: NDArray[Any]) -> PillDetection:
        m = self._match(image)
        if m is None:
            return PillDetection(False, 0.0, (0, 0, 0, 0))
        score, box = m
        return PillDetection(score >= _DETECT_THRESHOLD, score, box)

    def _footprint_box(self, image: NDArray[Any]) -> tuple[int, int, int, int] | None:
        h, w = image.shape[:2]
        x0, y0 = int(_MASK_X0 * w), int(_MASK_Y0 * h)
        x1, y1 = min(w, x0 + int(_MASK_W * w)), min(h, y0 + int(_MASK_H * w))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    def footprint_texture(self, image: NDArray[Any]) -> float:
        """Median gradient magnitude over the fixed top-left footprint box at a
        normalized width. A robust flatness proxy: low = flat (sky / wall / solid,
        inpaint invisible), high = textured (ceiling fixtures / structure, inpaint
        smears). Median (not mean) so the pill's own edges -- a minority of the box --
        do not inflate it. Backs the metadata-only arm's safe-inpaint gate."""
        if image is None or image.size == 0:
            return 0.0
        box = self._footprint_box(image)
        if box is None:
            return 0.0
        x0, y0, x1, y1 = box
        crop = image[y0:y1, x0:x1]
        gray = cv2.cvtColor(image_io.to_bgr(crop), cv2.COLOR_BGR2GRAY)
        tw = 220
        gray = cv2.resize(gray, (tw, max(1, int(gray.shape[0] * tw / gray.shape[1])))).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return float(np.median(cv2.magnitude(gx, gy)))

    def footprint_is_flat(self, image: NDArray[Any], *, thresh: float = _FLAT_TEXTURE_MAX) -> bool:
        """True when the top-left footprint is flat enough for an invisible inpaint."""
        return self.footprint_texture(image) <= thresh

    def footprint_mask(self, image: NDArray[Any], *, force: bool = False) -> NDArray[Any] | None:
        """Full-frame uint8 mask (255 = pill) over the pill's known top-left region.

        Uses stable GEOMETRY (a generous fixed box), not the NCC match position: the
        synthetic silhouette localizes only approximately, so a match-positioned mask
        leaves outline residue, while the top-left corner is negative space, so a
        generous geometric box removes the pill cleanly and harmlessly. The caller
        gates on :meth:`detect`, so a clean corner is never masked. ``force`` is
        accepted for a uniform engine signature but ignored (the geometry box is
        fixed regardless)."""
        if image is None or image.size == 0:
            return None
        box = self._footprint_box(image)
        if box is None:
            return None
        # Same primitive the shared fill uses, rather than a private zeros/fill copy.
        # `dilate=0` because this footprint is already generous by construction; the
        # box is clamped to the frame in _footprint_box and both origins are positive
        # fractions, so boxes_to_mask's own clamping is a no-op here.
        from pagedmark import region_eraser

        x0, y0, x1, y1 = box
        return region_eraser.boxes_to_mask(image.shape[:2], [(x0, y0, x1 - x0, y1 - y0)], dilate=0)
