"""Pure motion-compensated helpers shared by video pipelines."""

from __future__ import annotations

# OpenCV exposes incomplete types for optical-flow and remap operations.
# Public signatures remain annotated while this third-party boundary is relaxed.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportMissingTypeStubs=false, reportCallIssue=false, reportArgumentType=false
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


def _backward_map(
    current_gray: NDArray[Any],
    previous_gray: NDArray[Any],
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Build a remap from a previous frame into current coordinates."""
    flow = cv2.calcOpticalFlowFarneback(
        current_gray,
        previous_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )
    height, width = current_gray.shape
    flow[..., 0] += np.arange(width, dtype=np.float32)[None, :]
    flow[..., 1] += np.arange(height, dtype=np.float32)[:, None]
    return flow[..., 0], flow[..., 1]


def _backward_warp(
    image: NDArray[Any],
    maps: tuple[NDArray[Any], NDArray[Any]],
    *,
    interpolation: int = cv2.INTER_LINEAR,
) -> NDArray[Any]:
    """Apply a precomputed backward optical-flow map."""
    return cv2.remap(
        image,
        maps[0],
        maps[1],
        interpolation=interpolation,
        borderMode=cv2.BORDER_REFLECT,
    )


def _motion_residual(
    current: NDArray[Any],
    previous: NDArray[Any],
    maps: tuple[NDArray[Any], NDArray[Any]],
) -> float:
    """Return mean absolute residual after warping the previous frame."""
    current_f32 = np.asarray(current, dtype=np.float32)
    previous_f32 = np.asarray(previous, dtype=np.float32)
    warped_previous = _backward_warp(previous_f32, maps)
    return float(np.mean(np.abs(current_f32 - warped_previous)))


def build_temporal_reference(
    reference: Sequence[NDArray[Any]],
) -> tuple[tuple[tuple[NDArray[Any], NDArray[Any]], ...], float]:
    """Precompute source motion maps and its mean residual."""
    if len(reference) < 2:
        raise ValueError("Temporal metric needs at least two frames")
    maps: list[tuple[NDArray[Any], NDArray[Any]]] = []
    reference_residuals: list[float] = []
    for index in range(1, len(reference)):
        current_gray = cv2.cvtColor(reference[index], cv2.COLOR_BGR2GRAY)
        previous_gray = cv2.cvtColor(reference[index - 1], cv2.COLOR_BGR2GRAY)
        frame_maps = _backward_map(current_gray, previous_gray)
        maps.append(frame_maps)
        reference_residuals.append(_motion_residual(reference[index], reference[index - 1], frame_maps))
    return tuple(maps), float(np.mean(reference_residuals))


def temporal_residual_ratio(
    candidate: Sequence[NDArray[Any]],
    maps: Sequence[tuple[NDArray[Any], NDArray[Any]]],
    baseline: float,
) -> float:
    """Measure candidate flicker against a precomputed source residual."""
    if len(candidate) != len(maps) + 1:
        raise ValueError("Temporal metric needs one map per adjacent frame pair")
    candidate_residuals: list[float] = []
    for index, frame_maps in enumerate(maps, start=1):
        candidate_residuals.append(_motion_residual(candidate[index], candidate[index - 1], frame_maps))
    measured = float(np.mean(candidate_residuals))
    return measured / max(baseline, 1e-6)


def stabilize_filled_frame(
    previous_source: NDArray[Any],
    previous_cleaned: NDArray[Any],
    previous_mask: NDArray[Any],
    current_source: NDArray[Any],
    current_cleaned: NDArray[Any],
    current_mask: NDArray[Any],
    *,
    blend: float = 0.5,
    max_context_residual: float = 12.0,
    copy: bool = True,
) -> NDArray[Any]:
    """Blend a motion-aligned prior fill when nearby source pixels agree.

    The prior contributes only where its warped removal mask covers the current
    mask. A context ring outside both masks gates the blend, so scene cuts or
    non-rigid local changes keep the independent current-frame fill.
    """
    if not 0.0 <= blend <= 1.0:
        raise ValueError("Temporal blend must be between 0 and 1")
    if max_context_residual <= 0.0:
        raise ValueError("Context residual threshold must be positive")
    if (
        previous_source.shape != current_source.shape
        or previous_cleaned.shape != current_cleaned.shape
        or previous_source.shape != previous_cleaned.shape
        or previous_mask.shape != current_mask.shape
        or previous_mask.shape != current_source.shape[:2]
    ):
        raise ValueError("Temporal fill inputs must share frame and mask geometry")

    union = (previous_mask > 0) | (current_mask > 0)
    ys, xs = np.where(union)
    if len(xs) == 0:
        return current_cleaned
    height, width = current_mask.shape
    mask_width = int(xs.max() - xs.min() + 1)
    mask_height = int(ys.max() - ys.min() + 1)
    padding = max(24, round(max(mask_width, mask_height) * 0.75))
    x0 = max(0, int(xs.min()) - padding)
    y0 = max(0, int(ys.min()) - padding)
    x1 = min(width, int(xs.max()) + padding + 1)
    y1 = min(height, int(ys.max()) + padding + 1)

    previous_source_crop = previous_source[y0:y1, x0:x1]
    current_source_crop = current_source[y0:y1, x0:x1]
    current_cleaned_crop = current_cleaned[y0:y1, x0:x1]
    previous_cleaned_crop = previous_cleaned[y0:y1, x0:x1]
    previous_mask_crop = previous_mask[y0:y1, x0:x1]
    current_mask_crop = current_mask[y0:y1, x0:x1]
    maps = _backward_map(
        cv2.cvtColor(current_source_crop, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(previous_source_crop, cv2.COLOR_BGR2GRAY),
    )
    warped_previous_source = _backward_warp(previous_source_crop, maps)
    warped_previous_cleaned = _backward_warp(previous_cleaned_crop, maps)
    warped_previous_mask = _backward_warp(
        previous_mask_crop,
        maps,
        interpolation=cv2.INTER_NEAREST,
    )

    current_hole = current_mask_crop > 0
    if not np.any(current_hole):
        return current_cleaned
    covered = current_hole & (warped_previous_mask > 0)
    if float(np.mean(covered[current_hole])) < 0.85:
        return current_cleaned

    occupied = current_hole | (warped_previous_mask > 0)
    dilation = max(7, round(max(mask_width, mask_height) * 0.25))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation | 1, dilation | 1))
    context = cv2.dilate(occupied.astype(np.uint8), kernel).astype(bool) & ~occupied
    if np.count_nonzero(context) < 64:
        return current_cleaned
    residual = np.abs(current_source_crop.astype(np.float32) - warped_previous_source.astype(np.float32))
    context_residual = float(np.mean(residual[context]))
    if context_residual > max_context_residual:
        return current_cleaned

    effective_blend = blend * (1.0 - context_residual / max_context_residual)
    blended = (1.0 - effective_blend) * current_cleaned_crop[covered].astype(
        np.float32
    ) + effective_blend * warped_previous_cleaned[covered].astype(np.float32)
    result = current_cleaned.copy() if copy else current_cleaned
    result_crop = result[y0:y1, x0:x1]
    result_crop[covered] = np.clip(blended, 0, 255).astype(np.uint8)
    return result
