"""Regression tests for motion-compensated visible-video fill."""

from __future__ import annotations

import cv2
import numpy as np

from pagedmark.video_temporal import stabilize_filled_frame


def _translated_pair() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    previous = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
    previous = cv2.GaussianBlur(previous, (5, 5), 0)
    current = cv2.warpAffine(
        previous,
        np.float32(((1, 0, 2), (0, 1, 1))),
        (128, 96),
        borderMode=cv2.BORDER_REFLECT,
    )
    return previous, current


def test_motion_aligned_prior_reduces_independent_fill_error() -> None:
    previous, current = _translated_pair()
    mask = np.zeros(current.shape[:2], dtype=np.uint8)
    mask[30:66, 45:85] = 255
    rng = np.random.default_rng(11)
    current_fill = current.copy()
    noise = rng.normal(0, 18, (36, 40, 3))
    current_fill[30:66, 45:85] = np.clip(
        current_fill[30:66, 45:85].astype(np.float32) + noise,
        0,
        255,
    ).astype(np.uint8)

    stabilized = stabilize_filled_frame(
        previous,
        previous,
        mask,
        current,
        current_fill,
        mask,
    )

    hole = mask > 0
    before = float(np.mean((current_fill[hole].astype(np.float32) - current[hole]) ** 2))
    after = float(np.mean((stabilized[hole].astype(np.float32) - current[hole]) ** 2))
    assert after < before * 0.5
    assert np.array_equal(stabilized[~hole], current_fill[~hole])


def test_owned_fill_can_be_stabilized_without_full_frame_copy() -> None:
    previous, current = _translated_pair()
    mask = np.zeros(current.shape[:2], dtype=np.uint8)
    mask[30:66, 45:85] = 255
    current_fill = current.copy()
    current_fill[mask > 0] = 127

    stabilized = stabilize_filled_frame(
        previous,
        previous,
        mask,
        current,
        current_fill,
        mask,
        copy=False,
    )

    assert stabilized is current_fill


def test_scene_cut_keeps_independent_current_fill() -> None:
    previous, _current = _translated_pair()
    rng = np.random.default_rng(13)
    current = rng.integers(0, 256, previous.shape, dtype=np.uint8)
    mask = np.zeros(current.shape[:2], dtype=np.uint8)
    mask[30:66, 45:85] = 255
    current_fill = current.copy()
    current_fill[mask > 0] = 0

    stabilized = stabilize_filled_frame(
        previous,
        previous,
        mask,
        current,
        current_fill,
        mask,
    )

    assert np.array_equal(stabilized, current_fill)


def test_disjoint_prior_mask_cannot_reintroduce_old_mark() -> None:
    previous, current = _translated_pair()
    previous_mask = np.zeros(current.shape[:2], dtype=np.uint8)
    previous_mask[8:24, 8:24] = 255
    current_mask = np.zeros(current.shape[:2], dtype=np.uint8)
    current_mask[60:76, 96:112] = 255
    current_fill = current.copy()
    current_fill[current_mask > 0] = 127

    stabilized = stabilize_filled_frame(
        previous,
        previous,
        previous_mask,
        current,
        current_fill,
        current_mask,
    )

    assert np.array_equal(stabilized, current_fill)
