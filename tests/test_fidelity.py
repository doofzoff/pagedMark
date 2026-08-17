"""Tests for the fidelity report.

Synthetic images throughout: the point of these metrics is that they are decidable
without a model, so their tests must be too.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pagedmark import fidelity


def _flat_dark_frame(width: int = 256, height: int = 256, level: int = 24) -> np.ndarray:
    """A dark, featureless BGR frame -- the case where a prior has room to invent."""
    return np.full((height, width, 3), level, dtype=np.uint8)


def _with_grain(frame: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    """Per-pixel noise: what a real camera adds, and what must NOT read as invention."""
    rng = np.random.default_rng(seed)
    noisy = frame.astype(np.float32) + rng.normal(0.0, sigma, frame.shape).astype(np.float32)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _with_blotches(frame: np.ndarray, amplitude: float, seed: int = 0) -> np.ndarray:
    """Mid-band coloured structure: what a diffusion prior actually leaves behind."""
    rng = np.random.default_rng(seed)
    height, width = frame.shape[:2]
    coarse = rng.normal(0.0, amplitude, (height // 8, width // 8, 3)).astype(np.float32)
    blotches = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    return np.clip(frame.astype(np.float32) + blotches, 0, 255).astype(np.uint8)


class TestPsnr:
    def test_identical_input_is_infinite(self):
        frame = _flat_dark_frame()
        assert fidelity.psnr(frame, frame) == float("inf")

    def test_a_known_offset_gives_the_known_value(self):
        """PSNR is arithmetic, so pin it against the closed form rather than a snapshot."""
        frame = _flat_dark_frame(level=100)
        shifted = frame.astype(np.int16) + 10
        expected = 10 * np.log10(255.0**2 / 100.0)
        assert fidelity.psnr(frame, shifted.astype(np.uint8)) == pytest.approx(expected, abs=1e-6)

    def test_mismatched_shapes_are_refused(self):
        with pytest.raises(ValueError, match="identical shapes"):
            fidelity.psnr(_flat_dark_frame(64, 64), _flat_dark_frame(32, 32))


class TestInventedTexture:
    """The metric exists because the obvious ones were blind to the artifact."""

    def test_an_untouched_frame_reads_one(self):
        frame = _with_grain(_flat_dark_frame(), sigma=3.0)
        assert fidelity.invented_texture(frame, frame) == pytest.approx(1.0)

    def test_mid_band_blotches_read_far_above_one(self):
        source = _with_grain(_flat_dark_frame(), sigma=3.0)
        regenerated = _with_blotches(source, amplitude=6.0)
        assert fidelity.invented_texture(source, regenerated) > 3.0

    def test_scaling_the_grain_does_not_move_it(self):
        """The discriminating case, and the reason this is a shape ratio, not an energy one.

        A plain band ratio reports doubled grain as 2.00x -- identically to doubled
        blotches -- because a linear filter preserves a factor of two in its input.
        Normalising the band by the fine detail cancels it: the number describes how the
        energy is distributed, not how much of it there is.
        """
        source = _with_grain(_flat_dark_frame(), sigma=3.0)
        grainier = _with_grain(_flat_dark_frame(), sigma=6.0, seed=1)
        assert fidelity.invented_texture(source, grainier) == pytest.approx(1.0, abs=0.2)

    def test_a_smoother_output_does_not_read_as_invention_either(self):
        """Denoising is a different loss, and conflating the two hides both."""
        source = _with_grain(_flat_dark_frame(), sigma=3.0)
        smoother = _with_grain(_flat_dark_frame(), sigma=1.0, seed=2)
        assert fidelity.invented_texture(source, smoother) == pytest.approx(1.0, abs=0.2)

    def test_a_region_too_small_to_measure_returns_none(self):
        tiny = _flat_dark_frame(24, 24)
        assert fidelity.invented_texture(tiny, tiny) is None

    def test_mismatched_shapes_are_refused(self):
        with pytest.raises(ValueError, match="identical shapes"):
            fidelity.invented_texture(_flat_dark_frame(64, 64), _flat_dark_frame(32, 32))


class TestCompare:
    def test_an_unchanged_pair_reports_no_change(self, tmp_path):
        path = tmp_path / "source.png"
        cv2.imwrite(str(path), _with_grain(_flat_dark_frame(), sigma=3.0))

        report = fidelity.compare(path, path, faces=False)

        assert (report.width, report.height) == (256, 256)
        assert report.psnr == float("inf")
        assert report.invented_texture == pytest.approx(1.0)
        assert report.faces == ()
        assert report.face_psnr is None

    def test_a_candidate_of_another_size_is_still_comparable(self, tmp_path):
        """Diffusers rounds to its latent grid, so an 8-pixel difference is routine."""
        source = tmp_path / "source.png"
        candidate = tmp_path / "candidate.png"
        frame = _with_grain(_flat_dark_frame(264, 264), sigma=3.0)
        cv2.imwrite(str(source), frame)
        cv2.imwrite(str(candidate), frame[:256, :256])

        report = fidelity.compare(source, candidate, faces=False)

        assert (report.width, report.height) == (264, 264)
        assert report.psnr > 0.0

    def test_each_face_is_measured_where_the_detector_finds_one(self, tmp_path, monkeypatch):
        source = tmp_path / "source.png"
        candidate = tmp_path / "candidate.png"
        frame = _with_grain(_flat_dark_frame(), sigma=3.0)
        cv2.imwrite(str(source), frame)
        # Damage only the face box, so a frame-wide average would barely notice.
        damaged = frame.copy()
        damaged[40:120, 40:120] = 200
        cv2.imwrite(str(candidate), damaged)
        monkeypatch.setattr(fidelity, "_detected_faces", lambda _path: [(40, 40, 120, 120)])

        report = fidelity.compare(source, candidate)

        assert len(report.faces) == 1
        assert report.faces[0].box == (40, 40, 120, 120)
        assert report.face_psnr is not None
        assert report.face_psnr < report.psnr, "a damaged face must score worse than the frame"

    def test_a_missing_detector_costs_one_row_not_the_report(self, tmp_path, monkeypatch):
        """The frame and texture numbers answer most questions on their own."""
        path = tmp_path / "source.png"
        cv2.imwrite(str(path), _with_grain(_flat_dark_frame(), sigma=3.0))
        monkeypatch.setattr(
            "pagedmark._internal.qwen_zimage_pipeline.detect_faces",
            lambda _image: (_ for _ in ()).throw(RuntimeError("detector unavailable")),
        )

        report = fidelity.compare(path, path)

        assert report.faces == ()
        assert report.psnr == float("inf")
