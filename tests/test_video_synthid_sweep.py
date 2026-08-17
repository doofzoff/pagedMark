"""Pure regression tests for the oracle-gated video SynthID experiment."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

if TYPE_CHECKING:
    from types import ModuleType

_SCRIPT = Path(__file__).parent.parent / "scripts" / "video_synthid_sweep.py"


@pytest.fixture(scope="module")
def sweep() -> ModuleType:
    spec = importlib.util.spec_from_file_location("video_synthid_sweep", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fit_size_preserves_landscape_aspect_and_vae_alignment(sweep: ModuleType) -> None:
    assert sweep._fit_size(1280, 720, 512) == (512, 288)


def test_fit_size_rejects_invalid_dimensions(sweep: ModuleType) -> None:
    with pytest.raises(ValueError, match="positive"):
        sweep._fit_size(0, 720, 512)


def test_shared_latent_noise_is_one_spatial_field(sweep: ModuleType) -> None:
    noise = sweep._shared_latent_noise(
        (4, 8, 8),
        seed=7,
        device="cpu",
        dtype=torch.float32,
    )
    assert noise.shape == (1, 4, 8, 8)


def test_shared_latent_noise_is_seeded(sweep: ModuleType) -> None:
    first = sweep._shared_latent_noise((4, 8, 8), seed=7, device="cpu", dtype=torch.float32)
    repeated = sweep._shared_latent_noise((4, 8, 8), seed=7, device="cpu", dtype=torch.float32)
    other = sweep._shared_latent_noise((4, 8, 8), seed=8, device="cpu", dtype=torch.float32)
    assert torch.equal(first, repeated)
    assert not torch.equal(first, other)


def test_psnr_is_infinite_for_identical_frames(sweep: ModuleType) -> None:
    frame = np.full((2, 8, 8, 3), 120, dtype=np.uint8)
    assert sweep.paired_psnr(frame, frame.copy()) == pytest.approx(float("inf"))


def test_temporal_residual_ratio_is_one_for_identical_sequences(sweep: ModuleType) -> None:
    first = np.zeros((32, 32, 3), dtype=np.uint8)
    second = first.copy()
    second[:, 8:16] = 80
    sequence = [first, second]
    maps, baseline = sweep.build_temporal_reference(sequence)
    assert sweep.temporal_residual_ratio([frame.copy() for frame in sequence], maps, baseline) == pytest.approx(1.0)
