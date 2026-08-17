"""Unit tests for the --cpu-offload device-placement branch.

``WatermarkRemover._move_to_device_and_optimize`` chooses between a full
``pipeline.to("cuda")`` and ``enable_model_cpu_offload()``. The placement
decision is exercised with a mock pipeline and an uninitialized remover, so the
core CI matrix needs no diffusion dependency, model download, or GPU.
"""

from __future__ import annotations

from pagedmark._internal.watermark_remover import WatermarkRemover


def _remover(device: str, cpu_offload: bool) -> WatermarkRemover:
    remover = WatermarkRemover.__new__(WatermarkRemover)
    remover.device = device
    remover.cpu_offload = cpu_offload
    remover._progress_callback = None
    return remover
