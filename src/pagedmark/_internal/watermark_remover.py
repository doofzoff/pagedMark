"""Project-native orchestration for diffusion-based pixel regeneration."""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportPrivateUsage=false, reportInvalidTypeForm=false, reportConstantRedefinition=false, reportUnnecessaryComparison=false
from __future__ import annotations

import contextlib
import logging
import os
import subprocess
from typing import TYPE_CHECKING, Any

from PIL import Image

from pagedmark import optional_deps
from pagedmark._internal.watermark_profiles import (
    CPU_DEVICE,
    CUDA_DEVICE,
    MPS_DEVICE,
    REMOVAL_MODULES,
    SDXL_ZIMAGE_PROFILE,
    install_extra_for_device,
    plan_profile,
    required_modules,
    resolve_seed,
    resolve_strength,
)
from pagedmark.optional_deps import module_available

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pagedmark._internal.text_restoration import VerifiedTextManifest

logger = logging.getLogger(__name__)

# Read by torch's MPS backend when it initializes, so it has to be set before the
# import below rather than at the call site. Metal implements every op the SDXL global
# stage uses today; the fallback is here so a future op gap degrades to a slow CPU
# kernel instead of aborting a run that is minutes in. Never overrides an operator's
# own choice -- `setdefault`, so `PYTORCH_ENABLE_MPS_FALLBACK=0` still wins.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

# Probed once at import. ``torch`` is imported above rather than probed because this
# module needs the object, not just the answer.
_HAS_REMOVAL_MODULES = module_available(*(name for name in REMOVAL_MODULES if name != "torch"))


def is_watermark_removal_available(device: str | None = None) -> bool:
    """Return whether the removal runtime for ``device`` can be imported.

    ``device`` defaults to the auto-detected one. The answer is per-device because the
    module list is: an MPS host never reaches the DiffSynth face stage, so demanding
    diffsynth there refuses a run that works.
    """
    if not _HAS_TORCH:
        return False
    resolved = device or get_device()
    if resolved == CUDA_DEVICE:
        return _HAS_REMOVAL_MODULES
    # Through the module rather than the name bound at import: the CUDA answer is a
    # probe cached at import time and patchable as a flag, and the per-device answer
    # has to be interceptable at the same seam or a test can only reach one of them.
    return optional_deps.module_available(*(name for name in required_modules(resolved) if name != "torch"))


def _ensure_watermark_deps(device: str) -> None:
    if not is_watermark_removal_available(device):
        missing = ", ".join(required_modules(device))
        raise ImportError(
            f"Invisible watermark regeneration on {device} requires {missing}: "
            f"pip install {install_extra_for_device(device)}."
        )


def _has_nvidia_gpu() -> bool:
    try:
        subprocess.run(
            ["nvidia-smi"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return True


def _backend_works(device: str) -> bool:
    """Run one real op on ``device``: availability flags can outlive a usable build."""
    try:
        probe = torch.tensor([1.0], device=device)  # type: ignore[union-attr]
        _ = probe + probe
    except (AssertionError, RuntimeError):
        return False
    return True


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)  # type: ignore[union-attr]
    return bool(backend is not None and backend.is_available())


def get_device() -> str:
    """Return the best device a removal profile can run on: cuda, mps, or cpu.

    CUDA first: it is the only device that runs both profiles and the only one the
    operating points were certified on. MPS second, where ``sdxl-zimage`` runs its
    global stage. ``"cpu"`` means "no GPU backend", which is what the refusal in
    :class:`WatermarkRemover` then reports.
    """
    if not _HAS_TORCH:
        return CPU_DEVICE
    if torch.cuda.is_available() and _backend_works(CUDA_DEVICE):  # type: ignore[union-attr]
        return CUDA_DEVICE
    if _has_nvidia_gpu():
        logger.warning("NVIDIA GPU detected, but the installed PyTorch build has no working CUDA backend")
    if _mps_available() and _backend_works(MPS_DEVICE):
        return MPS_DEVICE
    return CPU_DEVICE


def device_memory_gib(device: str) -> float:
    """Usable device memory in GiB, or 0.0 when it cannot be read.

    CUDA reports its card capacity. MPS reports the *recommended* working-set limit
    rather than installed RAM: on a 16 GiB Mac that is ~11.8 GiB, and exceeding it is
    what makes Metal start swapping instead of failing, so it is the number the memory
    policies must be written against.
    """
    if not _HAS_TORCH:
        return 0.0
    if device == CUDA_DEVICE:
        with contextlib.suppress(Exception):
            return torch.cuda.get_device_properties(CUDA_DEVICE).total_memory / (1024**3)  # type: ignore[union-attr]
        return 0.0
    if device == MPS_DEVICE:
        with contextlib.suppress(Exception):
            return torch.mps.recommended_max_memory() / (1024**3)  # type: ignore[union-attr]
    return 0.0


def empty_device_cache(device: str) -> None:
    """Release cached blocks between stages. A no-op where the backend has no cache."""
    if not _HAS_TORCH:
        return
    with contextlib.suppress(Exception):
        if device == CUDA_DEVICE:
            torch.cuda.empty_cache()  # type: ignore[union-attr]
        elif device == MPS_DEVICE:
            torch.mps.empty_cache()  # type: ignore[union-attr]


class WatermarkRemover:
    """Load one regeneration profile and write a metadata-clean raster output."""

    def __init__(
        self,
        device: str | None = None,
        progress_callback: Callable[[str], None] | None = None,
        hf_token: str | None = None,
        pipeline: str | None = None,
        controlnet_conditioning_scale: float = 1.0,
        cpu_offload: bool = False,
    ) -> None:
        # There is no ``model_id`` parameter and no ``model_id`` attribute: each
        # profile pins a fixed model stack, and the dtype below is bound to that
        # stack's weights. Both used to be constructor overrides that existed only to
        # be rejected or to break the run, and the attribute only existed to echo the
        # rejected value back.
        selected_device = (device or get_device()).casefold()
        self.device = get_device() if selected_device == "auto" else selected_device
        # Device support is a precondition of the object, not of the run: a profile
        # whose stack cannot load here would otherwise fail at model-load time, several
        # layers down and under the wrong profile's name. ``pipeline=None`` means the
        # caller expressed no preference, which is what lets MPS pick the one profile
        # it can run instead of failing on a default it never chose.
        self.plan = plan_profile(pipeline, self.device)
        self.model_profile = self.plan.profile
        self.face_stage = self.plan.face_stage
        if self.plan.note:
            logger.info("%s", self.plan.note)
            if progress_callback is not None:
                with contextlib.suppress(Exception):
                    progress_callback(self.plan.note)
        _ensure_watermark_deps(self.device)

        if self.model_profile == SDXL_ZIMAGE_PROFILE:
            # SDXL ships fp16 weights and an fp16-safe VAE; bf16 would give up the
            # variant without buying anything on this architecture. MPS runs the same
            # fp16 stack: Metal implements half natively, and the fp16-fix VAE is what
            # keeps the decode from overflowing there as much as on CUDA.
            self.torch_dtype = torch.float16  # type: ignore[union-attr]
        else:
            self.torch_dtype = torch.bfloat16  # type: ignore[union-attr]

        self.cpu_offload = cpu_offload
        self.controlnet_conditioning_scale = controlnet_conditioning_scale
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self._progress_callback = progress_callback
        self._qwen_zimage_pipeline: Any = None

    def preload(self, *, global_only: bool = False) -> None:
        """Materialize the selected model stack before the first request."""
        self._load_qwen_zimage_pipeline().preload(global_only=global_only)

    def _load_qwen_zimage_pipeline(self) -> Any:
        if self._qwen_zimage_pipeline is None:
            if self.model_profile == SDXL_ZIMAGE_PROFILE:
                from pagedmark._internal.sdxl_zimage_pipeline import (
                    SdxlZImagePipeline as _Pipeline,
                )
            else:
                from pagedmark._internal.qwen_zimage_pipeline import (
                    QwenZImagePipeline as _Pipeline,
                )

            self._qwen_zimage_pipeline = _Pipeline(
                device=self.device,
                torch_dtype=self.torch_dtype,
                hf_token=self.hf_token,
                progress_callback=self._progress_callback,
                controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                keep_face_models_on_device=False if self.cpu_offload else None,
                keep_global_models_on_device=False if self.cpu_offload else None,
                face_stage=self.face_stage,
            )
        return self._qwen_zimage_pipeline

    def _write_output(self, image: Image.Image, output_path: Path) -> None:
        import numpy as np

        from pagedmark import image_io

        output_path.parent.mkdir(parents=True, exist_ok=True)
        bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
        if not image_io.imwrite(str(output_path), bgr):
            image.save(output_path)
        from pagedmark.metadata import remove_ai_metadata

        remove_ai_metadata(output_path, output_path, keep_standard=True)

    def remove_watermark(
        self,
        image_path: Path,
        output_path: Path | None = None,
        strength: float | None = None,
        seed: int | None = None,
        vendor: str | None = None,
        tile: bool = False,
        tile_size: int = 1024,
        tile_overlap: int = 128,
        text_manifest: VerifiedTextManifest | None = None,
    ) -> Path:
        """Regenerate image pixels and write the result without AI metadata.

        Step count and CFG are not parameters. Each stage of both profiles is a
        distilled schedule that owns its own, so the only thing a caller-supplied
        value could do is break the run or be rejected.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        destination = output_path or image_path
        with Image.open(image_path) as opened:
            source = opened.convert("RGB")

        resolved_strength = resolve_strength(strength, vendor, self.model_profile, size=source.size)
        if not 0.0 <= resolved_strength <= 1.0:
            raise ValueError(f"Strength must be between 0.0 and 1.0, got {resolved_strength}")
        if text_manifest is not None and self.model_profile == SDXL_ZIMAGE_PROFILE:
            raise ValueError("Verified text restoration is supported only by the qwen-zimage profile")
        if text_manifest is not None and tile:
            raise ValueError("Verified text restoration is not calibrated with tiled diffusion")

        result = self._load_qwen_zimage_pipeline().run(
            source,
            strength=resolved_strength,
            seed=resolve_seed(seed),
            tile=tile,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            text_manifest=text_manifest,
        )
        self._write_output(result, destination)
        return destination
