"""The qwen-zimage recipe with an SDXL global stage.

Only the global regeneration model changes. The face stage is inherited verbatim
from :class:`QwenZImagePipeline` -- same YuNet detection, same SAM masks, same
Z-Image Turbo repair of the original crops, same feathered compositing -- so a
change there cannot silently diverge between the two profiles.

Three pieces cannot be shared, because they are bound to the architecture: the
ControlNet, the four-step distillation LoRA, and the sampler. Strength is bound to
it too, which is the part that is easy to miss: an SDXL global pass leaves SynthID
at the strength Qwen needs. See ``watermark_profiles.SDXL_ZIMAGE_OPENAI_STRENGTH``.

This is also the only profile that runs on Apple Silicon. Its global stage is plain
Diffusers in fp16, which Metal implements; the inherited face stage is not, so on MPS
``face_stage`` is False and this stage runs alone. Three MPS-specific details carry
weight, and each is commented at its site: memory-saving execution (attention slicing,
and a tiled VAE only where it is needed), a CPU generator, and cache release between
the load and the run.
"""

# Diffusers and torch expose mostly untyped tensor APIs. Keep the relaxation local
# to this optional ML boundary.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportPrivateUsage=false, reportPrivateImportUsage=false
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, ClassVar

from PIL import Image

from pagedmark._internal.qwen_zimage_pipeline import (
    GLOBAL_STEPS,
    QwenZImagePipeline,
    build_canny_control_image,
)
from pagedmark._internal.watermark_profiles import (
    CONTROLNET_CANNY_MODEL,
    CPU_DEVICE,
    MPS_DEVICE,
    PROFILE_DEVICES,
    SDXL_FACE_CROP_CAP,
    SDXL_FACE_STEPS,
    SDXL_LIGHTNING_MODEL_ID,
    SDXL_LIGHTNING_PATTERN,
    SDXL_MODEL_ID,
    SDXL_ZIMAGE_PROFILE,
    face_stage_uses_zimage,
    sdxl_face_strength,
    sdxl_global_steps,
    sdxl_lightning_enabled,
)

log = logging.getLogger(__name__)

SDXL_VAE_MODEL_ID = "madebyollin/sdxl-vae-fp16-fix"
# SDXL aligns to an 8-pixel latent grid, against Qwen's 16.
_LATENT_GRID = 8

# Loaded weights for this stack: SDXL UNet + the two text encoders + the Canny
# ControlNet + the VAE, all fp16. Measured resident on MPS (9.67 GiB driver / 9.16 GiB
# live on an M5) rather than derived from parameter counts, because the ControlNet
# ships fp32 on the Hub and is cast on load.
_SDXL_STACK_GIB = 9.7
# What one forward pass adds per megapixel of input on the memory-saving execution path
# (attention slicing + tiled VAE). This is a THRESHOLD, not a fitted curve: it is
# bracketed by two measured runs on an M5 with an 11.84 GiB recommended working set,
# where a 1.57 MP pass peaked at 10.92 GiB and finished in 23 s, and a 2.5 MP pass went
# into swap and did not finish in twelve minutes. Any value in (0.86, 1.36) separates
# those two observations; the midpoint is taken. Refine it with a measurement, not with
# an estimate. Full record: docs/module-internals.md, "Apple Silicon (MPS)".
_SDXL_ACTIVATION_GIB_PER_MP = 1.15


# What the VAE decode adds over the weights when it runs on the whole frame at once,
# per megapixel. Measured on the same M5: the 1.57 MP run peaked at 18.74 GiB with
# tiling disabled against 10.92 GiB with it, so ~9.1 GiB over a 9.67 GiB stack.
_SDXL_UNTILED_VAE_GIB_PER_MP = 5.8


def vae_tiling_needed(width: int, height: int, memory_gib: float) -> bool:
    """Whether the VAE has to decode in tiles for this frame to fit the working set.

    Tiling is not free: its boundaries leave a faint texture. On one regenerated night
    photo the CJK text-mark detector read the tiled output as a 0.38-confidence Yuanbao
    mark, and dropping EITHER the tiling or the distilled sampler (which invented the
    texture the tiling then patterned) cleared it. So tiling is applied where it buys
    something rather than everywhere. On a 16 GiB Mac that is every size -- an untiled
    1.57 MP decode needs 18.74 GiB against an 11.84 GiB budget and survives only by
    paging; on a 64 GiB Mac it is only the large inputs.

    An unreadable budget (0.0) tiles, for the same reason the single-pass check treats
    it as "does not fit": unknown is when the bounded path is the right default.
    """
    if memory_gib <= 0.0:
        return True
    megapixels = (width * height) / 1_000_000
    return _SDXL_STACK_GIB + megapixels * _SDXL_UNTILED_VAE_GIB_PER_MP > memory_gib


def fits_in_one_pass(width: int, height: int, memory_gib: float) -> bool:
    """Whether a native-resolution pass fits the device's working set.

    0.0 means "unknown", which must not be read as "unlimited": an unreadable budget
    is exactly the case where a native pass on a large input is the risky choice, and
    the caller's fallback (tiling) is correct at every size.

    Pure arithmetic so the tiling decision is unit-testable without a GPU -- the same
    reason ``_target_size`` is a free function in the engine.
    """
    if memory_gib <= 0.0:
        return False
    megapixels = (width * height) / 1_000_000
    return _SDXL_STACK_GIB + megapixels * _SDXL_ACTIVATION_GIB_PER_MP <= memory_gib


def sdxl_target_size(width: int, height: int) -> tuple[int, int]:
    """Floor dimensions to SDXL's latent grid without changing aspect."""
    return max(_LATENT_GRID, (width // _LATENT_GRID) * _LATENT_GRID), max(
        _LATENT_GRID, (height // _LATENT_GRID) * _LATENT_GRID
    )


def requested_steps(effective_steps: int, strength: float) -> int:
    """Translate "spend N denoising steps" into what Diffusers has to be asked for.

    The two runtimes truncate differently and it is easy to port this wrong.
    DiffSynth sets ``sigma_start = denoising_strength`` and then runs *every*
    requested step across the shortened sigma range. Diffusers img2img instead
    truncates the step *count* (``init_timestep = int(steps * strength)``), so
    asking it for four steps at strength 0.15 runs **zero** and returns nothing but
    a VAE round-trip. Ask for enough that ``effective_steps`` actually execute.
    """
    return max(1, math.ceil(effective_steps / max(float(strength), 1e-6)))


@dataclass
class SdxlZImagePipeline(QwenZImagePipeline):
    """Lazy runtime for the SDXL global stage plus the inherited face stage."""

    supported_devices: ClassVar[tuple[str, ...]] = PROFILE_DEVICES[SDXL_ZIMAGE_PROFILE]

    def __post_init__(self) -> None:
        super().__post_init__()
        self._sdxl_pipe: Any = None

    def _load_sdxl(self) -> Any:
        if self._sdxl_pipe is not None:
            return self._sdxl_pipe
        self._require_supported_device()
        import torch
        from diffusers import (
            AutoencoderKL,
            ControlNetModel,
            EulerDiscreteScheduler,
            StableDiffusionXLControlNetImg2ImgPipeline,
        )
        from huggingface_hub import hf_hub_download

        lightning = sdxl_lightning_enabled(self.device)
        self._progress(
            "Loading SDXL, Lightning LoRA, and Canny ControlNet..."
            if lightning
            else "Loading SDXL and Canny ControlNet (undistilled sampling)..."
        )
        token = {"token": self.hf_token} if self.hf_token else {}
        controlnet = ControlNetModel.from_pretrained(CONTROLNET_CANNY_MODEL, torch_dtype=torch.float16, **token)
        vae = AutoencoderKL.from_pretrained(SDXL_VAE_MODEL_ID, torch_dtype=torch.float16, **token)
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            SDXL_MODEL_ID,
            controlnet=controlnet,
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            add_watermarker=False,
            **token,
        ).to(self.device)
        if lightning:
            # SDXL's own four-step distillation, at the strength its authors document.
            # The reference graph loads the Qwen LoRA at 0.8; carrying that number to a
            # different LoRA on a different architecture would be imitation, not parity.
            pipe.load_lora_weights(hf_hub_download(SDXL_LIGHTNING_MODEL_ID, SDXL_LIGHTNING_PATTERN, **token))
            pipe.fuse_lora()
            # SDXL-Lightning is distilled against trailing timestep spacing.
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        # Without the LoRA the checkpoint's own scheduler config is the right one: the
        # trailing spacing above exists for the distillation, not for the base model.
        if self.device == MPS_DEVICE:
            from pagedmark._internal.watermark_remover import empty_device_cache

            # Metal has no equivalent of CUDA's memory pooling across a long attention
            # matmul, and the working set it recommends is a fraction of installed RAM
            # (~11.8 of 16 GiB). Sliced attention keeps the peak bounded by the weights
            # rather than by the resolution, at no cost to the output. VAE tiling does
            # cost something, so it is decided per image in `_run_global` instead.
            pipe.enable_attention_slicing()
            pipe.vae.enable_slicing()
            # The LoRA fuse and the fp32->fp16 ControlNet cast leave a full copy of
            # their inputs in Metal's cache. Releasing here means the first forward
            # pass starts from the weights alone.
            empty_device_cache(self.device)
        self._sdxl_pipe = pipe
        return pipe

    def preload(self, *, global_only: bool = False) -> None:
        """Eagerly load the mandatory stage and, by default, the face stack."""
        from pagedmark._internal.qwen_zimage_pipeline import _yunet_model_path

        self._load_sdxl()
        if not self.face_stage:
            return
        _yunet_model_path()
        if not global_only:
            self._load_zimage()
            self._load_sam()

    def _face_crop_cap(self) -> int | None:
        """Cap the crop where the crop's size, not the face's, decides the wall time."""
        return None if face_stage_uses_zimage(self.device) else SDXL_FACE_CROP_CAP

    def _release_sam_if_transient(self) -> None:
        """On Metal, SAM and the SDXL stack do not fit next to each other comfortably."""
        if face_stage_uses_zimage(self.device):
            return
        self._sam_model = None
        self._sam_processor = None
        from pagedmark._internal.watermark_remover import empty_device_cache

        empty_device_cache(self.device)

    def _regenerate_face_crop(
        self,
        crop: Image.Image,
        index: int,
        total: int,
        *,
        strength: float,
        seed: int | None,
    ) -> Image.Image:
        """Repair one face with the resident SDXL stack where Z-Image cannot load.

        Same contract as the stage it stands in for -- the crop comes from the ORIGINAL
        and is regenerated at a lower strength than the global pass, then feathered into
        the global result by the caller -- so identity survives a pass that was tuned for
        the frame. The strength is floored because fp16 sampling stops producing an image
        below it, not because a softer repair was wanted.
        """
        if face_stage_uses_zimage(self.device):
            return super()._regenerate_face_crop(crop, index, total, strength=strength, seed=seed)

        import torch

        from pagedmark._internal.watermark_remover import empty_device_cache

        pipe = self._load_sdxl()
        floored = sdxl_face_strength(strength)
        steps = requested_steps(SDXL_FACE_STEPS, floored)
        self._progress(f"Repairing face {index}/{total} with SDXL: strength={floored:.4f}, steps={SDXL_FACE_STEPS}...")
        generator = torch.Generator(device=CPU_DEVICE).manual_seed(seed) if seed is not None else None
        result = pipe(
            prompt=self._global_prompt(),
            negative_prompt=self._global_negative(),
            image=crop,
            control_image=build_canny_control_image(crop),
            controlnet_conditioning_scale=float(self.controlnet_conditioning_scale),
            strength=float(floored),
            num_inference_steps=steps,
            guidance_scale=1.0,
            generator=generator,
        ).images[0]
        empty_device_cache(self.device)
        return result.convert("RGB")

    def _should_tile(self, image: Image.Image, tile: bool, tile_size: int) -> bool:
        """Tile when the caller asked, or when a native pass does not fit the device.

        The automatic half exists only where the budget is small and knowable: CUDA
        surfaces an OOM the caller can act on, while Metal answers an oversized
        allocation by paging, so the same run "succeeds" after an hour of swap. Tiling
        the input keeps native geometry, which is what the alternative
        (``--max-resolution``) gives up.
        """
        if super()._should_tile(image, tile, tile_size):
            return True
        if self.device != MPS_DEVICE or max(image.size) <= tile_size:
            return False
        if fits_in_one_pass(image.width, image.height, self._device_memory_gib()):
            return False
        self._progress(
            f"{image.width}x{image.height} exceeds this device's single-pass budget; "
            f"regenerating in {tile_size}px tiles at native resolution."
        )
        return True

    def _run_global(self, image: Image.Image, strength: float, seed: int | None) -> Image.Image:
        import torch

        pipe = self._load_sdxl()
        target = sdxl_target_size(image.width, image.height)
        prepared = image if image.size == target else image.resize(target, Image.Resampling.LANCZOS)
        if self.device == MPS_DEVICE:
            # Per frame, because the answer depends on the frame: a tile is what fits
            # a 16 GiB Mac and what a 64 GiB one does not need. Set on every call
            # rather than once, so a tiled run and a native run of different sizes on
            # one loaded pipeline each get their own answer.
            if vae_tiling_needed(*prepared.size, self._device_memory_gib()):
                pipe.vae.enable_tiling()
            else:
                pipe.vae.disable_tiling()
        control = build_canny_control_image(prepared)
        effective = sdxl_global_steps(self.device, GLOBAL_STEPS)
        steps = requested_steps(effective, strength)
        self._progress(f"Running SDXL Canny pass: strength={strength:.4f}, steps={effective} of {steps}...")
        # A Metal generator exists in current torch, but Diffusers reads the generator's
        # device to decide where the initial noise is drawn, and the two RNGs do not
        # produce the same stream. Drawing on the CPU is what keeps one seed meaning one
        # result across devices -- the property the fixed profile seed exists for.
        generator_device = CPU_DEVICE if self.device == MPS_DEVICE else self.device
        generator = torch.Generator(device=generator_device).manual_seed(seed) if seed is not None else None
        result = pipe(
            prompt=self._global_prompt(),
            negative_prompt=self._global_negative(),
            image=prepared,
            control_image=control,
            controlnet_conditioning_scale=float(self.controlnet_conditioning_scale),
            strength=float(strength),
            num_inference_steps=steps,
            guidance_scale=1.0,
            generator=generator,
        ).images[0]
        if self.device == MPS_DEVICE:
            # Per tile, not just per image: a tiled run holds every tile's activations
            # in Metal's cache otherwise, and the peak grows with the tile count
            # instead of staying at one tile.
            from pagedmark._internal.watermark_remover import empty_device_cache

            empty_device_cache(self.device)
        if result.size != image.size:
            result = result.resize(image.size, Image.Resampling.LANCZOS)
        return result.convert("RGB")

    @staticmethod
    def _global_prompt() -> str:
        from pagedmark._internal.qwen_zimage_pipeline import _GLOBAL_PROMPT

        return _GLOBAL_PROMPT

    @staticmethod
    def _global_negative() -> str:
        from pagedmark._internal.qwen_zimage_pipeline import _GLOBAL_NEGATIVE

        return _GLOBAL_NEGATIVE
