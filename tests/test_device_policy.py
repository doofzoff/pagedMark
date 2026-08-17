"""Tests for the profile/device policy that makes Apple Silicon a supported path.

Every assertion here is pure: no model loads, no GPU, no downloads. The policy is a
free function precisely so the decisions it encodes -- which profile a device gets,
which stage it can run, which extra installs it, and when a native pass has to become
a tiled one -- are testable on the CI hosts that have none of the hardware.
"""

from __future__ import annotations

import pytest

from pagedmark._internal.watermark_profiles import (
    CUDA_DEVICE,
    DEFAULT_PROFILE,
    MPS_DEVICE,
    PROFILE_CHOICES,
    QWEN_ZIMAGE_PROFILE,
    SDXL_ZIMAGE_PROFILE,
    default_profile_for_device,
    face_stage_runs_on,
    install_extra_for_device,
    plan_profile,
    profile_supports_device,
    profiles_for_device,
    required_modules,
)


class TestProfileDeviceSupport:
    """Which stack loads where. The split is a hardware fact, not a preference."""

    def test_the_float8_profile_is_cuda_only(self):
        """qwen-zimage streams float8 weights; Metal has no float8 tensor type."""
        assert profile_supports_device(QWEN_ZIMAGE_PROFILE, CUDA_DEVICE)
        assert not profile_supports_device(QWEN_ZIMAGE_PROFILE, MPS_DEVICE)

    def test_the_diffusers_profile_runs_on_both_gpus(self):
        assert profile_supports_device(SDXL_ZIMAGE_PROFILE, CUDA_DEVICE)
        assert profile_supports_device(SDXL_ZIMAGE_PROFILE, MPS_DEVICE)

    def test_no_profile_runs_on_the_cpu(self):
        """The CPU answer stays "none", so the refusal keeps naming a device."""
        assert profiles_for_device("cpu") == ()
        assert default_profile_for_device("cpu") is None

    def test_the_underscore_spelling_reaches_the_same_answer(self):
        """Support is asked with whatever the user typed, so it normalizes first."""
        assert profile_supports_device("sdxl_zimage", MPS_DEVICE)
        assert profile_supports_device("  SDXL_ZImage ", MPS_DEVICE)

    def test_every_profile_declares_at_least_one_device(self):
        """A profile no device can run is unreachable code with a CLI choice attached."""
        for profile in PROFILE_CHOICES:
            assert profiles_for_device(CUDA_DEVICE) or profiles_for_device(MPS_DEVICE)
            assert any(profile_supports_device(profile, device) for device in (CUDA_DEVICE, MPS_DEVICE))


class TestDefaultProfilePerDevice:
    """The default follows the device, and CUDA's must not move."""

    def test_cuda_keeps_the_certified_default(self):
        assert default_profile_for_device(CUDA_DEVICE) == DEFAULT_PROFILE == QWEN_ZIMAGE_PROFILE

    def test_mps_defaults_to_the_only_profile_it_can_load(self):
        assert default_profile_for_device(MPS_DEVICE) == SDXL_ZIMAGE_PROFILE


class TestFaceStagePolicy:
    """A face stage runs on both GPUs; only its model differs."""

    def test_a_face_stage_runs_on_both_gpus_but_zimage_only_on_cuda(self):
        from pagedmark._internal.watermark_profiles import face_stage_uses_zimage

        assert face_stage_runs_on(CUDA_DEVICE)
        assert face_stage_runs_on(MPS_DEVICE)
        assert not face_stage_runs_on("cpu")
        # Z-Image is DiffSynth with float8 storage, so Metal stands in the SDXL stack
        # it already has resident.
        assert face_stage_uses_zimage(CUDA_DEVICE)
        assert not face_stage_uses_zimage(MPS_DEVICE)

    def test_the_face_strength_floor_is_a_numerical_limit_not_a_preference(self):
        """Below it the fp16 crop path stops producing an image.

        Upstream's policy resolves to 0.025 for the measured photo; at that value three
        of four faces came back at ~12 dB PSNR against the source, while 0.05 produced
        30.5 dB. The clamp is what keeps a small-face image from silently returning
        garbage, so it is pinned rather than tuned.
        """
        from pagedmark._internal.watermark_profiles import (
            SDXL_FACE_STRENGTH_FLOOR,
            sdxl_face_strength,
        )

        assert SDXL_FACE_STRENGTH_FLOOR == 0.05
        assert sdxl_face_strength(0.025) == SDXL_FACE_STRENGTH_FLOOR
        assert sdxl_face_strength(0.0) == SDXL_FACE_STRENGTH_FLOOR
        # Above the floor the caller's value is respected, or the clamp would be a
        # constant with extra steps.
        assert sdxl_face_strength(0.12) == pytest.approx(0.12)

    def test_the_module_requirement_follows_the_stages_that_actually_run(self):
        assert "diffsynth" in required_modules(CUDA_DEVICE)
        assert "diffsynth" not in required_modules(MPS_DEVICE)
        assert "diffusers" in required_modules(MPS_DEVICE)
        assert "torch" in required_modules(MPS_DEVICE)

    def test_the_install_hint_is_the_extra_that_works_on_that_device(self):
        """qwen-zimage on a Mac downloads a stack Metal cannot load: a hint that lies."""
        assert install_extra_for_device(CUDA_DEVICE) == "'pagedmark[qwen-zimage]'"
        assert install_extra_for_device(MPS_DEVICE) == "'pagedmark[diffusion]'"

    def test_every_hint_is_shell_quoted(self):
        """Bare brackets are a zsh glob, so an unquoted hint dies before pip runs."""
        for device in (CUDA_DEVICE, MPS_DEVICE, "cpu"):
            hint = install_extra_for_device(device)
            assert hint.startswith("'")
            assert hint.endswith("'")


class TestPlanProfile:
    """One function resolves profile + device + face stage, and explains itself."""

    def test_cuda_with_no_preference_runs_the_full_recipe_silently(self):
        plan = plan_profile(None, CUDA_DEVICE)
        assert (plan.profile, plan.device, plan.face_stage) == (QWEN_ZIMAGE_PROFILE, CUDA_DEVICE, True)
        assert plan.note is None, "an unchanged CUDA run must not narrate"

    def test_mps_with_no_preference_selects_sdxl_and_says_so(self):
        plan = plan_profile(None, MPS_DEVICE)
        assert plan.profile == SDXL_ZIMAGE_PROFILE
        assert plan.face_stage is True
        assert plan.note is not None
        # Both facts a user needs: which profile ran, and which model repairs faces.
        assert QWEN_ZIMAGE_PROFILE in plan.note
        assert "face" in plan.note.casefold()
        assert "SDXL" in plan.note

    def test_an_explicit_runnable_profile_is_kept(self):
        plan = plan_profile(SDXL_ZIMAGE_PROFILE, MPS_DEVICE)
        assert plan.profile == SDXL_ZIMAGE_PROFILE
        # Still reports the substituted stage: the user chose the profile, not the
        # substitution, and a silent swap is what makes a face artifact a mystery.
        assert plan.note is not None
        assert "face" in plan.note.casefold()

    def test_an_explicit_cuda_only_profile_on_mps_raises_with_the_way_forward(self):
        with pytest.raises(ValueError, match="cannot run on mps") as excinfo:
            plan_profile(QWEN_ZIMAGE_PROFILE, MPS_DEVICE)
        message = str(excinfo.value)
        assert "float8" in message, "the refusal must name the reason, not just the verdict"
        assert SDXL_ZIMAGE_PROFILE in message, "and the profile that does run"

    def test_a_device_that_runs_nothing_raises_and_names_what_still_works(self):
        with pytest.raises(ValueError, match="needs a GPU") as excinfo:
            plan_profile(None, "cpu")
        message = str(excinfo.value)
        assert "cpu" in message
        assert "CUDA" in message
        assert "MPS" in message
        assert "identify" in message, "the CPU-only paths that still work belong in the refusal"

    def test_an_unknown_profile_name_is_rejected_before_the_device_question(self):
        with pytest.raises(ValueError, match="Unsupported pipeline"):
            plan_profile("controlnet", CUDA_DEVICE)

    def test_the_device_is_normalized_like_the_profile_is(self):
        assert plan_profile(None, "MPS").device == MPS_DEVICE


class TestSinglePassMemoryBudget:
    """When a native pass does not fit, tiling is the answer that keeps geometry."""

    @staticmethod
    def _fits(width: int, height: int, memory_gib: float) -> bool:
        from pagedmark._internal.sdxl_zimage_pipeline import fits_in_one_pass

        return fits_in_one_pass(width, height, memory_gib)

    def test_the_threshold_separates_the_two_measured_runs(self):
        """The constants are bracketed by measurement, so pin the bracket, not the value.

        On an M5 with an 11.84 GiB recommended working set, the shipped configuration
        ran 1448x1086 (1.57 MP) at a 10.92 GiB peak in 23 s, while 1824x1368 (2.5 MP)
        went into swap and did not finish in twelve minutes. A change to either constant
        that stops separating those two observations is a regression in the only
        evidence this policy has. Record: docs/module-internals.md, "Apple Silicon".
        """
        assert self._fits(1448, 1086, 11.84) is True
        assert self._fits(1824, 1368, 11.84) is False

    def test_a_bigger_budget_moves_the_boundary_rather_than_the_verdict_shape(self):
        """The same 2.5 MP input fits a machine that actually has the memory."""
        assert self._fits(1824, 1368, 24.0) is True

    def test_an_unreadable_budget_is_not_an_unlimited_one(self):
        """0.0 means "unknown", and unknown is exactly when tiling is the safe answer."""
        assert self._fits(512, 512, 0.0) is False

    def test_a_small_input_fits_a_16gib_mac(self):
        assert self._fits(1024, 1024, 11.8) is True

    def test_a_large_input_does_not(self):
        assert self._fits(4096, 4096, 11.8) is False

    def test_the_verdict_is_monotonic_in_both_directions(self):
        """More pixels never help, more memory never hurts."""
        assert self._fits(2048, 2048, 40.0) is True
        assert self._fits(2048, 2048, 4.0) is False
        assert self._fits(256, 256, 11.8) is True


class TestSamplingPolicy:
    """The four-step distillation runs where its operating point was measured."""

    @staticmethod
    def _enabled(device: str) -> bool:
        from pagedmark._internal.watermark_profiles import sdxl_lightning_enabled

        return sdxl_lightning_enabled(device)

    @staticmethod
    def _steps(device: str, distilled: int = 4) -> int:
        from pagedmark._internal.watermark_profiles import sdxl_global_steps

        return sdxl_global_steps(device, distilled)

    def test_cuda_keeps_the_distilled_schedule_it_was_certified_with(self):
        assert self._enabled(CUDA_DEVICE) is True
        assert self._steps(CUDA_DEVICE, 4) == 4

    def test_mps_runs_the_undistilled_schedule(self):
        """Measured, on the flat dark regions of a night photo at 4-16 px:

        the distilled LoRA invents 2.10x the source's luma energy there, because this
        profile runs it on the TAIL of a long schedule rather than the full range it
        was distilled for. Asking it for more steps makes that worse (8 -> 1.80x,
        16 -> 1.84x of the chroma measure), which is what identifies the distillation
        rather than the step count. The base model at 16 steps reads 1.19x and gains
        fidelity (PSNR 29.25 against 28.54).
        """
        assert self._enabled(MPS_DEVICE) is False
        assert self._steps(MPS_DEVICE, 4) == 16

    def test_the_undistilled_step_count_is_the_measured_knee(self):
        """24 steps measured 1.20x/PSNR 29.17 against 16 steps' 1.19x/29.25, for twice
        the wall time -- so 16 is the knee, not a round number."""
        from pagedmark._internal.watermark_profiles import SDXL_UNDISTILLED_GLOBAL_STEPS

        assert SDXL_UNDISTILLED_GLOBAL_STEPS == 16

    def test_the_step_count_is_not_read_from_the_distilled_constant_on_mps(self):
        """A profile that changes its own GLOBAL_STEPS must not move the MPS answer."""
        assert self._steps(MPS_DEVICE, 8) == self._steps(MPS_DEVICE, 4)


class TestResidencyPolicy:
    """Where the weights live, decided from the device budget rather than from hope."""

    @staticmethod
    def _plan(memory_gib: float):
        from pagedmark._internal.watermark_profiles import sdxl_residency_plan

        return sdxl_residency_plan(memory_gib)

    def test_a_16gib_mac_keeps_the_weights_resident(self):
        """11.84 GiB is the measured working set there, against a 7.7 GiB stack."""
        plan = self._plan(11.84)
        assert plan.offload is False
        assert plan.note is None, "the fast path must not narrate"

    def test_an_8gib_mac_streams_them_and_says_why(self):
        """~5.3 GiB cannot hold the stack, and the run is three times slower for it.

        Silence here is the worst outcome: the user waits through a run that looks
        broken. Measured on an M5: 0.28 GiB peak offloaded against 7.70 GiB resident,
        24.1 s against 7.1 s on the same frame.
        """
        plan = self._plan(5.3)
        assert plan.offload is True
        assert plan.note is not None
        assert "three times" in plan.note

    def test_an_unreadable_budget_takes_the_path_that_runs_anywhere(self):
        plan = self._plan(0.0)
        assert plan.offload is True
        assert plan.note is not None

    def test_a_budget_too_small_for_either_plan_is_refused_with_what_still_works(self):
        with pytest.raises(ValueError, match="smallest workable") as excinfo:
            self._plan(1.0)
        message = str(excinfo.value)
        assert "1.0 GiB" in message
        assert "identify" in message, "the refusal names the commands that still run"

    def test_the_text_encoders_are_released_only_where_that_was_measured(self):
        from pagedmark._internal.watermark_profiles import sdxl_releases_text_encoders

        assert sdxl_releases_text_encoders(MPS_DEVICE) is True
        assert sdxl_releases_text_encoders(CUDA_DEVICE) is False


class TestVaeTilingPolicy:
    """Tiling the VAE decode is a memory trade, so it is made where it pays."""

    @staticmethod
    def _needed(width: int, height: int, memory_gib: float) -> bool:
        from pagedmark._internal.sdxl_zimage_pipeline import vae_tiling_needed

        return vae_tiling_needed(width, height, memory_gib)

    def test_a_16gib_mac_tiles_even_a_small_frame(self):
        """An untiled 1.57 MP decode measured 18.74 GiB against an 11.84 GiB budget."""
        assert self._needed(1448, 1086, 11.84) is True
        assert self._needed(724, 542, 11.84) is True

    def test_a_large_budget_decodes_a_normal_frame_whole(self):
        """The tile texture is what the CJK detector read as a 0.38 mark, so a machine
        with the memory should not pay for it."""
        assert self._needed(1448, 1086, 48.0) is False

    def test_a_large_budget_still_tiles_a_large_enough_frame(self):
        assert self._needed(6000, 4000, 48.0) is True

    def test_an_unreadable_budget_takes_the_bounded_path(self):
        assert self._needed(512, 512, 0.0) is True


class TestPipelineDeviceGuards:
    """The pipelines refuse a device their stack cannot load, without loading it."""

    def test_the_qwen_pipeline_declares_cuda_only(self):
        from pagedmark._internal.qwen_zimage_pipeline import QwenZImagePipeline

        assert QwenZImagePipeline.supported_devices == (CUDA_DEVICE,)

    def test_the_sdxl_pipeline_declares_both_gpus(self):
        from pagedmark._internal.sdxl_zimage_pipeline import SdxlZImagePipeline

        assert set(SdxlZImagePipeline.supported_devices) == {CUDA_DEVICE, MPS_DEVICE}

    def test_a_directly_constructed_pipeline_still_refuses_an_unsupported_device(self):
        """The remover's gate is in front of this one, not instead of it."""
        from pagedmark._internal.qwen_zimage_pipeline import QwenZImagePipeline

        pipeline = QwenZImagePipeline(device=MPS_DEVICE, torch_dtype=None)
        with pytest.raises(RuntimeError, match="cannot run on 'mps'"):
            pipeline._require_supported_device()

    def test_the_sdxl_pipeline_tiles_an_input_that_does_not_fit_the_device(self, monkeypatch):
        """Automatic only on MPS, where an oversized allocation pages instead of failing."""
        from PIL import Image

        from pagedmark._internal.sdxl_zimage_pipeline import SdxlZImagePipeline

        pipeline = SdxlZImagePipeline(device=MPS_DEVICE, torch_dtype=None)
        monkeypatch.setattr(pipeline, "_device_memory_gib", lambda: 11.8)

        assert pipeline._should_tile(Image.new("RGB", (4096, 4096)), tile=False, tile_size=1024) is True
        assert pipeline._should_tile(Image.new("RGB", (1200, 900)), tile=False, tile_size=1024) is False
        # Below the tile size there is nothing to tile, whatever the budget says.
        assert pipeline._should_tile(Image.new("RGB", (512, 512)), tile=False, tile_size=1024) is False

    def test_cuda_never_tiles_unless_the_caller_asked(self, monkeypatch):
        """CUDA raises a legible OOM; silently changing its execution path would hide it."""
        from PIL import Image

        from pagedmark._internal.sdxl_zimage_pipeline import SdxlZImagePipeline

        pipeline = SdxlZImagePipeline(device=CUDA_DEVICE, torch_dtype=None)
        monkeypatch.setattr(pipeline, "_device_memory_gib", lambda: 8.0)

        assert pipeline._should_tile(Image.new("RGB", (4096, 4096)), tile=False, tile_size=1024) is False
        assert pipeline._should_tile(Image.new("RGB", (4096, 4096)), tile=True, tile_size=1024) is True


class TestSamplingPolicyReachesTheLoader:
    """The policy has to change what is LOADED, not just what a function returns."""

    @staticmethod
    def _load_with_recorder(monkeypatch, device: str, budget_gib: float = 11.84):
        """Build the SDXL stage against stubbed loaders and report what it did."""
        from unittest.mock import MagicMock

        import diffusers

        from pagedmark._internal import sdxl_zimage_pipeline as module

        calls: dict[str, object] = {"lora": 0}
        pipe = MagicMock()
        pipe.to.return_value = pipe
        pipe.load_lora_weights.side_effect = lambda *a, **k: calls.__setitem__("lora", calls["lora"] + 1)
        # The loader encodes the fixed prompts before dropping the encoders, so the fake
        # has to answer that call with the four values SDXL's encoder returns.
        pipe.encode_prompt.return_value = (MagicMock(), None, MagicMock(), None)

        monkeypatch.setattr(diffusers.ControlNetModel, "from_pretrained", staticmethod(lambda *a, **k: MagicMock()))
        monkeypatch.setattr(diffusers.AutoencoderKL, "from_pretrained", staticmethod(lambda *a, **k: MagicMock()))
        monkeypatch.setattr(
            diffusers.StableDiffusionXLControlNetImg2ImgPipeline,
            "from_pretrained",
            staticmethod(lambda *a, **k: pipe),
        )
        # The CUDA branch rebuilds the scheduler from the pipeline's config, which is a
        # mock here; without this it would try to resolve that mock against the Hub.
        monkeypatch.setattr(diffusers.EulerDiscreteScheduler, "from_config", staticmethod(lambda *a, **k: MagicMock()))
        monkeypatch.setattr(module, "hf_hub_download", lambda *a, **k: "lora.safetensors", raising=False)
        monkeypatch.setattr(
            "huggingface_hub.hf_hub_download", staticmethod(lambda *a, **k: "lora.safetensors"), raising=False
        )
        pipeline = module.SdxlZImagePipeline(device=device, torch_dtype=None)
        # A fixed budget, so the plan under test does not depend on the host's memory.
        monkeypatch.setattr(pipeline, "_device_memory_gib", lambda: budget_gib)
        pipeline._load_sdxl()
        return calls, pipe

    def test_mps_never_fuses_the_distillation_lora(self, monkeypatch):
        pytest.importorskip("diffusers")
        calls, pipe = self._load_with_recorder(monkeypatch, MPS_DEVICE)
        assert calls["lora"] == 0
        pipe.fuse_lora.assert_not_called()

    def test_a_small_budget_installs_the_offload_hooks(self, monkeypatch):
        """The plan has to change what the loader DOES, not only what it returns."""
        pytest.importorskip("diffusers")
        _calls, pipe = self._load_with_recorder(monkeypatch, MPS_DEVICE, budget_gib=5.3)

        pipe.enable_sequential_cpu_offload.assert_called_once()
        assert pipe.enable_sequential_cpu_offload.call_args.kwargs == {"device": MPS_DEVICE}

    def test_a_large_budget_does_not(self, monkeypatch):
        pytest.importorskip("diffusers")
        _calls, pipe = self._load_with_recorder(monkeypatch, MPS_DEVICE, budget_gib=11.84)

        pipe.enable_sequential_cpu_offload.assert_not_called()

    def test_mps_encodes_the_prompts_once_and_drops_the_encoders(self, monkeypatch):
        pytest.importorskip("diffusers")
        _calls, pipe = self._load_with_recorder(monkeypatch, MPS_DEVICE)

        pipe.encode_prompt.assert_called_once()
        assert pipe.text_encoder is None
        assert pipe.text_encoder_2 is None

    def test_cuda_still_fuses_it(self, monkeypatch):
        """The discriminating half: without this, a loader that never fuses passes."""
        pytest.importorskip("diffusers")
        calls, pipe = self._load_with_recorder(monkeypatch, CUDA_DEVICE)
        assert calls["lora"] == 1
        pipe.fuse_lora.assert_called_once()


class TestSdxlFaceStageReachesThePipeline:
    """The substituted stage has to be the one that RUNS, not just the one declared."""

    @staticmethod
    def _pipeline(device: str, monkeypatch):
        from PIL import Image

        from pagedmark._internal import qwen_zimage_pipeline as base
        from pagedmark._internal.sdxl_zimage_pipeline import SdxlZImagePipeline

        pipeline = SdxlZImagePipeline(device=device, torch_dtype=None)
        calls: dict[str, int] = {"sdxl": 0, "zimage": 0}

        class FakeSdxl:
            def __call__(self, **kwargs):
                calls["sdxl"] += 1
                calls["strength"] = kwargs["strength"]
                return type("Out", (), {"images": [Image.new("RGB", kwargs["image"].size)]})()

        monkeypatch.setattr(pipeline, "_load_sdxl", lambda: FakeSdxl())
        monkeypatch.setattr(
            base.QwenZImagePipeline,
            "_regenerate_face_crop",
            lambda self, crop, index, total, **kw: (
                calls.__setitem__("zimage", calls["zimage"] + 1) or Image.new("RGB", crop.size)
            ),
        )
        return pipeline, calls

    def test_mps_repairs_faces_with_sdxl_at_the_floored_strength(self, monkeypatch):
        pytest.importorskip("diffusers")
        from PIL import Image

        pipeline, calls = self._pipeline(MPS_DEVICE, monkeypatch)
        pipeline._regenerate_face_crop(Image.new("RGB", (256, 256)), 1, 1, strength=0.025, seed=0)

        assert calls["sdxl"] == 1
        assert calls["zimage"] == 0
        assert calls["strength"] == 0.05, "the floor has to reach the sampler, not just the policy"

    def test_cuda_still_repairs_faces_with_zimage(self, monkeypatch):
        """The discriminating half: without it, a stage that never calls Z-Image passes."""
        pytest.importorskip("diffusers")
        from PIL import Image

        pipeline, calls = self._pipeline(CUDA_DEVICE, monkeypatch)
        pipeline._regenerate_face_crop(Image.new("RGB", (256, 256)), 1, 1, strength=0.025, seed=0)

        assert calls["zimage"] == 1
        assert calls["sdxl"] == 0

    def test_the_crop_cap_applies_only_where_the_sdxl_stage_runs(self, monkeypatch):
        pytest.importorskip("diffusers")
        from pagedmark._internal.sdxl_zimage_pipeline import SdxlZImagePipeline
        from pagedmark._internal.watermark_profiles import SDXL_FACE_CROP_CAP

        assert SdxlZImagePipeline(device=MPS_DEVICE, torch_dtype=None)._face_crop_cap() == SDXL_FACE_CROP_CAP
        assert SdxlZImagePipeline(device=CUDA_DEVICE, torch_dtype=None)._face_crop_cap() is None


class TestFaceStageFlagReachesThePipeline:
    """A skipped stage must be skipped by the pipeline, not just by the plan."""

    def test_the_run_skips_detection_when_the_face_stage_is_off(self, monkeypatch):
        """Detection itself is skipped: YuNet would download a model to report faces
        this device will not touch, and a detection line reads as a stage that ran."""
        from PIL import Image

        from pagedmark._internal import qwen_zimage_pipeline as module

        detect_calls: list[object] = []
        monkeypatch.setattr(module, "detect_faces", lambda image: detect_calls.append(image) or [(0, 0, 8, 8)])

        pipeline = module.QwenZImagePipeline(device=CUDA_DEVICE, torch_dtype=None, face_stage=False)
        global_result = Image.new("RGB", (32, 32), (10, 20, 30))
        monkeypatch.setattr(pipeline, "_run_global", lambda *args, **kwargs: global_result)

        result = pipeline.run(Image.new("RGB", (32, 32)), strength=0.15, seed=0)

        assert result is global_result
        assert detect_calls == [], "the face detector must not run when its repair cannot"

    def test_the_run_still_detects_and_repairs_when_the_face_stage_is_on(self, monkeypatch):
        """The discriminating half: the skip above must come from the flag, not from
        a detector that never fires in this harness."""
        from PIL import Image

        from pagedmark._internal import qwen_zimage_pipeline as module

        repaired = Image.new("RGB", (32, 32), (99, 99, 99))
        monkeypatch.setattr(module, "detect_faces", lambda image: [(0, 0, 8, 8)])

        pipeline = module.QwenZImagePipeline(device=CUDA_DEVICE, torch_dtype=None, face_stage=True)
        monkeypatch.setattr(pipeline, "_run_global", lambda *args, **kwargs: Image.new("RGB", (32, 32)))
        monkeypatch.setattr(pipeline, "_sam_masks", lambda *args, **kwargs: [None])
        monkeypatch.setattr(pipeline, "_run_faces", lambda *args, **kwargs: repaired)

        assert pipeline.run(Image.new("RGB", (32, 32)), strength=0.15, seed=0) is repaired
