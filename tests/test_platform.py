"""Tests for device detection, profile resolution, and platform-specific paths.

Invisible-watermark removal needs a GPU: CUDA runs both profiles, MPS runs the one
whose stack Metal can load, and every other device is refused. The device tests here
assert that ladder and the refusal at its end, not a silent fallback.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pagedmark._internal.utils import get_image_format, is_supported_format
from pagedmark._internal.watermark_profiles import (
    PROFILE_CHOICES,
    REMOVAL_MODULES,
    SDXL_ZIMAGE_GEMINI_STRENGTH,
    SDXL_ZIMAGE_OPENAI_STRENGTH,
    SDXL_ZIMAGE_UNKNOWN_STRENGTH,
    normalize_profile,
    required_modules,
    resolve_strength,
    strength_default_help,
)
from pagedmark._internal.watermark_remover import get_device, is_watermark_removal_available

# ── Device detection ────────────────────────────────────────────────


class TestDeviceDetection:
    """get_device() answers with a device a profile can run, or the "cpu" that names none."""

    def test_answer_is_a_supported_device_or_cpu(self):
        """No xpu answer exists: nothing here runs on it, so it would only travel one
        frame to the same refusal while costing a probe. MPS does exist, because
        sdxl-zimage's global stage runs there.
        """
        assert get_device() in ("cpu", "cuda", "mps")

    @patch("pagedmark._internal.watermark_remover._HAS_TORCH", False)
    def test_no_torch_returns_cpu(self):
        assert get_device() == "cpu"

    def test_working_cuda_is_selected_and_probed(self):
        """A reported CUDA device is smoke-tested before it is returned.

        torch.cuda.is_available() can be True on a build whose CUDA backend then
        raises on the first real op; without the probe that surfaced much later.
        """
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        with patch("pagedmark._internal.watermark_remover.torch", fake_torch):
            assert get_device() == "cuda"
        fake_torch.tensor.assert_called_with([1.0], device="cuda")

    def test_broken_cuda_backend_falls_back_to_cpu(self):
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.tensor.side_effect = RuntimeError("no kernel image")
        with patch("pagedmark._internal.watermark_remover.torch", fake_torch):
            assert get_device() == "cpu"

    def test_cuda_is_preferred_over_a_working_mps_backend(self):
        """A machine reporting both takes CUDA: it is where the profiles were certified."""
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.backends.mps.is_available.return_value = True
        with patch("pagedmark._internal.watermark_remover.torch", fake_torch):
            assert get_device() == "cuda"

    def test_mps_is_selected_and_probed_when_cuda_is_absent(self):
        """Apple Silicon is a real answer now, and gets the same op probe CUDA gets."""
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        fake_torch.backends.mps.is_available.return_value = True
        with (
            patch("pagedmark._internal.watermark_remover.torch", fake_torch),
            patch("pagedmark._internal.watermark_remover._has_nvidia_gpu", return_value=False),
        ):
            assert get_device() == "mps"
        fake_torch.tensor.assert_called_with([1.0], device="mps")

    def test_a_broken_mps_backend_falls_back_to_cpu(self):
        """``is_available()`` is a build flag; the probe is what proves a usable backend."""
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        fake_torch.backends.mps.is_available.return_value = True
        fake_torch.tensor.side_effect = RuntimeError("MPS backend out of memory")
        with (
            patch("pagedmark._internal.watermark_remover.torch", fake_torch),
            patch("pagedmark._internal.watermark_remover._has_nvidia_gpu", return_value=False),
        ):
            assert get_device() == "cpu"

    def test_devices_that_run_nothing_are_refused_at_construction(self):
        """Device support is a precondition of the object, not of the run.

        Accepting a device no profile can load only defers a guaranteed failure to
        model-load time - several layers down, after the dependency check and the
        pipeline import, under a message naming whichever profile the internal
        pipeline happens to be.

        Deliberately NOT gated on the diffusion stack. It used to be, and since no CI
        job installs diffusers or diffsynth (the dev extra pulls torch only, via
        invisible-watermark) the guard skipped in every environment it ran in --
        including the maintainer's. The refusal fires before any torch attribute is
        touched, so faking the dependency probe is enough to reach it.
        """
        from pagedmark._internal import watermark_remover as module

        with patch.object(module, "is_watermark_removal_available", return_value=True):
            for device in ("cpu", "xpu"):
                with pytest.raises(ValueError, match="needs a GPU"):
                    module.WatermarkRemover(device=device)

    def test_the_cuda_only_profile_is_refused_on_mps_by_name(self):
        """Naming qwen-zimage on a Mac must say why, not fail inside DiffSynth.

        The auto-selected path is the other test; this is the user who typed it. The
        two are only distinguishable because ``pipeline`` defaults to None.
        """
        from pagedmark._internal import watermark_remover as module

        with (
            patch.object(module, "is_watermark_removal_available", return_value=True),
            pytest.raises(ValueError, match="cannot run on mps"),
        ):
            module.WatermarkRemover(device="mps", pipeline="qwen-zimage")

    def test_the_refusal_names_the_resolved_device_not_a_bare_none(self):
        """``device=None`` on a GPU-less host must report "cpu", not "None".

        The message used to interpolate the raw argument, so the common auto-detect
        path told the user that ``'None'`` cannot run the removal.
        """
        from pagedmark._internal import watermark_remover as module

        with (
            patch.object(module, "is_watermark_removal_available", return_value=True),
            patch.object(module, "get_device", return_value="cpu"),
            pytest.raises(ValueError, match="'cpu' cannot run it"),
        ):
            module.WatermarkRemover(device=None)

    def test_a_cuda_remover_picks_the_profile_dtype(self):
        """The dtype half still needs real torch, so it keeps its skip."""
        if not is_watermark_removal_available("cuda"):
            pytest.skip("the qwen-zimage extra is not installed")
        import torch

        from pagedmark._internal.watermark_remover import WatermarkRemover

        remover = WatermarkRemover(device="cuda")
        assert remover.device == "cuda"
        assert remover.torch_dtype == torch.bfloat16
        assert WatermarkRemover(device="cuda", pipeline="sdxl-zimage").torch_dtype == torch.float16

    def test_an_mps_remover_selects_the_runnable_profile_in_its_dtype(self):
        """The Apple path resolves to sdxl-zimage in fp16, with a face stage of its own.

        Constructed for real rather than asserted through the policy function: the
        dtype and the face-stage flag are what the pipeline is handed, and both have
        drifted from their policy before (see the face-stage dtype incident).
        """
        if not is_watermark_removal_available("mps"):
            pytest.skip("the diffusion extra is not installed")
        import torch

        from pagedmark._internal.watermark_remover import WatermarkRemover

        remover = WatermarkRemover(device="mps")
        assert remover.device == "mps"
        assert remover.model_profile == "sdxl-zimage"
        assert remover.torch_dtype == torch.float16
        # True, but served by the SDXL crops rather than Z-Image, which is CUDA-only.
        assert remover.face_stage is True


class TestModelProfiles:
    """Only the two CUDA-only two-stage profiles remain."""

    def test_canonical_profiles_unchanged(self):
        assert normalize_profile("qwen-zimage") == "qwen-zimage"
        assert normalize_profile("sdxl-zimage") == "sdxl-zimage"

    def test_underscore_spellings_resolve(self):
        assert normalize_profile("qwen_zimage") == "qwen-zimage"
        assert normalize_profile("  SDXL_ZImage ") == "sdxl-zimage"

    def test_retired_names_no_longer_resolve_to_a_profile(self):
        """default/sdxl/controlnet/qwen were removed, not aliased onward.

        Silently mapping them at the alias layer would route an old script into a
        profile it never asked for; the remover raises on the unknown name instead.
        """
        for retired in ("default", "sdxl", "controlnet", "qwen"):
            assert normalize_profile(retired) not in PROFILE_CHOICES


class TestResolveAdaptivePolish:
    """The polish default is per-profile data, not a CLI parameter-source inference."""

    def test_unset_follows_the_profile(self):
        from pagedmark._internal.watermark_profiles import resolve_adaptive_polish

        # qwen-zimage already matches the input's detail level, so polishing it only
        # moves the output away from upstream. An SDXL global pass leaves the softer
        # result the polish exists for.
        assert resolve_adaptive_polish(None, "qwen-zimage") is False
        assert resolve_adaptive_polish(None, "sdxl-zimage") is True
        assert resolve_adaptive_polish(None, "qwen_zimage") is False

    def test_an_explicit_choice_always_wins(self):
        from pagedmark._internal.watermark_profiles import resolve_adaptive_polish

        assert resolve_adaptive_polish(True, "qwen-zimage") is True
        assert resolve_adaptive_polish(False, "sdxl-zimage") is False


class TestNoReembeddedWatermark:
    """F2 regression: the SDXL global stage must disable the diffusers watermarker.

    diffusers stamps an open "Stable Diffusion XL" DWT-DCT watermark onto every SDXL
    output whenever ``invisible-watermark`` is installed. A watermark REMOVER that left
    it on would replace one detectable AI watermark (SynthID) with another -- the
    cleaned output re-reads as AI. The ControlNet sub-model load must NOT receive the
    kwarg, since it is not a pipeline and does not accept it.

    Only sdxl-zimage carries an SDXL pipeline now; qwen-zimage's global stage is
    DiffSynth, which has no such watermarker.
    """

    def test_sdxl_global_stage_disables_watermarker(self, monkeypatch: pytest.MonkeyPatch):
        if not is_watermark_removal_available():
            pytest.skip("the qwen-zimage extra is not installed")
        import diffusers

        from pagedmark._internal.sdxl_zimage_pipeline import SdxlZImagePipeline

        calls: dict[str, dict] = {}

        def record(name):
            def fake(*_args, **kwargs):
                calls[name] = kwargs
                return MagicMock()

            return fake

        monkeypatch.setattr(diffusers.ControlNetModel, "from_pretrained", record("controlnet"))
        monkeypatch.setattr(diffusers.AutoencoderKL, "from_pretrained", record("vae"))
        monkeypatch.setattr(diffusers.StableDiffusionXLControlNetImg2ImgPipeline, "from_pretrained", record("pipeline"))
        monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: "lora.safetensors")
        # from_config would otherwise resolve the mock's config as a repo id.
        monkeypatch.setattr(diffusers.EulerDiscreteScheduler, "from_config", lambda *a, **k: MagicMock())

        pipeline = SdxlZImagePipeline(device="cuda", torch_dtype=None)
        monkeypatch.setattr(type(pipeline), "_require_cuda", lambda self: None)
        pipeline._load_sdxl()

        assert calls["pipeline"].get("add_watermarker") is False
        assert "add_watermarker" not in calls["controlnet"]


class TestResolveStrength:
    """resolve_strength answers for sdxl-zimage and defers for qwen-zimage."""

    def test_qwen_zimage_answers_from_the_resolution_curve(self):
        """The function is total: it owns both policies rather than returning None.

        qwen-zimage picks strength from image area, so it takes the size. Returning
        None for it would push that branch onto every caller and leave one of the two
        strength policies living outside this module. The vendor is ignored here on
        purpose - the curve, not the issuer, is what was calibrated.
        """
        assert resolve_strength(None, "google", "qwen-zimage", size=(2000, 1850)) == pytest.approx(0.154)
        assert resolve_strength(None, None, "qwen-zimage", size=(600, 500)) == pytest.approx(0.084)

    def test_qwen_zimage_without_a_size_fails_loudly(self):
        """A missing size must not silently fall back to some vendor value."""
        with pytest.raises(ValueError, match="size is required"):
            resolve_strength(None, "google", "qwen-zimage")

    def test_sdxl_zimage_uses_its_flat_vendor_ladder(self):

        assert SDXL_ZIMAGE_OPENAI_STRENGTH == 0.15
        assert SDXL_ZIMAGE_GEMINI_STRENGTH == 0.25
        assert SDXL_ZIMAGE_UNKNOWN_STRENGTH == SDXL_ZIMAGE_GEMINI_STRENGTH
        assert resolve_strength(None, "openai", "sdxl-zimage") == SDXL_ZIMAGE_OPENAI_STRENGTH
        assert resolve_strength(None, "google", "sdxl-zimage") == SDXL_ZIMAGE_GEMINI_STRENGTH
        # An unrecognised issuer takes the stricter Gemini value, not the OpenAI one.
        assert resolve_strength(None, "adobe", "sdxl-zimage") == SDXL_ZIMAGE_UNKNOWN_STRENGTH
        assert resolve_strength(None, None, "sdxl-zimage") == SDXL_ZIMAGE_UNKNOWN_STRENGTH

    def test_strength_default_help_derives_from_constants(self):

        h = strength_default_help()
        assert str(SDXL_ZIMAGE_OPENAI_STRENGTH) in h
        assert str(SDXL_ZIMAGE_GEMINI_STRENGTH) in h

    def test_explicit_value_overrides_vendor(self):

        assert resolve_strength(0.3, "openai", "sdxl-zimage") == 0.3
        assert resolve_strength(0.3, None, "qwen-zimage") == 0.3

    def test_explicit_zero_is_respected_not_treated_as_unset(self):
        # 0.0 is falsy but explicit -- it must not fall through to the vendor default
        # (the old `strength or DEFAULT` bug would have). Range validation lives in
        # remove_watermark, not here.

        assert resolve_strength(0.0, "google", "sdxl-zimage") == 0.0
        assert resolve_strength(0.0, None, "qwen-zimage") == 0.0


class TestVendorForStrength:
    """vendor_for_strength normalizes SynthID provenance to openai/google/None."""

    @staticmethod
    def _patch(value):
        return patch("pagedmark.metadata.synthid_source", return_value=value)

    def test_openai(self):
        from pagedmark._internal.watermark_profiles import vendor_for_strength

        with self._patch("OpenAI"):
            assert vendor_for_strength(Path("x.png")) == "openai"

    def test_google(self):
        from pagedmark._internal.watermark_profiles import vendor_for_strength

        with self._patch("Google"):
            assert vendor_for_strength(Path("x.png")) == "google"

    def test_both_issuers_google_wins(self):
        # The more-robust watermark wins -> safer (higher) strength.
        from pagedmark._internal.watermark_profiles import vendor_for_strength

        with self._patch("OpenAI, Google"):
            assert vendor_for_strength(Path("x.png")) == "google"

    def test_none_when_no_synthid_source(self):
        from pagedmark._internal.watermark_profiles import vendor_for_strength

        with self._patch(None):
            assert vendor_for_strength(Path("x.png")) is None

    def test_unreadable_metadata_is_none(self):
        from pagedmark._internal.watermark_profiles import vendor_for_strength

        with patch("pagedmark.metadata.synthid_source", side_effect=OSError):
            assert vendor_for_strength(Path("x.png")) is None


# ── Format utilities ────────────────────────────────────────────────


class TestFormatUtils:
    """Tests for utils.py format helpers."""

    def test_supported_png(self, tmp_path):
        assert is_supported_format(tmp_path / "test.png")

    def test_supported_jpg(self, tmp_path):
        assert is_supported_format(tmp_path / "test.jpg")

    def test_supported_jpeg(self, tmp_path):
        assert is_supported_format(tmp_path / "test.jpeg")

    def test_supported_webp(self, tmp_path):
        assert is_supported_format(tmp_path / "test.webp")

    def test_unsupported_bmp(self, tmp_path):
        assert not is_supported_format(tmp_path / "test.bmp")

    def test_unsupported_gif(self, tmp_path):
        assert not is_supported_format(tmp_path / "test.gif")

    def test_get_format_png(self, tmp_path):
        assert get_image_format(tmp_path / "x.png") == "PNG"

    def test_get_format_jpg(self, tmp_path):
        assert get_image_format(tmp_path / "x.jpg") == "JPEG"

    def test_get_format_jpeg(self, tmp_path):
        assert get_image_format(tmp_path / "x.jpeg") == "JPEG"

    def test_get_format_webp_defaults_png(self, tmp_path):
        # .webp falls through to PNG in current implementation
        assert get_image_format(tmp_path / "x.webp") == "PNG"


# ── Availability checks ────────────────────────────────────────────


class TestAvailability:
    """The CLI gate and the remover precondition must answer from the same module list.

    Both used to hardcode (torch, diffusers) while the code moved to REMOVAL_MODULES,
    which includes diffsynth. In a torch+diffusers-only environment the assertions were
    then simply wrong -- and, worse, comparing each gate against a tuple copied from
    itself can never catch the two disagreeing, which is the drift that let the CLI pass
    an environment the run then died in.
    """

    @pytest.mark.parametrize("device", ["cuda", "mps"])
    def test_both_gates_agree_and_read_the_shared_module_list(self, device: str):
        import importlib.util

        from pagedmark.invisible_engine import is_available

        modules = required_modules(device)
        expected = all(importlib.util.find_spec(m) is not None for m in modules)
        assert is_watermark_removal_available(device) is expected
        assert is_available(device) is expected

    def test_the_cuda_list_carries_the_face_stage_runtime_and_the_mps_list_does_not(self):
        """The per-device split is the point, so pin both halves of it.

        diffsynth is what a CUDA run's face stage needs and what an MPS run can never
        use. Requiring it on MPS refuses a working run and prints an extra that does
        not install there; dropping it on CUDA re-lands the original drift, where the
        gate passed and the run then died at the face stage.
        """
        assert "diffsynth" in required_modules("cuda")
        assert "diffsynth" not in required_modules("mps")
        assert set(required_modules("mps")) <= set(required_modules("cuda"))
        assert "diffsynth" in REMOVAL_MODULES  # the CUDA answer stays the module default

    @pytest.mark.parametrize("device", ["cuda", "mps"])
    def test_a_missing_module_closes_both_gates(self, monkeypatch: pytest.MonkeyPatch, device: str):
        """Discriminating, not vacuous: it must FAIL if either gate stops requiring one.

        Comparing the live answer to a tuple derived from the same constant passes on
        any host -- with the full stack (True == True) and with none of it
        (False == False). Simulate each module's absence instead, per device, because
        the requirement itself is per-device now.
        """
        import pagedmark.invisible_engine as engine_module
        from pagedmark._internal import watermark_remover as remover_module

        for missing in required_modules(device):
            present = {name: name != missing for name in REMOVAL_MODULES}
            monkeypatch.setattr(
                "pagedmark.optional_deps.module_available",
                lambda *names, _p=present: all(_p.get(n, True) for n in names),
            )
            # The remover probes torch and the CUDA list at import time, so drive its
            # cached flags directly.
            monkeypatch.setattr(remover_module, "_HAS_TORCH", missing != "torch")
            monkeypatch.setattr(remover_module, "_HAS_REMOVAL_MODULES", missing == "torch")
            assert engine_module.is_available(device) is False, f"engine gate ignores a missing {missing} on {device}"
            assert remover_module.is_watermark_removal_available(device) is False, (
                f"remover gate ignores a missing {missing} on {device}"
            )


# ── Platform-specific path handling ─────────────────────────────────


class TestPlatformPaths:
    """Verify path handling works on current platform."""

    def test_pathlib_works_for_assets(self):
        from pathlib import Path

        asset_dir = Path(__file__).parent.parent / "src" / "pagedmark" / "assets"
        assert (asset_dir / "gemini_bg_48.png").exists()
        assert (asset_dir / "gemini_bg_96.png").exists()

    def test_asset_loading_works(self):
        """Verify embedded assets load correctly (critical for packaging)."""
        from pagedmark.gemini_engine import GeminiEngine

        engine = GeminiEngine()
        # If we get here without error, asset loading works
        assert engine._alpha_small.shape == (48, 48)
        assert engine._alpha_large.shape == (96, 96)
