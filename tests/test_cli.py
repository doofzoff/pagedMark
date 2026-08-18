"""Tests for the CLI entry point."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

import cv2
import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from pagedmark.cli import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def sample_png(tmp_path: Path) -> Path:
    """Create a sample PNG for CLI testing."""
    # Seeded: an unseeded random corner can occasionally trip the Doubao
    # visible-mark detector, making `visible --mark auto` flaky.
    img = np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8)
    path = tmp_path / "input.png"
    cv2.imwrite(str(path), img)
    return path


def _make_batch_dir(tmp_path: Path, count: int = 3) -> Path:
    """Create a directory with test images for batch testing."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(count):
        img = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(input_dir / f"img_{i}.png"), img)
    return input_dir


def _make_batch_dir_with_metadata(tmp_path: Path, count: int = 3) -> Path:
    """Create a directory with PNG images containing AI metadata."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for i in range(count):
        img = Image.new("RGB", (64, 64), color=(100 + i, 150, 200))
        pnginfo = PngInfo()
        pnginfo.add_text("parameters", f"Steps: 20, Sampler: Euler, img_{i}")
        pnginfo.add_text("prompt", "a test landscape")
        img.save(input_dir / f"img_{i}.png", pnginfo=pnginfo)
    return input_dir


def _mock_invisible_engine():
    """Create a mock InvisibleEngine that writes a copy of the input image."""

    def _mock_remove_watermark(image_path, output_path=None, **kwargs):
        out = output_path or image_path.with_stem(image_path.stem + "_clean")
        out.parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(image_path)
        img.save(out)
        return out

    mock_engine = MagicMock()
    mock_engine.remove_watermark.side_effect = _mock_remove_watermark
    mock_cls = MagicMock(return_value=mock_engine)
    return mock_cls, mock_engine


def _mock_invisible_engine_drops_alpha():
    """Mock InvisibleEngine that mimics the real engine's BGR-only output path.

    The real diffusion-based engine reads with cv2.IMREAD_COLOR and writes a
    3-channel result. This mock simulates that so we can regression-test alpha
    preservation across the ``all`` pipeline.
    """

    def _mock_remove_watermark(image_path, output_path=None, **kwargs):
        out = output_path or image_path.with_stem(image_path.stem + "_clean")
        out.parent.mkdir(parents=True, exist_ok=True)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        cv2.imwrite(str(out), bgr)
        return out

    mock_engine = MagicMock()
    mock_engine.remove_watermark.side_effect = _mock_remove_watermark
    mock_cls = MagicMock(return_value=mock_engine)
    return mock_cls, mock_engine


class TestMainGroup:
    """Tests for the top-level CLI group."""

    def test_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Remove visible and invisible" in result.output

    def test_version(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output

    def test_no_command_shows_banner(self, runner):
        result = runner.invoke(main, [])
        assert result.exit_code == 0
        assert "pagedMark" in result.output


class TestVisibleCommand:
    """Tests for the 'visible' subcommand."""

    def test_visible_help(self, runner):
        result = runner.invoke(main, ["visible", "--help"])
        assert result.exit_code == 0
        assert "visible AI watermark" in result.output
        assert "--mark" in result.output

    def test_visible_auto_no_mark_exits_two_with_eraser_hint(self, runner, sample_png, tmp_path):
        # No known visible mark and no AI provenance signal: the command must not
        # re-serve the input as a finished result. It exits EXIT_NO_VISIBLE_MARK
        # (2) -- distinct from success (0) and a hard error (1) -- writes no
        # output file, and points the user at the region eraser.
        output = tmp_path / "clean.png"
        result = runner.invoke(main, ["visible", str(sample_png), "-o", str(output)])
        assert result.exit_code == 2, result.output
        assert not output.exists()
        assert "erase" in result.output
        # The "no signal" branch must NOT imply the image is clean: a missing
        # metadata proxy is not proof an invisible pixel watermark (SynthID) is
        # absent, so the message preserves that uncertainty and routes to 'all'.
        assert "SynthID" in result.output
        assert "all" in result.output

    def test_visible_auto_no_mark_routes_to_all_when_metadata(self, runner, tmp_path):
        # An image whose only signal is an invisible/metadata watermark (here SD
        # generation parameters) has no visible mark to remove; the command must
        # exit 2 and upsell the full 'all' pipeline rather than the eraser.
        img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8))
        pnginfo = PngInfo()
        pnginfo.add_text("parameters", "Steps: 20, Sampler: Euler, a test landscape")
        src = tmp_path / "ai.png"
        img.save(src, pnginfo=pnginfo)
        output = tmp_path / "clean.png"
        result = runner.invoke(main, ["visible", str(src), "-o", str(output)])
        assert result.exit_code == 2, result.output
        assert not output.exists()
        assert "all" in result.output

    def test_visible_basic(self, runner, sample_png, tmp_path):
        output = tmp_path / "clean.png"
        result = runner.invoke(
            main,
            ["visible", str(sample_png), "-o", str(output), "--no-detect"],
        )
        assert result.exit_code == 0
        assert output.exists()
        assert "Saved" in result.output

    def test_visible_default_output_name(self, runner, sample_png):
        result = runner.invoke(main, ["visible", str(sample_png), "--no-detect"])
        assert result.exit_code == 0
        expected = sample_png.with_stem(sample_png.stem + "_clean")
        assert expected.exists()

    def test_visible_backend_cv2(self, runner, sample_png, tmp_path):
        # The old --inpaint/--no-inpaint flags are gone; the fill backend is picked
        # with --backend (localize -> fill). cv2 needs no ONNX model.
        output = tmp_path / "clean.png"
        result = runner.invoke(
            main,
            [
                "visible",
                str(sample_png),
                "-o",
                str(output),
                "--backend",
                "cv2",
                "--no-detect",
            ],
        )
        assert result.exit_code == 0
        assert output.exists()

    def test_visible_no_detect(self, runner, sample_png, tmp_path):
        output = tmp_path / "clean.png"
        result = runner.invoke(
            main,
            ["visible", str(sample_png), "-o", str(output), "--no-detect"],
        )
        assert result.exit_code == 0

    def test_visible_nonexistent_file(self, runner):
        result = runner.invoke(main, ["visible", "/nonexistent/file.png"])
        assert result.exit_code != 0

    def test_visible_preserves_rgba_transparency(self, runner, tmp_path):
        """Visible removal on an RGBA PNG must keep the alpha channel,
        not silently flatten the image onto an opaque background.
        """
        rgba = np.zeros((200, 200, 4), dtype=np.uint8)
        rgba[:, :, :3] = 200  # light grey foreground
        rgba[50:150, 50:150, 3] = 255  # opaque square in the middle, rest transparent
        src = tmp_path / "rgba_in.png"
        cv2.imwrite(str(src), rgba)

        output = tmp_path / "rgba_out.png"
        result = runner.invoke(
            main,
            ["visible", str(src), "-o", str(output), "--no-detect"],
        )

        assert result.exit_code == 0, result.output
        assert output.exists()

        out = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
        assert out.ndim == 3, f"output is not 3D: shape={out.shape}"
        assert out.shape[2] == 4, f"output is not RGBA: shape={out.shape}"
        # The transparent corners must remain transparent.
        assert out[0, 0, 3] == 0
        assert out[199, 199, 3] == 0
        # The opaque centre remains opaque (the watermark region default is bottom-right,
        # which doesn't overlap the centre square at 200x200).
        assert out[100, 100, 3] == 255

    def test_visible_keeps_alpha_opaque_in_watermark_region(self, runner, tmp_path):
        """Regression for issue #30 (white box): on an opaque RGBA image, the
        watermark region must stay OPAQUE. Reverse-alpha recovers real pixels
        there, so zeroing alpha would punch a transparent hole that renders as a
        solid white box on any non-transparent viewer.
        """
        rgba = np.full((200, 200, 4), 255, dtype=np.uint8)  # fully opaque white
        src = tmp_path / "rgba_full.png"
        cv2.imwrite(str(src), rgba)

        output = tmp_path / "rgba_kept.png"
        result = runner.invoke(
            main,
            ["visible", str(src), "-o", str(output), "--no-detect"],
        )

        assert result.exit_code == 0, result.output
        out = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
        assert out.shape[2] == 4
        # Default sparkle position is in the bottom-right; alpha there must stay 255.
        from pagedmark.gemini_engine import get_watermark_config

        cfg = get_watermark_config(200, 200)
        px, py = cfg.get_position(200, 200)
        size = cfg.logo_size
        assert out[py + size // 2, px + size // 2, 3] == 255, "watermark region alpha was zeroed (white-box regression)"
        # No pixel anywhere should have been forced transparent.
        assert int((out[:, :, 3] == 0).sum()) == 0, "spurious transparent pixels introduced"

    def test_visible_rgb_input_stays_rgb(self, runner, sample_png, tmp_path):
        """Regression: a plain RGB PNG must NOT gain a spurious alpha channel."""
        output = tmp_path / "rgb_out.png"
        result = runner.invoke(
            main,
            ["visible", str(sample_png), "-o", str(output), "--no-detect"],
        )

        assert result.exit_code == 0, result.output
        out = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
        assert out.ndim == 3, f"output is not 3D: shape={out.shape}"
        assert out.shape[2] == 3, f"RGB input produced non-RGB output: shape={out.shape}"


class TestInvisibleCommand:
    """Tests for the 'invisible' subcommand."""

    def test_invisible_help(self, runner):
        result = runner.invoke(main, ["invisible", "--help"])
        assert result.exit_code == 0
        assert "invisible" in result.output.lower()

    def test_invisible_basic(self, runner, sample_png, tmp_path):
        mock_cls, mock_engine = _mock_invisible_engine()
        output = tmp_path / "clean.png"
        with (
            patch("pagedmark.invisible_engine.is_available", return_value=True),
            patch("pagedmark.cli.InvisibleEngine", mock_cls, create=True),
            patch("pagedmark.invisible_engine.InvisibleEngine", mock_cls),
        ):
            result = runner.invoke(
                main,
                ["invisible", str(sample_png), "-o", str(output), "--force"],
            )
        assert result.exit_code == 0, result.output
        assert output.exists()
        mock_engine.remove_watermark.assert_called_once()

    def test_invisible_cpu_offload_flows_to_engine(self, runner, sample_png, tmp_path):
        mock_cls, _mock_engine = _mock_invisible_engine()
        output = tmp_path / "clean.png"
        with (
            patch("pagedmark.invisible_engine.is_available", return_value=True),
            patch("pagedmark.cli.InvisibleEngine", mock_cls, create=True),
            patch("pagedmark.invisible_engine.InvisibleEngine", mock_cls),
        ):
            result = runner.invoke(
                main,
                ["invisible", str(sample_png), "-o", str(output), "--cpu-offload", "--force"],
            )

        assert result.exit_code == 0, result.output
        assert mock_cls.call_args.kwargs["cpu_offload"] is True

    def test_invisible_default_output(self, runner, sample_png):
        mock_cls, _mock_engine = _mock_invisible_engine()
        with (
            patch("pagedmark.invisible_engine.is_available", return_value=True),
            patch("pagedmark.cli.InvisibleEngine", mock_cls, create=True),
            patch("pagedmark.invisible_engine.InvisibleEngine", mock_cls),
        ):
            result = runner.invoke(main, ["invisible", str(sample_png), "--force"])
        assert result.exit_code == 0, result.output
        expected = sample_png.with_stem(sample_png.stem + "_clean")
        assert expected.exists()

    def test_invisible_leaves_the_polish_default_to_the_library(self, runner, sample_png):
        """An untyped --adaptive-polish reaches the engine as None, not as a value.

        The per-profile default lives in watermark_profiles, so the CLI must pass the
        user's non-choice through rather than resolving it here. Resolving in the CLI
        is how the library and the CLI came to disagree on the same profile.
        """
        mock_cls, mock_engine = _mock_invisible_engine()
        with (
            patch("pagedmark.invisible_engine.is_available", return_value=True),
            patch("pagedmark.cli.InvisibleEngine", mock_cls, create=True),
            patch("pagedmark.invisible_engine.InvisibleEngine", mock_cls),
        ):
            result = runner.invoke(main, ["invisible", str(sample_png), "--force"])
        assert result.exit_code == 0, result.output
        assert mock_engine.remove_watermark.call_args.kwargs["adaptive_polish"] is None

    def test_invisible_no_adaptive_polish_disables(self, runner, sample_png):
        mock_cls, mock_engine = _mock_invisible_engine()
        with (
            patch("pagedmark.invisible_engine.is_available", return_value=True),
            patch("pagedmark.cli.InvisibleEngine", mock_cls, create=True),
            patch("pagedmark.invisible_engine.InvisibleEngine", mock_cls),
        ):
            result = runner.invoke(main, ["invisible", str(sample_png), "--no-adaptive-polish", "--force"])
        assert result.exit_code == 0, result.output
        assert mock_engine.remove_watermark.call_args.kwargs["adaptive_polish"] is False

    def test_knobs_the_fixed_stack_cannot_honor_are_not_offered(self, runner, sample_png):
        """--model/--steps/--guidance-scale/--device/--auto are gone, not rejected.

        Each pinned a value the profiles fix (model stack, per-stage schedule, CFG 1.0,
        CUDA), so accepting one only produced an error several layers down -- a flag
        that advertises a capability the library does not have. Click now refuses the
        option itself, which is the honest answer and the one a caller can act on.
        """
        retired = (
            ["--model", "org/custom-sdxl"],
            ["--steps", "20"],
            ["--guidance-scale", "5.5"],
            ["--device", "cpu"],
            ["--auto"],
        )
        # All THREE diffusion commands, not just `invisible`. Each used to declare these
        # inline, so removing them from one and not the others is a live possibility.
        for command in ("invisible", "all", "batch"):
            target = str(sample_png.parent) if command == "batch" else str(sample_png)
            for args in retired:
                result = runner.invoke(main, [command, target, *args, "--force"])
                assert result.exit_code == 2, f"{command} {args[0]}: {result.output}"
                assert "No such option" in result.output, f"{command} {args[0]}: {result.output}"

    def test_retired_pipeline_names_are_rejected_not_silently_remapped(self, runner, sample_png):
        """default/sdxl/controlnet/qwen were removed with their CPU code paths.

        Click rejects them at parse time. Mapping them onward would run a profile the
        caller never chose, at a different strength and a different quality.
        """
        for retired in ("default", "sdxl", "controlnet", "qwen"):
            result = runner.invoke(main, ["invisible", str(sample_png), "--pipeline", retired, "--force"])
            assert result.exit_code == 2, result.output
            assert "is not one of" in result.output

    def test_invisible_nonexistent_file(self, runner):
        result = runner.invoke(main, ["invisible", "/nonexistent/file.png"])
        assert result.exit_code != 0

    def test_invisible_no_signal_skips_and_exits_two(self, runner, sample_png, tmp_path):
        """P0#5: when no invisible AI watermark is locally detectable, the diffusion
        scrub must NOT run (it would only degrade a clean image). Mirrors the visible
        no-mark contract: write no output, exit 2, and DO NOT imply the image is
        clean (stripped SynthID provenance is not proof of absence)."""
        mock_cls, mock_engine = _mock_invisible_engine()
        output = tmp_path / "clean.png"
        with (
            patch("pagedmark.invisible_engine.is_available", return_value=True),
            patch("pagedmark.cli.InvisibleEngine", mock_cls, create=True),
            patch("pagedmark.invisible_engine.InvisibleEngine", mock_cls),
        ):
            result = runner.invoke(main, ["invisible", str(sample_png), "-o", str(output)])
        assert result.exit_code == 2, result.output
        assert not output.exists()
        mock_engine.remove_watermark.assert_not_called()
        assert "--force" in result.output
        assert "SynthID" in result.output  # the message must preserve removal uncertainty

    def test_invisible_force_runs_scrub_on_no_signal(self, runner, sample_png, tmp_path):
        """--force overrides the no-signal skip: the scrub runs regardless."""
        mock_cls, mock_engine = _mock_invisible_engine()
        output = tmp_path / "clean.png"
        with (
            patch("pagedmark.invisible_engine.is_available", return_value=True),
            patch("pagedmark.cli.InvisibleEngine", mock_cls, create=True),
            patch("pagedmark.invisible_engine.InvisibleEngine", mock_cls),
        ):
            result = runner.invoke(main, ["invisible", str(sample_png), "-o", str(output), "--force"])
        assert result.exit_code == 0, result.output
        mock_engine.remove_watermark.assert_called_once()

    def test_invisible_runs_without_force_when_signal_present(self, runner, tmp_path):
        """An image carrying an AI metadata signal IS a scrub target, so the run
        proceeds with no --force needed."""
        img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8))
        pnginfo = PngInfo()
        pnginfo.add_text("parameters", "Steps: 20, Sampler: Euler, a test landscape")
        src = tmp_path / "ai.png"
        img.save(src, pnginfo=pnginfo)
        output = tmp_path / "clean.png"
        mock_cls, mock_engine = _mock_invisible_engine()
        with (
            patch("pagedmark.invisible_engine.is_available", return_value=True),
            patch("pagedmark.cli.InvisibleEngine", mock_cls, create=True),
            patch("pagedmark.invisible_engine.InvisibleEngine", mock_cls),
        ):
            result = runner.invoke(main, ["invisible", str(src), "-o", str(output)])
        assert result.exit_code == 0, result.output
        mock_engine.remove_watermark.assert_called_once()


class TestAllCommand:
    """Tests for the 'all' subcommand (full pipeline)."""

    def test_all_help(self, runner):
        result = runner.invoke(main, ["all", "--help"])
        assert result.exit_code == 0
        assert "visible" in result.output.lower()

    def test_all_basic(self, runner, sample_png, tmp_path):
        mock_cls, _mock_engine = _mock_invisible_engine()
        output = tmp_path / "clean.png"
        with (
            patch("pagedmark.cli.InvisibleEngine", mock_cls, create=True),
            patch("pagedmark.invisible_engine.InvisibleEngine", mock_cls),
            patch("pagedmark.invisible_engine.is_available", return_value=True),
        ):
            result = runner.invoke(
                main,
                ["all", str(sample_png), "-o", str(output), "--force"],
            )
        assert result.exit_code == 0, result.output
        assert output.exists()

    def test_all_cpu_offload_flows_to_engine(self, runner, sample_png, tmp_path):
        mock_cls, _mock_engine = _mock_invisible_engine()
        output = tmp_path / "clean.png"
        with (
            patch("pagedmark.cli.InvisibleEngine", mock_cls, create=True),
            patch("pagedmark.invisible_engine.InvisibleEngine", mock_cls),
            patch("pagedmark.invisible_engine.is_available", return_value=True),
        ):
            result = runner.invoke(
                main,
                ["all", str(sample_png), "-o", str(output), "--cpu-offload", "--force"],
            )

        assert result.exit_code == 0, result.output
        assert mock_cls.call_args.kwargs["cpu_offload"] is True

    def test_all_controlnet_scale_flows_to_engine(self, runner, sample_png, tmp_path):
        """The click option is `--controlnet-scale`, the engine parameter is
        `controlnet_conditioning_scale`, and `InvisibleOptions` now uses the engine's
        spelling so the translation happens once. Pin the value, not just the name:
        hardcoding the constant in `_run_invisible` passed the whole suite before."""
        mock_cls, _mock_engine = _mock_invisible_engine()
        output = tmp_path / "clean.png"
        with (
            patch("pagedmark.cli.InvisibleEngine", mock_cls, create=True),
            patch("pagedmark.invisible_engine.InvisibleEngine", mock_cls),
            patch("pagedmark.invisible_engine.is_available", return_value=True),
        ):
            result = runner.invoke(
                main,
                ["all", str(sample_png), "-o", str(output), "--controlnet-scale", "0.65", "--force"],
            )

        assert result.exit_code == 0, result.output
        assert mock_cls.call_args.kwargs["controlnet_conditioning_scale"] == pytest.approx(0.65)

    def test_all_nonexistent_file(self, runner):
        result = runner.invoke(main, ["all", "/nonexistent/file.png"])
        assert result.exit_code != 0

    def test_all_visible_step_uses_registry(self, runner, sample_png, tmp_path):
        """Regression (#1): the `all` visible step must route through the registry
        (remove_auto_marks), so Doubao/Jimeng/Samsung/pill marks are handled -- not
        just the Gemini sparkle via a hardcoded GeminiEngine."""
        mock_cls, _mock_engine = _mock_invisible_engine()
        output = tmp_path / "clean.png"

        def _fake_remove_auto(image, **kwargs):
            return image, []

        with (
            patch("pagedmark.cli.InvisibleEngine", mock_cls, create=True),
            patch("pagedmark.invisible_engine.InvisibleEngine", mock_cls),
            patch("pagedmark.invisible_engine.is_available", return_value=True),
            patch("pagedmark.watermark_registry.remove_auto_marks", side_effect=_fake_remove_auto) as mock_auto,
        ):
            result = runner.invoke(main, ["all", str(sample_png), "-o", str(output), "--force"])
        assert result.exit_code == 0, result.output
        mock_auto.assert_called()  # the registry auto-detector drove the visible pass

    def test_all_skips_invisible_on_no_signal_but_succeeds(self, runner, sample_png, tmp_path):
        """P0#5: with no detectable invisible watermark and no --force, `all` skips
        the destructive step 2 (pixels left intact) but STILL succeeds (exit 0) --
        visible removal + metadata strip ran and a file is written. Distinct from the
        GPU-missing skip, which is a non-zero failure."""
        mock_cls, mock_engine = _mock_invisible_engine()
        output = tmp_path / "clean.png"
        with (
            patch("pagedmark.cli.InvisibleEngine", mock_cls, create=True),
            patch("pagedmark.invisible_engine.InvisibleEngine", mock_cls),
            patch("pagedmark.invisible_engine.is_available", return_value=True),
        ):
            result = runner.invoke(main, ["all", str(sample_png), "-o", str(output)])
        assert result.exit_code == 0, result.output
        assert output.exists()
        mock_engine.remove_watermark.assert_not_called()
        assert "Skipped (no invisible" in result.output

    def test_all_loud_warning_and_nonzero_exit_when_gpu_missing(self, runner, sample_png, tmp_path):
        """Regression (#14/#47): when the GPU extra is absent the invisible step is
        skipped, but the output still looks processed -- the run must fail loudly
        (prominent banner + non-zero exit) so a skipped SynthID pass is not mistaken
        for a clean result. The output file is still written (visible + metadata).

        The printed extra is derived from the host's device rather than pinned to the
        CUDA one: on a Mac, ``qwen-zimage`` is an install hint that does not install.
        """
        from pagedmark._internal.watermark_profiles import install_extra_for_device
        from pagedmark._internal.watermark_remover import get_device

        expected_extra = install_extra_for_device(get_device()).strip("'")
        output = tmp_path / "clean.png"
        with patch("pagedmark.invisible_engine.is_available", return_value=False):
            result = runner.invoke(main, ["all", str(sample_png), "-o", str(output)])
        assert result.exit_code != 0, result.output
        assert "NOT removed" in result.output
        assert expected_extra in result.output
        assert output.exists()  # visible + metadata still produced a file

    def test_all_reports_metadata_that_survived_stripping(self, runner, sample_png, tmp_path):
        """The full pipeline must verify the metadata result before reporting success.

        ``remove_ai_metadata`` is deliberately fail-safe and may copy an undecodable
        input through unchanged. Calling it directly let ``all`` print a successful
        strip even when a marker survived.
        """
        output = tmp_path / "clean.png"
        with (
            patch("pagedmark.invisible_engine.is_available", return_value=True),
            # The no-signal gate lives in the library now; patch it there, not on the
            # CLI module, or the diffusion stage runs for real.
            patch("pagedmark.api._SourceEvidence.has_invisible_target", return_value=False),
            patch(
                "pagedmark.metadata.strip_and_verify",
                return_value=(tmp_path / "intermediate.png", {"c2pa": True}),
            ),
        ):
            result = runner.invoke(main, ["all", str(sample_png), "-o", str(output)])

        assert result.exit_code != 0
        assert "metadata" in result.output.lower()
        assert "survived" in result.output.lower()
        assert "AI metadata stripped" not in result.output
        # An incomplete strip must leave NOTHING on disk: an AI-readable output plus a
        # non-zero exit is the failure mode the pre-write raise exists to prevent.
        assert not output.exists()

    def test_all_preserves_rgba_across_invisible_step(self, runner, tmp_path):
        """Regression: ``all`` must keep transparency even when the invisible
        step writes a 3-channel result (as the real diffusion engine does).
        """
        rgba = np.zeros((200, 200, 4), dtype=np.uint8)
        rgba[:, :, :3] = 200
        rgba[50:150, 50:150, 3] = 255  # opaque square; corners transparent
        src = tmp_path / "rgba_in.png"
        cv2.imwrite(str(src), rgba)

        output = tmp_path / "rgba_out.png"
        mock_cls, _engine = _mock_invisible_engine_drops_alpha()
        with (
            patch("pagedmark.cli.InvisibleEngine", mock_cls, create=True),
            patch("pagedmark.invisible_engine.InvisibleEngine", mock_cls),
            patch("pagedmark.cli.invisible_available", return_value=True, create=True),
            patch("pagedmark.invisible_engine.is_available", return_value=True),
        ):
            result = runner.invoke(main, ["all", str(src), "-o", str(output), "--force"])

        assert result.exit_code == 0, result.output
        out = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
        assert out.ndim == 3, f"output not 3D: shape={out.shape}"
        assert out.shape[2] == 4, f"output is not RGBA: shape={out.shape}"
        assert out[0, 0, 3] == 0
        assert out[100, 100, 3] == 255


class TestMetadataCommand:
    """Tests for the 'metadata' subcommand."""

    def test_metadata_help(self, runner):
        result = runner.invoke(main, ["metadata", "--help"])
        assert result.exit_code == 0

    def test_metadata_check_clean(self, runner, tmp_clean_png):
        result = runner.invoke(main, ["metadata", str(tmp_clean_png), "--check"])
        assert result.exit_code == 0
        assert "No AI metadata" in result.output

    def test_metadata_check_ai(self, runner, tmp_png_with_ai_metadata):
        result = runner.invoke(main, ["metadata", str(tmp_png_with_ai_metadata), "--check"])
        assert result.exit_code == 0
        assert "AI metadata detected" in result.output

    def test_metadata_remove(self, runner, tmp_png_with_ai_metadata, tmp_path):
        output = tmp_path / "stripped.png"
        result = runner.invoke(
            main,
            [
                "metadata",
                str(tmp_png_with_ai_metadata),
                "--remove",
                "-o",
                str(output),
            ],
        )
        assert result.exit_code == 0
        assert "stripped" in result.output

    def test_metadata_remove_reports_failure_when_the_strip_was_a_no_op(self, runner, tmp_path):
        """A file PIL cannot decode is copied through UNCHANGED by the fail-safe.

        That is correct (never crash a worker on a partial upload) but the command used
        to print "AI metadata stripped ->" and exit 0 for it, so a caller could not tell
        a real strip from a no-op and the output still read as AI. A Samsung C2PA
        compatibility case exposed the defect.
        """
        # PNG signature + a C2PA (caBX) chunk, then garbage: the byte scanner sees the
        # marker, PIL cannot decode it.
        src = tmp_path / "undecodable.png"
        payload = b"c2pa" + b"\x00" * 32
        chunk = len(payload).to_bytes(4, "big") + b"caBX" + payload + b"\x00\x00\x00\x00"
        src.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk + b"NOTAPNG" * 8)
        out = tmp_path / "cleaned.png"

        result = runner.invoke(main, ["metadata", str(src), "--remove", "-o", str(out)])

        from pagedmark.metadata import get_ai_metadata

        if not get_ai_metadata(src):
            pytest.skip("fixture does not register as an AI-metadata carrier")
        assert result.exit_code != 0, "a no-op strip must not report success"
        assert "stripped ->" not in result.output

    def test_metadata_remove_in_place(self, runner, tmp_png_with_ai_metadata):
        """With ``-o`` omitted, the strip overwrites the source in place (default
        output_path=None). Previously every test passed an explicit ``-o``."""
        from pagedmark.metadata import has_ai_metadata

        assert has_ai_metadata(tmp_png_with_ai_metadata)  # precondition
        result = runner.invoke(main, ["metadata", str(tmp_png_with_ai_metadata), "--remove"])
        assert result.exit_code == 0, result.output
        assert not has_ai_metadata(tmp_png_with_ai_metadata)  # source overwritten, AI metadata gone


class TestIdentifyCommand:
    """Tests for the 'identify' subcommand."""

    def test_identify_help(self, runner):
        result = runner.invoke(main, ["identify", "--help"])
        assert result.exit_code == 0

    def test_identify_clean_png(self, runner, tmp_clean_png):
        result = runner.invoke(main, ["identify", str(tmp_clean_png), "--no-visible"])
        assert result.exit_code == 0
        assert "unknown" in result.output

    def test_identify_unknown_explains_why(self, runner, tmp_clean_png):
        # An unknown verdict must explain itself inline (issue #22: users read a bare
        # "unknown" as the tool being broken) rather than only in the caveats section.
        result = runner.invoke(main, ["identify", str(tmp_clean_png), "--no-visible"])
        assert result.exit_code == 0
        assert "No locally-readable AI signal found" in result.output
        assert "not the same as 'clean'" in result.output

    def test_identify_ai_png_reports_platform(self, runner, tmp_png_with_ai_metadata):
        result = runner.invoke(main, ["identify", str(tmp_png_with_ai_metadata), "--no-visible"])
        assert result.exit_code == 0
        assert "AI-generated" in result.output
        assert "Stable Diffusion" in result.output

    def test_identify_reports_generated_source_kind(self, runner):
        """The C2PA trainedAlgorithmicMedia source type sharpens the verdict to
        'AI-generated (fully synthetic)' at the CLI (the ai_source_kind branch)."""
        from pathlib import Path

        sample = Path(__file__).resolve().parent.parent / "data" / "fixtures" / "provenance" / "chatgpt-1.png"
        if not sample.exists():
            pytest.skip("chatgpt sample not present")
        result = runner.invoke(main, ["identify", str(sample), "--no-visible"])
        assert result.exit_code == 0
        assert "AI-generated (fully synthetic)" in result.output
        assert "C2PA validation: integrity=valid, signature=valid" in result.output
        assert "signer trust=untrusted, signer validity=expired" in result.output

    def test_identify_json_is_valid(self, runner, tmp_png_with_ai_metadata):
        result = runner.invoke(main, ["identify", str(tmp_png_with_ai_metadata), "--no-visible", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["is_ai_generated"] is True
        assert payload["confidence"] == "high"

    def test_identify_nonexistent_file(self, runner):
        result = runner.invoke(main, ["identify", "/nonexistent/file.png"])
        assert result.exit_code != 0


class TestBatchCommand:
    """Tests for the 'batch' subcommand."""

    def test_batch_help(self, runner):
        result = runner.invoke(main, ["batch", "--help"])
        assert result.exit_code == 0

    def test_batch_empty_dir(self, runner, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = runner.invoke(main, ["batch", str(empty_dir)])
        assert result.exit_code == 0
        assert "No supported images" in result.output

    def test_batch_visible_mode(self, runner, tmp_path):
        input_dir = _make_batch_dir(tmp_path)
        output_dir = tmp_path / "output"
        result = runner.invoke(
            main,
            ["batch", str(input_dir), "-o", str(output_dir), "--mode", "visible"],
        )
        assert result.exit_code == 0
        assert "3 processed" in result.output
        assert output_dir.exists()
        assert len(list(output_dir.glob("*.png"))) == 3

    def test_batch_metadata_mode(self, runner, tmp_path):
        input_dir = _make_batch_dir_with_metadata(tmp_path)
        output_dir = tmp_path / "output"
        result = runner.invoke(
            main,
            ["batch", str(input_dir), "-o", str(output_dir), "--mode", "metadata"],
        )
        assert result.exit_code == 0
        assert "3 processed" in result.output
        assert output_dir.exists()
        assert len(list(output_dir.glob("*.png"))) == 3
        # Verify AI metadata was stripped
        for out_img in output_dir.glob("*.png"):
            with Image.open(out_img) as img:
                assert "parameters" not in img.info

    def test_batch_invisible_mode(self, runner, tmp_path):
        input_dir = _make_batch_dir(tmp_path)
        output_dir = tmp_path / "output"
        mock_cls, _mock_engine = _mock_invisible_engine()
        with (
            patch("pagedmark.cli.InvisibleEngine", mock_cls, create=True),
            patch("pagedmark.invisible_engine.InvisibleEngine", mock_cls),
            patch("pagedmark.cli.invisible_available", return_value=True, create=True),
            patch("pagedmark.invisible_engine.is_available", return_value=True),
        ):
            result = runner.invoke(
                main,
                ["batch", str(input_dir), "-o", str(output_dir), "--mode", "invisible", "--force"],
            )
        assert result.exit_code == 0, result.output
        assert "3 processed" in result.output

    def test_batch_cpu_offload_flows_to_cached_engine(self, runner, tmp_path):
        input_dir = _make_batch_dir(tmp_path)
        output_dir = tmp_path / "output"
        mock_cls, _mock_engine = _mock_invisible_engine()
        with (
            patch("pagedmark.cli.InvisibleEngine", mock_cls, create=True),
            patch("pagedmark.invisible_engine.InvisibleEngine", mock_cls),
            patch("pagedmark.cli.invisible_available", return_value=True, create=True),
            patch("pagedmark.invisible_engine.is_available", return_value=True),
        ):
            result = runner.invoke(
                main,
                [
                    "batch",
                    str(input_dir),
                    "-o",
                    str(output_dir),
                    "--mode",
                    "invisible",
                    "--cpu-offload",
                    "--force",
                ],
            )

        assert result.exit_code == 0, result.output
        assert mock_cls.call_args.kwargs["cpu_offload"] is True

    @pytest.mark.parametrize("mode", ["invisible", "all"])
    def test_batch_controlnet_scale_flows_to_the_cached_engine(self, runner, tmp_path, mode):
        """`_batch_engine` is a SECOND engine-construction site: `all` builds its engine
        inside `_run_invisible`, batch prebuilds one for the whole directory. Pinning the
        value on the `all` path alone leaves this one free to hardcode a constant."""
        input_dir = _make_batch_dir(tmp_path)
        output_dir = tmp_path / "output"
        mock_cls, _mock_engine = _mock_invisible_engine()
        args = ["batch", str(input_dir), "-o", str(output_dir), "--mode", mode]
        with (
            patch("pagedmark.cli.InvisibleEngine", mock_cls, create=True),
            patch("pagedmark.invisible_engine.InvisibleEngine", mock_cls),
            patch("pagedmark.cli.invisible_available", return_value=True, create=True),
            patch("pagedmark.invisible_engine.is_available", return_value=True),
        ):
            result = runner.invoke(main, [*args, "--controlnet-scale", "0.65", "--force"])

        assert result.exit_code == 0, result.output
        assert mock_cls.call_args.kwargs["controlnet_conditioning_scale"] == pytest.approx(0.65)

    def test_batch_invisible_skips_no_signal_and_copies_through(self, runner, tmp_path):
        """P0#5: batch invisible mode skips the scrub on signal-less images (no
        --force) and copies the input through, so the output dir is complete with the
        pixels left intact and the engine never called."""
        input_dir = _make_batch_dir(tmp_path)
        output_dir = tmp_path / "output"
        mock_cls, mock_engine = _mock_invisible_engine()
        with (
            patch("pagedmark.cli.InvisibleEngine", mock_cls, create=True),
            patch("pagedmark.invisible_engine.InvisibleEngine", mock_cls),
            patch("pagedmark.cli.invisible_available", return_value=True, create=True),
            patch("pagedmark.invisible_engine.is_available", return_value=True),
        ):
            result = runner.invoke(
                main,
                ["batch", str(input_dir), "-o", str(output_dir), "--mode", "invisible"],
            )
        assert result.exit_code == 0, result.output
        assert "3 processed" in result.output
        assert len(list(output_dir.glob("*.png"))) == 3  # inputs copied through
        mock_engine.remove_watermark.assert_not_called()

    def test_batch_all_mode(self, runner, tmp_path):
        input_dir = _make_batch_dir(tmp_path)
        output_dir = tmp_path / "output"
        mock_cls, _mock_engine = _mock_invisible_engine()
        with (
            patch("pagedmark.cli.InvisibleEngine", mock_cls, create=True),
            patch("pagedmark.invisible_engine.InvisibleEngine", mock_cls),
            patch("pagedmark.cli.invisible_available", return_value=True, create=True),
            patch("pagedmark.invisible_engine.is_available", return_value=True),
        ):
            result = runner.invoke(
                main,
                ["batch", str(input_dir), "-o", str(output_dir), "--mode", "all", "--force"],
            )
        assert result.exit_code == 0, result.output
        assert "3 processed" in result.output

    def test_batch_all_mode_preserves_rgba(self, runner, tmp_path):
        """Regression: batch ``all`` must keep transparency across the
        alpha-dropping invisible step (mirrors test_all_preserves_rgba_...).
        """
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        rgba = np.zeros((200, 200, 4), dtype=np.uint8)
        rgba[:, :, :3] = 200
        rgba[50:150, 50:150, 3] = 255
        cv2.imwrite(str(input_dir / "rgba.png"), rgba)

        output_dir = tmp_path / "output"
        mock_cls, _engine = _mock_invisible_engine_drops_alpha()
        with (
            patch("pagedmark.cli.InvisibleEngine", mock_cls, create=True),
            patch("pagedmark.invisible_engine.InvisibleEngine", mock_cls),
            patch("pagedmark.cli.invisible_available", return_value=True, create=True),
            patch("pagedmark.invisible_engine.is_available", return_value=True),
        ):
            result = runner.invoke(
                main,
                ["batch", str(input_dir), "-o", str(output_dir), "--mode", "all", "--force"],
            )
        assert result.exit_code == 0, result.output

        out = cv2.imread(str(output_dir / "rgba.png"), cv2.IMREAD_UNCHANGED)
        assert out.ndim == 3, f"output not 3D: shape={out.shape}"
        assert out.shape[2] == 4, f"output is not RGBA: shape={out.shape}"
        assert out[0, 0, 3] == 0
        assert out[100, 100, 3] == 255

    def test_batch_forwards_an_explicit_adaptive_polish(self, runner, tmp_path):
        """A typed --adaptive-polish must reach the engine as True, not as None.

        This mocks the engine, so it covers the CLI's forwarding only; the per-profile
        resolution of an UNSET flag happens inside the real engine and is guarded by
        test_invisible_engine.py::TestEngineResolvesThePolishPerProfile.
        """
        input_dir = _make_batch_dir(tmp_path, count=2)
        output_dir = tmp_path / "output"
        mock_cls, mock_engine = _mock_invisible_engine()
        with (
            patch("pagedmark.cli.InvisibleEngine", mock_cls, create=True),
            patch("pagedmark.invisible_engine.InvisibleEngine", mock_cls),
            patch("pagedmark.cli.invisible_available", return_value=True, create=True),
            patch("pagedmark.invisible_engine.is_available", return_value=True),
        ):
            result = runner.invoke(
                main,
                [
                    "batch",
                    str(input_dir),
                    "-o",
                    str(output_dir),
                    "--mode",
                    "invisible",
                    "--adaptive-polish",
                    "--force",
                ],
            )
        assert result.exit_code == 0, result.output
        assert "2 processed" in result.output
        # Unset --pipeline forwards None, not a literal: the engine resolves the
        # profile from the device it detects. A CLI-side default would pin the CUDA
        # profile on every host, which is the bug this None replaced.
        assert mock_cls.call_args.kwargs["pipeline"] is None
        assert mock_engine.remove_watermark.call_args.kwargs["adaptive_polish"] is True

    def test_batch_forwards_an_explicit_pipeline_verbatim(self, runner, tmp_path):
        """The other half of the None default: a typed profile must not be re-resolved."""
        input_dir = _make_batch_dir(tmp_path, count=1)
        output_dir = tmp_path / "output"
        mock_cls, _ = _mock_invisible_engine()
        with (
            patch("pagedmark.cli.InvisibleEngine", mock_cls, create=True),
            patch("pagedmark.invisible_engine.InvisibleEngine", mock_cls),
            patch("pagedmark.cli.invisible_available", return_value=True, create=True),
            patch("pagedmark.invisible_engine.is_available", return_value=True),
        ):
            result = runner.invoke(
                main,
                [
                    "batch",
                    str(input_dir),
                    "-o",
                    str(output_dir),
                    "--mode",
                    "invisible",
                    "--pipeline",
                    "sdxl-zimage",
                    "--force",
                ],
            )
        assert result.exit_code == 0, result.output
        assert mock_cls.call_args.kwargs["pipeline"] == "sdxl-zimage"

    def test_batch_default_output_dir(self, runner, tmp_path):
        input_dir = _make_batch_dir(tmp_path)
        result = runner.invoke(
            main,
            ["batch", str(input_dir), "--mode", "visible"],
        )
        assert result.exit_code == 0
        expected_dir = tmp_path / "input_clean"
        assert expected_dir.exists()

    def test_batch_errors_exit_nonzero(self, runner, tmp_path):
        """Regression: batch used to always exit 0 even when every image errored,
        hiding failure from a wrapping service. A corrupt image must yield a non-zero
        exit and an error count."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "corrupt.png").write_bytes(b"this is not a PNG at all" * 50)
        result = runner.invoke(main, ["batch", str(input_dir), "--mode", "visible"])
        assert result.exit_code != 0, result.output
        assert "error" in result.output.lower()

    def test_batch_invisible_gpu_missing_writes_output_and_exits_nonzero(self, runner, tmp_path):
        """Regression: batch --mode invisible with a signal-bearing image but no GPU
        deps used to write NO output for that image and still exit 0, silently dropping
        the files that most needed processing. It must now copy the input through (so the
        output dir is complete), warn about the retained SynthID watermark, and exit
        non-zero -- mirroring the single ``all`` command."""
        input_dir = _make_batch_dir_with_metadata(tmp_path, count=3)  # SD params = invisible signal
        output_dir = tmp_path / "output"
        with patch("pagedmark.invisible_engine.is_available", return_value=False):
            result = runner.invoke(
                main,
                ["batch", str(input_dir), "-o", str(output_dir), "--mode", "invisible"],
            )
        assert result.exit_code != 0, result.output
        assert "NOT removed" in result.output
        assert len(list(output_dir.glob("*.png"))) == 3  # every input copied through, none dropped


class TestGpuHintMarkup:
    """The install hint must name the extra that actually makes a removal run HERE.

    It must also survive to the user with its ``[...]`` token intact (plain output
    prints it verbatim, with no markup parsing). It used to say ``[diffusion]`` on
    every host, which installs torch and diffusers but not the DiffSynth face stage a
    CUDA run reaches -- so following the advice produced a second, different failure.
    The mirror-image failure is the reason the hint is per-device now: on Apple
    Silicon ``qwen-zimage`` downloads a stack Metal cannot load.
    """

    @staticmethod
    def _hint(device: str) -> str:
        from pagedmark._internal.watermark_profiles import install_extra_for_device

        return install_extra_for_device(device)

    def test_each_device_is_told_the_extra_that_works_on_it(self):
        """Pinned per device, so neither half can drift back to one constant."""
        assert self._hint("cuda") == "'pagedmark[qwen-zimage]'"
        assert self._hint("mps") == "'pagedmark[diffusion]'"

    def test_the_hint_is_a_command_the_user_can_actually_paste(self, runner, sample_png):
        """The extras bracket must be shell-quoted.

        A bulk replace once folded these hints into one constant and dropped the quotes
        the originals had. `pip install pagedmark[qwen-zimage]` dies with
        "zsh: no matches found" on the macOS default shell before pip ever runs -- an
        install hint that does not install, which is the failure this hint was fixed to
        stop producing in the first place.
        """
        from pagedmark._internal.watermark_remover import get_device

        with patch("pagedmark.invisible_engine.is_available", return_value=False):
            result = runner.invoke(main, ["invisible", str(sample_png)])
        assert f"pip install {self._hint(get_device())}" in result.output

    def test_invisible_install_hint_names_the_working_extra(self, runner, sample_png):
        from pagedmark._internal.watermark_remover import get_device

        with patch("pagedmark.invisible_engine.is_available", return_value=False):
            result = runner.invoke(main, ["invisible", str(sample_png)])
        assert result.exit_code != 0
        assert self._hint(get_device()).strip("'") in result.output

    def test_all_install_hint_names_the_working_extra(self, runner, sample_png):
        # The `all` pipeline skips the invisible step with a warning that carries
        # the same hint; it must name the same extra.
        from pagedmark._internal.watermark_remover import get_device

        with patch("pagedmark.invisible_engine.is_available", return_value=False):
            result = runner.invoke(main, ["all", str(sample_png)])
        assert self._hint(get_device()).strip("'") in result.output


class TestEraseCommand:
    """Tests for the 'erase' universal region eraser subcommand."""

    def test_erase_help(self, runner):
        result = runner.invoke(main, ["erase", "--help"])
        assert result.exit_code == 0
        assert "--region" in result.output
        assert "--backend" in result.output

    def test_erase_single_region(self, runner, sample_png, tmp_path):
        output = tmp_path / "erased.png"
        result = runner.invoke(
            main,
            ["erase", str(sample_png), "--region", "10,10,40,40", "-o", str(output)],
        )
        assert result.exit_code == 0, result.output
        assert output.exists()

    def test_erase_two_regions(self, runner, sample_png, tmp_path):
        output = tmp_path / "erased2.png"
        result = runner.invoke(
            main,
            [
                "erase",
                str(sample_png),
                "--region",
                "10,10,30,30",
                "--region",
                "120,120,30,30",
                "-o",
                str(output),
            ],
        )
        assert result.exit_code == 0, result.output
        assert output.exists()
        # The banner reports the region count it processed.
        assert "2 region(s)" in result.output

    def test_erase_default_output_name(self, runner, sample_png):
        result = runner.invoke(main, ["erase", str(sample_png), "--region", "10,10,40,40"])
        assert result.exit_code == 0, result.output
        assert sample_png.with_stem(sample_png.stem + "_clean").exists()

    def test_erase_malformed_region_exits_nonzero(self, runner, sample_png, tmp_path):
        output = tmp_path / "x.png"
        # Only three values: click.BadParameter -> non-zero exit, no output file.
        result = runner.invoke(
            main,
            ["erase", str(sample_png), "--region", "1,2,3", "-o", str(output)],
        )
        assert result.exit_code != 0
        assert not output.exists()

    def test_erase_nonexistent_file(self, runner):
        result = runner.invoke(main, ["erase", "/nonexistent/file.png", "--region", "0,0,10,10"])
        assert result.exit_code != 0

    def test_erase_lama_backend_without_onnxruntime(self, runner, sample_png, tmp_path):
        # The LaMa backend needs onnxruntime; without it the CLI must surface a
        # clear error and exit non-zero rather than crash. When onnxruntime IS
        # installed there is no missing-dep path to exercise, so skip.
        from pagedmark.region_eraser import lama_available

        if lama_available():
            pytest.skip("onnxruntime installed; missing-dep error path not reachable")
        output = tmp_path / "y.png"
        result = runner.invoke(
            main,
            ["erase", str(sample_png), "--region", "10,10,40,40", "--backend", "lama", "-o", str(output)],
        )
        assert result.exit_code != 0
        assert "onnxruntime" in result.output.lower()
        assert not output.exists()


def test_visible_backend_runtime_error_exits_cleanly(runner, tmp_path, monkeypatch):
    # A backend whose extra is missing raises RuntimeError in region_eraser.erase;
    # the visible --mark auto path must surface it cleanly, not as a raw traceback (#9).
    from pathlib import Path

    from pagedmark import region_eraser

    doubao = Path(__file__).resolve().parent.parent / "data" / "fixtures" / "provenance" / "doubao-1.png"
    if not doubao.exists():
        pytest.skip("doubao sample not present")

    def boom(*_a, **_k):
        raise RuntimeError("MI-GAN backend requires onnxruntime. Install the extra: ...")

    monkeypatch.setattr(region_eraser, "erase", boom)
    out = tmp_path / "out.png"
    result = runner.invoke(main, ["visible", str(doubao), "-o", str(out), "--backend", "migan"])
    assert result.exit_code == 1
    assert not isinstance(result.exception, RuntimeError), "RuntimeError leaked as a traceback"


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("empty.png", b""),
        ("notimage.jpg", b"plain text, not an image at all " * 20),
        ("truncated.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 40),
    ],
)
@pytest.mark.parametrize("cmd", [["metadata", "--remove"], ["visible", "--backend", "cv2"]])
def test_unreadable_input_exits_cleanly(runner, tmp_path, name, content, cmd):
    """Regression: a corrupt / empty / non-image file (real prod uploads include ~0.2%
    truncated files) must NEVER leak a raw PIL/OSError/ValueError traceback. `metadata
    --remove` is fail-safe -- an undecodable file is copied through unchanged (exit 0),
    a strip that cannot parse the file is a no-op, not a crash; `visible` must decode to
    remove a mark, so it is a clean error (exit 1). Found by the runtime mode fuzz."""
    bad = tmp_path / name
    bad.write_bytes(content)
    out = tmp_path / "out.png"
    result = runner.invoke(main, [cmd[0], str(bad), "-o", str(out), *cmd[1:]])
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        f"leaked a raw traceback: {result.exception!r}"
    )
    if cmd[0] == "metadata":
        assert result.exit_code == 0, result.output  # fail-safe copy-through
        assert out.read_bytes() == content  # input passed through unchanged
    else:
        assert result.exit_code == 1, result.output  # cannot remove a mark from unreadable input
        assert "Error" in result.output


class TestMeasureCommand:
    """The fidelity report is the one command that must work with no GPU and no models."""

    @staticmethod
    def _pair(tmp_path: Path, damage: bool) -> tuple[Path, Path]:
        rng = np.random.default_rng(0)
        frame = np.clip(np.full((256, 256, 3), 24, np.float32) + rng.normal(0, 3, (256, 256, 3)), 0, 255)
        source = tmp_path / "source.png"
        candidate = tmp_path / "candidate.png"
        cv2.imwrite(str(source), frame.astype(np.uint8))
        output = frame.copy()
        if damage:
            output[40:120, 40:120] = 200
        cv2.imwrite(str(candidate), output.astype(np.uint8))
        return source, candidate

    def test_reports_the_three_numbers(self, runner, tmp_path):
        source, candidate = self._pair(tmp_path, damage=True)

        result = runner.invoke(main, ["measure", str(source), str(candidate), "--no-faces"])

        assert result.exit_code == 0, result.output
        assert "Whole frame" in result.output
        assert "Invented texture" in result.output

    def test_json_is_strictly_parseable_even_for_an_identical_pair(self, runner, tmp_path):
        """`Infinity` is what Python emits for identical files, and strict parsers reject it.

        So the identical case travels as a boolean with a null PSNR. This test exists
        because the first version of the command emitted the bare token.
        """
        source, _ = self._pair(tmp_path, damage=False)

        result = runner.invoke(main, ["measure", str(source), str(source), "--no-faces", "--json"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["identical"] is True
        assert payload["psnr"] is None
        assert payload["invented_texture"] == pytest.approx(1.0)

    def test_no_faces_skips_the_detector_entirely(self, runner, tmp_path):
        """--no-faces is also how a user avoids the detector's model download."""
        source, candidate = self._pair(tmp_path, damage=True)
        with patch("pagedmark.fidelity._detected_faces", side_effect=AssertionError("detector was called")):
            result = runner.invoke(main, ["measure", str(source), str(candidate), "--no-faces"])

        assert result.exit_code == 0, result.output
        assert "face 1" not in result.output

    def test_a_missing_file_is_refused_by_the_argument(self, runner, tmp_path):
        source, _ = self._pair(tmp_path, damage=False)

        result = runner.invoke(main, ["measure", str(source), str(tmp_path / "absent.png")])

        assert result.exit_code != 0


class TestPreviewFlag:
    """A preview must be unmistakable for the result, and cheaper than it."""

    def test_it_writes_its_own_file_rather_than_the_clean_one(self, runner, sample_png, tmp_path):
        """Overwriting <source>_clean.png with a 512px look would be the worst outcome."""
        mock_cls, _engine = _mock_invisible_engine()
        with (
            patch("pagedmark.cli.InvisibleEngine", mock_cls, create=True),
            patch("pagedmark.invisible_engine.InvisibleEngine", mock_cls),
            patch("pagedmark.invisible_engine.is_available", return_value=True),
            patch("pagedmark.cli._should_skip_invisible_scrub", return_value=False, create=True),
        ):
            result = runner.invoke(main, ["invisible", str(sample_png), "--preview", "--force"])

        assert result.exit_code == 0, result.output
        assert sample_png.with_stem(sample_png.stem + "_preview").exists()
        assert not sample_png.with_stem(sample_png.stem + "_clean").exists()

    def test_it_reaches_the_engine_and_says_what_it_is(self, runner, sample_png):
        mock_cls, _engine = _mock_invisible_engine()
        with (
            patch("pagedmark.cli.InvisibleEngine", mock_cls, create=True),
            patch("pagedmark.invisible_engine.InvisibleEngine", mock_cls),
            patch("pagedmark.invisible_engine.is_available", return_value=True),
        ):
            result = runner.invoke(main, ["invisible", str(sample_png), "--preview", "--force"])

        assert mock_cls.call_args.kwargs["preview"] is True
        assert "not a result" in result.output

    def test_without_the_flag_nothing_changes(self, runner, sample_png):
        mock_cls, _engine = _mock_invisible_engine()
        with (
            patch("pagedmark.cli.InvisibleEngine", mock_cls, create=True),
            patch("pagedmark.invisible_engine.InvisibleEngine", mock_cls),
            patch("pagedmark.invisible_engine.is_available", return_value=True),
        ):
            result = runner.invoke(main, ["invisible", str(sample_png), "--force"])

        assert mock_cls.call_args.kwargs["preview"] is False
        assert "Preview:" not in result.output


class TestPixelRuntimeGuard:
    """The default install carries no OpenCV, and every pixel command has to say so."""

    @staticmethod
    def _without_pixels(monkeypatch):
        """Simulate the metadata-only install the default `pip install` produces."""
        import pagedmark.optional_deps as optional_deps

        real = optional_deps.module_available
        monkeypatch.setattr(
            optional_deps,
            "module_available",
            lambda *names: False if "cv2" in names else real(*names),
        )

    @pytest.mark.parametrize(
        ("argv", "feature"),
        [
            (["visible", "IMG"], "Visible-mark removal"),
            (["erase", "IMG", "--region", "0,0,4,4"], "Region erasing"),
            (["all", "IMG"], "The full pipeline"),
            (["measure", "IMG", "IMG"], "Measuring"),
        ],
    )
    def test_a_pixel_command_explains_itself_instead_of_crashing(self, runner, sample_png, monkeypatch, argv, feature):
        """Before this guard the first pixel command of a default install died on
        ``ModuleNotFoundError: No module named 'cv2'`` several frames inside the library,
        and the obvious next move -- ``pip install cv2`` -- names a package that does not
        exist on PyPI."""
        self._without_pixels(monkeypatch)

        result = runner.invoke(main, [arg.replace("IMG", str(sample_png)) for arg in argv])

        assert result.exit_code == 1, result.output
        assert feature in result.output
        assert "pagedmark[visible]" in result.output
        assert "Traceback" not in result.output

    def test_the_batch_command_is_guarded_too(self, runner, tmp_path, monkeypatch):
        self._without_pixels(monkeypatch)
        directory = _make_batch_dir(tmp_path, count=1)

        result = runner.invoke(main, ["batch", str(directory), "--mode", "visible"])

        assert result.exit_code == 1, result.output
        assert "Batch processing" in result.output

    def test_metadata_commands_still_run_without_it(self, runner, sample_png, monkeypatch):
        """The discriminating half: the guard must not spread to the light paths, which
        are the reason the default install is metadata-only in the first place."""
        self._without_pixels(monkeypatch)

        result = runner.invoke(main, ["metadata", str(sample_png), "--check"])

        assert result.exit_code == 0, result.output


class TestDoctorCommand:
    """The diagnostic has to answer on the install that cannot run anything else."""

    def test_it_reports_without_any_optional_runtime(self, runner, monkeypatch):
        """The person asking "why does this not work here" is usually the person who has
        not installed the heavy extras, so doctor must not need them."""
        import pagedmark.optional_deps as optional_deps

        monkeypatch.setattr(optional_deps, "module_available", lambda *_names: False)

        result = runner.invoke(main, ["doctor"])

        assert result.exit_code == 0, result.output
        assert "pagedMark:" in result.output
        assert "Device:" in result.output
        assert "Traceback" not in result.output

    def test_it_names_the_extra_that_is_missing(self, runner, monkeypatch):
        import pagedmark.optional_deps as optional_deps

        monkeypatch.setattr(optional_deps, "module_available", lambda *_names: False)

        result = runner.invoke(main, ["doctor"])

        assert "pip install" in result.output

    def test_json_is_machine_readable(self, runner):
        result = runner.invoke(main, ["doctor", "--json"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["version"]
        assert "installed" in payload
        assert "model_cache" in payload
