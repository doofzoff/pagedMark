"""Tests for what the pixel path cannot carry.

The regeneration is RGB-only and writes through OpenCV, so an embedded colour profile
and an alpha channel are both dropped silently. Neither loss is visible to a pixel
metric -- the pixels are fine, it is their interpretation that changed -- so these
tests assert on the container rather than on the image.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageCms

from pagedmark import color_profile


def _profile_bytes() -> bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _rgba(tmp_path, name="source.png", *, size=(64, 64), icc=True, alpha=True):
    rng = np.random.default_rng(0)
    rgb = np.clip(np.full((size[1], size[0], 3), 120.0) + rng.normal(0, 20, (size[1], size[0], 3)), 0, 255)
    if alpha:
        band = np.zeros((size[1], size[0]), np.uint8)
        band[16:48, 16:48] = 255
        image = Image.fromarray(np.dstack([rgb.astype(np.uint8), band]), "RGBA")
    else:
        image = Image.fromarray(rgb.astype(np.uint8), "RGB")
    path = tmp_path / name
    image.save(path, **({"icc_profile": _profile_bytes()} if icc else {}))
    return path


def _written_as_the_model_would(source, destination):
    """What the regeneration leaves behind: RGB, no profile."""
    with Image.open(source) as opened:
        opened.convert("RGB").save(destination)
    return destination


class TestRead:
    def test_it_takes_both_off_the_source(self, tmp_path):
        carryover = color_profile.read(_rgba(tmp_path))

        assert carryover.icc_profile == _profile_bytes()
        assert carryover.alpha is not None
        assert bool(carryover) is True

    def test_a_plain_rgb_source_carries_nothing(self, tmp_path):
        carryover = color_profile.read(_rgba(tmp_path, icc=False, alpha=False))

        assert carryover.icc_profile is None
        assert carryover.alpha is None
        assert bool(carryover) is False

    def test_an_unreadable_source_is_not_an_error(self, tmp_path):
        """The run is about to fail on that file anyway; a lost profile is not the
        failure worth reporting."""
        broken = tmp_path / "broken.png"
        broken.write_bytes(b"not an image")

        assert color_profile.read(broken) == color_profile.Carryover(None, None)


class TestApply:
    def test_it_restores_both(self, tmp_path):
        source = _rgba(tmp_path)
        carryover = color_profile.read(source)
        destination = _written_as_the_model_would(source, tmp_path / "out.png")

        assert color_profile.apply(destination, carryover) is True

        with Image.open(destination) as result:
            assert result.mode == "RGBA"
            assert result.info["icc_profile"] == _profile_bytes()
            assert np.array_equal(np.asarray(result)[..., 3], np.asarray(Image.open(source))[..., 3])

    def test_the_rgb_pixels_are_untouched(self, tmp_path):
        """The whole promise: this restores a container, it does not edit an image."""
        source = _rgba(tmp_path)
        destination = _written_as_the_model_would(source, tmp_path / "out.png")
        before = np.asarray(Image.open(destination).convert("RGB")).copy()

        color_profile.apply(destination, color_profile.read(source))

        after = np.asarray(Image.open(destination).convert("RGB"))
        assert np.array_equal(before, after)

    def test_an_alpha_of_another_size_is_resized_to_the_output(self, tmp_path):
        """Diffusers rounds to its latent grid, so an eight-pixel difference is routine."""
        source = _rgba(tmp_path, size=(72, 72))
        carryover = color_profile.read(source)
        destination = tmp_path / "out.png"
        Image.new("RGB", (64, 64), (10, 20, 30)).save(destination)

        assert color_profile.apply(destination, carryover) is True

        with Image.open(destination) as result:
            assert result.size == (64, 64)
            assert result.mode == "RGBA"

    def test_nothing_to_carry_leaves_the_file_alone(self, tmp_path):
        destination = tmp_path / "out.png"
        Image.new("RGB", (32, 32)).save(destination)
        before = destination.read_bytes()

        assert color_profile.apply(destination, color_profile.Carryover(None, None)) is False
        assert destination.read_bytes() == before

    def test_a_jpeg_keeps_its_scan_data_byte_for_byte(self, tmp_path):
        """A re-encode to attach a profile would undo the quality-preserving writes
        upstream, so the segment is spliced in instead."""
        destination = tmp_path / "out.jpg"
        Image.fromarray(np.full((64, 64, 3), 128, np.uint8)).save(destination, quality=95)
        before = destination.read_bytes()
        icc = _profile_bytes()

        assert color_profile.apply(destination, color_profile.Carryover(icc, None)) is True

        after = destination.read_bytes()
        with Image.open(destination) as result:
            assert result.info["icc_profile"] == icc
        # The original bytes survive as a contiguous run: only a segment was inserted.
        assert before[2:] in after

    def test_a_jpeg_is_not_given_an_alpha_channel(self, tmp_path):
        """The format has none, and the caller does not know the format."""
        source = _rgba(tmp_path)
        destination = tmp_path / "out.jpg"
        Image.fromarray(np.full((64, 64, 3), 128, np.uint8)).save(destination)

        color_profile.apply(destination, color_profile.read(source))

        with Image.open(destination) as result:
            assert result.mode == "RGB"

    def test_a_destination_it_cannot_read_is_reported_not_raised(self, tmp_path):
        broken = tmp_path / "broken.png"
        broken.write_bytes(b"not an image")

        assert color_profile.apply(broken, color_profile.Carryover(_profile_bytes(), None)) is False
