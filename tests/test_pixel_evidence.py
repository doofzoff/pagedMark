"""Tests for the pixel-forensics collector.

These pin the service contract and the edge cases a consumer cannot infer safely:
a family is empty rather than wrong when the image is too small for it, one failing
family does not lose the other five, and artifacts that identify the source image
stay behind their opt-in.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

from pagedmark.pixel_evidence import (
    AC_POSITIONS,
    PIXEL_EVIDENCE_SCHEMA_VERSION,
    PixelEvidence,
    _dct_matrix,
    dct_features,
    extract_pixel_evidence,
    is_available,
    perceptual_hash,
)

if TYPE_CHECKING:
    from pathlib import Path

FAMILIES = ("dct", "fft", "noise", "ela", "gradient", "color")


def _textured(path: Path, size: tuple[int, int] = (192, 160), *, seed: int = 0) -> Path:
    """A textured image. Flat color would make several families degenerate (zero
    residual, empty gradient histogram) and hide a real break."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    ramp = np.linspace(0, 255, size[0], dtype=np.uint8)[None, :, None]
    Image.fromarray(np.clip(base // 2 + ramp // 2, 0, 255).astype(np.uint8)).save(path)
    return path


class TestFamilies:
    def test_every_family_is_measured_on_a_textured_image(self, tmp_path: Path):
        evidence = extract_pixel_evidence(_textured(tmp_path / "textured.png"))

        assert evidence.decoded
        for family in FAMILIES:
            assert getattr(evidence, family), family

    def test_the_same_image_measures_the_same_twice(self, tmp_path: Path):
        """Determinism is what makes these comparable across runs and machines; the
        residual is computed in row chunks, which is exactly the kind of optimization
        that can perturb the last bits."""
        path = _textured(tmp_path / "textured.png")

        first, second = extract_pixel_evidence(path), extract_pixel_evidence(path)

        assert dataclasses.asdict(first) == dataclasses.asdict(second)

    def test_source_dimensions_survive_the_downscale(self, tmp_path: Path):
        """The recorded size is the SOURCE size, read before the 2048px cap. Recording
        the analysed size instead would still pass every statistic check, because
        those run on the downscaled array either way."""
        path = tmp_path / "oversize.png"
        Image.fromarray(np.zeros((80, 3000, 3), dtype=np.uint8)).save(path)

        assert extract_pixel_evidence(path).decode == {"width": 3000, "height": 80}

    def test_selected_dct_coefficients_match_the_full_transform(self):
        gray = np.random.default_rng(7).uniform(0, 255, (24, 32)).astype(np.float32)
        basis = _dct_matrix(np)
        blocks = gray.reshape(3, 8, 4, 8).swapaxes(1, 2)
        full = np.einsum("ij,abjk,lk->abil", basis, blocks, basis)
        bins = np.linspace(-20.5, 20.5, 22)
        expected = [
            np.histogram(full[:, :, row, column].ravel(), bins=bins)[0].tolist() for row, column in AC_POSITIONS
        ]

        assert dct_features(gray)["dct_ac_hist"] == expected

    def test_perceptual_hash_matches_the_full_transform(self):
        gray = np.random.default_rng(8).uniform(0, 255, (32, 32)).astype(np.float32)
        basis = _dct_matrix(np, 32)
        coefficients = basis @ gray @ basis.T
        low = coefficients[:8, :8].ravel()[1:]
        bits = low > np.median(low)
        expected = f"{int(''.join('1' if bit else '0' for bit in bits), 2):016x}"

        assert perceptual_hash(gray) == expected


class TestDegenerateInputs:
    def test_undecodable_file_reports_the_error_and_stays_empty(self, tmp_path: Path):
        path = tmp_path / "broken.png"
        path.write_bytes(b"not an image")

        evidence = extract_pixel_evidence(path, artifacts=True)

        assert evidence.decoded is False
        assert "error" in evidence.decode
        assert all(getattr(evidence, family) == {} for family in FAMILIES)
        assert evidence.artifacts == {}

    def test_image_too_small_for_a_family_leaves_it_empty(self, tmp_path: Path):
        """8x8 is below the FFT's 32px floor but at the block DCT's. A consumer must
        not assume a fixed feature width, so the narrow case is pinned."""
        path = tmp_path / "tiny.png"
        Image.fromarray(np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)).save(path)

        evidence = extract_pixel_evidence(path)

        assert evidence.decoded is True
        assert evidence.fft == {}
        assert evidence.color != {}

    def test_a_failing_family_does_not_lose_the_others(self, tmp_path: Path, monkeypatch):
        path = _textured(tmp_path / "textured.png")

        def boom(*args, **kwargs):
            raise ValueError(f"family failed for {path}")

        monkeypatch.setattr("pagedmark.pixel_evidence.color_features", boom)
        evidence = extract_pixel_evidence(path)

        assert evidence.color == {"error": "ValueError"}
        assert evidence.status == "partial"
        assert evidence.to_dict()["status"] == "partial"
        assert evidence.dct != {}
        assert evidence.gradient != {}


class TestArtifactsAreOptIn:
    """The artifacts identify the source image -- a thumbnail is a picture, a
    perceptual hash matches one. Everything else is a scalar or a fixed-length
    histogram. That difference in kind is the reason for the flag, so the flag is
    what these guard."""

    def test_off_by_default(self, tmp_path: Path):
        assert extract_pixel_evidence(_textured(tmp_path / "textured.png")).artifacts == {}

    def test_on_request_it_returns_the_spatial_layer(self, tmp_path: Path):
        evidence = extract_pixel_evidence(_textured(tmp_path / "textured.png"), artifacts=True)

        assert set(evidence.artifacts) == {"phash", "thumbnail_jpeg_b64", "ela_map", "noise_residual", "fft_phase"}
        assert len(evidence.artifacts["phash"]) == 16

    def test_the_thumbnail_is_a_readable_image_of_the_source(self, tmp_path: Path):
        """Stated plainly because it is the privacy claim: this field reconstructs
        the picture, at 128px."""
        import base64
        import io

        evidence = extract_pixel_evidence(_textured(tmp_path / "textured.png"), artifacts=True)

        with Image.open(io.BytesIO(base64.b64decode(evidence.artifacts["thumbnail_jpeg_b64"]))) as thumb:
            assert max(thumb.size) <= 128

    def test_a_different_image_hashes_differently(self, tmp_path: Path):
        one = extract_pixel_evidence(_textured(tmp_path / "a.png", seed=1), artifacts=True)
        two = extract_pixel_evidence(_textured(tmp_path / "b.png", seed=2), artifacts=True)

        assert one.artifacts["phash"] != two.artifacts["phash"]

    def test_the_statistics_carry_no_array_payloads(self, tmp_path: Path):
        """Without the flag nothing array-shaped may appear: that is what makes the
        default set aggregates rather than content."""
        evidence = extract_pixel_evidence(_textured(tmp_path / "textured.png"))

        for family in FAMILIES:
            for key, value in getattr(evidence, family).items():
                assert isinstance(value, (int, float, str, list)), (family, key)
                if isinstance(value, list):
                    for item in value:
                        assert isinstance(item, (int, float, list)), (family, key)


def test_is_available_reports_the_optional_dependency():
    assert is_available() is True  # the test environment installs the pixels extra


def test_evidence_is_frozen(tmp_path: Path):
    evidence = extract_pixel_evidence(_textured(tmp_path / "textured.png"))
    assert isinstance(evidence, PixelEvidence)
    with pytest.raises(dataclasses.FrozenInstanceError):
        evidence.decode = {}  # type: ignore[misc]


def test_transport_contract_is_versioned_json_and_omits_path(tmp_path: Path):
    import json

    evidence = extract_pixel_evidence(_textured(tmp_path / "textured.png"), timings=True)
    payload = evidence.to_dict()

    assert payload["schema_version"] == PIXEL_EVIDENCE_SCHEMA_VERSION == 1
    assert payload["status"] == "complete"
    assert "path" not in payload
    assert payload["timing_ms"]["total"] >= 0
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


@pytest.mark.parametrize("schema_version", [2, True, 1.0])
def test_transport_rejects_unsupported_output_schema(tmp_path: Path, schema_version: object):
    evidence = extract_pixel_evidence(_textured(tmp_path / "textured.png"))

    with pytest.raises(ValueError, match="Unsupported pixel evidence schema"):
        evidence.to_dict(schema_version=schema_version)  # type: ignore[arg-type]


def test_decode_error_transport_does_not_leak_the_local_path(tmp_path: Path):
    import json

    path = tmp_path / "broken.png"
    path.write_bytes(b"not an image")

    payload = extract_pixel_evidence(path).to_dict()

    assert payload["status"] == "error"
    assert payload["decode"]["error"] == "UnidentifiedImageError"
    assert str(path) not in json.dumps(payload)


def test_timings_are_opt_in(tmp_path: Path):
    path = _textured(tmp_path / "textured.png")

    assert extract_pixel_evidence(path).timing_ms == {}
    assert set(extract_pixel_evidence(path, artifacts=True, timings=True).timing_ms) == {
        "decode",
        "noise",
        "fft",
        "ela",
        "dct",
        "gradient",
        "color",
        "full_artifacts",
        "total",
    }


def test_pixel_decode_registers_optional_heif_opener(tmp_path: Path, monkeypatch):
    registered = False

    def mark_registered():
        nonlocal registered
        registered = True

    monkeypatch.setattr("pagedmark.image_io._register_heif", mark_registered)

    extract_pixel_evidence(_textured(tmp_path / "textured.png"))

    assert registered is True
