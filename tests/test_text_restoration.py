"""Verified-text manifest and compositor tests without model downloads."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from pagedmark._internal.text_restoration import (
    FIDELITY_BLEND_ALPHA,
    VerifiedTextLine,
    blend_fidelity_anchor,
    load_verified_text_manifest,
    restore_verified_text,
    source_pixel_sha256,
)


def _manifest(image: Image.Image) -> dict[str, object]:
    return {
        "schema_version": 1,
        "verified": True,
        "source_pixel_sha256": source_pixel_sha256(image),
        "width": image.width,
        "height": image.height,
        "lines": [
            {
                "box": [8, 8, 40, 24],
                "text": "Exact text",
                "script": "alphabetic",
                "angle": 0.0,
            }
        ],
    }


def test_pixel_hash_ignores_container_metadata(tmp_path) -> None:
    image = Image.new("RGB", (48, 32), (10, 20, 30))
    plain = tmp_path / "plain.png"
    tagged = tmp_path / "tagged.png"
    image.save(plain)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("note", "different container bytes")
    image.save(tagged, pnginfo=metadata)

    with Image.open(plain) as left, Image.open(tagged) as right:
        assert plain.read_bytes() != tagged.read_bytes()
        assert source_pixel_sha256(left) == source_pixel_sha256(right)


def test_verified_manifest_is_bound_to_source_pixels(tmp_path) -> None:
    source = Image.new("RGB", (48, 32), (10, 20, 30))
    path = tmp_path / "lines.json"
    path.write_text(json.dumps(_manifest(source)), encoding="utf-8")

    loaded = load_verified_text_manifest(path, source)

    assert loaded.width == 48
    assert loaded.height == 32
    assert loaded.lines == (VerifiedTextLine((8, 8, 40, 24), "Exact text", "alphabetic", 0.0),)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"verified": False}, "verified=true"),
        ({"source_pixel_sha256": "0" * 64}, "does not match"),
        ({"width": 49}, "dimensions"),
        ({"lines": []}, "non-empty"),
    ],
)
def test_manifest_rejects_unverified_or_unbound_input(tmp_path, mutation, message) -> None:
    source = Image.new("RGB", (48, 32), (10, 20, 30))
    payload = _manifest(source)
    payload.update(mutation)
    path = tmp_path / "lines.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_verified_text_manifest(path, source)


def test_fidelity_anchor_uses_the_calibrated_rounding() -> None:
    clean = Image.fromarray(np.array([[[1, 2, 3], [100, 150, 200]]], dtype=np.uint8))
    donor = Image.fromarray(np.array([[[255, 254, 253], [200, 100, 50]]], dtype=np.uint8))

    result = np.asarray(blend_fidelity_anchor(clean, donor))
    expected = np.rint(
        np.asarray(clean, dtype=np.float32) * (1.0 - FIDELITY_BLEND_ALPHA)
        + np.asarray(donor, dtype=np.float32) * FIDELITY_BLEND_ALPHA
    ).astype(np.uint8)

    assert np.array_equal(result, expected)


def test_restoration_uses_lama_and_qwen_vae_core(monkeypatch) -> None:
    from pagedmark import region_eraser

    source = np.full((40, 64, 3), 20, dtype=np.uint8)
    source[12:24, 12:44] = 235
    candidate = np.full_like(source, 30)
    candidate[12:24, 12:44] = 150
    donor = np.full_like(source, 40)
    donor[12:24, 12:44] = (210, 220, 230)
    calls: list[np.ndarray] = []

    def fake_erase(image_bgr, mask):
        calls.append(mask.copy())
        output = image_bgr.copy()
        output[mask > 0] = (30, 30, 30)
        return output

    monkeypatch.setattr(region_eraser, "lama_available", lambda: True)
    monkeypatch.setattr(region_eraser, "erase_lama", fake_erase)

    result = restore_verified_text(
        Image.fromarray(source),
        Image.fromarray(candidate),
        Image.fromarray(donor),
        (VerifiedTextLine((8, 8, 48, 28), "Exact text", "alphabetic"),),
    )

    restored = np.asarray(result)
    assert calls
    assert np.all(restored[16, 20] == donor[16, 20])
    assert np.all(restored[0, 0] == candidate[0, 0])
