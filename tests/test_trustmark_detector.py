"""Tests for the optional Adobe TrustMark detector.

TrustMark is an optional dependency (extra ``trustmark``) that downloads model
weights on first use, so the decode path is only exercised when it is installed
(mirrors the imwatermark handling). The always-on test pins the graceful
absent/error behaviour: detect must return None, never raise.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pagedmark import trustmark_detector
from pagedmark.identify import identify
from pagedmark.trustmark_detector import detect_trustmark, is_available

_OFFICIAL_FIXTURE = (
    Path(__file__).resolve().parent.parent / "data" / "fixtures" / "provenance" / "adobe-trustmark-p.png"
)
_OFFICIAL_FIXTURE_SHA256 = "e58c5825ed7e5d9fb04710ea541b61bd55879cad65554c0d46260aa24b3d0755"


class _FakeDecoder:
    """A TrustMark decoder whose successive ``decode`` calls return scripted
    ``(secret, present, schema)`` tuples -- the first for the original image, the
    second for the re-encoded copy used by the false-positive durability gate."""

    def __init__(self, *results: tuple[str, bool, int]):
        self._results = list(results)
        self.calls = 0

    def decode(self, _img: object, mode: str = "binary") -> tuple[str, bool, int]:
        assert mode == "binary"
        result = self._results[min(self.calls, len(self._results) - 1)]
        self.calls += 1
        return result


def test_detect_never_raises(tmp_clean_png: Path):
    # Whether or not trustmark is installed, a clean image must yield None
    # (no watermark) without raising. When absent, the import guard returns None.
    assert detect_trustmark(tmp_clean_png) is None


def test_unreadable_file_returns_none(tmp_path: Path):
    bad = tmp_path / "not_an_image.txt"
    bad.write_bytes(b"not an image")
    assert detect_trustmark(bad) is None


@pytest.mark.skipif(not is_available(), reason="trustmark not installed")
def test_clean_image_reports_no_watermark(tmp_clean_png: Path):
    # With the decoder present, an un-watermarked image must report absent.
    assert detect_trustmark(tmp_clean_png) is None


class TestFalsePositiveGate:
    """The re-encode durability gate keeps real (durable) TrustMarks and drops
    BCH false positives that collapse under a mild JPEG round-trip."""

    @pytest.fixture(autouse=True)
    def _force_available(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(trustmark_detector, "is_available", lambda: True)

    def _patch_decoder(self, monkeypatch: pytest.MonkeyPatch, decoder: _FakeDecoder) -> None:
        monkeypatch.setattr(trustmark_detector, "_decoder", lambda: decoder)

    def test_durable_watermark_survives_and_is_reported(self, monkeypatch, tmp_clean_png: Path):
        decoder = _FakeDecoder(("secret", True, 2), ("secret", True, 2))
        self._patch_decoder(monkeypatch, decoder)
        result = detect_trustmark(tmp_clean_png)
        assert result == "Adobe TrustMark (variant P, schema 2)"
        assert decoder.calls == 2  # original + re-encode

    def test_false_positive_collapsing_on_reencode_is_dropped(self, monkeypatch, tmp_clean_png: Path):
        # Present on the original, absent after re-encode -> content-noise FP.
        decoder = _FakeDecoder(("01", True, 2), ("", False, -1))
        self._patch_decoder(monkeypatch, decoder)
        assert detect_trustmark(tmp_clean_png) is None

    def test_schema_drift_on_reencode_is_dropped(self, monkeypatch, tmp_clean_png: Path):
        # Present both times but the schema changes -> not a stable watermark.
        decoder = _FakeDecoder(("01", True, 2), ("01", True, 3))
        self._patch_decoder(monkeypatch, decoder)
        assert detect_trustmark(tmp_clean_png) is None

    def test_payload_drift_on_reencode_is_dropped(self, monkeypatch, tmp_clean_png: Path):
        decoder = _FakeDecoder(("first", True, 2), ("second", True, 2))
        self._patch_decoder(monkeypatch, decoder)
        assert detect_trustmark(tmp_clean_png) is None

    def test_weak_schema_three_is_dropped(self, monkeypatch, tmp_clean_png: Path):
        decoder = _FakeDecoder(("secret", True, 3))
        self._patch_decoder(monkeypatch, decoder)
        assert detect_trustmark(tmp_clean_png) is None
        assert decoder.calls == 1

    def test_absent_skips_reencode(self, monkeypatch, tmp_clean_png: Path):
        decoder = _FakeDecoder(("", False, -1))
        self._patch_decoder(monkeypatch, decoder)
        assert detect_trustmark(tmp_clean_png) is None
        assert decoder.calls == 1  # no second decode when the first is absent


@pytest.mark.skipif(not is_available(), reason="trustmark not installed")
def test_official_adobe_variant_p_fixture():
    assert detect_trustmark(_OFFICIAL_FIXTURE) == "Adobe TrustMark (variant P, schema 1)"


@pytest.mark.skipif(not is_available(), reason="trustmark not installed")
def test_official_adobe_variant_p_fixture_is_provenance_not_ai():
    report = identify(_OFFICIAL_FIXTURE, check_visible=False)
    assert report.is_ai_generated is None
    assert report.ai_from_metadata is False
    assert any(signal.name == "trustmark" for signal in report.signals)


def test_official_adobe_variant_p_fixture_digest():
    assert hashlib.sha256(_OFFICIAL_FIXTURE.read_bytes()).hexdigest() == _OFFICIAL_FIXTURE_SHA256
