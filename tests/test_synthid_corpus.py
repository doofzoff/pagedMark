"""Tests for the SynthID corpus ingestion script."""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

# scripts/ is not an installed package; add it to the path for import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import synthid_corpus

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "fixtures" / "provenance"
CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "synthid"
QUALITY_SET = CORPUS_DIR / "full-pipeline-quality.csv"
CURRENT_OPENAI_SAMPLE = CORPUS_DIR / "originals" / "ChatGPT Image May 30, 2026, 10_31_08 AM.png"

EXPECTED_QUALITY_SOURCE_FILENAMES = {
    "ChatGPT Image May 30, 2026, 10_31_08 AM.png",
    "ChatGPT Image May 31, 2026, 02_02_23 PM.png",
    "ChatGPT Image May 31, 2026, 02_03_55 PM.png",
    "Gemini_Generated_Image_3mc4t93mc4t93mc4.png",
    "Gemini_Generated_Image_633uuy633uuy633u.png",
    "Gemini_Generated_Image_akdbeiakdbeiakdb.png",
    "Gemini_Generated_Image_y48j3cy48j3cy48j.png",
}


def _manifest_rows(root: Path) -> list[dict[str, str]]:
    with open(root / "manifest.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_reusable_quality_set_has_expected_inputs_and_valid_hashes() -> None:
    with open(QUALITY_SET, newline="") as f:
        rows = list(csv.DictReader(f))

    # Keep this literal independent of the CSV so deleting a fixture fails.
    assert {row["source_filename"] for row in rows} == EXPECTED_QUALITY_SOURCE_FILENAMES
    for row in rows:
        corpus_path = CORPUS_DIR / row["corpus_path"]
        assert corpus_path.is_file(), corpus_path
        assert hashlib.sha256(corpus_path.read_bytes()).hexdigest() == row["sha256"]


def test_manifest_matches_canonical_originals() -> None:
    rows = _manifest_rows(CORPUS_DIR)
    originals = {path.name for path in (CORPUS_DIR / "originals").iterdir() if path.is_file()}

    assert {row["filename"] for row in rows} == originals
    assert len({row["sha256"] for row in rows}) == len(rows)


@pytest.mark.skipif(not SAMPLES_DIR.exists(), reason="data/fixtures/provenance not present")
class TestIngest:
    @pytest.mark.skipif(not CURRENT_OPENAI_SAMPLE.exists(), reason="current OpenAI SynthID fixture not present")
    def test_ingest_current_openai_flags_synthid_metadata(self, tmp_path: Path):
        runner = CliRunner()
        result = runner.invoke(
            synthid_corpus.cli,
            ["ingest", str(CURRENT_OPENAI_SAMPLE), "--label", "pos", "--root", str(tmp_path)],
        )
        assert result.exit_code == 0, result.output

        rows = _manifest_rows(tmp_path)
        assert len(rows) == 1
        row = rows[0]
        assert row["label"] == "pos"
        assert row["synthid_metadata"] == "yes"
        assert int(row["width"]) > 0
        assert int(row["height"]) > 0
        assert row["filename"] == CURRENT_OPENAI_SAMPLE.name
        # Every label shares one canonical originals/ directory.
        assert (tmp_path / "originals" / row["filename"]).exists()

    def test_ingest_firefly_not_flagged(self, tmp_path: Path):
        runner = CliRunner()
        runner.invoke(
            synthid_corpus.cli,
            ["ingest", str(SAMPLES_DIR / "firefly-1.png"), "--label", "neg", "--root", str(tmp_path)],
        )
        rows = _manifest_rows(tmp_path)
        assert len(rows) == 1
        assert rows[0]["synthid_metadata"] == ""  # Adobe signs C2PA but not SynthID

    def test_ingest_dedupes_by_sha256(self, tmp_path: Path):
        runner = CliRunner()
        args = ["ingest", str(SAMPLES_DIR / "chatgpt-1.png"), "--label", "pos", "--root", str(tmp_path)]
        runner.invoke(synthid_corpus.cli, args)
        runner.invoke(synthid_corpus.cli, args)  # second time: duplicate
        assert len(_manifest_rows(tmp_path)) == 1

    def test_status_on_empty_corpus(self, tmp_path: Path):
        runner = CliRunner()
        result = runner.invoke(synthid_corpus.cli, ["status", "--root", str(tmp_path)])
        assert result.exit_code == 0
        assert "empty" in result.output.lower()
