from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from pagedmark._internal import text_restoration

SCRIPT = Path(__file__).parents[1] / "scripts" / "selective_text_restoration.py"
SPEC = importlib.util.spec_from_file_location("selective_text_restoration", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_normalized_edit_distance_ignores_case_and_whitespace() -> None:
    assert module.normalized_edit_distance(" Sample text ", "sample\ntext") == 0.0


def test_preserve_requires_source_candidate_agreement() -> None:
    assert module.should_preserve_line("clean text", "clean text", 0.9, "clean text", 0.9)
    assert not module.should_preserve_line("clean text", "clean text", 0.9, "damaged", 0.9)


def test_preserve_rejects_unreliable_source_recognition() -> None:
    assert not module.should_preserve_line("expected", "unrelated", 0.9, "unrelated", 0.9)
    assert not module.should_preserve_line("expected", "expected", 0.7, "expected", 0.9)


def test_cjk_recognition_box_excludes_overlapping_neighbor_lines() -> None:
    line = module.TextLine((1281, 650, 2357, 818), "每天都是一个新的机会。", "cjk")

    assert module._recognition_box(line, 2816, 1536) == (1247, 650, 2458, 818)


def test_latin_recognition_box_keeps_context_padding() -> None:
    line = module.TextLine((100, 200, 300, 260), "Sample text", "latin")

    assert module._recognition_box(line, 1000, 1000) == (40, 180, 360, 280)
    assert module._recognition_box(line, 1000, 1000, 0.1) == (40, 192, 360, 268)


def test_verified_lines_cover_each_ground_truth_string() -> None:
    root = Path(__file__).parents[1]
    lines = json.loads((root / "data/evaluations/fidelity/text-lines.json").read_text(encoding="utf-8"))
    ground_truth = json.loads((root / "data/evaluations/fidelity/ground-truth.json").read_text(encoding="utf-8"))

    assert lines.keys() == ground_truth.keys()
    for source, expected in ground_truth.items():
        observed = " ".join(line["text"] for line in lines[source])
        assert module.normalize_text(observed) == module.normalize_text(expected)


def test_group_word_boxes_merges_words_but_not_neighboring_lines() -> None:
    boxes = [(10, 10, 30, 30), (32, 12, 60, 29), (10, 35, 50, 55)]

    assert module.group_word_boxes(boxes) == [(10, 10, 60, 30), (10, 35, 50, 55)]


def test_group_word_boxes_does_not_merge_distant_columns() -> None:
    boxes = [(10, 10, 60, 30), (500, 11, 560, 31)]

    assert module.group_word_boxes(boxes) == boxes


def test_source_glyph_composite_keeps_masked_pixels_exact() -> None:
    source = np.zeros((9, 9, 3), dtype=np.uint8)
    source[:, :] = (220, 180, 40)
    background = np.zeros((9, 9, 3), dtype=np.uint8)
    background[:, :] = (10, 20, 30)
    mask = np.zeros((9, 9), dtype=np.uint8)
    mask[3:6, 3:6] = 255

    result = module.composite_source_glyphs(source, background, mask, feather=0.7)

    np.testing.assert_array_equal(result[3:6, 3:6], source[3:6, 3:6])
    np.testing.assert_array_equal(result[0, 0], background[0, 0])


def test_fresh_silhouette_uses_new_color_instead_of_source_pixels() -> None:
    background = np.zeros((9, 9, 3), dtype=np.uint8)
    background[:, :] = (10, 20, 30)
    mask = np.zeros((9, 9), dtype=np.uint8)
    mask[3:6, 3:6] = 255

    result = text_restoration.composite_fresh_silhouette(background, mask, (220, 180, 40), feather=0)

    assert np.all(result[3:6, 3:6] == (220, 180, 40))
    np.testing.assert_array_equal(result[0, 0], background[0, 0])


def test_fresh_silhouette_antialiasing_softens_binary_edges() -> None:
    background = np.zeros((9, 9, 3), dtype=np.uint8)
    background[:, :] = (10, 20, 30)
    mask = np.zeros((9, 9), dtype=np.uint8)
    mask[3:6, 3:6] = 255

    result = text_restoration.composite_fresh_silhouette(background, mask, (220, 180, 40), feather=1.0)

    assert np.all(result[3, 3] > background[3, 3])
    assert np.all(result[3, 3] < (220, 180, 40))


def test_reconstructed_glyphs_keep_exact_donor_core_and_fresh_edge() -> None:
    donor = np.zeros((9, 9, 3), dtype=np.uint8)
    donor[:, :] = (180, 140, 60)
    background = np.zeros((9, 9, 3), dtype=np.uint8)
    background[:, :] = (10, 20, 30)
    mask = np.zeros((9, 9), dtype=np.uint8)
    mask[3:6, 3:6] = 255

    fresh_edge = text_restoration.composite_fresh_silhouette(background, mask, (220, 180, 40))
    result = module.composite_reconstructed_glyphs(donor, fresh_edge, mask, feather=0.5)

    np.testing.assert_array_equal(result[3:6, 3:6], donor[3:6, 3:6])
    assert np.any(result[2, 3] != fresh_edge[2, 3])
    np.testing.assert_array_equal(result[0, 0], background[0, 0])


def test_source_silhouette_discards_foreground_amplitudes() -> None:
    source = np.full((15, 15, 3), 20, dtype=np.uint8)
    source[5:10, 6:9] = 230
    source[6:9, 7] = 180

    mask = module.source_silhouette_mask(source, (4, 4, 11, 11))

    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) <= {0, 255}
    assert mask[7, 7] == 255
    assert mask[4, 4] == 0


def test_rotated_source_silhouette_excludes_axis_aligned_corners() -> None:
    source = np.full((80, 160, 3), 20, dtype=np.uint8)
    source[10:70, 10:150] = 230

    mask = module.source_silhouette_mask(source, (0, 0, 160, 80), angle=12)

    assert mask[0, 0] == 0
    assert mask[79, 159] == 0


def test_source_box_mask_pads_and_clips_boxes() -> None:
    mask = module.source_box_mask((20, 30), [(1, 2, 11, 10), (25, 15, 30, 20)])

    assert mask.shape == (20, 30)
    assert mask[0, 0] == 255
    assert mask[19, 29] == 255
    assert mask[0, 22] == 0


def test_detect_line_boxes_fails_closed_on_count_mismatch() -> None:
    class Engine:
        def predict(self, _image):
            return [{"rec_scores": [0.9], "rec_boxes": [[10, 10, 30, 30]]}]

    with pytest.raises(module.click.ClickException, match="detected 1 source lines; expected exactly 2"):
        module.detect_line_boxes(Engine(), np.zeros((50, 50, 3), dtype=np.uint8), expected_count=2)


def test_residual_mask_is_limited_to_original_glyph_positions(monkeypatch) -> None:
    from pagedmark._internal import text_restoration

    background = np.zeros((8, 8, 3), dtype=np.uint8)
    original = np.zeros((8, 8), dtype=np.uint8)
    original[3, 3] = 255
    detected = np.zeros((8, 8), dtype=np.uint8)
    detected[3, 3] = 255
    detected[6, 6] = 255
    monkeypatch.setattr(text_restoration, "_foreground_mask", lambda _image, _box: detected)

    residual = module.residual_glyph_mask(background, original, (0, 0, 8, 8))

    assert residual[3, 3] == 255
    assert residual[6, 6] == 0
