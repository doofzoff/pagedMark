"""Unit tests for the sliding-window tiled-diffusion helpers (no GPU/model).

``tiling`` is pure numpy/PIL: the geometry (``plan_tiles`` / ``_axis_positions``),
the feather window (``feather_weights``), and the blend loop (``run_tiled``) are all
exercised here with a plain callable standing in for the diffusion pass, so the
seam-free reconstruction and the tile layout are guarded without loading SDXL.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pagedmark._internal.tiling import (
    Tile,
    _axis_positions,
    feather_weights,
    plan_tiles,
    run_tiled,
)


class TestAxisPositions:
    def test_single_tile_when_length_fits(self):
        assert _axis_positions(800, 1024, 128) == [0]
        assert _axis_positions(1024, 1024, 128) == [0]

    def test_two_tiles_last_flush_to_edge(self):
        # 1500 wide, tile 1024, step 1024-128=896: starts [0], then last = 1500-1024=476.
        assert _axis_positions(1500, 1024, 128) == [0, 476]

    def test_uniform_step_then_flush(self):
        # 3000, tile 1024, step 896 -> 0, 896, 1792; last = 1976 appended (2688 would
        # overrun). The regular range stops at 1792 (next 2688 > 1976), so flush adds 1976.
        assert _axis_positions(3000, 1024, 128) == [0, 896, 1792, 1976]

    def test_no_duplicate_when_range_already_hits_edge(self):
        # length-tile divisible by step: the last range entry already equals the edge.
        # 1024*2-128 = 1920 width, step 896, last = 896 -> range gives [0, 896]; no dup.
        assert _axis_positions(1920, 1024, 128) == [0, 896]

    def test_overlap_at_least_tile_still_progresses(self):
        # Pathological overlap >= tile is clamped to tile-1 so step stays >= 1.
        positions = _axis_positions(2000, 1024, 5000)
        assert positions[0] == 0
        assert positions[-1] == 2000 - 1024
        assert positions == sorted(positions)

    def test_invalid_tile_raises(self):
        with pytest.raises(ValueError, match="tile must be positive"):
            _axis_positions(100, 0, 10)


class TestPlanTiles:
    def test_single_tile_for_small_image(self):
        tiles = plan_tiles(800, 600, 1024, 128)
        assert tiles == [Tile(0, 0, 800, 600)]

    def test_grid_is_row_major_and_uniform_size(self):
        tiles = plan_tiles(1500, 1500, 1024, 128)
        # 2x2 grid: xs/ys both [0, 476].
        assert [(t.x, t.y) for t in tiles] == [(0, 0), (476, 0), (0, 476), (476, 476)]
        # Every tile is exactly tile_size (uniform -> SDXL-friendly).
        assert all((t.width, t.height) == (1024, 1024) for t in tiles)

    def test_tiles_cover_the_full_canvas(self):
        width, height = 2600, 1800
        tiles = plan_tiles(width, height, 1024, 128)
        covered = np.zeros((height, width), dtype=bool)
        for t in tiles:
            covered[t.y : t.y + t.height, t.x : t.x + t.width] = True
        assert covered.all()


class TestFeatherWeights:
    def test_shape_matches_tile(self):
        assert feather_weights(64, 48, 8).shape == (48, 64)

    def test_strictly_positive(self):
        assert (feather_weights(64, 64, 16) > 0).all()

    def test_interior_higher_than_edge(self):
        w = feather_weights(64, 64, 16)
        assert w[32, 32] == pytest.approx(1.0)
        # The corner sits in the taper, so it is well below the interior.
        assert w[0, 0] < w[32, 32]

    def test_symmetric(self):
        w = feather_weights(64, 64, 16)
        assert np.allclose(w, w[::-1, :])
        assert np.allclose(w, w[:, ::-1])

    def test_zero_overlap_is_flat(self):
        # No taper requested -> a flat window of ones.
        assert np.allclose(feather_weights(32, 32, 0), 1.0)


class TestRunTiled:
    def test_identity_generate_reconstructs_image(self):
        # A blend of identical (unchanged) tiles must reproduce the input exactly,
        # regardless of overlap -- the feather weights are a partition-of-unity once
        # normalised. This is the seam-free guarantee.
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 256, size=(1500, 1300, 3), dtype=np.uint8)
        image = Image.fromarray(arr)

        out = run_tiled(lambda tile: tile, image, tile_size=512, overlap=64)

        assert out.size == image.size
        assert np.abs(np.asarray(out, dtype=np.int16) - arr.astype(np.int16)).max() <= 1

    def test_generate_called_once_per_tile(self):
        calls: list[tuple[int, int]] = []

        def generate(tile: Image.Image) -> Image.Image:
            calls.append(tile.size)
            return tile

        image = Image.new("RGB", (1500, 1500), (120, 130, 140))
        run_tiled(generate, image, tile_size=1024, overlap=128)

        assert len(calls) == len(plan_tiles(1500, 1500, 1024, 128)) == 4

    def test_single_tile_path_for_small_image(self):
        image = Image.new("RGB", (300, 200), (10, 20, 30))
        out = run_tiled(lambda tile: tile, image, tile_size=1024, overlap=128)
        assert out.size == (300, 200)
        assert np.asarray(out)[0, 0].tolist() == [10, 20, 30]

    def test_mismatched_generate_output_is_resized_back(self):
        # A pipeline that rounds dims to the latent grid returns a slightly different
        # size; run_tiled must resize it back so the blend buffers line up.
        def generate(tile: Image.Image) -> Image.Image:
            w, h = tile.size
            return tile.resize((w - w % 8, h - h % 8), Image.Resampling.LANCZOS)

        image = Image.new("RGB", (1500, 1100), (200, 100, 50))
        out = run_tiled(generate, image, tile_size=1024, overlap=128)
        assert out.size == (1500, 1100)
