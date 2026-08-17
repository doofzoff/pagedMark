"""Sliding-window tiled diffusion for large images.

The global stage denoises the WHOLE image in one forward pass, so it OOMs on a
GPU above ~2K (issue #10). Tiling splits the image into overlapping tiles -- each
kept near the ~1024 training size -- regenerates each tile independently, and
feather-blends the overlaps. The result retains the input's native dimensions
without an explicit ``--max-resolution`` downscale, but it is not pixel-lossless
because every tile is regenerated.

The geometry (``plan_tiles``) and the blend weighting (``feather_weights``) are
pure functions, unit-tested without the diffusion model. ``run_tiled`` is the
orchestration loop; it takes a ``generate_tile`` callable (one img2img/ControlNet
pass on a single PIL tile) so it stays decoupled from the pipeline internals.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray
    from PIL import Image as PILImage

logger = logging.getLogger(__name__)

# Strictly-positive floor for the accumulated blend weights so a region covered
# by a single feathered tile edge (an image corner, no neighbour to blend with)
# never divides by zero.
_WEIGHT_EPS = 1e-3


class Tile(NamedTuple):
    """A tile crop box in the source image: top-left ``(x, y)`` + ``width``/``height``."""

    x: int
    y: int
    width: int
    height: int


def _axis_positions(length: int, tile: int, overlap: int) -> list[int]:
    """Tile start offsets along one axis, last tile flush to the far edge.

    Every interior tile is exactly ``tile`` long; the final tile is pulled back
    to ``length - tile`` so it ends exactly at the edge (it simply overlaps its
    predecessor a little more). Keeping all tiles the same size is what lets the
    diffusion pass run at SDXL's preferred dimension on every tile.
    """
    if tile <= 0:
        raise ValueError(f"tile must be positive, got {tile}")
    if length <= tile:
        return [0]
    # Guarantee forward progress even on a pathological overlap >= tile.
    overlap = min(max(overlap, 0), tile - 1)
    step = tile - overlap
    positions = list(range(0, length - tile + 1, step))
    last = length - tile
    if positions[-1] != last:
        positions.append(last)
    return positions


def plan_tiles(width: int, height: int, tile_size: int, overlap: int) -> list[Tile]:
    """Lay out a grid of overlapping tiles covering ``width`` x ``height``.

    All tiles are ``min(tile_size, width)`` x ``min(tile_size, height)`` (uniform
    size; the image itself when it fits in one tile). Returned in row-major order.
    """
    xs = _axis_positions(width, tile_size, overlap)
    ys = _axis_positions(height, tile_size, overlap)
    tile_w = min(tile_size, width)
    tile_h = min(tile_size, height)
    return [Tile(x, y, tile_w, tile_h) for y in ys for x in xs]


def feather_weights(width: int, height: int, overlap: int) -> NDArray[Any]:
    """A 2D blend window: ~1 in the interior, ramping down toward each edge.

    Separable linear taper over ``overlap`` pixels from every edge (capped at
    half the tile so short tiles still taper symmetrically). Strictly positive
    everywhere, so the normalized blend is well-defined even at an image corner
    that only one tile covers.
    """
    import numpy as np

    def ramp(n: int) -> NDArray[Any]:
        w = np.ones(n, dtype=np.float32)
        if overlap > 0 and n > 1:
            ramp_len = min(overlap, max(1, n // 2))
            taper = (np.arange(ramp_len, dtype=np.float32) + 1.0) / (ramp_len + 1.0)
            w[:ramp_len] = taper
            w[n - ramp_len :] = taper[::-1]
        return w

    weights = np.outer(ramp(height), ramp(width))
    np.maximum(weights, _WEIGHT_EPS, out=weights)
    return weights


def run_tiled(
    generate_tile: Callable[[PILImage.Image], PILImage.Image],
    image: PILImage.Image,
    tile_size: int,
    overlap: int,
    set_progress: Callable[[str], None] | None = None,
) -> PILImage.Image:
    """Tile ``image``, run ``generate_tile`` per tile, and feather-blend the result.

    ``generate_tile`` runs one diffusion pass on a single RGB PIL tile and returns
    the regenerated tile (the ControlNet control image is built per tile inside it,
    so each tile gets its own edge map). A pass that rounds dimensions to the latent
    grid is resized back to the exact tile size before blending.
    """
    import numpy as np
    from PIL import Image

    width, height = image.size
    tiles = plan_tiles(width, height, tile_size, overlap)
    accum = np.zeros((height, width, 3), dtype=np.float32)
    weight_sum = np.zeros((height, width, 1), dtype=np.float32)

    # All tiles share one size (plan_tiles is uniform), so the feather window is
    # loop-invariant -- compute it once.
    weights = feather_weights(tiles[0].width, tiles[0].height, overlap)[:, :, None]

    total = len(tiles)
    for index, tile in enumerate(tiles, start=1):
        if set_progress is not None:
            set_progress(f"Tiled diffusion: tile {index}/{total} at ({tile.x},{tile.y}) {tile.width}x{tile.height}...")
        crop = image.crop((tile.x, tile.y, tile.x + tile.width, tile.y + tile.height))
        result = generate_tile(crop)
        if result.size != (tile.width, tile.height):
            result = result.resize((tile.width, tile.height), Image.Resampling.LANCZOS)
        arr = np.asarray(result.convert("RGB"), dtype=np.float32)
        accum[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width] += arr * weights
        weight_sum[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width] += weights

    blended = accum / np.maximum(weight_sum, _WEIGHT_EPS)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
