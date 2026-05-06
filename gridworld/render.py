"""Tiny int-grid -> RGB renderer.

Replaces the heavyweight per-cell coordinate-fill rendering of the old
`utils/rendering.py`. We just paint solid color tiles using numpy slicing.
"""

from __future__ import annotations

import numpy as np

from gridworld.env import CELL_COLORS


def grid_to_rgb(grid: np.ndarray, tile_size: int = 16) -> np.ndarray:
    """Render a 2D int grid to an (H*tile, W*tile, 3) uint8 image.

    Parameters
    ----------
    grid : (W, H) int ndarray
        Integer cell-type grid (as produced by `GridEnv.get_grid()[..., 0]`).
    tile_size : int
        Pixel size of a single grid cell in the output image.
    """
    if grid.ndim == 3:
        grid = grid[..., 0]
    w, h = grid.shape
    img = np.full((h * tile_size, w * tile_size, 3), 200, dtype=np.uint8)
    for x in range(w):
        for y in range(h):
            color = CELL_COLORS.get(int(grid[x, y]), (200, 200, 200))
            img[
                y * tile_size : (y + 1) * tile_size,
                x * tile_size : (x + 1) * tile_size,
            ] = color
    return img
