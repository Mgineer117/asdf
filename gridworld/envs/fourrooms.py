"""FourRooms env. Reaches goal in a 4-room layout."""

from __future__ import annotations

from typing import Literal

from gridworld.env import GridEnv
from gridworld.maps import FOURROOMS_G_MAPS, FOURROOMS_MAPS


class FourRooms(GridEnv):
    def __init__(
        self,
        grid_type: str,
        max_steps: int,
        goal_conditioned: bool = False,
        tile_size: int = 16,
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
    ):
        maps = FOURROOMS_G_MAPS if goal_conditioned else FOURROOMS_MAPS
        if grid_type not in maps:
            raise ValueError(f"Unknown grid_type {grid_type!r} (choose from {list(maps)})")
        super().__init__(
            ascii_map=maps[grid_type],
            max_steps=max_steps,
            tile_size=tile_size,
            render_mode=render_mode,
        )
