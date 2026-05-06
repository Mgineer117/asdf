"""Maze env. Navigates a corridor maze to a goal cell."""

from __future__ import annotations

from typing import Literal

from gridworld.env import FLOOR, GOAL, LAVA, GridEnv
from gridworld.maps import MAZE_G_MAPS, MAZE_MAPS


class Maze(GridEnv):
    def __init__(
        self,
        grid_type: str,
        max_steps: int,
        goal_conditioned: bool = False,
        tile_size: int = 16,
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
    ):
        maps = MAZE_G_MAPS if goal_conditioned else MAZE_MAPS
        if grid_type not in maps:
            raise ValueError(f"Unknown grid_type {grid_type!r} (choose from {list(maps)})")
        super().__init__(
            ascii_map=maps[grid_type],
            max_steps=max_steps,
            tile_size=tile_size,
            render_mode=render_mode,
        )

    def cell_effects(self):
        # Maze: floor gives a small bonus and the agent walks across it
        # (no termination). Lava and goal behave as defaults.
        return {
            GOAL: (1.0, True, True),
            LAVA: (-1.0, True, True),
            FLOOR: (0.01, False, True),
        }
