"""Single-agent capture-the-flag.

The agent navigates a field with scattered obstacles to reach the flag.
Optionally pair with `gridworld.policy.ctf` for scripted reference policies.
"""

from __future__ import annotations

from typing import Literal

from gridworld.env import GOAL, OBSTACLE, GridEnv
from gridworld.maps import CTF_MAPS


class CaptureTheFlag(GridEnv):
    def __init__(
        self,
        grid_type: str = "v1",
        max_steps: int = 200,
        tile_size: int = 16,
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
    ):
        if grid_type not in CTF_MAPS:
            raise ValueError(f"Unknown grid_type {grid_type!r} (choose from {list(CTF_MAPS)})")
        super().__init__(
            ascii_map=CTF_MAPS[grid_type],
            max_steps=max_steps,
            tile_size=tile_size,
            render_mode=render_mode,
        )

    def cell_effects(self):
        return {
            GOAL: (1.0, True, True),         # flag captured
            OBSTACLE: (-0.1, False, False),  # bumping costs a small penalty
        }
