"""Minimal single-agent discrete gridworld package.

Public API:
    GridEnv          — base class for new environments (subclass + ASCII map).
    FourRooms / Maze — built-in environments.
    CaptureTheFlag   — single-agent CTF variant.
    maps             — ASCII map registry (FOURROOMS_MAPS, MAZE_MAPS, ...).

Quick start:

    from gridworld import FourRooms
    env = FourRooms(grid_type="v1", max_steps=100)
    obs, info = env.reset()
    obs, r, term, trunc, info = env.step(action_one_hot)
"""

from gridworld.env import (
    AGENT,
    EMPTY,
    FLOOR,
    GOAL,
    LAVA,
    OBSTACLE,
    WALL,
    GridEnv,
)
from gridworld.envs.ctf import CaptureTheFlag
from gridworld.envs.fourrooms import FourRooms
from gridworld.envs.maze import Maze

__all__ = [
    "GridEnv",
    "FourRooms",
    "Maze",
    "CaptureTheFlag",
    "WALL", "EMPTY", "GOAL", "LAVA", "FLOOR", "OBSTACLE", "AGENT",
]
