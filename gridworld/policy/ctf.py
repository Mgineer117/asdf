"""Lightweight scripted policies for the single-agent CTF env.

Two policies are provided:

* :class:`RandomPolicy` — uniform random action.
* :class:`AStarPolicy` — A* shortest-path to the goal/flag, with optional
  per-step randomness.

Both consume the dict observation produced by `GridEnv` (keys
``achieved_goal`` and ``desired_goal``).
"""

from __future__ import annotations

from heapq import heappop, heappush
from typing import NamedTuple

import numpy as np

from gridworld.env import OBSTACLE, WALL


class _Node(NamedTuple):
    f: int
    g: int
    pos: tuple[int, int]
    parent: "tuple | None"


def _manhattan(a, b) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def a_star(
    start: tuple[int, int],
    goal: tuple[int, int],
    grid: np.ndarray,
    blocked: tuple[int, ...] = (WALL, OBSTACLE),
) -> list[tuple[int, int]]:
    """Plain A* on a 2D int grid.

    `grid` is shape (W, H). Returns the path as a list of (x, y) tuples
    (including both endpoints), or ``[start]`` if no path exists.
    """
    w, h = grid.shape
    blocked_set = set(blocked)

    open_q: list[_Node] = []
    heappush(open_q, _Node(_manhattan(start, goal), 0, tuple(start), None))
    seen: dict[tuple[int, int], int] = {tuple(start): 0}

    while open_q:
        node = heappop(open_q)
        if node.pos == tuple(goal):
            path = []
            cur: tuple | None = node
            while cur is not None:
                path.append(cur.pos)
                cur = cur.parent
            return list(reversed(path))

        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = node.pos[0] + dx, node.pos[1] + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            cell = int(grid[nx, ny])
            if cell in blocked_set and (nx, ny) != tuple(goal):
                continue
            ng = node.g + 1
            if (nx, ny) in seen and seen[(nx, ny)] <= ng:
                continue
            seen[(nx, ny)] = ng
            heappush(open_q, _Node(ng + _manhattan((nx, ny), goal), ng, (nx, ny), node))
    return [tuple(start)]


# Action ids match `GridEnv.ACTION_TO_DELTA`: 0=left,1=up,2=right,3=down.
_DELTA_TO_ACTION = {(0, -1): 0, (-1, 0): 1, (0, 1): 2, (1, 0): 3}


def _delta_to_action(dx: int, dy: int) -> int:
    return _DELTA_TO_ACTION.get((dx, dy), 0)


class RandomPolicy:
    """Uniform random over the 4-action space."""

    def __init__(self, n_actions: int = 4, seed: int | None = None):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def act(self, obs: dict, env=None) -> int:
        return int(self.rng.integers(0, self.n_actions))


class AStarPolicy:
    """Greedy A* policy toward `desired_goal`, with optional noise.

    Pass the env to ``act`` so the policy can read the current grid; this
    keeps the policy stateless across episodes.
    """

    def __init__(self, randomness: float = 0.0, seed: int | None = None):
        self.randomness = float(randomness)
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def act(self, obs: dict, env) -> int:
        if self.randomness > 0 and self.rng.random() < self.randomness:
            return int(self.rng.integers(0, 4))

        start = tuple(int(v) for v in obs["achieved_goal"])
        goal = tuple(int(v) for v in obs["desired_goal"])
        grid = env.get_grid()[..., 0]
        path = a_star(start, goal, grid)
        if len(path) < 2:
            return int(self.rng.integers(0, 4))
        nx, ny = path[1]
        return _delta_to_action(nx - start[0], ny - start[1])


HEURISTIC_POLICIES = {
    "random": RandomPolicy,
    "astar": AStarPolicy,
}
