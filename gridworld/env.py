"""Minimal single-agent gridworld base class.

Authoring a new environment
---------------------------
Subclass `GridEnv` and either pass a map at construction or set the
`MAP` class attribute (a list of equal-length strings). ASCII vocabulary:

    '#'  wall          (impassable)
    ' '  empty         (free space)
    'g'  goal          (terminates with +1)
    'r'  agent start
    'c'  combined      (one such cell becomes the goal, another the agent)
    'L'  lava          (terminates with -1)
    '@'  floor         (configurable hazard)

Override `cell_effects()` to change rewards or termination behavior.
The default action space is 4-direction discrete (left/up/right/down),
passed as a one-hot vector to keep parity with the historical API.
"""

from __future__ import annotations

import random
from typing import Literal

import numpy as np
import torch
from gymnasium import Env, spaces

# Cell-type integer indices (kept stable for downstream consumers
# that grep `get_grid()` for wall=2, goal=8, agent=10, etc.).
WALL = 2
EMPTY = 1
GOAL = 8
LAVA = 9
FLOOR = 3
OBSTACLE = 13
AGENT = 10

CHAR_TO_CELL = {
    "#": WALL,
    " ": EMPTY,
    "g": GOAL,
    "r": EMPTY,        # 'r' marks an agent spawn; cell itself is empty
    "c": EMPTY,        # 'c' marks combined goal/agent spawn; cell itself is empty
    "L": LAVA,
    "@": FLOOR,
    "o": OBSTACLE,
}

# Action -> (dx, dy) in (row, col) coordinates used internally
ACTION_TO_DELTA = {
    0: (0, -1),   # left  (dy = -1)
    1: (-1, 0),   # up    (dx = -1)
    2: (0, +1),   # right
    3: (+1, 0),   # down
}

# (R, G, B) per cell type for the simple renderer
CELL_COLORS = {
    WALL: (60, 60, 60),
    EMPTY: (240, 240, 240),
    GOAL: (0, 200, 80),
    LAVA: (255, 90, 0),
    FLOOR: (0, 128, 38),
    OBSTACLE: (120, 79, 23),
    AGENT: (0, 77, 255),
}


class GridEnv(Env):
    """Single-agent ASCII-driven gridworld.

    Observation
    -----------
    Dict-like ndarray pair: ``{"achieved_goal": (x, y), "desired_goal": (x, y)}``.

    Action
    ------
    One-hot vector of length 4 (left, up, right, down). ``np.argmax`` is
    applied internally so passing a discrete int also works.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    MAP: list[str] | None = None  # subclasses may set this

    def __init__(
        self,
        ascii_map: list[str] | None = None,
        max_steps: int = 100,
        tile_size: int = 16,
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
    ):
        ascii_map = ascii_map if ascii_map is not None else self.MAP
        if ascii_map is None:
            raise ValueError("GridEnv needs an ASCII map (constructor arg or MAP attr).")

        self.ascii_map: list[str] = list(ascii_map)
        self.max_steps = max_steps
        self.tile_size = tile_size
        self.render_mode = render_mode

        self.height = len(self.ascii_map)
        self.width = len(self.ascii_map[0])
        self.grid_size = (self.width, self.height)

        # Static layout (walls/lava/floor) baked from the ASCII map once.
        self._base_grid: np.ndarray = np.array(
            [[CHAR_TO_CELL.get(ch, EMPTY) for ch in row] for row in self.ascii_map],
            dtype=np.int64,
        )

        # Spawn candidates from 'g'/'c' (goals) and 'r'/'c' (agents).
        self._goal_candidates: list[tuple[int, int]] = self._find_chars("g") + self._find_chars("c")
        self._agent_candidates: list[tuple[int, int]] = self._find_chars("r") + self._find_chars("c")
        if not self._goal_candidates:
            raise ValueError("Map has no 'g' or 'c' (goal) cells.")
        if not self._agent_candidates:
            raise ValueError("Map has no 'r' or 'c' (agent) cells.")

        self.action_space = spaces.Discrete(4)
        self.observation_space = self._build_observation_space()

        # Episode state filled in by reset().
        self.agent_pos: tuple[int, int] = (0, 0)
        self.goal_pos: tuple[int, int] = self._goal_candidates[0]
        self.step_count = 0

    # ------------------------------------------------------------------
    # Hooks subclasses commonly override
    # ------------------------------------------------------------------

    def cell_effects(self) -> dict[int, tuple[float, bool, bool]]:
        """Map cell-int -> (reward, terminate, can_enter).

        Walls are always impassable regardless of this table. Cells without
        an entry behave like empty (reward 0, no terminate, enterable).
        """
        return {
            GOAL: (1.0, True, True),
            LAVA: (-1.0, True, True),
            FLOOR: (0.01, True, False),  # touch-and-die hazard, agent doesn't enter
        }

    def _build_observation_space(self) -> spaces.Space:
        high = np.array([self.width, self.height, self.width, self.height], dtype=np.float32)
        return spaces.Box(low=np.zeros(4, dtype=np.float32), high=high, dtype=np.float32)

    def _get_obs(self) -> dict[str, np.ndarray]:
        return {
            "achieved_goal": np.array(self.agent_pos, dtype=np.float32),
            "desired_goal": np.array(self.goal_pos, dtype=np.float32),
        }

    # ------------------------------------------------------------------
    # gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed, options=options)
        if seed is not None:
            random.seed(seed)

        self.goal_pos = random.choice(self._goal_candidates)
        spawn_pool = [p for p in self._agent_candidates if p != self.goal_pos]
        if not spawn_pool:
            raise RuntimeError("No agent spawn left after placing the goal.")
        self.agent_pos = random.choice(spawn_pool)
        self.step_count = 0

        return self._get_obs(), {"success": False}

    def step(self, action):
        # Accept either a discrete int or one-hot vector. Legacy callers
        # pass one-hot, current callers pass an int — branch on shape.
        arr = np.atleast_1d(np.asarray(action))
        action = int(np.argmax(arr)) if arr.size > 1 else int(arr.item())
        self.step_count += 1

        dx, dy = ACTION_TO_DELTA[action]
        ax, ay = self.agent_pos
        target = (ax + dx, ay + dy)

        reward = 0.0
        terminated = False
        info = {"success": False}

        target_cell = self._cell_at(*target)

        if target_cell == WALL:
            pass  # blocked; stay in place
        else:
            effects = self.cell_effects().get(target_cell)
            if effects is None:
                self.agent_pos = target
            else:
                r, term, enter = effects
                reward += r
                terminated = term
                info["success"] = bool(term and r > 0)
                if enter:
                    self.agent_pos = target

        truncated = self.step_count >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        from gridworld.render import grid_to_rgb
        return grid_to_rgb(self.get_grid()[..., 0], self.tile_size)

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Helpers used by external code (algorithms/, utils/, etc.)
    # ------------------------------------------------------------------

    def get_grid(self) -> np.ndarray:
        """Return the current grid as a (W, H, 1) int array.

        Layout matches what downstream visitation/heatmap code expects:
        wall=2, empty=1, goal=8, lava=9, floor=3, obstacle=13, agent=10.
        The agent is included only after `reset()` has set its position.
        """
        grid = self._base_grid.copy()
        grid[self.goal_pos[1], self.goal_pos[0]] = GOAL
        if self.step_count > 0 or self.agent_pos != (0, 0):
            grid[self.agent_pos[1], self.agent_pos[0]] = AGENT
        return grid.T[..., None]  # transpose to (W, H), then add channel

    def get_rewards_heatmap(self, extractor: torch.nn.Module, eigenvectors):
        """Render per-eigenvector reward maps as RGB images.

        Walks the agent over every reachable (non-wall, non-goal) cell and
        evaluates ``extractor(state)`` to produce a scalar reward per cell.
        Positives map to red, negatives to blue, walls/goal stay grey.
        """
        state = self.get_grid()
        state[state == AGENT] = EMPTY
        grid = state
        mask = (grid != WALL) & (grid != GOAL)
        valid_coords = np.argwhere(mask)  # (N, 3) — last col is the channel

        state_batch = np.stack([np.array([c[0], c[1]]) for c in valid_coords])

        heatmaps = []
        shape = (self.width, self.height, 1)
        for vec_idx, vec_sign in eigenvectors:
            reward_map = np.zeros(shape, dtype=np.float32)
            with torch.no_grad():
                features, _ = extractor(state_batch)
                features = features.cpu().numpy()
            for i, coord in enumerate(state_batch):
                reward_map[coord[0], coord[1], 0] = vec_sign * features[i, vec_idx]

            self._normalize_signed(reward_map, mask)
            heatmaps.append(self.reward_map_to_rgb(reward_map, mask))
        return heatmaps

    def reward_map_to_rgb(self, reward_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
        rgb = np.zeros((self.width, self.height, 3), dtype=np.float32)
        pos = mask & (reward_map > 0)
        neg = mask & (reward_map < 0)
        rgb[neg[:, :, 0], 2] = -reward_map[neg]
        rgb[pos[:, :, 0], 0] = reward_map[pos]
        rgb[~mask[:, :, 0], :] = 0.5
        return rgb

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_chars(self, ch: str) -> list[tuple[int, int]]:
        """Return (x, y) coordinates of every occurrence of ``ch``."""
        return [(x, y) for y, row in enumerate(self.ascii_map) for x, c in enumerate(row) if c == ch]

    def _cell_at(self, x: int, y: int) -> int:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return WALL
        c = self._base_grid[y, x]
        # The goal cell only "becomes" a goal at runtime; in `_base_grid`
        # spawn-marker cells are stored as EMPTY so we overlay it here.
        if (x, y) == self.goal_pos:
            return GOAL
        return int(c)

    @staticmethod
    def _normalize_signed(reward_map: np.ndarray, mask: np.ndarray) -> None:
        """In-place: scale positives to [0,1] and negatives to [-1,0]."""
        pos = mask & (reward_map > 0)
        neg = mask & (reward_map < 0)
        if np.any(pos):
            lo, hi = reward_map[pos].min(), reward_map[pos].max()
            if hi != lo:
                reward_map[pos] = (reward_map[pos] - lo) / (hi - lo + 1e-4)
        if np.any(neg):
            lo, hi = reward_map[neg].min(), reward_map[neg].max()
            if hi != lo:
                reward_map[neg] = (reward_map[neg] - lo) / (hi - lo + 1e-4) - 1.0
