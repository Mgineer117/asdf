import random
from itertools import chain
from typing import (
    Any,
    Final,
    Iterable,
    Literal,
    SupportsFloat,
    TypeAlias,
    TypedDict,
    TypeVar,
)

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from numpy.typing import NDArray

from gridworld.core.agent import Agent, AgentT, GridActions, PolicyAgent
from gridworld.core.constants import *
from gridworld.core.grid import Grid
from gridworld.core.object import Floor, Goal, Lava, Obstacle, Wall
from gridworld.core.world import GridWorld
from gridworld.envs.__init__ import FOURROOMS_G_MAPS, FOURROOMS_MAPS
from gridworld.multigrid import MultiGridEnv
from gridworld.policy.ctf.heuristic import (
    HEURISTIC_POLICIES,
    CtfPolicyT,
    RoombaPolicy,
    RwPolicy,
)
from gridworld.typing import Position
from gridworld.utils.window import Window


class FourRooms(MultiGridEnv):
    """
    Environment for capture the flag with multiple agents with N blue agents and M red agents.
    """

    def __init__(
        self,
        grid_type: str,
        max_steps: int,
        goal_conditioned: bool = False,
        highlight_visible_cells: bool = False,
        tile_size: int = 10,
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
    ):
        self.max_steps = max_steps
        self.grid_type = grid_type

        self.world = GridWorld
        self.actions_set = GridActions

        see_through_walls: bool = False

        self.agent = Agent(
            self.world,
            color="blue",
            bg_color="light_blue",
            actions=self.actions_set,
            type="agent",
        )

        # Map the structure
        self.map_structure = FOURROOMS_G_MAPS if goal_conditioned else FOURROOMS_MAPS
        self.maze = self.map_structure[self.grid_type]

        self.width = len(self.maze[0])
        self.height = len(self.maze)
        self.grid_size = (self.width, self.height)

        # Scan the maze for 'c' or 'g' for goals, and 'c' or 'r' for agents
        self.goal_positions = self.find_obj_coordinates(
            "g"
        ) + self.find_obj_coordinates("c")
        self.agent_positions = self.find_obj_coordinates(
            "r"
        ) + self.find_obj_coordinates("c")

        # Safety check to ensure the map isn't missing required characters
        assert (
            len(self.goal_positions) > 0
        ), "No valid goal positions ('c' or 'g') found in the maze."
        assert (
            len(self.agent_positions) > 0
        ), "No valid agent positions ('c' or 'r') found in the maze."

        self.grids = {}
        self.grid_imgs = {}

        super().__init__(
            width=self.width,
            height=self.height,
            max_steps=self.max_steps,
            see_through_walls=see_through_walls,
            agents=[self.agent],
            actions_set=self.actions_set,
            world=self.world,
            render_mode=render_mode,
            highlight_visible_cells=highlight_visible_cells,
            tile_size=tile_size,
        )

    def get_grid(self):
        self.reset()
        grid = self.grid.encode()
        self.close()
        return grid

    def _set_observation_space(self) -> spaces.Dict | spaces.Box:
        observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array(
                [self.width, self.height, self.width, self.height],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        return observation_space

    def _gen_grid(self, width, height, options):
        # Create the grid
        self.grid = Grid(width, height, self.world)

        # Translate the maze structure into the grid
        for y, row in enumerate(self.maze):
            for x, cell in enumerate(row):
                if cell == "#":
                    self.grid.set(x, y, Wall(self.world))
                elif cell == " ":
                    self.grid.set(x, y, None)
                elif cell == "@":
                    self.grid.set(x, y, Floor(self.world, color="green"))

        # Place the goal first
        self.current_goal_position = random.choice(self.goal_positions)

        # Filter out the goal position from the available agent positions
        valid_agent_positions = [
            p for p in self.agent_positions if p != self.current_goal_position
        ]

        if not valid_agent_positions:
            raise ValueError("No valid agent positions left after placing the goal!")

        # Pick the agent position from the remaining valid spots
        self.current_agent_position = random.choice(valid_agent_positions)

        # Place the goal
        goal = Goal(self.world, index=1)
        self.put_obj(goal, *self.current_goal_position)
        goal.init_pos, goal.cur_pos = self.current_goal_position

        # Place agent
        self.place_agent(self.agent, pos=self.current_agent_position)

    def find_obj_coordinates(self, obj: str) -> list[tuple[int, int]]:
        """
        Finds all (x, y) coordinates of a specific string character in the maze.
        Returns an empty list if the object is not found.
        """
        coord_list = []

        # Iterate through rows (y) and columns (x)
        for y, row in enumerate(self.maze):
            for x, cell in enumerate(row):
                if cell == obj:  # Use == for string comparison
                    coord_list.append((x, y))

        return coord_list

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict = {},
    ):
        super().reset(seed=seed, options=options)

        observations = self.get_obs()
        info = {"success": False}

        return observations, info

    def _restore_previous_cell(self, pos):
        """Restores the cell to its original state based on the map structure."""
        x, y = pos
        # Look up the character in the original string array
        cell_char = self.maze[y][x]

        if cell_char == "@":
            self.grid.set(x, y, Floor(self.world, color="green"))
        else:
            self.grid.set(x, y, None)

    def step(self, action):
        self.step_count += 1

        action = np.argmax(action)

        # Get the current agent position
        curr_pos = self.agent.pos

        reward, done, info = 0.0, False, {}

        # Rotate left
        if action == self.actions.left:
            fwd_pos = tuple(a + b for a, b in zip(curr_pos, (0, -1)))
            fwd_cell = self.grid.get(*fwd_pos)

            if fwd_cell is not None:
                if self.agent.type == "agent":
                    if fwd_cell.type == "goal":
                        done = True
                        reward += 1.0
                        info["success"] = True
                    elif fwd_cell.type == "lava":
                        done = True
                        reward += -1
                        info["success"] = False
                    elif fwd_cell.type == "floor":
                        reward += 0.1
                        done = True
                        info["success"] = False
            elif fwd_cell is None or fwd_cell.can_overlap():
                self._restore_previous_cell(self.agent.pos)
                self.grid.set(*fwd_pos, self.agent)
                self.agent.pos = fwd_pos

        # Rotate right
        elif action == self.actions.right:
            fwd_pos = tuple(a + b for a, b in zip(curr_pos, (0, +1)))
            fwd_cell = self.grid.get(*fwd_pos)

            if fwd_cell is not None:
                if self.agent.type == "agent":
                    if fwd_cell.type == "goal":
                        done = True
                        reward += 1.0
                        info["success"] = True
                    elif fwd_cell.type == "lava":
                        done = True
                        reward += -1
                        info["success"] = False
                    elif fwd_cell.type == "floor":
                        reward += 0.1
                        done = True
                        info["success"] = False
            elif fwd_cell is None or fwd_cell.can_overlap():
                self._restore_previous_cell(self.agent.pos)
                self.grid.set(*fwd_pos, self.agent)
                self.agent.pos = fwd_pos

        # Move forward (Up)
        elif action == self.actions.up:
            fwd_pos = tuple(a + b for a, b in zip(curr_pos, (-1, 0)))
            fwd_cell = self.grid.get(*fwd_pos)

            if fwd_cell is not None:
                if self.agent.type == "agent":
                    if fwd_cell.type == "goal":
                        done = True
                        reward += 1.0
                        info["success"] = True
                    elif fwd_cell.type == "lava":
                        done = True
                        reward += -1
                        info["success"] = False
                    elif fwd_cell.type == "floor":
                        reward += 0.1
                        done = True
                        info["success"] = False
                        # self._restore_previous_cell(self.agent.pos)
                        # self.grid.set(*fwd_pos, self.agent)
                        # self.agent.pos = fwd_pos
            elif fwd_cell is None or fwd_cell.can_overlap():
                self._restore_previous_cell(self.agent.pos)
                self.grid.set(*fwd_pos, self.agent)
                self.agent.pos = fwd_pos

        # Move Backward (Down)
        elif action == self.actions.down:
            fwd_pos = tuple(a + b for a, b in zip(curr_pos, (+1, 0)))
            fwd_cell = self.grid.get(*fwd_pos)

            if fwd_cell is not None:
                if self.agent.type == "agent":
                    if fwd_cell.type == "goal":
                        done = True
                        reward += 1.0
                        info["success"] = True
                    elif fwd_cell.type == "lava":
                        done = True
                        reward += -1
                        info["success"] = False
                    elif fwd_cell.type == "floor":
                        reward += 0.1
                        done = True
                        info["success"] = False
                        # self._restore_previous_cell(self.agent.pos)
                        # self.grid.set(*fwd_pos, self.agent)
                        # self.agent.pos = fwd_pos
            elif fwd_cell is None or fwd_cell.can_overlap():
                self._restore_previous_cell(self.agent.pos)
                self.grid.set(*fwd_pos, self.agent)
                self.agent.pos = fwd_pos
        else:
            assert False, "unknown action"

        terminated = done
        truncated = True if self.step_count >= self.max_steps else False

        observations = self.get_obs()

        return observations, reward, terminated, truncated, info

    def get_obs(self):
        # Default to first goal position before reset finishes setting it
        goal_pos = getattr(self, "current_goal_position", self.goal_positions[0])

        obs = {
            "achieved_goal": np.array([self.agent.pos[0], self.agent.pos[1]]),
            "desired_goal": np.array(
                [
                    goal_pos[0],
                    goal_pos[1],
                ]
            ),
        }
        return obs

    def get_rewards_heatmap(
        self, extractor: torch.nn.Module, eigenvectors: np.ndarray | list
    ):

        # Environment indices
        empty_idx = 1
        goal_idx = 8
        agent_idx = 10
        obs_idx = 13
        wall_idx = 2

        # Get base state
        state = self.get_grid()
        agent_pos = np.where(state == agent_idx)
        state[agent_pos] = empty_idx
        grid = state

        mask = (grid != wall_idx) & (grid != goal_idx)

        # Get coordinates where agent can be placed
        valid_coords = np.argwhere(mask)  # shape: [num_valid, 2]

        # Generate a batch of states
        state_batch = []
        for coord in valid_coords:
            state = np.array([coord[0], coord[1]])
            state_batch.append(state)

        # Stack the batch: shape = [num_valid, H, W] or [num_valid, H, W, C]
        state_batch = np.stack(state_batch)

        heatmaps = []
        grid_shape = (self.width, self.height, 1)
        for n in range(len(eigenvectors)):
            reward_map = np.full(grid_shape, fill_value=0.0)

            with torch.no_grad():
                features, _ = extractor(state_batch)
                features = features.cpu().numpy()

            for i in range(features.shape[0]):
                agent_pos = [state_batch[i][0], state_batch[i][1]]
                x, y = agent_pos[0], agent_pos[1]

                eigenvector_idx, eigenvector_sign = eigenvectors[n]
                reward = eigenvector_sign * features[i, eigenvector_idx]

                reward_map[x, y, 0] = reward

            # reward_map = # normalize between -1 to 1
            pos_mask = np.logical_and(mask, (reward_map > 0))
            neg_mask = np.logical_and(mask, (reward_map < 0))

            # Normalize positive values to [0, 1]
            if np.any(pos_mask):
                pos_max, pos_min = (
                    reward_map[pos_mask].max(),
                    reward_map[pos_mask].min(),
                )
                if pos_max != pos_min:
                    reward_map[pos_mask] = (reward_map[pos_mask] - pos_min) / (
                        pos_max - pos_min + 1e-4
                    )

            # Normalize negative values to [-1, 0]
            if np.any(neg_mask):
                neg_max, neg_min = (
                    reward_map[neg_mask].max(),
                    reward_map[neg_mask].min(),
                )
                if neg_max != neg_min:
                    reward_map[neg_mask] = (reward_map[neg_mask] - neg_min) / (
                        neg_max - neg_min + 1e-4
                    ) - 1.0

            # Set all other entries (walls, empty) to 0
            # print(reward_map[:, :, 0])
            reward_map = self.reward_map_to_rgb(reward_map, mask)

            # set color theme as blue and red (blue = -1 and red = 1)
            # set wall color at value 0 and goal idx as 1
            heatmaps.append(reward_map)

        return heatmaps

    def reward_map_to_rgb(self, reward_map: np.ndarray, mask) -> np.ndarray:
        rgb_img = np.zeros((self.width, self.height, 3), dtype=np.float32)

        pos_mask = np.logical_and(mask, (reward_map > 0))
        neg_mask = np.logical_and(mask, (reward_map < 0))

        # Blue for negative: map [-1, 0] → [1, 0]
        rgb_img[neg_mask[:, :, 0], 2] = -reward_map[neg_mask]  # blue channel

        # Red for positive: map [0, 1] → [0, 1]
        rgb_img[pos_mask[:, :, 0], 0] = reward_map[pos_mask]  # red channel

        # rgb_img.flatten()[mask] to grey
        rgb_img[~mask[:, :, 0], :] = 0.5

        return rgb_img
