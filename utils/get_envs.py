import ale_py
import gymnasium as gym

import gymnasium_robotics

gym.register_envs(gymnasium_robotics)
gym.register_envs(ale_py)

from utils.__init__ import ANTMAZE_MAPS, EPI_LENGTH, POINTMAZE_MAPS, POS_IDX, GOAL_IDX
from utils.wrapper import (
    AntMazeWrapper,
    ArcadeWrapper,
    FetchWrapper,
    GridWrapper,
    MazeWrapper,
    MujocoWrapper,
)


def get_env(args):
    # Determine episode length
    episode_len = EPI_LENGTH[args.env_name]
    env_name, _, version = args.env_name.partition("-")

    if env_name == "fourrooms":
        from gridworld.envs.fourrooms import FourRooms

        env = FourRooms(grid_type=version, max_steps=episode_len)
        env = GridWrapper(env)
    elif env_name == "maze":
        from gridworld.envs.maze import Maze

        env = Maze(grid_type=version, max_steps=episode_len)
        env = GridWrapper(env)
    elif env_name == "fetchreach":
        env = gym.make(
            "FetchReach-v4",
            max_episode_steps=episode_len,
            render_mode="rgb_array",
        )

        env = FetchWrapper(env, episode_len, args.seed)
    elif env_name == "fetchpush":
        env = gym.make(
            "FetchPush-v4",
            max_episode_steps=episode_len,
            render_mode="rgb_array",
        )

        env = FetchWrapper(env, episode_len, args.seed)
    elif env_name == "fetchpusheasy":
        env = gym.make(
            "FetchPushEasy-v4",
            max_episode_steps=episode_len,
            render_mode="rgb_array",
        )

        env = FetchWrapper(env, episode_len, args.seed)
    elif env_name == "pointmaze":
        key = f"{env_name}-{version}"
        example_map = POINTMAZE_MAPS[key]

        env = gym.make(
            "PointMaze_UMaze-v3",
            maze_map=example_map,
            max_episode_steps=episode_len,
            continuing_task=False,
            render_mode="rgb_array",
        )

        env = MazeWrapper(env, example_map, episode_len, args.seed, cell_size=1.0)

    elif env_name == "antmaze":
        key = f"{env_name}-{version}"
        example_map = ANTMAZE_MAPS[key]

        env = gym.make(
            "AntMaze_UMaze-v5",
            maze_map=example_map,
            max_episode_steps=episode_len,
            continuing_task=False,
            render_mode="rgb_array",
        )

        env = AntMazeWrapper(env, example_map, episode_len, args.seed, cell_size=4.0)
    elif env_name == "pacman":
        from extractor.base.image_encoder import pretrain_image_encoder

        _raw_env = gym.make(
            "ALE/Pacman-v5",
            render_mode="rgb_array",
            max_episode_steps=episode_len,
            obs_type="grayscale",
        )
        # Pretrain (or load) the CNN encoder on raw frames, then encode at the
        # wrapper level so the batch stores compact vectors, not pixel arrays.
        _encoder = pretrain_image_encoder(
            _raw_env,
            seed=args.seed,
            encoder_dim=getattr(args, "encoder_dim", 256),
            device=getattr(args, "device", "cpu"),
        )
        env = ArcadeWrapper(_raw_env, encoder=_encoder, device=getattr(args, "device", "cpu"))
    elif env_name == "ant":
        env = gym.make(
            "Ant-v5",
            render_mode="rgb_array",
        )
        env = MujocoWrapper(env, vel_threshold=3.0)
    elif env_name == "walker":
        env = gym.make(
            "Walker2d-v5",
            render_mode="rgb_array",
        )
        env = MujocoWrapper(env, vel_threshold=4.0)

    elif env_name == "halfcheetah":
        env = gym.make(
            "HalfCheetah-v5",
            render_mode="rgb_array",
        )
        env = MujocoWrapper(env, vel_threshold=4.0)

    elif env_name == "hopper":
        env = gym.make(
            "Hopper-v5",
            render_mode="rgb_array",
        )
        env = MujocoWrapper(env, vel_threshold=3.0)
    else:
        raise NotImplementedError(f"Environment {env_name} is not supported.")

    args.episode_len = episode_len
    args.pos_idx = POS_IDX[env_name]
    args.goal_idx = GOAL_IDX[env_name]
    args.is_discrete = env.action_space.__class__.__name__ == "Discrete"

    if env_name in ["fourrooms", "maze"]:
        args.state_dim = env.observation_space.shape
        args.action_dim = env.action_space.n
    elif env_name in ["pointmaze", "antmaze", "fetchreach", "fetchpush", "fetchpusheasy"]:
        args.state_dim = (
            env.observation_space["observation"].shape[0]
            + env.observation_space["achieved_goal"].shape[0]
            + env.observation_space["desired_goal"].shape[0],
        )
        args.action_dim = env.action_space.shape[0]
    elif env_name in ["pacman"]:
        # observation_space.shape is (encoder_dim,) after ArcadeWrapper encoding
        args.state_dim = env.observation_space.shape[0]
        args.action_dim = env.action_space.n
    elif env_name in ["ant", "walker", "halfcheetah", "hopper"]:
        args.state_dim = (env.observation_space.shape[0] + len(args.pos_idx),)
        args.action_dim = env.action_space.shape[0]
    else:
        raise NotImplementedError(f"Environment {env_name} is not supported.")

    return env
