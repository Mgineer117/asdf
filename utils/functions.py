import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from log.wandb_logger import WandbLogger


def build_activation(activation_name):
    """Resolve an activation-name arg into an nn.Module. Default: ReLU."""
    if activation_name is None:
        return nn.ReLU()
    key = str(activation_name).strip().lower()
    if key == "relu":
        return nn.ReLU()
    if key == "tanh":
        return nn.Tanh()
    raise ValueError(
        f"Unsupported activation '{activation_name}'. Use 'relu' or 'tanh'."
    )


def build_actor_critic_pair(args, action_dim=None, is_discrete=None, activation=None):
    """
    Build a (PPO_Actor, PPO_Critic) pair from `args`.

    Centralises the boilerplate that every algorithm in `algorithms/`
    repeats. `action_dim` and `is_discrete` may be overridden (e.g. for
    HRL's discrete-option high-level actor); both default to `args`.
    """
    # Local import to avoid pulling torch-heavy modules at package-import
    # time (this helper is loaded by main.py via utils.functions).
    from policy.layers.ppo_networks import PPO_Actor, PPO_Critic

    if activation is None:
        activation = build_activation(getattr(args, "actor_activation", None))

    actor = PPO_Actor(
        input_dim=args.state_dim,
        hidden_dim=args.actor_fc_dim,
        action_dim=args.action_dim if action_dim is None else action_dim,
        is_discrete=args.is_discrete if is_discrete is None else is_discrete,
        activation=activation,
        device=args.device,
    )
    critic = PPO_Critic(
        args.state_dim,
        hidden_dim=args.critic_fc_dim,
        activation=activation,
        device=args.device,
    )
    return actor, critic


def setup_logger(args, unique_id, exp_time, seed, is_sweep: False):
    """
    setup logger both using WandB and Tensorboard
    Return: WandB logger, Tensorboard logger
    """
    # Get the current date and time
    if args.group is None:
        args.group = "-".join((exp_time, unique_id))

    if args.name is None:
        args.name = "-".join(
            (args.algo_name, args.env_name, unique_id, "seed:" + str(seed))
        )

    if args.project is None:
        args.project = args.task

    args.logdir = os.path.join(args.logdir, args.group)

    default_cfg = vars(args)
    sweep_metric_prefix = getattr(args, "sweep_metric_prefix", None)
    logger = WandbLogger(
        config=default_cfg,
        project=args.project,
        group=args.group,
        name=args.name,
        log_dir=args.logdir,
        log_txt=True,
        is_sweep=is_sweep,
        sweep_metric_prefix=sweep_metric_prefix,
    )
    logger.save_config(default_cfg, verbose=True)

    tensorboard_path = os.path.join(logger.log_dir, "tensorboard")
    os.makedirs(tensorboard_path, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_path)

    return logger, writer


def seed_all(seed=0):
    # Set the seed for hash-based operations in Python
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Set the seed for Python's random module
    random.seed(seed)

    # Set the seed for NumPy's random number generator
    np.random.seed(seed)

    # Set the seed for PyTorch (both CPU and GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU setups

    # Ensure reproducibility of PyTorch operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def concat_csv_columnwise_and_delete(folder_path, output_file="output.csv"):
    if not os.path.exists(folder_path):
        print(f"Log folder not found at {folder_path}. Skipping CSV concatenation.")
        return

    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    if not csv_files:
        print("No CSV files found in the folder.")
        return

    dataframes = []

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        dataframes.append(df)

    # Concatenate column-wise (axis=1)
    combined_df = pd.concat(dataframes, axis=1)

    # Save to output file
    output_file = os.path.join(folder_path, output_file)
    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV saved to {output_file}")

    # Delete original CSV files
    for file in csv_files:
        os.remove(os.path.join(folder_path, file))

    print("Original CSV files deleted.")
