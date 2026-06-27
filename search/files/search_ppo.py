import datetime
import os
import random
import sys
import uuid
import argparse

# Allow running from any directory by anchoring imports to repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import wandb

from utils.get_args import get_args, override_args
from main import run


def train():
    wandb.init()
    config = wandb.config

    args = get_args()
    args.algo_name = "ppo"
    args.env_name = search_args.env

    args = override_args(args)

    if "actor_lr" in config:
        args.actor_lr = config.actor_lr
    if "critic_lr" in config:
        args.critic_lr = config.critic_lr
    if "gae" in config:
        args.gae = config.gae
    if "eps_clip" in config:
        args.eps_clip = config.eps_clip
    if "target_kl" in config:
        args.target_kl = config.target_kl
    if "entropy_scaler" in config:
        args.entropy_scaler = config.entropy_scaler

    unique_id = str(uuid.uuid4())[:4]
    exp_time = datetime.datetime.now().strftime("%m-%d_%H-%M-%S.%f")
    seed = random.randint(1, 10000)
    args.seed = seed
    args.num_runs = 1

    print(f"-------------------------------------------------------")
    print(f"      PPO {search_args.env} Sweep Trial ID: {unique_id}")
    print(f"      Seed: {seed}")
    print(f"      Time Begun: {exp_time}")
    print(f"-------------------------------------------------------")

    if hasattr(args, "run_id"):
        run(args, seed, unique_id, exp_time, args.run_id)
    else:
        run(args, seed, unique_id, exp_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WandB Sweep Launcher for PPO")
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--env", type=str, default="pacman")

    search_args, remaining_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_args

    if search_args.project is None:
        search_args.project = f"PPO-{search_args.env.upper()}-SWEEP"

    torch.set_default_dtype(torch.float32)

    if search_args.sweep_id is None:
        sweep_config = {
            "method": "bayes",
            "metric": {
                "name": "eval/return_mean",
                "goal": "maximize"
            },
            "parameters": {
                "actor_lr": {
                    "min": 1e-5,
                    "max": 1e-2,
                    "distribution": "log_uniform_values"
                },
                "critic_lr": {
                    "min": 1e-5,
                    "max": 1e-2,
                    "distribution": "log_uniform_values"
                },
                "gae": {
                    "min": 0.9,
                    "max": 1.0,
                    "distribution": "uniform"
                },
                "eps_clip": {
                    "values": [0.1, 0.2, 0.3]
                },
                "target_kl": {
                    "min": 1e-3,
                    "max": 0.1,
                    "distribution": "log_uniform_values"
                },
                # "entropy_scaler": {
                #     "min": 1e-4,
                #     "max": 1e-1,
                #     "distribution": "log_uniform_values"
                # },
            }
        }

        sweep_id = wandb.sweep(sweep_config, project=search_args.project)
        print(f"\n=======================================================")
        print(f"Created NEW wandb sweep with ID: {sweep_id}")
        print(f"To run additional agents in parallel, run:")
        print(f"python search_ppo.py --sweep_id {sweep_id} --env {search_args.env}")
        print(f"=======================================================\n")

        if search_args.count == 0:
            sys.exit(0)
    else:
        sweep_id = search_args.sweep_id
        print(f"\nJoining EXISTING wandb sweep with ID: {sweep_id}\n")

    print(f"Starting wandb agent for sweep {sweep_id}")
    wandb.agent(sweep_id, train, count=search_args.count, project=search_args.project)
