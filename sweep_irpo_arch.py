"""IRPO architecture ablation as a wandb sweep on pointmaze-v4.

Sweeps actor/critic FC dims and activation. Each sweep run executes multiple
seeds and logs per-seed training curves under prefixes seed_<seed>/* (with their
own custom step metric `seed_<seed>/env_step`) so wandb's monotonic-step
constraint across seeds inside a single sweep run is bypassed. The sweep
optimizer still receives a single scalar `mean_return` per run.
"""

from __future__ import annotations

import datetime
import uuid

import numpy as np

import wandb
from main import run
from utils.get_args import get_args, override_args


PROJECT = "irpo_arch_ablation"
ENV_NAME = "pointmaze-v4"
ALGO_NAME = "irpo"
SEEDS = [1825, 410, 4507, 4013, 3658]  # 5 seeds (compute-reduced)


sweep_config = {
    "method": "grid",
    "metric": {"name": "mean_return", "goal": "maximize"},
    "parameters": {
        "fc_dim": {"values": ["32-32", "512-512", "256-128-64"]},
        "actor_activation": {"values": ["relu", "tanh"]},
    },
}


def _parse_dims(token: str) -> list[int]:
    return [int(x) for x in token.split("-")]


def sweep_evaluate():
    with wandb.init() as sweep_run:
        config = sweep_run.config
        fc_dim = _parse_dims(config.fc_dim)
        activation = config.actor_activation
        sweep_run.name = f"arch_{config.fc_dim}_{activation}"

        init_args = get_args()
        init_args.project = PROJECT
        init_args.env_name = ENV_NAME
        init_args.algo_name = ALGO_NAME
        init_args.actor_fc_dim = fc_dim
        init_args.critic_fc_dim = fc_dim
        init_args.actor_activation = activation

        unique_id = str(uuid.uuid4())[:4]
        exp_time = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")

        performances = []
        for seed in SEEDS:
            args = override_args(init_args)
            args.seed = seed
            args.unique_id = unique_id
            # Per-seed prefix → metrics logged as seed_<seed>/<key> with their
            # own step axis seed_<seed>/env_step. Avoids wandb monotonic-step
            # collisions across seeds.
            args.sweep_metric_prefix = f"seed_{seed}"

            perf = run(args, seed, unique_id, exp_time, is_sweep=True)
            performances.append(float(perf))
            wandb.log({f"seed_{seed}/final_return": float(perf)})

        wandb.log({
            "mean_return": float(np.mean(performances)),
            "std_return": float(np.std(performances)),
        })


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project=PROJECT)
    # 3 archs × 2 activations = 6 grid points
    wandb.agent(sweep_id, function=sweep_evaluate, count=6)
