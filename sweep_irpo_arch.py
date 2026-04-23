#!/usr/bin/env python3
"""Launch a small IRPO architecture sweep with nohup.

Run this from a bash shell or WSL so `nohup` and background jobs work as expected.
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

PROJECT = "trap_reward_ablation"
ENV_NAME = "fourrooms-v2"
ALGO_NAME = "irpo"
NUM_RUNS = 10
NUM_EXP_UPDATES = 10
GPU_IDX = 0

ARCHITECTURES = [
    ([64, 64], "relu"),
    ([64, 64], "tanh"),
    ([256, 256], "relu"),
    ([256, 256], "tanh"),
    ([100, 50, 25], "relu"),
    ([100, 50, 25], "tanh"),
]


def build_command(hidden_dims: list[int], activation: str) -> str:
    dims = " ".join(str(dim) for dim in hidden_dims)
    run_tag = "-".join(str(dim) for dim in hidden_dims)
    log_dir = Path("sweep_logs") / PROJECT / ENV_NAME / ALGO_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{run_tag}_{activation}.out"

    return (
        "nohup python3 main.py "
        f"--project {PROJECT} "
        f"--env-name {ENV_NAME} "
        f"--algo-name {ALGO_NAME} "
        f"--num-runs {NUM_RUNS} "
        f"--num-exp-updates {NUM_EXP_UPDATES} "
        f"--gpu-idx {GPU_IDX} "
        f"--actor-fc-dim {dims} "
        f"--actor-activation {activation} "
        f"> {log_file} 2>&1 &"
    )


def main() -> None:
    for hidden_dims, activation in ARCHITECTURES:
        command = build_command(hidden_dims, activation)
        print(command)
        if platform.system().lower().startswith("win"):
            subprocess.Popen(["wsl", "bash", "-lc", command])
        else:
            subprocess.Popen(command, shell=True)


if __name__ == "__main__":
    main()
