import datetime
import uuid

import numpy as np

import wandb

# Import your modified run function and args utilities
from main import run
from utils.get_args import get_args, override_args

# 1. Rename to IRPO and adjust the search space
sweep_config_irpo = {
    "method": "bayes",
    "metric": {"name": "mean_return", "goal": "maximize"},
    "parameters": {
        # Integers can stay discrete using q_uniform, or you can keep it as values
        # if there are only a few choices, but 'int_uniform' is better for Bayes.
        "num_exp_updates": {"distribution": "int_uniform", "min": 2, "max": 10},
        # Learning rates should almost always be searched on a log scale
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-3,
            "max": 1e-1,
        },
        "base_target_kl": {
            "distribution": "log_uniform_values",
            "min": 1e-3,
            "max": 1e-1,
        },
    },
}


def sweep_evaluate():
    with wandb.init() as sweep_run:
        config = sweep_run.config

        # 2. Dynamically set the run name using the IRPO hyperparameters
        custom_name = f"base_{config.base_target_kl}"
        sweep_run.name = custom_name

        init_args = get_args()

        # 3. Force the algorithm choice to IRPO
        init_args.algo_name = "irpo"

        seeds = [1825, 410, 4507]
        unique_id = str(uuid.uuid4())[:4]
        exp_time = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")

        performances = []

        for seed in seeds:
            args = override_args(init_args)
            args.seed = seed
            args.unique_id = unique_id

            # Apply W&B suggested hyperparameters
            for key, val in config.items():
                setattr(args, key, val)

            perf = run(args, seed, unique_id, exp_time, is_sweep=True)
            performances.append(perf)

        # Log the final mean to the sweep controller
        mean_perf = np.mean(performances)
        wandb.log({"mean_return": mean_perf})


if __name__ == "__main__":
    # 4. Initialize the sweep using the IRPO configuration
    sweep_id = wandb.sweep(sweep_config_irpo, project="rl-algorithm-comparison")

    # Launch the agent
    wandb.agent(sweep_id, function=sweep_evaluate, count=24)
