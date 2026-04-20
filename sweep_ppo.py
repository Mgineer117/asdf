import datetime
import uuid

import numpy as np

import wandb

# Import your modified run function and args utilities
from main import run
from utils.get_args import get_args, override_args

sweep_config_ppo = {
    "method": "bayes",  # 'bayes' is excellent for finding optimal RL params
    "metric": {"name": "mean_return", "goal": "maximize"},
    "parameters": {
        # Learning rates should almost always be searched on a log scale
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-4,  # 0.0001
            "max": 1e-2,  # 0.01
        },
        # KL divergence targets also benefit from log scale exploration
        "target_kl": {
            "distribution": "log_uniform_values",
            "min": 1e-3,
            "max": 1e-1,
        },
        # Clipping ranges are linear, so standard uniform is best
        "eps_clip": {
            "distribution": "uniform",
            "min": 0.1,
            "max": 0.3,
        },
    },
}


def sweep_evaluate():
    # wandb.init() creates the parent run for this specific hyperparameter config
    with wandb.init() as sweep_run:
        config = sweep_run.config

        # ✅ Dynamically set the run name using the current hyperparameters
        # We format actor_lr to scientific notation (e.g., 3.2e-04) for cleaner UI
        custom_name = (
            f"lr_{config.learning_rate:.1e}_kl_{config.target_kl}_eps_{config.eps_clip}"
        )
        sweep_run.name = custom_name

        init_args = get_args()
        init_args.algo_name = "ppo"

        # Use 5 fixed seeds for a fair comparison between configs
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

            # Note: If your setup_logger calls wandb.init(), you may want to set it up
            # to group by sweep_run.id, or temporarily disable child logging during sweeps
            perf = run(args, seed, unique_id, exp_time, is_sweep=True)
            performances.append(perf)

        # Log the final mean to the sweep controller
        mean_perf = np.mean(performances)
        wandb.log({"mean_return": mean_perf})


if __name__ == "__main__":
    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config_ppo, project="rl-algorithm-comparison")
    # Launch the agent (e.g., test 20 different configurations)
    wandb.agent(sweep_id, function=sweep_evaluate, count=24)
