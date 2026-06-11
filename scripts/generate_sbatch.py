import os

envs = ["pacman", "amidar", "bankheist", "alien"]
algos = ["ppo", "irpo", "maml", "hrl", "drnd", "trpo", "psne"]

sbatch_template = """#!/bin/bash
#SBATCH --job-name={env}_{algo}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --gres=gpu:{gpu_count}
#SBATCH --time={time_limit}
#SBATCH --output=logs/{env}_{algo}.o%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu

ulimit -n 4096  # raise file descriptor limit

# 1. Block Python from reading ~/.local/lib to prevent cross-contamination (CRITICAL)
export PYTHONNOUSERSITE=1

# 2. Hook conda into the non-interactive shell
eval "$(conda shell.bash hook)"

# 3. Add MuJoCo and NVIDIA to path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/minjae5/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# 4. Safely activate your environment
conda activate irpo

# Create logs directory if missing
mkdir -p logs

{run_commands}

# === Wait for all 5 runs to finish ===
wait
"""

run_commands_1gpu = """# === Run 5 seeds (0-4) in parallel on 1 GPU / Node === #
for SEED in {{0..4}}; do
    python3 main.py --project Atari --env {env} --algo {algo} --seed $SEED --gpu-idx 0 --actor-activation elu &
    sleep 3
done"""

run_commands_2gpus = """# === Run 5 seeds across 2 GPUs === #
# First three seeds (0, 1, 2) on GPU 0
for SEED in 0 1 2; do
    python3 main.py --project Atari --env {env} --algo {algo} --seed $SEED --gpu-idx 0 --actor-activation elu &
    sleep 3
done

# Next two seeds (3, 4) on GPU 1
for SEED in 3 4; do
    python3 main.py --project Atari --env {env} --algo {algo} --seed $SEED --gpu-idx 1 --actor-activation elu &
    sleep 3
done"""

os.makedirs("scripts", exist_ok=True)

# Generate sbatch files
for env in envs:
    for algo in algos:
        if algo in ["psne", "trpo", "ppo"]:
            partition = "eng-research-gpu"
            account = "huytran1-ic"
            time_limit = "2-00:00:00"
            gpu_count = 2
            run_commands = run_commands_2gpus.format(env=env, algo=algo)
        elif algo in ["irpo", "maml"]:
            partition = "IllinoisComputes-GPU"
            account = "huytran1-ic"
            time_limit = "3-00:00:00"
            gpu_count = 1
            run_commands = run_commands_1gpu.format(env=env, algo=algo)
        elif algo in ["hrl", "drnd"]:
            partition = "csl"
            account = "huytran1-ic"
            time_limit = "3-00:00:00"
            gpu_count = 2
            run_commands = run_commands_2gpus.format(env=env, algo=algo)
        else:
            partition = "csl"
            account = "huytran1-ic"
            time_limit = "3-00:00:00"
            gpu_count = 2
            run_commands = run_commands_2gpus.format(env=env, algo=algo)
            
        filepath = f"scripts/{env}_{algo}.sbatch"
        content = sbatch_template.format(env=env, algo=algo, partition=partition, account=account, time_limit=time_limit, gpu_count=gpu_count, run_commands=run_commands)
        with open(filepath, "w") as f:
            f.write(content)
        os.chmod(filepath, 0o755)
        print(f"Generated {filepath} (Partition: {partition}, Account: {account}, Time: {time_limit})")

# Generate submit_all.sh
submit_content = f"""#!/bin/bash

# Create logs directory locally
mkdir -p logs

echo "========================================================="
echo " Submitting all {len(envs) * len(algos)} Experiment Jobs to the Cluster"
echo " Each job (algorithm + environment) is allocated to 1 Node"
echo " running 5 seeds across 1 or 2 GPUs depending on partition."
echo "========================================================="

"""

for env in envs:
    for algo in algos:
        submit_content += f'echo "[SUBMIT] scripts/{env}_{algo}.sbatch"\nsbatch scripts/{env}_{algo}.sbatch\nsleep 0.5\n\n'

submit_content += f"""echo "========================================================="
echo " All {len(envs) * len(algos)} jobs successfully submitted to queue!"
echo " Check progress using: squeue -u $(whoami)"
echo " Logs will be stored in: logs/"
echo "========================================================="
"""

submit_filepath = "scripts/submit_all.sh"
with open(submit_filepath, "w") as f:
    f.write(submit_content)
os.chmod(submit_filepath, 0o755)
print(f"Generated {submit_filepath}")

# Generate env-specific submit scripts
for env in envs:
    env_submit_content = f"""#!/bin/bash

# Create logs directory locally
mkdir -p logs

echo "========================================================="
echo " Submitting all {len(algos)} Experiment Jobs for {env} to the Cluster"
echo " Each job (algorithm + environment) is allocated to 1 Node"
echo " running 5 seeds (0-4) across 1 or 2 GPUs."
echo "========================================================="

"""
    for algo in algos:
        env_submit_content += f'echo "[SUBMIT] scripts/{env}_{algo}.sbatch"\nsbatch scripts/{env}_{algo}.sbatch\nsleep 0.5\n\n'

    env_submit_content += f"""echo "========================================================="
echo " All {len(algos)} jobs for {env} successfully submitted to queue!"
echo " Check progress using: squeue -u $(whoami)"
echo " Logs will be stored in: logs/"
echo "========================================================="
"""

    env_submit_filepath = f"scripts/run_{env}.sh"
    with open(env_submit_filepath, "w") as f:
        f.write(env_submit_content)
    os.chmod(env_submit_filepath, 0o755)
    print(f"Generated {env_submit_filepath}")
