import os

envs = ["pacman", "amidar"]
algos = ["ppo", "irpo", "maml", "hrl", "drnd", "trpo", "psne"]

sbatch_template = """#!/bin/bash
#SBATCH --job-name={env}_{algo}
#SBATCH --account=huytran1-ic
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
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

# === Run 10 seeds (0-9) in parallel on 1 GPU / Node === #
for SEED in {{0..9}}; do
    python3 main.py --project NEW_IRPO --env-name {env} --algo-name {algo} --seed $SEED --gpu-idx 0 &
    sleep 3
done

# === Wait for all 10 runs to finish ===
wait
"""

os.makedirs("scripts", exist_ok=True)

# Generate sbatch files
for env in envs:
    for algo in algos:
        if algo in ["irpo", "maml", "hrl"]:
            partition = "IllinoisComputes-GPU"
            time_limit = "3-00:00:00"
        else:
            partition = "eng-research-gpu"
            time_limit = "2-00:00:00"
            
        filepath = f"scripts/{env}_{algo}.sbatch"
        content = sbatch_template.format(env=env, algo=algo, partition=partition, time_limit=time_limit)
        with open(filepath, "w") as f:
            f.write(content)
        os.chmod(filepath, 0o755)
        print(f"Generated {filepath} (Partition: {partition}, Time: {time_limit})")

# Generate submit_all.sh
submit_content = """#!/bin/bash

# Create logs directory locally
mkdir -p logs

echo "========================================================="
echo " Submitting all 14 Experiment Jobs to the Cluster"
echo " Each job (algorithm + environment) is allocated to 1 Node"
echo " running 10 seeds (0-9) in parallel on 1 GPU."
echo "========================================================="

"""

for env in envs:
    for algo in algos:
        submit_content += f'echo "[SUBMIT] scripts/{env}_{algo}.sbatch"\nsbatch scripts/{env}_{algo}.sbatch\nsleep 0.5\n\n'

submit_content += """echo "========================================================="
echo " All 14 jobs successfully submitted to queue!"
echo " Check progress using: squeue -u $(whoami)"
echo " Logs will be stored in: logs/"
echo "========================================================="
"""

submit_filepath = "scripts/submit_all.sh"
with open(submit_filepath, "w") as f:
    f.write(submit_content)
os.chmod(submit_filepath, 0o755)
print(f"Generated {submit_filepath}")
