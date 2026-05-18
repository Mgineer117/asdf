#!/bin/bash
#SBATCH --job-name=rl-run
#SBATCH --account=huytran1-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu

# Raise file descriptor limit
ulimit -n 4096

# 1. Block Python from reading ~/.local/lib to prevent cross-contamination (CRITICAL)
export PYTHONNOUSERSITE=1

# 2. Hook conda into the non-interactive shell
eval "$(conda shell.bash hook)"

# 3. Add MuJoCo and NVIDIA libraries to path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/minjae5/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# 4. Activate your conda environment
conda activate irpo

# Create logs directory if missing
mkdir -p logs

echo "Starting job: ${SLURM_JOB_NAME} (Job ID: ${SLURM_JOB_ID})"
echo "Environment: ${ENV}"
echo "Algorithm: ${ALGO}"
echo "Running on node: $(hostname)"
echo "GPU Allocated: ${CUDA_VISIBLE_DEVICES}"

# 5. Run the experiment
python3 main.py --algo-name "${ALGO}" --env-name "${ENV}" --override-results
