#!/bin/bash

# Create logs directory if missing
mkdir -p logs

echo "========================================================="
echo " Starting IRPO for pacman and amidar on 4 GPUs"
echo " 5 seeds per environment (10 runs total)"
echo " Maximum 3 runs per GPU (5GB VRAM per run, 16GB total)"
echo "========================================================="

# 1. Block Python from reading ~/.local/lib to prevent cross-contamination
export PYTHONNOUSERSITE=1

# 2. Hook conda into the non-interactive shell
eval "$(conda shell.bash hook)"

# 3. Add MuJoCo and NVIDIA to path (if needed)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/minjae5/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# 4. Safely activate the environment
conda activate IRPO

# Array of jobs: "env_name seed gpu_idx"
# We have 10 jobs total. Max 3 per GPU.
# GPU 0: 3 runs
# GPU 1: 3 runs
# GPU 2: 2 runs
# GPU 3: 2 runs

JOBS=(
    "pacman 0 0"
    "pacman 1 0"
    "pacman 2 0"
    "pacman 3 1"
    "pacman 4 1"
    "amidar 0 1"
    "amidar 1 2"
    "amidar 2 2"
    "amidar 3 3"
    "amidar 4 3"
)

echo "Launching background processes..."

for JOB in "${JOBS[@]}"; do
    read -r ENV SEED GPU <<< "$JOB"
    
    echo "Launching: env=$ENV, seed=$SEED, gpu=$GPU"
    python3 main.py \
        --project Atari \
        --env "$ENV" \
        --algo irpo \
        --seed "$SEED" \
        --gpu-idx "$GPU" > "logs/${ENV}_irpo_seed${SEED}_gpu${GPU}.log" 2>&1 &
        
    # Small sleep to prevent simultaneous initialization crashes
    sleep 5
done

echo "========================================================="
echo " All 10 jobs successfully launched in the background!"
echo " Logs are being saved to logs/<env>_irpo_seed<seed>_gpu<gpu>.log"
echo " Waiting for all processes to finish..."
echo "========================================================="

# Wait for all background jobs to complete
wait

echo "All runs completed!"
