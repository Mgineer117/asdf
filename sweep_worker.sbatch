#!/bin/bash
#SBATCH --job-name=irpo_sweep
#SBATCH --account=huytran1-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --gres=gpu:A100:2
#SBATCH --time=3-00:00:00
#SBATCH --nice=0
#SBATCH --output=logs/irpo_sweep.o%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu

# The sweep ID is provided as the first argument to sbatch
SWEEP_ID=$1

if [ -z "$SWEEP_ID" ]; then
    echo "Error: No sweep ID provided!"
    exit 1
fi

echo "Starting WandB agents for Sweep ID: $SWEEP_ID on 2 A100 GPUs (20 agents total)"

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook)"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/minjae5/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
conda activate irpo

# Run 10 agents on GPU 0
echo "Starting 10 agents on GPU 0..."
for i in {1..10}; do
    echo "  -> Launching Agent $i on GPU 0..."
    CUDA_VISIBLE_DEVICES=0 python search_irpo_random.py --sweep_id "$SWEEP_ID" --count 20 &
    sleep 3
done

# Run 10 agents on GPU 1
echo "Starting 10 agents on GPU 1..."
for i in {1..10}; do
    echo "  -> Launching Agent $((i+10)) on GPU 1..."
    CUDA_VISIBLE_DEVICES=1 python search_irpo_random.py --sweep_id "$SWEEP_ID" --count 20 &
    sleep 3
done

echo "All 20 agents successfully deployed! Waiting for completion..."
wait
