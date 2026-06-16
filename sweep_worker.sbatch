#!/bin/bash
#SBATCH --job-name=irpo_sweep
#SBATCH --account=huytran1-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=3-00:00:00
#SBATCH --array=1-10
#SBATCH --nice=0
#SBATCH --output=logs/irpo_sweep.o%A_%a
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu

# The sweep ID is provided as the first argument to sbatch
SWEEP_ID=$1

if [ -z "$SWEEP_ID" ]; then
    echo "Error: No sweep ID provided!"
    exit 1
fi

echo "Starting WandB agent for Sweep ID: $SWEEP_ID [Array Task $SLURM_ARRAY_TASK_ID]"

# Run the agent (each parallel worker will execute 20 trials before exiting)
python search_irpo_random.py --sweep_id "$SWEEP_ID" --count 20
