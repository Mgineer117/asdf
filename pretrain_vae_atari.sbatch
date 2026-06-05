#!/bin/bash
#SBATCH --job-name=pretrain_vae_atari
#SBATCH --account=huytran1-ic
#SBATCH --partition=csl
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/pretrain_vae_atari.o%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu

ulimit -n 4096  # raise file descriptor limit

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook)"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/minjae5/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

conda activate irpo
mkdir -p logs

# Pre-train VAE for all 4 Atari envs in parallel (seeds 0-4)
# Each env gets a separate GPU slot, so sequential is fine

for ENV in pacman amidar bankheist alien; do
    for SEED in {0..4}; do
        python3 pretrain_vae.py --env $ENV --seed $SEED --epochs 50 --collect-samples 100000 &
        sleep 2
    done
done

# Wait for all to finish
wait

echo "✓ VAE pre-training complete for all environments"
