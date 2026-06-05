#!/bin/bash
#SBATCH --job-name=atari_vae_pretrain_finetune
#SBATCH --account=huytran1-ic
#SBATCH --partition=csl
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=10-00:00:00
#SBATCH --output=logs/atari_vae_pretrain_finetune.o%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu

ulimit -n 4096  # raise file descriptor limit

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook)"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/minjae5/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

conda activate irpo
mkdir -p logs

# ============================================================================
# STAGE 1: Pre-train VAE on random policy samples
# ============================================================================
echo "=========================================="
echo "STAGE 1: Pre-training VAE encoders..."
echo "=========================================="

for ENV in pacman amidar bankheist alien; do
    for SEED in {0..4}; do
        python3 pretrain_vae.py --env $ENV --seed $SEED --epochs 50 --collect-samples 100000 &
        sleep 1
    done
done

# Wait for all pre-training to complete
wait
echo "✓ Stage 1 complete: All VAE encoders pre-trained"
echo ""

# ============================================================================
# STAGE 2: Fine-tune VAE + Train IRPO with random intrinsic reward
# ============================================================================
echo "=========================================="
echo "STAGE 2: Fine-tuning VAE + IRPO training..."
echo "=========================================="
echo "IRPO will load pre-trained encoders and continue fine-tuning"
echo ""

# Run IRPO for each Atari env with 5 seeds in parallel
# IRPO will:
#  1. Load pre-trained VAE encoder (if exists)
#  2. Continue training VAE via VAE loss every 5 steps
#  3. Train policy via IRPO with random intrinsic reward
#
for ENV in pacman amidar bankheist alien; do
    for SEED in {0..4}; do
        python3 main.py --project IRPO_ATARI_PRETRAIN_FINETUNE --env $ENV --algo irpo \
            --int-reward-type random --seed $SEED --gpu-idx 0 &
        sleep 3
    done
done

# Wait for all IRPO training to complete
wait
echo ""
echo "=========================================="
echo "✓ Stage 2 complete: All IRPO training finished"
echo "=========================================="
echo "Results saved to log/train_log/"
