#!/bin/bash

# Ensure logs directory exists
mkdir -p logs

echo "=========================================================="
echo " Starting ALLO training for Atari environments on 3 GPUs"
echo " (Running sequentially over seeds to conserve VRAM)"
echo "=========================================================="

echo "Launching Alien and Pacman sequentially on GPU 0..."
(
for SEED in {0..4}; do
    echo "Starting Alien Seed $SEED on GPU 0..."
    python3 train_models.py --project Exp --env alien --algo irpo --int-reward-type allo --seed $SEED --gpu-idx 0 > logs/alien_allo_${SEED}.log 2>&1
    
    echo "Starting Pacman Seed $SEED on GPU 0..."
    python3 train_models.py --project Exp --env pacman --algo irpo --int-reward-type allo --seed $SEED --gpu-idx 0 > logs/pacman_allo_${SEED}.log 2>&1
done
) &

echo "Launching Amidar sequentially on GPU 1..."
(
for SEED in {0..4}; do
    echo "Starting Amidar Seed $SEED on GPU 1..."
    python3 train_models.py --project Exp --env amidar --algo irpo --int-reward-type allo --seed $SEED --gpu-idx 1 > logs/amidar_allo_${SEED}.log 2>&1
done
) &

echo "Launching Bankheist sequentially on GPU 2..."
(
for SEED in {0..4}; do
    echo "Starting Bankheist Seed $SEED on GPU 2..."
    python3 train_models.py --project Exp --env bankheist --algo irpo --int-reward-type allo --seed $SEED --gpu-idx 2 > logs/bankheist_allo_${SEED}.log 2>&1
done
) &

echo "=========================================================="
echo " All 3 GPUs are now running their queues in the background!"
echo " You can monitor output using: tail -f logs/alien_allo_0.log"
echo " Use 'nvidia-smi' or 'top' to monitor system resources."
echo "=========================================================="

# Wait for all background subshells to complete
wait

echo "All training runs have finished successfully!"
