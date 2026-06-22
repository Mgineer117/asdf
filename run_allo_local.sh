#!/bin/bash

# Ensure logs directory exists
mkdir -p logs

echo "=========================================================="
echo " Starting ALLO training for Atari environments on 3 GPUs"
echo " (train_models.py handles all seeds internally)"
echo "=========================================================="

echo "Launching Alien then Pacman on GPU 0..."
(
    echo "Starting Alien (All Seeds) on GPU 0..."
    python3 train_models.py --project Exp --env alien --algo irpo --int-reward-type allo --gpu-idx 0 > logs/alien_allo.log 2>&1
    
    echo "Starting Pacman (All Seeds) on GPU 0..."
    python3 train_models.py --project Exp --env pacman --algo irpo --int-reward-type allo --gpu-idx 0 > logs/pacman_allo.log 2>&1
) &

echo "Launching Amidar on GPU 1..."
(
    echo "Starting Amidar (All Seeds) on GPU 1..."
    python3 train_models.py --project Exp --env amidar --algo irpo --int-reward-type allo --gpu-idx 1 > logs/amidar_allo.log 2>&1
) &

echo "Launching Bankheist on GPU 2..."
(
    echo "Starting Bankheist (All Seeds) on GPU 2..."
    python3 train_models.py --project Exp --env bankheist --algo irpo --int-reward-type allo --gpu-idx 2 > logs/bankheist_allo.log 2>&1
) &

echo "=========================================================="
echo " All 3 GPUs are now running in the background!"
echo " You can monitor output using: tail -f logs/alien_allo.log"
echo " Use 'nvidia-smi' or 'top' to monitor system resources."
echo "=========================================================="

# Wait for all background subshells to complete
wait

echo "All training runs have finished successfully!"
