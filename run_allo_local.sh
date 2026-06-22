#!/bin/bash

# Ensure logs directory exists
mkdir -p logs

echo "=========================================================="
echo " Starting ALLO training for Atari environments on 3 GPUs"
echo " (train_models.py handles all seeds internally)"
echo "=========================================================="

echo "Launching Alien on GPU 0..."
python3 train_models.py --project Exp --env alien --algo irpo --int-reward-type allo --gpu-idx 0 > logs/alien_allo.log 2>&1 &

echo "Launching Pacman on GPU 1..."
python3 train_models.py --project Exp --env pacman --algo irpo --int-reward-type allo --gpu-idx 1 > logs/pacman_allo.log 2>&1 &

echo "Launching Amidar then Bankheist on GPU 2..."
(
    python3 train_models.py --project Exp --env amidar --algo irpo --int-reward-type allo --gpu-idx 2 > logs/amidar_allo.log 2>&1
    python3 train_models.py --project Exp --env bankheist --algo irpo --int-reward-type allo --gpu-idx 2 > logs/bankheist_allo.log 2>&1
) &

echo "=========================================================="
echo " All jobs launched!"
echo " Monitor with: tail -f logs/alien_allo.log"
echo "=========================================================="

wait
echo "All training runs have finished successfully!"
