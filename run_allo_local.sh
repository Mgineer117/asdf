#!/bin/bash

# Ensure logs directory exists
mkdir -p logs

echo "=========================================================="
echo " Starting ALLO training for all 4 Atari environments"
echo " All 4 envs run simultaneously; seeds are sequential"
echo " GPU 0: alien | GPU 1: pacman | GPU 2: amidar + bankheist"
echo "=========================================================="

python3 train_models.py --project Exp --env alien     --algo irpo --int-reward-type allo --gpu-idx 0 > logs/alien_allo.log     2>&1 &
python3 train_models.py --project Exp --env pacman    --algo irpo --int-reward-type allo --gpu-idx 1 > logs/pacman_allo.log    2>&1 &
python3 train_models.py --project Exp --env amidar    --algo irpo --int-reward-type allo --gpu-idx 2 > logs/amidar_allo.log    2>&1 &
python3 train_models.py --project Exp --env bankheist --algo irpo --int-reward-type allo --gpu-idx 2 > logs/bankheist_allo.log 2>&1 &

echo "All 4 jobs launched! Monitor with:"
echo "  tail -f logs/alien_allo.log"
echo "  tail -f logs/pacman_allo.log"
echo "  tail -f logs/amidar_allo.log"
echo "  tail -f logs/bankheist_allo.log"

wait
echo "All training runs have finished successfully!"
