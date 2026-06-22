#!/bin/bash

# Ensure logs directory exists
mkdir -p logs

echo "=========================================================="
echo " Starting ALLO training for Atari environments on 3 GPUs"
echo "=========================================================="

echo "Launching Alien and Pacman on GPU 0..."
for SEED in {0..4}; do
    python3 main.py --project Atari --env alien --algo irpo --int-reward-type allo --seed $SEED --gpu-idx 0 > logs/alien_allo_${SEED}.log 2>&1 &
    sleep 2
    python3 main.py --project Atari --env pacman --algo irpo --int-reward-type allo --seed $SEED --gpu-idx 0 > logs/pacman_allo_${SEED}.log 2>&1 &
    sleep 2
done

echo "Launching Amidar on GPU 1..."
for SEED in {0..4}; do
    python3 main.py --project Atari --env amidar --algo irpo --int-reward-type allo --seed $SEED --gpu-idx 1 > logs/amidar_allo_${SEED}.log 2>&1 &
    sleep 2
done

echo "Launching Bankheist on GPU 2..."
for SEED in {0..4}; do
    python3 main.py --project Atari --env bankheist --algo irpo --int-reward-type allo --seed $SEED --gpu-idx 2 > logs/bankheist_allo_${SEED}.log 2>&1 &
    sleep 2
done

echo "=========================================================="
echo " All jobs have been launched in the background!"
echo " You can monitor output using: tail -f logs/alien_allo_0.log"
echo " Use 'nvidia-smi' or 'top' to monitor system resources."
echo "=========================================================="

# Wait for all background processes to complete
wait

echo "All training runs have finished successfully!"
