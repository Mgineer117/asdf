#!/bin/bash
set -e

# Run each algorithm for a very small number of timesteps to ensure it doesn't crash during initialization and the first update block.

for algo in ppo trpo psne maml irpo; do
    echo "======================================"
    echo "Testing algorithm: $algo"
    echo "======================================"
    python main.py --algo $algo --env CartPole-v1 --timesteps 100 --batch-size 64 --num-minibatch 4 --minibatch-size 16 || {
        echo "Algorithm $algo FAILED on CartPole-v1!"
        exit 1
    }
done

for algo in ppo trpo psne maml irpo; do
    echo "======================================"
    echo "Testing algorithm: $algo on Atari (Amidar)"
    echo "======================================"
    python main.py --algo $algo --env amidar --timesteps 256 --batch-size 256 --num-minibatch 4 --minibatch-size 64 --cnn-mode simultaneous || {
        echo "Algorithm $algo FAILED on Atari Amidar (simultaneous CNN mode)!"
        exit 1
    }
done

echo "ALL TESTS PASSED SUCCESSFULLY!"
