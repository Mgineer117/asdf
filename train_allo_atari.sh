#!/bin/bash

# Define the 4 Atari environments
ENVS=("pacman" "amidar" "bankheist" "alien")

echo "Starting ALLO training for Atari environments..."

# Loop through the environments and assign them to GPUs 0 through 3
for i in "${!ENVS[@]}"; do
    ENV=${ENVS[$i]}
    GPU=$i
    
    echo "  -> Launching ${ENV} on GPU ${GPU}"
    
    # Run the training script in the background, redirecting output to a log file
    python train_models.py --env $ENV --gpu-idx $GPU > "log_${ENV}_allo.out" 2>&1 &
    
    # Sleep for 5 minutes before launching the next one, except for the last environment
    if [ $i -lt $((${#ENVS[@]} - 1)) ]; then
        echo "  -> Sleeping for 5 minutes to prevent RAM bottleneck..."
        sleep 300
    fi
done

echo "All 4 jobs have been launched in the background!"
echo "You can monitor the progress of each environment using:"
echo "  tail -f log_pacman_allo.out"
echo "  tail -f log_amidar_allo.out"
echo "  tail -f log_bankheist_allo.out"
echo "  tail -f log_alien_allo.out"
