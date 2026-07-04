#!/bin/bash

# Change to the directory where this script is located (scripts/)
cd "$(dirname "$0")"

# Set the environment variables to exclude iRPO
export ALGO_SUBSET="ppo,trpo,psne,maml,hrl,drnd"
export SUBMIT_NAME="submit_no_irpo.sh"

echo "Generating sbatch files for $ALGO_SUBSET..."
python3 generate_sbatch.py

echo "Submitting jobs via $SUBMIT_NAME..."
bash submit_no_irpo.sh
