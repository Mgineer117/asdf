#!/bin/bash
# Change to the root directory (one level up from where the script is)
cd "$(dirname "$0")/.."

# Set the environment variables to exclude iRPO
export ALGO_SUBSET="ppo,trpo,psne,maml,hrl,drnd"
export SUBMIT_NAME="submit_no_irpo.sh"

echo "Generating sbatch files for $ALGO_SUBSET..."
python3 scripts/generate_sbatch.py

echo "Submitting jobs via $SUBMIT_NAME..."
bash scripts/submit_no_irpo.sh
