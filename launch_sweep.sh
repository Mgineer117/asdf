#!/bin/bash

echo "Initializing new WandB sweep for IRPO Random parameter search..."

# Run the python script with count=0 just to create the sweep and exit
OUTPUT=$(python search_irpo_random.py --count 0 2>&1)

# Extract the sweep ID from the script output
SWEEP_ID=$(echo "$OUTPUT" | grep "Created NEW wandb sweep with ID:" | awk '{print $NF}')

if [ -z "$SWEEP_ID" ]; then
    echo "Failed to create WandB sweep. Output was:"
    echo "$OUTPUT"
    exit 1
fi

echo "Successfully created sweep ID: $SWEEP_ID"
echo "Submitting 20 parallel agents (10 per GPU) to IllinoisComputes-GPU on 2 GPUs..."

# Pass the sweep ID to the sbatch array script
sbatch sweep_worker.sbatch $SWEEP_ID
