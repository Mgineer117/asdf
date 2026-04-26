#!/usr/bin/env bash
for script in commands/psne/*.sbatch; do
    echo "Submitting $script..."
    sbatch "$script"
    sleep 2
done