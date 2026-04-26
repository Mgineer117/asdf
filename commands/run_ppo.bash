#!/usr/bin/env bash
for script in commands/ppo/*.sbatch; do
    echo "Submitting $script..."
    sbatch "$script"
    sleep 2
done