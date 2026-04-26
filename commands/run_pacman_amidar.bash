#!/usr/bin/env bash
# Submit pacman + amidar SLURM jobs for every algorithm.

ALGOS=(drnd hrl irpo ppo psne trpo)
ENVS=(pacman amidar)

for algo in "${ALGOS[@]}"; do
    for env in "${ENVS[@]}"; do
        script="commands/${algo}/run_${env}.sbatch"
        if [ -f "${script}" ]; then
            echo "Submitting ${script}..."
            sbatch "${script}"
            sleep 2
        else
            echo "Skipping missing ${script}"
        fi
    done
done
