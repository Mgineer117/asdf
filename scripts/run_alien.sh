#!/bin/bash

# Create logs directory locally
mkdir -p logs

echo "========================================================="
echo " Submitting all 7 Experiment Jobs for alien to the Cluster"
echo " Each job (algorithm + environment) is allocated to 1 Node"
echo " running 5 seeds (0-4) across 1 GPU."
echo "========================================================="

echo "[SUBMIT] scripts/alien_ppo.sbatch"
sbatch scripts/alien_ppo.sbatch
sleep 0.5

echo "[SUBMIT] scripts/alien_irpo.sbatch"
sbatch scripts/alien_irpo.sbatch
sleep 0.5

echo "[SUBMIT] scripts/alien_maml.sbatch"
sbatch scripts/alien_maml.sbatch
sleep 0.5

echo "[SUBMIT] scripts/alien_hrl.sbatch"
sbatch scripts/alien_hrl.sbatch
sleep 0.5

echo "[SUBMIT] scripts/alien_drnd.sbatch"
sbatch scripts/alien_drnd.sbatch
sleep 0.5

echo "[SUBMIT] scripts/alien_trpo.sbatch"
sbatch scripts/alien_trpo.sbatch
sleep 0.5

echo "[SUBMIT] scripts/alien_psne.sbatch"
sbatch scripts/alien_psne.sbatch
sleep 0.5

echo "========================================================="
echo " All 7 jobs for alien successfully submitted to queue!"
echo " Check progress using: squeue -u $(whoami)"
echo " Logs will be stored in: logs/"
echo "========================================================="
