#!/bin/bash

# Create logs directory locally
mkdir -p logs

echo "========================================================="
echo " Submitting all 14 Experiment Jobs to the Cluster"
echo " Each job (algorithm + environment) is allocated to 1 Node"
echo " running 10 seeds (0-9) in parallel on 1 GPU."
echo "========================================================="

echo "[SUBMIT] scripts/pacman_ppo.sbatch"
sbatch scripts/pacman_ppo.sbatch
sleep 0.5

echo "[SUBMIT] scripts/pacman_irpo.sbatch"
sbatch scripts/pacman_irpo.sbatch
sleep 0.5

echo "[SUBMIT] scripts/pacman_maml.sbatch"
sbatch scripts/pacman_maml.sbatch
sleep 0.5

echo "[SUBMIT] scripts/pacman_hrl.sbatch"
sbatch scripts/pacman_hrl.sbatch
sleep 0.5

echo "[SUBMIT] scripts/pacman_drnd.sbatch"
sbatch scripts/pacman_drnd.sbatch
sleep 0.5

echo "[SUBMIT] scripts/pacman_trpo.sbatch"
sbatch scripts/pacman_trpo.sbatch
sleep 0.5

echo "[SUBMIT] scripts/pacman_psne.sbatch"
sbatch scripts/pacman_psne.sbatch
sleep 0.5

echo "[SUBMIT] scripts/amidar_ppo.sbatch"
sbatch scripts/amidar_ppo.sbatch
sleep 0.5

echo "[SUBMIT] scripts/amidar_irpo.sbatch"
sbatch scripts/amidar_irpo.sbatch
sleep 0.5

echo "[SUBMIT] scripts/amidar_maml.sbatch"
sbatch scripts/amidar_maml.sbatch
sleep 0.5

echo "[SUBMIT] scripts/amidar_hrl.sbatch"
sbatch scripts/amidar_hrl.sbatch
sleep 0.5

echo "[SUBMIT] scripts/amidar_drnd.sbatch"
sbatch scripts/amidar_drnd.sbatch
sleep 0.5

echo "[SUBMIT] scripts/amidar_trpo.sbatch"
sbatch scripts/amidar_trpo.sbatch
sleep 0.5

echo "[SUBMIT] scripts/amidar_psne.sbatch"
sbatch scripts/amidar_psne.sbatch
sleep 0.5

echo "========================================================="
echo " All 14 jobs successfully submitted to queue!"
echo " Check progress using: squeue -u $(whoami)"
echo " Logs will be stored in: logs/"
echo "========================================================="
