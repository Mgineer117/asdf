#!/bin/bash

# Create logs directory locally
mkdir -p logs

echo "========================================================="
echo " Submitting all 28 Experiment Jobs to the Cluster"
echo " Each job (algorithm + environment) is allocated to 1 Node"
echo " running 5 seeds across 1 or 2 GPUs depending on partition."
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

echo "[SUBMIT] scripts/bankheist_ppo.sbatch"
sbatch scripts/bankheist_ppo.sbatch
sleep 0.5

echo "[SUBMIT] scripts/bankheist_irpo.sbatch"
sbatch scripts/bankheist_irpo.sbatch
sleep 0.5

echo "[SUBMIT] scripts/bankheist_maml.sbatch"
sbatch scripts/bankheist_maml.sbatch
sleep 0.5

echo "[SUBMIT] scripts/bankheist_hrl.sbatch"
sbatch scripts/bankheist_hrl.sbatch
sleep 0.5

echo "[SUBMIT] scripts/bankheist_drnd.sbatch"
sbatch scripts/bankheist_drnd.sbatch
sleep 0.5

echo "[SUBMIT] scripts/bankheist_trpo.sbatch"
sbatch scripts/bankheist_trpo.sbatch
sleep 0.5

echo "[SUBMIT] scripts/bankheist_psne.sbatch"
sbatch scripts/bankheist_psne.sbatch
sleep 0.5

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
echo " All 28 jobs successfully submitted to queue!"
echo " Check progress using: squeue -u $(whoami)"
echo " Logs will be stored in: logs/"
echo "========================================================="
