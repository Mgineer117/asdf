#!/bin/bash

# Create logs directory locally
mkdir -p logs

echo "========================================================="
echo " Submitting all 7 Experiment Jobs for bankheist to the Cluster"
echo " Each job (algorithm + environment) is allocated to 1 Node"
echo " running 5 seeds (0-4) across 1 or 2 GPUs."
echo "========================================================="

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

echo "========================================================="
echo " All 7 jobs for bankheist successfully submitted to queue!"
echo " Check progress using: squeue -u $(whoami)"
echo " Logs will be stored in: logs/"
echo "========================================================="
