#!/bin/bash

# Create logs directory locally
mkdir -p logs

echo "========================================================="
echo " Submitting HRL Atari Jobs to IllinoisComputes-GPU Cluster"
echo " Each job is allocated to 1 Node running 5 seeds across 1 GPU."
echo "========================================================="

echo "[SUBMIT] scripts/pacman_hrl_ic.sbatch"
OUTPUT=$(sbatch scripts/pacman_hrl_ic.sbatch)
echo "$OUTPUT"
JOBID=$(echo "$OUTPUT" | grep -o '[0-9]*$')
echo "Waiting 5s for SLURM to estimate start time..."
sleep 5
squeue -j $JOBID --start
sleep 0.5

echo "[SUBMIT] scripts/amidar_hrl_ic.sbatch"
OUTPUT=$(sbatch scripts/amidar_hrl_ic.sbatch)
echo "$OUTPUT"
JOBID=$(echo "$OUTPUT" | grep -o '[0-9]*$')
echo "Waiting 5s for SLURM to estimate start time..."
sleep 5
squeue -j $JOBID --start
sleep 0.5

echo "[SUBMIT] scripts/bankheist_hrl_ic.sbatch"
OUTPUT=$(sbatch scripts/bankheist_hrl_ic.sbatch)
echo "$OUTPUT"
JOBID=$(echo "$OUTPUT" | grep -o '[0-9]*$')
echo "Waiting 5s for SLURM to estimate start time..."
sleep 5
squeue -j $JOBID --start
sleep 0.5

echo "[SUBMIT] scripts/alien_hrl_ic.sbatch"
OUTPUT=$(sbatch scripts/alien_hrl_ic.sbatch)
echo "$OUTPUT"
JOBID=$(echo "$OUTPUT" | grep -o '[0-9]*$')
echo "Waiting 5s for SLURM to estimate start time..."
sleep 5
squeue -j $JOBID --start
sleep 0.5

echo "========================================================="
echo " All 4 HRL jobs successfully submitted to IllinoisComputes-GPU!"
echo " Check progress using: squeue -u \$(whoami)"
echo " Logs will be stored in: logs/"
echo "========================================================="
