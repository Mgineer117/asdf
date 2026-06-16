#!/bin/bash

# Create logs directory locally
mkdir -p logs

echo "========================================================="
echo " Submitting all 7 Experiment Jobs for bankheist to the Cluster"
echo " Each job (algorithm + environment) is allocated to 1 Node"
echo " running 5 seeds (0-4) across 1 GPU."
echo "========================================================="

echo "[SUBMIT] scripts/bankheist_ppo.sbatch"
OUTPUT=$(sbatch scripts/bankheist_ppo.sbatch)
echo "$OUTPUT"
JOBID=$(echo "$OUTPUT" | grep -o '[0-9]*$')
echo "Waiting 5s for SLURM to estimate start time..."
sleep 5
squeue -j $JOBID --start
sleep 0.5

echo "[SUBMIT] scripts/bankheist_irpo.sbatch"
OUTPUT=$(sbatch scripts/bankheist_irpo.sbatch)
echo "$OUTPUT"
JOBID=$(echo "$OUTPUT" | grep -o '[0-9]*$')
echo "Waiting 5s for SLURM to estimate start time..."
sleep 5
squeue -j $JOBID --start
sleep 0.5

echo "[SUBMIT] scripts/bankheist_maml.sbatch"
OUTPUT=$(sbatch scripts/bankheist_maml.sbatch)
echo "$OUTPUT"
JOBID=$(echo "$OUTPUT" | grep -o '[0-9]*$')
echo "Waiting 5s for SLURM to estimate start time..."
sleep 5
squeue -j $JOBID --start
sleep 0.5

echo "[SUBMIT] scripts/bankheist_hrl.sbatch"
OUTPUT=$(sbatch scripts/bankheist_hrl.sbatch)
echo "$OUTPUT"
JOBID=$(echo "$OUTPUT" | grep -o '[0-9]*$')
echo "Waiting 5s for SLURM to estimate start time..."
sleep 5
squeue -j $JOBID --start
sleep 0.5

echo "[SUBMIT] scripts/bankheist_drnd.sbatch"
OUTPUT=$(sbatch scripts/bankheist_drnd.sbatch)
echo "$OUTPUT"
JOBID=$(echo "$OUTPUT" | grep -o '[0-9]*$')
echo "Waiting 5s for SLURM to estimate start time..."
sleep 5
squeue -j $JOBID --start
sleep 0.5

echo "[SUBMIT] scripts/bankheist_trpo.sbatch"
OUTPUT=$(sbatch scripts/bankheist_trpo.sbatch)
echo "$OUTPUT"
JOBID=$(echo "$OUTPUT" | grep -o '[0-9]*$')
echo "Waiting 5s for SLURM to estimate start time..."
sleep 5
squeue -j $JOBID --start
sleep 0.5

echo "[SUBMIT] scripts/bankheist_psne.sbatch"
OUTPUT=$(sbatch scripts/bankheist_psne.sbatch)
echo "$OUTPUT"
JOBID=$(echo "$OUTPUT" | grep -o '[0-9]*$')
echo "Waiting 5s for SLURM to estimate start time..."
sleep 5
squeue -j $JOBID --start
sleep 0.5

echo "========================================================="
echo " All 7 jobs for bankheist successfully submitted to queue!"
echo " Check progress using: squeue -u $(whoami)"
echo " Logs will be stored in: logs/"
echo "========================================================="
