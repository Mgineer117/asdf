#!/bin/bash

# Create logs directory locally
mkdir -p logs

echo "========================================================="
echo " Submitting all 7 Experiment Jobs for amidar to the Cluster"
echo " Each job (algorithm + environment) is allocated to 1 Node"
echo " running 5 seeds (0-4) across 1 GPU."
echo "========================================================="

echo "[SUBMIT] scripts/amidar_ppo.sbatch"
OUTPUT=$(sbatch scripts/amidar_ppo.sbatch)
echo "$OUTPUT"
JOBID=$(echo "$OUTPUT" | grep -o '[0-9]*$')
echo "Waiting 5s for SLURM to estimate start time..."
sleep 5
squeue -j $JOBID --start
sleep 0.5

echo "[SUBMIT] scripts/amidar_irpo.sbatch"
OUTPUT=$(sbatch scripts/amidar_irpo.sbatch)
echo "$OUTPUT"
JOBID=$(echo "$OUTPUT" | grep -o '[0-9]*$')
echo "Waiting 5s for SLURM to estimate start time..."
sleep 5
squeue -j $JOBID --start
sleep 0.5

echo "[SUBMIT] scripts/amidar_maml.sbatch"
OUTPUT=$(sbatch scripts/amidar_maml.sbatch)
echo "$OUTPUT"
JOBID=$(echo "$OUTPUT" | grep -o '[0-9]*$')
echo "Waiting 5s for SLURM to estimate start time..."
sleep 5
squeue -j $JOBID --start
sleep 0.5

echo "[SUBMIT] scripts/amidar_hrl.sbatch"
OUTPUT=$(sbatch scripts/amidar_hrl.sbatch)
echo "$OUTPUT"
JOBID=$(echo "$OUTPUT" | grep -o '[0-9]*$')
echo "Waiting 5s for SLURM to estimate start time..."
sleep 5
squeue -j $JOBID --start
sleep 0.5

echo "[SUBMIT] scripts/amidar_drnd.sbatch"
OUTPUT=$(sbatch scripts/amidar_drnd.sbatch)
echo "$OUTPUT"
JOBID=$(echo "$OUTPUT" | grep -o '[0-9]*$')
echo "Waiting 5s for SLURM to estimate start time..."
sleep 5
squeue -j $JOBID --start
sleep 0.5

echo "[SUBMIT] scripts/amidar_trpo.sbatch"
OUTPUT=$(sbatch scripts/amidar_trpo.sbatch)
echo "$OUTPUT"
JOBID=$(echo "$OUTPUT" | grep -o '[0-9]*$')
echo "Waiting 5s for SLURM to estimate start time..."
sleep 5
squeue -j $JOBID --start
sleep 0.5

echo "[SUBMIT] scripts/amidar_psne.sbatch"
OUTPUT=$(sbatch scripts/amidar_psne.sbatch)
echo "$OUTPUT"
JOBID=$(echo "$OUTPUT" | grep -o '[0-9]*$')
echo "Waiting 5s for SLURM to estimate start time..."
sleep 5
squeue -j $JOBID --start
sleep 0.5

echo "========================================================="
echo " All 7 jobs for amidar successfully submitted to queue!"
echo " Check progress using: squeue -u $(whoami)"
echo " Logs will be stored in: logs/"
echo "========================================================="
