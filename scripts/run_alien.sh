#!/bin/bash

# Create logs directory locally
mkdir -p logs

echo "========================================================="
echo " Submitting all 7 Experiment Jobs for alien to the Cluster"
echo " Each job (algorithm + environment) is allocated to 1 Node"
echo " running 5 seeds (0-4) across 1 GPU."
echo "========================================================="

echo "[SUBMIT] scripts/alien_ppo.sbatch"
OUTPUT=$(sbatch scripts/alien_ppo.sbatch)
echo "$OUTPUT"
JOBID=$(echo "$OUTPUT" | grep -o '[0-9]*$')
echo "Waiting 5s for SLURM to estimate start time..."
sleep 5
squeue -j $JOBID --start
sleep 0.5

echo "[SUBMIT] scripts/alien_irpo.sbatch"
OUTPUT=$(sbatch scripts/alien_irpo.sbatch)
echo "$OUTPUT"
JOBID=$(echo "$OUTPUT" | grep -o '[0-9]*$')
echo "Waiting 5s for SLURM to estimate start time..."
sleep 5
squeue -j $JOBID --start
sleep 0.5

echo "[SUBMIT] scripts/alien_maml.sbatch"
OUTPUT=$(sbatch scripts/alien_maml.sbatch)
echo "$OUTPUT"
JOBID=$(echo "$OUTPUT" | grep -o '[0-9]*$')
echo "Waiting 5s for SLURM to estimate start time..."
sleep 5
squeue -j $JOBID --start
sleep 0.5

echo "[SUBMIT] scripts/alien_hrl.sbatch"
OUTPUT=$(sbatch scripts/alien_hrl.sbatch)
echo "$OUTPUT"
JOBID=$(echo "$OUTPUT" | grep -o '[0-9]*$')
echo "Waiting 5s for SLURM to estimate start time..."
sleep 5
squeue -j $JOBID --start
sleep 0.5

echo "[SUBMIT] scripts/alien_drnd.sbatch"
OUTPUT=$(sbatch scripts/alien_drnd.sbatch)
echo "$OUTPUT"
JOBID=$(echo "$OUTPUT" | grep -o '[0-9]*$')
echo "Waiting 5s for SLURM to estimate start time..."
sleep 5
squeue -j $JOBID --start
sleep 0.5

echo "[SUBMIT] scripts/alien_trpo.sbatch"
OUTPUT=$(sbatch scripts/alien_trpo.sbatch)
echo "$OUTPUT"
JOBID=$(echo "$OUTPUT" | grep -o '[0-9]*$')
echo "Waiting 5s for SLURM to estimate start time..."
sleep 5
squeue -j $JOBID --start
sleep 0.5

echo "[SUBMIT] scripts/alien_psne.sbatch"
OUTPUT=$(sbatch scripts/alien_psne.sbatch)
echo "$OUTPUT"
JOBID=$(echo "$OUTPUT" | grep -o '[0-9]*$')
echo "Waiting 5s for SLURM to estimate start time..."
sleep 5
squeue -j $JOBID --start
sleep 0.5

echo "========================================================="
echo " All 7 jobs for alien successfully submitted to queue!"
echo " Check progress using: squeue -u $(whoami)"
echo " Logs will be stored in: logs/"
echo "========================================================="
