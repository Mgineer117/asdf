#!/bin/bash

# Create logs directory locally
mkdir -p logs

echo "========================================================="
echo " Submitting all 28 Experiment Jobs to the Cluster"
echo " Each job (algorithm + environment) is allocated to 1 Node"
echo " running 5 seeds across 1 GPU."
echo "========================================================="

echo "[SUBMIT] scripts/pacman_ppo.sbatch"
OUTPUT=$(sbatch scripts/pacman_ppo.sbatch)
echo "$OUTPUT"
JOBID=$(echo "$OUTPUT" | grep -o '[0-9]*$')
echo "Waiting 5s for SLURM to estimate start time..."
sleep 5
squeue -j $JOBID --start
sleep 0.5

echo "[SUBMIT] scripts/pacman_irpo.sbatch"
OUTPUT=$(sbatch scripts/pacman_irpo.sbatch)
echo "$OUTPUT"
JOBID=$(echo "$OUTPUT" | grep -o '[0-9]*$')
echo "Waiting 5s for SLURM to estimate start time..."
sleep 5
squeue -j $JOBID --start
sleep 0.5

echo "[SUBMIT] scripts/pacman_maml.sbatch"
OUTPUT=$(sbatch scripts/pacman_maml.sbatch)
echo "$OUTPUT"
JOBID=$(echo "$OUTPUT" | grep -o '[0-9]*$')
echo "Waiting 5s for SLURM to estimate start time..."
sleep 5
squeue -j $JOBID --start
sleep 0.5

echo "[SUBMIT] scripts/pacman_hrl.sbatch"
OUTPUT=$(sbatch scripts/pacman_hrl.sbatch)
echo "$OUTPUT"
JOBID=$(echo "$OUTPUT" | grep -o '[0-9]*$')
echo "Waiting 5s for SLURM to estimate start time..."
sleep 5
squeue -j $JOBID --start
sleep 0.5

echo "[SUBMIT] scripts/pacman_drnd.sbatch"
OUTPUT=$(sbatch scripts/pacman_drnd.sbatch)
echo "$OUTPUT"
JOBID=$(echo "$OUTPUT" | grep -o '[0-9]*$')
echo "Waiting 5s for SLURM to estimate start time..."
sleep 5
squeue -j $JOBID --start
sleep 0.5

echo "[SUBMIT] scripts/pacman_trpo.sbatch"
OUTPUT=$(sbatch scripts/pacman_trpo.sbatch)
echo "$OUTPUT"
JOBID=$(echo "$OUTPUT" | grep -o '[0-9]*$')
echo "Waiting 5s for SLURM to estimate start time..."
sleep 5
squeue -j $JOBID --start
sleep 0.5

echo "[SUBMIT] scripts/pacman_psne.sbatch"
OUTPUT=$(sbatch scripts/pacman_psne.sbatch)
echo "$OUTPUT"
JOBID=$(echo "$OUTPUT" | grep -o '[0-9]*$')
echo "Waiting 5s for SLURM to estimate start time..."
sleep 5
squeue -j $JOBID --start
sleep 0.5

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
echo " All 28 jobs successfully submitted to queue!"
echo " Check progress using: squeue -u $(whoami)"
echo " Logs will be stored in: logs/"
echo "========================================================="
