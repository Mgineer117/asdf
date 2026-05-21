#!/bin/bash
#
# CSL SBATCH Scripts Quick Reference
# 
# Time Limit: 7 hours (07:00:00)
# GPU: 1x GPU
# CPU: 8 cores
# Memory: 32GB
#
# ════════════════════════════════════════════════════════════════
# SINGLE ALGORITHM + ENVIRONMENT
# ════════════════════════════════════════════════════════════════
#
# IRPO (default: pointmaze-v4):
#   sbatch commands/run_irpo_csl.sbatch
#
# IRPO with custom environment:
#   sbatch --export=ENV_NAME=maze-v1 commands/run_irpo_csl.sbatch
#
# IRPO with custom seed and project:
#   sbatch --export=ENV_NAME=antmaze,SEED=42,PROJECT=my-project commands/run_irpo_csl.sbatch
#
# PPO:
#   sbatch commands/run_ppo_csl.sbatch
#
# TRPO:
#   sbatch commands/run_trpo_csl.sbatch
#
# DRND:
#   sbatch commands/run_drnd_csl.sbatch
#
# ════════════════════════════════════════════════════════════════
# MULTI-SEED RUN (Sequential in single job)
# ════════════════════════════════════════════════════════════════
#
# Run IRPO on pointmaze-v4 with seeds 0,1,2:
#   sbatch --export=ALGO_NAME=irpo,ENV_NAME=pointmaze-v4,SEEDS="0 1 2" commands/run_multi_seed_csl.sbatch
#
# Run PPO on fourrooms with seeds 0-3:
#   sbatch --export=ALGO_NAME=ppo,ENV_NAME=fourrooms,SEEDS="0 1 2 3" commands/run_multi_seed_csl.sbatch
#
# ════════════════════════════════════════════════════════════════
# AVAILABLE ENVIRONMENTS
# ════════════════════════════════════════════════════════════════
#
# RL Environments:
#   - pointmaze-v4      (default)
#   - pointmaze-v3, pointmaze-v2, pointmaze-v1
#   - maze-v1, maze-v2
#   - fourrooms
#   - halfcheetah
#   - walker
#   - hopper
#   - ant
#   - fetchreach
#   - fetchpush
#   - fetchpusheasy
#   - antmaze
#   - amidar
#   - pacman
#
# ════════════════════════════════════════════════════════════════
# AVAILABLE ALGORITHMS
# ════════════════════════════════════════════════════════════════
#
#   - irpo   (Intrinsic Reward Policy Optimization)
#   - ppo    (Proximal Policy Optimization)
#   - trpo   (Trust Region Policy Optimization)
#   - drnd   (Diversity-driven Random Network Distillation)
#   - hrl    (Hierarchical Reinforcement Learning)
#   - maml   (Model-Agnostic Meta-Learning)
#   - psne   (Policy Search with Natural Evolution)
#   - htrpo  (Hierarchical TRPO)
#
# ════════════════════════════════════════════════════════════════
# MONITOR JOBS
# ════════════════════════════════════════════════════════════════
#
# View job status:
#   squeue -u $USER
#
# View job details:
#   scontrol show job <job_id>
#
# Cancel job:
#   scancel <job_id>
#
# View logs:
#   tail -f logs/irpo_<job_id>.log
#
# ════════════════════════════════════════════════════════════════
# NOTES
# ════════════════════════════════════════════════════════════════
#
# - Logs are saved to: logs/<algo>_<job_id>.log
# - WandB logging is enabled with --wandb-mode online
# - Ensure conda environment is properly configured
# - GPU must be available on partition=gpu
#
