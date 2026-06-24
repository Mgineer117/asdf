#!/bin/bash
# =========================================================
#  submit_atari.sh – submit all chains for pacman + amidar
# =========================================================
set -euo pipefail
mkdir -p logs

echo "========================================================="
echo " Submitting 2 envs × 8 algos with dependency chains"
echo "========================================================="

# ── amidar / drnd  [csl, 1 chain(s)] ──
JID_AMIDAR_DRND_C1=$(sbatch --parsable scripts/amidar_drnd.sbatch)
echo "[SUBMIT] scripts/amidar_drnd.sbatch → job ID: ${JID_AMIDAR_DRND_C1}"

# ── amidar / hrl  [csl, 1 chain(s)] ──
JID_AMIDAR_HRL_C1=$(sbatch --parsable scripts/amidar_hrl.sbatch)
echo "[SUBMIT] scripts/amidar_hrl.sbatch → job ID: ${JID_AMIDAR_HRL_C1}"

# ── amidar / irpo_allo  [IllinoisComputes-GPU, 2 chain(s)] ──
JID_AMIDAR_IRPO_ALLO_C1=$(sbatch --parsable scripts/amidar_irpo_allo.sbatch)
echo "[SUBMIT] scripts/amidar_irpo_allo.sbatch → job ID: ${JID_AMIDAR_IRPO_ALLO_C1}"
JID_AMIDAR_IRPO_ALLO_C2=$(sbatch --parsable --dependency=afterany:${JID_AMIDAR_IRPO_ALLO_C1} scripts/amidar_irpo_allo_c2.sbatch)
echo "[SUBMIT] scripts/amidar_irpo_allo_c2.sbatch → job ID: ${JID_AMIDAR_IRPO_ALLO_C2}"

# ── amidar / irpo_random  [IllinoisComputes-GPU, 2 chain(s)] ──
JID_AMIDAR_IRPO_RANDOM_C1=$(sbatch --parsable scripts/amidar_irpo_random.sbatch)
echo "[SUBMIT] scripts/amidar_irpo_random.sbatch → job ID: ${JID_AMIDAR_IRPO_RANDOM_C1}"
JID_AMIDAR_IRPO_RANDOM_C2=$(sbatch --parsable --dependency=afterany:${JID_AMIDAR_IRPO_RANDOM_C1} scripts/amidar_irpo_random_c2.sbatch)
echo "[SUBMIT] scripts/amidar_irpo_random_c2.sbatch → job ID: ${JID_AMIDAR_IRPO_RANDOM_C2}"

# ── amidar / maml  [IllinoisComputes-GPU, 2 chain(s)] ──
JID_AMIDAR_MAML_C1=$(sbatch --parsable scripts/amidar_maml.sbatch)
echo "[SUBMIT] scripts/amidar_maml.sbatch → job ID: ${JID_AMIDAR_MAML_C1}"
JID_AMIDAR_MAML_C2=$(sbatch --parsable --dependency=afterany:${JID_AMIDAR_MAML_C1} scripts/amidar_maml_c2.sbatch)
echo "[SUBMIT] scripts/amidar_maml_c2.sbatch → job ID: ${JID_AMIDAR_MAML_C2}"

# ── amidar / ppo  [eng-research-gpu, 3 chain(s)] ──
JID_AMIDAR_PPO_C1=$(sbatch --parsable scripts/amidar_ppo.sbatch)
echo "[SUBMIT] scripts/amidar_ppo.sbatch → job ID: ${JID_AMIDAR_PPO_C1}"
JID_AMIDAR_PPO_C2=$(sbatch --parsable --dependency=afterany:${JID_AMIDAR_PPO_C1} scripts/amidar_ppo_c2.sbatch)
echo "[SUBMIT] scripts/amidar_ppo_c2.sbatch → job ID: ${JID_AMIDAR_PPO_C2}"
JID_AMIDAR_PPO_C3=$(sbatch --parsable --dependency=afterany:${JID_AMIDAR_PPO_C2} scripts/amidar_ppo_c3.sbatch)
echo "[SUBMIT] scripts/amidar_ppo_c3.sbatch → job ID: ${JID_AMIDAR_PPO_C3}"

# ── amidar / psne  [eng-research-gpu, 3 chain(s)] ──
JID_AMIDAR_PSNE_C1=$(sbatch --parsable scripts/amidar_psne.sbatch)
echo "[SUBMIT] scripts/amidar_psne.sbatch → job ID: ${JID_AMIDAR_PSNE_C1}"
JID_AMIDAR_PSNE_C2=$(sbatch --parsable --dependency=afterany:${JID_AMIDAR_PSNE_C1} scripts/amidar_psne_c2.sbatch)
echo "[SUBMIT] scripts/amidar_psne_c2.sbatch → job ID: ${JID_AMIDAR_PSNE_C2}"
JID_AMIDAR_PSNE_C3=$(sbatch --parsable --dependency=afterany:${JID_AMIDAR_PSNE_C2} scripts/amidar_psne_c3.sbatch)
echo "[SUBMIT] scripts/amidar_psne_c3.sbatch → job ID: ${JID_AMIDAR_PSNE_C3}"

# ── amidar / trpo  [eng-research-gpu, 3 chain(s)] ──
JID_AMIDAR_TRPO_C1=$(sbatch --parsable scripts/amidar_trpo.sbatch)
echo "[SUBMIT] scripts/amidar_trpo.sbatch → job ID: ${JID_AMIDAR_TRPO_C1}"
JID_AMIDAR_TRPO_C2=$(sbatch --parsable --dependency=afterany:${JID_AMIDAR_TRPO_C1} scripts/amidar_trpo_c2.sbatch)
echo "[SUBMIT] scripts/amidar_trpo_c2.sbatch → job ID: ${JID_AMIDAR_TRPO_C2}"
JID_AMIDAR_TRPO_C3=$(sbatch --parsable --dependency=afterany:${JID_AMIDAR_TRPO_C2} scripts/amidar_trpo_c3.sbatch)
echo "[SUBMIT] scripts/amidar_trpo_c3.sbatch → job ID: ${JID_AMIDAR_TRPO_C3}"

# ── pacman / drnd  [csl, 1 chain(s)] ──
JID_PACMAN_DRND_C1=$(sbatch --parsable scripts/pacman_drnd.sbatch)
echo "[SUBMIT] scripts/pacman_drnd.sbatch → job ID: ${JID_PACMAN_DRND_C1}"

# ── pacman / hrl  [csl, 1 chain(s)] ──
JID_PACMAN_HRL_C1=$(sbatch --parsable scripts/pacman_hrl.sbatch)
echo "[SUBMIT] scripts/pacman_hrl.sbatch → job ID: ${JID_PACMAN_HRL_C1}"

# ── pacman / irpo_allo  [IllinoisComputes-GPU, 2 chain(s)] ──
JID_PACMAN_IRPO_ALLO_C1=$(sbatch --parsable scripts/pacman_irpo_allo.sbatch)
echo "[SUBMIT] scripts/pacman_irpo_allo.sbatch → job ID: ${JID_PACMAN_IRPO_ALLO_C1}"
JID_PACMAN_IRPO_ALLO_C2=$(sbatch --parsable --dependency=afterany:${JID_PACMAN_IRPO_ALLO_C1} scripts/pacman_irpo_allo_c2.sbatch)
echo "[SUBMIT] scripts/pacman_irpo_allo_c2.sbatch → job ID: ${JID_PACMAN_IRPO_ALLO_C2}"

# ── pacman / irpo_random  [IllinoisComputes-GPU, 2 chain(s)] ──
JID_PACMAN_IRPO_RANDOM_C1=$(sbatch --parsable scripts/pacman_irpo_random.sbatch)
echo "[SUBMIT] scripts/pacman_irpo_random.sbatch → job ID: ${JID_PACMAN_IRPO_RANDOM_C1}"
JID_PACMAN_IRPO_RANDOM_C2=$(sbatch --parsable --dependency=afterany:${JID_PACMAN_IRPO_RANDOM_C1} scripts/pacman_irpo_random_c2.sbatch)
echo "[SUBMIT] scripts/pacman_irpo_random_c2.sbatch → job ID: ${JID_PACMAN_IRPO_RANDOM_C2}"

# ── pacman / maml  [IllinoisComputes-GPU, 2 chain(s)] ──
JID_PACMAN_MAML_C1=$(sbatch --parsable scripts/pacman_maml.sbatch)
echo "[SUBMIT] scripts/pacman_maml.sbatch → job ID: ${JID_PACMAN_MAML_C1}"
JID_PACMAN_MAML_C2=$(sbatch --parsable --dependency=afterany:${JID_PACMAN_MAML_C1} scripts/pacman_maml_c2.sbatch)
echo "[SUBMIT] scripts/pacman_maml_c2.sbatch → job ID: ${JID_PACMAN_MAML_C2}"

# ── pacman / ppo  [eng-research-gpu, 3 chain(s)] ──
JID_PACMAN_PPO_C1=$(sbatch --parsable scripts/pacman_ppo.sbatch)
echo "[SUBMIT] scripts/pacman_ppo.sbatch → job ID: ${JID_PACMAN_PPO_C1}"
JID_PACMAN_PPO_C2=$(sbatch --parsable --dependency=afterany:${JID_PACMAN_PPO_C1} scripts/pacman_ppo_c2.sbatch)
echo "[SUBMIT] scripts/pacman_ppo_c2.sbatch → job ID: ${JID_PACMAN_PPO_C2}"
JID_PACMAN_PPO_C3=$(sbatch --parsable --dependency=afterany:${JID_PACMAN_PPO_C2} scripts/pacman_ppo_c3.sbatch)
echo "[SUBMIT] scripts/pacman_ppo_c3.sbatch → job ID: ${JID_PACMAN_PPO_C3}"

# ── pacman / psne  [eng-research-gpu, 3 chain(s)] ──
JID_PACMAN_PSNE_C1=$(sbatch --parsable scripts/pacman_psne.sbatch)
echo "[SUBMIT] scripts/pacman_psne.sbatch → job ID: ${JID_PACMAN_PSNE_C1}"
JID_PACMAN_PSNE_C2=$(sbatch --parsable --dependency=afterany:${JID_PACMAN_PSNE_C1} scripts/pacman_psne_c2.sbatch)
echo "[SUBMIT] scripts/pacman_psne_c2.sbatch → job ID: ${JID_PACMAN_PSNE_C2}"
JID_PACMAN_PSNE_C3=$(sbatch --parsable --dependency=afterany:${JID_PACMAN_PSNE_C2} scripts/pacman_psne_c3.sbatch)
echo "[SUBMIT] scripts/pacman_psne_c3.sbatch → job ID: ${JID_PACMAN_PSNE_C3}"

# ── pacman / trpo  [eng-research-gpu, 3 chain(s)] ──
JID_PACMAN_TRPO_C1=$(sbatch --parsable scripts/pacman_trpo.sbatch)
echo "[SUBMIT] scripts/pacman_trpo.sbatch → job ID: ${JID_PACMAN_TRPO_C1}"
JID_PACMAN_TRPO_C2=$(sbatch --parsable --dependency=afterany:${JID_PACMAN_TRPO_C1} scripts/pacman_trpo_c2.sbatch)
echo "[SUBMIT] scripts/pacman_trpo_c2.sbatch → job ID: ${JID_PACMAN_TRPO_C2}"
JID_PACMAN_TRPO_C3=$(sbatch --parsable --dependency=afterany:${JID_PACMAN_TRPO_C2} scripts/pacman_trpo_c3.sbatch)
echo "[SUBMIT] scripts/pacman_trpo_c3.sbatch → job ID: ${JID_PACMAN_TRPO_C3}"

echo "========================================================="
echo " All jobs submitted. Check with: squeue -u $(whoami)"
echo "========================================================="
