#!/bin/bash
# =========================================================
#  submit_baselines.sh – submit all chains for pacman + amidar + bankheist + alien
#  algos: ppo, trpo, psne, hrl
# =========================================================
set -euo pipefail
mkdir -p logs

# Collected job IDs (every chain segment), used for the start-time report.
SUBMITTED_JIDS=()

echo "========================================================="
echo " Submitting 4 envs × 4 algos with dependency chains"
echo "========================================================="

# ── alien / hrl  [csl, 1 chain(s)] ──
JID_ALIEN_HRL_C1=$(sbatch --parsable scripts/alien_hrl.sbatch)
echo "[SUBMIT] scripts/alien_hrl.sbatch → job ID: ${JID_ALIEN_HRL_C1}"
SUBMITTED_JIDS+=("${JID_ALIEN_HRL_C1}")

# ── alien / ppo  [eng-research-gpu, 3 chain(s)] ──
JID_ALIEN_PPO_C1=$(sbatch --parsable scripts/alien_ppo.sbatch)
echo "[SUBMIT] scripts/alien_ppo.sbatch → job ID: ${JID_ALIEN_PPO_C1}"
SUBMITTED_JIDS+=("${JID_ALIEN_PPO_C1}")
JID_ALIEN_PPO_C2=$(sbatch --parsable --dependency=afterany:${JID_ALIEN_PPO_C1} scripts/alien_ppo_c2.sbatch)
echo "[SUBMIT] scripts/alien_ppo_c2.sbatch → job ID: ${JID_ALIEN_PPO_C2}"
SUBMITTED_JIDS+=("${JID_ALIEN_PPO_C2}")
JID_ALIEN_PPO_C3=$(sbatch --parsable --dependency=afterany:${JID_ALIEN_PPO_C2} scripts/alien_ppo_c3.sbatch)
echo "[SUBMIT] scripts/alien_ppo_c3.sbatch → job ID: ${JID_ALIEN_PPO_C3}"
SUBMITTED_JIDS+=("${JID_ALIEN_PPO_C3}")

# ── alien / psne  [eng-research-gpu, 3 chain(s)] ──
JID_ALIEN_PSNE_C1=$(sbatch --parsable scripts/alien_psne.sbatch)
echo "[SUBMIT] scripts/alien_psne.sbatch → job ID: ${JID_ALIEN_PSNE_C1}"
SUBMITTED_JIDS+=("${JID_ALIEN_PSNE_C1}")
JID_ALIEN_PSNE_C2=$(sbatch --parsable --dependency=afterany:${JID_ALIEN_PSNE_C1} scripts/alien_psne_c2.sbatch)
echo "[SUBMIT] scripts/alien_psne_c2.sbatch → job ID: ${JID_ALIEN_PSNE_C2}"
SUBMITTED_JIDS+=("${JID_ALIEN_PSNE_C2}")
JID_ALIEN_PSNE_C3=$(sbatch --parsable --dependency=afterany:${JID_ALIEN_PSNE_C2} scripts/alien_psne_c3.sbatch)
echo "[SUBMIT] scripts/alien_psne_c3.sbatch → job ID: ${JID_ALIEN_PSNE_C3}"
SUBMITTED_JIDS+=("${JID_ALIEN_PSNE_C3}")

# ── alien / trpo  [eng-research-gpu, 3 chain(s)] ──
JID_ALIEN_TRPO_C1=$(sbatch --parsable scripts/alien_trpo.sbatch)
echo "[SUBMIT] scripts/alien_trpo.sbatch → job ID: ${JID_ALIEN_TRPO_C1}"
SUBMITTED_JIDS+=("${JID_ALIEN_TRPO_C1}")
JID_ALIEN_TRPO_C2=$(sbatch --parsable --dependency=afterany:${JID_ALIEN_TRPO_C1} scripts/alien_trpo_c2.sbatch)
echo "[SUBMIT] scripts/alien_trpo_c2.sbatch → job ID: ${JID_ALIEN_TRPO_C2}"
SUBMITTED_JIDS+=("${JID_ALIEN_TRPO_C2}")
JID_ALIEN_TRPO_C3=$(sbatch --parsable --dependency=afterany:${JID_ALIEN_TRPO_C2} scripts/alien_trpo_c3.sbatch)
echo "[SUBMIT] scripts/alien_trpo_c3.sbatch → job ID: ${JID_ALIEN_TRPO_C3}"
SUBMITTED_JIDS+=("${JID_ALIEN_TRPO_C3}")

# ── amidar / hrl  [csl, 1 chain(s)] ──
JID_AMIDAR_HRL_C1=$(sbatch --parsable scripts/amidar_hrl.sbatch)
echo "[SUBMIT] scripts/amidar_hrl.sbatch → job ID: ${JID_AMIDAR_HRL_C1}"
SUBMITTED_JIDS+=("${JID_AMIDAR_HRL_C1}")

# ── amidar / ppo  [eng-research-gpu, 3 chain(s)] ──
JID_AMIDAR_PPO_C1=$(sbatch --parsable scripts/amidar_ppo.sbatch)
echo "[SUBMIT] scripts/amidar_ppo.sbatch → job ID: ${JID_AMIDAR_PPO_C1}"
SUBMITTED_JIDS+=("${JID_AMIDAR_PPO_C1}")
JID_AMIDAR_PPO_C2=$(sbatch --parsable --dependency=afterany:${JID_AMIDAR_PPO_C1} scripts/amidar_ppo_c2.sbatch)
echo "[SUBMIT] scripts/amidar_ppo_c2.sbatch → job ID: ${JID_AMIDAR_PPO_C2}"
SUBMITTED_JIDS+=("${JID_AMIDAR_PPO_C2}")
JID_AMIDAR_PPO_C3=$(sbatch --parsable --dependency=afterany:${JID_AMIDAR_PPO_C2} scripts/amidar_ppo_c3.sbatch)
echo "[SUBMIT] scripts/amidar_ppo_c3.sbatch → job ID: ${JID_AMIDAR_PPO_C3}"
SUBMITTED_JIDS+=("${JID_AMIDAR_PPO_C3}")

# ── amidar / psne  [eng-research-gpu, 3 chain(s)] ──
JID_AMIDAR_PSNE_C1=$(sbatch --parsable scripts/amidar_psne.sbatch)
echo "[SUBMIT] scripts/amidar_psne.sbatch → job ID: ${JID_AMIDAR_PSNE_C1}"
SUBMITTED_JIDS+=("${JID_AMIDAR_PSNE_C1}")
JID_AMIDAR_PSNE_C2=$(sbatch --parsable --dependency=afterany:${JID_AMIDAR_PSNE_C1} scripts/amidar_psne_c2.sbatch)
echo "[SUBMIT] scripts/amidar_psne_c2.sbatch → job ID: ${JID_AMIDAR_PSNE_C2}"
SUBMITTED_JIDS+=("${JID_AMIDAR_PSNE_C2}")
JID_AMIDAR_PSNE_C3=$(sbatch --parsable --dependency=afterany:${JID_AMIDAR_PSNE_C2} scripts/amidar_psne_c3.sbatch)
echo "[SUBMIT] scripts/amidar_psne_c3.sbatch → job ID: ${JID_AMIDAR_PSNE_C3}"
SUBMITTED_JIDS+=("${JID_AMIDAR_PSNE_C3}")

# ── amidar / trpo  [eng-research-gpu, 3 chain(s)] ──
JID_AMIDAR_TRPO_C1=$(sbatch --parsable scripts/amidar_trpo.sbatch)
echo "[SUBMIT] scripts/amidar_trpo.sbatch → job ID: ${JID_AMIDAR_TRPO_C1}"
SUBMITTED_JIDS+=("${JID_AMIDAR_TRPO_C1}")
JID_AMIDAR_TRPO_C2=$(sbatch --parsable --dependency=afterany:${JID_AMIDAR_TRPO_C1} scripts/amidar_trpo_c2.sbatch)
echo "[SUBMIT] scripts/amidar_trpo_c2.sbatch → job ID: ${JID_AMIDAR_TRPO_C2}"
SUBMITTED_JIDS+=("${JID_AMIDAR_TRPO_C2}")
JID_AMIDAR_TRPO_C3=$(sbatch --parsable --dependency=afterany:${JID_AMIDAR_TRPO_C2} scripts/amidar_trpo_c3.sbatch)
echo "[SUBMIT] scripts/amidar_trpo_c3.sbatch → job ID: ${JID_AMIDAR_TRPO_C3}"
SUBMITTED_JIDS+=("${JID_AMIDAR_TRPO_C3}")

# ── bankheist / hrl  [csl, 1 chain(s)] ──
JID_BANKHEIST_HRL_C1=$(sbatch --parsable scripts/bankheist_hrl.sbatch)
echo "[SUBMIT] scripts/bankheist_hrl.sbatch → job ID: ${JID_BANKHEIST_HRL_C1}"
SUBMITTED_JIDS+=("${JID_BANKHEIST_HRL_C1}")

# ── bankheist / ppo  [eng-research-gpu, 3 chain(s)] ──
JID_BANKHEIST_PPO_C1=$(sbatch --parsable scripts/bankheist_ppo.sbatch)
echo "[SUBMIT] scripts/bankheist_ppo.sbatch → job ID: ${JID_BANKHEIST_PPO_C1}"
SUBMITTED_JIDS+=("${JID_BANKHEIST_PPO_C1}")
JID_BANKHEIST_PPO_C2=$(sbatch --parsable --dependency=afterany:${JID_BANKHEIST_PPO_C1} scripts/bankheist_ppo_c2.sbatch)
echo "[SUBMIT] scripts/bankheist_ppo_c2.sbatch → job ID: ${JID_BANKHEIST_PPO_C2}"
SUBMITTED_JIDS+=("${JID_BANKHEIST_PPO_C2}")
JID_BANKHEIST_PPO_C3=$(sbatch --parsable --dependency=afterany:${JID_BANKHEIST_PPO_C2} scripts/bankheist_ppo_c3.sbatch)
echo "[SUBMIT] scripts/bankheist_ppo_c3.sbatch → job ID: ${JID_BANKHEIST_PPO_C3}"
SUBMITTED_JIDS+=("${JID_BANKHEIST_PPO_C3}")

# ── bankheist / psne  [eng-research-gpu, 3 chain(s)] ──
JID_BANKHEIST_PSNE_C1=$(sbatch --parsable scripts/bankheist_psne.sbatch)
echo "[SUBMIT] scripts/bankheist_psne.sbatch → job ID: ${JID_BANKHEIST_PSNE_C1}"
SUBMITTED_JIDS+=("${JID_BANKHEIST_PSNE_C1}")
JID_BANKHEIST_PSNE_C2=$(sbatch --parsable --dependency=afterany:${JID_BANKHEIST_PSNE_C1} scripts/bankheist_psne_c2.sbatch)
echo "[SUBMIT] scripts/bankheist_psne_c2.sbatch → job ID: ${JID_BANKHEIST_PSNE_C2}"
SUBMITTED_JIDS+=("${JID_BANKHEIST_PSNE_C2}")
JID_BANKHEIST_PSNE_C3=$(sbatch --parsable --dependency=afterany:${JID_BANKHEIST_PSNE_C2} scripts/bankheist_psne_c3.sbatch)
echo "[SUBMIT] scripts/bankheist_psne_c3.sbatch → job ID: ${JID_BANKHEIST_PSNE_C3}"
SUBMITTED_JIDS+=("${JID_BANKHEIST_PSNE_C3}")

# ── bankheist / trpo  [eng-research-gpu, 3 chain(s)] ──
JID_BANKHEIST_TRPO_C1=$(sbatch --parsable scripts/bankheist_trpo.sbatch)
echo "[SUBMIT] scripts/bankheist_trpo.sbatch → job ID: ${JID_BANKHEIST_TRPO_C1}"
SUBMITTED_JIDS+=("${JID_BANKHEIST_TRPO_C1}")
JID_BANKHEIST_TRPO_C2=$(sbatch --parsable --dependency=afterany:${JID_BANKHEIST_TRPO_C1} scripts/bankheist_trpo_c2.sbatch)
echo "[SUBMIT] scripts/bankheist_trpo_c2.sbatch → job ID: ${JID_BANKHEIST_TRPO_C2}"
SUBMITTED_JIDS+=("${JID_BANKHEIST_TRPO_C2}")
JID_BANKHEIST_TRPO_C3=$(sbatch --parsable --dependency=afterany:${JID_BANKHEIST_TRPO_C2} scripts/bankheist_trpo_c3.sbatch)
echo "[SUBMIT] scripts/bankheist_trpo_c3.sbatch → job ID: ${JID_BANKHEIST_TRPO_C3}"
SUBMITTED_JIDS+=("${JID_BANKHEIST_TRPO_C3}")

# ── pacman / hrl  [csl, 1 chain(s)] ──
JID_PACMAN_HRL_C1=$(sbatch --parsable scripts/pacman_hrl.sbatch)
echo "[SUBMIT] scripts/pacman_hrl.sbatch → job ID: ${JID_PACMAN_HRL_C1}"
SUBMITTED_JIDS+=("${JID_PACMAN_HRL_C1}")

# ── pacman / ppo  [eng-research-gpu, 3 chain(s)] ──
JID_PACMAN_PPO_C1=$(sbatch --parsable scripts/pacman_ppo.sbatch)
echo "[SUBMIT] scripts/pacman_ppo.sbatch → job ID: ${JID_PACMAN_PPO_C1}"
SUBMITTED_JIDS+=("${JID_PACMAN_PPO_C1}")
JID_PACMAN_PPO_C2=$(sbatch --parsable --dependency=afterany:${JID_PACMAN_PPO_C1} scripts/pacman_ppo_c2.sbatch)
echo "[SUBMIT] scripts/pacman_ppo_c2.sbatch → job ID: ${JID_PACMAN_PPO_C2}"
SUBMITTED_JIDS+=("${JID_PACMAN_PPO_C2}")
JID_PACMAN_PPO_C3=$(sbatch --parsable --dependency=afterany:${JID_PACMAN_PPO_C2} scripts/pacman_ppo_c3.sbatch)
echo "[SUBMIT] scripts/pacman_ppo_c3.sbatch → job ID: ${JID_PACMAN_PPO_C3}"
SUBMITTED_JIDS+=("${JID_PACMAN_PPO_C3}")

# ── pacman / psne  [eng-research-gpu, 3 chain(s)] ──
JID_PACMAN_PSNE_C1=$(sbatch --parsable scripts/pacman_psne.sbatch)
echo "[SUBMIT] scripts/pacman_psne.sbatch → job ID: ${JID_PACMAN_PSNE_C1}"
SUBMITTED_JIDS+=("${JID_PACMAN_PSNE_C1}")
JID_PACMAN_PSNE_C2=$(sbatch --parsable --dependency=afterany:${JID_PACMAN_PSNE_C1} scripts/pacman_psne_c2.sbatch)
echo "[SUBMIT] scripts/pacman_psne_c2.sbatch → job ID: ${JID_PACMAN_PSNE_C2}"
SUBMITTED_JIDS+=("${JID_PACMAN_PSNE_C2}")
JID_PACMAN_PSNE_C3=$(sbatch --parsable --dependency=afterany:${JID_PACMAN_PSNE_C2} scripts/pacman_psne_c3.sbatch)
echo "[SUBMIT] scripts/pacman_psne_c3.sbatch → job ID: ${JID_PACMAN_PSNE_C3}"
SUBMITTED_JIDS+=("${JID_PACMAN_PSNE_C3}")

# ── pacman / trpo  [eng-research-gpu, 3 chain(s)] ──
JID_PACMAN_TRPO_C1=$(sbatch --parsable scripts/pacman_trpo.sbatch)
echo "[SUBMIT] scripts/pacman_trpo.sbatch → job ID: ${JID_PACMAN_TRPO_C1}"
SUBMITTED_JIDS+=("${JID_PACMAN_TRPO_C1}")
JID_PACMAN_TRPO_C2=$(sbatch --parsable --dependency=afterany:${JID_PACMAN_TRPO_C1} scripts/pacman_trpo_c2.sbatch)
echo "[SUBMIT] scripts/pacman_trpo_c2.sbatch → job ID: ${JID_PACMAN_TRPO_C2}"
SUBMITTED_JIDS+=("${JID_PACMAN_TRPO_C2}")
JID_PACMAN_TRPO_C3=$(sbatch --parsable --dependency=afterany:${JID_PACMAN_TRPO_C2} scripts/pacman_trpo_c3.sbatch)
echo "[SUBMIT] scripts/pacman_trpo_c3.sbatch → job ID: ${JID_PACMAN_TRPO_C3}"
SUBMITTED_JIDS+=("${JID_PACMAN_TRPO_C3}")

echo "========================================================="
echo " All jobs submitted. Waiting 10s for the scheduler to register them..."
echo "========================================================="
sleep 10

# Comma-separated job-id list for squeue.
JOBS_CSV=$(IFS=,; echo "${SUBMITTED_JIDS[*]}")

echo "========================================================="
echo " Estimated start times (squeue --start)"
echo "========================================================="
# --start prints the scheduler's estimated START_TIME and the pending
# reason (Resources/Priority/Dependency). Restricted to the jobs we just
# submitted; falls back to all of the user's jobs if the id filter errors.
squeue --start --jobs="$JOBS_CSV" \
    -o "%.12i %.34j %.16P %.20S %.10T %.18r" \
    || squeue --start -u "$(whoami)"

echo "========================================================="
echo " Live queue status (squeue)"
echo "========================================================="
squeue --jobs="$JOBS_CSV" \
    -o "%.12i %.34j %.16P %.10T %.10M %.10l %.18r" \
    || squeue -u "$(whoami)"

echo "Re-check anytime with:  squeue --start --jobs=$JOBS_CSV"
