#!/bin/bash
# =========================================================
#  submit_maml.sh – submit all chains for pacman + amidar
#  algos: maml
# =========================================================
set -euo pipefail
mkdir -p logs

# Collected job IDs (every chain segment), used for the start-time report.
SUBMITTED_JIDS=()

echo "========================================================="
echo " Submitting 2 envs × 1 algos with dependency chains"
echo "========================================================="

# ── amidar / maml  [IllinoisComputes-GPU, 2 chain(s)] ──
JID_AMIDAR_MAML_C1=$(sbatch --parsable scripts/amidar_maml.sbatch)
echo "[SUBMIT] scripts/amidar_maml.sbatch → job ID: ${JID_AMIDAR_MAML_C1}"
SUBMITTED_JIDS+=("${JID_AMIDAR_MAML_C1}")
JID_AMIDAR_MAML_C2=$(sbatch --parsable --dependency=afterany:${JID_AMIDAR_MAML_C1} scripts/amidar_maml_c2.sbatch)
echo "[SUBMIT] scripts/amidar_maml_c2.sbatch → job ID: ${JID_AMIDAR_MAML_C2}"
SUBMITTED_JIDS+=("${JID_AMIDAR_MAML_C2}")

# ── pacman / maml  [IllinoisComputes-GPU, 2 chain(s)] ──
JID_PACMAN_MAML_C1=$(sbatch --parsable scripts/pacman_maml.sbatch)
echo "[SUBMIT] scripts/pacman_maml.sbatch → job ID: ${JID_PACMAN_MAML_C1}"
SUBMITTED_JIDS+=("${JID_PACMAN_MAML_C1}")
JID_PACMAN_MAML_C2=$(sbatch --parsable --dependency=afterany:${JID_PACMAN_MAML_C1} scripts/pacman_maml_c2.sbatch)
echo "[SUBMIT] scripts/pacman_maml_c2.sbatch → job ID: ${JID_PACMAN_MAML_C2}"
SUBMITTED_JIDS+=("${JID_PACMAN_MAML_C2}")

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
