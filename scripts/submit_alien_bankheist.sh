#!/bin/bash
# =========================================================
#  submit_alien_bankheist.sh – submit all chains for alien + bankheist
#  algos: irpo_random, irpo_allo, maml
# =========================================================
set -euo pipefail
mkdir -p logs

# Collected job IDs (every chain segment), used for the start-time report.
SUBMITTED_JIDS=()

echo "========================================================="
echo " Submitting 2 envs × 3 algos with dependency chains"
echo "========================================================="

# ── alien / irpo_allo  [IllinoisComputes-GPU, 2 chain(s)] ──
JID_ALIEN_IRPO_ALLO_C1=$(sbatch --parsable scripts/alien_irpo_allo.sbatch)
echo "[SUBMIT] scripts/alien_irpo_allo.sbatch → job ID: ${JID_ALIEN_IRPO_ALLO_C1}"
SUBMITTED_JIDS+=("${JID_ALIEN_IRPO_ALLO_C1}")
JID_ALIEN_IRPO_ALLO_C2=$(sbatch --parsable --dependency=afterany:${JID_ALIEN_IRPO_ALLO_C1} scripts/alien_irpo_allo_c2.sbatch)
echo "[SUBMIT] scripts/alien_irpo_allo_c2.sbatch → job ID: ${JID_ALIEN_IRPO_ALLO_C2}"
SUBMITTED_JIDS+=("${JID_ALIEN_IRPO_ALLO_C2}")

# ── alien / irpo_random  [IllinoisComputes-GPU, 2 chain(s)] ──
JID_ALIEN_IRPO_RANDOM_C1=$(sbatch --parsable scripts/alien_irpo_random.sbatch)
echo "[SUBMIT] scripts/alien_irpo_random.sbatch → job ID: ${JID_ALIEN_IRPO_RANDOM_C1}"
SUBMITTED_JIDS+=("${JID_ALIEN_IRPO_RANDOM_C1}")
JID_ALIEN_IRPO_RANDOM_C2=$(sbatch --parsable --dependency=afterany:${JID_ALIEN_IRPO_RANDOM_C1} scripts/alien_irpo_random_c2.sbatch)
echo "[SUBMIT] scripts/alien_irpo_random_c2.sbatch → job ID: ${JID_ALIEN_IRPO_RANDOM_C2}"
SUBMITTED_JIDS+=("${JID_ALIEN_IRPO_RANDOM_C2}")

# ── alien / maml  [IllinoisComputes-GPU, 2 chain(s)] ──
JID_ALIEN_MAML_C1=$(sbatch --parsable scripts/alien_maml.sbatch)
echo "[SUBMIT] scripts/alien_maml.sbatch → job ID: ${JID_ALIEN_MAML_C1}"
SUBMITTED_JIDS+=("${JID_ALIEN_MAML_C1}")
JID_ALIEN_MAML_C2=$(sbatch --parsable --dependency=afterany:${JID_ALIEN_MAML_C1} scripts/alien_maml_c2.sbatch)
echo "[SUBMIT] scripts/alien_maml_c2.sbatch → job ID: ${JID_ALIEN_MAML_C2}"
SUBMITTED_JIDS+=("${JID_ALIEN_MAML_C2}")

# ── bankheist / irpo_allo  [IllinoisComputes-GPU, 2 chain(s)] ──
JID_BANKHEIST_IRPO_ALLO_C1=$(sbatch --parsable scripts/bankheist_irpo_allo.sbatch)
echo "[SUBMIT] scripts/bankheist_irpo_allo.sbatch → job ID: ${JID_BANKHEIST_IRPO_ALLO_C1}"
SUBMITTED_JIDS+=("${JID_BANKHEIST_IRPO_ALLO_C1}")
JID_BANKHEIST_IRPO_ALLO_C2=$(sbatch --parsable --dependency=afterany:${JID_BANKHEIST_IRPO_ALLO_C1} scripts/bankheist_irpo_allo_c2.sbatch)
echo "[SUBMIT] scripts/bankheist_irpo_allo_c2.sbatch → job ID: ${JID_BANKHEIST_IRPO_ALLO_C2}"
SUBMITTED_JIDS+=("${JID_BANKHEIST_IRPO_ALLO_C2}")

# ── bankheist / irpo_random  [IllinoisComputes-GPU, 2 chain(s)] ──
JID_BANKHEIST_IRPO_RANDOM_C1=$(sbatch --parsable scripts/bankheist_irpo_random.sbatch)
echo "[SUBMIT] scripts/bankheist_irpo_random.sbatch → job ID: ${JID_BANKHEIST_IRPO_RANDOM_C1}"
SUBMITTED_JIDS+=("${JID_BANKHEIST_IRPO_RANDOM_C1}")
JID_BANKHEIST_IRPO_RANDOM_C2=$(sbatch --parsable --dependency=afterany:${JID_BANKHEIST_IRPO_RANDOM_C1} scripts/bankheist_irpo_random_c2.sbatch)
echo "[SUBMIT] scripts/bankheist_irpo_random_c2.sbatch → job ID: ${JID_BANKHEIST_IRPO_RANDOM_C2}"
SUBMITTED_JIDS+=("${JID_BANKHEIST_IRPO_RANDOM_C2}")

# ── bankheist / maml  [IllinoisComputes-GPU, 2 chain(s)] ──
JID_BANKHEIST_MAML_C1=$(sbatch --parsable scripts/bankheist_maml.sbatch)
echo "[SUBMIT] scripts/bankheist_maml.sbatch → job ID: ${JID_BANKHEIST_MAML_C1}"
SUBMITTED_JIDS+=("${JID_BANKHEIST_MAML_C1}")
JID_BANKHEIST_MAML_C2=$(sbatch --parsable --dependency=afterany:${JID_BANKHEIST_MAML_C1} scripts/bankheist_maml_c2.sbatch)
echo "[SUBMIT] scripts/bankheist_maml_c2.sbatch → job ID: ${JID_BANKHEIST_MAML_C2}"
SUBMITTED_JIDS+=("${JID_BANKHEIST_MAML_C2}")

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
