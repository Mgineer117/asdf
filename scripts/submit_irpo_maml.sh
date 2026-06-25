#!/bin/bash
# =========================================================
#  submit_irpo_maml.sh – submit all chains for pacman + amidar
#  algos: irpo_allo, irpo_random, maml
# =========================================================
set -euo pipefail
mkdir -p logs

echo "========================================================="
echo " Submitting 2 envs × 3 algos with dependency chains"
echo "========================================================="

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

echo "========================================================="
echo " All jobs submitted. Check with: squeue -u $(whoami)"
echo "========================================================="
