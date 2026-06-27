#!/bin/bash
# Submit N rounds of all 4 searches (default N=3 → 24 h of search per algo).
# Each job is 8 h; they queue up and all join the same persistent sweep.
# Run from repo root: bash search/submit_search.sh [N]
#
# To stop: scancel --name=search_ppo --name=search_trpo \
#                  --name=search_psne --name=search_drnd
# To reset sweeps: rm search/logs/*_sweep_id.txt

set -e
cd "$(dirname "$0")/.."   # repo root

N=${1:-8}
mkdir -p search/logs

echo "Submitting $N rounds per algorithm..."

for round in $(seq 1 $N); do
    sbatch search/search_ppo.sbatch
    sbatch search/search_trpo.sbatch
    sbatch search/search_psne.sbatch
    sbatch search/search_drnd.sbatch
done

echo ""
echo "Queued $N rounds × 4 jobs = $((N * 4)) jobs total ($((N * 8)) h of wall-clock per algo)."
echo "Monitor: squeue -u \$USER"
