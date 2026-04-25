#!/usr/bin/env bash
# Run four ablations in parallel, one per A4000 GPU:
#   GPU 0 → ablation_thompson
#   GPU 1 → ablation_num_exp_updates
#   GPU 2 → ablation_num_options
#   GPU 3 → ablation_arch
#
# Each child is launched with `nohup ... &` so it survives logout / SIGHUP,
# and this script returns to the shell immediately (no trailing `wait`).
# Monitor with: tail -f log/A4000_*.out  or  nvidia-smi.

set -u
mkdir -p log

GPU=0 nohup bash ablation_thompson.bash         > log/A4000_thompson.out         2>&1 &
GPU=1 nohup bash ablation_num_exp_updates.bash  > log/A4000_num_exp_updates.out  2>&1 &
GPU=2 nohup bash ablation_num_options.bash      > log/A4000_num_options.out      2>&1 &
GPU=3 nohup bash ablation_arch.bash             > log/A4000_arch.out             2>&1 &

disown -a
echo "Launched 4 ablations in background. PIDs:"
jobs -p
