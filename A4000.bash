#!/usr/bin/env bash
# Run four ablations in parallel, one per A4000 GPU:
#   GPU 0 → ablation_thompson
#   GPU 1 → ablation_num_exp_updates
#   GPU 2 → ablation_num_options
#   GPU 3 → ablation_arch
#
# Each child script already backgrounds its own runs and waits internally.

set -u
mkdir -p log

GPU=0 bash ablation_thompson.bash         > log/A4000_thompson.out         2>&1 &
GPU=1 bash ablation_num_exp_updates.bash  > log/A4000_num_exp_updates.out  2>&1 &
GPU=2 bash ablation_num_options.bash      > log/A4000_num_options.out      2>&1 &
GPU=3 bash ablation_arch.bash             > log/A4000_arch.out             2>&1 &

wait
