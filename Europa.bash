#!/usr/bin/env bash
# Europa — NEW_IRPO pacman: hrl on GPU 0, maml on GPU 1, 5 seeds each.

mkdir -p log

nohup python3 main.py --project NEW_IRPO --env-name pacman --algo-name hrl --seed 0 --gpu-idx 2 > log/NEW_IRPO_pacman_hrl_seed0.out 2>&1 &
sleep 3
nohup python3 main.py --project NEW_IRPO --env-name pacman --algo-name hrl --seed 1 --gpu-idx 2 > log/NEW_IRPO_pacman_hrl_seed1.out 2>&1 &
sleep 3
nohup python3 main.py --project NEW_IRPO --env-name pacman --algo-name hrl --seed 2 --gpu-idx 2 > log/NEW_IRPO_pacman_hrl_seed2.out 2>&1 &
sleep 3
# nohup python3 main.py --project NEW_IRPO --env-name pacman --algo-name hrl --seed 3 --gpu-idx 0 > log/NEW_IRPO_pacman_hrl_seed3.out 2>&1 &
# sleep 3
# nohup python3 main.py --project NEW_IRPO --env-name pacman --algo-name hrl --seed 4 --gpu-idx 0 > log/NEW_IRPO_pacman_hrl_seed4.out 2>&1 &
# sleep 3

nohup python3 main.py --project NEW_IRPO --env-name pacman --algo-name maml --seed 0 --gpu-idx 1 > log/NEW_IRPO_pacman_maml_seed0.out 2>&1 &
sleep 3
nohup python3 main.py --project NEW_IRPO --env-name pacman --algo-name maml --seed 1 --gpu-idx 1 > log/NEW_IRPO_pacman_maml_seed1.out 2>&1 &
sleep 3
nohup python3 main.py --project NEW_IRPO --env-name pacman --algo-name maml --seed 2 --gpu-idx 1 > log/NEW_IRPO_pacman_maml_seed2.out 2>&1 &
# sleep 3
# nohup python3 main.py --project NEW_IRPO --env-name pacman --algo-name maml --seed 3 --gpu-idx 1 > log/NEW_IRPO_pacman_maml_seed3.out 2>&1 &
# sleep 3
# nohup python3 main.py --project NEW_IRPO --env-name pacman --algo-name maml --seed 4 --gpu-idx 1 > log/NEW_IRPO_pacman_maml_seed4.out 2>&1 &

disown -a
echo "Launched 10 NEW_IRPO jobs."
