#!/usr/bin/env bash
# Europa — NEW_IRPO pacman: hrl on GPU 0, maml on GPU 1, 5 seeds each.

mkdir -p log

nohup python3 main.py --project NEW_IRPO --env-name fetchreach --algo-name maml --seed 0 --gpu-idx 0 > log/NEW_IRPO_pacman_irpo_seed0.out 2>&1 &
sleep 1
nohup python3 main.py --project NEW_IRPO --env-name fetchreach --algo-name maml --seed 1 --gpu-idx 0 > log/NEW_IRPO_pacman_irpo_seed1.out 2>&1 &
sleep 1
nohup python3 main.py --project NEW_IRPO --env-name fetchreach --algo-name maml --seed 2 --gpu-idx 0 > log/NEW_IRPO_pacman_irpo_seed2.out 2>&1 &
sleep 1
nohup python3 main.py --project NEW_IRPO --env-name fetchreach --algo-name maml --seed 3 --gpu-idx 0 > log/NEW_IRPO_pacman_irpo_seed3.out 2>&1 &
sleep 1
nohup python3 main.py --project NEW_IRPO --env-name fetchreach --algo-name maml --seed 4 --gpu-idx 0 > log/NEW_IRPO_pacman_irpo_seed4.out 2>&1 &
sleep 1

nohup python3 main.py --project NEW_IRPO --env-name fetchreach --algo-name maml --seed 5 --gpu-idx 0 > log/NEW_IRPO_pacman_irpo_seed5.out 2>&1 &
sleep 1
nohup python3 main.py --project NEW_IRPO --env-name fetchreach --algo-name maml --seed 6 --gpu-idx 0 > log/NEW_IRPO_pacman_irpo_seed6.out 2>&1 &
sleep 1
nohup python3 main.py --project NEW_IRPO --env-name fetchreach --algo-name maml --seed 7 --gpu-idx 0 > log/NEW_IRPO_pacman_irpo_seed7.out 2>&1 &
sleep 1
nohup python3 main.py --project NEW_IRPO --env-name fetchreach --algo-name maml --seed 8 --gpu-idx 0 > log/NEW_IRPO_pacman_irpo_seed8.out 2>&1 &
sleep 1
nohup python3 main.py --project NEW_IRPO --env-name fetchreach --algo-name maml --seed 9 --gpu-idx 0 > log/NEW_IRPO_pacman_irpo_seed9.out 2>&1 &

disown -a
echo "Launched 10 NEW_IRPO jobs."
