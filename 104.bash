#!/usr/bin/env bash
# Europa — NEW_IRPO pacman: hrl on GPU 0, maml on GPU 1, 5 seeds each.

mkdir -p log

nohup python3 main.py --project NEW_IRPO --env-name pacman --algo-name irpo --int-reward-type random --seed 0 --gpu-idx 0 > log/NEW_IRPO_pacman_irpo_seed0.out 2>&1 &
sleep 3
nohup python3 main.py --project NEW_IRPO --env-name pacman --algo-name irpo --int-reward-type random --seed 1 --gpu-idx 0 > log/NEW_IRPO_pacman_irpo_seed1.out 2>&1 &
sleep 3
nohup python3 main.py --project NEW_IRPO --env-name pacman --algo-name irpo --int-reward-type random --seed 2 --gpu-idx 0 > log/NEW_IRPO_pacman_irpo_seed2.out 2>&1 &
sleep 3
# nohup python3 main.py --project NEW_IRPO --env-name pacman --algo-name irpo --int-reward-type random --seed 3 --gpu-idx 0 > log/NEW_IRPO_pacman_irpo_seed3.out 2>&1 &
# sleep 3
# nohup python3 main.py --project NEW_IRPO --env-name pacman --algo-name irpo --int-reward-type random --seed 4 --gpu-idx 0 > log/NEW_IRPO_pacman_irpo_seed4.out 2>&1 &
# sleep 3

nohup python3 main.py --project NEW_IRPO --env-name pacman --algo-name irpo --int-reward-type allo --seed 0 --gpu-idx 0 > log/NEW_IRPO_pacman_irpo_seed0.out 2>&1 &
sleep 3
nohup python3 main.py --project NEW_IRPO --env-name pacman --algo-name irpo --int-reward-type allo --seed 1 --gpu-idx 0 > log/NEW_IRPO_pacman_irpo_seed1.out 2>&1 &
sleep 3
nohup python3 main.py --project NEW_IRPO --env-name pacman --algo-name irpo --int-reward-type allo --seed 2 --gpu-idx 0 > log/NEW_IRPO_pacman_irpo_seed2.out 2>&1 &
# sleep 3
# nohup python3 main.py --project NEW_IRPO --env-name pacman --algo-name irpo --int-reward-type allo --seed 3 --gpu-idx 0 > log/NEW_IRPO_pacman_irpo_seed3.out 2>&1 &
# sleep 3
# nohup python3 main.py --project NEW_IRPO --env-name pacman --algo-name irpo --int-reward-type allo --seed 4 --gpu-idx 0 > log/NEW_IRPO_pacman_irpo_seed4.out 2>&1 &
# sleep 3

disown -a
echo "Launched 10 NEW_IRPO jobs."
