# A4000
# TRAP REWARD
# TRAIN FOR ANTMAZE

nohup python3 main.py --project trap_reward --env-name fourrooms-v2 --algo-name irpo --num-runs 10 --num-exp-updates 5 --gpu-idx 0 &
nohup python3 main.py --project trap_reward --env-name fourrooms-v2 --algo-name irpo --num-runs 10 --num-exp-updates 10 --gpu-idx 0 &
nohup python3 main.py --project trap_reward --env-name fourrooms-v2 --algo-name irpo --num-runs 10 --num-exp-updates 15 --gpu-idx 0 &
nohup python3 main.py --project trap_reward --env-name fourrooms-v2 --algo-name irpo --num-runs 10 --num-exp-updates 20 --gpu-idx 0 &

nohup python3 main.py --project trap_reward --env-name fourrooms-v2 --algo-name irpo --num-runs 10 --num-exp-updates 10 --gpu-idx 0 --temperature 0.5 &
nohup python3 main.py --project trap_reward --env-name fourrooms-v2 --algo-name irpo --num-runs 10 --num-exp-updates 10 --gpu-idx 0 --temperature 0.8 &

# nohup python3 main.py --project pacman --env-name pacman --algo-name irpo --num-runs 10 --num-exp-updates 5 --gpu-idx 3 &
# nohup python3 main.py --project pacman --env-name pacman --algo-name ppo --num-runs 10 --gpu-idx 1 &
# nohup python3 main.py --project pacman --env-name pacman --algo-name drnd --num-runs 10 --gpu-idx 1 &
# nohup python3 main.py --project pacman --env-name pacman --algo-name psne --num-runs 10 --gpu-idx 1 &
# nohup python3 main.py --project pacman --env-name pacman --algo-name hrl --num-runs 10 --gpu-idx 1 &
# nohup python3 main.py --project pacman --env-name pacman --algo-name maml --num-runs 10 --gpu-idx 1 &

nohup python3 main.py --project maml --env-name fourrooms-v1 --algo-name maml --num-runs 10 --gpu-idx 2 &
nohup python3 main.py --project maml --env-name maze-v1 --algo-name maml --num-runs 10 --gpu-idx 2 &
nohup python3 main.py --project maml --env-name maze-v2 --algo-name maml --num-runs 10 --gpu-idx 2 &
nohup python3 main.py --project maml --env-name pointmaze-v1 --algo-name maml --num-runs 10 --gpu-idx 3 &
nohup python3 main.py --project maml --env-name pointmaze-v2 --algo-name maml --num-runs 10 --gpu-idx 3 &
nohup python3 main.py --project maml --env-name fetchreach --algo-name maml --num-runs 10 --gpu-idx 3 &

# nohup python3 train_allo.py --project allo --env-name antmaze-v1 --num-runs 10 --gpu-idx 1 &
# nohup python3 train_allo.py --project allo --env-name antmaze-v2 --num-runs 10 --gpu-idx 1 &
# nohup python3 train_allo.py --project allo --env-name antmaze-v3 --num-runs 10 --gpu-idx 1 &