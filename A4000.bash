# A4000
# TRAP REWARD
# TRAIN FOR ANTMAZE

nohup python3 main.py --project trap_reward --env-name fourrooms-v2 --algo-name irpo --num-runs 10 --num-exp-updates 5 --rendering --gpu-idx 2 &
nohup python3 main.py --project trap_reward --env-name fourrooms-v2 --algo-name irpo --num-runs 10 --num-exp-updates 10 --rendering --gpu-idx 2 &
nohup python3 main.py --project trap_reward --env-name fourrooms-v2 --algo-name irpo --num-runs 10 --num-exp-updates 15 --rendering --gpu-idx 2 &
nohup python3 main.py --project trap_reward --env-name fourrooms-v2 --algo-name irpo --num-runs 10 --num-exp-updates 20 --rendering --gpu-idx 2 &

nohup python3 main.py --project pacman --env-name pacman --algo-name irpo --num-runs 10 --num-exp-updates 5 --rendering --gpu-idx 3 &
nohup python3 main.py --project pacman --env-name pacman --algo-name ppo --num-runs 10 --rendering --gpu-idx 3 &
nohup python3 main.py --project pacman --env-name pacman --algo-name drnd --num-runs 10 --rendering --gpu-idx 3 &

# nohup python3 train_allo.py --project allo --env-name antmaze-v1 --num-runs 10 --gpu-idx 1 &
# nohup python3 train_allo.py --project allo --env-name antmaze-v2 --num-runs 10 --gpu-idx 1 &
# nohup python3 train_allo.py --project allo --env-name antmaze-v3 --num-runs 10 --gpu-idx 1 &