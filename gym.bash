nohup python3 main.py --project gym --env-name hopper --algo-name irpo --num-runs 10 --int-reward-type random --rendering &
nohup python3 main.py --project gym --env-name walker --algo-name irpo --num-runs 10 --int-reward-type random --rendering &
nohup python3 main.py --project gym --env-name halfcheetah --algo-name irpo --num-runs 10 --int-reward-type random --rendering &
nohup python3 main.py --project gym --env-name ant --algo-name irpo --num-runs 10 --int-reward-type random --rendering &

nohup python3 main.py --project gym --env-name hopper --algo-name ppo --num-runs 10 --rendering &
nohup python3 main.py --project gym --env-name walker --algo-name ppo --num-runs 10 --rendering &
nohup python3 main.py --project gym --env-name halfcheetah --algo-name ppo --num-runs 10 --rendering &
nohup python3 main.py --project gym --env-name ant --algo-name ppo --num-runs 10 --rendering &