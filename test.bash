# nohup python3 main.py --env-name fourrooms-v2 --algo-name ppo --rendering --num-runs 10 & 
# # nohup python3 main.py --env-name fourrooms-v2 --algo-name irpo --rendering --num-runs 10 #  --noise-std 0.0 & 
# nohup python3 main.py --env-name fourrooms-v2 --algo-name irpo --rendering --num-runs 10 --aggregation-method softmax & # --noise-std 0.3 & 
# nohup python3 main.py --env-name fourrooms-v2 --algo-name irpo --rendering --num-runs 10 --aggregation-method uniform & # --noise-std 0.5 & 
# nohup python3 main.py --env-name fourrooms-v2 --algo-name irpo --rendering --num-runs 10 --aggregation-method argmax & # --noise-std 1.0 & 

nohup python3 main.py --project ablation_irpo --env-name pointmaze-v4 --algo-name irpo --num-runs 10 --aggregation-method softmax --beta 0.0 &  # softmax -> argnax

nohup python3 main.py --project ablation_irpo --env-name pointmaze-v4 --algo-name irpo --num-runs 10 --aggregation-method softmax &  # softmax -> argnax
nohup python3 main.py --project ablation_irpo --env-name pointmaze-v4 --algo-name irpo --num-runs 10 --aggregation-method uniform &  # uniform -> argmax
nohup python3 main.py --project ablation_irpo --env-name pointmaze-v4 --algo-name irpo --num-runs 10 --aggregation-method argmax &  # argmax

nohup python3 main.py --project ablation_irpo --env-name fourrooms-v2 --algo-name irpo --num-runs 10 --aggregation-method argmax --noise-std 0.0 &  # argmax
nohup python3 main.py --project ablation_irpo --env-name fourrooms-v2 --algo-name irpo --num-runs 10 --aggregation-method argmax --noise-std 0.3 &  # argmax
nohup python3 main.py --project ablation_irpo --env-name fourrooms-v2 --algo-name irpo --num-runs 10 --aggregation-method argmax --noise-std 0.5 &  # argmax
nohup python3 main.py --project ablation_irpo --env-name fourrooms-v2 --algo-name irpo --num-runs 10 --aggregation-method argmax --noise-std 1.0 &  # argmax