# EUROPA
nohup python3 main.py --project ablation_irpo --env-name pointmaze-v1 --algo-name irpo --num-runs 10 --aggregation-method softmax --beta 0.0 --gpu-idx 1 &  # softmax -> argnax
nohup python3 main.py --project ablation_irpo --env-name pointmaze-v1 --algo-name irpo --num-runs 10 --aggregation-method softmax --beta 0.99 --gpu-idx 1 &  # softmax -> argnax
nohup python3 main.py --project ablation_irpo --env-name pointmaze-v1 --algo-name irpo --num-runs 10 --aggregation-method uniform --beta 0.0 --gpu-idx 1 &  # uniform -> argmax
nohup python3 main.py --project ablation_irpo --env-name pointmaze-v1 --algo-name irpo --num-runs 10 --aggregation-method uniform --beta 0.99 --gpu-idx 2 &  # uniform -> argmax
nohup python3 main.py --project ablation_irpo --env-name pointmaze-v1 --algo-name irpo --num-runs 10 --aggregation-method argmax --beta 0.0 --gpu-idx 2 &  # argmax
nohup python3 main.py --project ablation_irpo --env-name pointmaze-v1 --algo-name irpo --num-runs 10 --aggregation-method argmax --beta 0.99 --gpu-idx 2 &  # argmax
