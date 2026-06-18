import json
import subprocess
import os

with open('config/algos/irpo.json', 'r') as f:
    config = json.load(f)
config["extractor_total_samples"] = 2000
config["extractor_collect_batch_size"] = 2000
with open('config/algos/irpo.json', 'w') as f:
    json.dump(config, f, indent=4)

try:
    subprocess.run([
        "python3", "main.py", "--env", "pacman", "--algo", "irpo",
        "--int-reward-type", "allo", "--timesteps", "128",
        "--minibatch-size", "16", "--num-minibatch", "4", "--num-runs", "1",
        "--extractor-epochs", "1", "--wandb-mode", "disabled"
    ], check=True)
finally:
    # restore config
    del config["extractor_total_samples"]
    del config["extractor_collect_batch_size"]
    with open('config/algos/irpo.json', 'w') as f:
        json.dump(config, f, indent=4)
