import glob
import json
import re

# 1. Revert algorithms/*.py
for algo_file in glob.glob("algorithms/*.py"):
    with open(algo_file, "r") as f:
        content = f.read()
    content = content.replace("lr=self.args.learning_rate,\n            actor_lr=getattr(self.args, 'actor_lr', None),", "lr=self.args.learning_rate,")
    with open(algo_file, "w") as f:
        f.write(content)

# 2. Revert policy/*.py
for policy_file in glob.glob("policy/*.py"):
    with open(policy_file, "r") as f:
        content = f.read()
    
    # Remove actor_lr from signature
    content = re.sub(r'\n\s*actor_lr:\s*float\s*=\s*None,', '', content)
    
    # Remove the init block logic
    content = re.sub(r'super\(\)\.__init__\(device=device\)\n\s*if actor_lr is None:\n\s*actor_lr = lr\n', 'super().__init__(device=device)\n', content)
    
    # Revert Adam lr usage
    content = content.replace('"lr": actor_lr', '"lr": lr')
    
    with open(policy_file, "w") as f:
        f.write(content)

# 3. Revert config/envs/*.json
atari_envs = ["pacman", "amidar", "bankheist", "alien"]
for env in atari_envs:
    path = f"config/envs/{env}.json"
    with open(path, "r") as f:
        data = json.load(f)
    if "atari_actor_lr" in data:
        del data["atari_actor_lr"]
    data["is_atari"] = True
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

# 4. Modify config/algos/*.json to add atari_learning_rate (1/5 of learning_rate)
for algo_file in glob.glob("config/algos/*.json"):
    with open(algo_file, "r") as f:
        data = json.load(f)
    if "learning_rate" in data:
        data["atari_learning_rate"] = data["learning_rate"] / 5.0
    with open(algo_file, "w") as f:
        json.dump(data, f, indent=4)

# 5. Fix get_args.py
get_args_path = "utils/get_args.py"
with open(get_args_path, "r") as f:
    content = f.read()

# Remove old additions from get_args
content = content.replace(
    'parser.add_argument(\n        "--atari-actor-lr",\n        action="store_true",\n        help="Set actor lr to 1/5 of learning rate for Atari",\n    )\n    parser.add_argument(\n        "--actor-lr",\n        type=float,\n        default=None,\n        help="Actor learning rate (overrides learning_rate if set)",\n    )\n    parser.add_argument(\n        "--actor-activation",',
    'parser.add_argument(\n        "--is-atari",\n        action="store_true",\n        help="Flag to indicate if environment is Atari",\n    )\n    parser.add_argument(\n        "--atari-learning-rate",\n        type=float,\n        default=None,\n        help="Learning rate for Atari environments",\n    )\n    parser.add_argument(\n        "--actor-activation",'
)

# Replace logic in get_args
old_logic = """    if getattr(args, "atari_actor_lr", False) and getattr(args, "learning_rate", None) is not None:
        args.actor_lr = args.learning_rate / 5.0
    elif getattr(args, "actor_lr", None) is None:
        args.actor_lr = getattr(args, "learning_rate", None)"""

new_logic = """    if getattr(args, "is_atari", False) and getattr(args, "atari_learning_rate", None) is not None:
        args.learning_rate = args.atari_learning_rate"""

content = content.replace(old_logic, new_logic)

with open(get_args_path, "w") as f:
    f.write(content)

print("Done")
