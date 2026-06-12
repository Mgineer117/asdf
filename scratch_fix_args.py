import json

get_args_path = "utils/get_args.py"
with open(get_args_path, "r") as f:
    content = f.read()

# Revert previous change
old_block = """    if getattr(args, "project", "") == "Atari" and getattr(args, "learning_rate", None) is not None:
        args.actor_lr = args.learning_rate / 5.0
    else:
        args.actor_lr = getattr(args, "learning_rate", None)

    return args"""
new_block = """    if getattr(args, "atari_actor_lr", False) and getattr(args, "learning_rate", None) is not None:
        args.actor_lr = args.learning_rate / 5.0
    elif getattr(args, "actor_lr", None) is None:
        args.actor_lr = getattr(args, "learning_rate", None)

    return args"""

content = content.replace(old_block, new_block)

# Add --atari-actor-lr and --actor-lr
if "--atari-actor-lr" not in content:
    content = content.replace(
        'parser.add_argument(\n        "--actor-activation",',
        'parser.add_argument(\n        "--atari-actor-lr",\n        action="store_true",\n        help="Set actor lr to 1/5 of learning rate for Atari",\n    )\n    parser.add_argument(\n        "--actor-lr",\n        type=float,\n        default=None,\n        help="Actor learning rate (overrides learning_rate if set)",\n    )\n    parser.add_argument(\n        "--actor-activation",'
    )

with open(get_args_path, "w") as f:
    f.write(content)

# Update config/envs/*.json
atari_envs = ["pacman", "amidar", "bankheist", "alien"]
for env in atari_envs:
    path = f"config/envs/{env}.json"
    with open(path, "r") as f:
        data = json.load(f)
    data["atari_actor_lr"] = True
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

print("Done")
