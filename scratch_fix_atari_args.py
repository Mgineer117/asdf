import json
import glob

# 1. Remove is_atari from env configs
atari_envs = ["pacman", "amidar", "bankheist", "alien"]
for env in atari_envs:
    path = f"config/envs/{env}.json"
    with open(path, "r") as f:
        data = json.load(f)
    if "is_atari" in data:
        del data["is_atari"]
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

# 2. Update get_args.py
get_args_path = "utils/get_args.py"
with open(get_args_path, "r") as f:
    content = f.read()

# Remove the --is-atari argument
content = content.replace(
    'parser.add_argument(\n        "--is-atari",\n        action="store_true",\n        help="Flag to indicate if environment is Atari",\n    )\n    ',
    ''
)

# Update logic in override_args
old_logic_override = """    if getattr(args, "is_atari", False) and getattr(args, "atari_learning_rate", None) is not None:
        args.learning_rate = args.atari_learning_rate"""

new_logic_override = """    atari_envs = ["pacman", "amidar", "bankheist", "alien"]
    if config_env_name in atari_envs and getattr(args, "atari_learning_rate", None) is not None:
        args.learning_rate = args.atari_learning_rate"""

content = content.replace(old_logic_override, new_logic_override)

# Update logic in get_args
old_logic_get_args = """    if getattr(args, "is_atari", False) and getattr(args, "atari_learning_rate", None) is not None:
        args.learning_rate = args.atari_learning_rate"""

new_logic_get_args = """    atari_envs = ["pacman", "amidar", "bankheist", "alien"]
    env_name, _, _ = args.env_name.partition("-")
    if env_name in atari_envs and getattr(args, "atari_learning_rate", None) is not None:
        args.learning_rate = args.atari_learning_rate"""

content = content.replace(old_logic_get_args, new_logic_get_args)

with open(get_args_path, "w") as f:
    f.write(content)

print("Done")
