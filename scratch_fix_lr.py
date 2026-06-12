import os
import glob

# 1. Update get_args.py
get_args_path = "utils/get_args.py"
with open(get_args_path, "r") as f:
    content = f.read()

if "args.actor_lr" not in content:
    content = content.replace("    return args\n", """    if getattr(args, "project", "") == "Atari" and getattr(args, "learning_rate", None) is not None:
        args.actor_lr = args.learning_rate / 5.0
    else:
        args.actor_lr = getattr(args, "learning_rate", None)

    return args\n""")
    with open(get_args_path, "w") as f:
        f.write(content)

# 2. Update algorithms/*.py
algos = ["ppo", "drnd", "hrl", "maml", "irpo"]
for algo in algos:
    algo_path = f"algorithms/{algo}.py"
    with open(algo_path, "r") as f:
        content = f.read()
    
    if "actor_lr=" not in content:
        content = content.replace("lr=self.args.learning_rate,", "lr=self.args.learning_rate,\n            actor_lr=getattr(self.args, 'actor_lr', None),")
        with open(algo_path, "w") as f:
            f.write(content)

# 3. Update policy/*.py
import re

for algo in algos:
    policy_path = f"policy/{algo}.py"
    with open(policy_path, "r") as f:
        content = f.read()
    
    if "actor_lr: float = None" not in content:
        # Add actor_lr to init signature
        content = re.sub(r'(lr:\s*float\s*=\s*[^,]+,)', r'\1\n        actor_lr: float = None,', content)
        
        # In Adam, replace self.actor.parameters(), "lr": lr with "lr": actor_lr
        # but first we need to initialize actor_lr if None
        init_block = "super().__init__(device=device)\n"
        replacement_block = init_block + "        if actor_lr is None:\n            actor_lr = lr\n"
        content = content.replace(init_block, replacement_block, 1)

        if "PPO" in content or "IRPO" in content or "MAML" in content or "DRND" in content or "HRL" in content:
            # We must be careful to only replace the actor's learning rate in Adam
            content = re.sub(r'({"params":\s*self\.actor\.parameters\(\),\s*"lr":\s*)lr(\s*})', r'\g<1>actor_lr\g<2>', content)
            
            # for IRPO, MAML, HRL there might be base_policy or high_level_policy actor.
            # MAML uses `self.actor`, IRPO uses `self.actor` (base policy).
            
        with open(policy_path, "w") as f:
            f.write(content)

print("Done")
