import re

files_to_fix = [
    "policy/htrpo.py",
    "policy/maml.py",
    "policy/psne.py",
    "policy/trpo.py",
    "policy/irpo.py"
]

target_pattern = r'([ \t]*)else:\n[ \t]*lm = torch\.sqrt\(sAs / self\.target_kl\)\n[ \t]*full_step = step_dir / lm'

replacement = r'''\1else:
\1    lm = torch.sqrt(sAs / self.target_kl)
\1    full_step = step_dir / lm
\1
\1if not torch.isfinite(full_step).all():
\1    print("WARNING: full_step contains NaN/Inf! Rejecting update.")
\1    full_step = torch.zeros_like(step_dir)'''

for filepath in files_to_fix:
    with open(filepath, "r") as f:
        content = f.read()
    
    new_content = re.sub(target_pattern, replacement, content)
    
    with open(filepath, "w") as f:
        f.write(new_content)

print("Done adding NaN safety layer")
