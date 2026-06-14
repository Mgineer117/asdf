import re

files_to_fix = [
    "policy/htrpo.py",
    "policy/maml.py",
    "policy/psne.py",
    "policy/trpo.py",
    "policy/irpo.py"
]

target_pattern = r'([ \t]*)sAs = 0\.5 \* torch\.dot\(step_dir, Hv\(step_dir\)\)\n[ \t]*sAs = torch\.clamp\(sAs, min=1e-8\)\n[ \t]*lm = torch\.sqrt\(sAs / self\.target_kl\)\n[ \t]*full_step = step_dir / \(lm \+ 1e-8\)'

replacement = r'''\1sAs = 0.5 * torch.dot(step_dir, Hv(step_dir))
\1if sAs < 1e-8:
\1    full_step = torch.zeros_like(step_dir)
\1else:
\1    lm = torch.sqrt(sAs / self.target_kl)
\1    full_step = step_dir / lm'''

for filepath in files_to_fix:
    with open(filepath, "r") as f:
        content = f.read()
    
    new_content = re.sub(target_pattern, replacement, content)
    
    with open(filepath, "w") as f:
        f.write(new_content)

print("Done fixing sAs explosion bug")
