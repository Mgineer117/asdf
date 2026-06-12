import re

files_to_fix = [
    "policy/htrpo.py",
    "policy/maml.py",
    "policy/psne.py",
    "policy/trpo.py"
]

target_str = "sAs = 0.5 * torch.dot(step_dir, Hv(step_dir))"
replacement_str = "sAs = 0.5 * torch.dot(step_dir, Hv(step_dir))\n        sAs = torch.clamp(sAs, min=1e-8)"

# MAML and HTRPO might have different indentation
for filepath in files_to_fix:
    with open(filepath, "r") as f:
        content = f.read()
    
    # We use regex to preserve indentation
    content = re.sub(
        r'([ \t]*)sAs = 0\.5 \* torch\.dot\(step_dir, Hv\(step_dir\)\)',
        r'\1sAs = 0.5 * torch.dot(step_dir, Hv(step_dir))\n\1sAs = torch.clamp(sAs, min=1e-8)',
        content
    )
    
    with open(filepath, "w") as f:
        f.write(content)

print("Done fixing sAs")
