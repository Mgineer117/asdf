"""
generate_sbatch.py
==================
Generates SLURM sbatch files and a master submission script for the
amidar + pacman Atari experiments.

Cluster assignment
------------------
- irpo_random, irpo_allo, maml  →  IllinoisComputes-GPU  (H200/A100)
    time_per_job = 3-00:00:00, n_chains = 2  (total 6 days coverage)
- ppo, trpo, psne               →  eng-research-gpu
    time_per_job = 6-00:00:00, n_chains = 3  (total 18 days coverage)
- hrl, drnd                     →  csl
    time_per_job = 7-00:00:00, n_chains = 1  (no chain needed)

All chain segments except the first include --resume-run so the job
picks up from the saved WandB run-ID and latest checkpoint.
"""

import os

# ── Focused environments ──────────────────────────────────────────────────────
_envs_raw = os.environ.get("ENVS_SUBSET", "").strip()
if _envs_raw:
    ENVS = [e.strip() for e in _envs_raw.split(",") if e.strip()]
else:
    ENVS = ["pacman", "amidar", "bankheist", "alien"]

# ── Per-algo cluster config ───────────────────────────────────────────────────
# algo_key → (partition, account, time_per_job, n_chains, extra_flags)
ALGO_CONFIG = {
    "ppo":         ("eng-research-gpu",    "huytran1-ic", "2-00:00:00", 4, ""),
    "trpo":        ("eng-research-gpu",    "huytran1-ic", "2-00:00:00", 4, ""),
    "psne":        ("eng-research-gpu",    "huytran1-ic", "2-00:00:00", 4, ""),
    "irpo_random": ("IllinoisComputes-GPU","huytran1-ic", "3-00:00:00", 3,
                    "--int-reward-type random"),
    "irpo_allo":   ("IllinoisComputes-GPU","huytran1-ic", "3-00:00:00", 3,
                    "--int-reward-type allo"),
    "maml_random": ("IllinoisComputes-GPU","huytran1-ic", "3-00:00:00", 3,
                    "--int-reward-type random"),
    "maml_allo":   ("IllinoisComputes-GPU","huytran1-ic", "3-00:00:00", 3,
                    "--int-reward-type allo"),
    "maml":        ("IllinoisComputes-GPU","huytran1-ic", "3-00:00:00", 3,
                    "--int-reward-type allo"),
    "hrl":         ("csl",                "huytran1-ic", "7-00:00:00", 1,
                    "--int-reward-type allo"),
    "drnd":        ("csl",                "huytran1-ic", "7-00:00:00", 1, ""),
}

# The CLI algo name may differ from the algo_key (irpo_random → irpo)
ALGO_CLI_NAME = {
    "irpo_random": "irpo",
    "irpo_allo":   "irpo",
    "maml_random": "maml",
    "maml_allo":   "maml",
}

# ── Templates ─────────────────────────────────────────────────────────────────
SBATCH_HEADER = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time={time_limit}
#SBATCH --output=logs/{job_name}.o%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu

ulimit -n 4096  # raise file descriptor limit

# 1. Block Python from reading ~/.local/lib to prevent cross-contamination (CRITICAL)
export PYTHONNOUSERSITE=1

# 2. Hook conda into the non-interactive shell
eval "$(conda shell.bash hook)"

# 3. Add MuJoCo and NVIDIA to path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/minjae5/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# 4. Safely activate your environment
conda activate irpo

# Create logs directory if missing
mkdir -p logs

{run_commands}

# === Wait for all 5 seeds to finish ===
wait
"""

RUN_COMMANDS_TEMPLATE = """\
# === Run 5 seeds (0-4) in parallel on 1 GPU / Node === #
for SEED in {{0..4}}; do
    python3 main.py --project Atari-final --env {env} --algo {algo_cli} {extra_flags} \\
        --seed $SEED --gpu-idx 0{resume_flag} &
    sleep 3
done"""


def make_run_commands(env: str, algo_key: str, is_resume: bool) -> str:
    algo_cli = ALGO_CLI_NAME.get(algo_key, algo_key)
    extra_flags = ALGO_CONFIG[algo_key][4]
    resume_flag = " \\\n        --resume-run"
    return RUN_COMMANDS_TEMPLATE.format(
        env=env,
        algo_cli=algo_cli,
        extra_flags=extra_flags,
        resume_flag=resume_flag,
    )


def chain_suffix(chain_idx: int, n_chains: int) -> str:
    """Return empty string for chain 1, '_c2' for chain 2, etc."""
    if n_chains == 1:
        return ""          # no suffix – single job
    return "" if chain_idx == 1 else f"_c{chain_idx}"


def generate_sbatch(env: str, algo_key: str, chain_idx: int) -> str:
    partition, account, time_limit, n_chains, _ = ALGO_CONFIG[algo_key]
    suffix = chain_suffix(chain_idx, n_chains)
    job_name = f"{env}_{algo_key}{suffix}"
    is_resume = chain_idx > 1

    run_commands = make_run_commands(env, algo_key, is_resume)
    content = SBATCH_HEADER.format(
        job_name=job_name,
        account=account,
        partition=partition,
        time_limit=time_limit,
        run_commands=run_commands,
    )
    return job_name, content


# ── Subset / submit-script selection ──────────────────────────────────────────
# ALGO_SUBSET filters which algo_keys are emitted and which master submit script
# is written. Set via env var, e.g. ALGO_SUBSET=irpo_allo,irpo_random,maml.
# Defaults to all algos and the canonical submit_atari.sh.
_subset_raw = os.environ.get("ALGO_SUBSET", "").strip()
if _subset_raw:
    ALGO_SUBSET = [a.strip() for a in _subset_raw.split(",") if a.strip()]
    unknown = [a for a in ALGO_SUBSET if a not in ALGO_CONFIG]
    if unknown:
        raise SystemExit(f"Unknown algo key(s) in ALGO_SUBSET: {unknown}")
    SUBMIT_NAME = os.environ.get("SUBMIT_NAME", "submit_subset.sh")
else:
    ALGO_SUBSET = list(ALGO_CONFIG.keys())
    SUBMIT_NAME = os.environ.get("SUBMIT_NAME", "submit_atari.sh")

# ── Output directories ────────────────────────────────────────────────────────
os.makedirs("scripts", exist_ok=True)

generated = []  # list of (env, algo_key, chain_idx, filepath)

for env in ENVS:
    for algo_key in ALGO_SUBSET:
        partition, account, time_limit, n_chains, _ = ALGO_CONFIG[algo_key]
        for chain_idx in range(1, n_chains + 1):
            job_name, content = generate_sbatch(env, algo_key, chain_idx)
            suffix = chain_suffix(chain_idx, n_chains)
            filepath = f"scripts/{env}_{algo_key}{suffix}.sbatch"
            with open(filepath, "w") as f:
                f.write(content)
            os.chmod(filepath, 0o755)
            print(f"  Generated {filepath}  [{partition} | {time_limit} | chain {chain_idx}/{n_chains}]")
            generated.append((env, algo_key, chain_idx, n_chains, filepath))

# ── master submit script ──────────────────────────────────────────────────────
# Group by (env, algo_key) to build the dependency chain submission blocks.
from collections import defaultdict

blocks = defaultdict(list)  # (env, algo_key) → [filepath, ...]
for env, algo_key, chain_idx, n_chains, filepath in generated:
    blocks[(env, algo_key)].append((chain_idx, n_chains, filepath))

lines = [
    "#!/bin/bash",
    "# =========================================================",
    f"#  {SUBMIT_NAME} – submit all chains for {' + '.join(ENVS)}",
    f"#  algos: {', '.join(ALGO_SUBSET)}",
    "# =========================================================",
    "set -euo pipefail",
    "mkdir -p logs",
    "",
    "# Collected job IDs (every chain segment), used for the start-time report.",
    "SUBMITTED_JIDS=()",
    "",
    'echo "========================================================="',
    f'echo " Submitting {len(ENVS)} envs × {len(ALGO_SUBSET)} algos with dependency chains"',
    'echo "========================================================="',
    "",
]

for (env, algo_key), chain_list in sorted(blocks.items()):
    n_chains = chain_list[0][1]
    partition, *_ = ALGO_CONFIG[algo_key]
    lines.append(f"# ── {env} / {algo_key}  [{partition}, {n_chains} chain(s)] ──")

    prev_var = None
    for chain_idx, n_chains, filepath in sorted(chain_list):
        sbatch_file = os.path.basename(filepath)
        var = f"JID_{env.upper()}_{algo_key.upper()}_C{chain_idx}"
        var = var.replace("-", "_")

        if prev_var is None:
            lines.append(f'{var}=$(sbatch --parsable scripts/{sbatch_file})')
        else:
            lines.append(
                f'{var}=$(sbatch --parsable --dependency=afterany:${{{prev_var}}} scripts/{sbatch_file})'
            )
        lines.append(f'echo "[SUBMIT] scripts/{sbatch_file} → job ID: ${{{var}}}"')
        lines.append(f'SUBMITTED_JIDS+=("${{{var}}}")')
        prev_var = var

    lines.append("")

lines += [
    'echo "========================================================="',
    'echo " All jobs submitted. Waiting 10s for the scheduler to register them..."',
    'echo "========================================================="',
    "sleep 10",
    "",
    "# Comma-separated job-id list for squeue.",
    'JOBS_CSV=$(IFS=,; echo "${SUBMITTED_JIDS[*]}")',
    "",
    'echo "========================================================="',
    'echo " Estimated start times (squeue --start)"',
    'echo "========================================================="',
    "# --start prints the scheduler's estimated START_TIME and the pending",
    "# reason (Resources/Priority/Dependency). Restricted to the jobs we just",
    "# submitted; falls back to all of the user's jobs if the id filter errors.",
    'squeue --start --jobs="$JOBS_CSV" \\',
    '    -o "%.12i %.34j %.16P %.20S %.10T %.18r" \\',
    '    || squeue --start -u "$(whoami)"',
    "",
    'echo "========================================================="',
    'echo " Live queue status (squeue)"',
    'echo "========================================================="',
    'squeue --jobs="$JOBS_CSV" \\',
    '    -o "%.12i %.34j %.16P %.10T %.10M %.10l %.18r" \\',
    '    || squeue -u "$(whoami)"',
    "",
    'echo "Re-check anytime with:  squeue --start --jobs=$JOBS_CSV"',
]

submit_path = f"scripts/{SUBMIT_NAME}"
with open(submit_path, "w") as f:
    f.write("\n".join(lines) + "\n")
os.chmod(submit_path, 0o755)
print(f"\n  Generated {submit_path}")
print(f"\nDone. Total sbatch files: {len(generated)}  +  1 submission script.")
