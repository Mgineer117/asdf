#!/bin/bash
# ---------------------------------------------------------------------------
# Smoke test for the SLURM job-chain resume path (IRPO / MAML).
#
# Run from the repo root with the conda env active:
#     conda activate irpo
#     bash scripts/test_resume.sh
#
# It does NOT touch your real logs or W&B:
#   - logs to an isolated dir (log/test_resume_log/, wiped at start)
#   - forces --wandb-mode disabled
#
# What it checks end to end, in ~a couple minutes on CPU:
#   1. Phase 1 trains a tiny run and writes checkpoint/latest_full.pt.
#   2. latest_full.pt is a FULL training checkpoint: policy_state carries the
#      critic optimizer moments, the perf-gain EMA, and the reward normalizer
#      (rms) -- not just the network weights.
#   3. Phase 2 (same env/algo/seed, larger --timesteps, with --resume-run)
#      auto-detects the checkpoint, prints "[Resume] Loaded checkpoint", and
#      continues from phase 1's timestep instead of restarting.
#
# All sizes are tiny and overridable from the environment, e.g.:
#     T1=4000 T2=8000 ENV=maze-v1 ALGO=maml bash scripts/test_resume.sh
# ---------------------------------------------------------------------------
set -euo pipefail

# --- knobs (override via env) ---
ENV=${ENV:-fourrooms-v1}        # cheap discrete gridworld by default
ALGO=${ALGO:-irpo}              # irpo | maml
INT_REWARD=${INT_REWARD:-random}  # random avoids ALLO extractor pretraining
SEED=${SEED:-0}                 # index into main.py's seed_pool (num-runs 1)
T1=${T1:-3000}                  # phase-1 timesteps
T2=${T2:-6000}                  # phase-2 timesteps (> T1 to force continued training)
CKPT_EVERY=${CKPT_EVERY:-2}     # rolling checkpoint cadence, in learn() updates

LOGROOT="log/test_resume_log"
COMMON_ARGS=(
    --project irpo_resume_test
    --logdir "$LOGROOT"
    --env "$ENV"
    --algo "$ALGO"
    --int-reward-type "$INT_REWARD"
    --seed "$SEED"
    --num-runs 1
    --eval-num 1
    --log-interval 4
    --minibatch-size 64
    --num-minibatch 1
    --checkpoint-interval "$CKPT_EVERY"
    --wandb-mode disabled
)

red()   { printf "\033[31m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
bold()  { printf "\033[1m%s\033[0m\n" "$*"; }
fail()  { red "FAIL: $*"; exit 1; }

# Find the newest latest_full.pt under LOGROOT and print its saved step (or
# MISSING). The step lives inside the checkpoint dict, not a sidecar json.
read_step() {
    python3 - "$LOGROOT" <<'PY'
import glob, sys, torch
hits = glob.glob(sys.argv[1] + "/**/checkpoint/latest_full.pt", recursive=True)
if not hits:
    print("MISSING"); sys.exit(0)
ck = torch.load(max(hits, key=__import__("os").path.getmtime),
                map_location="cpu", weights_only=False)
print(int(ck.get("step", -1)))
PY
}

bold "== Cleaning $LOGROOT =="
rm -rf "$LOGROOT"
mkdir -p "$LOGROOT"

# ------------------------------------------------------------------ Phase 1
bold "== Phase 1: fresh training to ~$T1 timesteps ($ALGO/$ENV/$INT_REWARD) =="
python3 main.py "${COMMON_ARGS[@]}" --timesteps "$T1" 2>&1 | tee "$LOGROOT/phase1.out"

CKPT=$(find "$LOGROOT" -name latest_full.pt | head -1 || true)
[ -n "$CKPT" ] || fail "phase 1 did not produce latest_full.pt"
green "phase 1 wrote checkpoint: $CKPT"

STEP1=$(read_step)
[ "$STEP1" != "MISSING" ] || fail "phase 1 checkpoint has no step"
green "phase 1 checkpoint step = $STEP1"

# Verify the checkpoint is FULL training state, not weights-only.
bold "== Verifying full-state checkpoint contents =="
python3 - "$CKPT" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if not (isinstance(ck, dict) and "policy_state" in ck):
    raise SystemExit(f"FAIL: checkpoint missing 'policy_state'; keys={list(ck)}")
ps = ck["policy_state"]
if not (isinstance(ps, dict) and "modules" in ps):
    raise SystemExit(f"FAIL: policy_state is not a full-state dict; keys={list(ps) if isinstance(ps, dict) else type(ps)}")
has_optim = any(k in ps for k in ("ext_critic_optim", "critic_optim"))
checks = {
    "modules (weights)": "modules" in ps,
    "critic optimizer moments": has_optim,
    "perf_gains EMA": "perf_gains" in ps,
    "reward_rms normalizer": "reward_rms" in ps,
}
for name, ok in checks.items():
    print(f"  {'OK ' if ok else 'MISSING'} {name}")
missing = [k for k, ok in checks.items() if not ok]
if missing:
    raise SystemExit(f"FAIL: checkpoint missing {missing}")
print("OK: full training state present")
PY
green "full-state checkpoint verified (weights + optimizer + perf_gains + rms)"

# ------------------------------------------------------------------ Phase 2
bold "== Phase 2: --resume-run, --timesteps $T2 (should resume + continue) =="
python3 main.py "${COMMON_ARGS[@]}" --timesteps "$T2" --resume-run 2>&1 | tee "$LOGROOT/phase2.out"

grep -q "\[Resume\] Loaded checkpoint" "$LOGROOT/phase2.out" \
    || fail "phase 2 did NOT resume (no '[Resume] Loaded checkpoint' message)"
green "phase 2 resumed from the phase-1 checkpoint"

STEP2=$(read_step)
[ "$STEP2" != "MISSING" ] || fail "phase 2 lost the checkpoint"
python3 -c "import sys; sys.exit(0 if $STEP2 > $STEP1 else 1)" \
    || fail "phase 2 step ($STEP2) did not advance past phase 1 step ($STEP1)"
green "phase 2 advanced the timestep: $STEP1 -> $STEP2"

echo
green "===================================================="
green " PASS: resume works -- full-state checkpoint, detected"
green "       automatically, training continued ($STEP1 -> $STEP2)."
green "===================================================="
