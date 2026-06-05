# VAE Pre-training + IRPO Fine-tuning Workflow

This document describes the two-stage workflow for training a VAE encoder on Atari environments.

## Overview

**Stage 1 (Optional)**: Pre-train VAE on random policy samples
- Fast, deterministic training on diverse random trajectories
- Creates `model/<env>/encoder/<seed>.pth`

**Stage 2**: Fine-tune VAE + Train IRPO
- IRPO automatically loads pre-trained encoder (if exists)
- Continues fine-tuning VAE via reconstruction loss
- Trains policy via IRPO with random intrinsic reward

## Why Two Stages?

| Aspect | Stage 1 (Random) | Stage 2 (IRPO) |
|--------|------------------|----------------|
| **Data source** | Random policy (diverse, unbiased) | On-policy (focused on learned behavior) |
| **Convergence** | Fast (no RL optimization) | Slow (RL convergence) |
| **Purpose** | Warm-start CNN on visual features | Specialize CNN for learned policy |
| **Time** | Hours | Days |

**Combined**: CNN gets general visual knowledge (Stage 1) + policy-specific knowledge (Stage 2)

## Usage

### Option A: Full Two-Stage Workflow (Recommended)

```bash
sbatch atari_vae_pretrain_finetune.sbatch
```

**Timeline:**
- Stage 1: ~4-5 hours (4 envs × 5 seeds × 50 epochs on 100K random samples)
- Stage 2: ~7 days (4 envs × 5 seeds × 500M timesteps IRPO)
- **Total: ~7 days**

### Option B: Pre-train Only

```bash
sbatch pretrain_vae_atari.sbatch
```

Saves encoders to `model/<env>/encoder/<seed>.pth`. Later, run IRPO normally:

```bash
sbatch pacman_irpo_random.sbatch
sbatch amidar_irpo_random.sbatch
# ... etc
```

### Option C: IRPO Only (No Pre-training)

```bash
sbatch pacman_irpo_random.sbatch
sbatch amidar_irpo_random.sbatch
# ... etc
```

IRPO trains VAE from scratch via reconstruction loss (Stage 2 only).

### Option D: Manual Single Env

```bash
# Pre-train
python pretrain_vae.py --env amidar --seed 1825 --epochs 100 --collect-samples 200000

# Fine-tune (loads pre-trained encoder automatically)
python main.py --env amidar --algo irpo --int-reward-type random --seed 1825
```

## How IRPO Loads Pre-trained Encoders

In `algorithms/irpo.py`, when creating the VAE encoder:

```python
# Try to load pre-trained VAE encoder (from pretrain_vae.py)
pretrain_path = os.path.join("model", env_base, "encoder", f"{seed}.pth")
if os.path.exists(pretrain_path):
    vae_encoder.load_state_dict(torch.load(pretrain_path, ...))
    print(f"[IRPO] Loaded pre-trained VAE encoder from {pretrain_path}")
```

**Behavior:**
- ✓ Encoder exists → Load it, continue fine-tuning
- ✗ Encoder missing → Create fresh, train from scratch

## Encoder Storage

Pre-trained encoders are saved to:
```
model/
├── pacman/
│   └── encoder/
│       ├── 0.pth      (seed 0)
│       ├── 1.pth      (seed 1)
│       └── ...
├── amidar/
│   └── encoder/
│       └── ...
├── bankheist/
│   └── encoder/
│       └── ...
└── alien/
    └── encoder/
        └── ...
```

Same location where IRPO saves checkpoints during fine-tuning.

## Hyperparameters

### Pre-training (Stage 1)

- `--epochs 50`: Number of training passes over collected samples
- `--collect-samples 100000`: Number of random policy trajectories
- `--batch-size 256`: Training batch size
- `--learning-rate 3e-4`: VAE optimizer LR (Adam)

**Note:** Only tunes CNN reconstruction loss, not RL objective.

### Fine-tuning (Stage 2)

IRPO hyperparameters in config (e.g., `amidar.json`):
- `batch_size: 4096`: On-policy batch size
- `minibatch_size: 256`: Gradient computation batch (controls memory)
- `num_options: 6`: Number of exploratory options
- `num_exp_updates: 5`: Exploration steps per option
- `int_reward_type: random`: Use random feature learning

VAE-specific in `policy/irpo.py`:
- `vae_lr: 3e-4`: VAE optimizer LR during IRPO
- `_vae_step_counter % 5`: VAE updates every 5 IRPO steps (slow temporal scale)
- Gradient annealing: LR decays linearly from 100% to 10%

## Expected Performance

**Pre-training (Stage 1):**
- VAE reconstruction loss: ~0.1-0.3 (MSE on [0,1] pixels)
- Time: 30-60 min per seed per env

**Fine-tuning (Stage 2):**
- VAE loss: increases slightly as CNN specializes for policy features
- IRPO reward: steadily improves with exploration + option aggregation
- Time: 2-3 weeks per seed per env (depending on convergence)

## Debugging

**Q: Pre-training is slow**
- Reduce `--collect-samples` (e.g., 50K instead of 100K)
- Reduce `--epochs` (e.g., 25 instead of 50)

**Q: IRPO not loading encoder**
- Check `model/<env>/encoder/<seed>.pth` exists
- Check file size > 1 MB (not corrupted)
- IRPO prints `[IRPO] Loaded pre-trained VAE encoder` if successful

**Q: Encoder not improving during IRPO**
- Check VAE loss is logging to wandb (should spike every 5 steps)
- Check minibatch_size is small enough for GPU memory
- Increase `vae_lr` if VAE loss plateaus

## References

- `pretrain_vae.py`: VAE pre-training script
- `policy/irpo.py`: IRPO learner (VAE fine-tuning logic)
- `algorithms/irpo.py`: Algorithm setup (encoder loading)
- `policy/layers/building_blocks.py`: ConvVAEEncoder implementation
