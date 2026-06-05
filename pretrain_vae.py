"""
Pre-train the VAE encoder on random-policy samples for Atari environments.

IRPO on Atari uses this encoder as a FROZEN, shared visual representation, so it
must be trained first and saved to:

    model/<env>/encoder/<seed>.pth

IMPORTANT — seed convention:
    main.py maps the CLI --seed to an INDEX into a fixed seed_pool when
    num_runs == 1 (e.g. --seed 0 -> 1825). This script applies the SAME mapping
    so the encoder filename matches the seed IRPO actually runs with.

Usage:
    python pretrain_vae.py --env amidar --seed 0 --epochs 50 --collect-samples 100000
"""

import argparse
import datetime
import os
from argparse import Namespace

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm

from policy.layers.building_blocks import ConvVAEEncoder
from utils.functions import seed_all
from utils.get_args import select_device
from utils.get_envs import get_env

# Must match main.py
SEED_POOL = [1825, 410, 4507, 4013, 3658, 2287, 1680, 8936, 1425, 9675]


def map_seed(seed_arg: int) -> int:
    """Replicate main.py's --seed -> seed_pool[index] mapping (num_runs == 1)."""
    if 0 <= seed_arg < len(SEED_POOL):
        return SEED_POOL[seed_arg]
    return seed_arg


def collect_random_samples(env, num_samples: int):
    """Roll out a uniform-random policy and collect grayscale frames (N, H, W)."""
    states = []
    obs, _ = env.reset()
    for _ in tqdm(range(num_samples), desc="Collecting random samples"):
        action = env.action_space.sample()
        next_obs, _, terminated, truncated, _ = env.step(action)

        # ArcadeWrapper returns grayscale (H, W); guard for stray channel dim.
        frame = obs
        if frame.ndim == 3:
            frame = frame.mean(axis=2) if frame.shape[2] == 3 else frame[:, :, 0]
        states.append(frame.astype(np.float32))

        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

    return np.stack(states)  # (N, H, W)


def train_vae_encoder(vae, env=None, states=None, num_epochs=50, batch_size=256,
                      learning_rate=1e-3, num_samples=100000, device=None,
                      verbose=True):
    """Train an existing ConvVAEEncoder in memory on random-policy rollouts.

    Pass pre-collected `states` (N, H, W) to skip sampling, otherwise random
    samples are collected from `env`. Used both by the standalone pretraining
    script and by IRPO with int_reward_type=random, which pretrains a FRESH
    encoder at the start of every run instead of loading one from disk.
    Returns `vae`.
    """
    device = device or next(vae.parameters()).device

    if states is None:
        if env is None:
            raise ValueError("train_vae_encoder requires either `env` or `states`.")
        if verbose:
            print(f"Collecting {num_samples} random samples for VAE pretraining...")
        states = collect_random_samples(env, num_samples)

    optimizer = Adam(vae.parameters(), lr=learning_rate)
    num_batches = max(1, len(states) // batch_size)

    if verbose:
        print(f"Pre-training VAE for {num_epochs} epochs over {len(states)} frames...")
    for epoch in range(num_epochs):
        perm = np.random.permutation(len(states))
        epoch_loss = 0.0
        for b in range(num_batches):
            sl = perm[b * batch_size : (b + 1) * batch_size]
            batch = torch.from_numpy(states[sl]).to(device)  # (B, H, W)

            loss = vae.vae_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        if verbose:
            print(f"Epoch {epoch + 1}/{num_epochs} | VAE loss: {epoch_loss / num_batches:.4f}")

    return vae


def pretrain_vae(env_name, seed_arg, num_epochs, batch_size, learning_rate=1e-3,
                 num_samples=100000, gpu_idx=0):
    seed = map_seed(seed_arg)
    seed_all(seed)
    device = select_device(gpu_idx)

    # Build the same args namespace get_env expects; it fills in state_dim etc.
    args = Namespace(env_name=env_name, seed=seed, gpu_idx=gpu_idx, device=device)
    env = get_env(args)
    env_base = env_name.split("-")[0]

    print(f"Collecting {num_samples} random samples from {env_name} (seed {seed})...")
    states = collect_random_samples(env, num_samples)
    try:
        env.close()
    except Exception:
        pass

    H, W = states.shape[1:]
    print(f"State shape: ({H}, {W}); collected {states.shape[0]} frames")

    # latent_dim must match PPO_Actor's CNN feature dim (256) so IRPO's MLP head
    # stays compatible with this encoder.
    vae = ConvVAEEncoder(input_shape=(1, H, W), latent_dim=256, device=device).to(device)

    train_vae_encoder(
        vae,
        states=states,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
    )

    model_dir = os.path.join("model", env_base, "encoder")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{seed}.pth")
    torch.save(vae.state_dict(), model_path)
    print(f"\n[OK] Saved encoder to {model_path}")
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train VAE encoder on random samples")
    parser.add_argument("--env", type=str, default="amidar", help="Environment name")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed INDEX into seed_pool (matches main.py --seed)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--collect-samples", type=int, default=100000)
    parser.add_argument("--gpu-idx", type=int, default=0)
    args = parser.parse_args()

    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"Pre-training VAE | env={args.env} | seed_index={args.seed} "
          f"-> seed={map_seed(args.seed)}")

    pretrain_vae(
        env_name=args.env,
        seed_arg=args.seed,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_samples=args.collect_samples,
        gpu_idx=args.gpu_idx,
    )
