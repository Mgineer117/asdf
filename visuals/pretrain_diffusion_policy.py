"""Pretrain a state-diffusing policy on FourRooms and save its weights.

Purpose
-------
Produces a well-initialized *exploratory* policy that can be loaded into
`visuals/ppo_exploration_failure_modes.py` via its `--init-from` flag to test
the third hypothesis: that the cold-start reward problem in sparse-reward
PPO can be sidestepped by initializing from a policy that already covers
state space well, rather than from the uniform random policy that has near-
zero probability of seeing the first reward in time-bounded episodes.

Method (lighter than the full ALLO pipeline, same intent)
---------------------------------------------------------
We pretrain with PPO using a *count-based intrinsic reward*

    r_int(s) = 1 / sqrt( N(s) + 1 )

where N(s) is the cumulative visit count of grid cell s. This is a discrete
analogue of state-density penalization (cf. ALLO/Laplacian-eigenvector or
RND-style bonuses) and it matches the *intent* of ALLO on tabular gridworlds:
push the policy to spread visitation as uniformly as possible.

We use exactly the same `ActorCritic` architecture, `compute_gae`,
`ppo_update`, and `flat_obs` helpers as `ppo_exploration_failure_modes.py`,
imported directly so the saved checkpoint is plug-compatible.

Usage
-----
    python visuals/pretrain_diffusion_policy.py
    python visuals/pretrain_diffusion_policy.py --updates 80 --steps 1024 --seed 0

Outputs the actor state_dict to `visuals/checkpoints/diffusion_actor.pt` (by
default). Load it in the failure-mode experiment with

    python visuals/ppo_exploration_failure_modes.py \\
        --init-from visuals/checkpoints/diffusion_actor.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import torch
import torch.nn.functional as F

from gridworld import FourRooms

# Shared building blocks from the failure-mode experiment.
from visuals.ppo_exploration_failure_modes import (
    ActorCritic,
    Config,
    compute_gae,
    flat_obs,
    ppo_update,
)

import matplotlib.pyplot as plt


def intrinsic_rollout(env, model, T, w, h, count_grid, reachable_mask, beta: float):
    """Like the experiment's `rollout`, but rewards are *count-based* and
    extrinsic reward is ignored.

    `count_grid` is a (W, H) int array of cumulative visits across pretraining;
    we update it as we go so the bonus decays at the right rate.
    """
    obs_buf = np.zeros((T, 4), dtype=np.float32)
    act_buf = np.zeros(T, dtype=np.int64)
    logp_buf = np.zeros(T, dtype=np.float32)
    rew_buf = np.zeros(T, dtype=np.float32)
    val_buf = np.zeros(T, dtype=np.float32)
    done_buf = np.zeros(T, dtype=np.float32)
    local_visit = np.zeros_like(count_grid)

    obs, _ = env.reset()
    fobs = flat_obs(obs, w, h)

    for t in range(T):
        x = torch.from_numpy(fobs).unsqueeze(0)
        with torch.no_grad():
            dist = model.pi(x)
            v = model.v(x).item()
        a_t = dist.sample()
        a = int(a_t.item())
        logp = dist.log_prob(a_t).item()

        obs_buf[t] = fobs
        act_buf[t] = a
        logp_buf[t] = logp
        val_buf[t] = v

        ax, ay = obs["achieved_goal"]
        cx, cy = int(ax), int(ay)
        local_visit[cx, cy] += 1
        # Update the count BEFORE computing the bonus so revisits decay.
        count_grid[cx, cy] += 1
        intrinsic_r = beta / np.sqrt(count_grid[cx, cy])

        obs, _r_ext, term, trunc, _ = env.step(a)  # extrinsic reward ignored
        rew_buf[t] = float(intrinsic_r)
        done = bool(term or trunc)
        done_buf[t] = float(done)
        if done:
            obs, _ = env.reset()
        fobs = flat_obs(obs, w, h)

    with torch.no_grad():
        last_v = model.v(torch.from_numpy(fobs).unsqueeze(0)).item()

    visited = (local_visit > 0) & reachable_mask
    return dict(
        obs=obs_buf,
        act=act_buf,
        logp=logp_buf,
        rew=rew_buf,
        val=val_buf,
        done=done_buf,
        last_v=last_v,
        unique_cells=int(visited.sum()),
        ep_returns=[float(rew_buf.sum())],  # for diagnostic logging only
    )


def plot_pretraining_metrics(metrics_history: dict, out_dir: Path):
    """Plot training metrics: unique_cells, cumulative coverage, and intrinsic reward sum."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    updates = metrics_history["updates"]

    # Plot 1: Unique cells per update
    ax = axes[0]
    ax.plot(
        updates, metrics_history["unique_cells"], marker="o", linewidth=2, markersize=5
    )
    ax.set_xlabel("Update")
    ax.set_ylabel("Unique cells per rollout")
    ax.set_title("Exploration (unique cells)")
    ax.grid(alpha=0.3)

    # Plot 2: Cumulative coverage
    ax = axes[1]
    ax.plot(
        updates,
        metrics_history["cumulative_cov"],
        marker="o",
        linewidth=2,
        markersize=5,
        color="orange",
    )
    ax.set_xlabel("Update")
    ax.set_ylabel("Cumulative coverage")
    ax.set_title("Coverage (fraction of reachable cells)")
    ax.grid(alpha=0.3)

    # Plot 3: Intrinsic reward sum per update
    ax = axes[2]
    ax.plot(
        updates,
        metrics_history["int_reward_sum"],
        marker="o",
        linewidth=2,
        markersize=5,
        color="green",
    )
    ax.set_xlabel("Update")
    ax.set_ylabel("Intrinsic reward sum")
    ax.set_title("Intrinsic reward (per rollout)")
    ax.grid(alpha=0.3)

    fig.suptitle("Pretraining Metrics Across Updates", fontsize=13, fontweight="bold")
    fig.tight_layout()

    # Save as SVG
    fig_path = out_dir / "pretraining_metrics.svg"
    fig.savefig(fig_path, dpi=130, format="svg")
    plt.close(fig)

    print(f"\nMetrics plot saved to: {fig_path.resolve()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--updates",
        type=int,
        default=500,
        help="number of PPO updates with intrinsic reward",
    )
    ap.add_argument("--steps", type=int, default=1024, help="env steps per rollout")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beta", type=float, default=1.0, help="intrinsic reward scale")
    ap.add_argument("--grid-type", type=str, default="v1")
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--out", type=str, default="visuals/checkpoints/diffusion_actor.pt")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = FourRooms(grid_type=args.grid_type, max_steps=args.max_steps)
    w, h = env.width, env.height
    model = ActorCritic()
    opt_pi = torch.optim.Adam(model.actor.parameters(), lr=3e-4)
    opt_v = torch.optim.Adam(model.critic.parameters(), lr=1e-3)

    # PPO hyperparams matching the experiment so the resulting policy is in
    # the same regime; we just swap the reward channel.
    cfg = Config(
        name="pretrain",
        n_updates=args.updates,
        steps_per_rollout=args.steps,
        gae_lambda=0.95,
        gamma=0.99,
    )

    base_grid = env.get_grid()[..., 0]
    reachable_mask = base_grid != 2
    n_reachable = int(reachable_mask.sum())
    count_grid = np.zeros((w, h), dtype=np.int64)

    print(
        f"Pretraining diffusion policy on FourRooms-{args.grid_type} "
        f"({args.updates} updates × {args.steps} steps, seed={args.seed})"
    )

    # Track metrics across all updates for plotting
    metrics_history = {
        "updates": [],
        "unique_cells": [],
        "cumulative_cov": [],
        "int_reward_sum": [],
    }

    for k in range(args.updates):
        batch = intrinsic_rollout(
            env, model, args.steps, w, h, count_grid, reachable_mask, args.beta
        )
        ppo_update(model, opt_pi, opt_v, batch, cfg, freeze_actor=False)
        cov = (count_grid > 0).sum() / max(n_reachable, 1)

        # Record metrics at every update
        metrics_history["updates"].append(k + 1)
        metrics_history["unique_cells"].append(batch["unique_cells"])
        metrics_history["cumulative_cov"].append(cov)
        metrics_history["int_reward_sum"].append(batch["rew"].sum())

        if (k + 1) % max(1, args.updates // 10) == 0:
            print(
                f"  update {k + 1:3d}/{args.updates}  "
                f"unique_cells={batch['unique_cells']:3d}  "
                f"cumulative_cov={cov:.3f}  "
                f"int_reward_sum={batch['rew'].sum():.2f}"
            )

    torch.save(
        {
            "actor_state_dict": model.actor.state_dict(),
            "critic_state_dict": model.critic.state_dict(),
            "config": dict(
                grid_type=args.grid_type,
                max_steps=args.max_steps,
                seed=args.seed,
                updates=args.updates,
                steps=args.steps,
                beta=args.beta,
            ),
            "count_grid": count_grid,
        },
        out_path,
    )
    print(f"\nSaved pretrained actor (and critic) to: {out_path.resolve()}")
    print(
        f"Final cumulative coverage: {cov:.3f} of reachable cells "
        f"({(count_grid > 0).sum()}/{n_reachable})"
    )

    # Plot metrics across updates
    plot_pretraining_metrics(metrics_history, out_path.parent)


if __name__ == "__main__":
    main()
