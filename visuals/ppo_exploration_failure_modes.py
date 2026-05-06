"""Verify two PPO exploration failure modes on sparse-reward FourRooms.

We test two hypotheses on `gridworld.FourRooms` (v1, single goal at a fixed
location, reward = 1 only at the goal — purely sparse).

"Exploratory" here means *practical state-space coverage* — does the policy
visit a broad set of cells in early training? — not the entropy of its action
distribution. Action entropy can stay high while the policy oscillates in a
two-cell loop. Conversely, a deterministic policy can explore broadly. So we
measure exploration at the state level.

Per-rollout metrics (each is a number per PPO update):

    * unique_cells       — distinct grid cells visited *within this rollout*
                           (rate of fresh exploration; drops sharply when the
                           policy collapses to a narrow region)
    * visit_entropy      — Shannon entropy of the per-rollout visit-count
                           distribution over reachable cells (uniform spread
                           => high; concentration => low)
    * coverage           — cumulative fraction of reachable cells ever visited
    * mean_return        — sparse-reward signal (1 at goal, 0 otherwise)

Scenario A — Critic warmup
    With a randomly initialized critic, the *first* PPO actor update uses
    near-meaningless advantages. We compare:
      * naive_ppo   : actor trains from update 0
      * warmup_5    : freeze actor for the first 5 updates (critic-only)
      * warmup_15   : freeze actor for the first 15 updates
    If the hypothesis holds, naive_ppo's per-rollout unique_cells / visit_entropy
    drop fast and stay low, and the cumulative visit heatmap shows tight
    clustering near the start cell.

Scenario B — GAE bias / variance
    Even with a reasonable critic, GAE's lambda trades bias for variance:
      * lam=0.0  : pure TD(0) bootstrap — biased by the critic
      * lam=0.95 : usual setting
      * lam=1.0  : pure Monte Carlo — unbiased but high variance
    We expect different coverage signatures for each.

Visit heatmaps (final cumulative visits, log scale, walls masked) are rendered
per variant so the *spatial* pattern of exploration is directly visible.

Run (defaults take ~3 minutes on CPU):

    python visuals/ppo_exploration_failure_modes.py
    python visuals/ppo_exploration_failure_modes.py --rollouts 120 --seeds 5

Outputs PNGs in `visuals/figs_ppo_exploration/`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from gridworld import FourRooms

DEVICE = torch.device("cpu")
N_ACTIONS = 4


# -------- Model ------------------------------------------------------------

class ActorCritic(nn.Module):
    """Tiny MLP actor + MLP critic over the 4-D (agent_xy, goal_xy) obs."""

    def __init__(self, obs_dim: int = 4, n_actions: int = N_ACTIONS, hidden: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def pi(self, x: torch.Tensor) -> Categorical:
        return Categorical(logits=self.actor(x))

    def v(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x).squeeze(-1)


# -------- Config -----------------------------------------------------------

@dataclass
class Config:
    name: str
    rollouts: int = 60
    steps_per_rollout: int = 1024
    epochs_per_update: int = 4
    minibatch: int = 256
    lr_pi: float = 3e-4
    lr_v: float = 1e-3
    clip: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    # entropy_coef = 0 keeps the comparison "natural" — we want to see the
    # *intrinsic* tendency of the loss to collapse exploration, not a forced
    # bonus.
    entropy_coef: float = 0.0
    critic_warmup: int = 0
    grid_type: str = "v1"
    max_steps: int = 100


# -------- GAE --------------------------------------------------------------

def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_v: float,
    gamma: float,
    lam: float,
):
    """Standard GAE with proper episode-boundary masking.

    `dones[t]` should be 1 when the episode terminates *at step t* (either
    natural termination or truncation — both stop the bootstrap chain so the
    advantage at t doesn't leak across resets).
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        nonterm = 1.0 - dones[t]
        next_v = last_v if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_v * nonterm - values[t]
        gae = delta + gamma * lam * nonterm * gae
        adv[t] = gae
    returns = adv + values
    return adv, returns


# -------- Helpers ----------------------------------------------------------

def flat_obs(d: dict, w: int, h: int) -> np.ndarray:
    a, g = d["achieved_goal"], d["desired_goal"]
    return np.array([a[0] / w, a[1] / h, g[0] / w, g[1] / h], dtype=np.float32)


# -------- Rollout ----------------------------------------------------------

def rollout(env, model, T, w, h, cumulative_visit, reachable_mask):
    """Collect T transitions and return both the PPO batch and per-rollout
    state-coverage metrics.

    ``cumulative_visit`` is a (W, H) grid mutated in place so the caller can
    track coverage across the whole training run. We additionally build a
    fresh per-rollout visit grid (`local_visit`) so we can report two
    coverage signals: how broadly the policy explores *right now*, and how
    much of the reachable space the agent has *ever* touched.
    """
    obs_buf = np.zeros((T, 4), dtype=np.float32)
    act_buf = np.zeros(T, dtype=np.int64)
    logp_buf = np.zeros(T, dtype=np.float32)
    rew_buf = np.zeros(T, dtype=np.float32)
    val_buf = np.zeros(T, dtype=np.float32)
    done_buf = np.zeros(T, dtype=np.float32)
    ep_returns: list[float] = []
    ep_lens: list[int] = []
    local_visit = np.zeros_like(cumulative_visit)

    obs, _ = env.reset()
    fobs = flat_obs(obs, w, h)
    cur_ret, cur_len = 0.0, 0

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
        cumulative_visit[int(ax), int(ay)] += 1
        local_visit[int(ax), int(ay)] += 1

        obs, r, term, trunc, _ = env.step(a)
        rew_buf[t] = r
        cur_ret += r
        cur_len += 1
        done = bool(term or trunc)
        done_buf[t] = float(done)
        if done:
            ep_returns.append(cur_ret)
            ep_lens.append(cur_len)
            cur_ret, cur_len = 0.0, 0
            obs, _ = env.reset()
        fobs = flat_obs(obs, w, h)

    with torch.no_grad():
        last_v = model.v(torch.from_numpy(fobs).unsqueeze(0)).item()

    # Per-rollout state-coverage measures.
    visited_mask = (local_visit > 0) & reachable_mask
    unique_cells = int(visited_mask.sum())
    p = local_visit[reachable_mask].astype(np.float64)
    if p.sum() > 0:
        p = p / p.sum()
        # Shannon entropy in nats; uniform over reachable cells == log(n_reachable).
        visit_entropy = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
    else:
        visit_entropy = 0.0

    return dict(
        obs=obs_buf, act=act_buf, logp=logp_buf, rew=rew_buf,
        val=val_buf, done=done_buf, last_v=last_v,
        ep_returns=ep_returns, ep_lens=ep_lens,
        unique_cells=unique_cells, visit_entropy=visit_entropy,
    )


# -------- PPO update -------------------------------------------------------

def ppo_update(model, opt_pi, opt_v, batch, cfg, freeze_actor: bool):
    obs = torch.from_numpy(batch["obs"])
    act = torch.from_numpy(batch["act"])
    old_logp = torch.from_numpy(batch["logp"])
    adv, ret = compute_gae(
        batch["rew"], batch["val"], batch["done"], batch["last_v"],
        cfg.gamma, cfg.gae_lambda,
    )
    adv_t = torch.from_numpy((adv - adv.mean()) / (adv.std() + 1e-8))
    ret_t = torch.from_numpy(ret)

    N = obs.shape[0]
    idx = np.arange(N)
    losses_pi, losses_v = [], []
    for _ in range(cfg.epochs_per_update):
        np.random.shuffle(idx)
        for s in range(0, N, cfg.minibatch):
            mb = idx[s : s + cfg.minibatch]
            mb_obs, mb_act, mb_oldlogp = obs[mb], act[mb], old_logp[mb]
            mb_adv, mb_ret = adv_t[mb], ret_t[mb]

            dist = model.pi(mb_obs)
            logp = dist.log_prob(mb_act)
            ratio = torch.exp(logp - mb_oldlogp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * mb_adv
            entropy = dist.entropy().mean()
            loss_pi = -torch.min(surr1, surr2).mean() - cfg.entropy_coef * entropy

            v = model.v(mb_obs)
            loss_v = F.mse_loss(v, mb_ret)

            if not freeze_actor:
                opt_pi.zero_grad()
                loss_pi.backward()
                opt_pi.step()
            opt_v.zero_grad()
            loss_v.backward()
            opt_v.step()

            losses_pi.append(loss_pi.item())
            losses_v.append(loss_v.item())

    return dict(loss_pi=float(np.mean(losses_pi)),
                loss_v=float(np.mean(losses_v)))


# -------- Single-config training run --------------------------------------

def run_config(cfg: Config, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = FourRooms(grid_type=cfg.grid_type, max_steps=cfg.max_steps)
    w, h = env.width, env.height
    model = ActorCritic().to(DEVICE)
    opt_pi = torch.optim.Adam(model.actor.parameters(), lr=cfg.lr_pi)
    opt_v = torch.optim.Adam(model.critic.parameters(), lr=cfg.lr_v)

    visit = np.zeros((w, h), dtype=np.float32)
    base = env.get_grid()[..., 0]
    # Reachable = anything that isn't a wall (we count the goal as reachable
    # because the agent can stand on the goal cell on the terminating step).
    reachable_mask = base != 2
    n_reachable = int(reachable_mask.sum())

    history = {"unique_cells": [], "visit_entropy": [],
               "coverage": [], "mean_return": [],
               "first_success_update": None}

    for k in range(cfg.rollouts):
        batch = rollout(env, model, cfg.steps_per_rollout,
                        w, h, visit, reachable_mask)
        ppo_update(model, opt_pi, opt_v, batch, cfg,
                   freeze_actor=(k < cfg.critic_warmup))

        ret_mean = float(np.mean(batch["ep_returns"])) if batch["ep_returns"] else 0.0
        cov = float((visit > 0).sum() / max(n_reachable, 1))
        history["unique_cells"].append(batch["unique_cells"])
        history["visit_entropy"].append(batch["visit_entropy"])
        history["coverage"].append(cov)
        history["mean_return"].append(ret_mean)
        if history["first_success_update"] is None and ret_mean > 0:
            history["first_success_update"] = k

    return history, visit


# -------- Multi-seed runner ------------------------------------------------

def run_variants(variants: list[dict], seeds: int, base: Config):
    out: dict[str, dict] = {}
    for v in variants:
        cfg = replace(base, **v)
        hists, visits = [], []
        for s in range(seeds):
            print(f"  [{cfg.name}] seed {s} ...", flush=True)
            h, vg = run_config(cfg, seed=s)
            hists.append(h)
            visits.append(vg)
        out[cfg.name] = dict(history=hists, visit=visits, cfg=cfg)
    return out


# -------- Plotting ---------------------------------------------------------

def plot_curves(results: dict, key: str, ax, title: str, ylabel: str):
    for name, d in results.items():
        ys = np.array([h[key] for h in d["history"]])  # (seeds, T)
        m, s = ys.mean(0), ys.std(0)
        x = np.arange(m.shape[0])
        ax.plot(x, m, label=name, linewidth=2)
        ax.fill_between(x, m - s, m + s, alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel("PPO update")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


def plot_visit(results: dict, fig_path: Path, base_grid: np.ndarray):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.4))
    if n == 1:
        axes = [axes]
    walls = (base_grid == 2).T  # (H, W)
    for ax, (name, d) in zip(axes, results.items()):
        v = np.mean(d["visit"], axis=0).T  # (H, W)
        v = np.log1p(v)
        # Mask walls visually: paint them black on top.
        img = np.ma.masked_where(walls, v)
        ax.imshow(img, cmap="magma")
        ax.imshow(np.where(walls, 1.0, np.nan), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{name}\nlog visitation (mean over seeds)")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)


def summary_table(results: dict) -> str:
    lines = [f"{'variant':<14}  {'final_ent':>10}  {'final_cov':>10}  "
             f"{'final_ret':>10}  {'first_succ':>10}"]
    for name, d in results.items():
        ent = np.mean([h["entropy"][-1] for h in d["history"]])
        cov = np.mean([h["coverage"][-1] for h in d["history"]])
        ret = np.mean([h["mean_return"][-1] for h in d["history"]])
        firsts = [h["first_success_update"] for h in d["history"]
                  if h["first_success_update"] is not None]
        first = float(np.mean(firsts)) if firsts else float("nan")
        lines.append(f"{name:<14}  {ent:>10.3f}  {cov:>10.3f}  "
                     f"{ret:>10.3f}  {first:>10.1f}")
    return "\n".join(lines)


# -------- Entry point -----------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=int, default=60)
    ap.add_argument("--steps", type=int, default=1024)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=str, default="visuals/figs_ppo_exploration")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = Config(name="base", rollouts=args.rollouts,
                  steps_per_rollout=args.steps)

    # Reference grid for plotting (walls etc.) — use a fresh env so we don't
    # share state with the training envs.
    ref_env = FourRooms(grid_type=base.grid_type, max_steps=base.max_steps)
    base_grid = ref_env.get_grid()[..., 0]

    print(f"\n=== Scenario A: critic warmup (sparse-reward FourRooms-{base.grid_type}) ===")
    A = run_variants([
        dict(name="naive_ppo",  critic_warmup=0),
        dict(name="warmup_5",   critic_warmup=5),
        dict(name="warmup_15",  critic_warmup=15),
    ], seeds=args.seeds, base=base)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_curves(A, "entropy",     axes[0], "policy entropy",       "H(π)")
    plot_curves(A, "mean_return", axes[1], "mean episode return",  "return")
    plot_curves(A, "coverage",    axes[2], "cumulative state coverage",
                "fraction of reachable cells")
    fig.suptitle("Scenario A: actor update with poor critic kills exploration",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "scenario_A_curves.png", dpi=130)
    plt.close(fig)
    plot_visit(A, out_dir / "scenario_A_visits.png", base_grid)
    print(summary_table(A))

    print(f"\n=== Scenario B: GAE bias/variance ===")
    B = run_variants([
        dict(name="lam=0.0",  gae_lambda=0.0),
        dict(name="lam=0.95", gae_lambda=0.95),
        dict(name="lam=1.0",  gae_lambda=1.0),
    ], seeds=args.seeds, base=base)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_curves(B, "entropy",     axes[0], "policy entropy",       "H(π)")
    plot_curves(B, "mean_return", axes[1], "mean episode return",  "return")
    plot_curves(B, "coverage",    axes[2], "cumulative state coverage",
                "fraction of reachable cells")
    fig.suptitle("Scenario B: GAE λ — bias (low λ) vs variance (high λ)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "scenario_B_curves.png", dpi=130)
    plt.close(fig)
    plot_visit(B, out_dir / "scenario_B_visits.png", base_grid)
    print(summary_table(B))

    print(f"\nFigures written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
