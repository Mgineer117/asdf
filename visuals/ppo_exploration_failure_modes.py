"""Verify the early-update exploration-collapse hypothesis on sparse FourRooms.

Setup (single experiment, single figure):

* Policy is initialized with a *zero-mean* final actor layer so the initial
  action distribution is uniform — i.e., the agent starts as a 4-action
  random walker, which on FourRooms covers a broad set of cells per rollout.
* Critic is randomly initialized (poor) — values are essentially noise.
* Reward is purely sparse: r = 1 only when the agent reaches the goal cell.

Hypothesis being tested:
    With a poor random critic, the *first 1-5 PPO updates* compute advantages
    that are essentially noise. PPO follows that noise with clipped gradient
    steps that pull the actor away from uniform — toward whatever direction
    the random V function happens to favor. On a sparse-reward task no real
    learning signal is yet available to correct this drift, so the policy
    *prematurely collapses* into a narrow region of state space and may
    never recover. Warming up the critic first (with the actor frozen at
    uniform) lets V settle near 0 everywhere, so the first actor update has
    tiny, near-isotropic advantages and the policy stays exploratory.

Ablation:
    GAE λ ∈ {0.90, 0.95, 1.00}  ×  warmup ∈ {off, on}
    -> 6 conditions, plotted side-by-side as no-warmup (left) vs warmup (right).

Metric:
    `unique cells per rollout` — distinct grid cells the policy visits within
    a single 1024-step rollout. Direct measure of *practical state-space
    exploration*. We log it over snapshots t = 0..N where t=0 is the initial
    uniform policy and t=k is the policy after k actor updates.

Run (defaults take ~30 seconds on CPU):

    python visuals/ppo_exploration_failure_modes.py
    python visuals/ppo_exploration_failure_modes.py --seeds 5

Outputs to `visuals/figs_ppo_exploration/initial_update_exploration.svg`.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

# Make project root importable regardless of cwd.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

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
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # Zero-init actor's final projection: logits = 0 => softmax uniform
        # over actions. This isolates the experiment to "what does the *first*
        # PPO update do to a known-uniform policy paired with a poor critic?".
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)

    def pi(self, x: torch.Tensor) -> Categorical:
        return Categorical(logits=self.actor(x))

    def v(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x).squeeze(-1)


# -------- Config -----------------------------------------------------------


@dataclass
class Config:
    name: str
    n_updates: int = 100  # actor-PPO updates we plot (1..N)
    pre_critic_updates: int = 0  # critic-only updates BEFORE the main loop
    steps_per_rollout: int = 1024
    epochs_per_update: int = 4
    minibatch: int = 256
    lr_pi: float = 3e-4
    lr_v: float = 3e-4
    clip: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    # entropy_coef = 0 keeps the comparison "natural" — we want to see the
    # *intrinsic* tendency of the loss to collapse exploration, not a forced
    # bonus.
    entropy_coef: float = 0.0
    grid_type: str = "v1"
    max_steps: int = 50
    # If non-None, load actor weights from this checkpoint at construction
    # time (overrides the zero-init final layer). This is how we plug in a
    # pretrained "diffusion" policy from `pretrain_diffusion_policy.py`.
    init_from: str | None = None


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
        obs=obs_buf,
        act=act_buf,
        logp=logp_buf,
        rew=rew_buf,
        val=val_buf,
        done=done_buf,
        last_v=last_v,
        ep_returns=ep_returns,
        ep_lens=ep_lens,
        unique_cells=unique_cells,
        visit_entropy=visit_entropy,
    )


# -------- PPO update -------------------------------------------------------


def ppo_update(model, opt_pi, opt_v, batch, cfg, freeze_actor: bool):
    obs = torch.from_numpy(batch["obs"])
    act = torch.from_numpy(batch["act"])
    old_logp = torch.from_numpy(batch["logp"])
    adv, ret = compute_gae(
        batch["rew"],
        batch["val"],
        batch["done"],
        batch["last_v"],
        cfg.gamma,
        cfg.gae_lambda,
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

    return dict(loss_pi=float(np.mean(losses_pi)), loss_v=float(np.mean(losses_v)))


# -------- Single-config training run --------------------------------------


def _actor_spectral_norms(model: ActorCritic) -> dict[str, float]:
    """Largest singular value of each Linear weight in the actor.

    We expose this as a compact "weight property" diagnostic — the user's
    third hypothesis is that intrinsic-reward pretraining changes the
    spectral structure of the policy network in ways that matter for the
    *next* phase of learning. Comparing these values between random-init and
    pretrained-init runs gives a quantitative read on that.
    """
    out = {}
    for i, m in enumerate(model.actor):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                s = float(torch.linalg.matrix_norm(m.weight, ord=2).item())
            out[f"actor.linear[{i}]"] = s
    return out


def run_config(cfg: Config, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = FourRooms(grid_type=cfg.grid_type, max_steps=cfg.max_steps)
    w, h = env.width, env.height
    model = ActorCritic().to(DEVICE)

    # Optionally override the zero-init actor with a pretrained checkpoint.
    # Only the actor is loaded — the critic stays randomly initialized so we
    # can keep testing the "poor critic" half of the original hypothesis even
    # when the *policy* is no longer uniform-random.
    if cfg.init_from:
        ckpt = torch.load(cfg.init_from, map_location=DEVICE, weights_only=False)
        model.actor.load_state_dict(ckpt["actor_state_dict"])

    opt_pi = torch.optim.Adam(model.actor.parameters(), lr=cfg.lr_pi)
    opt_v = torch.optim.Adam(model.critic.parameters(), lr=cfg.lr_v)

    visit = np.zeros((w, h), dtype=np.float32)
    base = env.get_grid()[..., 0]
    # Reachable = anything that isn't a wall (we count the goal as reachable
    # because the agent can stand on the goal cell on the terminating step).
    reachable_mask = base != 2
    n_reachable = int(reachable_mask.sum())

    # ---- Pre-phase: K critic-only updates, no logging ---------------------
    # The actor stays at its zero-init uniform distribution because we freeze
    # it; the critic gets to see uniform-policy rollouts and (with all-zero
    # reward) typically learns V≈0 everywhere. After this, the *first* actor
    # update sees small/calibrated advantages instead of a random V function.
    for _ in range(cfg.pre_critic_updates):
        batch = rollout(env, model, cfg.steps_per_rollout, w, h, visit, reachable_mask)
        ppo_update(model, opt_pi, opt_v, batch, cfg, freeze_actor=True)
    # Don't let warmup rollouts contaminate the visitation heatmap.
    visit *= 0
    # Re-seed before the main phase so both warmup-on and warmup-off variants
    # start from the *same* RNG state (env reset, action sampling). Otherwise
    # warmup advances the global RNG by ~K * steps_per_rollout extra draws and
    # the t=0 unique-cells point differs purely from RNG drift, masking the
    # effect we want to measure.
    main_seed = seed + 1000
    torch.manual_seed(main_seed)
    np.random.seed(main_seed)

    # ---- Main phase: log N+1 snapshots, run N actor+critic updates --------
    # snapshot k records the rollout collected with the policy after k actor
    # updates (k=0 is the initial uniform policy).
    history = {
        "unique_cells": [],
        "visit_entropy": [],
        "coverage": [],
        "mean_return": [],
        "first_success_update": None,
        # Spectral norms of actor Linear weights at t=0 (start of main phase).
        # Constant for the run; useful for the "init effects spectral norm"
        # diagnostic from the user's hypothesis.
        "spectral_norms_at_main_start": _actor_spectral_norms(model),
    }

    # Track per-update visitation snapshots for visualization
    visit_snapshots = []

    for k in range(cfg.n_updates + 1):
        batch = rollout(env, model, cfg.steps_per_rollout, w, h, visit, reachable_mask)
        ret_mean = float(np.mean(batch["ep_returns"])) if batch["ep_returns"] else 0.0
        cov = float((visit > 0).sum() / max(n_reachable, 1))
        history["unique_cells"].append(batch["unique_cells"])
        history["visit_entropy"].append(batch["visit_entropy"])
        history["coverage"].append(cov)
        history["mean_return"].append(ret_mean)
        if history["first_success_update"] is None and ret_mean > 0:
            history["first_success_update"] = k

        # Store a snapshot of visitation at this update
        visit_snapshots.append(visit.copy())

        if k < cfg.n_updates:
            ppo_update(model, opt_pi, opt_v, batch, cfg, freeze_actor=False)

    return history, visit, visit_snapshots


# -------- Multi-seed runner ------------------------------------------------


def run_variants(variants: list[dict], seeds: int, base: Config):
    out: dict[str, dict] = {}
    for v in variants:
        cfg = replace(base, **v)
        hists, visits, visit_snaps = [], [], []
        for s in range(seeds):
            print(f"  [{cfg.name}] seed {s} ...", flush=True)
            h, vg, vs = run_config(cfg, seed=s)
            hists.append(h)
            visits.append(vg)
            visit_snaps.append(vs)
        out[cfg.name] = dict(
            history=hists, visit=visits, visit_snapshots=visit_snaps, cfg=cfg
        )
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
    n_variants = len(results)
    n_update_slices = 10

    # Figure with 10 rows (updates) and 6 columns (variants)
    fig, axes = plt.subplots(
        n_update_slices, n_variants, figsize=(3 * n_variants, 3 * n_update_slices)
    )
    if n_variants == 1:
        axes = axes.reshape(-1, 1)

    walls = (base_grid == 2).T  # (H, W)
    variant_names = list(results.keys())

    # Get the number of updates from the first variant's snapshots
    n_updates = len(results[variant_names[0]]["visit_snapshots"][0])
    update_slice_size = max(1, n_updates // n_update_slices)

    for col, name in enumerate(variant_names):
        d = results[name]
        visit_snapshots_all_seeds = d["visit_snapshots"]  # List[List[(W, H) arrays]]

        for row in range(n_update_slices):
            ax = axes[row, col]
            update_idx = min(row * update_slice_size, n_updates - 1)

            # Average the visitation snapshot at this update across all seeds
            avg_visit = np.zeros_like(visit_snapshots_all_seeds[0][0])
            for seed_idx, visit_snapshots in enumerate(visit_snapshots_all_seeds):
                avg_visit += visit_snapshots[update_idx]
            avg_visit /= len(visit_snapshots_all_seeds)

            # Convert to log scale and transpose for display
            v = np.log1p(avg_visit).T  # (H, W)

            # Mask walls visually: paint them black on top.
            img = np.ma.masked_where(walls, v)
            ax.imshow(img, cmap="magma")
            ax.imshow(np.where(walls, 1.0, np.nan), cmap="gray", vmin=0, vmax=1)

            if row == 0:
                ax.set_title(f"{name}", fontsize=9)

            if col == 0:
                ax.set_ylabel(f"Update ~{update_idx}", fontsize=8)

            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        "Visitation heatmaps across training updates (10 slices × 6 variants)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)


def summary_table(results: dict) -> str:
    """Per-variant final-state summary, averaged over seeds.

    `early_uniq` is the mean per-rollout unique-cell count over the first
    25% of training updates — this is the cleanest snapshot of *initial*
    exploration, which is what the hypothesis is about.

    `||W||₂ avg` is the mean (over actor Linear layers and seeds) spectral
    norm of the actor's weights at the start of the main phase. Comparing
    this between random-init and pretrained-init runs gives a quantitative
    read on whether pretraining changed weight properties in addition to
    behavior.
    """
    lines = [
        f"{'variant':<22}  {'early_uniq':>10}  {'final_uniq':>10}  "
        f"{'final_cov':>10}  {'final_ret':>10}  {'first_succ':>10}  "
        f"{'||W||2':>8}"
    ]
    for name, d in results.items():
        uniq_curves = np.array([h["unique_cells"] for h in d["history"]])
        T = uniq_curves.shape[1]
        early = float(uniq_curves[:, : max(1, T // 4)].mean())
        final_uniq = float(uniq_curves[:, -1].mean())
        cov = np.mean([h["coverage"][-1] for h in d["history"]])
        ret = np.mean([h["mean_return"][-1] for h in d["history"]])
        firsts = [
            h["first_success_update"]
            for h in d["history"]
            if h["first_success_update"] is not None
        ]
        first = float(np.mean(firsts)) if firsts else float("nan")
        # Average spectral norm across actor Linear layers, then over seeds.
        sn_per_seed = [
            float(np.mean(list(h["spectral_norms_at_main_start"].values())))
            for h in d["history"]
        ]
        sn = float(np.mean(sn_per_seed))
        lines.append(
            f"{name:<22}  {early:>10.1f}  {final_uniq:>10.1f}  "
            f"{cov:>10.3f}  {ret:>10.3f}  {first:>10.1f}  {sn:>8.3f}"
        )
    return "\n".join(lines)


# -------- Entry point -----------------------------------------------------

GAE_LAMBDAS = [0.90, 0.95, 1.00]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--n-updates",
        type=int,
        default=200,
        help="number of plotted PPO actor updates (1..N)",
    )
    ap.add_argument("--steps", type=int, default=1024, help="env steps per rollout")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="critic-only updates before the warmup variant's " "first actor update",
    )
    ap.add_argument("--out", type=str, default="visuals/figs_ppo_exploration")
    ap.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="path to a checkpoint produced by pretrain_diffusion_policy.py; "
        "if given, the actor is initialized from this checkpoint instead of "
        "the zero-init uniform policy. The critic is still randomly initialized.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = Config(
        name="base",
        n_updates=args.n_updates,
        steps_per_rollout=args.steps,
        init_from=args.init_from,
    )

    init_tag = "pretrained" if args.init_from else "uniform"
    print(
        f"actor init: {init_tag}" + (f" ({args.init_from})" if args.init_from else "")
    )

    # 6 = 3 λs × {warmup off, warmup on}
    variants: list[dict] = []
    for lam in GAE_LAMBDAS:
        variants.append(
            dict(name=f"lam={lam:.2f},warmup=0", gae_lambda=lam, pre_critic_updates=0)
        )
        variants.append(
            dict(
                name=f"lam={lam:.2f},warmup={args.warmup}",
                gae_lambda=lam,
                pre_critic_updates=args.warmup,
            )
        )

    print("=== Initial PPO updates with poor critic — λ × warmup ablation ===")
    print(
        f"FourRooms-{base.grid_type}, sparse reward, {init_tag}-init policy, "
        f"{args.n_updates} updates, {args.seeds} seeds"
    )
    results = run_variants(variants, seeds=args.seeds, base=base)

    # Reference grid (walls) for the heatmap.
    ref_env = FourRooms(grid_type=base.grid_type, max_steps=base.max_steps)
    base_grid = ref_env.get_grid()[..., 0]

    # ---- Single combined plot: 3 rows (metrics) × 2 cols (warmup) --------
    # Reading guide:
    #   * unique_cells dropping AND mean_return staying ~0 ⇒ unexploratory
    #     collapse (the failure mode the hypothesis predicts).
    #   * unique_cells dropping AND mean_return rising ⇒ legitimate goal
    #     seeking; the policy concentrated *toward* the goal.
    #   * visit_entropy is the spread of the in-rollout visit distribution;
    #     it complements unique_cells (uniform vs clumpy among visited
    #     cells, beyond just "how many cells").
    metrics = [
        ("unique_cells", "unique cells per rollout"),
        ("visit_entropy", "state-visitation entropy [nats]"),
        ("mean_return", "mean episode return"),
    ]
    cmap = plt.cm.viridis
    colors = {
        lam: cmap(i / (len(GAE_LAMBDAS) - 1)) for i, lam in enumerate(GAE_LAMBDAS)
    }
    columns = [
        (0, "no critic warmup"),
        (args.warmup, f"critic warmup ({args.warmup} updates)"),
    ]

    fig, axes = plt.subplots(
        len(metrics), 2, figsize=(12, 3.2 * len(metrics)), sharex=True
    )
    for r, (mkey, ylabel) in enumerate(metrics):
        for c, (warmup_val, col_title) in enumerate(columns):
            ax = axes[r, c]
            for lam in GAE_LAMBDAS:
                key = f"lam={lam:.2f},warmup={warmup_val}"
                d = results[key]
                ys = np.array([h[mkey] for h in d["history"]])
                m, s = ys.mean(0), ys.std(0)
                x = np.arange(m.shape[0])
                ax.plot(
                    x,
                    m,
                    label=f"λ={lam:.2f}",
                    linewidth=2.0,
                    color=colors[lam],
                    marker="o",
                    markersize=4,
                )
                ax.fill_between(x, m - s, m + s, alpha=0.18, color=colors[lam])
            if r == 0:
                ax.set_title(col_title, fontsize=11)
            if r == len(metrics) - 1:
                ax.set_xlabel("PPO update (0 = initial uniform policy)")
            ax.set_xticks(np.arange(args.n_updates + 1))
            if c == 0:
                ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            if r == 0 and c == 0:
                ax.legend(fontsize=9, title="GAE λ", loc="best")
        # Share y-axis across the two warmup columns within a row so the
        # left/right comparison is visually fair.
        ymin = min(axes[r, c].get_ylim()[0] for c in range(2))
        ymax = max(axes[r, c].get_ylim()[1] for c in range(2))
        for c in range(2):
            axes[r, c].set_ylim(ymin, ymax)

    fig.suptitle(
        f"Poor-critic early-update exploration on sparse FourRooms — "
        f"{init_tag}-init policy\n"
        "(mean ± std over seeds; drop in unique_cells with flat return ⇒ "
        "collapse, drop with rising return ⇒ goal-seeking)",
        fontsize=11,
    )
    fig.tight_layout()
    # Tag filenames so init=uniform and init=pretrained runs don't overwrite
    # each other in the same output directory.
    fig_path = out_dir / f"initial_update_exploration_{init_tag}.svg"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    print(f"\nFigure: {fig_path.resolve()}")

    # ---- Optional: visitation heatmap (helpful for spatial intuition) ----
    plot_visit(results, out_dir / f"initial_update_visits_{init_tag}.svg", base_grid)

    # ---- Summary table ---------------------------------------------------
    print()
    print(summary_table(results))


if __name__ == "__main__":
    main()
