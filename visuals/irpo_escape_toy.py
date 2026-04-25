"""
Toy visualization of the IRPO mechanism on a 2D non-convex smooth loss.

We strip IRPO down to the essentials:
  * parameters theta in R^2
  * extrinsic loss L(theta): smooth, non-convex, with a shallow local min and a
    deeper global min
  * intrinsic loss L_int(theta) = -L(theta)
  * inner "exploratory" rollout: K gradient steps on L_int (ascends L,
    escaping the local basin)
  * outer (base) gradient: chain-rule backprop through the inner rollout,
    exactly as in policy/irpo.py :: backprop

We compare plain GD on L against IRPO's base update on the same starting point
(inside a local basin) and plot:
  1) the loss landscape with both trajectories,
  2) the inner rollout fans at a few outer iterations,
  3) L vs outer iteration for both methods.

Run:
    python visuals/irpo_escape_toy.py
"""

from __future__ import annotations

from copy import deepcopy
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import grad


# -------- loss landscape --------------------------------------------------


# Each tuple: (center_x, center_y, amplitude). The well with the largest
# amplitude is the global minimum of L; the others are local minima.
WELLS: List[Tuple[float, float, float]] = [
    (1.6, 1.4, 1.40),  # <-- global min
    (-1.7, 1.3, 0.75),
    (-1.4, -1.0, 0.70),
    (0.6, -1.4, 0.80),
    (1.7, -0.5, 0.85),
    (0.0, 0.6, 0.65),
]
SIGMA = 0.75
BOWL_COEF = 0.001  # A small non-zero value to make the landscape almost flat but still connected.

# Non-smooth ridge — a tent-shaped "wall" along the line x - y = RIDGE_CENTER.
# L picks up a + RIDGE_HEIGHT * relu(RIDGE_HALF_WIDTH - |x - y - RIDGE_CENTER|)
# term, which is piecewise-linear: zero outside the strip, rising linearly
# toward the crest.  The gradient flips sign abruptly across the crest and
# vanishes outside the strip — a drastic, non-smooth feature the optimizer has
# to deal with on the way to the global well.
RIDGE_CENTER = 1.0
RIDGE_HALF_WIDTH = 0.5
RIDGE_HEIGHT = 0.3

# Multi-option IRPO: each "option" corresponds to a different intrinsic reward
# in the real algorithm (policy/irpo.py: num_options).  In the toy we model
# this as a fixed directional drift added to every inner (exploratory) step,
# so different options push the rollout into different halves of parameter
# space.  After all options have rolled out, we pick the one whose final θ_K
# has the lowest extrinsic L — argmax aggregation by performance gain (see
# policy/irpo.py:277 greedy_idx and :296-298 aggregation_method="argmax").
NUM_OPTIONS = 8
_opt_angles = np.linspace(0.0, 2.0 * np.pi, NUM_OPTIONS, endpoint=False)
OPTION_DIRS = np.stack([np.cos(_opt_angles), np.sin(_opt_angles)], axis=1)
OPTION_BIAS = 0.6  # per-step drift magnitude; larger ⇒ wider exploration fan

# Gradient-evaluation noise.  Mirrors IRPO's `noise_std` (see policy/irpo.py).
# Disabled by default — the story here is landscape non-smoothness, not sample
# noise.  Bump above 0 to add stochastic-policy-style noise on top.
NOISE_STD = 0.0


def _noisy_grad(
    loss: torch.Tensor, theta: torch.Tensor, create_graph: bool = False
) -> torch.Tensor:
    """grad(loss, theta) with additive iid Gaussian noise.

    The noise is detached, so it does not contribute to the downstream autograd
    graph — the Hessian-vector product used in IRPO's chain-rule backprop is
    still the true Hessian, while the forward rollout sees noisy gradients
    (just like a stochastic policy-gradient method in RL).
    """
    g = grad(loss, theta, create_graph=create_graph)[0]
    if NOISE_STD > 0:
        g = g + (NOISE_STD * torch.randn_like(g)).detach()
    return g


def L(theta: torch.Tensor) -> torch.Tensor:
    """Multi-well landscape + a very weak quadratic bowl + one non-smooth tent-ridge.

    The Gaussian part is smooth and non-convex with multiple local minima at
    WELLS (global = largest amplitude).  The ridge term is piecewise-linear
    (relu ∘ abs), giving a crease where the gradient changes abruptly — a
    non-smooth "valley-and-hill" obstacle between the start and the global
    well.
    """
    x, y = theta[0], theta[1]
    val = BOWL_COEF * (x**2 + y**2)
    for cx, cy, amp in WELLS:
        val = val - amp * torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * SIGMA**2))
    val = val + RIDGE_HEIGHT * torch.relu(
        RIDGE_HALF_WIDTH - torch.abs(x - y - RIDGE_CENTER)
    )
    return val


def L_int(theta: torch.Tensor) -> torch.Tensor:
    return -L(theta)


# -------- IRPO-style update ----------------------------------------------


def irpo_step(
    theta0: torch.Tensor,
    inner_lr: float,
    K: int,
    option_drift: torch.Tensor = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """One IRPO update for a single option.

    Mirrors policy/irpo.py:
      * K-1 inner steps on L_int  (exploratory phase; irpo uses int advantages
        for the non-final inner updates)
      * final inner step uses the extrinsic gradient
      * backprop through the inner chain via grad(..., grad_outputs=grads)

    ``option_drift`` is added to every non-final inner step as a constant
    direction bias — this is how we model different intrinsic rewards pushing
    the exploratory policy in different directions.  Being constant in θ, it
    contributes zero to the Hessian chain, so the meta-gradient backprop is
    unaffected.
    """
    thetas: List[torch.Tensor] = [theta0.detach().clone().requires_grad_(True)]
    grads_per_step: List[torch.Tensor] = []

    for j in range(K):
        last_step = j == K - 1
        loss = L(thetas[-1]) if last_step else L_int(thetas[-1])
        g = _noisy_grad(loss, thetas[-1], create_graph=True)
        grads_per_step.append(g)
        step = -inner_lr * g
        if (not last_step) and (option_drift is not None):
            step = step + option_drift.detach()
        thetas.append(thetas[-1] + step)

    # Chain-rule backprop (see policy/irpo.py :: backprop).  option_drift is
    # a constant in θ so it drops out of d/dθ; the recursion below is the
    # same as in the unbiased case.
    grads = grads_per_step[-1]
    for j in reversed(range(K - 1)):
        Hv = grad(grads_per_step[j], thetas[j], grad_outputs=grads, retain_graph=True)[
            0
        ]
        grads = grads - inner_lr * Hv

    rollout = [t.detach().clone() for t in thetas]
    return grads.detach(), rollout


def irpo_step_multi(
    theta0: torch.Tensor,
    inner_lr: float,
    K: int,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[List[torch.Tensor]], int]:
    """Run NUM_OPTIONS parallel exploratory rollouts and pick the best.

    Selection: argmax over extrinsic performance at the rollout endpoint θ_K,
    i.e. the option with the lowest L(θ_K) wins.  This is the same greedy
    aggregation used in policy/irpo.py when temperature → 0.
    """
    best_L = float("inf")
    best_grad: torch.Tensor = None
    best_rollout: List[torch.Tensor] = None
    best_idx = 0
    all_rollouts: List[List[torch.Tensor]] = []
    for m in range(NUM_OPTIONS):
        drift = torch.tensor(OPTION_BIAS * OPTION_DIRS[m], dtype=torch.float64)
        g, rollout = irpo_step(theta0, inner_lr, K, option_drift=drift)
        all_rollouts.append(rollout)
        L_K = L(rollout[-1]).item()
        if L_K < best_L:
            best_L = L_K
            best_grad = g
            best_rollout = rollout
            best_idx = m
    return best_grad, best_rollout, all_rollouts, best_idx


# -------- experiment ------------------------------------------------------


def run_gd(theta0: np.ndarray, lr: float, n_iters: int) -> np.ndarray:
    theta = torch.tensor(theta0, dtype=torch.float64)
    path = [theta.numpy().copy()]
    for _ in range(n_iters):
        t = theta.clone().requires_grad_(True)
        g = _noisy_grad(L(t), t)
        theta = theta - lr * g.detach()
        path.append(theta.numpy().copy())
    return np.array(path)


def run_irpo(
    theta0: np.ndarray,
    outer_lr: float,
    inner_lr: float,
    K: int,
    n_iters: int,
) -> Tuple[np.ndarray, List[List[np.ndarray]]]:
    theta = torch.tensor(theta0, dtype=torch.float64)
    path = [theta.numpy().copy()]
    rollouts: List[List[np.ndarray]] = []
    for _ in range(n_iters):
        result = irpo_step_multi(theta, inner_lr=inner_lr, K=K)
        g, rollout = result[0], result[1]
        rollouts.append([t.numpy() for t in rollout])
        theta = theta - outer_lr * g
        path.append(theta.numpy().copy())
    return np.array(path), rollouts


def loss_curve(path: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return np.array([L(torch.tensor(p, dtype=torch.float64)).item() for p in path])


# -------- plotting --------------------------------------------------------


def _plot_wells(ax) -> None:
    """Mark every well on the landscape: largest-amplitude = global, rest = local."""
    global_idx = int(np.argmax([amp for _, _, amp in WELLS]))
    first_local_idx = 0 if global_idx != 0 else 1
    for i, well in enumerate(WELLS):
        cx, cy = well[0], well[1]
        if i == global_idx:
            ax.plot(
                cx,
                cy,
                "*",
                color="white",
                markersize=18,
                markeredgecolor="black",
                label="global min",
            )
        else:
            ax.plot(
                cx,
                cy,
                "X",
                color="white",
                markersize=10,
                markeredgecolor="black",
                label="local min" if i == first_local_idx else None,
            )


def plot(
    gd_path: np.ndarray,
    irpo_path: np.ndarray,
    irpo_rollouts: List[List[np.ndarray]],
    out_path: str,
) -> None:
    # contour grid
    xs = np.linspace(-3.5, 3.5, 300)
    ys = np.linspace(-3.5, 3.5, 300)
    X, Y = np.meshgrid(xs, ys)
    pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float64)
    with torch.no_grad():
        Z = np.array([L(p).item() for p in pts]).reshape(X.shape)

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0])

    # ---- left: landscape + trajectories -----------------------------------
    ax = fig.add_subplot(gs[0, 0])
    cf = ax.contourf(X, Y, Z, levels=30, cmap="viridis", alpha=0.85)
    ax.contour(X, Y, Z, levels=15, colors="k", linewidths=0.3, alpha=0.4)
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label="L(θ)")

    # show a few inner rollouts (thin dashed lines) to make the "climb uphill"
    # step visible.  Pick ~6 outer iterations evenly across the run.
    n = len(irpo_rollouts)
    show_idx = np.linspace(0, n - 1, min(8, n)).astype(int)
    for idx in show_idx:
        r = np.array(irpo_rollouts[idx])
        ax.plot(r[:, 0], r[:, 1], "--", color="white", linewidth=0.9, alpha=0.9)
        ax.plot(
            r[-1, 0],
            r[-1, 1],
            "o",
            color="white",
            markersize=3,
            markeredgecolor="black",
            markeredgewidth=0.4,
        )

    ax.plot(
        gd_path[:, 0],
        gd_path[:, 1],
        "-",
        color="#e74c3c",
        linewidth=2.2,
        label="plain GD",
        marker="o",
        markersize=3,
    )
    ax.plot(
        irpo_path[:, 0],
        irpo_path[:, 1],
        "-",
        color="#f1c40f",
        linewidth=2.2,
        label="IRPO (base θ)",
        marker="o",
        markersize=3,
    )

    ax.plot(
        gd_path[0, 0], gd_path[0, 1], "s", color="black", markersize=9, label="start"
    )
    _plot_wells(ax)

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel("θ₁")
    ax.set_ylabel("θ₂")
    ax.set_title(
        "Landscape + trajectories\n(white dashed = inner exploratory rollouts)"
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_aspect("equal")

    # ---- right: loss vs outer iteration -----------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(
        loss_curve(gd_path),
        color="#e74c3c",
        linewidth=2,
        label="plain GD",
        marker="o",
        markersize=3,
    )
    ax2.plot(
        loss_curve(irpo_path),
        color="#f1c40f",
        linewidth=2,
        label="IRPO",
        marker="o",
        markersize=3,
    )
    ax2.set_xlabel("outer iteration")
    ax2.set_ylabel("L(θ)")
    ax2.set_title("Extrinsic loss vs iteration")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")


def plot_K_sweep(
    gd_path: np.ndarray,
    irpo_results: List[Tuple[int, np.ndarray, List[List[np.ndarray]]]],
    out_path: str,
) -> None:
    """Left: landscape with one trajectory per K.  Right: L vs iter, one line per K."""
    xs = np.linspace(-3.5, 3.5, 300)
    ys = np.linspace(-3.5, 3.5, 300)
    X, Y = np.meshgrid(xs, ys)
    pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float64)
    with torch.no_grad():
        Z = np.array([L(p).item() for p in pts]).reshape(X.shape)

    cmap = plt.get_cmap("plasma")
    Ks = [k for k, _, _ in irpo_results]
    color_for = {k: cmap(i / max(1, len(Ks) - 1)) for i, k in enumerate(Ks)}

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0])

    # ---- left: landscape + per-K base trajectories ------------------------
    ax = fig.add_subplot(gs[0, 0])
    cf = ax.contourf(X, Y, Z, levels=30, cmap="viridis", alpha=0.85)
    ax.contour(X, Y, Z, levels=15, colors="k", linewidths=0.3, alpha=0.4)
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label="L(θ)")

    ax.plot(
        gd_path[:, 0],
        gd_path[:, 1],
        "-",
        color="#e74c3c",
        linewidth=1.8,
        alpha=0.9,
        label="plain GD (K=0)",
        marker="o",
        markersize=2.5,
    )

    for K, irpo_path, _ in irpo_results:
        ax.plot(
            irpo_path[:, 0],
            irpo_path[:, 1],
            "-",
            color=color_for[K],
            linewidth=2.0,
            label=f"IRPO K={K}",
            marker="o",
            markersize=3,
        )

    ax.plot(
        gd_path[0, 0], gd_path[0, 1], "s", color="black", markersize=9, label="start"
    )
    _plot_wells(ax)

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel("θ₁")
    ax.set_ylabel("θ₂")
    ax.set_title("Base-policy trajectory vs num_exp_updates K")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_aspect("equal")

    # ---- right: L vs iter per K ------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(
        loss_curve(gd_path),
        color="#e74c3c",
        linewidth=1.8,
        label="plain GD (K=0)",
        marker="o",
        markersize=2.5,
    )
    for K, irpo_path, _ in irpo_results:
        ax2.plot(
            loss_curve(irpo_path),
            color=color_for[K],
            linewidth=2,
            label=f"IRPO K={K}",
            marker="o",
            markersize=3,
        )
    ax2.set_xlabel("outer iteration")
    ax2.set_ylabel("L(θ)")
    ax2.set_title("Extrinsic loss vs iteration, per K")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")


# -------- main ------------------------------------------------------------


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    # Start inside the shallow local basin, slightly off-center so there is
    # a non-zero gradient signal (at the exact well center ∇L ≈ 0, which is a
    # degenerate starting point for both methods).
    theta0 = np.array([0.0, 3.0])

    n_iters = 100
    outer_lr = 0.1
    inner_lr = 0.1

    # --- figure 1: single K, with inner rollouts shown ---------------------
    K = 5
    gd_path = run_gd(theta0, lr=outer_lr, n_iters=n_iters)
    irpo_path, irpo_rollouts = run_irpo(
        theta0,
        outer_lr=outer_lr,
        inner_lr=inner_lr,
        K=K,
        n_iters=n_iters,
    )
    print(f"final L — plain GD       : {L(torch.tensor(gd_path[-1])).item():+.4f}")
    print(f"final L — IRPO (K={K})    : {L(torch.tensor(irpo_path[-1])).item():+.4f}")
    plot(gd_path, irpo_path, irpo_rollouts, "visuals/irpo_escape_toy.png")

    # --- figure 2: sweep over K --------------------------------------------
    # K=2 is the minimum IRPO allows (see policy/irpo.py: assert num_exp_updates >= 2).
    Ks = [5, 10]  # , 15, 20]
    irpo_results: List[Tuple[int, np.ndarray, List[List[np.ndarray]]]] = []
    for K in Ks:
        # Re-seed for each K so the starting conditions are identical; the only
        # thing that changes is the number of inner steps.
        torch.manual_seed(0)
        np.random.seed(0)
        path, rollouts = run_irpo(
            theta0,
            outer_lr=outer_lr,
            inner_lr=inner_lr,
            K=K,
            n_iters=n_iters,
        )
        final = L(torch.tensor(path[-1])).item()
        print(f"final L — IRPO (K={K:>2}) : {final:+.4f}")
        irpo_results.append((K, path, rollouts))

    plot_K_sweep(gd_path, irpo_results, "visuals/irpo_escape_K_sweep.png")


if __name__ == "__main__":
    main()
