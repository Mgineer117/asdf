"""Toy 2D demo: IRPO vs gradient descent vs BFGS, two scenarios.

Both scenarios share the same MAML-style IRPO update:
    exp = base − lr_in · ∇f_dense(...)  ×K   # inner loop, autograd graph kept
    base ← base − lr_out · ∂f_sparse(exp) / ∂base   # meta-gradient
    primary = exp                            # deployed parameter

Scenario A — convex intrinsic, intrinsic optimum coincides with extrinsic:
    f_sparse: narrow Gaussian basin at p* = (3, 3) — sparse signal
    f_dense:  wide Gaussian basin also at p*       — abundant signal
    Story: IRPO works well; GD / BFGS stall in flat regions.

Scenario B — multimodal sparse, dense centered at a *local* optimum:
    f_sparse: sum of narrow Gaussian wells. Global min at (3, 3); two
              shallower local mins elsewhere.
    f_dense:  wide Gaussian centered at one of the local mins, NOT the
              global. This is a misleading intrinsic reward.
    Story: GD / BFGS still stall, but IRPO is misled — its primary
    parameter rides the dense gradient into the wrong basin.

Run:
    python visuals/irpo_sparse_vs_dense.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize
from torch.autograd import grad as tgrad


# -------- scenario types --------------------------------------------------


@dataclass
class Scenario:
    name: str
    title: str
    f_sparse: Callable[[np.ndarray], np.ndarray]
    grad_sparse: Callable[[np.ndarray], np.ndarray]
    f_dense: Callable[[np.ndarray], np.ndarray]
    f_sparse_t: Callable[[torch.Tensor], torch.Tensor]
    f_dense_t: Callable[[torch.Tensor], torch.Tensor]
    p_global: np.ndarray
    extra_marks: (
        list  # list of (name, point) for non-global features (e.g., dense center)
    )


# -------- scenario A: convex intrinsic ------------------------------------

P_STAR_A = np.array([3.0, 3.0])
SIGMA2_SPARSE_A = 0.5
SIGMA2_DENSE_A = 8.0

P_STAR_A_T = torch.tensor(P_STAR_A, dtype=torch.float64)


def fA_sparse(p):
    p = np.asarray(p)
    d2 = np.sum((p - P_STAR_A) ** 2, axis=-1)
    return -np.exp(-d2 / SIGMA2_SPARSE_A)


def fA_grad_sparse(p):
    d = p - P_STAR_A
    return (2.0 / SIGMA2_SPARSE_A) * np.exp(-np.sum(d**2) / SIGMA2_SPARSE_A) * d


def fA_dense(p):
    p = np.asarray(p)
    d2 = np.sum((p - P_STAR_A) ** 2, axis=-1)
    return -np.exp(-d2 / SIGMA2_DENSE_A)


def fA_sparse_t(p):
    return -torch.exp(-((p - P_STAR_A_T) ** 2).sum() / SIGMA2_SPARSE_A)


def fA_dense_t(p):
    return -torch.exp(-((p - P_STAR_A_T) ** 2).sum() / SIGMA2_DENSE_A)


SCENARIO_A = Scenario(
    name="convex",
    title="convex intrinsic (dense optimum = sparse global)",
    f_sparse=fA_sparse,
    grad_sparse=fA_grad_sparse,
    f_dense=fA_dense,
    f_sparse_t=fA_sparse_t,
    f_dense_t=fA_dense_t,
    p_global=P_STAR_A,
    extra_marks=[],
)


# -------- scenario B: multimodal sparse, dense at a local optimum ---------

WELLS_B = [
    {"center": np.array([3.0, 3.0]), "amp": 1.00, "sigma2": 0.45},  # global
    {
        "center": np.array([-1.0, 2.0]),
        "amp": 0.55,
        "sigma2": 0.45,
    },  # local — dense centered here
    {"center": np.array([2.5, -1.5]), "amp": 0.60, "sigma2": 0.45},  # local
    {"center": np.array([-2.0, -2.0]), "amp": 0.50, "sigma2": 0.45},  # local
]
# Wide low-amplitude halos around each well — they overlap with neighbors
# and produce a small but nonzero curvature in the inter-well regions, so
# the landscape is gently connected rather than flat. Amplitude is small
# enough that the deep narrow wells still dominate near each center.
HALO_AMP = 0.04
HALO_SIGMA2 = 7.0
DENSE_B_CENTER = np.array([-1.0, 2.0])  # at one of the LOCAL minima
SIGMA2_DENSE_B = 6.0

DENSE_B_CENTER_T = torch.tensor(DENSE_B_CENTER, dtype=torch.float64)
WELLS_B_T = [
    {
        "center": torch.tensor(w["center"], dtype=torch.float64),
        "amp": w["amp"],
        "sigma2": w["sigma2"],
    }
    for w in WELLS_B
]
P_GLOBAL_B = WELLS_B[0]["center"]


def fB_sparse(p):
    p = np.asarray(p)
    val = np.zeros(p.shape[:-1])
    for w in WELLS_B:
        d2 = np.sum((p - w["center"]) ** 2, axis=-1)
        val = val - w["amp"] * np.exp(-d2 / w["sigma2"])
        val = val - HALO_AMP * np.exp(-d2 / HALO_SIGMA2)
    return val


def fB_grad_sparse(p):
    g = np.zeros_like(p)
    for w in WELLS_B:
        d = p - w["center"]
        d2 = np.sum(d**2)
        g = g + w["amp"] * (2.0 / w["sigma2"]) * np.exp(-d2 / w["sigma2"]) * d
        g = g + HALO_AMP * (2.0 / HALO_SIGMA2) * np.exp(-d2 / HALO_SIGMA2) * d
    return g


def fB_dense(p):
    p = np.asarray(p)
    d2 = np.sum((p - DENSE_B_CENTER) ** 2, axis=-1)
    return -np.exp(-d2 / SIGMA2_DENSE_B)


def fB_sparse_t(p):
    val = torch.zeros((), dtype=torch.float64)
    for w in WELLS_B_T:
        d2 = ((p - w["center"]) ** 2).sum()
        val = val - w["amp"] * torch.exp(-d2 / w["sigma2"])
        val = val - HALO_AMP * torch.exp(-d2 / HALO_SIGMA2)
    return val


def fB_dense_t(p):
    return -torch.exp(-((p - DENSE_B_CENTER_T) ** 2).sum() / SIGMA2_DENSE_B)


SCENARIO_B = Scenario(
    name="multimodal",
    title="multimodal sparse, dense at a LOCAL optimum",
    f_sparse=fB_sparse,
    grad_sparse=fB_grad_sparse,
    f_dense=fB_dense,
    f_sparse_t=fB_sparse_t,
    f_dense_t=fB_dense_t,
    p_global=P_GLOBAL_B,
    extra_marks=[("dense center (local min)", DENSE_B_CENTER)],
)


# -------- optimizers ------------------------------------------------------


def run_gd(p0, lr, steps, grad_fn):
    traj = [p0.copy()]
    p = p0.copy()
    for _ in range(steps):
        p = p - lr * grad_fn(p)
        traj.append(p.copy())
    return np.array(traj)


def run_bfgs(p0, max_iter, f_fn, grad_fn):
    traj = [p0.copy()]
    res = minimize(
        f_fn,
        p0,
        jac=grad_fn,
        method="BFGS",
        callback=lambda xk: traj.append(xk.copy()),
        options={"maxiter": max_iter, "gtol": 1e-12},
    )
    if not np.allclose(traj[-1], res.x):
        traj.append(res.x.copy())
    return np.array(traj)


@dataclass
class IRPOResult:
    base_traj: np.ndarray
    exp_traj: np.ndarray
    inner_paths: list
    meta_grad_norms: np.ndarray  # |∂L_true(exp)/∂base| each outer iter


def irpo_step(base, inner_lr, K, f_dense_t_fn, f_sparse_t_fn):
    base_v = base.detach().clone().requires_grad_(True)
    theta = base_v
    inner = [base_v.detach().numpy().copy()]
    for _ in range(K):
        L_int = f_dense_t_fn(theta)
        g = tgrad(L_int, theta, create_graph=True)[0]
        theta = theta - inner_lr * g
        inner.append(theta.detach().numpy().copy())
    L_true = f_sparse_t_fn(theta)
    base_grad = tgrad(L_true, base_v)[0]
    return theta.detach(), base_grad.detach(), inner


def run_irpo(p0, outer_steps, outer_lr, inner_lr, K, f_dense_t_fn, f_sparse_t_fn):
    base = torch.tensor(p0, dtype=torch.float64)
    base_traj = [base.numpy().copy()]
    exp_traj = [base.numpy().copy()]
    inner_paths = []
    meta_norms = []
    for _ in range(outer_steps):
        exp, g, inner = irpo_step(base, inner_lr, K, f_dense_t_fn, f_sparse_t_fn)
        inner_paths.append(np.array(inner))
        meta_norms.append(float(g.norm()))
        base = base - outer_lr * g
        base_traj.append(base.numpy().copy())
        exp_traj.append(exp.numpy().copy())
    return IRPOResult(
        np.array(base_traj),
        np.array(exp_traj),
        inner_paths,
        np.array(meta_norms),
    )


# -------- visualization ---------------------------------------------------

# Soft monochrome background lets the trajectory colors stay readable.
BG_CMAP = "Greys"
BG_ALPHA = 0.7
CONTOUR_LINE_ALPHA = 0.45


def _draw_landscape(
    ax, f_fn, p_global, extra_marks, xlim, ylim, n=240, cbar_label=None, fig=None
):
    xs = np.linspace(*xlim, n)
    ys = np.linspace(*ylim, n)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel()], axis=-1)
    Z = f_fn(pts).reshape(X.shape)
    cs = ax.contourf(X, Y, Z, levels=30, cmap=BG_CMAP, alpha=BG_ALPHA)
    ax.contour(X, Y, Z, levels=10, colors="k", alpha=CONTOUR_LINE_ALPHA, linewidths=0.5)
    ax.scatter(
        *p_global,
        marker="*",
        c="gold",
        edgecolors="k",
        s=260,
        zorder=12,
        label=r"$p^\star$ (global)",
    )
    for name, pt in extra_marks:
        ax.scatter(
            *pt, marker="P", c="white", edgecolors="black", s=140, zorder=12, label=name
        )
    if fig is not None and cbar_label is not None:
        fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    return cs


def _annotate_stalled(ax, p0, traj, label, color, offset):
    """If the trajectory barely moves, add an annotation pointing to the
    stalled marker so it's not invisible behind the start X. `offset`
    is added to p0 to position the annotation text."""
    final = traj[-1]
    if np.linalg.norm(final - p0) < 1e-6:
        ax.annotate(
            f"{label}\nstalled",
            xy=p0,
            xytext=(p0[0] + offset[0], p0[1] + offset[1]),
            fontsize=8,
            color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
            bbox=dict(
                boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.8, alpha=0.9
            ),
        )


def make_figure(
    scen: Scenario,
    p0: np.ndarray,
    save_dir: str,
    outer_steps=100,
    # outer_lr=1.0,
    # inner_lr=4.0,
    outer_lr=0.1,
    inner_lr=1.0,
    K=10,
    gd_lr=1.0,
    gd_steps=80,
    bfgs_iters=80,
    xlim=(-4, 6),
    ylim=(-4, 6),
):
    gd_traj = run_gd(p0, gd_lr, gd_steps, scen.grad_sparse)
    bfgs_traj = run_bfgs(p0, bfgs_iters, scen.f_sparse, scen.grad_sparse)
    irpo = run_irpo(
        p0, outer_steps, outer_lr, inner_lr, K, scen.f_dense_t, scen.f_sparse_t
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(f"Scenario: {scen.title}", fontsize=13, y=1.02)

    # ---- (1) sparse landscape with all trajectories ----
    ax = axes[0]
    _draw_landscape(
        ax,
        scen.f_sparse,
        scen.p_global,
        scen.extra_marks,
        xlim,
        ylim,
        cbar_label=r"$f_{\rm sparse}$",
        fig=fig,
    )
    ax.plot(
        *gd_traj.T,
        "-o",
        c="tab:red",
        ms=4,
        lw=1.5,
        mec="white",
        mew=0.5,
        label="GD on sparse",
        zorder=4,
    )
    ax.plot(
        *bfgs_traj.T,
        "-s",
        c="tab:orange",
        ms=5,
        lw=1.5,
        mec="white",
        mew=0.5,
        label="BFGS on sparse",
        zorder=5,
    )
    rollout_idx = [0, len(irpo.inner_paths) // 2, len(irpo.inner_paths) - 1]
    for j, t in enumerate(rollout_idx):
        path = irpo.inner_paths[t]
        ax.plot(
            *path.T,
            c="tab:cyan",
            lw=1.3,
            alpha=0.85,
            label="inner intrinsic rollout" if j == 0 else None,
            zorder=6,
        )
    ax.plot(
        *irpo.base_traj.T,
        "-^",
        c="black",
        mec="white",
        ms=8,
        lw=2.0,
        label="IRPO base (meta-init)",
        zorder=8,
    )
    ax.plot(
        *irpo.exp_traj.T,
        "-D",
        c="magenta",
        mec="black",
        ms=6,
        lw=1.8,
        label="IRPO primary = exp(base)",
        zorder=9,
    )
    ax.scatter(
        *p0,
        c="black",
        marker="X",
        s=110,
        edgecolors="white",
        linewidth=1.3,
        zorder=11,
        label="start",
    )
    _annotate_stalled(ax, p0, gd_traj, "GD", "tab:red", offset=(-1.6, -0.6))
    _annotate_stalled(ax, p0, bfgs_traj, "BFGS", "tab:orange", offset=(-1.7, -1.6))
    ax.set_title("Sparse landscape: deployed exp(base) vs moving base")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.92)

    # ---- (2) dense landscape with dense inner map only ----
    ax = axes[1]
    _draw_landscape(
        ax,
        scen.f_dense,
        scen.p_global,
        scen.extra_marks,
        xlim,
        ylim,
        cbar_label=r"$f_{\rm dense}$",
        fig=fig,
    )
    for j, t in enumerate(rollout_idx):
        path = irpo.inner_paths[t]
        base_t = irpo.base_traj[t]
        exp_t = irpo.exp_traj[t + 1]
        ax.plot(
            *path.T,
            c="tab:cyan",
            lw=1.3,
            alpha=0.85,
            label=(
                r"dense inner map: $\mathrm{base}_t \rightarrow \mathrm{exp}_t$"
                if j == 0
                else None
            ),
            zorder=6,
        )
        ax.scatter(
            *base_t,
            c="black",
            marker="o",
            s=44,
            edgecolors="white",
            linewidth=0.9,
            zorder=8,
            label=r"$\mathrm{base}_t$" if j == 0 else None,
        )
        ax.scatter(
            *exp_t,
            c="magenta",
            marker="D",
            s=48,
            edgecolors="black",
            linewidth=0.9,
            zorder=9,
            label=r"$\mathrm{exp}_t$" if j == 0 else None,
        )
        ax.text(
            base_t[0] + 0.08,
            base_t[1] + 0.08,
            f"t={t}",
            fontsize=8,
            color="black",
        )
    ax.scatter(
        *p0,
        c="black",
        marker="X",
        s=110,
        edgecolors="white",
        linewidth=1.3,
        zorder=11,
        label="start",
    )
    ax.set_title("Dense objective only determines base -> exp")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.92)

    # ---- (3) sparse-reward value vs iteration ----
    ax = axes[2]
    ax.axhline(
        scen.f_sparse(scen.p_global),
        color="gold",
        ls="--",
        lw=1.2,
        label=r"$f_{\rm sparse}(p^\star)$ (best possible)",
    )
    ax.plot(scen.f_sparse(gd_traj), c="tab:red", lw=2, label="GD on sparse")
    ax.plot(
        scen.f_sparse(bfgs_traj),
        c="tab:orange",
        lw=2,
        marker="s",
        ms=4,
        label="BFGS on sparse",
    )
    ax.plot(
        np.arange(len(irpo.base_traj)),
        scen.f_sparse(irpo.base_traj),
        c="black",
        lw=2,
        marker="^",
        label="IRPO base",
    )
    ax.plot(
        np.arange(len(irpo.exp_traj)),
        scen.f_sparse(irpo.exp_traj),
        c="magenta",
        lw=2,
        marker="D",
        label="IRPO primary (exp)",
    )
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$f_{\rm sparse}(p)$")
    ax.set_title("Sparse-reward value over iterations")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    plt.tight_layout()
    out_path = os.path.join(save_dir, f"irpo_sparse_vs_dense_{scen.name}.svg")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[{scen.name}] saved {out_path}")
    print(
        f"[{scen.name}] final sparse — "
        f"GD: {scen.f_sparse(gd_traj[-1]):.4f}, "
        f"BFGS: {scen.f_sparse(bfgs_traj[-1]):.4f}, "
        f"IRPO base: {scen.f_sparse(irpo.base_traj[-1]):.4f}, "
        f"IRPO primary: {scen.f_sparse(irpo.exp_traj[-1]):.4f}, "
        f"global: {scen.f_sparse(scen.p_global):.4f}"
    )
    print(
        f"[{scen.name}] meta-grad norms across outer iters (max={irpo.meta_grad_norms.max():.3e}, "
        f"min={irpo.meta_grad_norms.min():.3e}, mean={irpo.meta_grad_norms.mean():.3e})"
    )


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    p0 = np.array([0.0, 0.0])
    make_figure(
        SCENARIO_A,
        p0,
        out_dir,
        # outer_steps=20,
        # outer_lr=0.1,
        # inner_lr=1.0,
        # K=8,
        xlim=(-2, 6),
        ylim=(-2, 6),
    )
    make_figure(
        SCENARIO_B,
        p0,
        out_dir,
        # outer_steps=25,
        # outer_lr=0.1,
        # inner_lr=1.0,
        # K=10,
        # gd_lr=1.0,
        # gd_steps=120,
        xlim=(-4, 5),
        ylim=(-4, 5),
    )


if __name__ == "__main__":
    main()
