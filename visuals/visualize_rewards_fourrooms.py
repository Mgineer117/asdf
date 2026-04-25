import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn

from policy.layers.building_blocks import MLP


GRID = [
    "#############",
    "#    #      #",
    "# c  #      #",
    "#        c  #",
    "#    #      #",
    "#    #      #",
    "## ###### ###",
    "#     #     #",
    "#     # c   #",
    "#  c  #     #",
    "#           #",
    "#     #     #",
    "#############",
]

# ---------------------------------------------------------------------------
# Kernel catalogue
# (label, fn(sq_dist, feat_grid, phi_g, sigma2) -> (H,W) array in [0,1])
# ---------------------------------------------------------------------------


def _rbf(sq_dist, sigma2, scale, **_):
    return np.exp(-sq_dist / (2.0 * sigma2 * scale))


def _laplacian(sq_dist, sigma2, scale, **_):
    dist = np.sqrt(np.maximum(sq_dist, 0.0))
    return np.exp(-dist / (np.sqrt(sigma2) * scale))


def _cauchy(sq_dist, sigma2, scale, **_):
    return 1.0 / (1.0 + sq_dist / (sigma2 * scale))


def _cosine(feat_grid, phi_g, **_):
    dot = np.sum(feat_grid * phi_g[:, None, None], axis=0)  # (H, W)
    norm_s = np.sqrt(np.sum(feat_grid**2, axis=0)) + 1e-8  # (H, W)
    norm_g = np.linalg.norm(phi_g) + 1e-8
    cosine = dot / (norm_s * norm_g)
    return np.clip((cosine + 1.0) / 2.0, 0.0, 1.0)  # shift → [0,1]


KERNEL_SPECS = [
    # (row label,         fn,           scale)
    (r"RBF  $\sigma/2$", _rbf, 0.5),
    (r"RBF  $\sigma$", _rbf, 1.0),
    (r"RBF  $2\sigma$", _rbf, 2.0),
    (r"Laplacian $\sigma$", _laplacian, 1.0),
    (r"Cauchy $\sigma$", _cauchy, 1.0),
    (r"Cosine", _cosine, 1.0),  # σ unused
]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Model:

    def __init__(self):
        self.num_options = 4
        self.eigenvectors = [
            n // 2 + 1 for n in range(self.num_options)
        ]  # [1, 1, 2, 2]
        self.sings = [
            2 * (n % 2) - 1 for n in range(self.num_options)
        ]  # [-1, 1, -1, 1]

        model_path = "model/fourrooms-v1/ALLO_410_50000_0.9.pth"
        self.extractor = MLP(
            input_dim=2,
            hidden_dims=[512, 512, 512, 512],
            output_dim=10,
            activation=nn.LeakyReLU(),
        )

        state_dict = torch.load(model_path, map_location="cpu")
        new_state_dict = {
            k.replace("network.encoder.", "").replace("network.", ""): v
            for k, v in state_dict.items()
        }
        self.extractor.load_state_dict(new_state_dict, strict=False)
        self.extractor.eval()

    def get_features(self, states: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.extractor(torch.from_numpy(states)).numpy()[
                :, : self.num_options
            ]

    def forward(self, states: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            feature = self.extractor(torch.from_numpy(states))
            signs = torch.tensor(self.sings, dtype=torch.float32)
            return feature[:, self.eigenvectors] * signs  # (B, 4)


class RandomModel(Model):
    """Same architecture, random weights — baseline with no learned geometry."""

    def __init__(self):
        self.num_options = 4
        self.eigenvectors = [n // 2 + 1 for n in range(self.num_options)]
        self.sings = [2 * (n % 2) - 1 for n in range(self.num_options)]
        self.extractor = MLP(
            input_dim=2,
            hidden_dims=[512, 512, 512, 512],
            output_dim=10,
            activation=nn.LeakyReLU(),
        )
        self.extractor.eval()


# ---------------------------------------------------------------------------
# Grid utilities
# ---------------------------------------------------------------------------


def parse_grid(grid):
    walls, goals, walkable = set(), [], []
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if cell == "#":
                walls.add((x, y))
            elif cell == "c":
                goals.append((x, y))
                walkable.append((x, y))
            else:
                walkable.append((x, y))
    return walls, goals, walkable


def build_reward_grids(model, walkable, grid_h, grid_w):
    """4 signed eigenvector maps normalised to [-1, 1]."""
    n = 4
    s = np.array(walkable, dtype=np.float32)
    r = model.forward(s).numpy()  # (N, 4)

    grids = np.full((n, grid_h, grid_w), np.nan)
    for i in range(n):
        m = np.full((grid_h, grid_w), np.nan)
        for idx, (x, y) in enumerate(walkable):
            m[y, x] = r[idx, i]
        for mask, offset in [(m > 0, 0.0), (m < 0, -1.0)]:
            if np.any(mask):
                lo, hi = np.nanmin(m[mask]), np.nanmax(m[mask])
                if hi > lo:
                    m[mask] = (m[mask] - lo) / (hi - lo + 1e-8) + offset
        grids[i] = m
    return grids


def build_feature_grid(model, walkable, grid_h, grid_w):
    """10-dim encoder features on every walkable cell."""
    s = np.array(walkable, dtype=np.float32)
    phi = model.get_features(s)  # (N, D)
    grid = np.full((phi.shape[1], grid_h, grid_w), np.nan)
    for idx, (x, y) in enumerate(walkable):
        grid[:, y, x] = phi[idx]
    return grid, phi


def estimate_sigma2(phi_walkable, n_sample=300):
    """Median pairwise squared distance in feature space."""
    idx = np.random.choice(
        len(phi_walkable), min(len(phi_walkable), n_sample), replace=False
    )
    sub = phi_walkable[idx]
    sq = np.sum((sub[:, None] - sub[None, :]) ** 2, axis=-1)
    valid_sq = sq[sq > 1e-6]
    if len(valid_sq) == 0:
        return 1.0  # Fallback to prevent NaN if features collapsed
    return float(np.median(valid_sq))


def goal_reward_map(kernel_fn, scale, feat_grid, phi_goals, sigma2):
    """
    Returns (n_goals, H, W) array using the given kernel.
    NaN propagates naturally from wall cells in feat_grid.
    """
    n_goals = len(phi_goals)
    _, H, W = feat_grid.shape
    out = np.full((n_goals, H, W), np.nan)
    wall_mask = np.isnan(feat_grid[0])
    for i, phi_g in enumerate(phi_goals):
        diff = feat_grid - phi_g[:, None, None]  # (D, H, W)
        sq_dist = np.sum(diff**2, axis=0)  # (H, W)
        res = kernel_fn(
            sq_dist=sq_dist,
            feat_grid=feat_grid,
            phi_g=phi_g,
            sigma2=sigma2,
            scale=scale,
        )
        res[wall_mask] = np.nan  # Force walls to NaN so they plot as gray
        out[i] = res
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_all(
    reward_grids_allo,
    reward_grids_rand,
    feat_grid_allo,
    phi_allo,
    feat_grid_rand,
    phi_rand,
    goals,
    model_allo,
    model_rand,
):
    n_goals = len(goals)
    n_kernels = len(KERNEL_SPECS)
    n_cols = n_goals  # one column per goal
    # rows: 1 (raw ALLO) + n_kernels (ALLO kernels) + 1 (raw random) + n_kernels (random kernels)
    n_rows = 2 + n_kernels + 2 + n_kernels
    height_ratios = [0.15, 1.0] + [1.0] * n_kernels + [0.15, 1.0] + [1.0] * n_kernels

    goal_states = np.array(goals, dtype=np.float32)
    phi_goals_allo = model_allo.get_features(goal_states)  # (G, D)
    phi_goals_rand = model_rand.get_features(goal_states)

    sigma2_allo = estimate_sigma2(phi_allo)
    sigma2_rand = estimate_sigma2(phi_rand)

    cmap_div = plt.get_cmap("RdBu_r").copy()
    cmap_div.set_bad("gray")
    cmap_seq = plt.get_cmap("viridis").copy()
    cmap_seq.set_bad("gray")

    cell_w, cell_h = 3.2, 3.0
    fig = plt.figure(figsize=(cell_w * n_cols, cell_h * (n_rows - 2 * 0.85)))
    gs = gridspec.GridSpec(
        n_rows,
        n_cols,
        figure=fig,
        hspace=0.55,
        wspace=0.25,
        height_ratios=height_ratios,
    )

    def _plot_map(ax, data, cmap, vmin, vmax, title, star_xy=None, ylabel=None):
        im = ax.imshow(
            data, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal"
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize=7.5, pad=3)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=8, fontweight="bold")
        if star_xy is not None:
            ax.plot(*star_xy, "r*", markersize=9, zorder=5)
        plt.colorbar(im, ax=ax, shrink=0.72, pad=0.02)

    def _section_label(row, text, color):
        """Invisible axis spanning the full row used as a section header."""
        ax = fig.add_subplot(gs[row, :])
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            text,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=color,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", fc=color, alpha=0.15),
        )

    reward_labels = [
        f"eigvec={ev}, sign={s:+d}"
        for ev, s in zip(
            [n // 2 + 1 for n in range(4)], [2 * (n % 2) - 1 for n in range(4)]
        )
    ]

    # ── Row 0: raw ALLO intrinsic rewards ──────────────────────────────────
    _section_label(0, "ALLO — Raw Intrinsic Rewards", "#1a6faf")
    raw_allo_row = 1
    for g in range(n_goals):
        ax = fig.add_subplot(gs[raw_allo_row, g])
        if g < len(reward_labels):
            d = reward_grids_allo[g]
            vmax = np.nanmax(np.abs(d))
            if vmax == 0 or np.isnan(vmax):
                vmax = 1.0
            _plot_map(ax, d, cmap_div, -vmax, vmax, reward_labels[g])
            for gx, gy in goals:
                ax.plot(gx, gy, "k*", markersize=7)
        else:
            ax.set_axis_off()

    # ── Rows 1…n_kernels: ALLO features × each kernel ──────────────────────
    for ki, (label, fn, scale) in enumerate(KERNEL_SPECS):
        row = raw_allo_row + 1 + ki
        maps = goal_reward_map(fn, scale, feat_grid_allo, phi_goals_allo, sigma2_allo)
        for g in range(n_goals):
            ax = fig.add_subplot(gs[row, g])
            gx, gy = goals[g]
            title = f"Goal ({gx},{gy})" if ki == 0 else f"({gx},{gy})"
            full_title = f"{label}\n{title}" if g == 0 else title
            ylabel = label if g == 0 else None
            _plot_map(
                ax, maps[g], cmap_seq, 0, 1, full_title, star_xy=(gx, gy), ylabel=ylabel
            )

    # ── Row n_kernels+1: raw Random intrinsic rewards ──────────────────────
    header_rand_row = raw_allo_row + 1 + n_kernels
    _section_label(
        header_rand_row, "Random (uninformed) — Raw Intrinsic Rewards", "#a03010"
    )
    raw_rand_row = header_rand_row + 1
    for g in range(n_goals):
        ax = fig.add_subplot(gs[raw_rand_row, g])
        if g < len(reward_labels):
            d = reward_grids_rand[g]
            vmax = np.nanmax(np.abs(d))
            if vmax == 0 or np.isnan(vmax):
                vmax = 1.0
            _plot_map(ax, d, cmap_div, -vmax, vmax, reward_labels[g])
            for gx, gy in goals:
                ax.plot(gx, gy, "k*", markersize=7)
        else:
            ax.set_axis_off()

    # ── Rows raw_row+1…: Random features × each kernel ─────────────────────
    for ki, (label, fn, scale) in enumerate(KERNEL_SPECS):
        row = raw_rand_row + 1 + ki
        maps = goal_reward_map(fn, scale, feat_grid_rand, phi_goals_rand, sigma2_rand)
        for g in range(n_goals):
            ax = fig.add_subplot(gs[row, g])
            gx, gy = goals[g]
            title = f"Goal ({gx},{gy})" if ki == 0 else f"({gx},{gy})"
            full_title = f"{label}\n{title}" if g == 0 else title
            ylabel = label if g == 0 else None
            _plot_map(
                ax, maps[g], cmap_seq, 0, 1, full_title, star_xy=(gx, gy), ylabel=ylabel
            )

    fig.suptitle(
        "Goal-Conditioned Reward: ALLO (learned geometry) vs Random features\n"
        r"Kernels vary across rows; $\sigma^2$ = median pairwise dist in $\varphi$-space",
        fontsize=11,
        y=1.002,
    )
    plt.savefig("reward_visualization.svg", dpi=150, bbox_inches="tight")
    print("Saved reward_visualization.png")
    # plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    np.random.seed(0)

    grid_h = len(GRID)
    grid_w = len(GRID[0])

    walls, goals, walkable = parse_grid(GRID)
    print(f"Goals (c): {goals}")

    model_allo = Model()
    model_rand = RandomModel()

    reward_grids_allo = build_reward_grids(model_allo, walkable, grid_h, grid_w)
    reward_grids_rand = build_reward_grids(model_rand, walkable, grid_h, grid_w)

    feat_grid_allo, phi_allo = build_feature_grid(model_allo, walkable, grid_h, grid_w)
    feat_grid_rand, phi_rand = build_feature_grid(model_rand, walkable, grid_h, grid_w)

    plot_all(
        reward_grids_allo,
        reward_grids_rand,
        feat_grid_allo,
        phi_allo,
        feat_grid_rand,
        phi_rand,
        goals,
        model_allo,
        model_rand,
    )
