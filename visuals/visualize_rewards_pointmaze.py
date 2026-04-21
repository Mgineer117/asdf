import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn

from policy.layers.building_blocks import MLP


GRID = [
    [1, 1, 1, 1, 1, 1],
    [1, "c", 1, "c", 0, 1],
    [1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1],
]

CELL_SIZE = 1.0
RESOLUTION = 30  # sample points per cell (increase for higher fidelity)

# ---------------------------------------------------------------------------
# Kernel catalogue
# ---------------------------------------------------------------------------


def _rbf(sq_dist, sigma2, scale, **_):
    return np.exp(-sq_dist / (2.0 * sigma2 * scale))


def _laplacian(sq_dist, sigma2, scale, **_):
    dist = np.sqrt(np.maximum(sq_dist, 0.0))
    return np.exp(-dist / (np.sqrt(sigma2) * scale))


def _cauchy(sq_dist, sigma2, scale, **_):
    return 1.0 / (1.0 + sq_dist / (sigma2 * scale))


def _cosine(feat_grid, phi_g, **_):
    dot = np.sum(feat_grid * phi_g[:, None, None], axis=0)
    norm_s = np.sqrt(np.sum(feat_grid**2, axis=0)) + 1e-8
    norm_g = np.linalg.norm(phi_g) + 1e-8
    cosine = dot / (norm_s * norm_g)
    return np.clip((cosine + 1.0) / 2.0, 0.0, 1.0)


KERNEL_SPECS = [
    (r"RBF  $\sigma/2$", _rbf, 0.5),
    (r"RBF  $\sigma$", _rbf, 1.0),
    (r"RBF  $2\sigma$", _rbf, 2.0),
    (r"Laplacian $\sigma$", _laplacian, 1.0),
    (r"Cauchy $\sigma$", _cauchy, 1.0),
    (r"Cosine", _cosine, 1.0),
]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Model:

    def __init__(self):
        self.num_options = 4
        self.eigenvectors = [n // 2 + 1 for n in range(self.num_options)]
        self.sings = [2 * (n % 2) - 1 for n in range(self.num_options)]

        model_path = "model/pointmaze-v1/ALLO_410_50000_0.99.pth"
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
            # feature = feature[:, : self.num_options]
            signs = torch.tensor(self.sings, dtype=torch.float32)
            return feature[:, self.eigenvectors] * signs


class RandomModel(Model):
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
# Continuous grid utilities
# ---------------------------------------------------------------------------


def parse_grid(grid, cell_size=CELL_SIZE):
    """Return goal positions in continuous env coordinates (1=wall, c=goal)."""
    maze_h, maze_w = len(grid), len(grid[0])
    goals_cont = []
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell == "c":
                x = (j + 0.5) * cell_size - maze_w * cell_size / 2
                y = maze_h * cell_size / 2 - (i + 0.5) * cell_size
                goals_cont.append((x, y))
    return goals_cont


def _build_mesh(grid, resolution=RESOLUTION, cell_size=CELL_SIZE):
    """
    Build a continuous coordinate mesh and boolean wall mask.
    Returns: XX, YY (H,W), wall_mask (H,W), extent [x_min,x_max,y_min,y_max]
    """
    maze_h, maze_w = len(grid), len(grid[0])
    x_min = -maze_w * cell_size / 2
    x_max = maze_w * cell_size / 2
    y_min = -maze_h * cell_size / 2
    y_max = maze_h * cell_size / 2

    W = maze_w * resolution
    H = maze_h * resolution
    half_px = cell_size / (2 * resolution)

    xs = np.linspace(x_min, x_max, W, endpoint=False) + half_px
    ys = np.linspace(y_max, y_min, H, endpoint=False) - half_px

    XX, YY = np.meshgrid(xs, ys)  # (H, W), rows = y descending

    # Convert each sample point to grid index and look up wall flag
    J = np.clip(((XX - x_min) / cell_size).astype(int), 0, maze_w - 1)
    I = np.clip(((y_max - YY) / cell_size).astype(int), 0, maze_h - 1)
    grid_int = np.array([[1 if cell == 1 else 0 for cell in row] for row in grid])
    wall_mask = grid_int[I, J].astype(bool)

    extent = [x_min, x_max, y_min, y_max]
    return XX, YY, wall_mask, extent


def build_feature_grid(model, grid, resolution=RESOLUTION, cell_size=CELL_SIZE):
    """Encode every non-wall sample in continuous space → (D, H, W) feature grid."""
    XX, YY, wall_mask, extent = _build_mesh(grid, resolution, cell_size)
    H, W = XX.shape

    non_wall = ~wall_mask.ravel()
    states = np.stack([XX.ravel()[non_wall], YY.ravel()[non_wall]], axis=1).astype(
        np.float32
    )

    phi = model.get_features(states)  # (N_free, D)
    D = phi.shape[1]

    feat_grid = np.full((D, H, W), np.nan)
    rows, cols = np.where(~wall_mask)
    feat_grid[:, rows, cols] = phi.T

    return feat_grid, phi, extent


def build_reward_grids(model, grid, resolution=RESOLUTION, cell_size=CELL_SIZE):
    """4 signed eigenvector maps over continuous space, normalised to [-1,1]."""
    XX, YY, wall_mask, extent = _build_mesh(grid, resolution, cell_size)
    H, W = XX.shape
    n = 4

    non_wall = ~wall_mask.ravel()
    states = np.stack([XX.ravel()[non_wall], YY.ravel()[non_wall]], axis=1).astype(
        np.float32
    )

    r = model.forward(states).numpy()  # (N_free, 4)

    grids = np.full((n, H, W), np.nan)
    rows, cols = np.where(~wall_mask)
    for i in range(n):
        m = np.full((H, W), np.nan)
        m[rows, cols] = r[:, i]
        for mask, offset in [(m > 0, 0.0), (m < 0, -1.0)]:
            if np.any(mask):
                lo, hi = np.nanmin(m[mask]), np.nanmax(m[mask])
                if hi > lo:
                    m[mask] = (m[mask] - lo) / (hi - lo + 1e-8) + offset
        grids[i] = m

    return grids, extent


def estimate_sigma2(phi_walkable, n_sample=300):
    """Median pairwise squared distance in feature space."""
    idx = np.random.choice(
        len(phi_walkable), min(len(phi_walkable), n_sample), replace=False
    )
    sub = phi_walkable[idx]
    sq = np.sum((sub[:, None] - sub[None, :]) ** 2, axis=-1)
    valid_sq = sq[sq > 1e-6]
    return float(np.median(valid_sq)) if len(valid_sq) > 0 else 1.0


def goal_reward_map(kernel_fn, scale, feat_grid, phi_goals, sigma2):
    """
    (n_goals, H, W) kernel reward maps in continuous space.
    NaN propagates naturally from wall cells.
    """
    n_goals = len(phi_goals)
    _, H, W = feat_grid.shape
    out = np.full((n_goals, H, W), np.nan)
    wall_mask = np.isnan(feat_grid[0])
    for i, phi_g in enumerate(phi_goals):
        diff = feat_grid - phi_g[:, None, None]
        sq_dist = np.sum(diff**2, axis=0)
        res = kernel_fn(
            sq_dist=sq_dist,
            feat_grid=feat_grid,
            phi_g=phi_g,
            sigma2=sigma2,
            scale=scale,
        )
        res[wall_mask] = np.nan
        out[i] = res
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_all(
    reward_grids_allo,
    extent_allo,
    reward_grids_rand,
    extent_rand,
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
    n_cols = n_goals
    n_rows = 2 + n_kernels + 2 + n_kernels
    height_ratios = [0.15, 1.0] + [1.0] * n_kernels + [0.15, 1.0] + [1.0] * n_kernels

    goal_states = np.array(goals, dtype=np.float32)  # (G, 2) continuous coords
    phi_goals_allo = model_allo.get_features(goal_states)
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

    def _plot_map(ax, data, extent, cmap, vmin, vmax, title, star_xy=None, ylabel=None):
        im = ax.imshow(
            data,
            origin="upper",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
            interpolation="bilinear",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize=7.5, pad=3)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=8, fontweight="bold")
        if star_xy is not None:
            # star_xy is a continuous (x, y) coord — plots correctly with extent
            ax.plot(star_xy[0], star_xy[1], "r*", markersize=9, zorder=5)
        plt.colorbar(im, ax=ax, shrink=0.72, pad=0.02)

    def _section_label(row, text, color):
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

    # ── Row 0: raw ALLO intrinsic rewards ─────────────────────────────────────
    _section_label(0, "ALLO — Raw Intrinsic Rewards", "#1a6faf")
    raw_allo_row = 1
    for g in range(n_goals):
        ax = fig.add_subplot(gs[raw_allo_row, g])
        if g < len(reward_labels):
            d = reward_grids_allo[g]
            vmax = np.nanmax(np.abs(d))
            vmax = 1.0 if (vmax == 0 or np.isnan(vmax)) else vmax
            _plot_map(ax, d, extent_allo, cmap_div, -vmax, vmax, reward_labels[g])
            for gx, gy in goals:
                ax.plot(gx, gy, "k*", markersize=7)
        else:
            ax.set_axis_off()

    # ── Rows 1…n_kernels: ALLO features × each kernel ─────────────────────────
    for ki, (label, fn, scale) in enumerate(KERNEL_SPECS):
        row = raw_allo_row + 1 + ki
        maps = goal_reward_map(fn, scale, feat_grid_allo, phi_goals_allo, sigma2_allo)
        for g in range(n_goals):
            ax = fig.add_subplot(gs[row, g])
            gx, gy = goals[g]
            title = f"Goal ({gx:.1f},{gy:.1f})" if ki == 0 else f"({gx:.1f},{gy:.1f})"
            full_title = f"{label}\n{title}" if g == 0 else title
            ylabel = label if g == 0 else None
            _plot_map(
                ax,
                maps[g],
                extent_allo,
                cmap_seq,
                0,
                1,
                full_title,
                star_xy=(gx, gy),
                ylabel=ylabel,
            )

    # ── Row n_kernels+1: raw Random intrinsic rewards ─────────────────────────
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
            vmax = 1.0 if (vmax == 0 or np.isnan(vmax)) else vmax
            _plot_map(ax, d, extent_rand, cmap_div, -vmax, vmax, reward_labels[g])
            for gx, gy in goals:
                ax.plot(gx, gy, "k*", markersize=7)
        else:
            ax.set_axis_off()

    # ── Rows raw_row+1…: Random features × each kernel ────────────────────────
    for ki, (label, fn, scale) in enumerate(KERNEL_SPECS):
        row = raw_rand_row + 1 + ki
        maps = goal_reward_map(fn, scale, feat_grid_rand, phi_goals_rand, sigma2_rand)
        for g in range(n_goals):
            ax = fig.add_subplot(gs[row, g])
            gx, gy = goals[g]
            title = f"Goal ({gx:.1f},{gy:.1f})" if ki == 0 else f"({gx:.1f},{gy:.1f})"
            full_title = f"{label}\n{title}" if g == 0 else title
            ylabel = label if g == 0 else None
            _plot_map(
                ax,
                maps[g],
                extent_rand,
                cmap_seq,
                0,
                1,
                full_title,
                star_xy=(gx, gy),
                ylabel=ylabel,
            )

    fig.suptitle(
        "Goal-Conditioned Reward: ALLO (learned geometry) vs Random features\n"
        r"Kernels vary across rows; $\sigma^2$ = median pairwise dist in $\varphi$-space",
        fontsize=11,
        y=1.002,
    )
    plt.savefig("reward_visualization_pointmaze.svg", dpi=150, bbox_inches="tight")
    print("Saved reward_visualization.svg")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    np.random.seed(0)

    goals = parse_grid(GRID)
    print(f"Goals (c) in continuous coords: {goals}")

    model_allo = Model()
    model_rand = RandomModel()

    reward_grids_allo, extent_allo = build_reward_grids(model_allo, GRID)
    reward_grids_rand, extent_rand = build_reward_grids(model_rand, GRID)

    feat_grid_allo, phi_allo, _ = build_feature_grid(model_allo, GRID)
    feat_grid_rand, phi_rand, _ = build_feature_grid(model_rand, GRID)

    plot_all(
        reward_grids_allo,
        extent_allo,
        reward_grids_rand,
        extent_rand,
        feat_grid_allo,
        phi_allo,
        feat_grid_rand,
        phi_rand,
        goals,
        model_allo,
        model_rand,
    )
