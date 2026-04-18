import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


# --- 1. Core Classes ---
class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count

        self.mean = new_mean
        self.var = M2 / tot_count
        self.count = tot_count


class NeuralNet(nn.Module):
    def __init__(self, state_dim, feature_dim, encoder_fc_dim, activation):
        super(NeuralNet, self).__init__()
        layers = []
        in_dim = state_dim
        for out_dim in encoder_fc_dim:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(activation)
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, feature_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x), None


# --- 2. Shared Configuration --- #
FEATURE_DIM = 16
INDICES_TO_PLOT = [0, 1, 2, 3]  # Plotted as columns
DELTA = 0.1

# Expanded Activation Dictionary
activations = {
    "ReLU": nn.ReLU(),
    "LeakyReLU": nn.LeakyReLU(),
    "ELU": nn.ELU(),
    "GELU": nn.GELU(),
    "Tanh": nn.Tanh(),
}


def get_model(state_dim, activation):
    """Initializes model with a fixed seed so weights are identical across ablations."""
    torch.manual_seed(42)
    return NeuralNet(state_dim, FEATURE_DIM, [128, 128], activation)


# ==========================================
# 3. 1D Ablation Study (X Input)
# ==========================================
def run_1d_ablation():
    STATE_DIM_1D = 1
    num_points = 100
    x_positions = torch.linspace(-5, 5, num_points).unsqueeze(1)
    next_x_positions = x_positions + DELTA

    # Dynamically adjust height based on number of activations
    fig_height = 2.5 * len(activations)
    fig, axes = plt.subplots(
        len(activations),
        len(INDICES_TO_PLOT),
        figsize=(16, fig_height),
        sharex=True,
        sharey="row",
    )
    fig.suptitle(
        "1D Ablation: Activation Functions vs. Intrinsic Rewards (X-axis only)\nBlue = (+), Red Dashed = (-)",
        fontsize=16,
    )

    for row_idx, (act_name, act_fn) in enumerate(activations.items()):
        extractor = get_model(STATE_DIM_1D, act_fn).eval()
        rms = RunningMeanStd(shape=(len(INDICES_TO_PLOT),))

        with torch.no_grad():
            feat, _ = extractor(x_positions)
            next_feat, _ = extractor(next_x_positions)
            raw_diff = (next_feat - feat)[:, INDICES_TO_PLOT]

            rms.update(raw_diff.numpy())
            var_tensor = torch.as_tensor(rms.var, dtype=raw_diff.dtype)
            norm_diff = raw_diff / (torch.sqrt(var_tensor) + 1e-8)

        for col_idx, feat_idx in enumerate(INDICES_TO_PLOT):
            ax = axes[row_idx, col_idx]
            curve = norm_diff[:, col_idx].numpy()

            ax.plot(x_positions.squeeze().numpy(), curve, color="royalblue", lw=2)
            ax.plot(
                x_positions.squeeze().numpy(),
                -curve,
                color="firebrick",
                ls="--",
                lw=1.5,
            )

            ax.grid(True, ls="--", alpha=0.5)
            if row_idx == 0:
                ax.set_title(f"Feature {feat_idx}")
            if col_idx == 0:
                ax.set_ylabel(f"{act_name}\nNorm. Reward")
            if row_idx == len(activations) - 1:
                ax.set_xlabel("X Position")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# ==========================================
# 4. 2D Ablation Study (X, Y Input Heatmaps)
# ==========================================
def run_2d_ablation():
    STATE_DIM_2D = 2
    grid_res = 50
    x_vals = torch.linspace(-5, 5, grid_res)
    y_vals = torch.linspace(-5, 5, grid_res)

    X, Y = torch.meshgrid(x_vals, y_vals, indexing="ij")
    states_2d = torch.stack([X.flatten(), Y.flatten()], dim=1)
    next_states_2d = states_2d + DELTA

    fig_height = 3.0 * len(activations)
    fig, axes = plt.subplots(
        len(activations),
        len(INDICES_TO_PLOT),
        figsize=(16, fig_height),
        sharex=True,
        sharey=True,
    )
    fig.suptitle(
        "2D Ablation: Activation Functions vs. Intrinsic Rewards (X,Y Plane)\nHeatmaps show the (+) direction reward gradient",
        fontsize=16,
    )

    for row_idx, (act_name, act_fn) in enumerate(activations.items()):
        extractor = get_model(STATE_DIM_2D, act_fn).eval()
        rms = RunningMeanStd(shape=(len(INDICES_TO_PLOT),))

        with torch.no_grad():
            feat, _ = extractor(states_2d)
            next_feat, _ = extractor(next_states_2d)
            raw_diff = (next_feat - feat)[:, INDICES_TO_PLOT]

            rms.update(raw_diff.numpy())
            var_tensor = torch.as_tensor(rms.var, dtype=raw_diff.dtype)
            norm_diff = raw_diff / (torch.sqrt(var_tensor) + 1e-8)

        for col_idx, feat_idx in enumerate(INDICES_TO_PLOT):
            ax = axes[row_idx, col_idx]
            reward_grid = norm_diff[:, col_idx].reshape(grid_res, grid_res).numpy()

            contour = ax.contourf(
                X.numpy(), Y.numpy(), reward_grid, levels=20, cmap="viridis"
            )

            if row_idx == 0:
                ax.set_title(f"Feature {feat_idx}")
            if col_idx == 0:
                ax.set_ylabel(f"{act_name}\nY Position")
            if row_idx == len(activations) - 1:
                ax.set_xlabel("X Position")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(contour, cax=cbar_ax, label="Normalized Intrinsic Reward")

    plt.subplots_adjust(right=0.9, top=0.92, bottom=0.08, wspace=0.2, hspace=0.3)
    plt.show()


# --- 5. Execute ---
if __name__ == "__main__":
    run_1d_ablation()
    run_2d_ablation()
