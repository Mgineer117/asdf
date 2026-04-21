import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA


class Base(nn.Module):
    def __init__(self, device):
        super(Base, self).__init__()

        self.dtype = torch.float32
        self.device = device

        # utils
        self.l1_loss = F.l1_loss
        self.mse_loss = F.mse_loss
        self.huber_loss = F.smooth_l1_loss

        self.state_visitation = None

    def setup_obs_rms(self, state_dim, pos_idx=None, goal_idx=None):
        """
        Call once in learner __init__ to enable per-dimension state normalisation.

        For goal-conditioned envs, pass pos_idx (achieved_goal indices) and
        goal_idx (desired_goal indices). After every RMS update, desired_goal
        dim stats are overwritten with achieved_goal dim stats, ensuring both
        are normalised by the same well-estimated (high-variance) statistics.
        This prevents the near-zero-variance problem that arises because the
        desired_goal is constant within an episode while achieved_goal varies.
        """
        from utils.wrapper import RunningMeanStd

        # Skip for image inputs (2-D or 3-D shape) — encoder already handles scale
        if isinstance(state_dim, (tuple, list)) and len(state_dim) >= 2:
            self.obs_rms = None
            self._rms_pos_idx = None
            self._rms_goal_idx = None
        else:
            total_dim = state_dim[0] if isinstance(state_dim, tuple) else state_dim
            self.obs_rms = RunningMeanStd(shape=(total_dim,))
            # Convert potentially-negative indices to positive for numpy indexing
            to_pos = lambda idxs: [i % total_dim for i in idxs] if idxs else None
            self._rms_pos_idx = to_pos(pos_idx)
            self._rms_goal_idx = to_pos(goal_idx)
            # Only sync when pos and goal are genuinely separate dimensions
            if (self._rms_pos_idx is not None and self._rms_goal_idx is not None
                    and self._rms_pos_idx == self._rms_goal_idx):
                self._rms_goal_idx = None  # same dims — no sync needed

    def _sync_goal_stats(self):
        """
        Copy achieved_goal dim statistics into desired_goal dim slots.
        Must be called after every obs_rms.update() in goal-conditioned learners.
        """
        if (self.obs_rms is None
                or self._rms_pos_idx is None
                or self._rms_goal_idx is None):
            return
        for p, g in zip(self._rms_pos_idx, self._rms_goal_idx):
            self.obs_rms.mean[g] = self.obs_rms.mean[p]
            self.obs_rms.var[g] = self.obs_rms.var[p]

    def _normalize_obs(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize x with running stats; no-op if obs_rms is None."""
        rms = getattr(self, "obs_rms", None)
        if rms is None:
            return x
        return rms.normalize(x)

    def print_parameter_devices(self, model):
        for name, param in model.named_parameters():
            print(f"{name}: {param.device}")

    def to_device(self, device):
        self.device = device
        # because actor is coded to be independent nn.Module for decision-making
        if hasattr(self, "actor"):
            self.actor.device = device
        if hasattr(self, "sampled_actor"):
            self.sampled_actor.device = device
        if hasattr(self, "policies"):
            for policy in self.policies:
                if policy is not None:
                    policy.device = device
                    if hasattr(policy, "actor"):
                        policy.actor.device = device
        self.to(device)

    def preprocess_state(self, state: torch.Tensor | np.ndarray) -> torch.Tensor:
        """
        Convert input state to a torch.Tensor on the correct device and dtype.
        Advanced shape formatting (Grayscale/RGB/Flattened) is handled safely inside
        the network's forward pass using self.input_shape.
        """
        if isinstance(state, np.ndarray):
            state = torch.as_tensor(state)
        elif not isinstance(state, torch.Tensor):
            raise ValueError("Unsupported state type. Must be a tensor or numpy array.")

        state = state.to(self.device).to(self.dtype)

        # Ensure batch dimension exists for basic unbatched 1D vectors
        if state.ndim == 1:
            state = state.unsqueeze(0)

        return state

    def compute_gradient_norm(self, models, names, device, dir="None", norm_type=2):
        grad_dict = {}
        for i, model in enumerate(models):
            if model is not None:
                total_norm = torch.tensor(0.0, device=device)
                try:
                    for param in model.parameters():
                        if (
                            param.grad is not None
                        ):  # Only consider parameters that have gradients
                            param_grad_norm = torch.norm(param.grad, p=norm_type)
                            total_norm += param_grad_norm**norm_type
                except:
                    try:
                        param_grad_norm = torch.norm(model.grad, p=norm_type)
                    except:
                        param_grad_norm = torch.tensor(0.0)
                    total_norm += param_grad_norm**norm_type

                total_norm = total_norm ** (1.0 / norm_type)
                grad_dict[dir + "/grad/" + names[i]] = total_norm.item()

        return grad_dict

    def compute_weight_norm(self, models, names, device, dir="None", norm_type=2):
        norm_dict = {}
        for i, model in enumerate(models):
            if model is not None:
                total_norm = torch.tensor(0.0, device=device)
                try:
                    for param in model.parameters():
                        param_norm = torch.norm(param, p=norm_type)
                        total_norm += param_norm**norm_type
                except:
                    param_norm = torch.norm(model, p=norm_type)
                    total_norm += param_norm**norm_type
                total_norm = total_norm ** (1.0 / norm_type)
                norm_dict[dir + "/weight/" + names[i]] = total_norm.item()

        return norm_dict

    def average_dict_values(self, dict_list):
        if not dict_list:
            return {}

        # Initialize a dictionary to hold the sum of values and counts for each key
        sum_dict = {}
        count_dict = {}

        # Iterate over each dictionary in the list
        for d in dict_list:
            for key, value in d.items():
                if key not in sum_dict:
                    sum_dict[key] = 0
                    count_dict[key] = 0
                sum_dict[key] += value
                count_dict[key] += 1

        # Calculate the average for each key
        avg_dict = {key: sum_val / count_dict[key] for key, sum_val in sum_dict.items()}

        return avg_dict

    def flat_grads(self, grads: tuple):
        """
        Flatten the gradients into a single tensor.
        """
        flat_grad = torch.cat([g.view(-1) for g in grads])
        return flat_grad

    def clip_grad_norm(self, grads, max_norm, eps=1e-6):
        # Compute total norm
        total_norm = torch.norm(torch.stack([g.norm(2) for g in grads]), 2)
        clip_coef = max_norm / (total_norm + eps)

        if clip_coef < 1:
            grads = tuple(g * clip_coef for g in grads)

        return grads

    def record_state_visitations(
        self, states: np.ndarray | torch.Tensor, alpha: float | None = None
    ):
        if alpha is None:
            alpha = 0.01

        wall_idx = 2
        agent_idx = 10
        goal_idx = 8

        if isinstance(states, torch.Tensor):
            states = states.cpu().numpy()

        is_discrete = self.is_discrete

        # if hasattr(self, "grid"):
        #     grid = self.grid
        # elif hasattr(self, "actor"):
        #     is_discrete = self.actor.is_discrete

        if is_discrete:
            if self.state_visitation is None:
                self.state_visitation = np.zeros_like(self.grid, dtype=np.float32)

            # Mask out wall and goal
            mask = (self.grid == wall_idx) | (self.grid == goal_idx)

            # Compute where agent is
            # agent_mask = (batch["states"] == agent_idx).astype(np.float32)
            # visitation = batch["states"].mean(0) + 1e-8  # average across batch
            visitation = np.zeros_like(self.grid, dtype=np.float32) + 1e-8
            for s in states:
                visitation[int(s[0]), int(s[1]), 0] += 1 / (states.shape[0])

            visitation[mask] = 0.0  # remove static or irrelevant regions

            # EMA update
            if self.state_visitation is None:
                self.state_visitation = visitation
            else:
                self.state_visitation = (
                    alpha * visitation + (1 - alpha) * self.state_visitation
                )
        else:
            # ----- CONTINUOUS CASE -----
            if len(states.shape) >= 3:
                states = states.reshape(states.shape[0], -1)

            # Initialize PCA once and fix basis
            if not hasattr(self, "pca_fitted") or not self.pca_fitted:
                self.pca = PCA(n_components=2)
                self.pca.fit(states)  # fit on first batch or a replay buffer
                self.pca_fitted = True
                # Optional: fix global bin edges based on the projected range
                proj = self.pca.transform(states)
                self.visitation_x_bounds = (
                    proj[:, 0].min() - 3,
                    proj[:, 0].max() + 3,
                )
                self.visitation_y_bounds = (
                    proj[:, 1].min() - 3,
                    proj[:, 1].max() + 3,
                )

            # Project using fixed PCA
            projected = self.pca.transform(states)
            x_min, x_max = self.visitation_x_bounds
            y_min, y_max = self.visitation_y_bounds

            bins = 100
            heatmap, _, _ = np.histogram2d(
                projected[:, 0],
                projected[:, 1],
                bins=bins,
                range=[[x_min, x_max], [y_min, y_max]],
            )
            heatmap = heatmap.T
            heatmap += 1e-8
            heatmap /= heatmap.sum()

            if self.state_visitation is None:
                self.state_visitation = heatmap
            else:
                self.state_visitation = (
                    alpha * heatmap + (1 - alpha) * self.state_visitation
                )
