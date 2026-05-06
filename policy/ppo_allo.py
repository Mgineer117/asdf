from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from policy.irpo import NUM_GOALS
from policy.ppo import PPO_Learner
from utils.intrinsic_rewards import ALLOIntRewardFunctionG
from utils.rl import estimate_advantages


class PPO_ALLO_Learner(PPO_Learner):
    """
    Goal-conditioned PPO trained on `r_ext + alpha * r_int_allo`.

    Inherits the entire PPO update from `PPO_Learner`; only the reward
    construction (`_compute_rewards`) and a post-update diagnostic hook
    (`_post_update_hook`) are overridden.

    `alpha` linearly anneals from `alpha_init` -> `alpha_final` across
    training (using `learning_progress` supplied by `OnPolicyTrainer`),
    so the policy shifts from exploration-shaped to pure-task reward by
    the end of training.

    The post-update hook produces a per-goal gradient-conflict analysis
    (cosine similarity + angle heatmaps + scalar conflict metrics),
    mirroring `IRPO_Learner._goal_gradient_conflict_figure`.
    """

    def __init__(
        self,
        intrinsic_reward_fn: ALLOIntRewardFunctionG,
        alpha: float = 1.0,
        alpha_final: float = 0.0,
        **ppo_kwargs,
    ):
        if not isinstance(intrinsic_reward_fn, ALLOIntRewardFunctionG):
            raise TypeError(
                "PPO_ALLO_Learner requires the goal-conditioned ALLO reward "
                f"(`ALLOIntRewardFunctionG`); got "
                f"{type(intrinsic_reward_fn).__name__}."
            )

        super().__init__(**ppo_kwargs)

        self.name = "PPO_ALLO"
        self.intrinsic_reward_fn = intrinsic_reward_fn
        self.alpha_init = float(alpha)
        self.alpha_final = float(alpha_final)

    def _current_alpha(self, learning_progress: float) -> float:
        p = max(0.0, min(1.0, float(learning_progress)))
        return self.alpha_init + (self.alpha_final - self.alpha_init) * p

    def _compute_rewards(self, batch, **kwargs):
        ext_rewards = self.preprocess_state(batch["rewards"])
        states = self.preprocess_state(batch["states"])
        next_states = self.preprocess_state(batch["next_states"])

        with torch.no_grad():
            int_rewards = self.intrinsic_reward_fn(states, next_states).to(
                dtype=ext_rewards.dtype, device=ext_rewards.device
            )

        alpha = self._current_alpha(kwargs.get("learning_progress", 0.0))
        rewards = ext_rewards + alpha * int_rewards

        log = {
            f"{self.name}/analytics/avg_ext_reward": ext_rewards.mean().item(),
            f"{self.name}/analytics/avg_int_reward": int_rewards.mean().item(),
            f"{self.name}/analytics/alpha": alpha,
        }
        return rewards, log

    def _post_update_hook(self, batch, **kwargs):
        """Per-goal gradient conflict analysis on the post-update policy."""
        return self._goal_gradient_conflict(batch, **kwargs)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def _goal_gradient_conflict(self, batch, **kwargs):
        """
        Cluster the batch by goal, then for each cluster compute the true
        policy-gradient direction g_c = ∇_θ -E[log π(a|s) · Â(s,a)] under
        the post-update actor. Report pairwise cosine similarity / angle
        between {g_c}, plus scalar summary metrics.

        Returns (loss_dict, image_dict) or ({}, {}) on failure / when
        goal_idx is unavailable.
        """
        from sklearn.cluster import KMeans

        args = self.intrinsic_reward_fn.args
        goal_idx = getattr(args, "goal_idx", None)
        if goal_idx is None:
            return {}, {}

        try:
            states = self.preprocess_state(batch["states"])
            actions = self.preprocess_state(batch["actions"])
            terminations = self.preprocess_state(batch["terminations"])
            truncations = self.preprocess_state(batch["truncations"])

            # Recompute combined reward + advantages (consistent with the
            # objective the actor was just optimized against).
            rewards, _ = self._compute_rewards(batch, **kwargs)
            with torch.no_grad():
                values = self.critic(states)
                advantages, _ = estimate_advantages(
                    rewards,
                    terminations,
                    truncations,
                    values,
                    gamma=self.gamma,
                    gae=self.gae,
                )
                advantages = advantages.detach()

            goals_np = states[:, goal_idx].detach().cpu().numpy()
            env_name = getattr(args, "env_name", "")
            n_clusters = NUM_GOALS.get(env_name, 6)
            n_clusters = min(n_clusters, goals_np.shape[0])
            if n_clusters < 2:
                return {}, {}

            kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=0)
            labels = kmeans.fit_predict(goals_np)

            grad_vecs, valid_clusters = [], []
            for c in range(n_clusters):
                mask = torch.from_numpy(labels == c).to(states.device)
                if mask.sum() < 4:
                    continue
                mb_states = states[mask].detach()
                mb_actions = actions[mask].detach()
                mb_adv = advantages[mask]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                probe = deepcopy(self.actor)
                probe.zero_grad()
                _, meta = probe(mb_states)
                logprobs = probe.log_prob(meta["dist"], mb_actions)
                loss = -(logprobs * mb_adv).mean()
                loss.backward()

                # Zero-fill any missing grads so the flattened vector has a
                # consistent shape across clusters (cosine similarity is
                # only meaningful when the dimensionalities match).
                flat = torch.cat(
                    [
                        (p.grad.detach() if p.grad is not None else torch.zeros_like(p))
                        .cpu()
                        .flatten()
                        for p in probe.parameters()
                    ]
                )
                grad_vecs.append(flat)
                valid_clusters.append(c)
                del probe

            if len(grad_vecs) < 2:
                return {}, {}

            n = len(grad_vecs)
            G = torch.stack(grad_vecs)  # (n, P)
            G_norm = F.normalize(G, dim=1, eps=1e-12)
            cos_sim = (G_norm @ G_norm.T).cpu().numpy()
            angles = np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0)))

            off_diag = cos_sim[~np.eye(n, dtype=bool)]
            conflict_frac = float((off_diag < 0).mean())
            mean_cos = float(off_diag.mean())
            min_cos = float(off_diag.min())

            metrics = {
                f"{self.name}/grad_conflict/conflict_fraction": conflict_frac,
                f"{self.name}/grad_conflict/mean_cosine_similarity": mean_cos,
                f"{self.name}/grad_conflict/min_cosine_similarity": min_cos,
                f"{self.name}/grad_conflict/num_clusters": float(n),
            }
            image = self._render_conflict_figure(
                cos_sim, angles, valid_clusters, conflict_frac, mean_cos
            )
            image_dict = {"goal_gradient_conflict": image} if image is not None else {}
            return metrics, image_dict

        except Exception as e:
            print(f"[WARNING] PPO_ALLO goal gradient conflict failed: {e}")
            return {}, {}

    @staticmethod
    def _render_conflict_figure(
        cos_sim: np.ndarray,
        angles: np.ndarray,
        valid_clusters: list,
        conflict_frac: float,
        mean_cos: float,
    ):
        n = cos_sim.shape[0]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        tick_labels = [f"G{c}" for c in valid_clusters]

        im0 = axes[0].imshow(cos_sim, vmin=-1, vmax=1, cmap="RdBu_r")
        axes[0].set_title(
            f"Cosine Similarity (conflict={conflict_frac:.2f}, mean={mean_cos:.3f})"
        )
        axes[0].set_xticks(range(n))
        axes[0].set_xticklabels(tick_labels)
        axes[0].set_yticks(range(n))
        axes[0].set_yticklabels(tick_labels)
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        for i in range(n):
            for j in range(n):
                axes[0].text(
                    j, i, f"{cos_sim[i, j]:.2f}",
                    ha="center", va="center", fontsize=7,
                )

        im1 = axes[1].imshow(angles, vmin=0, vmax=180, cmap="hot_r")
        axes[1].set_title("Gradient Angle (degrees)")
        axes[1].set_xticks(range(n))
        axes[1].set_xticklabels(tick_labels)
        axes[1].set_yticks(range(n))
        axes[1].set_yticklabels(tick_labels)
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        for i in range(n):
            for j in range(n):
                axes[1].text(
                    j, i, f"{angles[i, j]:.0f}",
                    ha="center", va="center", fontsize=7,
                )

        plt.tight_layout()
        try:
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
        finally:
            plt.close(fig)
        return buf
