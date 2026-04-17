import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from policy.layers.base import Base
from policy.layers.ppo_networks import PPO_Actor, PPO_Critic
from utils.rl import (
    compute_kl,
    conjugate_gradients,
    estimate_advantages,
    flat_params,
    hessian_vector_product,
    set_flat_params,
)

ACHIEVED_GOAL_IDX = {
    "maze": [0, 1],
    "fourrooms": [0, 1],
    "antmaze": [-4, -3],
    "pointmaze": [-4, -3],
    "fetchreach": [-6, -5, -4],
    "fetchpush": [-6, -5, -4],
}
DESIRED_GOAL_IDX = {
    "maze": [2, 3],
    "fourrooms": [2, 3],
    "antmaze": [-2, -1],
    "pointmaze": [-2, -1],
    "fetchreach": [-3, -2, -1],
    "fetchpush": [-3, -2, -1],
}

DIST_THRESHOLD = {
    "maze": 0.001,
    "fourrooms": 0.001,
    "antmaze": 0.45,
    "pointmaze": 0.45,
    "fetchreach": 0.1,
    "fetchpush": 0.1,
}


class HTRPO_Learner(Base):
    def __init__(
        self,
        actor: PPO_Actor,
        critic: PPO_Critic,
        is_discrete: bool,
        nupdates: int,
        env_name: str,
        lr: float = 5e-4,
        batch_size: int = 4096,
        entropy_scaler: float = 1e-3,
        l2_reg: float = 1e-8,
        target_kl: float = 0.03,
        damping: float = 1e-3,
        backtrack_iters: int = 10,
        backtrack_coeff: float = 0.8,
        gamma: float = 0.99,
        gae: float = 0.9,
        device: str = "cpu",
        num_hindsight_goals: int = 100,
        use_hgf: bool = True,
    ):
        super().__init__(device=device)

        self.name = "HTRPO"
        self.device = device
        self.env_name = env_name

        self.entropy_scaler = entropy_scaler
        self.batch_size = batch_size
        self.damping = damping
        self.gamma = gamma
        self.gae = gae
        self.l2_reg = l2_reg
        self.backtrack_iters = backtrack_iters
        self.backtrack_coeff = backtrack_coeff
        self.nupdates = nupdates

        # self.target_kl = target_kl
        self.target_kl = 0.00001

        self.num_hindsight_goals = num_hindsight_goals
        self.use_hgf = use_hgf

        self.achieved_goal_idx = ACHIEVED_GOAL_IDX[env_name]
        self.desired_goal_idx = DESIRED_GOAL_IDX[env_name]
        self.distance_threshold = DIST_THRESHOLD[env_name]

        self.actor = actor
        self.critic = critic

        self.optimizer = torch.optim.Adam(params=self.critic.parameters(), lr=lr)

        self.steps = 0
        self.to(self.dtype).to(self.device)

    def forward(self, state: np.ndarray, deterministic: bool = False, **kwargs):
        a, metaData = self.actor(state, deterministic=deterministic)
        return a, {
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
            "entropy": metaData["entropy"],
            "dist": metaData["dist"],
        }

    def _hindsight_goal_filtering(
        self, achieved_goals: np.ndarray, original_goals: np.ndarray, num_goals: int
    ) -> np.ndarray:
        dists = np.linalg.norm(
            achieved_goals[:, None, :] - original_goals[None, :, :], axis=2
        )
        min_dists_to_o = np.min(dists, axis=1)

        valid_mask = min_dists_to_o < self.distance_threshold
        G_v = achieved_goals[valid_mask]

        selected_goals = []

        if len(G_v) == 0:
            sorted_indices = np.argsort(min_dists_to_o)
            selected_goals = achieved_goals[sorted_indices[:num_goals]].tolist()
        else:
            g0_idx = np.random.choice(len(G_v))
            selected_goals.append(G_v[g0_idx])
            remaining_G_v = np.delete(G_v, g0_idx, axis=0)

            for _ in range(min(num_goals - 1, len(remaining_G_v))):
                d2s = np.linalg.norm(
                    remaining_G_v[:, None, :] - np.array(selected_goals)[None, :, :],
                    axis=2,
                )
                min_d2s = np.min(d2s, axis=1)
                best_idx = np.argmax(min_d2s)
                selected_goals.append(remaining_G_v[best_idx])
                remaining_G_v = np.delete(remaining_G_v, best_idx, axis=0)

            while len(selected_goals) < num_goals:
                selected_goals.append(
                    achieved_goals[np.random.choice(len(achieved_goals))]
                )

        return np.array(selected_goals)

    def _create_hindsight_batch(
        self, batch: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """
        Build the augmented hindsight batch and tag every sample with a goal-group ID.

        goal_id == 0  →  original data  (IS weight stays 1, no WIS correction)
        goal_id >= 1  →  hindsight data for the (goal_id - 1)-th sampled goal

        The goal_id array is consumed by _compute_wis_and_discounts so that WIS
        normalisation is performed across episodes per goal group, matching the
        denominator of eq. 79 in the HTRPO paper.  It is stored under the key
        "hs_goal_ids" and removed from the dict before tensor conversion in learn().
        """
        achieved_goals = batch["states"][:, self.achieved_goal_idx]
        original_goals = batch["states"][:, self.desired_goal_idx]

        if self.use_hgf:
            G = self._hindsight_goal_filtering(
                achieved_goals, original_goals, self.num_hindsight_goals
            )
        else:
            indices = np.random.choice(
                len(achieved_goals), size=self.num_hindsight_goals, replace=True
            )
            G = achieved_goals[indices]

        n_orig = len(batch["states"])
        augmented_batches = {k: [batch[k]] for k in batch.keys()}
        goal_ids_list = [np.zeros(n_orig, dtype=np.int64)]  # goal_id = 0 for original

        # Pre-compute episode boundaries so we can truncate early-terminating virtual episodes
        orig_dones = np.logical_or(batch["terminations"], batch["truncations"]).astype(
            bool
        )
        ep_ends = np.where(orig_dones)[0]
        ep_starts = []
        ep_ends_clean = []
        start_idx = 0
        for end_idx in ep_ends:
            ep_starts.append(start_idx)
            ep_ends_clean.append(end_idx)
            start_idx = end_idx + 1
        if start_idx < n_orig:
            ep_starts.append(start_idx)
            ep_ends_clean.append(n_orig - 1)

        for g_idx, g_prime in enumerate(G):
            hs_batch = {k: np.copy(v) for k, v in batch.items()}
            hs_batch["states"][:, self.desired_goal_idx] = g_prime

            current_achieved = hs_batch["states"][:, self.achieved_goal_idx]
            dists = np.linalg.norm(current_achieved - g_prime, axis=1)
            reached_mask = dists < self.distance_threshold

            hs_batch["rewards"][:] = 0.0
            hs_batch["rewards"][reached_mask] = 1.0

            if self.env_name in ["fetchreach", "fetchpush"]:
                # 1. FIXED HORIZON ENVS (Fetch)
                # Do NOT alter truncations or terminations. The episode boundaries
                # (T=50 timeouts) are already physically recorded in the buffer.
                hs_batch["rewards"][:] = 0.0
                hs_batch["rewards"][reached_mask] = 1.0

            else:
                # 2. EARLY TERMINATION ENVS (Maze, etc.)
                # If the environment physically halts upon success, reaching the
                # virtual goal means the episode is technically over.
                hs_batch["rewards"][:] = 0.0
                hs_batch["rewards"][reached_mask] = 1.0

                # # Use logical_or so you don't delete any existing terminations
                # # that were already in the rollout buffer.
                # if "terminations" in hs_batch:
                #     current_terms = hs_batch["terminations"].astype(bool)
                #     new_terms = current_terms | reached_mask
                #     hs_batch["terminations"] = new_terms.astype(np.float32)
                # Use logical_or so you don't delete any existing terminations
                # that were already in the rollout buffer.
                if "terminations" in hs_batch:
                    current_terms = hs_batch["terminations"].astype(bool)
                    new_terms = current_terms | reached_mask
                    hs_batch["terminations"] = new_terms.astype(np.float32)

            for k in batch.keys():
                augmented_batches[k].append(hs_batch[k])
            goal_ids_list.append(np.full(n_orig, g_idx + 1, dtype=np.int64))

        result = {k: np.concatenate(v, axis=0) for k, v in augmented_batches.items()}
        result["hs_goal_ids"] = np.concatenate(goal_ids_list, axis=0)
        return result

    def _compute_wis_and_discounts(
        self,
        step_is_ratios: torch.Tensor,
        dones: torch.Tensor,
        goal_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute WIS-normalised cumulative IS ratios and γ^t discount factors.

        WHY THE PREVIOUS VERSION WAS WRONG
        ───────────────────────────────────
        The previous code normalised IS weights *within* a single episode over
        its own timesteps.  That has no basis in the paper and made WIS a no-op
        for every transition except at episode boundaries, which is why FetchPush
        failed to replicate the paper results.

        CORRECT WIS (eq. 79, HTRPO paper)
        ───────────────────────────────────
        For each hindsight goal group g' and timestep position t, the WIS weight
        for episode τ is:

            w̃(τ, t, g') =        ∏_{k=0}^{t} π_θ̃(aₖ|sₖ,g') / π_θ̃(aₖ|sₖ,g)
                           ──────────────────────────────────────────────────────
                           Σ_{τ'} ∏_{k=0}^{t} π_θ̃(aₖ|sₖ,g') / π_θ̃(aₖ|sₖ,g)

        The denominator sums the cumulative IS ratio across ALL episodes at the
        same timestep position t.  Episodes shorter than t are simply excluded
        from the sum (equivalent to zero-padding in the original (Ne,Ng,T) tensor).

        This is implemented without a fixed-size (Ne, Ng, T) tensor so that
        variable-length episodes are handled correctly for maze-type environments
        where hindsight relabelling can shorten episodes.

        ORIGINAL DATA (goal_id == 0)
        ─────────────────────────────
        No IS correction is needed — cum_is_ratio stays 1.

        Args:
            step_is_ratios : π_θ̃(a|s,g') / π_θ̃(a|s,g) per step,  shape (N,)
            dones          : episode-end flags (1 = terminal),       shape (N,)
            goal_ids       : goal-group index per sample,            shape (N,)

        Returns:
            cum_is_ratios   : WIS-normalised cumulative IS weights,  shape (N,)
            discount_factors: γ^t for each sample,                   shape (N,)
        """
        n = len(step_is_ratios)
        device = step_is_ratios.device
        dtype = step_is_ratios.dtype

        cum_is_ratios = torch.ones(n, dtype=dtype, device=device)
        discount_factors = torch.ones(n, dtype=dtype, device=device)

        for gid in goal_ids.unique():

            # ── 1. All sample indices that belong to this goal group ───────────
            gid_mask = goal_ids == gid
            indices = gid_mask.nonzero(as_tuple=True)[0]  # ordered global positions

            # ── 2. Split into episodes using done flags ────────────────────────
            #    Works for variable-length episodes: the window between two
            #    consecutive done=1 flags is one episode.  A trailing window
            #    that never received done=1 is kept as a truncated episode.
            episodes: list[torch.Tensor] = []
            ep_start = 0
            for local_i in range(len(indices)):
                if dones[indices[local_i]]:
                    episodes.append(indices[ep_start : local_i + 1])
                    ep_start = local_i + 1
            if ep_start < len(indices):  # truncated final episode
                episodes.append(indices[ep_start:])

            # ── 3. Discount factors: γ^t, t = step index within episode ───────
            #    Computed per goal group so that envs where hindsight relabelling
            #    changes episode length (non-fetch) still get the right γ^t.
            for ep_idx in episodes:
                ep_len = len(ep_idx)
                t_vals = torch.arange(ep_len, dtype=dtype, device=device)
                discount_factors[ep_idx] = self.gamma**t_vals

            # ── 4. Original data: leave IS weight = 1 ─────────────────────────
            if gid == 0:
                continue

            # ── 5. Cumulative IS ratios per episode ───────────────────────────
            #    ep_cum_ratios[i][t] = ∏_{k=0}^{t} step_is_ratios  for episode i
            ep_cum_ratios: list[torch.Tensor] = [
                torch.cumprod(step_is_ratios[ep_idx], dim=0) for ep_idx in episodes
            ]

            # ── 6. WIS normalisation: divide by sum across episodes at each t ─
            #    Only episodes that reach position t contribute to the denominator,
            #    which correctly handles variable-length episodes without padding.
            max_ep_len = max(len(ep) for ep in episodes)
            for t in range(max_ep_len):
                # Indices into `episodes` / `ep_cum_ratios` that reach position t
                valid = [i for i, ep in enumerate(episodes) if len(ep) > t]
                if not valid:
                    continue

                # Σ_τ ∏_{k=0}^{t} π_θ̃(aₖ|sₖ,g') / π_θ̃(aₖ|sₖ,g)
                ratio_sum = sum(ep_cum_ratios[i][t] for i in valid)

                for i in valid:
                    cum_is_ratios[episodes[i][t]] = ep_cum_ratios[i][t] / (
                        ratio_sum + 1e-8
                    )

        return cum_is_ratios.unsqueeze(-1), discount_factors.unsqueeze(-1)

    def learn(self, env, sampler, seed, **kwargs):
        self.train()

        batch, sample_time = sampler.collect_samples(env, self.actor, seed)
        t0 = time.time()
        old_rewards = batch["rewards"]

        hs_batch = self._create_hindsight_batch(batch)

        # Remove WIS metadata before tensor conversion — preprocess_state must
        # never see integer goal-group indices.
        hs_goal_ids_np: np.ndarray = hs_batch.pop("hs_goal_ids")

        states = self.preprocess_state(hs_batch["states"])
        actions = self.preprocess_state(hs_batch["actions"])
        rewards = self.preprocess_state(hs_batch["rewards"])
        terminations = self.preprocess_state(hs_batch["terminations"])
        truncations = self.preprocess_state(hs_batch["truncations"])
        dones = torch.logical_or(terminations.bool(), truncations.bool())
        original_logprobs = self.preprocess_state(hs_batch["logprobs"])

        goal_ids = torch.from_numpy(hs_goal_ids_np).to(self.device)

        timesteps = states.shape[0]

        # log π_θ̃(a | s, g')
        # For goal_id == 0 (original data), the goal embedded in `states` is the
        # original goal g, so relabeled_logprobs == original_logprobs and
        # step_is_ratios == 1 for those samples — no correction applied.
        with torch.no_grad():
            _, old_metaData = self.actor(states)
            relabeled_logprobs = self.actor.log_prob(old_metaData["dist"], actions)

        # per-step IS ratio: π_θ̃(a|s,g') / π_θ̃(a|s,g)
        step_is_ratios = torch.exp(relabeled_logprobs - original_logprobs)

        # Cross-episode WIS-normalised cumulative IS ratios and γ^t discounts.
        cum_is_ratios, discount_factors = self._compute_wis_and_discounts(
            step_is_ratios, dones, goal_ids
        )

        with torch.no_grad():
            values = self.critic(states)

            # Compute V(s') by shifting the values array
            next_values = torch.zeros_like(values)
            next_values[:-1] = values[1:]

            # Mask out next values where the episode ended
            next_values[dones.bool()] = 0.0

            # One-step TD Advantage: r + gamma * V(s') - V(s)
            advantages = rewards + self.gamma * next_values - values

            # The return is the TD target
            returns = rewards + self.gamma * next_values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_gradients, actor_loss = self.actor_loss(
            states,
            actions,
            relabeled_logprobs,
            advantages,
            cum_is_ratios,
            discount_factors,
        )

        old_actor = deepcopy(self.actor)
        grad_flat = torch.cat([g.view(-1) for g in actor_gradients]).detach()

        def kl_fn():
            _, metaData = self.actor(states)
            current_logprobs = self.actor.log_prob(metaData["dist"], actions)
            # QKL approximation to KL divergence (eq. 80)
            kl_approx = 0.5 * (relabeled_logprobs - current_logprobs) ** 2
            return (discount_factors * cum_is_ratios * kl_approx).mean()

        Hv = lambda v: hessian_vector_product(kl_fn, self.actor, self.damping, v)
        step_dir = conjugate_gradients(Hv, grad_flat, nsteps=10)

        sAs = 0.5 * torch.dot(step_dir, Hv(step_dir))
        lm = torch.sqrt(sAs / self.target_kl)
        full_step = step_dir / (lm + 1e-8)

        with torch.no_grad():
            old_params = flat_params(old_actor)
            success = False
            for i in range(self.backtrack_iters):
                alpha = self.backtrack_coeff**i
                # Subtracting the step because we are minimizing the negative objective
                new_params = old_params - alpha * full_step
                set_flat_params(self.actor, new_params)

                kl = kl_fn()

                _, metaData = self.actor(states)
                logprobs = self.actor.log_prob(metaData["dist"], actions)
                ratios = torch.exp(logprobs - relabeled_logprobs)
                new_actor_loss = -(
                    ratios * discount_factors * cum_is_ratios * advantages
                ).mean()

                if kl <= self.target_kl and new_actor_loss <= actor_loss:
                    success = True
                    break

            if not success:
                set_flat_params(self.actor, old_params)
                kl = kl_fn()

        critic_iteration = 20
        batch_size_critic = states.size(0) // critic_iteration
        grad_dict_list = []
        for _ in range(critic_iteration):
            indices = torch.randperm(states.size(0))[:batch_size_critic]
            mb_states = states[indices]
            mb_returns = returns[indices]

            value_loss = self.critic_loss(mb_states, mb_returns)
            self.optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)

            grad_dict = self.compute_gradient_norm(
                [self.critic], ["critic"], dir=f"{self.name}", device=self.device
            )
            grad_dict_list.append(grad_dict)
            self.optimizer.step()

        grad_dict = self.average_dict_values(grad_dict_list)

        loss_dict = {
            f"{self.name}/loss/actor_loss": actor_loss.item(),
            f"{self.name}/loss/value_loss": value_loss.item(),
            f"{self.name}/analytics/backtrack_success": int(success),
            f"{self.name}/analytics/klDivergence": kl.item(),
            f"{self.name}/analytics/avg_rewards": torch.mean(rewards).item(),
            f"{self.name}/analytics/avg_old_rewards": np.mean(old_rewards).item(),
        }

        self.eval()
        return {"loss_dict": loss_dict, "timesteps": timesteps}

    def actor_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        relabeled_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        cum_is_ratios: torch.Tensor,
        discount_factors: torch.Tensor,
    ):
        _, metaData = self.actor(states)
        logprobs = self.actor.log_prob(metaData["dist"], actions)
        ratios = torch.exp(logprobs - relabeled_logprobs)
        actor_loss = -(ratios * discount_factors * cum_is_ratios * advantages).mean()

        actor_gradients = torch.autograd.grad(actor_loss, self.actor.parameters())
        return actor_gradients, actor_loss.detach()

    def critic_loss(self, states: torch.Tensor, returns: torch.Tensor):
        mb_values = self.critic(states)
        value_loss = self.mse_loss(mb_values, returns)
        return value_loss
