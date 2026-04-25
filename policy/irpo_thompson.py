"""Thompson-sampling option selection for IRPO.

Replaces the argmax/softmax/uniform aggregation with Thompson sampling: at
each iteration we maintain a Gaussian posterior over each option's expected
extrinsic return and *sample* one return per option, then commit to the
arg-max-of-samples option for at least `min_base_updates` base-policy updates
before re-sampling. This is a stochastic exploration variant of argmax IRPO.

For sample efficiency, while an option is locked we only collect rollouts and
update critics/actor for that option (parent IRPO_Learner.learn() already
honours this once we restrict `self.policy_indices` to a single index, since
all per-option work iterates over `active_indices`). We additionally override
`init_exp_policies` to skip deep-copying actors for unused options.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from policy.irpo import IRPO_Learner


class IRPO_Thompson_Learner(IRPO_Learner):
    name = "IRPO_Thompson"

    def __init__(self, *args, min_base_updates: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "IRPO_Thompson"
        self.min_base_updates = max(1, int(min_base_updates))

        # Welford running stats per option for Gaussian posterior over the mean.
        self.thompson_n = torch.zeros(self.num_options, device=self.device)
        self.thompson_mean = torch.zeros(self.num_options, device=self.device)
        self.thompson_m2 = torch.zeros(self.num_options, device=self.device)
        # Optimistic prior so each option is tried before the posterior tightens.
        self._prior_var = 1.0

        self._chosen_option: int | None = None
        self._lock_remaining = 0
        self._latest_ext_return: float | None = None

    # ------------------------------------------------------------------ #
    # Thompson-sampling utilities
    # ------------------------------------------------------------------ #
    def _sample_option(self) -> int:
        samples = np.empty(self.num_options, dtype=np.float64)
        for i in range(self.num_options):
            n = float(self.thompson_n[i].item())
            mean = float(self.thompson_mean[i].item())
            if n < 2:
                # Optimistic prior — always try unvisited options first.
                var = self._prior_var
            else:
                var = float(self.thompson_m2[i].item()) / (n - 1.0)
            std = (var / max(n, 1.0)) ** 0.5
            samples[i] = mean + std * float(np.random.randn())
        return int(np.argmax(samples))

    def _update_thompson(self, option_idx: int, value: float) -> None:
        # Welford's online algorithm for running mean / variance.
        self.thompson_n[option_idx] += 1
        n = float(self.thompson_n[option_idx].item())
        delta = value - float(self.thompson_mean[option_idx].item())
        self.thompson_mean[option_idx] += delta / n
        delta2 = value - float(self.thompson_mean[option_idx].item())
        self.thompson_m2[option_idx] += delta * delta2

    # ------------------------------------------------------------------ #
    # Overrides
    # ------------------------------------------------------------------ #
    def init_exp_policies(self):
        """Only deep-copy the locked option's actor (sample-efficient init)."""
        active = self.policy_indices
        return {f"{i}_{0}": deepcopy(self.actor) for i in active}

    def learn_exploratory_policy(self, actor, batch, i: int, flag: bool):
        out = super().learn_exploratory_policy(actor, batch, i, flag)
        if flag:
            self._latest_ext_return = float(out["ext_return"])
        return out

    def learn(self, env, sampler, seed, learning_progress, **kwargs):
        # Re-sample if no current selection or lock has expired.
        if self._chosen_option is None or self._lock_remaining <= 0:
            self._chosen_option = self._sample_option()
            self._lock_remaining = self.min_base_updates

        # Restrict downstream parent logic to the locked option only. This
        # gates rollout collection, critic updates, exploratory updates, and
        # gradient backprop to a single option.
        original_policy_indices = self.policy_indices
        self.policy_indices = [self._chosen_option]
        try:
            result = super().learn(env, sampler, seed, learning_progress, **kwargs)
        finally:
            self.policy_indices = original_policy_indices

        # Update posterior with the most recent observed return for the chosen
        # option (captured in learn_exploratory_policy via _latest_ext_return).
        if self._latest_ext_return is not None:
            self._update_thompson(self._chosen_option, self._latest_ext_return)

        # Annotate logs with Thompson state for diagnostics.
        loss_dict = result.get("loss_dict", {})
        loss_dict[f"{self.name}/thompson/chosen_option"] = self._chosen_option
        loss_dict[f"{self.name}/thompson/lock_remaining"] = self._lock_remaining
        for i in range(self.num_options):
            loss_dict[f"{self.name}/thompson/mean_{i}"] = float(
                self.thompson_mean[i].item()
            )
            loss_dict[f"{self.name}/thompson/n_{i}"] = float(
                self.thompson_n[i].item()
            )
        result["loss_dict"] = loss_dict

        self._lock_remaining -= 1
        return result
