"""IS-correction variant of IRPO (paper Eq. 20) with Thompson option selection.

Differences vs the default Thompson IRPO:
  - Option choice: Thompson sampling (inherited from IRPO_Thompson_Learner) is
    used to pick which exploratory policy's rollouts feed the base-policy
    update — we only collect samples for that one option.
  - Base-policy gradient: replaces the backpropagated IRPO gradient with the
    importance-sampling-corrected policy gradient

        ∇θ J_R(θ) ≈ E_{d^{π̃}} [ ρ(s, a) Q^{π̃}_R(s, a) ∇θ log π_θ(a | s) ]

    where ρ = π_θ / π̃_θ and Q^{π̃}_R is the *exploratory* policy's extrinsic
    critic (paper finding: better than using the base policy's critic).
  - Evaluation policy: the base policy `self.actor` is the primary evaluation
    policy — we override `forward()` accordingly. The exploratory policies
    are treated purely as data-collection / behaviour policies.
"""

from __future__ import annotations

import numpy as np
import torch

from policy.irpo_thompson import IRPO_Thompson_Learner


class IRPO_IS_Learner(IRPO_Thompson_Learner):
    name = "IRPO_IS"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "IRPO_IS"

    def forward(self, state: np.ndarray, deterministic: bool = False, **kwargs):
        # Base policy is the evaluation policy (exploratory policies are only
        # behaviour policies for IS-corrected updates).
        state = self.preprocess_state(state)
        a, metaData = self.actor(state, deterministic=deterministic)
        return a, {
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
            "entropy": metaData["entropy"],
            "dist": metaData["dist"],
        }

    def _per_option_gradient(
        self, option_idx: int, policy_dict: dict, gradient_dict: dict
    ):
        # Latest exploratory rollout for this option (cached during the inner
        # loop). When num_exp_updates < 2 there is no cached batch — fall back
        # to the standard backprop path so the run still progresses.
        batch = self._last_batches.get(option_idx)
        if batch is None:
            return super()._per_option_gradient(option_idx, policy_dict, gradient_dict)

        states = self.preprocess_state(batch["states"])
        actions = self.preprocess_state(batch["actions"])

        exp_actor_idx = f"{option_idx}_{self.num_exp_updates - 1}"
        exp_actor = policy_dict[exp_actor_idx]

        with torch.no_grad():
            _, exp_meta = exp_actor(states)
            exp_logprobs = exp_actor.log_prob(exp_meta["dist"], actions).detach()
            # Approximate Q^{π̃}_R with V from the exploratory ext critic;
            # paper neglects the discounted-occupancy correction term.
            q_R = self.ext_critics[option_idx](states).detach().squeeze(-1)

        _, base_meta = self.actor(states)
        base_logprobs = self.actor.log_prob(base_meta["dist"], actions)

        # ρ = π_θ / π̃, kept differentiable so ∇θ ρ = ρ ∇θ log π_θ. The
        # gradient of (ρ * Q.detach()).mean() equals E[ρ Q ∇log π_θ] (Eq. 20).
        ratio = torch.exp(base_logprobs - exp_logprobs)
        surrogate = (ratio * q_R).mean()

        gradients = torch.autograd.grad(
            surrogate,
            tuple(self.actor.parameters()),
            allow_unused=True,
        )
        gradients = tuple(
            g if g is not None else torch.zeros_like(p)
            for p, g in zip(self.actor.parameters(), gradients)
        )
        # learn_base_policy expects loss-form gradients (it does p -= lr * g),
        # so return -∇surrogate = ∇(loss).
        return tuple(-g for g in gradients)
