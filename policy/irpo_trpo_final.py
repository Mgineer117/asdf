"""IRPO -> standard-TRPO switch variant.

Pre-switch: standard Thompson IRPO (option selection + IRPO base-policy update).
Post-switch (i.e. once `learning_progress > trpo_switch_progress`): we switch
to a vanilla TRPO objective, training the *final exploratory policy with the
highest performance among others* directly with the standard policy gradient.
On first crossing the threshold we copy that best exploratory policy's weights
into `self.actor` (and the corresponding ext critic stays as the value
function); from then on `learn()` collects rollouts with `self.actor` and runs
a single TRPO step per call, fully bypassing the IRPO machinery.
"""

from __future__ import annotations

import time
from copy import deepcopy

import numpy as np
import torch

from policy.irpo_thompson import IRPO_Thompson_Learner
from utils.rl import estimate_advantages


class IRPO_TRPOFinal_Learner(IRPO_Thompson_Learner):
    name = "IRPO_TRPO"

    def __init__(self, *args, trpo_switch_progress: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "IRPO_TRPO"
        self.trpo_switch_progress = float(trpo_switch_progress)
        self._switched = False
        self._switch_option_idx: int | None = None

    # ------------------------------------------------------------------ #
    # Switch entry-point
    # ------------------------------------------------------------------ #
    def _maybe_switch(self) -> None:
        if self._switched:
            return
        # Pick the best-performing final exploratory policy and copy its
        # weights into the base policy so subsequent TRPO updates train it.
        best_idx = int(torch.argmax(self.perf_gains).item())
        try:
            self.actor.load_state_dict(
                self.final_exp_policies[best_idx].state_dict()
            )
        except Exception:
            # final_exp_policies[best_idx] may equal self.actor on the very
            # first iteration if the IRPO phase never produced an update;
            # in that case there is nothing to copy.
            pass
        self._switch_option_idx = best_idx
        self._switched = True

    # ------------------------------------------------------------------ #
    # Standard-TRPO step on the (transferred) base policy
    # ------------------------------------------------------------------ #
    def _trpo_step(self, env, sampler, seed):
        t0 = time.time()
        batch, sample_time = sampler.collect_samples(env, self.actor, seed)
        if isinstance(batch, list):
            batch = batch[0]

        states = self.preprocess_state(batch["states"])
        actions = self.preprocess_state(batch["actions"])
        rewards = self.preprocess_state(batch["rewards"])
        terminations = self.preprocess_state(batch["terminations"])
        truncations = self.preprocess_state(batch["truncations"])
        timesteps = states.shape[0]

        critic_idx = self._switch_option_idx if self._switch_option_idx is not None else 0
        critic = self.ext_critics[critic_idx]
        critic_optim = self.ext_critic_optim[critic_idx]

        with torch.no_grad():
            values = critic(states)
            advantages, returns = estimate_advantages(
                rewards,
                terminations,
                truncations,
                values,
                gamma=self.gamma,
                gae=self.gae,
            )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Standard PG surrogate (loss form so its gradient is -∇J).
        actor_loss = self.actor_loss(self.actor, states, actions, advantages)
        gradients = torch.autograd.grad(
            actor_loss, tuple(self.actor.parameters()), allow_unused=True
        )
        gradients = tuple(
            g if g is not None else torch.zeros_like(p)
            for p, g in zip(self.actor.parameters(), gradients)
        )

        # Reuse IRPO's learn_base_policy (TRPO branch) for the constrained step.
        prev_update_type = self.base_policy_update_type
        self.base_policy_update_type = "trpo"
        backtrack_iter, backtrack_success = self.learn_base_policy(
            states=batch["states"], grads=gradients
        )
        self.base_policy_update_type = prev_update_type

        # Refresh the value function on the new returns to avoid drift.
        critic_loss_value = float("nan")
        batch_size = states.shape[0]
        if batch_size > 0:
            num_minibatches = 4
            mb_size = max(1, batch_size // num_minibatches)
            losses = []
            for _ in range(5):
                perm = torch.randperm(batch_size)
                for start in range(0, batch_size, mb_size):
                    idx = perm[start : start + mb_size]
                    loss = self.critic_loss(critic, states[idx], returns[idx])
                    critic_optim.zero_grad()
                    loss.backward()
                    critic_optim.step()
                    losses.append(loss.item())
            critic_loss_value = float(np.mean(losses)) if losses else float("nan")

        elapsed = time.time() - t0

        loss_dict = {
            f"{self.name}/loss/actor_loss": actor_loss.item(),
            f"{self.name}/loss/critic_loss": critic_loss_value,
            f"{self.name}/analytics/Backtrack_iter": backtrack_iter,
            f"{self.name}/analytics/Backtrack_success": float(backtrack_success),
            f"{self.name}/analytics/switched": 1.0,
            f"{self.name}/analytics/switch_option_idx": float(
                self._switch_option_idx if self._switch_option_idx is not None else -1
            ),
            f"{self.name}/time_profile/total_sec": elapsed,
            f"{self.name}/time_profile/sample_sec": sample_time,
        }
        return {
            "loss_dict": loss_dict,
            "timesteps": timesteps,
            "supp_dict": {},
        }

    # ------------------------------------------------------------------ #
    # Forward (post-switch the base policy is canonical)
    # ------------------------------------------------------------------ #
    def forward(self, state: np.ndarray, deterministic: bool = False, **kwargs):
        if self._switched:
            state = self.preprocess_state(state)
            a, metaData = self.actor(state, deterministic=deterministic)
            return a, {
                "probs": metaData["probs"],
                "logprobs": metaData["logprobs"],
                "entropy": metaData["entropy"],
                "dist": metaData["dist"],
            }
        return super().forward(state, deterministic=deterministic, **kwargs)

    # ------------------------------------------------------------------ #
    # Top-level dispatch
    # ------------------------------------------------------------------ #
    def learn(self, env, sampler, seed, learning_progress, **kwargs):
        if learning_progress > self.trpo_switch_progress:
            self._maybe_switch()
            return self._trpo_step(env, sampler, seed)
        return super().learn(env, sampler, seed, learning_progress, **kwargs)
