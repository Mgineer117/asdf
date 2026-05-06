"""SB3 PPO adapter: reuse Stable-Baselines3 PPO optimizers while using
the repo's sampler and logging.

This module provides `SB3PPOAdapter` which subclasses SB3's PPO and
overrides `collect_rollouts` to pull batches from `sampler.collect_samples`
and push them into SB3's `rollout_buffer`.

Notes:
- The adapter performs best when `stable-baselines3` is installed.
- The adapter tries to be tolerant of SB3 RolloutBuffer API variations.
"""

from typing import Optional

try:
    from stable_baselines3.ppo import PPO as SB3PPO
    from stable_baselines3.common.buffers import RolloutBuffer
except Exception:  # pragma: no cover - optional dependency
    SB3PPO = None
    RolloutBuffer = object

import numpy as np


class SB3PPOAdapter:
    """A thin adapter that wraps SB3's PPO when available.

    Usage:
        adapter = SB3PPOAdapter(policy_class, policy_kwargs={...}, sampler=my_sampler, **sb3_kwargs)
        adapter.learn(total_timesteps=..., env=env)

    The adapter expects `sampler` to implement `collect_samples(env, policy, seed)`
    returning `(batch, sample_time)` where `batch` is a dict containing at
    least `states`, `actions`, `rewards`, `terminations` and `truncations`,
    and optionally `next_states` and `logprobs`.
    """

    def __init__(self, *args, sampler=None, **kwargs):
        if SB3PPO is None:
            raise ImportError(
                "stable-baselines3 is required for SB3PPOAdapter. Install via: pip install stable-baselines3"
            )
        # Create a real SB3 PPO instance and attach sampler
        self._sb3: SB3PPO = SB3PPO(*args, **kwargs)
        self.sampler = sampler

    # Proxy common attributes to the underlying SB3 object
    def __getattr__(self, name):
        return getattr(self._sb3, name)

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        """Use the repo sampler to fill SB3's rollout buffer.

        This method adheres to SB3's expected return: True on success.
        """
        if self.sampler is None:
            raise RuntimeError("SB3PPOAdapter requires a `sampler` instance.")

        # sampler.collect_samples may accept different signatures; try to call
        # with (env, policy, seed) or (env, policy, seed=None).
        try:
            batch, sample_time = self.sampler.collect_samples(env, self._sb3.policy, None)
        except TypeError:
            batch, sample_time = self.sampler.collect_samples(env, self._sb3.policy)

        # Ensure arrays are numpy
        states = np.asarray(batch["states"])
        actions = np.asarray(batch["actions"])
        rewards = np.asarray(batch["rewards"])
        terminations = np.asarray(batch.get("terminations", np.zeros_like(rewards, dtype=bool)))
        truncations = np.asarray(batch.get("truncations", np.zeros_like(rewards, dtype=bool)))
        next_states = np.asarray(batch.get("next_states")) if batch.get("next_states") is not None else None
        logprobs = np.asarray(batch.get("logprobs")) if batch.get("logprobs") is not None else None

        n = states.shape[0]

        # SB3 RolloutBuffer add() signatures differ between versions. Try both.
        for i in range(n):
            obs = states[i]
            act = actions[i]
            rew = float(rewards[i])
            done = bool(terminations[i] or truncations[i])
            nxt = next_states[i] if next_states is not None else None
            lp = float(logprobs[i]) if logprobs is not None else None

            # value and log_prob may be optional for some buffers; set to 0/None
            val = 0.0

            # Try common add signatures
            try:
                # signature: add(obs, action, reward, done, value, log_prob)
                rollout_buffer.add(obs, act, rew, done, val, lp)
            except TypeError:
                try:
                    # signature: add(obs, next_obs, action, reward, done, value, log_prob)
                    rollout_buffer.add(obs, nxt, act, rew, done, val, lp)
                except Exception as e:  # pragma: no cover - defensive
                    raise RuntimeError(f"Unsupported RolloutBuffer.add signature: {e}")

        return True
