"""SB3 PPO wrapper moved into the `wrappers` package.

This re-exports the previous adapter functionality under the
`wrappers` namespace as `SB3PPOWrapper`.
"""

from typing import Optional

try:
    from stable_baselines3.ppo import PPO as SB3PPO
    from stable_baselines3.common.buffers import RolloutBuffer
except Exception:  # pragma: no cover - optional dependency
    SB3PPO = None
    RolloutBuffer = object

import numpy as np


class SB3PolicyAdapter:
    """Adapter that wraps an SB3 policy to expose the repo's `to_device` and `device` interface."""

    def __init__(self, sb3_policy):
        self._sb3_policy = sb3_policy
        self._device = getattr(sb3_policy, "device", "cpu")

    @property
    def device(self):
        return self._device

    def to_device(self, device):
        """Move policy to device, matching repo policy interface."""
        self._device = device
        if hasattr(self._sb3_policy, "to"):
            self._sb3_policy.to(device)

    def __call__(self, obs, deterministic=False, **kwargs):
        """Return a repo-style `(action, metaData)` tuple."""
        try:
            obs_tensor, _ = self._sb3_policy.obs_to_tensor(obs)
        except Exception:
            import torch

            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
            if obs_tensor.ndim == 1:
                obs_tensor = obs_tensor.unsqueeze(0)

        try:
            actions, _, logprob = self._sb3_policy.forward(obs_tensor, deterministic=deterministic)
            distribution = self._sb3_policy.get_distribution(obs_tensor)
            entropy = distribution.entropy()
            if hasattr(entropy, "sum") and entropy.ndim > 1:
                entropy = entropy.sum(dim=1)
        except Exception:
            import torch

            actions, _ = self._sb3_policy.predict(obs, deterministic=deterministic)
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self._device)
            if actions.ndim == 1:
                actions = actions.unsqueeze(0)
            logprob = torch.zeros((actions.shape[0],), dtype=torch.float32, device=self._device)
            entropy = torch.zeros((actions.shape[0],), dtype=torch.float32, device=self._device)

        if not isinstance(actions, __import__("torch").Tensor):
            import torch

            actions = torch.as_tensor(actions, dtype=torch.float32, device=self._device)

        if not isinstance(logprob, __import__("torch").Tensor):
            import torch

            logprob = torch.as_tensor(logprob, dtype=torch.float32, device=self._device)

        if not isinstance(entropy, __import__("torch").Tensor):
            import torch

            entropy = torch.as_tensor(entropy, dtype=torch.float32, device=self._device)

        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        if logprob.ndim == 0:
            logprob = logprob.unsqueeze(0)
        if entropy.ndim == 0:
            entropy = entropy.unsqueeze(0)

        return actions, {"logprobs": logprob, "entropy": entropy}

    def __getattr__(self, name):
        return getattr(self._sb3_policy, name)


class SB3PPOWrapper:
    """Wrapper that instantiates an SB3 PPO and uses the repo sampler
    for data collection via a `collect_rollouts` method compatible with
    SB3's expectations.
    """

    def __init__(self, *args, sampler=None, **kwargs):
        if SB3PPO is None:
            raise ImportError(
                "stable-baselines3 is required for SB3PPOWrapper. Install via: pip install stable-baselines3"
            )
        self._sb3: SB3PPO = SB3PPO(*args, **kwargs)
        self.sampler = sampler

    def __getattr__(self, name):
        return getattr(self._sb3, name)

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        if self.sampler is None:
            raise RuntimeError("SB3PPOWrapper requires a `sampler` instance.")

        # Wrap SB3 policy to expose repo's to_device interface
        adapted_policy = SB3PolicyAdapter(self._sb3.policy)

        try:
            batch, sample_time = self.sampler.collect_samples(env, adapted_policy, None)
        except TypeError:
            batch, sample_time = self.sampler.collect_samples(env, adapted_policy)

        states = np.asarray(batch["states"])
        actions = np.asarray(batch["actions"])
        rewards = np.asarray(batch["rewards"])
        terminations = np.asarray(batch.get("terminations", np.zeros_like(rewards, dtype=bool)))
        truncations = np.asarray(batch.get("truncations", np.zeros_like(rewards, dtype=bool)))
        next_states = np.asarray(batch.get("next_states")) if batch.get("next_states") is not None else None
        logprobs = np.asarray(batch.get("logprobs")) if batch.get("logprobs") is not None else None

        n = states.shape[0]

        for i in range(n):
            obs = states[i]
            act = actions[i]
            rew = float(rewards[i])
            done = bool(terminations[i] or truncations[i])
            nxt = next_states[i] if next_states is not None else None
            lp = float(logprobs[i]) if logprobs is not None else None

            val = 0.0
            try:
                rollout_buffer.add(obs, act, rew, done, val, lp)
            except TypeError:
                try:
                    rollout_buffer.add(obs, nxt, act, rew, done, val, lp)
                except Exception as e:
                    raise RuntimeError(f"Unsupported RolloutBuffer.add signature: {e}")

        return True
