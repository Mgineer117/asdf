"""SB3 PPO algorithm bridge for the repo.

This module exposes `SB3_PPO_Algorithm` with the same constructor
signature used by other algorithms in `algorithms/` so it can be
selected via `--algo-name sb3_ppo` from `main.py`.

The implementation uses the `adapters.sb3_ppo_adapter.SB3PPOAdapter` to
instantiate an SB3 PPO model and monkeypatch its `collect_rollouts`
implementation to call the adapter's sampler-backed collector. This
lets the repo retain its `OnlineSampler` while using SB3's optimization.
"""

import types

from gymnasium import spaces

from utils.sampler import OnlineSampler
from wrappers.sb3_ppo_wrapper import SB3PPOWrapper as SB3PPOAdapter


class SB3_PPO_Algorithm:
    def __init__(self, env, logger, writer, args):
        self.env = env
        self.logger = logger
        self.writer = writer
        self.args = args

    def begin_training(self):
        policy_class = "MultiInputPolicy" if isinstance(self.env.observation_space, spaces.Dict) else "MlpPolicy"

        # Build a sampler compatible with the repo's sampling API
        sampler = OnlineSampler(
            state_dim=self.args.state_dim,
            action_dim=self.args.action_dim,
            episode_len=self.args.episode_len,
            batch_size=int(self.args.minibatch_size * self.args.num_minibatch),
        )

        # Create an SB3 PPO wrapped instance that will use our sampler.
        # We instantiate the adapter, then monkeypatch the underlying SB3
        # object's collect_rollouts to delegate to the adapter.collect_rollouts.
        adapter = SB3PPOAdapter(policy_class, self.env, verbose=0, sampler=sampler)

        # Monkeypatch the internal SB3 collect_rollouts to call adapter's
        # collector implementation which knows about the repo sampler.
        # Monkeypatch SB3's collect_rollouts with a method that accepts
        # both positional and keyword forms (e.g., n or n_rollout_steps).
        def _collect_rollouts(self, env, callback, rb, *args, **kwargs):
            n = kwargs.get("n_rollout_steps")
            if n is None and len(args) > 0:
                n = args[0]
            return adapter.collect_rollouts(env, callback, rb, n)

        adapter._sb3.collect_rollouts = types.MethodType(_collect_rollouts, adapter._sb3)

        # Run SB3 learning loop
        total_timesteps = int(self.args.timesteps)
        adapter._sb3.learn(total_timesteps=total_timesteps)

        # We don't have a canonical final performance value here; return 0.
        return 0
