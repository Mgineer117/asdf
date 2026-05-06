"""SB3 SAC bridge for the repo.

This module exposes `SB3_SAC_Algorithm` that constructs an SB3 `SAC`
model, pre-fills its replay buffer using the repo `OnlineSampler`, and
then runs SB3's learning loop. SB3 internal logging is disabled (verbose=0)
so your repo's wandb/tensorboard logging remains primary.
"""

import math
from typing import Optional

from gymnasium import spaces

try:
    from stable_baselines3.sac import SAC
except Exception:  # pragma: no cover - optional dependency
    SAC = None

from utils.sampler import OnlineSampler
from wrappers.env_wrapper import wrap_env_for_sb3
from wrappers.sb3_ppo_wrapper import SB3PolicyAdapter


class SB3_SAC_Algorithm:
    def __init__(self, env, logger, writer, args):
        self.env = env
        self.logger = logger
        self.writer = writer
        self.args = args

    def begin_training(self):
        if SAC is None:
            raise ImportError("stable-baselines3 is required for SB3 SAC. Install via: pip install stable-baselines3")

        # Wrap environment conservatively for SB3
        env = wrap_env_for_sb3(self.env, self.args)
        policy_class = "MultiInputPolicy" if isinstance(env.observation_space, spaces.Dict) else "MlpPolicy"

        # Create SB3 SAC model with SB3 logging disabled
        model = SAC(
            policy_class,
            env,
            verbose=0,
            tensorboard_log=None,
            batch_size=getattr(self.args, "batch_size", 256),
            buffer_size=int(getattr(self.args, "replay_buffer_size", 100000)),
        )

        # Build repo sampler to prefill replay buffer
        sampler = OnlineSampler(
            state_dim=self.args.state_dim,
            action_dim=self.args.action_dim,
            episode_len=self.args.episode_len,
            batch_size=int(getattr(self.args, "prefill_batch", self.args.minibatch_size * self.args.num_minibatch)),
        )

        # Prefill replay buffer with a few batches (best-effort)
        target_prefill = min(model.replay_buffer.buffer_size, int(getattr(self.args, "prefill_steps", sampler.batch_size * 4)))
        try:
            current = model.replay_buffer.size()
        except Exception:
            current = 0

        seed = getattr(self.args, "seed", None)
        while current < target_prefill:
            # Wrap SB3 policy to expose repo's to_device interface
            adapted_policy = SB3PolicyAdapter(model.policy)
            batch, _ = sampler.collect_samples(env, adapted_policy, seed)
            states = batch["states"]
            next_states = batch.get("next_states")
            actions = batch["actions"]
            rewards = batch["rewards"]
            terminations = batch.get("terminations")
            truncations = batch.get("truncations")

            n = states.shape[0]
            for i in range(n):
                obs = states[i]
                nxt = next_states[i] if next_states is not None else None
                act = actions[i]
                rew = float(rewards[i]) if hasattr(rewards[i], "__float__") else float(rewards[i][0])
                done = bool((terminations is not None and terminations[i]) or (truncations is not None and truncations[i]))

                # SB3 ReplayBuffer.add signature varies; try common one
                try:
                    model.replay_buffer.add(obs, nxt, act, rew, done, {})
                except TypeError:
                    try:
                        model.replay_buffer.add(obs, act, rew, done, {})
                    except Exception:
                        # If add fails, break out — best-effort prefill
                        break

            try:
                current = model.replay_buffer.size()
            except Exception:
                current += n

        # Run SB3 learning loop (SB3 will use its own env interactions thereafter)
        total_timesteps = int(self.args.timesteps)
        model.learn(total_timesteps=total_timesteps)

        return 0
