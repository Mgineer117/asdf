import time

import numpy as np
import torch
import torch.nn as nn

from policy.layers.base import Base
from policy.layers.ppo_networks import PPO_Actor, PPO_Critic
from utils.rl import estimate_advantages


class PPO_Learner(Base):
    def __init__(
        self,
        actor: PPO_Actor,
        critic: PPO_Critic,
        is_discrete: bool,
        nupdates: int,
        lr: float = 3e-4,
        num_minibatch: int = 8,
        minibatch_size: int = 256,
        eps_clip: float = 0.2,
        entropy_scaler: float = 1e-3,
        l2_reg: float = 1e-8,
        target_kl: float = 0.03,
        gamma: float = 0.99,
        gae: float = 0.9,
        K: int = 5,
        pos_idx: list = None,
        goal_idx: list = None,
        device=torch.device("cpu"),
    ):
        super().__init__(device=device)

        # constants
        self.name = "PPO"
        self.device = device

        self.num_minibatch = num_minibatch
        self.minibatch_size = minibatch_size
        self._entropy_scaler = entropy_scaler
        self.gamma = gamma
        self.gae = gae
        self.K = K
        self.l2_reg = l2_reg
        self.target_kl = target_kl
        self.eps_clip = eps_clip
        self.nupdates = nupdates

        # trainable networks
        self.actor = actor
        self.critic = critic

        # State normalisation — goal dims are synced to achieved_goal stats after each update
        self.setup_obs_rms(actor.input_shape, pos_idx=pos_idx, goal_idx=goal_idx)

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": lr},
                {"params": self.critic.parameters(), "lr": lr},
            ]
        )
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self.lr_lambda
        )

        #
        self.to(self.dtype).to(self.device)

    def lr_lambda(self, step):
        return 1.0 - float(step) / float(self.nupdates)

    def forward(self, state: np.ndarray, deterministic: bool = False, **kwargs):
        state = self.preprocess_state(state)
        state = self._normalize_obs(state)
        a, metaData = self.actor(state, deterministic=deterministic)

        return a, {
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
            "entropy": metaData["entropy"],
            "dist": metaData["dist"],
        }

    def learn(self, env, sampler, seed, **kwargs):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()

        # Collect initial data with the base policy
        batch, sample_time = sampler.collect_samples(env, self, seed)

        update_time = time.time()

        # Ingredients: Convert batch data to tensors
        states = self.preprocess_state(batch["states"])
        actions = self.preprocess_state(batch["actions"])
        rewards = self.preprocess_state(batch["rewards"])
        terminations = self.preprocess_state(batch["terminations"])
        truncations = self.preprocess_state(batch["truncations"])
        old_logprobs = self.preprocess_state(batch["logprobs"])

        # Update running obs stats from this batch, sync goal dims, then normalise
        if self.obs_rms is not None:
            self.obs_rms.update(states.detach().cpu().numpy())
            self._sync_goal_stats()
        states = self._normalize_obs(states)

        # FOR HRL option pretraining
        int_reward_fn = kwargs.get("intrinsic_reward_fn")
        option_idx = kwargs.get("option_idx")

        if int_reward_fn is not None and option_idx is not None:
            next_states = self.preprocess_state(batch["next_states"])
            rewards = int_reward_fn(states, next_states)[:, option_idx : option_idx + 1]

        # self.record_state_visitations(states)
        timesteps = states.shape[0]

        # Compute advantages and returns
        with torch.no_grad():
            values = self.critic(states)
            advantages, returns = estimate_advantages(
                rewards,
                terminations,
                truncations,
                values,
                gamma=self.gamma,
                gae=self.gae,
            )

        # Mini-batch training
        batch_size = states.size(0)

        # List to track actor loss over minibatches
        losses = []
        actor_losses = []
        value_losses = []
        entropy_losses = []

        clip_fractions = []
        target_kl = []
        grad_dicts = []

        for k in range(self.K):
            for n in range(self.num_minibatch):
                indices = torch.randperm(batch_size)[: self.minibatch_size]
                mb_states, mb_actions = states[indices], actions[indices]
                mb_old_logprobs, mb_returns = old_logprobs[indices], returns[indices]

                # advantages
                mb_advantages = advantages[indices]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                # 1. Critic Loss (with optional regularization)
                value_loss = self.critic_loss(mb_states, mb_returns)
                # Track value loss for logging
                value_losses.append(value_loss.item())

                # 2. actor Loss
                actor_loss, entropy_loss, clip_fraction, kl_div = self.actor_loss(
                    mb_states, mb_actions, mb_old_logprobs, mb_advantages
                )

                # Track actor loss for logging
                actor_losses.append(actor_loss.item())
                entropy_losses.append(entropy_loss.item())
                clip_fractions.append(clip_fraction)
                target_kl.append(kl_div.item())

                if kl_div.item() > self.target_kl:
                    break

                # Total loss
                loss = actor_loss - entropy_loss + 0.5 * value_loss
                losses.append(loss.item())

                # Update critic parameters
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                grad_dict = self.compute_gradient_norm(
                    [self.actor, self.critic],
                    ["actor", "critic"],
                    dir=f"{self.name}",
                    device=self.device,
                )
                grad_dicts.append(grad_dict)
                self.optimizer.step()

            if kl_div.item() > self.target_kl:
                break

        self.lr_scheduler.step()

        update_time = time.time() - update_time

        # Logging
        loss_dict = {
            f"{self.name}/loss/loss": np.mean(losses),
            f"{self.name}/loss/actor_loss": np.mean(actor_losses),
            f"{self.name}/loss/value_loss": np.mean(value_losses),
            f"{self.name}/loss/entropy_loss": np.mean(entropy_losses),
            f"{self.name}/analytics/clip_fraction": np.mean(clip_fractions),
            f"{self.name}/analytics/klDivergence": target_kl[-1],
            f"{self.name}/analytics/K-epoch": k + 1,
            f"{self.name}/analytics/avg_ext_returns": returns.mean().item(),
            f"{self.name}/analytics/policy_lr": self.optimizer.param_groups[0]["lr"],
            f"{self.name}/analytics/critic_lr": self.optimizer.param_groups[1]["lr"],
            f"{self.name}/time_profile/sample_time": sample_time,
            f"{self.name}/time_profile/update_time": update_time,
        }
        grad_dict = self.average_dict_values(grad_dicts)
        norm_dict = self.compute_weight_norm(
            [self.actor, self.critic],
            ["actor", "critic"],
            dir=f"{self.name}",
            device=self.device,
        )
        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        # Cleanup
        del states, actions, rewards, terminations, old_logprobs
        self.eval()

        return {"loss_dict": loss_dict, "timesteps": timesteps}

    def actor_loss(
        self,
        mb_states: torch.Tensor,
        mb_actions: torch.Tensor,
        mb_old_logprobs: torch.Tensor,
        mb_advantages: torch.Tensor,
    ):
        _, metaData = self.actor(mb_states)
        logprobs = self.actor.log_prob(metaData["dist"], mb_actions)
        entropy = self.actor.entropy(metaData["dist"])
        ratios = torch.exp(logprobs - mb_old_logprobs)

        surr1 = ratios * mb_advantages
        surr2 = (
            torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advantages
        )

        actor_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = self._entropy_scaler * entropy.mean()

        # Compute clip fraction (for logging)
        clip_fraction = torch.mean(
            (torch.abs(ratios - 1) > self.eps_clip).float()
        ).item()

        # Check if KL divergence exceeds target KL for early stopping
        kl_div = torch.mean(mb_old_logprobs - logprobs)

        return actor_loss, entropy_loss, clip_fraction, kl_div

    def critic_loss(self, mb_states: torch.Tensor, mb_returns: torch.Tensor):
        mb_values = self.critic(mb_states)
        value_loss = self.mse_loss(mb_values, mb_returns)

        return value_loss
