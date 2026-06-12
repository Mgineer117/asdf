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


class TRPO_Learner(Base):
    def __init__(
        self,
        actor: PPO_Actor,
        critic: PPO_Critic,
        is_discrete: bool,
        nupdates: int,
        lr: float = 1e-3,
        batch_size: int = 8,
        entropy_scaler: float = 1e-3,
        l2_reg: float = 1e-8,
        target_kl: float = 0.03,
        damping: float = 1e-3,
        backtrack_iters: int = 10,
        backtrack_coeff: float = 0.8,
        gamma: float = 0.99,
        gae: float = 0.9,
        pos_idx: list = None,
        goal_idx: list = None,
        grad_batch_size: int = 512,
        device: str = "cpu",
    ):
        super().__init__(device=device)

        # constants
        self.name = "TRPO"
        self.device = device

        self.entropy_scaler = entropy_scaler
        self.batch_size = batch_size
        self.damping = damping
        self.gamma = gamma
        self.gae = gae
        self.l2_reg = l2_reg
        self.backtrack_iters = backtrack_iters
        self.backtrack_coeff = backtrack_coeff
        self.nupdates = nupdates
        self.grad_batch_size = grad_batch_size

        self.init_target_kl = target_kl
        self.target_kl = target_kl

        # trainable networks
        self.actor = actor
        self.critic = critic
        self.setup_obs_rms(actor.input_shape, pos_idx=pos_idx, goal_idx=goal_idx)
        self.sync_obs_rms_to(self.actor, self.critic)

        self.optimizer = torch.optim.Adam(params=self.critic.parameters(), lr=lr)

        #
        self.steps = 0
        self.to(self.dtype).to(self.device)

    def lr_scheduler(self):
        self.target_kl = self.init_target_kl * (1 - self.steps / self.nupdates)
        self.steps += 1

    def forward(self, state: np.ndarray, deterministic: bool = False, **kwargs):
        state = self.preprocess_state(state)
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
        batch, sample_time = sampler.collect_samples(env, self.actor, seed)

        t0 = time.time()

        # Ingredients: Convert batch data to tensors (kept on CPU to save VRAM)
        states = torch.as_tensor(batch["states"], dtype=self.dtype, device="cpu")
        actions = torch.as_tensor(batch["actions"], dtype=self.dtype, device="cpu")
        rewards = torch.as_tensor(batch["rewards"], dtype=self.dtype, device="cpu")
        terminations = torch.as_tensor(batch["terminations"], dtype=self.dtype, device="cpu")
        truncations = torch.as_tensor(batch["truncations"], dtype=self.dtype, device="cpu")
        old_logprobs = torch.as_tensor(batch["logprobs"], dtype=self.dtype, device="cpu")

        timesteps = states.shape[0]

        # Compute advantages and returns in chunks to prevent full-batch OOM
        with torch.no_grad():
            values = []
            chunk_size = 512
            for start in range(0, timesteps, chunk_size):
                end = min(start + chunk_size, timesteps)
                mb_states = self.preprocess_state(states[start:end])
                mb_values = self.critic(mb_states)
                values.append(mb_values.cpu())
            values = torch.cat(values, dim=0)

            advantages, returns = estimate_advantages(
                rewards,
                terminations,
                truncations,
                values,
                gamma=self.gamma,
                gae=self.gae,
            )

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        from utils.rl import average_gradients_across_minibatches, average_loss_across_minibatches
        
        def actor_loss_fn(s, a, adv):
            s = self.preprocess_state(s)
            a = a.to(self.device)
            adv = adv.to(self.device)
            return self.actor_loss(s, a, adv)

        actor_gradients = average_gradients_across_minibatches(
            self.actor,
            actor_loss_fn,
            states,
            actions,
            advantages,
            self.grad_batch_size,
            create_graph=False,
        )

        with torch.no_grad():
            actor_loss = average_loss_across_minibatches(
                actor_loss_fn, states, actions, advantages, self.grad_batch_size
            )

        # === actor trpo update === #
        old_actor = deepcopy(self.actor)

        grad_flat = torch.cat([g.view(-1) for g in actor_gradients]).detach()

        B = states.shape[0]
        if B > self.grad_batch_size:
            idx = torch.randperm(B)[: self.grad_batch_size]
            trpo_states = self.preprocess_state(states[idx])
            trpo_actions = actions[idx].to(self.device)
            trpo_advantages = advantages[idx].to(self.device)
            trpo_old_logprobs = old_logprobs[idx].to(self.device)
        else:
            trpo_states = self.preprocess_state(states)
            trpo_actions = actions.to(self.device)
            trpo_advantages = advantages.to(self.device)
            trpo_old_logprobs = old_logprobs.to(self.device)

        self.update_obs_rms(trpo_states)
        self.sync_obs_rms_to(self.actor, self.critic)

        # KL function (closure)
        def kl_fn():
            return compute_kl(old_actor, self.actor, trpo_states)

        # Define HVP function
        Hv = lambda v: hessian_vector_product(kl_fn, self.actor, self.damping, v)

        # Compute step direction with CG
        step_dir = conjugate_gradients(Hv, grad_flat, nsteps=10)

        # Compute step size to satisfy KL constraint
        sAs = 0.5 * torch.dot(step_dir, Hv(step_dir))
        lm = torch.sqrt(sAs / self.target_kl)
        full_step = step_dir / (lm + 1e-8)

        old_loss = actor_loss.item()

        with torch.no_grad():
            old_params = flat_params(old_actor)
            success = False

            for i in range(self.backtrack_iters):
                step_frac = self.backtrack_coeff**i
                new_params = old_params - step_frac * full_step
                set_flat_params(self.actor, new_params)

                # Use the new variance-reduced kl_fn() for backtracking constraint checks
                kl = kl_fn()

                # === INLINE LOSS CALCULATION ===
                _, metaData = self.actor(trpo_states)
                new_logprobs = self.actor.log_prob(metaData["dist"], trpo_actions)
                new_entropy = self.actor.entropy(metaData["dist"])

                ratios = torch.exp(new_logprobs - trpo_old_logprobs)

                # Surrogate objective minus entropy bonus
                surrogate = -(ratios * trpo_advantages).mean()
                entropy_bonus = self.entropy_scaler * new_entropy.mean()
                new_loss = surrogate - entropy_bonus
                # ===============================

                # TRPO requires BOTH constraints to be met
                if kl <= self.target_kl and new_loss < old_loss:
                    success = True
                    break

            if not success:
                set_flat_params(self.actor, old_params)

        # === critic update === #
        # Use averaged loss across full batch for better gradient estimates
        from utils.rl import average_loss_across_minibatches

        critic_epochs = 5
        grad_dict_list = []

        for _ in range(critic_epochs):
            def critic_loss_fn(s, _, r):
                s = self.preprocess_state(s)
                r = r.to(self.device)
                return self.critic_loss(s, r)

            avg_loss = average_loss_across_minibatches(
                critic_loss_fn, states, None, returns, self.grad_batch_size
            )
            value_loss = avg_loss
            loss = avg_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
            grad_dict = self.compute_gradient_norm(
                [self.critic],
                ["critic"],
                dir=f"{self.name}",
                device=self.device,
            )
            grad_dict_list.append(grad_dict)
            self.optimizer.step()
        grad_dict = self.average_dict_values(grad_dict_list)

        # Logging
        loss_dict = {
            f"{self.name}/loss/loss": loss.item(),
            f"{self.name}/loss/actor_loss": actor_loss.item(),
            f"{self.name}/loss/value_loss": value_loss.item(),
            f"{self.name}/analytics/backtrack_iter": i,
            f"{self.name}/analytics/backtrack_success": int(success),
            f"{self.name}/analytics/klDivergence": kl.item(),
            f"{self.name}/analytics/avg_rewards": torch.mean(rewards).item(),
            f"{self.name}/analytics/target_kl": self.target_kl,
            f"{self.name}/analytics/critic_lr": self.optimizer.param_groups[0]["lr"],
            f"{self.name}/grad/actor": torch.linalg.norm(grad_flat).item(),
            f"{self.name}/analytics/step_norm": torch.linalg.norm(
                step_frac * full_step
            ).item(),
            f"{self.name}/analytics/sample_time": sample_time,
            f"{self.name}/analytics/update_time": time.time() - t0,
        }
        norm_dict = self.compute_weight_norm(
            [self.actor, self.critic],
            ["actor", "critic"],
            dir=f"{self.name}",
            device=self.device,
        )
        loss_dict.update(norm_dict)
        loss_dict.update(grad_dict)

        # Cleanup
        self.eval()

        # reduce target_kl for next iteration
        # self.lr_scheduler()

        return {"loss_dict": loss_dict, "timesteps": timesteps}

    def actor_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
    ):
        _, metaData = self.actor(states)
        logprobs = self.actor.log_prob(metaData["dist"], actions)
        entropy = self.actor.entropy(metaData["dist"])

        # surrogate loss
        actor_loss = -(logprobs * advantages).mean()
        entropy_loss = self.entropy_scaler * entropy.mean()

        loss = actor_loss - entropy_loss

        return loss

    def critic_loss(self, states: torch.Tensor, returns: torch.Tensor):
        mb_values = self.critic(states)
        value_loss = self.mse_loss(mb_values, returns)

        return value_loss
