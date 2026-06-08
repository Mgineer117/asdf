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


class PSNE_Learner(Base):
    def __init__(
        self,
        actor: PPO_Actor,
        critic: PPO_Critic,
        is_discrete: bool,
        states: np.ndarray,
        nupdates: int,
        lr: float = 1e-3,
        entropy_scaler: float = 1e-3,
        batch_size: int = 8,
        l2_reg: float = 1e-8,
        target_kl: float = 0.03,
        damping: float = 1e-1,
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
        self.name = "PSNE"
        self.device = device

        self.states = self.preprocess_state(states)

        self.entropy_scaler = entropy_scaler
        self.batch_size = batch_size
        self.damping = damping
        self.gamma = gamma
        self.gae = gae
        self.l2_reg = l2_reg
        self.init_target_kl = target_kl
        self.target_kl = target_kl
        self.backtrack_iters = backtrack_iters
        self.backtrack_coeff = backtrack_coeff
        self.nupdates = nupdates
        self.grad_batch_size = grad_batch_size

        # Adaptive noise scale (Plappert et al., 2017 — arXiv:1706.01905)
        self.sigma = 0.1            # initial noise standard deviation
        self.sigma_alpha = 1.01     # adaptation factor

        # trainable networks
        self.actor = actor
        self.sampled_actor = deepcopy(self.actor)

        self.critic = critic
        self.setup_obs_rms(actor.input_shape, pos_idx=pos_idx, goal_idx=goal_idx)
        self.update_obs_rms(self.states)
        self.sync_obs_rms_to(self.actor, self.sampled_actor, self.critic)
        self.optimizer = torch.optim.Adam(params=self.critic.parameters(), lr=lr)

        #
        self.steps = 0
        self.to(self.dtype).to(self.device)
        self.sample_policy()

    def lr_scheduler(self):
        self.target_kl = self.init_target_kl * (1 - self.steps / self.nupdates)
        self.steps += 1

    def sample_policy(self):
        """Perturb actor params: θ̃ = θ + N(0, σ²I), then adapt σ.

        Implements the adaptive noise scaling from Plappert et al. (2017):
        σ is increased (×α) when KL < δ and decreased (÷α) when KL ≥ δ,
        ensuring the perturbation stays in a useful exploration regime.
        """
        with torch.no_grad():
            clean_params = flat_params(self.actor)
            epsilon = torch.randn_like(clean_params)
            perturbed_params = clean_params + self.sigma * epsilon
            set_flat_params(self.sampled_actor, perturbed_params)

            # Adapt σ based on realized KL divergence
            kl = compute_kl(self.actor, self.sampled_actor, self.states)
            if kl < self.target_kl:
                self.sigma *= self.sigma_alpha   # noise too small → increase
            else:
                self.sigma /= self.sigma_alpha   # noise too large → decrease

    def forward(self, state: np.ndarray, deterministic: bool = False, **kwargs):
        state = self.preprocess_state(state)
        a, metaData = self.sampled_actor(state, deterministic=deterministic)

        return a, {
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
            "entropy": metaData["entropy"],
            "dist": metaData["dist"],
        }

    def learn(self, env, sampler, seed, **kwargs):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()

        # Collect data with the PERTURBED policy for exploration (paper §3)
        batch, sample_time = sampler.collect_samples(env, self.sampled_actor, seed)

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
        else:
            trpo_states = self.preprocess_state(states)

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

        # Apply update
        with torch.no_grad():
            old_params = flat_params(self.actor)

            # Backtracking line search
            success = False
            for i in range(self.backtrack_iters):
                step_frac = self.backtrack_coeff**i
                new_params = old_params - step_frac * full_step
                set_flat_params(self.actor, new_params)
                kl = compute_kl(old_actor, self.actor, trpo_states)

                if kl <= self.target_kl:
                    success = True
                    break

            if not success:
                set_flat_params(self.actor, old_params)

        # self.lr_scheduler()
        # given the update of new actor, sample a actor for exploration
        self.sample_policy()
        self.states = states

        # === critic update === #
        # Use averaged loss across full batch for better gradient estimates
        from utils.rl import average_loss_across_minibatches

        critic_epochs = 5
        grad_dict_list = []

        for _ in range(critic_epochs):
            def critic_loss_fn(s, _, r):
                s = self.preprocess_state(s)
                r = r.to(self.device)
                value_loss, l2_loss = self.critic_loss(s, r)
                return value_loss + l2_loss

            avg_loss = average_loss_across_minibatches(
                critic_loss_fn, states, None, returns, self.grad_batch_size
            )
            l2_loss = sum(param.pow(2).sum() for param in self.critic.parameters()) * self.l2_reg
            value_loss = avg_loss - l2_loss
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
            f"{self.name}/loss/l2_loss": l2_loss.item(),
            f"{self.name}/grad/actor": torch.linalg.norm(grad_flat).item(),
            f"{self.name}/analytics/backtrack_iter": i,
            f"{self.name}/analytics/backtrack_success": int(success),
            f"{self.name}/analytics/klDivergence": kl.item(),
            f"{self.name}/analytics/avg_rewards": torch.mean(rewards).item(),
            f"{self.name}/analytics/target_kl": self.target_kl,
            f"{self.name}/analytics/sigma": self.sigma,
            f"{self.name}/analytics/critic_lr": self.optimizer.param_groups[0]["lr"],
            f"{self.name}/analytics/sample_time": sample_time,
            f"{self.name}/analytics/update_time": time.time() - t0,
        }
        norm_dict = self.compute_weight_norm(
            [self.sampled_actor, self.critic],
            ["actor", "critic"],
            dir=f"{self.name}",
            device=self.device,
        )
        loss_dict.update(norm_dict)
        loss_dict.update(grad_dict)

        # Cleanup
        self.eval()

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
        l2_loss = (
            sum(param.pow(2).sum() for param in self.critic.parameters()) * self.l2_reg
        )

        return value_loss, l2_loss
