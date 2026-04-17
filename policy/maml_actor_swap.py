import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

from policy.layers.base import Base
from policy.layers.ppo_networks import PPO_Actor, PPO_Critic
from utils.intrinsic_rewards import BaseIntRewardFunctions
from utils.rl import *
from utils.sampler import OnlineSampler


class MAML_AS_Learner(Base):

    def __init__(
        self,
        actor: PPO_Actor,
        critic: PPO_Critic,
        intrinsic_reward_fn: BaseIntRewardFunctions,
        timesteps: int,
        num_exp_updates: int,
        base_policy_update_type: str = "sgd",
        lr: float = 3e-4,
        entropy_scaler: float = 1e-3,
        target_kl: float = 0.03,
        l2_reg: float = 1e-8,
        gamma: float = 0.99,
        gae: float = 0.95,
        device: str = "cpu",
    ):
        super().__init__(device=device)

        self.name = "MAML"
        self.device = device

        # Policy and Environment parameters
        self.state_dim = actor.state_dim
        self.action_dim = actor.action_dim
        self.actor_hidden_dim = actor.hidden_dim
        self.critic_hidden_dim = critic.hidden_dim
        self.is_discrete = actor.is_discrete
        self.timesteps = timesteps

        # MAML Optimization parameters
        self.base_policy_update_type = base_policy_update_type
        self.num_exp_updates = num_exp_updates
        assert self.num_exp_updates >= 2, "num_exp_updates must be at least 2"

        # Learning rates
        self.lr = lr

        # Hyperparameters
        self.entropy_scaler = entropy_scaler
        self.gamma = gamma
        self.gae = gae
        self.target_kl = target_kl  # Constraint for TRPO
        self.l2_reg = l2_reg

        # Neural Networks
        self.actor = actor  # The "Base" policy
        self.intrinsic_reward_fn = intrinsic_reward_fn
        self.num_options = self.intrinsic_reward_fn.num_rewards

        # Critics
        self.critics = nn.ModuleList(
            [deepcopy(critic) for _ in range(self.num_options)]
        )

        # Optimizers for the critics
        self.critic_optim = [
            torch.optim.Adam(critic.parameters(), lr=self.lr) for critic in self.critics
        ]

        # Tracking contributing (high reward-inducing) int rewards
        self.policy_indices = [i for i in range(self.num_options)]

        self.wall_clock_time = 0
        self.to(self.dtype).to(self.device)

    def forward(self, state: np.ndarray, deterministic: bool = False):
        state = self.preprocess_state(state)

        actor = self.final_exp_policies[torch.argmax(self.perf_gains)]
        a, metaData = actor(state, deterministic=deterministic)

        return a, {
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
            "entropy": metaData["entropy"],
            "dist": metaData["dist"],
        }

    def init_exp_policies(self):
        """
        Initializes the exploratory policies for each intrinsic reward type by cloning the base actor.
        """
        policy_dict = {}
        for i in range(self.num_options):
            actor_idx = f"{i}_{0}"
            policy_dict[actor_idx] = deepcopy(self.actor)
        return policy_dict

    def backprop(self, policy_dict: dict, gradient_dict: dict, option_idx: int):
        grads = gradient_dict[f"{option_idx}_{self.num_exp_updates - 1}"]
        for j in reversed(range(self.num_exp_updates - 1)):
            iter_idx = f"{option_idx}_{j}"
            Hv = grad(
                gradient_dict[iter_idx],
                policy_dict[iter_idx].parameters(),
                grad_outputs=grads,
            )
            grads = tuple(g - self.lr * h for g, h in zip(grads, Hv))
        return grads

    def learn(
        self, env, sampler: OnlineSampler, seed: int, learning_progress: float, **kwargs
    ):
        self.train()

        t_start_total = time.time()  # Start total timer

        # Initialize time tracking variables
        total_timesteps, total_sample_time = 0, 0
        total_exp_update_time, total_backprop_time, total_base_update_time = 0, 0, 0

        policy_dict, gradient_dict = self.init_exp_policies(), {}

        # Collect initial data with the base policy
        init_batch, init_sample_time = sampler.collect_samples(env, self.actor, seed)
        total_sample_time += init_sample_time  # Track initial sample
        # self.actor.record_state_visitations(init_batch["states"], alpha=1.0)

        total_timesteps += init_batch["states"].shape[0]

        loss_dict_list = []
        # EXPLORATORY PHASE: Iterate over each intrinsic reward type
        for j in range(self.num_exp_updates):
            flag = j == self.num_exp_updates - 1
            actor_indices = [f"{i}_{j}" for i in self.policy_indices]
            future_actor_indices = [f"{i}_{j+1}" for i in self.policy_indices]
            actors = [policy_dict[actor_idx] for actor_idx in actor_indices]

            if j == 0:
                init_batch["rewards"] = self.intrinsic_reward_fn(
                    init_batch["states"], init_batch["next_states"]
                )
                batches = [init_batch for _ in self.policy_indices]
                timesteps = 0
            else:
                batches, current_sample_time = sampler.collect_samples(
                    env, actors, seed
                )
                total_sample_time += current_sample_time

                # add intrinsic reward to batches
                for b in batches:
                    b["rewards"] = self.intrinsic_reward_fn(
                        b["states"], b["next_states"]
                    )
                timesteps = sum(batch["states"].shape[0] for batch in batches)
                total_timesteps += timesteps

            for i in range(len(self.policy_indices)):
                actor, batch = actors[i], batches[i]
                actor_idx, future_actor_idx = actor_indices[i], future_actor_indices[i]

                # Perform Gradient Descent Step (Exploratory Update)
                exp_dict = self.learn_exploratory_policy(actor, batch, i, flag)

                loss_dict_list.append(exp_dict["loss_dict"])
                gradient_dict[actor_idx] = exp_dict["gradients"]
                policy_dict[future_actor_idx] = exp_dict["updated_actor"]
                total_exp_update_time += exp_dict["update_time"]

                beta = 0.95
                self.perf_gains[i] = (
                    beta * self.perf_gains[i] + (1 - beta) * exp_dict["ext_return"]
                )

        # BACKPROP & AGGREGATION
        t_backprop_start = time.time()

        outer_gradients = [
            self.backprop(policy_dict, gradient_dict, i) for i in self.policy_indices
        ]

        outer_gradients_transposed = zip(*outer_gradients)
        gradients = tuple(
            torch.stack(grads_per_param).mean(dim=0)
            for grads_per_param in outer_gradients_transposed
        )

        total_backprop_time = time.time() - t_backprop_start  # Track backprop time

        # BASE POLICY UPDATE
        t_base_start = time.time()
        backtrack_iter, backtrack_success = self.learn_base_policy(
            states=init_batch["states"],
            grads=gradients,
        )
        total_base_update_time = time.time() - t_base_start  # Track base update time

        # CALCULATE TOTAL TIME AND PERCENTAGES
        total_n_irpo_time = time.time() - t_start_total
        self.wall_clock_time += total_n_irpo_time

        # Dictionary construction for logger
        loss_dict = self.average_dict_values(loss_dict_list)
        loss_dict[f"{self.name}/analytics/avg_ext_returns"] = (
            self.perf_gains.mean().item()
        )
        loss_dict[f"{self.name}/analytics/max_ext_returns"] = (
            self.perf_gains.max().item()
        )
        loss_dict[f"{self.name}/analytics/wall_clock_time (hr)"] = (
            self.wall_clock_time / 3600.0
        )
        loss_dict[f"{self.name}/analytics/Backtrack_iter"] = backtrack_iter
        loss_dict[f"{self.name}/analytics/Backtrack_success"] = backtrack_success
        loss_dict[f"{self.name}/analytics/target_kl"] = self.target_kl

        # --- LOGGING PROFILE DATA ---
        loss_dict[f"{self.name}/time_profile/total_sec"] = total_n_irpo_time
        loss_dict[f"{self.name}/time_profile/sample_pct"] = (
            total_sample_time / total_n_irpo_time
        )
        loss_dict[f"{self.name}/time_profile/exp_update_pct"] = (
            total_exp_update_time / total_n_irpo_time
        )
        loss_dict[f"{self.name}/time_profile/backprop_pct"] = (
            total_backprop_time / total_n_irpo_time
        )
        loss_dict[f"{self.name}/time_profile/base_update_pct"] = (
            total_base_update_time / total_n_irpo_time
        )

        self.eval()

        return {"loss_dict": loss_dict, "timesteps": total_timesteps}

    def learn_exploratory_policy(
        self, actor: nn.Module, batch: dict, i: int, flag: bool
    ):
        """
        Performs a single exploratory update.
        Calculates intrinsic rewards, updates critics, and performs a differentiable actor update.
        """
        t0 = time.time()

        # Preprocessing data
        states = self.preprocess_state(batch["states"])
        actions = self.preprocess_state(batch["actions"])
        rewards = self.preprocess_state(batch["rewards"][:, i : i + 1])
        terminations = self.preprocess_state(batch["terminations"])

        # Estimate Advantages
        with torch.no_grad():
            values = self.critics[i](states)

            advantages, returns = estimate_advantages(
                rewards,
                terminations,
                values,
                gamma=self.gamma,
                gae=self.gae,
            )

        # Critic Mini-batch Updates
        batch_size = states.shape[0]
        critic_epochs = 5  # Number of passes over the data
        num_minibatches = 4  # Split data into 4 chunks per epoch
        mb_size = max(1, batch_size // num_minibatches)

        critic = self.critics[i]
        optim = self.critic_optim[i]
        losses = []

        # Loop over the dataset multiple times (epochs)
        for _ in range(critic_epochs):
            # Shuffle the data at the start of each epoch for true SGD
            perm = torch.randperm(batch_size)

            # Iterate through the dataset in mini-batches
            for start_idx in range(0, batch_size, mb_size):
                indices = perm[start_idx : start_idx + mb_size]

                # Update Intrinsic Critic
                loss = self.critic_loss(critic, states[indices], returns[indices])

                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
                optim.step()

                losses.append(loss.item())

        # Average the accumulated losses for logging
        critic_loss = sum(losses) / len(losses)

        # 3. Update Actor (Exploratory Policy)
        actor_clone = deepcopy(actor)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        actor_loss = self.actor_loss(actor, states, actions, advantages)

        # Calculate Gradients with create_graph=True
        gradients = torch.autograd.grad(
            actor_loss, actor.parameters(), create_graph=True
        )
        # gradients = self.clip_grad_norm(gradients, max_norm=0.5)

        # Manual SGD update on the clone to keep the graph connected
        with torch.no_grad():
            for p, g in zip(actor_clone.parameters(), gradients):
                p -= self.lr * g

        # If this is the final exploratory step, save this policy as a potential inference policy
        if flag:
            self.final_exp_policies[i] = actor_clone

        loss_dict = {
            f"{self.name}/loss/actor_loss": actor_loss.item(),
            f"{self.name}/loss/critic_loss": critic_loss,
        }

        update_time = time.time() - t0

        return {
            "loss_dict": loss_dict,
            "update_time": update_time,
            "updated_actor": actor_clone,
            "gradients": gradients,
            "return": returns.mean().item(),
        }

    def learn_base_policy(
        self,
        states: np.ndarray,
        grads: tuple[torch.Tensor],
        damping: float = 1e-1,
        backtrack_iters: int = 15,
        backtrack_coeff: float = 0.7,
    ):
        if self.base_policy_update_type == "trpo":
            states = self.preprocess_state(states)
            old_actor = deepcopy(self.actor)

            # Flatten the aggregated gradients
            grad_flat = torch.cat([g.view(-1) for g in grads]).detach()

            # KL divergence closure for Hessian Vector Product
            def kl_fn():
                return compute_kl(old_actor, self.actor, states)

            Hv = lambda v: hessian_vector_product(kl_fn, self.actor, damping, v)

            # Compute search direction (F_inv * g) via CG
            step_dir = conjugate_gradients(Hv, grad_flat, nsteps=10)

            # Compute step size scaling (Lagrange multiplier)
            sAs = 0.5 * torch.dot(step_dir, Hv(step_dir))
            lm = torch.sqrt(sAs / self.target_kl)
            full_step = step_dir / (lm + 1e-8)

            # Line Search
            with torch.no_grad():
                old_params = flat_params(self.actor)
                success = False
                for i in range(backtrack_iters):
                    step_frac = backtrack_coeff**i
                    new_params = old_params - step_frac * full_step
                    set_flat_params(self.actor, new_params)

                    # Verify KL constraint
                    kl = compute_kl(old_actor, self.actor, states)
                    if kl <= self.target_kl:
                        success = True
                        break

                if not success:
                    set_flat_params(self.actor, old_params)

            return i, success

        elif self.base_policy_update_type == "sgd":
            # Simple fallback: vanilla Gradient Descent
            with torch.no_grad():
                for p, g in zip(self.actor.parameters(), grads):
                    p -= self.lr * g
            return 0, True

    def actor_loss(
        self,
        actor: nn.Module,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
    ):
        """Standard Policy Gradient Loss with Entropy Regularization."""
        _, metaData = actor(states)
        logprobs = actor.log_prob(metaData["dist"], actions)

        actor_loss = -(logprobs * advantages).mean()

        return actor_loss

    def critic_loss(
        self, critic: nn.Module, states: torch.Tensor, returns: torch.Tensor
    ):
        """Standard MSE Value Loss."""
        values = critic(states)
        value_loss = self.mse_loss(values, returns)
        return value_loss
