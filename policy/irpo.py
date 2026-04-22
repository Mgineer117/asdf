import time
from copy import deepcopy

import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.distributions import Normal
import matplotlib.pyplot as plt
from policy.layers.base import Base
from policy.layers.ppo_networks import PPO_Actor, PPO_Critic
from utils.intrinsic_rewards import BaseIntRewardFunctions
from utils.rl import *
from utils.sampler import OnlineSampler


NUM_GOALS = {
    "fourroomsG-v1": 4,
    "mazeG-v1": 3,
    "mazeG-v2": 4,
    "mazeG-v3": 4,
    "pointmazeG-v1": 2,
    "pointmazeG-v2": 2,
    "pointmazeG-v3": 3,
    "pointmazeG-v4": 4,
    "antmazeG-v1": 2,
    "antmazeG-v2": 2,
    "antmazeG-v3": 2,
    "antmazeG-v4": 3,
    "antmazeG-v5": 3,
    "antmazeG-v6": 2,
}


class IRPO_Learner(Base):

    def __init__(
        self,
        actor: PPO_Actor,
        critic: PPO_Critic,
        beta: float,
        intrinsic_reward_fn: BaseIntRewardFunctions,
        aggregation_method: str,
        noise_std: float,
        # find_lr: bool,
        num_exp_updates: int,
        base_policy_update_type: str = "trpo",
        lr: float = 3e-4,
        critic_lr: float = 3e-4,
        entropy_scaler: float = 1e-3,
        target_kl: float = 0.03,
        lock_threshold: float = 0.66,
        # base_target_kl: float = 0.001,
        l2_reg: float = 1e-8,
        gamma: float = 0.99,
        gae: float = 0.95,
        anneal_kl: bool = True,
        device: str = "cpu",
    ):
        super().__init__(device=device)

        self.name = "IRPO"
        self.device = device

        # IRPO Optimization parameters
        self.base_policy_update_type = base_policy_update_type
        self.num_exp_updates = num_exp_updates
        assert self.num_exp_updates >= 2, "num_exp_updates must be at least 2"

        # Learning rates
        self.beta = beta
        self.lr = lr
        # self.find_lr = find_lr
        self.critic_lr = critic_lr
        self.noise_std = noise_std

        # Hyperparameters
        self.entropy_scaler = entropy_scaler
        self.gamma = gamma
        self.gae = gae
        self.target_kl = target_kl  # Constraint for TRPO
        self.init_target_kl = target_kl
        self.lock_threshold = lock_threshold
        self.locked_option_idx = None
        # self.base_target_kl = base_target_kl
        self.l2_reg = l2_reg
        self.anneal_kl = anneal_kl

        # Neural Networks
        self.actor = actor  # The "Base" policy
        self.intrinsic_reward_fn = intrinsic_reward_fn
        self.num_options = self.intrinsic_reward_fn.num_rewards

        # Critics
        self.ext_critics = nn.ModuleList(
            [deepcopy(critic) for _ in range(self.num_options)]
        )
        self.int_critics = nn.ModuleList(
            [deepcopy(critic) for _ in range(self.num_options)]
        )

        # Optimizers for the critics
        self.ext_critic_optim = [
            torch.optim.Adam(critic.parameters(), lr=self.critic_lr)
            for critic in self.ext_critics
        ]
        self.int_critic_optim = [
            torch.optim.Adam(critic.parameters(), lr=self.critic_lr)
            for critic in self.int_critics
        ]

        # Tracking contributing (high reward-inducing) int rewards
        self.policy_indices = [i for i in range(self.num_options)]

        # Initialized to 0, updated via EMA.
        self.perf_gains = torch.zeros(self.num_options).to(self.device)
        self.aggregation_method = aggregation_method

        # Storage for the best policies found during exploration (for inference/eval)
        self.final_exp_policies = [deepcopy(actor) for _ in range(self.num_options)]

        self.wall_clock_time = 0
        self.to(self.dtype).to(self.device)

    def anneal_target_kl(self, learning_progress: float):
        # Optional: Anneal target KL over time (e.g., linearly decay)
        self.target_kl = self.init_target_kl * (1.0 - learning_progress) + 1e-5

    def forward(self, state: np.ndarray, deterministic: bool = False, **kwargs):
        state = self.preprocess_state(state)

        best_idx = (
            self.locked_option_idx
            if self.locked_option_idx is not None
            else torch.argmax(self.perf_gains).item()
        )
        actor = self.final_exp_policies[best_idx]
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
                tuple(policy_dict[iter_idx].parameters()),
                grad_outputs=grads,
                allow_unused=True,
            )
            Hv = tuple(
                h if h is not None else torch.zeros_like(p)
                for p, h in zip(policy_dict[iter_idx].parameters(), Hv)
            )
            grads = tuple(g - self.lr * h for g, h in zip(grads, Hv))
        return grads

    def measure_kl_among_exp_policies(self, batch: dict):
        if self.num_options <= 1:
            return 0.0

        # for each policy
        kl_div_list = []
        for i in range(self.num_options):
            actor = self.final_exp_policies[i]
            states = self.preprocess_state(batch["states"])
            with torch.no_grad():
                _, metaData = actor(states)
                dist = metaData["dist"]

            kl_divs = []
            for j in range(self.num_options):
                if j == i:
                    continue
                other_actor = self.final_exp_policies[j]
                with torch.no_grad():
                    _, other_metaData = other_actor(states)
                    other_dist = other_metaData["dist"]
                    kl_div = torch.distributions.kl_divergence(dist, other_dist).mean()
                    kl_divs.append(kl_div.item())
            kl_div_list.append(sum(kl_divs) / len(kl_divs))
        return sum(kl_div_list) / len(kl_div_list)

    def learn(
        self, env, sampler: OnlineSampler, seed: int, learning_progress: float, **kwargs
    ):
        self.train()

        t_start_total = time.time()  # Start total timer

        # Initialize time tracking variables
        total_timesteps, total_sample_time = 0, 0
        total_exp_update_time, total_backprop_time, total_base_update_time = 0, 0, 0

        # --- Option Lock-in (Late stage fine-tuning) ---
        if learning_progress >= self.lock_threshold and self.locked_option_idx is None:
            self.locked_option_idx = torch.argmax(self.perf_gains).item()

        # --- Select active options ---
        if self.locked_option_idx is not None:
            active_indices = [self.locked_option_idx]
        else:
            active_indices = self.policy_indices

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
            actor_indices = [f"{i}_{j}" for i in active_indices]
            future_actor_indices = [f"{i}_{j+1}" for i in active_indices]
            actors = [policy_dict[actor_idx] for actor_idx in actor_indices]

            if j == 0:
                init_batch["int_rewards"] = self.intrinsic_reward_fn(
                    init_batch["states"], init_batch["next_states"]
                )
                batches = [init_batch for _ in active_indices]
                timesteps = 0
            else:
                batches, current_sample_time = sampler.collect_samples(
                    env, actors, seed
                )
                if isinstance(batches, dict):
                    # for num_options = 1 case, sampler may return a single batch instead of a list
                    batches = [batches]

                total_sample_time += current_sample_time

                # add intrinsic reward to batches
                for b in batches:
                    b["int_rewards"] = self.intrinsic_reward_fn(
                        b["states"], b["next_states"]
                    )
                timesteps = sum(batch["states"].shape[0] for batch in batches)
                total_timesteps += timesteps

            for idx, i in enumerate(active_indices):
                actor, batch = actors[idx], batches[idx]
                actor_idx, future_actor_idx = (
                    actor_indices[idx],
                    future_actor_indices[idx],
                )

                # Perform Gradient Descent Step (Exploratory Update)
                exp_dict = self.learn_exploratory_policy(actor, batch, i, flag)

                loss_dict_list.append(exp_dict["loss_dict"])
                gradient_dict[actor_idx] = exp_dict["gradients"]
                policy_dict[future_actor_idx] = exp_dict["updated_actor"]
                # lr_dict[actor_idx] = exp_dict["used_lr"]
                total_exp_update_time += exp_dict["update_time"]

                # Update performance gains only on the fully adapted policy
                if flag:
                    self.perf_gains[i] = (
                        self.beta * self.perf_gains[i]
                        + (1 - self.beta) * exp_dict["ext_return"]
                    )

        # BACKPROP & AGGREGATION
        t_backprop_start = time.time()

        greedy_idx = torch.argmax(self.perf_gains).item()
        if self.locked_option_idx is not None:
            temperature = 1e-8
        else:
            temperature = max(1e-8, 1.0 - (learning_progress / self.lock_threshold))

        if temperature <= 1e-8 or len(active_indices) == 1:
            active_gains = self.perf_gains[active_indices]
            best_local_idx = active_indices[torch.argmax(active_gains).item()]
            gradients = self.backprop(policy_dict, gradient_dict, best_local_idx)
        else:
            active_gains = self.perf_gains[active_indices]
            # REVERSE SOFTMAX: Amplify low return option idx (Lower bound maximization)
            logits = active_gains
            # logits = -active_gains
            # weights = F.softmax(logits / temperature, dim=0)

            if self.aggregation_method == "argmax":
                weights = torch.zeros_like(logits)
                weights[torch.argmax(logits)] = 1.0
            elif self.aggregation_method == "uniform":
                weights = torch.ones_like(logits) / logits.shape[0]
            elif self.aggregation_method == "softmax":
                weights = F.softmax(logits / temperature, dim=0)
            else:
                raise ValueError(
                    f"Invalid aggregation method: {self.aggregation_method}"
                )

            outer_gradients = [
                self.backprop(policy_dict, gradient_dict, i) for i in active_indices
            ]

            outer_gradients_transposed = zip(*outer_gradients)
            gradients = tuple(
                sum(w * g for w, g in zip(weights, grads_per_param))
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

        kl_diff = self.measure_kl_among_exp_policies(init_batch)
        if self.anneal_kl:
            self.anneal_target_kl(learning_progress)

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
        loss_dict[f"{self.name}/analytics/Contributing Option"] = greedy_idx
        loss_dict[f"{self.name}/analytics/Backtrack_iter"] = backtrack_iter
        loss_dict[f"{self.name}/analytics/Backtrack_success"] = backtrack_success
        loss_dict[f"{self.name}/analytics/target_kl"] = self.target_kl
        loss_dict[f"{self.name}/analytics/kl_divergence"] = kl_diff
        loss_dict[f"{self.name}/analytics/temperature"] = temperature

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

        #### for all exploratory policies ####
        supp_dict = {}
        for option_idx, policy in enumerate(self.final_exp_policies):
            state, _ = env.reset(seed=seed)
            frames = []

            img = env.render()
            if img is not None:
                frames.append(img)

            done = False
            max_steps = getattr(env, "max_steps", 1000)
            step_count = 0

            while not done and step_count < max_steps:
                with torch.no_grad():
                    a, _ = policy(state)
                    a = a.cpu().numpy().flatten()

                state, _, term, trunc, _ = env.step(a)
                done = term or trunc
                step_count += 1

                img = env.render()
                if img is not None:
                    frames.append(img)

            if len(frames) > 0:
                # Convert to (T, C, H, W) shape, standard for wandb.Video and TensorBoard
                gif = np.array(frames).transpose(0, 3, 1, 2)
                supp_dict[f"rendering{option_idx}"] = gif

        self.eval()

        return {
            "loss_dict": loss_dict,
            "timesteps": total_timesteps,
            "supp_dict": supp_dict,
        }

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
        ext_rewards = self.preprocess_state(batch["rewards"])
        int_rewards = self.preprocess_state(batch["int_rewards"][:, i : i + 1])
        terminations = self.preprocess_state(batch["terminations"])
        truncations = self.preprocess_state(batch["truncations"])

        # MAKE PLOT (x-axis: states[:, 0], y-axis: int_rewards[:, 0])
        # plt.figure()
        # plt.scatter(states[:, 0], int_rewards[:, 0])
        # plt.xlabel("State")
        # plt.ylabel("Intrinsic Reward")
        # plt.title(f"Intrinsic Reward vs State (Option {i})")
        # plt.show()

        # Estimate Advantages
        with torch.no_grad():
            ext_values = self.ext_critics[i](states)
            int_values = self.int_critics[i](states)

            ext_advantages, ext_returns = estimate_advantages(
                ext_rewards,
                terminations,
                truncations,
                ext_values,
                gamma=self.gamma,
                gae=self.gae,
            )
            int_advantages, int_returns = estimate_advantages(
                int_rewards,
                terminations,
                truncations,
                int_values,
                gamma=self.gamma,
                gae=self.gae,
            )

        # Critic Mini-batch Updates
        batch_size = states.shape[0]
        critic_epochs = 5  # Number of passes over the data
        num_minibatches = 4  # Split data into 4 chunks per epoch
        mb_size = max(1, batch_size // num_minibatches)

        ext_critic = self.ext_critics[i]
        ext_optim = self.ext_critic_optim[i]
        ext_losses = []

        int_critic = self.int_critics[i]
        int_optim = self.int_critic_optim[i]
        int_losses = []

        # Loop over the dataset multiple times (epochs)
        for _ in range(critic_epochs):
            # Shuffle the data at the start of each epoch for true SGD
            perm = torch.randperm(batch_size)

            # Iterate through the dataset in mini-batches
            for start_idx in range(0, batch_size, mb_size):
                indices = perm[start_idx : start_idx + mb_size]

                # 1. Update Extrinsic Critic
                ext_loss = self.critic_loss(
                    ext_critic, states[indices], ext_returns[indices]
                )

                ext_optim.zero_grad()
                ext_loss.backward()
                # nn.utils.clip_grad_norm_(ext_critic.parameters(), max_norm=0.5)
                ext_optim.step()

                ext_losses.append(ext_loss.item())

                # 2. Update Intrinsic Critic
                int_loss = self.critic_loss(
                    int_critic, states[indices], int_returns[indices]
                )

                int_optim.zero_grad()
                int_loss.backward()
                # nn.utils.clip_grad_norm_(int_critic.parameters(), max_norm=0.5)
                int_optim.step()

                int_losses.append(int_loss.item())

        # Average the accumulated losses for logging
        ext_critic_loss = sum(ext_losses) / len(ext_losses)
        int_critic_loss = sum(int_losses) / len(int_losses)

        # 3. Update Actor (Exploratory Policy)
        actor_clone = deepcopy(actor)

        # Select advantage based on whether we are in the 'exploratory' (int)
        # or 'base' (ext) phase of the loop for this specific calculation.
        advantages = ext_advantages if flag else int_advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss = self.actor_loss(actor, states, actions, advantages)

        # Calculate Gradients with create_graph=True
        gradients = torch.autograd.grad(
            actor_loss, tuple(actor.parameters()), create_graph=True, allow_unused=True
        )
        gradients = tuple(
            g if g is not None else torch.zeros_like(p)
            for p, g in zip(actor.parameters(), gradients)
        )
        # gradients = self.clip_grad_norm(gradients, max_norm=0.5)

        with torch.no_grad():
            for p, g in zip(actor_clone.parameters(), gradients):
                p -= self.lr * g

        # If this is the final exploratory step, save this policy as a potential inference policy
        if flag:
            self.final_exp_policies[i] = actor_clone

        # # Update Int Reward Generator (if it has learnable parameters, e.g., DRND)
        # self.int_reward_fn.learn(states, next_states, i, source)

        loss_dict = {
            f"{self.name}/loss/actor_loss": actor_loss.item(),
            f"{self.name}/loss/ext_critic_loss": ext_critic_loss,
            f"{self.name}/loss/int_critic_loss": int_critic_loss,
        }

        update_time = time.time() - t0

        return {
            "loss_dict": loss_dict,
            "update_time": update_time,
            "updated_actor": actor_clone,
            "gradients": gradients,
            # "used_lr": lr,
            "ext_return": ext_returns.mean().item(),
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
            sAs = torch.clamp(sAs, min=1e-8)
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
        entropy = actor.entropy(metaData["dist"])

        actor_loss = -(logprobs * advantages).mean()
        entropy_loss = self.entropy_scaler * entropy.mean()

        return actor_loss - entropy_loss

    def critic_loss(
        self, critic: nn.Module, states: torch.Tensor, returns: torch.Tensor
    ):
        """Standard MSE Value Loss."""
        values = critic(states)
        value_loss = self.mse_loss(values, returns)
        return value_loss


class IRPO_G_Learner(IRPO_Learner):
    """
    Goal-conditioned IRPO: single diffusion-kernel reward, no gradient aggregation.

    Inherits all critic/actor update machinery from IRPO_Learner (num_options=1
    is enforced by the G reward functions setting num_rewards=1). Only learn()
    and forward() are overridden to remove multi-option and aggregation logic.
    """

    def __init__(
        self,
        actor: PPO_Actor,
        critic: PPO_Critic,
        env_name: str,
        beta: float,
        intrinsic_reward_fn: BaseIntRewardFunctions,
        noise_std: float,
        num_exp_updates: int,
        base_policy_update_type: str = "trpo",
        lr: float = 3e-4,
        critic_lr: float = 3e-4,
        entropy_scaler: float = 1e-3,
        target_kl: float = 0.03,
        l2_reg: float = 1e-8,
        gamma: float = 0.99,
        gae: float = 0.95,
        anneal_kl: bool = True,
        device: str = "cpu",
    ):
        super().__init__(
            actor=actor,
            critic=critic,
            beta=beta,
            intrinsic_reward_fn=intrinsic_reward_fn,
            aggregation_method="argmax",  # unused, required by parent signature
            noise_std=noise_std,
            num_exp_updates=num_exp_updates,
            base_policy_update_type=base_policy_update_type,
            lr=lr,
            critic_lr=critic_lr,
            entropy_scaler=entropy_scaler,
            target_kl=target_kl,
            l2_reg=l2_reg,
            gamma=gamma,
            gae=gae,
            anneal_kl=anneal_kl,
            device=device,
        )
        self.name = "IRPO_G"
        self.num_goals = NUM_GOALS.get(env_name, None)
        assert (
            self.num_goals is None
        ), f"Number of goals for environment {env_name} not found in NUM_GOALS dictionary."
        assert self.num_options == 1, (
            f"IRPO_G_Learner requires exactly 1 intrinsic reward (got {self.num_options}). "
            "Use ALLOIntRewardFunctionG or RandomIntRewardFunctionsG."
        )

    def forward(self, state: np.ndarray, deterministic: bool = False, **kwargs):
        state = self.preprocess_state(state)
        a, metaData = self.final_exp_policies[0](state, deterministic=deterministic)
        return a, {
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
            "entropy": metaData["entropy"],
            "dist": metaData["dist"],
        }

    def _goal_gradient_conflict_figure(self, batch, actor, n_clusters=None):
        """
        For each goal cluster in the batch, compute the true advantage-weighted
        policy-gradient direction ∇_θ E[log π(a|s) · Â(s,a)], then measure
        pairwise cosine similarities and angles.
        Returns (img_hwc, metrics_dict) or (None, {}) on failure.
        """
        from sklearn.cluster import KMeans

        goal_idx = getattr(self.intrinsic_reward_fn.args, "goal_idx", None)
        if goal_idx is None:
            return None, {}

        try:
            states = self.preprocess_state(batch["states"])  # (T, D)
            actions = self.preprocess_state(batch["actions"])  # (T, A)
            terminations = self.preprocess_state(batch["terminations"])
            truncations = self.preprocess_state(batch["truncations"])

            # Compute combined advantage (ext + int) from the trained critics.
            # We use the same critics that learn_exploratory_policy uses (index 0).
            with torch.no_grad():
                ext_rewards = self.preprocess_state(batch["rewards"])
                int_rewards = self.preprocess_state(batch["int_rewards"][:, 0:1])

                ext_values = self.ext_critics[0](states)
                int_values = self.int_critics[0](states)

                ext_adv, _ = estimate_advantages(
                    ext_rewards,
                    terminations,
                    truncations,
                    ext_values,
                    gamma=self.gamma,
                    gae=self.gae,
                )
                int_adv, _ = estimate_advantages(
                    int_rewards,
                    terminations,
                    truncations,
                    int_values,
                    gamma=self.gamma,
                    gae=self.gae,
                )
                advantages = (ext_adv + int_adv).detach()  # (T, 1)

            goals_np = states[:, goal_idx].detach().cpu().numpy()  # (T, g)
            env_name = getattr(self.intrinsic_reward_fn.args, "env_name", "")
            n_clusters = NUM_GOALS.get(
                env_name, n_clusters if n_clusters is not None else 6
            )

            kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=0)
            labels = kmeans.fit_predict(goals_np)

            # Per-cluster advantage-weighted policy-gradient vectors
            grad_vecs = []
            valid_clusters = []
            for c in range(n_clusters):
                mask = torch.from_numpy(labels == c).to(states.device)
                if mask.sum() < 4:
                    continue
                mb_states = states[mask].detach()
                mb_actions = actions[mask].detach()
                mb_adv = advantages[mask]
                # Normalize advantages within the cluster for scale consistency
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                probe = deepcopy(actor)
                probe.zero_grad()
                _, meta = probe(mb_states)
                logprobs = probe.log_prob(meta["dist"], mb_actions)  # (N, 1)
                # True policy gradient: -E[log π(a|s) · Â(s,a)]
                loss = -(logprobs * mb_adv).mean()
                loss.backward()

                flat = torch.cat(
                    [
                        p.grad.detach().cpu().flatten()
                        for p in probe.parameters()
                        if p.grad is not None
                    ]
                )
                grad_vecs.append(flat)
                valid_clusters.append(c)
                del probe

            if len(grad_vecs) < 2:
                return None, {}

            n = len(grad_vecs)
            cos_sim = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    cos_sim[i, j] = F.cosine_similarity(
                        grad_vecs[i].unsqueeze(0), grad_vecs[j].unsqueeze(0)
                    ).item()

            angles = np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0)))

            off_diag_mask = ~np.eye(n, dtype=bool)
            off_diag = cos_sim[off_diag_mask]
            conflict_frac = float((off_diag < 0).mean())
            mean_cos = float(off_diag.mean())

            # Figure: two side-by-side heatmaps
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            tick_labels = [f"G{c}" for c in valid_clusters]

            im0 = axes[0].imshow(cos_sim, vmin=-1, vmax=1, cmap="RdBu_r")
            axes[0].set_title(
                f"Cosine Similarity  (conflict={conflict_frac:.2f}, mean={mean_cos:.3f})"
            )
            axes[0].set_xticks(range(n))
            axes[0].set_xticklabels(tick_labels)
            axes[0].set_yticks(range(n))
            axes[0].set_yticklabels(tick_labels)
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            for i in range(n):
                for j in range(n):
                    axes[0].text(
                        j,
                        i,
                        f"{cos_sim[i,j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                    )

            im1 = axes[1].imshow(angles, vmin=0, vmax=180, cmap="hot_r")
            axes[1].set_title("Gradient Angle (°)")
            axes[1].set_xticks(range(n))
            axes[1].set_xticklabels(tick_labels)
            axes[1].set_yticks(range(n))
            axes[1].set_yticklabels(tick_labels)
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            for i in range(n):
                for j in range(n):
                    axes[1].text(
                        j,
                        i,
                        f"{angles[i,j]:.0f}°",
                        ha="center",
                        va="center",
                        fontsize=7,
                    )

            plt.tight_layout()
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
            plt.close(fig)

            metrics = {
                f"{self.name}/grad_conflict/conflict_fraction": conflict_frac,
                f"{self.name}/grad_conflict/mean_cosine_similarity": mean_cos,
            }
            return buf, metrics

        except Exception as e:
            print(f"[WARNING] Goal gradient conflict figure failed: {e}")
            return None, {}

    def measure_kl_among_exp_policies(self, batch: dict):
        """
        For a single option, measure KL divergence between the base policy and
        the single exploratory policy.
        """
        states = self.preprocess_state(batch["states"])
        with torch.no_grad():
            _, base_meta = self.actor(states)
            _, exp_meta = self.final_exp_policies[0](states)
            kl_div = torch.distributions.kl_divergence(
                base_meta["dist"], exp_meta["dist"]
            ).mean()
        return kl_div.item()

    def learn(self, env, sampler: OnlineSampler, seed: int, learning_progress: float):
        self.train()
        t_start = time.time()
        total_timesteps, total_sample_time = 0, 0
        total_exp_update_time, total_backprop_time, total_base_update_time = 0, 0, 0

        policy_dict, gradient_dict = self.init_exp_policies(), {}

        # Initial rollout with the base policy
        init_batch, init_sample_time = sampler.collect_samples(env, self.actor, seed)
        total_sample_time += init_sample_time
        total_timesteps += init_batch["states"].shape[0]

        loss_dict_list = []

        # Multi-step exploratory phase (MAML-style), single option (i=0)
        for j in range(self.num_exp_updates):
            flag = j == self.num_exp_updates - 1
            actor_idx = f"0_{j}"
            future_actor_idx = f"0_{j + 1}"
            actor = policy_dict[actor_idx]

            if j == 0:
                init_batch["int_rewards"] = self.intrinsic_reward_fn(
                    init_batch["states"], init_batch["next_states"]
                )
                batch = init_batch
            else:
                result, current_sample_time = sampler.collect_samples(
                    env, [actor], seed
                )
                batch = result[0] if isinstance(result, list) else result
                total_sample_time += current_sample_time
                batch["int_rewards"] = self.intrinsic_reward_fn(
                    batch["states"], batch["next_states"]
                )
                total_timesteps += batch["states"].shape[0]

            exp_dict = self.learn_exploratory_policy(actor, batch, 0, flag)

            loss_dict_list.append(exp_dict["loss_dict"])
            gradient_dict[actor_idx] = exp_dict["gradients"]
            policy_dict[future_actor_idx] = exp_dict["updated_actor"]
            total_exp_update_time += exp_dict["update_time"]

            if flag:
                self.perf_gains[0] = (
                    self.beta * self.perf_gains[0]
                    + (1 - self.beta) * exp_dict["ext_return"]
                )

        # Single backprop — no aggregation
        t_bp = time.time()
        gradients = self.backprop(policy_dict, gradient_dict, 0)
        total_backprop_time = time.time() - t_bp

        # Base policy update
        t_base = time.time()
        backtrack_iter, backtrack_success = self.learn_base_policy(
            states=init_batch["states"],
            grads=gradients,
        )
        total_base_update_time = time.time() - t_base

        total_time = time.time() - t_start
        self.wall_clock_time += total_time

        kl_diff = self.measure_kl_among_exp_policies(init_batch)
        if self.anneal_kl:
            self.anneal_target_kl(learning_progress)

        loss_dict = self.average_dict_values(loss_dict_list)
        loss_dict[f"{self.name}/analytics/avg_ext_returns"] = (
            self.perf_gains.mean().item()
        )
        loss_dict[f"{self.name}/analytics/wall_clock_time (hr)"] = (
            self.wall_clock_time / 3600.0
        )
        loss_dict[f"{self.name}/analytics/Backtrack_iter"] = backtrack_iter
        loss_dict[f"{self.name}/analytics/Backtrack_success"] = backtrack_success
        loss_dict[f"{self.name}/analytics/target_kl"] = self.target_kl
        loss_dict[f"{self.name}/analytics/kl_divergence"] = kl_diff
        loss_dict[f"{self.name}/time_profile/total_sec"] = total_time
        loss_dict[f"{self.name}/time_profile/sample_pct"] = (
            total_sample_time / total_time
        )
        loss_dict[f"{self.name}/time_profile/exp_update_pct"] = (
            total_exp_update_time / total_time
        )
        loss_dict[f"{self.name}/time_profile/backprop_pct"] = (
            total_backprop_time / total_time
        )
        loss_dict[f"{self.name}/time_profile/base_update_pct"] = (
            total_base_update_time / total_time
        )

        # Gradient conflict across goals
        conflict_img, conflict_metrics = self._goal_gradient_conflict_figure(
            init_batch, self.final_exp_policies[0]
        )
        loss_dict.update(conflict_metrics)

        # Render the single exploratory policy
        image_dict = {}
        if conflict_img is not None:
            image_dict["goal_gradient_conflict"] = conflict_img

        self.eval()
        return {
            "loss_dict": loss_dict,
            "timesteps": total_timesteps,
            "image_dict": image_dict,
        }
