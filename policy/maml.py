import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad

from policy.layers.base import Base
from policy.layers.ppo_networks import PPO_Actor, PPO_Critic
from utils.intrinsic_rewards import BaseIntRewardFunctions
from utils.rl import *
from utils.sampler import OnlineSampler


class MAML_Learner(Base):

    def __init__(
        self,
        actor: PPO_Actor,
        critic: PPO_Critic,
        intrinsic_reward_fn: BaseIntRewardFunctions,
        timesteps: int,
        num_exp_updates: int,
        base_policy_update_type: str = "sgd",
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        entropy_scaler: float = 1e-3,
        target_kl: float = 0.03,
        l2_reg: float = 1e-8,
        gamma: float = 0.99,
        gae: float = 0.95,
        pos_idx: list = None,
        goal_idx: list = None,
        device: str = "cpu",

        grad_batch_size: int = 256,
    ):
        super().__init__(device=device)

        self.name = "MAML"


        # MAML Optimization parameters
        self.base_policy_update_type = base_policy_update_type
        self.num_exp_updates = num_exp_updates
        assert self.num_exp_updates >= 2, "num_exp_updates must be at least 2"

        # Learning rates
        self.lr = actor_lr
        self.critic_lr = critic_lr

        # Hyperparameters
        self.entropy_scaler = entropy_scaler
        self.gamma = gamma
        self.gae = gae
        self.target_kl = target_kl  # Constraint for TRPO
        self.l2_reg = l2_reg

        # Neural Networks
        self.actor = actor  # The "Base" policy
        self.adapted_actor = actor  # Placeholder
        self.critic = critic
        self.intrinsic_reward_fn = intrinsic_reward_fn
        self.num_options = self.intrinsic_reward_fn.num_rewards

        # Critics
        self.critics = nn.ModuleList(
            [self._share_clone(critic) for _ in range(self.num_options)]
        )

        self.critic_optim = [
            torch.optim.Adam(
                critic.parameters(),
                lr=self.critic_lr,
            )
            for critic in self.critics
        ]

        # Tracking contributing (high reward-inducing) int rewards
        self.policy_indices = [i for i in range(self.num_options)]

        # Initialized to 0, updated via EMA.
        self.perf_gains = torch.zeros(self.num_options).to(self.device)
        self.setup_obs_rms(actor.input_shape, pos_idx=pos_idx, goal_idx=goal_idx)
        self.sync_obs_rms_to(self.actor, self.critic, self.critics)

        self.wall_clock_time = 0
        self.grad_batch_size = grad_batch_size
        self.to(self.dtype).to(self.device)



    def _share_clone(self, module: nn.Module) -> nn.Module:
        return deepcopy(module)

    def get_training_state(self) -> dict:
        """Full state for a faithful training resume: network weights (via
        state_dict, incl. obs-rms buffers), per-option EMA perf gains, every
        critic's Adam optimizer moments, and the intrinsic-reward normalizer.
        Saving only state_dict() restarts the critic optimizers / perf-gain EMA
        / reward normalizer from scratch on resume."""
        state = {
            "modules": self.state_dict(),
            "perf_gains": self.perf_gains.detach().cpu(),
            "critic_optim": [o.state_dict() for o in self.critic_optim],
            "wall_clock_time": self.wall_clock_time,
            "target_kl": self.target_kl,
        }
        rms = getattr(self.intrinsic_reward_fn, "reward_rms", None)
        if rms is not None:
            state["reward_rms"] = {"mean": rms.mean, "var": rms.var, "count": rms.count}
        return state

    def load_training_state(self, ckpt) -> None:
        """Inverse of :meth:`get_training_state`; accepts a bare module
        state_dict (legacy checkpoints) as a fallback."""
        if not (isinstance(ckpt, dict) and "modules" in ckpt):
            self.load_state_dict(ckpt, strict=False)  # legacy checkpoint
            return

        self.load_state_dict(ckpt["modules"], strict=False)
        if ckpt.get("perf_gains") is not None:
            self.perf_gains = ckpt["perf_gains"].to(self.device)
        for o, sd in zip(self.critic_optim, ckpt.get("critic_optim", [])):
            o.load_state_dict(sd)
        if ckpt.get("wall_clock_time") is not None:
            self.wall_clock_time = ckpt["wall_clock_time"]
        if ckpt.get("target_kl") is not None:
            self.target_kl = ckpt["target_kl"]
        rms_state = ckpt.get("reward_rms")
        rms = getattr(self.intrinsic_reward_fn, "reward_rms", None)
        if rms_state is not None and rms is not None:
            rms.mean = rms_state["mean"]
            rms.var = rms_state["var"]
            rms.count = rms_state["count"]

    def forward(
        self,
        state: np.ndarray,
        deterministic: bool = False,
        eval_mode: bool = False,
        **kwargs,
    ):
        state = self.preprocess_state(state)

        actor = self.adapted_actor if eval_mode else self.actor
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
            policy_dict[actor_idx] = self._share_clone(self.actor)
        return policy_dict

    def backprop(self, policy_dict: dict, gradient_dict: dict, option_idx: int):
        grads = gradient_dict[f"{option_idx}_{self.num_exp_updates - 1}"]
        for j in reversed(range(self.num_exp_updates - 1)):
            iter_idx = f"{option_idx}_{j}"
            all_params = tuple(policy_dict[iter_idx].parameters())
            all_grad_tensors = gradient_dict[iter_idx]

            active = [
                (k, g) for k, g in enumerate(all_grad_tensors)
                if g.grad_fn is not None
            ]

            if active:
                indices, filtered_outputs = zip(*active)
                filtered_grad_outputs = tuple(grads[k] for k in indices)
                filtered_inputs = tuple(all_params[k] for k in indices)

                Hv_partial = grad(
                    filtered_outputs,
                    filtered_inputs,
                    grad_outputs=filtered_grad_outputs,
                    allow_unused=True,
                )
                hv_map = {
                    idx: (h if h is not None else torch.zeros_like(all_params[idx]))
                    for idx, h in zip(indices, Hv_partial)
                }
                Hv = tuple(
                    hv_map.get(k, torch.zeros_like(p))
                    for k, p in enumerate(all_params)
                )
            else:
                Hv = tuple(torch.zeros_like(p) for p in all_params)

            grads = tuple(g - self.lr * h for g, h in zip(grads, Hv))
        return grads

    def adapt_actor(self, env, sampler: OnlineSampler, seed):
        """
        Performs the inner loop updates to adapt the actor using the exploratory policies and returns the final adapted actor.
        """

        actor = self._share_clone(self.actor)
        optim = torch.optim.Adam(
            [
                {"params": actor.parameters(), "lr": self.lr},
                {"params": self.critic.parameters(), "lr": self.critic_lr},
            ]
        )

        for j in range(self.num_exp_updates):
            self.sync_obs_rms_to(actor, self.critic)
            batch, _ = sampler.collect_samples(env, actor, seed)

            states = torch.as_tensor(batch["states"], dtype=self.dtype, device="cpu")
            actions = torch.as_tensor(batch["actions"], dtype=self.dtype, device="cpu")
            rewards = torch.as_tensor(batch["rewards"], dtype=self.dtype, device="cpu")
            terminations = torch.as_tensor(batch["terminations"], dtype=self.dtype, device="cpu")
            truncations = torch.as_tensor(batch["truncations"], dtype=self.dtype, device="cpu")
            timesteps = states.shape[0]

            with torch.no_grad():
                values = []
                chunk_size = 512
                for start in range(0, timesteps, chunk_size):
                    end = min(start + chunk_size, timesteps)
                    mb_states = self.preprocess_state(states[start:end])
                    values.append(self.critic(mb_states).cpu())
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

            optim.zero_grad()
            mb_size = self.grad_batch_size
            B = states.shape[0]

            device_type = "cuda" if "cuda" in str(self.device) else "cpu"
            use_amp = (device_type == "cuda")

            for start in range(0, B, mb_size):
                end = min(start + mb_size, B)
                weight = (end - start) / B
                mb_states = self.preprocess_state(states[start:end])
                self.update_obs_rms(mb_states)
                self.sync_obs_rms_to(actor, self.critic)
                mb_actions = actions[start:end].to(self.device)
                mb_advantages = advantages[start:end].to(self.device)
                mb_returns = returns[start:end].to(self.device)
                
                with torch.autocast(device_type=device_type, enabled=use_amp):
                    a_loss = self.actor_loss(actor, mb_states, mb_actions, mb_advantages)
                    c_loss = self.critic_loss(self.critic, mb_states, mb_returns)
                    loss = (a_loss + c_loss) * weight
                loss.backward()
                
            nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
            optim.step()

        self.adapted_actor = actor

    def learn(self, env, sampler: OnlineSampler, seed: int, learning_progress: float = 0.0, **kwargs):
        self.train()

        t_start_total = time.time()  # Start total timer

        self.adapt_actor(env, sampler, seed)

        # Initialize time tracking variables
        total_timesteps, total_sample_time = 0, 0
        total_exp_update_time, total_backprop_time, total_base_update_time = 0, 0, 0

        policy_dict, gradient_dict = self.init_exp_policies(), {}

        # Collect initial data with the base policy
        init_batch, init_sample_time = sampler.collect_samples(env, self.actor, seed)
        total_sample_time += init_sample_time  # Track initial sample
        self.update_obs_rms(init_batch["states"])
        self.sync_obs_rms_to(self.actor, self.critic, self.critics, self.adapted_actor)

        total_timesteps += init_batch["states"].shape[0]



        loss_dict_list = []
        for j in range(self.num_exp_updates):
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
                self.sync_obs_rms_to(actors)
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
                exp_dict = self.learn_exploratory_policy(actor, batch, i)

                loss_dict_list.append(exp_dict["loss_dict"])
                gradient_dict[actor_idx] = exp_dict["gradients"]
                policy_dict[future_actor_idx] = exp_dict["updated_actor"]
                total_exp_update_time += exp_dict["update_time"]

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
        total_base_update_time = time.time() - t_base_start

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

        # Clear GPU cache after learn step to reclaim memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return {"loss_dict": loss_dict, "timesteps": total_timesteps}

    def learn_exploratory_policy(self, actor: nn.Module, batch: dict, i: int):
        """
        Performs a single exploratory update.
        Calculates intrinsic rewards, updates critics, and performs a differentiable actor update.
        """
        t0 = time.time()

        # Extract batch data and keep on CPU
        states = torch.as_tensor(batch["states"], dtype=self.dtype, device="cpu")
        actions = torch.as_tensor(batch["actions"], dtype=self.dtype, device="cpu")
        rewards = torch.as_tensor(batch["rewards"][:, i : i + 1], dtype=self.dtype, device="cpu")
        terminations = torch.as_tensor(batch["terminations"], dtype=self.dtype, device="cpu")
        truncations = torch.as_tensor(batch["truncations"], dtype=self.dtype, device="cpu")

        # Estimate Advantages
        with torch.no_grad():
            values = []
            chunk_size = 512
            for start in range(0, states.shape[0], chunk_size):
                end = min(start + chunk_size, states.shape[0])
                mb_states = self.preprocess_state(states[start:end])
                values.append(self.critics[i](mb_states).cpu())
            values = torch.cat(values, dim=0)

            advantages, returns = estimate_advantages(
                rewards,
                terminations,
                truncations,
                values,
                gamma=self.gamma,
                gae=self.gae,
            )

        # Critic Mini-batch Updates
        critic = self.critics[i]
        optim = self.critic_optim[i]
        losses = []

        device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        use_amp = (device_type == "cuda")

        # Loop over the dataset multiple times (epochs)
        critic_epochs = 5  # Number of passes over the data
        for _ in range(critic_epochs):
            def critic_loss_fn(s, _, r):
                s = self.preprocess_state(s)
                r = r.to(self.device)
                with torch.autocast(device_type=device_type, enabled=use_amp):
                    return self.critic_loss(critic, s, r)

            from utils.rl import average_loss_across_minibatches
            avg_loss = average_loss_across_minibatches(
                critic_loss_fn, states, None, returns, self.grad_batch_size
            )

            optim.zero_grad()
            avg_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
            optim.step()

            losses.append(avg_loss.item())

        # Average the accumulated losses for logging
        critic_loss = sum(losses) / len(losses)

        # 3. Update Actor (Exploratory Policy)
        actor_clone = self._share_clone(actor)

        # Select advantage based on whether we are in the 'exploratory' (int)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        from utils.rl import average_gradients_across_minibatches

        def actor_loss_fn(s, a, adv):
            s = self.preprocess_state(s)
            self.update_obs_rms(s)
            self.sync_obs_rms_to(actor, self.critics[i])
            a = a.to(self.device)
            adv = adv.to(self.device)
            with torch.autocast(device_type=device_type, enabled=use_amp):
                return self.actor_loss(actor, s, a, adv)

        gradients = average_gradients_across_minibatches(
            actor,
            actor_loss_fn,
            states,
            actions,
            advantages,
            self.grad_batch_size,
            create_graph=True,
        )

        # Manual SGD update on the clone to keep the graph connected
        with torch.no_grad():
            for p, g in zip(actor_clone.parameters(), gradients):
                p -= self.lr * g

        with torch.no_grad():
            from utils.rl import average_loss_across_minibatches
            with torch.autocast(device_type=device_type, enabled=use_amp):
                actor_loss_log = average_loss_across_minibatches(
                    actor_loss_fn, states, actions, advantages, self.grad_batch_size
                )

        loss_dict = {
            f"{self.name}/loss/actor_loss": actor_loss_log.item(),
            f"{self.name}/loss/critic_loss": critic_loss,
        }

        update_time = time.time() - t0

        return {
            "loss_dict": loss_dict,
            "update_time": update_time,
            "updated_actor": actor_clone,
            "gradients": gradients,
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
            states = torch.as_tensor(states, dtype=self.dtype)

            # Subsample
            B = states.shape[0]
            if B > self.grad_batch_size:
                idx = torch.randperm(B)[: self.grad_batch_size]
                trpo_states = self.preprocess_state(states[idx])
            else:
                trpo_states = self.preprocess_state(states)

            self.sync_obs_rms_to(self.actor)
            old_actor = self._share_clone(self.actor)

            # Flatten the aggregated gradients
            grad_flat = torch.cat([g.view(-1) for g in grads]).detach()

            # KL divergence closure for Hessian Vector Product
            def kl_fn():
                return compute_kl(old_actor, self.actor, trpo_states)

            Hv = lambda v: hessian_vector_product(kl_fn, self.actor, damping, v)

            # Compute search direction (F_inv * g) via CG (reduced from 10 to 5 iterations to save memory)
            step_dir = conjugate_gradients(Hv, grad_flat, nsteps=5)

            # Compute step size scaling (Lagrange multiplier)
            # Recompute HVP to get fresh graph without accumulating old ones
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            sAs = 0.5 * torch.dot(step_dir, Hv(step_dir))
            if sAs < 1e-8:
                full_step = torch.zeros_like(step_dir)
            else:
                lm = torch.sqrt(sAs / self.target_kl)
                full_step = step_dir / lm
            
            if not torch.isfinite(full_step).all():
                print("WARNING: full_step contains NaN/Inf! Rejecting update.")
                full_step = torch.zeros_like(step_dir)

            # Line Search (reduced backtrack_iters from 15 to 10 to save memory)
            with torch.no_grad():
                old_params = flat_params(self.actor)
                success = False
                backtrack_iters_effective = min(10, backtrack_iters)  # Cap at 10 iterations
                for i in range(backtrack_iters_effective):
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

            # Clear GPU cache after TRPO update to reclaim memory for next step
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
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
