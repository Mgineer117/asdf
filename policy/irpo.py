import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import matplotlib.pyplot as plt
from policy.layers.base import Base
from policy.layers.ppo_networks import PPO_Actor, PPO_Critic
from utils import ANTMAZE_G_MAPS, POINTMAZE_G_MAPS
from utils.intrinsic_rewards import BaseIntRewardFunctions
from utils.rl import *
from utils.sampler import OnlineSampler


def _count_candidate_goals(maze_map):
    return sum(cell == "c" for row in maze_map for cell in row)


NUM_GOALS = {
    "fourroomsG-v1": 4,
    "mazeG-v1": 3,
    "mazeG-v2": 4,
    "mazeG-v3": 4,
    **{
        env_name.replace("pointmaze-", "pointmazeG-"): _count_candidate_goals(maze_map)
        for env_name, maze_map in POINTMAZE_G_MAPS.items()
    },
    **{
        env_name.replace("antmaze-", "antmazeG-"): _count_candidate_goals(maze_map)
        for env_name, maze_map in ANTMAZE_G_MAPS.items()
    },
}


class IRPO_Learner(Base):

    def __init__(
        self,
        actor: PPO_Actor,
        critic: PPO_Critic,
        beta: float,
        intrinsic_reward_fn: BaseIntRewardFunctions,
        aggregation_method: str,
        temperature: float,
        noise_std: float,
        # find_lr: bool,
        num_exp_updates: int,
        base_policy_update_type: str = "trpo",
        lr: float = 1e-2,
        critic_lr: float = 1e-3,
        entropy_scaler: float = 1e-3,
        target_kl: float = 0.03,
        # base_target_kl: float = 0.001,
        l2_reg: float = 1e-8,
        gamma: float = 0.99,
        gae: float = 0.95,
        anneal_kl: bool = False,
        device: str = "cpu",
        vae_encoder=None,  # ConvVAEEncoder for Atari (pre-trained init)
        vae_lr: float = 1e-4,  # fine-tune LR (pretrain uses 1e-3); annealed
        train_vae: bool = True,  # True = keep fine-tuning the CNN via VAE loss
        grad_batch_size: int = 256,  # rows per create_graph grad / TRPO HVP minibatch
    ):
        super().__init__(device=device)

        self.name = "IRPO"
        self.device = device

        # The encoder (if provided, i.e. Atari) is the single shared CNN. It is
        # SHARED — not deep-copied — across the actor, all exploratory clones, and
        # all critics, so the whole learner holds exactly ONE CNN regardless of
        # num_options. _share_clone() relies on self.shared_encoder.
        #
        # Sharing is independent of whether the CNN is trained: the policy/value
        # gradients are detached from the CNN (PPO_Actor/Critic detach_cnn=True),
        # so per-clone IRPO updates leave the CNN untouched. The CNN is fine-tuned
        # ONLY by the VAE objective (when train_vae=True), starting from the
        # pre-trained weights loaded by the algorithm.
        self.shared_encoder = vae_encoder

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
        # self.base_target_kl = base_target_kl
        self.l2_reg = l2_reg
        self.anneal_kl = anneal_kl

        # Neural Networks
        self.actor = actor  # The "Base" policy
        self.intrinsic_reward_fn = intrinsic_reward_fn
        self.num_options = self.intrinsic_reward_fn.num_rewards

        # Critics — share the encoder so all 2*num_options critics reuse the
        # single CNN; only their MLP heads are distinct.
        self.ext_critics = nn.ModuleList(
            [self._share_clone(critic) for _ in range(self.num_options)]
        )
        self.int_critics = nn.ModuleList(
            [self._share_clone(critic) for _ in range(self.num_options)]
        )

        # Critic optimizers must exclude the shared encoder params (by identity):
        # the CNN is trained only by the VAE objective, never by the value loss.
        enc_param_ids = (
            {id(p) for p in self.shared_encoder.parameters()}
            if self.shared_encoder is not None
            else set()
        )
        self.ext_critic_optim = [
            torch.optim.Adam(
                [p for p in critic.parameters() if id(p) not in enc_param_ids],
                lr=self.critic_lr,
            )
            for critic in self.ext_critics
        ]
        self.int_critic_optim = [
            torch.optim.Adam(
                [p for p in critic.parameters() if id(p) not in enc_param_ids],
                lr=self.critic_lr,
            )
            for critic in self.int_critics
        ]

        # Tracking contributing (high reward-inducing) int rewards
        self.policy_indices = [i for i in range(self.num_options)]

        # Initialized to 0, updated via EMA.
        self.perf_gains = torch.zeros(self.num_options).to(self.device)
        self.aggregation_method = aggregation_method
        self.temperature = temperature

        # Storage for the best policies found during exploration (for inference/eval)
        self.final_exp_policies = [
            self._share_clone(actor) for _ in range(self.num_options)
        ]

        self.wall_clock_time = 0
        self.grad_batch_size = grad_batch_size
        self.to(self.dtype).to(self.device)

        # Joint VAE fine-tuning: the shared CNN starts from pre-trained weights
        # and keeps being trained by the VAE objective on incoming policy samples
        # (every 5 learn() calls, on a minibatch). Set train_vae=False to freeze.
        self.vae_encoder = vae_encoder if train_vae else None
        self.vae_lr_init = vae_lr
        self.vae_optim = (
            torch.optim.Adam(vae_encoder.parameters(), lr=vae_lr)
            if (vae_encoder is not None and train_vae)
            else None
        )
        self._vae_step_counter = 0   # counts learn() calls; VAE runs every 5
        self._vae_save_path: str | None = None

    def anneal_target_kl(self, learning_progress: float):
        # Optional: Anneal target KL over time (e.g., linearly decay)
        self.target_kl = self.init_target_kl * (1.0 - learning_progress) + 1e-5

    def forward(self, state: np.ndarray, deterministic: bool = False, **kwargs):
        state = self.preprocess_state(state)

        best_idx = torch.argmax(self.perf_gains).item()
        actor = self.final_exp_policies[best_idx]
        a, metaData = actor(state, deterministic=deterministic)

        return a, {
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
            "entropy": metaData["entropy"],
            "dist": metaData["dist"],
        }

    def _share_clone(self, module: nn.Module) -> nn.Module:
        """Clone an actor/critic, sharing the encoder instead of copying it.

        Deep-copying a module with the CNN inside would duplicate the (large) CNN
        for every option/clone — the dominant OOM source on image envs. Here we
        temporarily detach the encoder, deep-copy only the small MLP head, then
        re-attach the single shared encoder to both the original and the clone.

        Sharing is safe whether or not the CNN is trained: policy/value gradients
        are detached from it (detach_cnn=True), so per-clone updates leave it
        unchanged; only the VAE optimizer (which holds this same instance) trains
        the CNN.
        """
        if self.shared_encoder is None:
            return deepcopy(module)

        enc = module.feature_extractor
        module.feature_extractor = nn.Identity()
        clone = deepcopy(module)
        module.feature_extractor = enc
        clone.feature_extractor = enc
        return clone

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

            # torch.autograd.grad errors if ANY output lacks grad_fn.
            # CNN params have zero placeholder gradients (no grad_fn) because
            # the actor detaches CNN features before the MLP.  Only pass the
            # MLP gradient tensors (those with grad_fn) to grad().
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

    def _per_option_gradient(
        self, option_idx: int, policy_dict: dict, gradient_dict: dict
    ):
        """Hook returning the per-option gradient w.r.t. base-policy parameters.

        Default IRPO behaviour: backpropagate through the inner-update graph.
        Subclasses (e.g. IRPO_IS) override to use an alternative estimator.
        """
        return self.backprop(policy_dict, gradient_dict, option_idx)

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

        active_indices = self.policy_indices

        policy_dict, gradient_dict = self.init_exp_policies(), {}

        # Collect initial data with the base policy
        init_batch, init_sample_time = sampler.collect_samples(env, self.actor, seed)
        total_sample_time += init_sample_time  # Track initial sample
        # self.actor.record_state_visitations(init_batch["states"], alpha=1.0)

        total_timesteps += init_batch["states"].shape[0]

        # VAE fine-tuning — the shared CNN (pre-trained init) is the only thing
        # trained by this objective. Each IRPO iteration runs ONE epoch over the
        # on-policy batch, split into minibatches of `grad_batch_size`
        # (== minibatch_size, i.e. num_minibatch steps per epoch). The LR starts
        # at vae_lr (1e-4) and is linearly annealed over training. The CNN is
        # detached from the policy/value losses (detach_cnn) so IRPO's
        # create_graph=True never touches the CNN graph, and the single shared
        # instance keeps the learner to exactly one CNN.
        self._vae_step_counter += 1
        vae_loss_val = 0.0
        if self.vae_encoder is not None:
            # Linearly anneal VAE LR from vae_lr_init down to 10 % of it.
            annealed_lr = self.vae_lr_init * max(0.1, 1.0 - learning_progress)
            for pg in self.vae_optim.param_groups:
                pg["lr"] = annealed_lr

            # VAE inputs kept on CPU until minibatched
            raw_states_cpu = torch.as_tensor(init_batch["states"], dtype=self.dtype)
            B = raw_states_cpu.shape[0]
            mb = max(1, self.grad_batch_size)  # == minibatch_size
            perm = torch.randperm(B)

            vae_losses = []
            for start in range(0, B, mb):  # one epoch == num_minibatch steps
                idx = perm[start : start + mb]
                mb_raw_states = self.preprocess_state(raw_states_cpu[idx])
                vae_loss = self.vae_encoder.vae_loss(mb_raw_states)
                self.vae_optim.zero_grad()
                vae_loss.backward()
                self.vae_optim.step()
                vae_losses.append(vae_loss.item())
            vae_loss_val = sum(vae_losses) / len(vae_losses)

            # Checkpoint the jointly-tuned encoder (int_reward_type=random). Note:
            # random pretrains a fresh encoder each run and never loads this file;
            # it's kept only as a recoverable artifact of the run.
            if self._vae_save_path is None:
                import os
                _args = self.intrinsic_reward_fn.args
                _env = _args.env_name.split("-")[0]
                _seed = _args.seed
                _dir = os.path.join("model", _env, "encoder")
                os.makedirs(_dir, exist_ok=True)
                self._vae_save_path = os.path.join(_dir, f"{_seed}.pth")
            torch.save(self.vae_encoder.state_dict(), self._vae_save_path)

        loss_dict_list = []
        # Cache the latest exploratory-rollout batches keyed by option index so
        # subclasses (e.g. IRPO_IS) can compute alternative gradients on samples
        # drawn from the exploratory policies.
        self._last_batches: dict[int, dict] = {}
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

                if flag:
                    # Snapshot the final-step exploratory batch + actor for any
                    # downstream alternative gradient (e.g. IRPO_IS).
                    self._last_batches[i] = batch
                    # Update performance gains only on the fully adapted policy
                    self.perf_gains[i] = (
                        self.beta * self.perf_gains[i]
                        + (1 - self.beta) * exp_dict["ext_return"]
                    )

        # BACKPROP & AGGREGATION
        t_backprop_start = time.time()

        greedy_idx = torch.argmax(self.perf_gains).item()
        # Temperature anneals linearly from 1.0 to 0 over self.temperature
        # learning progress, then stays effectively at 0 (argmax-like).
        if self.temperature <= 0:
            temperature = 1e-8
        else:
            temperature = max(1e-8, 1.0 - learning_progress / self.temperature)

        # Snapshot context that subclasses (e.g. IRPO_TRPOFinal) may consult.
        self._init_batch = init_batch
        self._temperature = temperature

        if temperature <= 1e-8 or len(active_indices) == 1:
            self._collapsed = True
            active_gains = self.perf_gains[active_indices]
            best_local_idx = active_indices[torch.argmax(active_gains).item()]
            gradients = self._per_option_gradient(
                best_local_idx, policy_dict, gradient_dict
            )
            self._collapsed = False
        else:
            active_gains = self.perf_gains[active_indices]
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
                self._per_option_gradient(i, policy_dict, gradient_dict)
                for i in active_indices
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
        if self.vae_encoder is not None:
            loss_dict[f"{self.name}/loss/vae_loss"] = vae_loss_val
            loss_dict[f"{self.name}/loss/vae_lr"] = self.vae_optim.param_groups[0]["lr"]
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
        if getattr(self.intrinsic_reward_fn.args, "rendering", False):
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

        # Clear GPU cache after learn step to reclaim memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

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

        # Critic Updates: Average loss across full batch, then optimize
        # This uses all data points while keeping gradient flow stable
        critic_epochs = 5

        ext_critic = self.ext_critics[i]
        ext_optim = self.ext_critic_optim[i]
        int_critic = self.int_critics[i]
        int_optim = self.int_critic_optim[i]

        ext_losses = []
        int_losses = []

        for _ in range(critic_epochs):
            # Compute average loss across all minibatches in full batch
            def ext_loss_fn(s, _, r):
                return self.critic_loss(ext_critic, s, r)

            def int_loss_fn(s, _, r):
                return self.critic_loss(int_critic, s, r)

            # Average extrinsic critic loss
            from utils.rl import average_loss_across_minibatches
            avg_ext_loss = average_loss_across_minibatches(
                ext_loss_fn, states, None, ext_returns, self.grad_batch_size
            )

            ext_optim.zero_grad()
            avg_ext_loss.backward()
            ext_optim.step()
            ext_losses.append(avg_ext_loss.item())

            # Average intrinsic critic loss
            avg_int_loss = average_loss_across_minibatches(
                int_loss_fn, states, None, int_returns, self.grad_batch_size
            )

            int_optim.zero_grad()
            avg_int_loss.backward()
            int_optim.step()
            int_losses.append(avg_int_loss.item())

        # Average the losses for logging
        ext_critic_loss = sum(ext_losses) / len(ext_losses)
        int_critic_loss = sum(int_losses) / len(int_losses)

        # 3. Update Actor (Exploratory Policy) — Averaged gradients across minibatches
        exp_actor = self._share_clone(actor)

        # Select advantage based on whether we are in the 'exploratory' (int)
        # or 'base' (ext) phase of the loop for this specific calculation.
        advantages = ext_advantages if flag else int_advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Average gradients across minibatches: uses full batch info but keeps
        # memory low by only storing one create_graph=True graph at a time.
        from utils.rl import average_gradients_across_minibatches, average_loss_across_minibatches

        # --- Compute gradients ---
        def actor_loss_fn(s, a, adv):
            s = self.preprocess_state(s)
            a = a.to(self.device)
            adv = adv.to(self.device)
            return self.actor_loss(exp_actor, s, a, adv)

        actor_gradients = average_gradients_across_minibatches(
            exp_actor,
            actor_loss_fn,
            states,
            actions,
            advantages,
            self.grad_batch_size,
            create_graph=True,
        )

        gradients = [g.detach() for g in actor_gradients]

        with torch.no_grad():
            for p, g in zip(exp_actor.parameters(), gradients):
                p -= self.lr * g

        # --- Optimize Critics ---
        # Extrinsic Critic
        def ext_critic_loss_fn(s, _, r):
            s = self.preprocess_state(s)
            r = r.to(self.device)
            return self.critic_loss(self.ext_critics[i], s, r)

        ext_critic_loss = average_loss_across_minibatches(
            ext_critic_loss_fn, states, None, ext_returns, self.grad_batch_size
        )
        self.ext_critic_optim[i].zero_grad()
        ext_critic_loss.backward()
        self.ext_critic_optim[i].step()

        # Intrinsic Critic
        def int_critic_loss_fn(s, _, r):
            s = self.preprocess_state(s)
            r = r.to(self.device)
            return self.critic_loss(self.int_critics[i], s, r)

        int_critic_loss = average_loss_across_minibatches(
            int_critic_loss_fn, states, None, int_returns, self.grad_batch_size
        )
        self.int_critic_optim[i].zero_grad()
        int_critic_loss.backward()
        self.int_critic_optim[i].step()

        # If this is the final exploratory step, save this policy as a potential inference policy
        if flag:
            self.final_exp_policies[i] = exp_actor

        # Compute actor loss on full batch for logging
        with torch.no_grad():
            actor_loss_log = average_loss_across_minibatches(
                actor_loss_fn, states, actions, advantages, self.grad_batch_size
            )

        loss_dict = {
            f"{self.name}/loss/actor_loss": actor_loss_log.item(),
            f"{self.name}/loss/ext_critic_loss": ext_critic_loss.item(),
            f"{self.name}/loss/int_critic_loss": int_critic_loss.item(),
        }

        update_time = time.time() - t0

        return {
            "loss_dict": loss_dict,
            "update_time": update_time,
            "updated_actor": exp_actor,
            "gradients": gradients,
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
            states = torch.as_tensor(states, dtype=self.dtype)

            # Subsample once for all HVP / line-search KL computations.
            B = states.shape[0]
            if B > self.grad_batch_size:
                idx = torch.randperm(B)[: self.grad_batch_size]
                trpo_states = self.preprocess_state(states[idx])
            else:
                trpo_states = self.preprocess_state(states)

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
            sAs = torch.clamp(sAs, min=1e-8)
            lm = torch.sqrt(sAs / self.target_kl)
            full_step = step_dir / (lm + 1e-8)

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
                    kl = compute_kl(old_actor, self.actor, trpo_states)
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
        temperature: float,
        noise_std: float,
        num_exp_updates: int,
        base_policy_update_type: str = "trpo",
        lr: float = 1e-2,
        critic_lr: float = 1e-3,
        entropy_scaler: float = 1e-3,
        target_kl: float = 0.03,
        l2_reg: float = 1e-8,
        gamma: float = 0.99,
        gae: float = 0.95,
        anneal_kl: bool = True,
        device: str = "cpu",
        grad_batch_size: int = 256,
    ):
        super().__init__(
            actor=actor,
            critic=critic,
            beta=beta,
            intrinsic_reward_fn=intrinsic_reward_fn,
            aggregation_method="argmax",  # unused, required by parent signature
            temperature=temperature,
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
            grad_batch_size=grad_batch_size,
        )
        self.name = "IRPO_G"
        self.num_goals = NUM_GOALS.get(env_name, None)
        assert (
            self.num_goals is not None
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

    @staticmethod
    def _pcgrad_surgery(grad_list: list) -> tuple:
        """
        PCGrad: for each per-goal gradient g_i, project out components
        that conflict (negative dot product) with every other g_j (j≠i).
        Returns the element-wise mean of surgered gradients as a tuple
        matching the shape of grad_list[0].
        """
        flat = [torch.cat([g.flatten() for g in grads]) for grads in grad_list]

        surgered = []
        for i in range(len(flat)):
            g_i = flat[i].clone()
            for j in range(len(flat)):
                if i == j:
                    continue
                g_j = flat[j]
                dot = torch.dot(g_i, g_j)
                if dot < 0:
                    g_i = g_i - (dot / (g_j.dot(g_j) + 1e-8)) * g_j
            surgered.append(g_i)

        mean_grad = torch.stack(surgered).mean(0)

        result, offset = [], 0
        for g in grad_list[0]:
            numel = g.numel()
            result.append(mean_grad[offset : offset + numel].view(g.shape))
            offset += numel
        return tuple(result)

    def learn_exploratory_policy(
        self, actor: nn.Module, batch: dict, i: int, flag: bool
    ):
        from sklearn.cluster import KMeans

        t0 = time.time()

        states = self.preprocess_state(batch["states"])
        actions = self.preprocess_state(batch["actions"])
        ext_rewards = self.preprocess_state(batch["rewards"])
        int_rewards = self.preprocess_state(batch["int_rewards"][:, i : i + 1])
        terminations = self.preprocess_state(batch["terminations"])
        truncations = self.preprocess_state(batch["truncations"])

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

        batch_size = states.shape[0]
        critic_epochs, num_minibatches = 5, 4
        mb_size = max(1, batch_size // num_minibatches)

        ext_critic = self.ext_critics[i]
        ext_optim = self.ext_critic_optim[i]
        int_critic = self.int_critics[i]
        int_optim = self.int_critic_optim[i]
        ext_losses, int_losses = [], []

        for _ in range(critic_epochs):
            perm = torch.randperm(batch_size)
            for start in range(0, batch_size, mb_size):
                idx = perm[start : start + mb_size]
                ext_loss = self.critic_loss(ext_critic, states[idx], ext_returns[idx])
                ext_optim.zero_grad()
                ext_loss.backward()
                ext_optim.step()
                ext_losses.append(ext_loss.item())
                int_loss = self.critic_loss(int_critic, states[idx], int_returns[idx])
                int_optim.zero_grad()
                int_loss.backward()
                int_optim.step()
                int_losses.append(int_loss.item())

        ext_critic_loss = sum(ext_losses) / len(ext_losses)
        int_critic_loss = sum(int_losses) / len(int_losses)

        actor_clone = self._share_clone(actor)
        advantages = ext_advantages if flag else int_advantages
        advantages_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        goal_idx = getattr(self.intrinsic_reward_fn.args, "goal_idx", None)
        per_goal_grads = None

        if goal_idx is not None:
            try:
                goals_np = states[:, goal_idx].detach().cpu().numpy()
                kmeans = KMeans(n_clusters=self.num_goals, n_init=5, random_state=0)
                labels = kmeans.fit_predict(goals_np)

                per_goal_grads = []
                for c in range(self.num_goals):
                    mask = torch.from_numpy(labels == c).to(states.device)
                    if mask.sum() < 4:
                        continue
                    mb_states = states[mask].detach()
                    mb_actions = actions[mask].detach()
                    mb_adv = advantages[mask]
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                    loss_c = self.actor_loss(actor, mb_states, mb_actions, mb_adv)
                    grads_c = torch.autograd.grad(
                        loss_c,
                        tuple(actor.parameters()),
                        create_graph=True,
                        allow_unused=True,
                    )
                    grads_c = tuple(
                        g if g is not None else torch.zeros_like(p)
                        for p, g in zip(actor.parameters(), grads_c)
                    )
                    per_goal_grads.append(grads_c)

            except Exception as e:
                print(f"[WARNING] PCGrad cluster step failed, falling back: {e}")
                per_goal_grads = None

        if per_goal_grads is not None and len(per_goal_grads) >= 2:
            gradients = self._pcgrad_surgery(per_goal_grads)
        elif per_goal_grads is not None and len(per_goal_grads) == 1:
            gradients = per_goal_grads[0]
        else:
            actor_loss_full = self.actor_loss(actor, states, actions, advantages_norm)
            gradients = torch.autograd.grad(
                actor_loss_full,
                tuple(actor.parameters()),
                create_graph=True,
                allow_unused=True,
            )
            gradients = tuple(
                g if g is not None else torch.zeros_like(p)
                for p, g in zip(actor.parameters(), gradients)
            )

        with torch.no_grad():
            for p, g in zip(actor_clone.parameters(), gradients):
                p -= self.lr * g

        if flag:
            self.final_exp_policies[i] = actor_clone

        with torch.no_grad():
            actor_loss_val = self.actor_loss(
                actor, states, actions, advantages_norm
            ).item()

        loss_dict = {
            f"{self.name}/loss/actor_loss": actor_loss_val,
            f"{self.name}/loss/ext_critic_loss": ext_critic_loss,
            f"{self.name}/loss/int_critic_loss": int_critic_loss,
        }

        return {
            "loss_dict": loss_dict,
            "update_time": time.time() - t0,
            "updated_actor": actor_clone,
            "gradients": gradients,
            "ext_return": ext_returns.mean().item(),
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

                probe = self._share_clone(actor)
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
