import os
import random
import time
from collections import deque
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm

from trainer.base_trainer import BaseTrainer


class HRLTrainer(BaseTrainer):
    def __init__(
        self,
        env,
        hl_policy,
        policies,
        intrinsic_reward_fn,
        hl_sampler,
        sampler,
        logger,
        writer,
        init_timesteps=0,
        timesteps=1_000_000,
        hl_timesteps=1_000_000,
        log_interval=100,
        eval_num=10,
        rendering=False,
        seed=0,
    ):
        self.env = env
        self.hl_policy = hl_policy
        self.policies = policies
        self.intrinsic_reward_fn = intrinsic_reward_fn
        self.num_vectors = len(policies) - 1
        self.hl_sampler = hl_sampler
        self.sampler = sampler
        self.logger = logger
        self.writer = writer
        self.eval_num = eval_num
        self.rendering = rendering
        self.seed = seed

        self.init_timesteps = init_timesteps
        self.timesteps = timesteps
        self.hl_timesteps = hl_timesteps
        self.eval_interval = int(timesteps / log_interval)

        self.best_return_mean = -float("inf")
        self.recent_returns = deque(maxlen=5)

    def train(self):
        start_time = time.time()
        self.eval_idx = 0
        self.current_step = self.init_timesteps
        total_option_timesteps = int(
            self.timesteps * self.num_vectors + self.init_timesteps
        )

        # === Phase 1: Train option policies ===
        with tqdm(
            total=total_option_timesteps,
            initial=self.init_timesteps,
            desc=f"{self.hl_policy.name} Option Training (Timesteps)",
        ) as pbar:
            for option_idx in range(self.num_vectors):
                policy = self.policies[option_idx]
                phase_limit = int(
                    (option_idx + 1) * (self.timesteps + self.init_timesteps)
                )

                while pbar.n < phase_limit:
                    self.current_step = pbar.n

                    learn_args = {
                        "env": self.env,
                        "sampler": self.sampler,
                        "seed": self.seed,
                        "learning_progress": self.current_step / phase_limit,
                        "intrinsic_reward_fn": self.intrinsic_reward_fn,
                        "option_idx": option_idx,
                    }

                    info = policy.learn(**learn_args)

                    self.write_log(info["loss_dict"], step=self.current_step)
                    pbar.update(info["timesteps"])

        # === Phase 2: Train high-level policy ===
        self.hl_policy.policies = self.policies
        init_timesteps = self.current_step
        total_hl_timesteps = init_timesteps + self.hl_timesteps

        with tqdm(
            total=total_hl_timesteps,
            initial=init_timesteps,
            desc=f"{self.hl_policy.name} HL Training (Timesteps)",
        ) as pbar:
            while pbar.n < total_hl_timesteps:
                self.current_step = pbar.n

                learn_args = {
                    "env": self.env,
                    "sampler": self.hl_sampler,
                    "seed": self.seed,
                    "learning_progress": self.current_step / total_hl_timesteps,
                }

                info = self.hl_policy.learn(**learn_args)

                self.write_log(info["loss_dict"], step=self.current_step)
                pbar.update(info["timesteps"])

                if self.current_step >= self.eval_interval * (self.eval_idx + 1):
                    self.evaluate_hl_policy()

            # End of training logging
            elapsed = (time.time() - start_time) / 3600
            self.logger.print(
                f"Total {self.hl_policy.name} training time: {elapsed:.2f} hours"
            )
            return self.current_step

    def evaluate_hl_policy(self):
        self.hl_policy.eval()
        self.eval_idx += 1

        ep_returns, image_array, trajectories, desired_goals = [], [], [], []

        for episode in range(self.eval_num):
            # Replaced static seed with the randomized eval_seed for proper distribution
            eval_seed = random.randint(0, 10000) + self.seed + episode
            state, _ = self.env.reset(seed=eval_seed)
            ep_reward = []

            for _ in range(self.env.max_steps):
                with torch.no_grad():
                    [option_idx, a], meta = self.hl_policy(
                        state, None, deterministic=True
                    )
                    a = a.cpu().numpy().squeeze(0) if a.shape[-1] > 1 else [a.item()]

                if episode == 0 and self.rendering:
                    image_array.append(self.env.render())

                if meta["is_option"]:
                    option_termination = False
                    for _ in range(10):
                        next_state, rew, term, trunc, _ = self.env.step(a)
                        done = term or trunc
                        ep_reward.append(rew)
                        if done or option_termination:
                            break
                        with torch.no_grad():
                            [_, a], opt_meta = self.hl_policy(
                                next_state, option_idx=option_idx, deterministic=True
                            )
                            a = (
                                a.cpu().numpy().squeeze(0)
                                if a.shape[-1] > 1
                                else [a.item()]
                            )
                        option_termination = opt_meta["option_termination"]
                else:
                    next_state, rew, term, trunc, _ = self.env.step(a)
                    done = term or trunc
                    ep_reward.append(rew)

                state = next_state
                if done:
                    ep_returns.append(
                        self.discounted_return(ep_reward, self.hl_policy.gamma)
                    )
                    traj, goal = self.env.get_trajectory_info()
                    trajectories.append(traj)
                    desired_goals.append(goal)
                    break

        return_mean, return_std = np.mean(ep_returns), np.std(ep_returns)

        eval_dict = {
            "eval/return_mean": return_mean,
            "eval/return_std": return_std,
            "eval/spectral_loss": self.spectral_loss(self.hl_policy.actor),
        }

        # Handle specific HL visualization
        if self.hl_policy.state_visitation is not None:
            v = self.hl_policy.state_visitation
            v = (v - v.min()) / (v.max() - v.min() + 1e-8)
            self.write_image(
                self.visitation_to_rgb(v),
                self.current_step,
                "Image",
                "visitation map",
            )

        self.write_log(eval_dict, step=self.current_step, eval_log=True)
        self.write_image(
            self.env.get_trajectory_plot(trajectories, desired_goals),
            self.current_step,
            "Image",
            "trajectory",
        )
        self.write_video(image_array, self.current_step, "Video", "running_video")

        self.recent_returns.append(eval_dict["eval/return_mean"])

        # Save specific HL policy
        self.save_model(self.current_step, self.hl_policy, "hl_policy")
        torch.cuda.empty_cache()

    def save_model(self, step, model, name):
        if model is None:
            raise ValueError("Model is not identifiable!")

        model = deepcopy(model).to("cpu")
        torch.save(
            model.state_dict(),
            os.path.join(self.logger.checkpoint_dir, f"{name}_{step}.pth"),
        )

        # Save best model based on mean return alone (fix: removed conflicting std condition)
        current_mean = np.mean(self.recent_returns)
        if current_mean > self.best_return_mean:
            self.best_return_mean = current_mean
            torch.save(
                model.state_dict(), os.path.join(self.logger.log_dir, "best_model.pth")
            )
