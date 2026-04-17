import os
import random
from abc import abstractmethod
from collections import deque
from copy import deepcopy

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from log.wandb_logger import WandbLogger
from policy.layers.base import Base
from utils.sampler import OnlineSampler


# model-free policy trainer
class BaseTrainer:
    def __init__(
        self,
        env: gym.Env,
        policy: Base,
        sampler: OnlineSampler,
        logger: WandbLogger,
        writer: SummaryWriter,
        init_timesteps: int = 0,
        timesteps: int = 1e6,
        log_interval: int = 100,
        eval_num: int = 10,
        rendering: bool = False,
        seed: int = 0,
    ) -> None:
        self.env = env
        self.policy = policy
        self.sampler = sampler
        self.eval_num = eval_num

        self.logger = logger
        self.writer = writer

        # training parameters
        self.init_timesteps = init_timesteps
        self.timesteps = timesteps

        self.log_interval = log_interval
        self.eval_interval = int(self.timesteps / self.log_interval)

        self.eval_idx, self.current_step = -1, 0

        # initialize the essential training components
        self.best_return_mean = -float("inf")
        self.best_success_mean = -float("inf")
        self.recent_returns = deque(maxlen=5)
        self.recent_successes = deque(maxlen=5)

        self.rendering = rendering
        self.seed = seed

    @abstractmethod
    def train(self) -> dict[str, float]:
        pass

    def evaluate_policy(self):
        self.policy.eval()
        self.eval_idx += 1

        (
            ep_returns,
            ep_discounted_returns,
            ep_successes,
            image_array,
            trajectories,
            desired_goals,
        ) = ([], [], [], [], [], [])

        for episode in range(self.eval_num):
            eval_seed = random.randint(0, 10000) + self.seed + episode
            state, _ = self.env.reset(seed=eval_seed)
            ep_reward, ep_success = [], 0.0

            fwd_vel = []

            # 10000 is hard limit
            for _ in range(self.sampler.episode_len):
                with torch.no_grad():
                    a, _ = self.policy(state, eval_mode=True)  # , deterministic=True)
                    a = a.cpu().numpy().flatten()

                if episode == 0 and self.rendering:
                    image_array.append(self.env.render())

                state, rew, term, trunc, info = self.env.step(a)
                done = term or trunc

                if info.get("x_velocity", None) is not None:
                    fwd_vel.append(info["x_velocity"])

                ep_reward.append(rew)
                ep_success = max(ep_success, info.get("success", 0.0))

                if done:
                    ep_returns.append(sum(ep_reward))
                    ep_discounted_returns.append(
                        self.discounted_return(ep_reward, self.policy.gamma)
                    )
                    ep_successes.append(ep_success)

                    if hasattr(self.env, "get_trajectory_info"):
                        traj, goal = self.env.get_trajectory_info()
                        trajectories.append(traj)
                        desired_goals.append(goal)

                    break

        return_mean, return_std = np.mean(ep_returns), np.std(ep_returns)
        discounted_return_mean, discounted_return_std = (
            np.mean(ep_discounted_returns),
            np.std(ep_discounted_returns),
        )
        success_mean, success_std = np.mean(ep_successes), np.std(ep_successes)

        eval_dict = {
            "eval/return_mean": return_mean,
            "eval/return_std": return_std,
            "eval/discounted_return_mean": discounted_return_mean,
            "eval/discounted_return_std": discounted_return_std,
            "eval/success_mean": success_mean,
            "eval/success_std": success_std,
            # "eval/spectral_loss": self.spectral_loss(self.policy.actor),
        }

        supp_dict = {
            "rendering": image_array,
        }

        if hasattr(self.env, "get_trajectory_plot") and len(trajectories) > 0:
            supp_dict["trajectory_plot"] = self.env.get_trajectory_plot(
                trajectories, desired_goals
            )
            self.write_image(
                supp_dict["trajectory_plot"], self.current_step, "Image", "trajectory"
            )

        if len(fwd_vel) > 0:
            supp_dict["forward_velocity"] = self.plot_fwd_velocity(fwd_vel)
            self.write_image(
                supp_dict["forward_velocity"],
                self.current_step,
                "Image",
                "forward_velocity",
            )

        self.write_log(eval_dict, step=self.current_step, eval_log=True)
        self.write_video(
            supp_dict["rendering"], self.current_step, "Video", "running_video"
        )

        self.recent_returns.append(eval_dict["eval/return_mean"])
        self.recent_successes.append(eval_dict["eval/success_mean"])
        self.save_model(self.current_step)

    def plot_fwd_velocity(self, fwd_vel):
        # This creates both the figure and the axes (the drawing area)
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(fwd_vel, label="Forward Velocity")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Velocity")
        ax.set_title("Forward Velocity Over Time")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()
        return fig

    def spectral_loss(
        self, policy: nn.Module, power_iters: int = 3, k: int = 2
    ) -> torch.Tensor:
        """Compute spectral loss of a given policy network.
        Loss → 0 when spectral norm of weight matrices = 1 and biases = 0.
        """
        with torch.no_grad():
            spectral_loss = 0.0
            device = next(policy.parameters()).device

            for name, param in policy.named_parameters():
                if "weight" in name:
                    W = param
                    A = W.T @ W

                    b_k = torch.randn(A.shape[1], device=device)
                    for _ in range(power_iters):
                        b_k1 = A @ b_k
                        b_k1_norm = torch.norm(b_k1) + 1e-12
                        b_k = b_k1 / b_k1_norm

                    sigma = torch.norm(W @ b_k)  # spectral norm approximation
                    spectral_loss += (sigma**k - 1) ** 2

                elif "bias" in name:
                    spectral_loss += torch.sum(param ** (2 * k))

        return spectral_loss.item()

    # power_iteration(np.array([[0.5, 0.5], [0.2, 0.8]]), 10)

    def discounted_return(self, rewards, gamma):
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
        return G

    def write_log(self, logging_dict: dict, step: int, eval_log: bool = False):
        # Logging to WandB and Tensorboard
        self.logger.store(**logging_dict)
        self.logger.write(step, eval_log=eval_log, display=False)
        for key, value in logging_dict.items():
            self.writer.add_scalar(key, value, step)

    def write_image(
        self, image: np.ndarray | plt.Figure, step: int, logdir: str, name: str
    ):
        image_list = [image]
        image_path = os.path.join(logdir, name)
        self.logger.write_images(step=step, images=image_list, logdir=image_path)

    def write_video(self, image: list, step: int, logdir: str, name: str):
        if len(image) > 0:
            tensor = np.stack(image, axis=0)
            video_path = os.path.join(logdir, name)
            self.logger.write_videos(step=step, images=tensor, logdir=video_path)

    def save_model(self, step):
        if self.policy.actor is None:
            raise ValueError("Model is not identifiable!")

        model = deepcopy(self.policy.actor).to("cpu")
        torch.save(
            model.state_dict(),
            os.path.join(self.logger.checkpoint_dir, f"model_{step}.pth"),
        )

        # Save best model based on mean return alone (fix: removed conflicting std condition)
        current_return_mean = np.mean(self.recent_returns)
        if current_return_mean > self.best_return_mean:
            self.best_return_mean = current_return_mean
            torch.save(
                model.state_dict(), os.path.join(self.logger.log_dir, "best_model.pth")
            )
        current_success_mean = np.mean(self.recent_successes)
        if current_success_mean > self.best_success_mean:
            self.best_success_mean = current_success_mean
            torch.save(
                model.state_dict(),
                os.path.join(self.logger.log_dir, "best_success_model.pth"),
            )

    def visitation_to_rgb(self, visitation_map: np.ndarray) -> np.ndarray:
        visitation_map = np.squeeze(visitation_map)  # Make sure it's 2D
        H, W = visitation_map.shape

        rgb_map = np.ones((H, W, 3), dtype=np.float32)  # Start with white

        # Zero visitation → gray
        zero_mask = visitation_map == 0
        rgb_map[zero_mask] = [0.5, 0.5, 0.5]

        # Nonzero visitation → white → blue gradient
        nonzero_mask = visitation_map > 0
        blue_intensity = visitation_map[nonzero_mask]

        rgb_map[nonzero_mask] = np.stack(
            [
                1.0 - blue_intensity,  # Red
                1.0 - blue_intensity,  # Green
                np.ones_like(blue_intensity),  # Blue
            ],
            axis=-1,
        )

        return rgb_map
