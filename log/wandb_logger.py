import uuid
from typing import Iterable

import numpy as np
import torch

import wandb
from log.base_logger import BaseLogger


class WandbLogger(BaseLogger):
    def __init__(
        self,
        config: dict = {},
        project: str = "project",
        group: str = "test",
        name: str = None,
        log_dir: str = "log",
        log_txt: bool = True,
        fps: int = 30,
        is_sweep: bool = False,  # 1. Add the is_sweep flag
    ) -> None:
        super().__init__(log_dir, log_txt, name)
        self.fps = fps
        self.is_sweep = is_sweep  # 2. Store the flag

        # 3. Only initialize WandB if we are NOT in a sweep
        if not self.is_sweep:
            self.wandb_run = (
                wandb.init(
                    project=project,
                    group=group,
                    name=name,
                    id=str(uuid.uuid4()),
                    resume="allow",
                    config=config,  # type: ignore
                    settings=wandb.Settings(init_timeout=120),
                )
                if not wandb.run
                else wandb.run
            )
        else:
            self.wandb_run = None

    def write(
        self,
        step: int,
        eval_log: bool = False,
        display: bool = True,
        display_keys: Iterable[str] = None,
    ) -> None:
        self.store(tab="update", env_step=step)
        self.write_without_reset(step)
        return super().write(step, eval_log, display, display_keys)

    def write_without_reset(self, step: int) -> None:
        """Sending data to wandb without resetting the current stored stats."""
        # 4. Bypass logging if it's a sweep
        if not self.is_sweep:
            wandb.log(self.stats_mean, step=int(step))

    def write_images(self, step: int, images: list | None, logdir: str) -> None:
        """Logs images to wandb."""
        # 4. Bypass logging if it's a sweep
        if self.is_sweep:
            return

        if isinstance(images, torch.Tensor):
            images = images.detach().cpu().numpy()
        if isinstance(images, np.ndarray):
            images = [images]

        image_list = []
        for img in images:
            if img is None:
                continue
            image_list.append(wandb.Image(img))

        wandb.log({f"{logdir}": image_list}, step=int(step))

    def write_videos(self, step: int, images: np.ndarray, logdir: str) -> None:
        """Logs a video to wandb using a list of images."""
        # 4. Bypass logging if it's a sweep
        if self.is_sweep:
            return

        images = np.transpose(images, (0, 3, 1, 2))
        wandb.log(
            {f"{logdir}": wandb.Video(images, fps=self.fps, format="gif")},
            step=int(step),
        )

    def restore_data(self) -> None:
        """Not implemented yet"""
        pass
