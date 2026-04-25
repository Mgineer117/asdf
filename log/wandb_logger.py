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
        is_sweep: bool = False,
        sweep_metric_prefix: str | None = None,
    ) -> None:
        super().__init__(log_dir, log_txt, name)
        self.fps = fps
        self.is_sweep = is_sweep
        # When is_sweep=True we attach to the parent sweep run instead of starting a
        # new one. Per-run metrics are prefixed (e.g. "seed_1825/...") and use a
        # per-prefix custom step metric to avoid wandb's global monotonic-step
        # constraint when multiple seeds share a single sweep run.
        self.sweep_metric_prefix = sweep_metric_prefix
        self._defined_metrics = False

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
            self.wandb_run = wandb.run
            if self.wandb_run is not None and self.sweep_metric_prefix:
                self._define_sweep_metrics()

    def _define_sweep_metrics(self) -> None:
        if self._defined_metrics or self.wandb_run is None:
            return
        prefix = self.sweep_metric_prefix
        wandb.define_metric(f"{prefix}/env_step")
        wandb.define_metric(f"{prefix}/*", step_metric=f"{prefix}/env_step")
        self._defined_metrics = True

    def _prefixed(self, payload: dict) -> dict:
        if not self.sweep_metric_prefix:
            return payload
        return {f"{self.sweep_metric_prefix}/{k}": v for k, v in payload.items()}

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
        if not self.is_sweep:
            wandb.log(self.stats_mean, step=int(step))
        elif self.sweep_metric_prefix and self.wandb_run is not None:
            payload = self._prefixed(self.stats_mean)
            payload[f"{self.sweep_metric_prefix}/env_step"] = int(step)
            wandb.log(payload)

    def write_images(self, step: int, images: list | None, logdir: str) -> None:
        """Logs images to wandb."""
        if self.is_sweep and not self.sweep_metric_prefix:
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

        if self.is_sweep and self.sweep_metric_prefix and self.wandb_run is not None:
            wandb.log({
                f"{self.sweep_metric_prefix}/{logdir}": image_list,
                f"{self.sweep_metric_prefix}/env_step": int(step),
            })
        else:
            wandb.log({f"{logdir}": image_list}, step=int(step))

    def write_videos(self, step: int, images: np.ndarray, logdir: str) -> None:
        """Logs a video to wandb using a list of images."""
        if self.is_sweep and not self.sweep_metric_prefix:
            return

        images = np.transpose(images, (0, 3, 1, 2))
        video = wandb.Video(images, fps=self.fps, format="gif")
        if self.is_sweep and self.sweep_metric_prefix and self.wandb_run is not None:
            wandb.log({
                f"{self.sweep_metric_prefix}/{logdir}": video,
                f"{self.sweep_metric_prefix}/env_step": int(step),
            })
        else:
            wandb.log({f"{logdir}": video}, step=int(step))

    def restore_data(self) -> None:
        """Not implemented yet"""
        pass
