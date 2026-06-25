import time

import numpy as np
from tqdm import tqdm

from trainer.base_trainer import BaseTrainer


class OnPolicyTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self):
        start_time = time.time()
        update_count = 0
        self.evaluate_policy()  # eval_idx aligned to init in BaseTrainer.__init__

        # Absolute budget: a resumed chunk continues toward the SAME target
        # instead of adding another full `timesteps` on top of where it resumed.
        total = self.total_timesteps
        with tqdm(
            total=total,
            initial=self.init_timesteps,
            desc=f"{self.policy.name} Training (Timesteps)",
        ) as pbar:
            while pbar.n < total:
                self.current_step = pbar.n
                # Create a dictionary containing ALL possible inputs any algorithm might need
                learn_args = {
                    "env": self.env,
                    "sampler": self.sampler,
                    "seed": self.seed,
                    "learning_progress": self.current_step / total,
                }

                # Unpack the dictionary using **
                info = self.policy.learn(**learn_args)
                update_count += 1
                self.write_log(info["loss_dict"], self.current_step)

                for key, video_array in info.get("supp_dict", {}).items():
                    if video_array is not None and len(video_array) > 0:
                        # Revert (T, C, H, W) to list of (H, W, C) frames for BaseTrainer's API
                        frames = list(np.transpose(video_array, (0, 2, 3, 1)))
                        self.write_video(frames, self.current_step, "Video", key)

                for key, img in info.get("image_dict", {}).items():
                    if img is not None:
                        self.write_image(img, self.current_step, "Image", key)

                pbar.update(info["timesteps"])

                if self.current_step >= self.eval_interval * (self.eval_idx + 1):
                    self.evaluate_policy()

                # Rolling resume checkpoint every `checkpoint_interval` learn()
                # updates, independent of the eval cadence, so a timeout loses at
                # most that many updates of progress.
                if (
                    self.checkpoint_interval > 0
                    and update_count % self.checkpoint_interval == 0
                ):
                    self._save_latest_checkpoint(self.current_step)

        self.evaluate_policy()

        elapsed = (time.time() - start_time) / 3600
        self.logger.print(
            f"Total {self.policy.name} training time: {elapsed:.2f} hours"
        )
        return self.current_step
