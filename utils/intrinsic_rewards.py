import glob
import os
from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Make sure to import your CNN and MLP building blocks
from policy.layers.building_blocks import CNN, MLP
from utils.get_envs import get_env
from utils.sampler import OnlineSampler
from utils.wrapper import RunningMeanStd


class AtariFeatureNet(nn.Module):
    """
    A wrapper network for image states that combines the CNN and MLP.
    Returns (features, None) to perfectly match the expected output format of NeuralNet.
    """

    def __init__(self, chw_shape, feature_dim, device):
        super().__init__()

        self.feature_dim = feature_dim

        # 1. Extract 512-dim spatial features
        self.cnn = CNN(
            input_shape=chw_shape,
            features_dim=512,
            initialization="default",
            device=device,
        )

        # 2. Project down to the required feature_dim for the intrinsic reward
        self.mlp = MLP(
            input_dim=512,
            hidden_dims=[512, 512, 512],
            output_dim=feature_dim,
            activation=nn.Tanh(),
            device=device,
        )

    def forward(self, x, deterministic=False):
        feat = self.cnn(x)
        feat = self.mlp(feat)
        return feat, {}


class BaseIntRewardFunctions(nn.Module):
    def __init__(self, logger, writer, args, **kwargs):
        super(BaseIntRewardFunctions, self).__init__()

        self.extractor_env = get_env(args)
        self.logger = logger
        self.writer = writer
        self.args = args

        self.current_timesteps = 0
        self.num_rewards = args.num_options
        self.reward_rms = RunningMeanStd(shape=(self.num_rewards,))

    def preprocess_inputs(self, states, next_states):
        if isinstance(states, np.ndarray):
            states = torch.as_tensor(
                states, device=self.args.device, dtype=torch.float32
            )
        if isinstance(next_states, np.ndarray):
            next_states = torch.as_tensor(
                next_states, device=self.args.device, dtype=torch.float32
            )

        if states.ndim == 1:
            states = states.unsqueeze(0)
        if next_states.ndim == 1:
            next_states = next_states.unsqueeze(0)

        return states, next_states

    @abstractmethod
    def forward(self, **kwargs):
        pass

    @abstractmethod
    def learn(self, **kwargs):
        pass

    @abstractmethod
    def define_reward_model(self):
        pass

    def define_eigenvectors(self):
        self.eigenvectors = [
            (n // 2 + 1, 2 * (n % 2) - 1) for n in range(self.num_rewards)
        ]

        try:
            heatmaps = self.extractor_env.get_rewards_heatmap(
                self.extractor, self.eigenvectors
            )
            log_dir = f"{self.logger.log_dir}/intrinsic_rewards"

            os.makedirs(f"{log_dir}", exist_ok=True)
            for i, fig in enumerate(heatmaps):
                if isinstance(fig, np.ndarray):
                    plt.imsave(f"{log_dir}/figure_{i}.pdf", fig, cmap="viridis")
                    plt.imsave(f"{log_dir}/figure_{i}.svg", fig, cmap="viridis")
                elif isinstance(fig, plt.Figure):
                    fig.savefig(f"{log_dir}/figure_{i}.pdf", format="pdf")
                    fig.savefig(f"{log_dir}/figure_{i}.svg", format="svg")
            self.logger.write_images(
                step=self.current_timesteps, images=heatmaps, logdir="Image/Heatmaps"
            )
        except Exception as e:
            print(f"[WARNING] Failed to generate heatmaps: {e}")


class ArbitraryIntRewardFunctions(BaseIntRewardFunctions):
    def __init__(self, target: list | float | None, **kwargs):
        super(ArbitraryIntRewardFunctions, self).__init__(**kwargs)

        # Save input shape to determine if we are working with Images or Vectors
        self.pos_idx = self.args.pos_idx
        self.target = target

        if self.args.num_options > 1:
            print(
                f"[WARNING] ArbitraryIntRewardFunctions is designed for a single reward function. Ignoring extra options and using only the first one."
            )

    def forward(
        self, states: np.ndarray | torch.Tensor, next_states: np.ndarray | torch.Tensor
    ):
        states, next_states = self.preprocess_inputs(states, next_states)

        states = states[:, self.pos_idx]
        # next_states = next_states[:, self.pos_idx]
        # delta = next_states - states

        # if target is list measure the euclidean norm distance to the target vector, if target is a scalar, measure the distance to the target scalar, if target is None, just return the state values as rewards (maximize)
        # if target is scaler take norm first and then apply the formula, if target is vector apply the formula elementwise and then sum across the reward dimensions
        # if target is None then just give the norm of states as rewards (maximize)
        if self.target is None:
            intrinsic_rewards = 1 / (1 + torch.norm(states, p=2, dim=1, keepdim=True))
        elif isinstance(self.target, (float, int)):
            state_norms = torch.norm(states, p=2, dim=1, keepdim=True)
            intrinsic_rewards = 1 / (1 + torch.abs(state_norms - self.target))
        elif isinstance(self.target, list):
            target_tensor = torch.tensor(
                self.target, device=states.device, dtype=states.dtype
            )
            intrinsic_rewards = torch.sum(
                1 / (1 + torch.abs(states - target_tensor)), dim=1, keepdim=True
            )
        else:
            raise ValueError(f"Invalid target type: {type(self.target)}")

        return intrinsic_rewards

    def learn(self, **kwargs):
        pass

    def define_reward_model(self):
        pass


class RandomIntRewardFunctions(BaseIntRewardFunctions):
    def __init__(self, **kwargs):
        super(RandomIntRewardFunctions, self).__init__(**kwargs)

        # Save input shape to determine if we are working with Images or Vectors
        self.input_shape = self.args.state_dim

        self.define_reward_model()
        self.define_eigenvectors()

    def forward(
        self, states: np.ndarray | torch.Tensor, next_states: np.ndarray | torch.Tensor
    ):
        states, next_states = self.preprocess_inputs(states, next_states)

        # --- Image vs Vector Formatting ---
        # An image will have a shape of length 2 (H, W) or 3 (H, W, C).
        # A MuJoCo vector has a shape of length 1, e.g., (19,).
        is_image = (
            isinstance(self.input_shape, (tuple, list)) and len(self.input_shape) > 1
        )

        if is_image:
            # 2. Handle RGB: [Batch, H, W, 3] -> [Batch, 3, H, W]
            if (
                len(self.input_shape) == 3
                and states.ndim == 4
                and states.shape[-1] == self.input_shape[-1]
            ):
                states = states.permute(0, 3, 1, 2)
                next_states = next_states.permute(0, 3, 1, 2)

            # 3. Handle Grayscale: [Batch, H, W] -> [Batch, 1, H, W]
            elif len(self.input_shape) == 2 and states.ndim == 3:
                states = states.unsqueeze(1)
                next_states = next_states.unsqueeze(1)
            else:
                raise ValueError(
                    f"Unexpected image input shape: {states.shape}. Expected [Batch, H, W] or [Batch, H, W, Channels]."
                )
        else:
            # Original vector behavior: slice using pos_idx
            if hasattr(self.args, "pos_idx") and self.args.pos_idx is not None:
                states = states[:, self.args.pos_idx]
                next_states = next_states[:, self.args.pos_idx]

        # Prevent NaN explosions - normalize pixels to [0, 1]
        if states.ndim >= 3 and states.max() > 1.0:
            states = states.float() / 255.0
            next_states = next_states.float() / 255.0

        with torch.no_grad():
            feature, _ = self.extractor(states)
            next_feature, _ = self.extractor(next_states)
            difference = next_feature - feature

            # Extract all indices and signs for every 'i'
            indices = [e[0] for e in self.eigenvectors]
            signs = torch.tensor(
                [e[1] for e in self.eigenvectors], device=states.device
            )

            intrinsic_rewards = difference[:, indices] * signs

        # # normalize rewards using running std
        # self.reward_rms.update(intrinsic_rewards.cpu().numpy())
        # var_tensor = torch.as_tensor(
        #     self.reward_rms.var,
        #     device=intrinsic_rewards.device,
        #     dtype=intrinsic_rewards.dtype,
        # )
        # intrinsic_rewards = intrinsic_rewards / (torch.sqrt(var_tensor) + 1e-8)

        intrinsic_rewards = self.reward_rms.normalize_var_only(intrinsic_rewards)

        return intrinsic_rewards

    def define_reward_model(self):
        from extractor.base.mlp import NeuralNet
        from extractor.extractor import ALLO as Random

        # === CREATE FEATURE EXTRACTOR === #
        if isinstance(self.input_shape, (tuple, list)) and len(self.input_shape) == 3:
            # RGB Image State: Use the Custom CNN Wrapper
            chw_shape = (self.input_shape[2], self.input_shape[0], self.input_shape[1])

            feature_network = AtariFeatureNet(
                chw_shape=chw_shape,
                feature_dim=self.args.feature_dim,
                device=self.args.device,
            )
            # Since we are using images, pos_idx slicing is disabled
            extractor_pos_idx = None

        elif isinstance(self.input_shape, (tuple, list)) and len(self.input_shape) == 2:
            # Grayscale Image State: Force add 1 channel -> (1, H, W)
            chw_shape = (1, self.input_shape[0], self.input_shape[1])

            feature_network = AtariFeatureNet(
                chw_shape=chw_shape,
                feature_dim=self.args.feature_dim,
                device=self.args.device,
            )
            # Since we are using images, pos_idx slicing is disabled
            extractor_pos_idx = None

        else:
            # Vector State: Use the original NeuralNet
            feature_network = NeuralNet(
                state_dim=len(self.args.pos_idx),
                feature_dim=self.args.feature_dim,
                encoder_fc_dim=[128, 128],
                activation=nn.ELU(),
            )
            extractor_pos_idx = self.args.pos_idx

        # === DEFINE LEARNING METHOD FOR EXTRACTOR === #
        self.extractor = Random(
            network=feature_network,
            positional_indices=extractor_pos_idx,
            extractor_lr=self.args.extractor_lr,
            epochs=self.args.extractor_epochs,
            batch_size=1024,
            lr_barrier_coeff=self.args.lr_barrier_coeff,
            discount=self.args.discount_sampling_factor,
            device=self.args.device,
        )


class DRNDIntRewardFunctions(BaseIntRewardFunctions):
    def __init__(self, **kwargs):
        super(DRNDIntRewardFunctions, self).__init__(**kwargs)

        assert (
            self.args.num_options == 1
        ), "DRND only supports 1 intrinsic reward function."

        self.define_reward_model()
        self.define_eigenvectors()

    def forward(
        self, states: np.ndarray | torch.Tensor, next_states: np.ndarray | torch.Tensor
    ):
        NotImplementedError("DRND intrinsic reward function is not implemented yet.")

    def define_reward_model(self):
        NotImplementedError("DRND intrinsic reward function is not implemented yet.")


class ALLOIntRewardFunctions(BaseIntRewardFunctions):
    def __init__(self, **kwargs):
        super(ALLOIntRewardFunctions, self).__init__(**kwargs)

        self.num_trials = 2_000

        self.define_reward_model()
        self.define_eigenvectors()

    def forward(
        self, states: np.ndarray | torch.Tensor, next_states: np.ndarray | torch.Tensor
    ):
        states, next_states = self.preprocess_inputs(states, next_states)

        states = states[:, self.args.pos_idx]
        next_states = next_states[:, self.args.pos_idx]

        with torch.no_grad():
            feature, _ = self.extractor(states)
            next_feature, _ = self.extractor(next_states)
            difference = next_feature - feature

            # Extract all indices and signs for every 'i'
            indices = [e[0] for e in self.eigenvectors]
            signs = torch.tensor(
                [e[1] for e in self.eigenvectors], device=states.device
            )

            intrinsic_rewards = difference[:, indices] * signs

        # self.reward_rms.update(intrinsic_rewards.cpu().numpy())
        # var_tensor = torch.as_tensor(
        #     self.reward_rms.var,
        #     device=intrinsic_rewards.device,
        #     dtype=intrinsic_rewards.dtype,
        # )
        # intrinsic_rewards = intrinsic_rewards / (torch.sqrt(var_tensor) + 1e-8)

        intrinsic_rewards = self.reward_rms.normalize_var_only(intrinsic_rewards)

        return intrinsic_rewards

    def define_reward_model(self):
        from extractor.base.mlp import NeuralNet
        from extractor.extractor import ALLO
        from policy.uniform_random import UniformRandom
        from trainer.extractor_trainer import ExtractorTrainer

        if not os.path.exists("model"):
            os.makedirs("model")
        if not os.path.exists(f"model/{self.args.env_name}"):
            os.makedirs(f"model/{self.args.env_name}")

        # === CREATE FEATURE EXTRACTOR === #
        feature_network = NeuralNet(
            state_dim=len(self.args.pos_idx),
            feature_dim=self.args.feature_dim,
            encoder_fc_dim=[512, 512, 512, 512],
            activation=nn.LeakyReLU(),
        )

        # === DEFINE LEARNING METHOD FOR EXTRACTOR === #
        extractor = ALLO(
            network=feature_network,
            positional_indices=self.args.pos_idx,
            extractor_lr=self.args.extractor_lr,
            epochs=self.args.extractor_epochs,
            batch_size=1024,
            lr_barrier_coeff=self.args.lr_barrier_coeff,  # ALLO uses 0.01 lr_barrier_coeff
            discount=self.args.discount_sampling_factor,  # ALLO uses 0.99 discount
            device=self.args.device,
        )

        # Step 1: Search for .pth files in the directory
        model_dir = f"model/{self.args.env_name}/"
        pth_files = glob.glob(os.path.join(model_dir, "*.pth"))

        if not pth_files:
            print(
                f"[INFO] No existing model found in {model_dir}. Training from scratch."
            )
            epochs = 0
            model_path = os.path.join(
                model_dir,
                f"ALLO_{self.args.seed}_{self.args.extractor_epochs}_{self.args.discount_sampling_factor}.pth",
            )
        else:
            print(f"[INFO] Found {len(pth_files)} .pth files in {model_dir}")
            epochs = []
            seeds = []
            discount_factors = []
            valid_files = []

            for pth_file in pth_files:
                filename = os.path.basename(pth_file)
                parts = filename.replace(".pth", "").split("_")
                if len(parts) != 4:
                    print(f"[WARNING] Skipping malformed file: {filename}")
                    continue

                _, seed_str, epoch_str, discount_str = parts
                try:
                    seeds.append(int(seed_str))
                    epochs.append(int(epoch_str))
                    discount_factors.append(float(discount_str))
                    valid_files.append(filename)
                except ValueError:
                    print(f"[WARNING] Failed to parse file: {filename}")
                    continue

            if self.args.seed not in seeds:
                print(
                    f"[INFO] No model with seed {self.args.seed} found. Starting fresh."
                )
                epochs = 0
                model_path = os.path.join(
                    model_dir,
                    f"ALLO_{self.args.seed}_{self.args.extractor_epochs}_{self.args.discount_sampling_factor}.pth",
                )
            elif self.args.discount_sampling_factor not in discount_factors:
                print(
                    f"[INFO] No model with discount factor {self.args.discount_sampling_factor} found. Starting fresh."
                )
                epochs = 0
                model_path = os.path.join(
                    model_dir,
                    f"ALLO_{self.args.seed}_{self.args.extractor_epochs}_{self.args.discount_sampling_factor}.pth",
                )
            else:
                matching = [
                    (e, s, f, filename)
                    for e, s, f, filename in zip(
                        epochs, seeds, discount_factors, valid_files
                    )
                    if f == self.args.discount_sampling_factor and s == self.args.seed
                ]

                max_epoch, _, _, _ = max(matching, key=lambda x: x[0])
                idx = epochs.index(max_epoch)
                filename = matching[idx][-1]
                model_path = os.path.join(model_dir, filename)
                print(
                    f"[INFO] Loading model from: {model_path} (epoch {max_epoch}, seed {self.args.seed}, discount {self.args.discount_sampling_factor})"
                )

                extractor.load_state_dict(
                    torch.load(model_path, map_location=self.args.device)
                )
                extractor.to(self.args.device)
                epochs = max_epoch  # set current epoch

        if epochs < self.args.extractor_epochs:
            uniform_random_policy = UniformRandom(
                state_dim=self.args.state_dim,
                action_dim=self.args.action_dim,
                is_discrete=self.args.is_discrete,
                device=self.args.device,
            )
            sampler = OnlineSampler(
                state_dim=self.args.state_dim,
                action_dim=self.args.action_dim,
                episode_len=self.args.episode_len,
                batch_size=self.num_trials * self.args.episode_len,
                verbose=False,
            )
            trainer = ExtractorTrainer(
                env=self.extractor_env,
                random_policy=uniform_random_policy,
                extractor=extractor,
                sampler=sampler,
                logger=self.logger,
                writer=self.writer,
                epochs=self.args.extractor_epochs - epochs,
            )
            final_timesteps = trainer.train()
            self.current_timesteps += final_timesteps

            torch.save(extractor.state_dict(), model_path)

        self.extractor = extractor
