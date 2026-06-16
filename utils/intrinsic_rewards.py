import glob
import os
from abc import abstractmethod
from math import ceil

import matplotlib

matplotlib.use("Agg")  # headless backend; avoids tk "main thread" GC errors

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Make sure to import your CNN and MLP building blocks
from policy.layers.building_blocks import CNN, MLP
from utils.get_envs import get_env
from utils.sampler import OnlineSampler
from utils.wrapper import RunningMeanStd

KERNEL_MODES = ("rbf", "laplacian", "cauchy", "cosine")


def _apply_kernel(phi_s: torch.Tensor, phi_g: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Goal-conditioned diffusion reward R(s,g) ∈ (0,1].

    σ² is estimated per-batch as 2·Var[φ(s)] summed over feature dims
    (standard bandwidth heuristic; scales automatically with the encoder).

    Modes
    -----
    rbf       : exp(-‖φ(s)−φ(g)‖² / 2σ²)           — Gaussian, peaks at 1
    laplacian : exp(-‖φ(s)−φ(g)‖ / σ)               — heavier tails than RBF
    cauchy    : 1 / (1 + ‖φ(s)−φ(g)‖² / σ²)         — polynomial decay
    cosine    : (φ(s)·φ(g) / ‖φ(s)‖‖φ(g)‖ + 1) / 2  — no σ, direction-only
    """
    if mode == "cosine":
        norm_s = phi_s.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        norm_g = phi_g.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        cosine = (phi_s / norm_s * phi_g / norm_g).sum(dim=-1, keepdim=True)
        return (cosine + 1.0) / 2.0  # shift [-1,1] → [0,1]

    sq_dist = ((phi_s - phi_g) ** 2).sum(dim=-1, keepdim=True)  # (B, 1)
    sigma2 = 2.0 * phi_s.var(dim=0).sum() + 1e-8  # scalar

    if mode == "rbf":
        return torch.exp(-sq_dist / (2.0 * sigma2))
    elif mode == "laplacian":
        return torch.exp(-sq_dist.sqrt() / sigma2.sqrt())
    elif mode == "cauchy":
        return 1.0 / (1.0 + sq_dist / sigma2)
    else:
        raise ValueError(f"Unknown kernel mode '{mode}'. Choose from {KERNEL_MODES}.")


# Canonical Atari visual-encoder width. The ALLO CNN, the IRPO policy CNN, and
# the VAE latent all use this so the encoder ALLO trains can be dropped into the
# policy (and vice-versa) without rebuilding MLP heads.
ATARI_ENCODER_DIM = 256


class AtariFeatureNet(nn.Module):
    """
    A wrapper network for image states that combines the CNN and MLP (or a custom backbone).
    Returns (features, infos) to perfectly match the expected output format of NeuralNet.

    `self.cnn` is the reusable visual encoder: ALLO trains it end-to-end (no VAE
    loss) and its weights are saved as the FROZEN encoder the IRPO policy loads.
    """

    def __init__(self, chw_shape, feature_dim, device, cnn_features_dim=ATARI_ENCODER_DIM, backbone=None):
        super().__init__()

        self.feature_dim = feature_dim
        self.cnn_features_dim = cnn_features_dim

        # 1. Visual encoder — shared (frozen) with the IRPO policy, so its width
        #    must match the policy CNN feature dim (ATARI_ENCODER_DIM).
        self.cnn = CNN(
            input_shape=chw_shape,
            features_dim=cnn_features_dim,
            initialization="default",
            device=device,
        )

        # 2. Project the encoder features down to the ALLO eigenvector feature_dim.
        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = MLP(
                input_dim=cnn_features_dim,
                hidden_dims=[512, 512, 512],
                output_dim=feature_dim,
                activation=nn.Tanh(),
                device=device,
            )

    def forward(self, x, deterministic=False):
        # Accept (B, H, W) grayscale or (B, 1, H, W) — normalise to [0, 1]
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.max() > 1.0:
            x = x / 255.0
        feat = self.cnn(x)
        if isinstance(self.backbone, MLP):
            feat = self.backbone(feat)
            return feat, {}
        else:
            return self.backbone(feat, deterministic=deterministic)


class EncoderWrappedNetwork(nn.Module):
    """
    Wraps a frozen ConvVAEEncoder with an ALLO MLP extractor.

    When used as the ALLO network for Atari:
    - encoder : ConvVAEEncoder (frozen, maps raw pixels → latent vector)
    - network : NeuralNet MLP  (trainable, maps latent → ALLO eigenvectors)

    This lets ALLO train on the same feature space the policy uses, without
    re-training the CNN.
    """

    def __init__(self, encoder: nn.Module, network: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.encoder.requires_grad_(False)  # frozen — only network trains
        self.network = network

    @property
    def feature_dim(self):
        return self.network.feature_dim

    def forward(self, x, deterministic=False):
        # x : (B, H, W) raw pixels or (B, 1, H, W) already formatted
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.max() > 1.0:
            x = x / 255.0
        with torch.no_grad():
            z = self.encoder(x)          # (B, latent_dim)
        features, infos = self.network(z, deterministic=deterministic)
        return features, infos


class BaseIntRewardFunctions(nn.Module):
    def __init__(self, logger, writer, args, init_timesteps: int = 0, **kwargs):
        super(BaseIntRewardFunctions, self).__init__()

        self.extractor_env = get_env(args)
        self.logger = logger
        self.writer = writer
        self.args = args

        self.current_timesteps = init_timesteps
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

        self.use_diff = kwargs.get("use_diff", True)

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
            if self.use_diff:
                feature, _ = self.extractor(states)
                next_feature, _ = self.extractor(next_states)
                difference = next_feature - feature
            else:
                difference, _ = self.extractor(states)

            # Extract all indices and signs for every 'i'
            indices = [e[0] for e in self.eigenvectors]
            signs = torch.tensor(
                [e[1] for e in self.eigenvectors], device=states.device
            )

            intrinsic_rewards = difference[:, indices] * signs

        intrinsic_rewards = self.reward_rms.normalize_var_only(intrinsic_rewards)

        return intrinsic_rewards

    def define_reward_model(self):
        from extractor.base.mlp import NeuralNet
        from extractor.extractor import ALLO as Random

        # === CREATE FEATURE EXTRACTOR === #
        if isinstance(self.input_shape, (tuple, list)) and len(self.input_shape) == 3:
            # RGB Image State: Use the Custom CNN Wrapper
            chw_shape = (self.input_shape[2], self.input_shape[0], self.input_shape[1])

            backbone = NeuralNet(
                state_dim=(ATARI_ENCODER_DIM,),
                feature_dim=self.args.feature_dim,
                encoder_fc_dim=[128, 128],
                activation=nn.LeakyReLU(),
                device=self.args.device,
            )
            feature_network = AtariFeatureNet(
                chw_shape=chw_shape,
                feature_dim=self.args.feature_dim,
                device=self.args.device,
                backbone=backbone,
            )
            # Since we are using images, pos_idx slicing is disabled
            extractor_pos_idx = None

        elif isinstance(self.input_shape, (tuple, list)) and len(self.input_shape) == 2:
            # Grayscale Image State: Force add 1 channel -> (1, H, W)
            chw_shape = (1, self.input_shape[0], self.input_shape[1])

            backbone = NeuralNet(
                state_dim=(ATARI_ENCODER_DIM,),
                feature_dim=self.args.feature_dim,
                encoder_fc_dim=[128, 128],
                activation=nn.LeakyReLU(),
                device=self.args.device,
            )
            feature_network = AtariFeatureNet(
                chw_shape=chw_shape,
                feature_dim=self.args.feature_dim,
                device=self.args.device,
                backbone=backbone,
            )
            # Since we are using images, pos_idx slicing is disabled
            extractor_pos_idx = None

        else:
            # Vector State: Use the original NeuralNet
            feature_network = NeuralNet(
                state_dim=len(self.args.pos_idx),
                feature_dim=self.args.feature_dim,
                encoder_fc_dim=[128, 128],
                activation=nn.LeakyReLU(),
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


class RandomIntRewardFunctionsG(RandomIntRewardFunctions):
    def __init__(self, mode: str = "rbf", **kwargs):
        super(RandomIntRewardFunctionsG, self).__init__(**kwargs)

        if mode not in KERNEL_MODES:
            raise ValueError(
                f"Unknown kernel mode '{mode}'. Choose from {KERNEL_MODES}."
            )
        self.mode = mode
        # Single combined reward regardless of args.num_options
        self.num_rewards = 1

    def forward(
        self, states: np.ndarray | torch.Tensor, next_states: np.ndarray | torch.Tensor
    ):
        states, next_states = self.preprocess_inputs(states, next_states)

        goals = states[:, self.args.goal_idx]  # (B, len(goal_idx))
        states = states[:, self.args.pos_idx]
        next_states = next_states[:, self.args.pos_idx]

        with torch.no_grad():
            phi_s, _ = self.extractor(states)  # (B, feature_dim)
            phi_g, _ = self.extractor(goals)  # (B, feature_dim)
            phi_s = phi_s[:, : self.args.num_options]  # (B, num_options)
            phi_g = phi_g[:, : self.args.num_options]  # (B, num_options)
            intrinsic_rewards = _apply_kernel(phi_s, phi_g, self.mode)

            if self.use_diff:
                n_phi_s, _ = self.extractor(next_states)
                n_phi_s = n_phi_s[:, : self.args.num_options]  # (B, num_options)
                n_intrinsic_rewards = _apply_kernel(n_phi_s, phi_g, self.mode)
                intrinsic_rewards = n_intrinsic_rewards - intrinsic_rewards

        return intrinsic_rewards


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
        self.use_diff = kwargs.get("use_diff", True)
        # Keep a local copy so define_reward_model / forward can check image vs vector
        self.input_shape = self.args.state_dim

        self.define_reward_model()
        self.define_eigenvectors()

    def forward(
        self, states: np.ndarray | torch.Tensor, next_states: np.ndarray | torch.Tensor
    ):
        states, next_states = self.preprocess_inputs(states, next_states)

        is_image = isinstance(self.input_shape, (tuple, list)) and len(self.input_shape) >= 2

        if is_image:
            # Grayscale (B, H, W) → (B, 1, H, W); RGB (B, H, W, C) → (B, C, H, W)
            if len(self.input_shape) == 2 and states.ndim == 3:
                states = states.unsqueeze(1)
                next_states = next_states.unsqueeze(1)
            elif len(self.input_shape) == 3 and states.ndim == 4 and states.shape[-1] == self.input_shape[-1]:
                states = states.permute(0, 3, 1, 2)
                next_states = next_states.permute(0, 3, 1, 2)

            if states.max() > 1.0:
                states = states.float() / 255.0
                next_states = next_states.float() / 255.0
        else:
            states = states[:, self.args.pos_idx]
            next_states = next_states[:, self.args.pos_idx]

        with torch.no_grad():
            if self.use_diff:
                feature, _ = self.extractor(states)
                next_feature, _ = self.extractor(next_states)
                difference = next_feature - feature
            else:
                difference, _ = self.extractor(states)

            # Extract all indices and signs for every 'i'
            indices = [e[0] for e in self.eigenvectors]
            signs = torch.tensor(
                [e[1] for e in self.eigenvectors], device=states.device
            )

            intrinsic_rewards = difference[:, indices] * signs

        intrinsic_rewards = self.reward_rms.normalize_var_only(intrinsic_rewards)

        return intrinsic_rewards

    def define_reward_model(self):
        from extractor.base.mlp import NeuralNet
        from extractor.extractor import ALLO
        from policy.uniform_random import UniformRandom
        from trainer.extractor_trainer import ExtractorTrainer

        # Goal-conditioned variants reuse the base env's trained models
        _g_to_base = {
            "fourroomsG": "fourrooms",
            "mazeG": "maze",
            "pointmazeG": "pointmaze",
            "antmazeG": "antmaze",
        }
        base_name, _, version = self.args.env_name.partition("-")
        model_env_name = _g_to_base.get(base_name, base_name)
        if version:
            model_env_name = f"{model_env_name}-{version}"

        if not os.path.exists("model"):
            os.makedirs("model")
        if not os.path.exists(f"model/{model_env_name}"):
            os.makedirs(f"model/{model_env_name}")

        # === CREATE FEATURE EXTRACTOR === #
        # Tracks the CNN visual encoder for image envs so it can be saved as the
        # frozen encoder the IRPO policy loads (None for vector envs).
        self._allo_pixel_encoder = None
        if isinstance(self.input_shape, (tuple, list)) and len(self.input_shape) in (2, 3):
            # Image state (grayscale (H,W) or RGB (H,W,C)).
            # ALLO trains its own CNN-augmented network end-to-end (NO VAE loss);
            # the trained CNN is then exported as the FROZEN encoder for the IRPO
            # policy, so both share the same visual representation.
            if len(self.input_shape) == 2:
                chw_shape = (1, self.input_shape[0], self.input_shape[1])
            else:
                chw_shape = (self.input_shape[2], self.input_shape[0], self.input_shape[1])

            backbone = NeuralNet(
                state_dim=(ATARI_ENCODER_DIM,),
                feature_dim=self.args.feature_dim,
                encoder_fc_dim=[512, 512, 512, 512],
                activation=nn.LeakyReLU(),
                device=self.args.device,
            )
            feature_network = AtariFeatureNet(
                chw_shape=chw_shape,
                feature_dim=self.args.feature_dim,
                device=self.args.device,
                backbone=backbone,
            )
            self._allo_pixel_encoder = feature_network.cnn
            extractor_pos_idx = None
        else:
            # Vector state: use the original MLP extractor
            feature_network = NeuralNet(
                state_dim=len(self.args.pos_idx),
                feature_dim=self.args.feature_dim,
                encoder_fc_dim=[512, 512, 512, 512],
                activation=nn.LeakyReLU(),
            )
            extractor_pos_idx = self.args.pos_idx

        # === DEFINE LEARNING METHOD FOR EXTRACTOR === #
        extractor = ALLO(
            network=feature_network,
            positional_indices=extractor_pos_idx,
            extractor_lr=self.args.extractor_lr,
            epochs=self.args.extractor_epochs,
            batch_size=1024,
            lr_barrier_coeff=self.args.lr_barrier_coeff,
            discount=self.args.discount_sampling_factor,
            device=self.args.device,
        )

        # Step 1: Search for .pth files in the directory
        model_dir = f"model/{model_env_name}/"
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
            total_target_samples = int(
                getattr(self.args, "extractor_total_samples", 200_000)
            )
            total_target_samples = max(1, total_target_samples)
            collect_batch_size = int(
                getattr(self.args, "extractor_collect_batch_size", 20_000)
            )
            collect_batch_size = max(10_000, min(50_000, collect_batch_size))
            collect_loops = ceil(total_target_samples / collect_batch_size)

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
                batch_size=collect_batch_size,
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
                init_step=self.current_timesteps,
                collect_loops=collect_loops,
                target_samples=total_target_samples,
            )
            final_timesteps = trainer.train()
            self.current_timesteps += final_timesteps

            torch.save(extractor.state_dict(), model_path)

        self.extractor = extractor

        # Export the ALLO-trained CNN as the FROZEN visual encoder for the IRPO
        # policy (image envs only). Saved unconditionally — even when ALLO was
        # already fully trained and just loaded — so the encoder file always
        # matches the extractor IRPO will run with.
        if self._allo_pixel_encoder is not None:
            enc_dir = os.path.join("model", model_env_name, "allo_encoder")
            os.makedirs(enc_dir, exist_ok=True)
            enc_path = os.path.join(enc_dir, f"{self.args.seed}.pth")
            torch.save(self._allo_pixel_encoder.state_dict(), enc_path)
            print(f"[ALLO] Saved frozen policy encoder to '{enc_path}'.")


class ALLOIntRewardFunctionG(ALLOIntRewardFunctions):
    def __init__(self, mode: str = "rbf", **kwargs):
        super(ALLOIntRewardFunctionG, self).__init__(**kwargs)

        if mode not in KERNEL_MODES:
            raise ValueError(
                f"Unknown kernel mode '{mode}'. Choose from {KERNEL_MODES}."
            )
        self.mode = mode
        # Single combined reward regardless of args.num_options
        self.num_rewards = 1

    def forward(
        self, states: np.ndarray | torch.Tensor, next_states: np.ndarray | torch.Tensor
    ):
        states, next_states = self.preprocess_inputs(states, next_states)

        goals = states[:, self.args.goal_idx]  # (B, len(goal_idx))
        states = states[:, self.args.pos_idx]
        next_states = next_states[:, self.args.pos_idx]

        with torch.no_grad():
            phi_s, _ = self.extractor(states)  # (B, feature_dim)
            phi_g, _ = self.extractor(goals)  # (B, feature_dim)

            phi_s = phi_s[:, : self.args.num_options]  # (B, num_options)
            phi_g = phi_g[:, : self.args.num_options]  # (B, num_options)

            intrinsic_rewards = _apply_kernel(phi_s, phi_g, self.mode)

            if self.use_diff:
                n_phi_s, _ = self.extractor(next_states)  # (B, feature_dim)
                n_phi_s = n_phi_s[:, : self.args.num_options]  # (B, num_options)
                n_intrinsic_rewards = _apply_kernel(n_phi_s, phi_g, self.mode)
                intrinsic_rewards = n_intrinsic_rewards - intrinsic_rewards

        return intrinsic_rewards
