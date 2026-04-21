from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from policy.layers.base import Base
from policy.layers.building_blocks import CNN, MLP
from utils.wrapper import RunningMeanStd


class PPO_Actor(Base):
    def __init__(
        self,
        input_dim: Union[int, tuple, list],
        hidden_dim: list,
        action_dim: int,
        is_discrete: bool,
        activation: nn.Module = nn.Tanh(),
        device=torch.device("cpu"),
    ):
        super().__init__(device=device)

        self.hidden_dim = hidden_dim
        self.action_dim = np.prod(action_dim)
        # self.rms = RunningMeanStd(shape=input_dim)
        self.is_discrete = is_discrete

        # Save the original input shape so we can rebuild flattened states
        self.input_shape = input_dim

        # Check if observation is an image (C, H, W) or a 1D vector
        if isinstance(input_dim, (tuple, list)) and len(input_dim) == 3:
            # RGB Image: Change (H, W, C) to (C, H, W)
            chw_shape = (input_dim[2], input_dim[0], input_dim[1])

            cnn_features_dim = 256
            self.feature_extractor = CNN(
                input_shape=chw_shape, features_dim=cnn_features_dim, device=device
            )
            mlp_input_dim = cnn_features_dim

        elif isinstance(input_dim, (tuple, list)) and len(input_dim) == 2:
            # Grayscale Image: (H, W) -> Force add a channel (1, H, W)
            chw_shape = (1, input_dim[0], input_dim[1])

            cnn_features_dim = 256
            self.feature_extractor = CNN(
                input_shape=chw_shape, features_dim=cnn_features_dim, device=device
            )
            mlp_input_dim = cnn_features_dim

        else:
            # Vector input
            self.feature_extractor = nn.Identity()
            mlp_input_dim = np.prod(input_dim)

        # The MLP acts as the policy head
        self.model = MLP(
            mlp_input_dim,
            hidden_dim,
            self.action_dim,
            activation=activation,
            initialization="actor",
            device=device,
        )

        if not self.is_discrete:
            self.logstd = nn.Parameter(torch.zeros(1, self.action_dim))

        self.device = device
        self.to(self.device).to(self.dtype)

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ):
        # state = self.rms.normalize(state)
        state = self.preprocess_state(state)

        # --- Universal Image Reshape Logic ---
        if isinstance(self.input_shape, (tuple, list)):
            expected_flat_size = int(np.prod(self.input_shape))

            # 1. Unbatched single image: (H, W) or (H, W, C) -> Add Batch Dimension
            if tuple(state.shape) == tuple(self.input_shape):
                state = state.unsqueeze(0)

            # 2. Flattened Rollout Batch: Rebuild back to original dimensions
            elif state.numel() % expected_flat_size == 0 and state.ndim <= 2:
                batch_size = state.numel() // expected_flat_size
                state = state.view(batch_size, *self.input_shape)

            # 3. RGB Format: [Batch, H, W, C] -> Permute to PyTorch's [Batch, C, H, W]
            if (
                len(self.input_shape) == 3
                and state.ndim == 4
                and state.shape[-1] == self.input_shape[-1]
            ):
                state = state.permute(0, 3, 1, 2)

            # 4. Grayscale Format: [Batch, H, W] -> Add Channel [Batch, 1, H, W]
            elif len(self.input_shape) == 2 and state.ndim == 3:
                state = state.unsqueeze(1)

        # Prevent NaN explosions by normalizing raw 0-255 pixels to 0-1
        if state.ndim >= 3 and state.max() > 1.0:
            state = state.float() / 255.0

        # Extract features (passes through CNN if image, or Identity if vector)
        features = self.feature_extractor(state)

        if self.is_discrete:
            a, metaData = self.discrete_forward(features, deterministic)
        else:
            a, metaData = self.continuous_forward(features, deterministic)

        return a, metaData

    def continuous_forward(
        self,
        features: torch.Tensor,
        deterministic: bool = False,
    ):
        logits = self.model(features)

        ### Shape the output as desired
        mu = logits
        logstd = torch.clip(self.logstd, -5, 2)  # Clip logstd to avoid numerical issues
        std = torch.exp(logstd.expand_as(mu))
        dist = Normal(loc=mu, scale=std)

        a = dist.rsample()

        logprobs = dist.log_prob(a).unsqueeze(-1).sum(1)
        probs = torch.exp(logprobs)
        entropy = dist.entropy().sum(1)

        return a, {
            "dist": dist,
            "probs": probs,
            "logprobs": logprobs,
            "entropy": entropy,
        }

    def discrete_forward(
        self,
        features: torch.Tensor,
        deterministic: bool = False,
    ):
        logits = self.model(features)

        if deterministic:
            a = torch.argmax(logits, dim=-1)
            dist = None
            logprobs = torch.zeros_like(logits[:, 0:1])
            probs = torch.ones_like(logprobs)
            entropy = torch.zeros_like(logprobs)
        else:
            dist = Categorical(logits=logits)
            a = dist.sample()

            logprobs = dist.log_prob(a).unsqueeze(-1)
            probs = torch.exp(logprobs)

            entropy = dist.entropy()

        a = F.one_hot(a, num_classes=logits.size(-1))
        return a, {
            "dist": dist,
            "probs": probs,
            "logprobs": logprobs,
            "entropy": entropy,
        }

    def log_prob(self, dist: torch.distributions, actions: torch.Tensor):
        """
        Actions must be tensor
        """
        actions = actions.squeeze() if actions.shape[-1] > 1 else actions

        if self.is_discrete:
            logprobs = dist.log_prob(torch.argmax(actions, dim=-1)).unsqueeze(-1)
        else:
            logprobs = dist.log_prob(actions).unsqueeze(-1).sum(1)
        return logprobs

    def entropy(self, dist: torch.distributions):
        """
        For code consistency
        """
        if self.is_discrete:
            return dist.entropy().unsqueeze(-1)
        else:
            return dist.entropy().unsqueeze(-1).sum(1)


class PPO_Critic(nn.Module):
    def __init__(
        self,
        input_dim: Union[int, tuple, list],
        hidden_dim: list,
        activation: nn.Module = nn.Tanh(),
        device=torch.device("cpu"),
    ):
        super(PPO_Critic, self).__init__()

        self.hidden_dim = hidden_dim

        # Save the original input shape so we can rebuild flattened states
        self.input_shape = input_dim
        # self.rms = RunningMeanStd(shape=input_dim)

        # Check if observation is an image (C, H, W) or a 1D vector
        if isinstance(input_dim, (tuple, list)) and len(input_dim) == 3:
            cnn_features_dim = 512

            # Change (H, W, C) to (C, H, W) to match PyTorch expectations
            chw_shape = (input_dim[2], input_dim[0], input_dim[1])

            self.feature_extractor = CNN(
                input_shape=chw_shape,
                features_dim=cnn_features_dim,
                initialization="critic",
                device=device,
            )
            mlp_input_dim = cnn_features_dim

        elif isinstance(input_dim, (tuple, list)) and len(input_dim) == 2:
            # Grayscale Image: (H, W) -> Force add a channel (1, H, W)
            chw_shape = (1, input_dim[0], input_dim[1])

            cnn_features_dim = 512
            self.feature_extractor = CNN(
                input_shape=chw_shape,
                features_dim=cnn_features_dim,
                initialization="critic",
                device=device,
            )
            mlp_input_dim = cnn_features_dim

        else:
            self.feature_extractor = nn.Identity()
            mlp_input_dim = np.prod(input_dim)

        self.model = MLP(
            mlp_input_dim,
            hidden_dim,
            1,
            activation=activation,
            initialization="critic",
            device=device,
        )

        self.to(device)

    def forward(self, x: torch.Tensor):
        # x = self.rms.normalize(x)

        # --- Universal Image Reshape Logic ---
        if isinstance(self.input_shape, (tuple, list)):
            expected_flat_size = int(np.prod(self.input_shape))

            # 1. Unbatched single image: (H, W) or (H, W, C) -> Add Batch Dimension
            if tuple(x.shape) == tuple(self.input_shape):
                x = x.unsqueeze(0)

            # 2. Flattened Rollout Batch: Rebuild back to original dimensions
            elif x.numel() % expected_flat_size == 0 and x.ndim <= 2:
                batch_size = x.numel() // expected_flat_size
                x = x.view(batch_size, *self.input_shape)

            # 3. RGB Format: [Batch, H, W, C] -> Permute to PyTorch's [Batch, C, H, W]
            if (
                len(self.input_shape) == 3
                and x.ndim == 4
                and x.shape[-1] == self.input_shape[-1]
            ):
                x = x.permute(0, 3, 1, 2)

            # 4. Grayscale Format: [Batch, H, W] -> Add Channel [Batch, 1, H, W]
            elif len(self.input_shape) == 2 and x.ndim == 3:
                x = x.unsqueeze(1)

        # Prevent NaN explosions by normalizing raw 0-255 pixels to 0-1
        if x.ndim >= 3 and x.max() > 1.0:
            x = x.float() / 255.0

        features = self.feature_extractor(x)
        value = self.model(features)
        return value


class FrozenEncoderActor(Base):
    """
    Policy actor for IRPO on image observations.

    A pretrained ImageEncoder is frozen (no gradient, excluded from .parameters())
    so IRPO Jacobians only flow through the small MLP head, dramatically reducing
    peak memory from create_graph=True during the exploratory update.

    Usage::

        from extractor.base.image_encoder import pretrain_image_encoder
        encoder = pretrain_image_encoder(env, seed=args.seed, device=args.device)
        actor = FrozenEncoderActor(
            encoder=encoder, encoder_dim=256,
            hidden_dim=[256, 256], action_dim=args.action_dim,
            is_discrete=True, device=args.device,
        )
    """

    def __init__(
        self,
        encoder,
        encoder_dim: int,
        hidden_dim: list,
        action_dim: int,
        is_discrete: bool,
        activation: nn.Module = nn.Tanh(),
        device=torch.device("cpu"),
    ):
        super().__init__(device=device)

        self.hidden_dim = hidden_dim
        self.action_dim = np.prod(action_dim)
        self.is_discrete = is_discrete
        # input_shape=None signals that states arrive as raw pixels but are
        # pre-encoded inside forward(); the rest of the codebase checks this attr.
        self.input_shape = None

        # Frozen encoder — registered as a proper submodule so .to(device) moves it,
        # but excluded from .parameters() so IRPO never backprops through the CNN.
        self.encoder = encoder
        self.encoder.requires_grad_(False)

        # Trainable MLP head — the only parameters IRPO differentiates through.
        self.model = MLP(
            encoder_dim,
            hidden_dim,
            self.action_dim,
            activation=activation,
            initialization="actor",
            device=device,
        )

        if not is_discrete:
            self.logstd = nn.Parameter(torch.zeros(1, self.action_dim))

        self.device = device
        self.to(device).to(self.dtype)

    # ------------------------------------------------------------------
    # Override parameters() so that IRPO Jacobians / TRPO HVP only see MLP
    # ------------------------------------------------------------------

    def parameters(self, recurse=True):
        yield from self.model.parameters(recurse=recurse)
        if not self.is_discrete and hasattr(self, "logstd"):
            yield self.logstd

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, state, deterministic=False):
        state = self.preprocess_state(state)  # → float32 tensor on device

        # Reshape raw pixels to (B, C, H, W)
        if state.ndim == 2:  # (H, W) single grayscale frame
            state = state.unsqueeze(0).unsqueeze(0)
        elif state.ndim == 3:
            if state.shape[-1] in (1, 3):  # (H, W, C) single colour frame
                state = state.permute(2, 0, 1).unsqueeze(0)
            else:  # (B, H, W) batched grayscale
                state = state.unsqueeze(1)
        elif state.ndim == 4 and state.shape[-1] in (1, 3):  # (B, H, W, C)
            state = state.permute(0, 3, 1, 2)

        if state.max() > 1.0:
            state = state.float() / 255.0

        # Frozen encoder — no autograd graph built, features detached as leaf
        with torch.no_grad():
            features = self.encoder(state)
        features = features.detach()

        if self.is_discrete:
            return self._discrete_forward(features, deterministic)
        else:
            return self._continuous_forward(features, deterministic)

    # ------------------------------------------------------------------
    # Policy distribution helpers (mirrors PPO_Actor)
    # ------------------------------------------------------------------

    def _continuous_forward(self, features, deterministic=False):
        mu = self.model(features)
        logstd = torch.clip(self.logstd, -5, 2)
        std = torch.exp(logstd.expand_as(mu))
        dist = Normal(loc=mu, scale=std)
        a = dist.rsample()
        logprobs = dist.log_prob(a).unsqueeze(-1).sum(1)
        probs = torch.exp(logprobs)
        entropy = dist.entropy().sum(1)
        return a, {"dist": dist, "probs": probs, "logprobs": logprobs, "entropy": entropy}

    def _discrete_forward(self, features, deterministic=False):
        logits = self.model(features)
        if deterministic:
            a = torch.argmax(logits, dim=-1)
            dist = None
            logprobs = torch.zeros_like(logits[:, 0:1])
            probs = torch.ones_like(logprobs)
            entropy = torch.zeros_like(logprobs)
        else:
            dist = Categorical(logits=logits)
            a = dist.sample()
            logprobs = dist.log_prob(a).unsqueeze(-1)
            probs = torch.exp(logprobs)
            entropy = dist.entropy()
        a = F.one_hot(a, num_classes=logits.size(-1))
        return a, {"dist": dist, "probs": probs, "logprobs": logprobs, "entropy": entropy}

    def log_prob(self, dist, actions):
        actions = actions.squeeze() if actions.shape[-1] > 1 else actions
        if self.is_discrete:
            return dist.log_prob(torch.argmax(actions, dim=-1)).unsqueeze(-1)
        else:
            return dist.log_prob(actions).unsqueeze(-1).sum(1)

    def entropy(self, dist):
        if self.is_discrete:
            return dist.entropy().unsqueeze(-1)
        else:
            return dist.entropy().unsqueeze(-1).sum(1)
