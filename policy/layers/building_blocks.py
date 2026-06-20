from typing import Optional, Union

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[list[int], tuple[int]],
        output_dim: Optional[int] = None,
        activation: nn.Module = nn.ReLU(),
        initialization: str = "default",
        dropout_rate: Optional[float] = None,
        device=torch.device("cpu"),
    ) -> None:
        super().__init__()
        hidden_dims = [input_dim] + hidden_dims
        model = []

        # Derive gain from actual activation type (isinstance avoids false-negative == on modules)
        if isinstance(activation, nn.ReLU):
            gain = nn.init.calculate_gain("relu")       # sqrt(2) ≈ 1.414
        elif isinstance(activation, nn.LeakyReLU):
            gain = nn.init.calculate_gain("leaky_relu")
        elif isinstance(activation, nn.Tanh):
            gain = nn.init.calculate_gain("tanh")       # 5/3 ≈ 1.667
        elif isinstance(activation, nn.Sigmoid):
            gain = nn.init.calculate_gain("sigmoid")
        else:
            gain = 1.0

        # Initialize hidden layers
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            linear_layer = nn.Linear(in_dim, out_dim)
            if initialization == "default":
                nn.init.xavier_uniform_(linear_layer.weight, gain=gain)
                linear_layer.bias.data.fill_(0.01)

            elif initialization == "actor":
                nn.init.orthogonal_(linear_layer.weight, gain=gain)
                linear_layer.bias.data.fill_(0.0)

            elif initialization == "critic":
                nn.init.orthogonal_(linear_layer.weight, gain=gain)
                linear_layer.bias.data.fill_(0.0)

            model += (
                [linear_layer, activation] if activation is not None else [linear_layer]
            )

            if dropout_rate is not None:
                model += [nn.Dropout(p=dropout_rate)]

        self.output_dim = hidden_dims[-1]

        # Initialize output layer
        if output_dim is not None:
            linear_layer = nn.Linear(hidden_dims[-1], output_dim)
            if initialization == "default":
                nn.init.xavier_uniform_(linear_layer.weight, gain=gain)
                linear_layer.bias.data.fill_(0.0)

            elif initialization == "actor":
                nn.init.orthogonal_(linear_layer.weight, gain=gain)
                linear_layer.bias.data.fill_(0.0)

            elif initialization == "critic":
                nn.init.orthogonal_(linear_layer.weight, gain=gain)
                linear_layer.bias.data.fill_(0.0)

            model += [linear_layer]
            self.output_dim = output_dim

        self.model = nn.Sequential(*model).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class CNN(nn.Module):
    """
    Classic CNN architecture from the Nature DQN paper.
    Used for processing pixel-based observations.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        features_dim: int = 512,
        initialization: str = "actor",
        activation: nn.Module = nn.ReLU(),
        device=torch.device("cpu"),
    ):
        super().__init__()
        # input_shape is expected to be (Channels, Height, Width)
        n_input_channels = input_shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=8, stride=4, padding=0),
            activation,
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            activation,
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            activation,
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass with a dummy tensor
        with torch.no_grad():
            dummy_tensor = torch.zeros(1, *input_shape)
            self.n_flatten = self.cnn(dummy_tensor).shape[1]

        # Made the linear head deeper here
        self.linear = nn.Sequential(
            nn.Linear(self.n_flatten, features_dim),
            activation,
        )
        self.output_dim = features_dim

        # Apply orthogonal initialization standard for PPO
        if initialization in ["actor", "critic"]:
            gain = nn.init.calculate_gain("relu")
            for module in self.cnn.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=gain)
                    module.bias.data.fill_(0.0)
            for module in self.linear.modules():
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=gain)
                    module.bias.data.fill_(0.0)

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PPO requires inputs in [0, 1]. Ensure you divide image arrays by 255.0 before this step
        x = self.cnn(x)
        x = self.linear(x)
        return x


class ConvVAEEncoder(nn.Module):
    """
    CNN encoder with VAE heads for jointly learning representations and RL.

    Used by IRPO on Atari: the CNN is trained simultaneously via VAE reconstruction
    loss and via the IRPO meta-gradient.

    forward(x) → mu (deterministic latent), used as features by the policy.
    vae_loss(x) → VAE loss (reconstruction + KL) on raw pixel inputs, used for
                  the separate representation-learning update.

    input_shape: (C, H, W) in CHW format.
    output_dim: latent_dim, compatible with CNN.output_dim for the MLP head.
    """

    def __init__(
        self,
        input_shape: tuple,
        latent_dim: int = 256,
        activation: nn.Module = nn.ReLU(),
        device=torch.device("cpu"),
    ):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.output_dim = latent_dim

        C = input_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(C, 16, kernel_size=8, stride=4, padding=0),
            activation,
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            activation,
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            activation,
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            n_flatten = self.cnn(dummy).shape[1]
            # Spatial shape before flatten (for decoder)
            conv_only = nn.Sequential(*list(self.cnn.children())[:-1])
            self._spatial_shape = tuple(conv_only(dummy).shape[1:])  # (C', H', W')

        self.mu_head = nn.Linear(n_flatten, latent_dim)
        self.logvar_head = nn.Linear(n_flatten, latent_dim)

        # Decoder mirrors the encoder for reconstruction loss
        Cs, Hs, Ws = self._spatial_shape
        self.decoder_project = nn.Linear(latent_dim, Cs * Hs * Ws)
        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(Cs, 32, kernel_size=3, stride=1),
            activation,
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, output_padding=1),
            activation,
            nn.ConvTranspose2d(16, C, kernel_size=8, stride=4),
            nn.Sigmoid(),
        )

        gain = nn.init.calculate_gain("relu")
        for module in self.cnn.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=gain)
                module.bias.data.fill_(0.0)

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) already preprocessed (channel added, normalized to [0,1]).
        Returns mu latent (B, latent_dim) for policy inference.
        """
        features = self.cnn(x)
        return self.mu_head(features)

    def vae_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: raw pixel tensor (B, H, W) or (B, C, H, W), values in [0, 255] or [0, 1].
        Returns scalar VAE loss (reconstruction + beta*KL).
        """
        import torch.nn.functional as F

        # Normalize and add channel if needed
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.max() > 1.0:
            x = x / 255.0

        features = self.cnn(x)
        mu = self.mu_head(features)
        logvar = self.logvar_head(features).clamp(-5, 2)

        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        Cs, Hs, Ws = self._spatial_shape
        proj = self.decoder_project(z).view(-1, Cs, Hs, Ws)
        recon = self.decoder_deconv(proj)
        recon = F.interpolate(recon, size=self.input_shape[1:], mode="bilinear", align_corners=False)

        recon_loss = F.mse_loss(recon, x)
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()

        return recon_loss + 1e-3 * kl_loss
