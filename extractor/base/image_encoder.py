import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from policy.layers.building_blocks import CNN


class ImageEncoder(nn.Module):
    """
    CNN encoder mapping (B, C, H, W) images → (B, encoder_dim) feature vectors.
    Wraps the existing Nature-DQN CNN building block.
    """

    def __init__(self, input_chw, encoder_dim=256, device=torch.device("cpu")):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.backbone = CNN(input_shape=input_chw, features_dim=encoder_dim, device=device)

    def forward(self, x):
        return self.backbone(x)


class _ConvDecoder(nn.Module):
    """
    Mirrored ConvTranspose decoder for the Nature-DQN encoder.
    Uses bilinear upsampling at the end to match the exact target spatial size,
    which avoids hardcoding output_padding for every possible input resolution.
    """

    def __init__(self, encoder_dim, intermediate_chw, target_chw, device=torch.device("cpu")):
        super().__init__()
        C, H, W = intermediate_chw        # spatial shape after CNN conv layers, e.g. (64, 22, 16)
        self.intermediate_chw = intermediate_chw
        self.target_chw = target_chw      # (C_out, H_out, W_out)

        self.project = nn.Linear(encoder_dim, C * H * W)
        self.deconv = nn.Sequential(
            # Reverse Conv3 (k=3, s=1): (C, H, W) → (32, H+2, W+2)
            nn.ConvTranspose2d(C, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            # Reverse Conv2 (k=4, s=2): → (16, ~2(H+2), ~2(W+2))
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, output_padding=1),
            nn.ReLU(),
            # Reverse Conv1 (k=8, s=4): → (C_out, ~, ~)
            nn.ConvTranspose2d(16, target_chw[0], kernel_size=8, stride=4),
            nn.Sigmoid(),
        )
        self.to(device)

    def forward(self, z):
        B = z.shape[0]
        C, H, W = self.intermediate_chw
        x = self.project(z).view(B, C, H, W)
        x = self.deconv(x)
        # Resize to exact target shape — handles any output_padding discrepancy
        x = F.interpolate(x, size=self.target_chw[1:], mode="bilinear", align_corners=False)
        return x


class ConvAutoEncoder(nn.Module):
    """
    Convolutional autoencoder for self-supervised image encoder pretraining.

    Encode: (B, C, H, W) → (B, encoder_dim)
    Decode: (B, encoder_dim) → (B, C, H, W) [reconstruction]
    """

    def __init__(self, input_chw, encoder_dim=256, device=torch.device("cpu")):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.input_chw = input_chw

        self.encoder = ImageEncoder(input_chw, encoder_dim, device)

        # Compute intermediate spatial shape after CNN conv layers (before flatten+linear)
        with torch.no_grad():
            # Put dummy on the same device as the encoder weights to avoid device mismatch
            _dev = next(self.encoder.parameters()).device
            dummy = torch.zeros(1, *input_chw, device=_dev)
            # CNN.cnn is a Sequential ending with nn.Flatten — strip it to get (C, H, W)
            conv_only = nn.Sequential(*list(self.encoder.backbone.cnn.children())[:-1])
            spatial = conv_only(dummy)                    # (1, C', H', W')
            intermediate_chw = tuple(spatial.shape[1:])  # (C', H', W')

        self.decoder = _ConvDecoder(encoder_dim, intermediate_chw, input_chw, device)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

    def reconstruction_loss(self, x):
        recon, _ = self(x)
        return F.mse_loss(recon, x)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pretrain_image_encoder(
    env,
    seed: int,
    encoder_dim: int = 256,
    epochs: int = 5000,
    batch_size: int = 64,
    lr: float = 3e-4,
    n_frames: int = 50_000,
    device: str = "cpu",
) -> ImageEncoder:
    """
    Load or train a CNN image encoder using reconstruction loss on random frames.

    Saves encoder weights to  model/<env_name>/encoders/encoder_<seed>.pth  so that
    subsequent runs skip training entirely.

    Args:
        env:         Gymnasium environment (already wrapped). Used only if training.
        seed:        Random seed, embedded in the checkpoint filename.
        encoder_dim: Dimension of the latent representation (default 256).
        epochs:      Number of gradient steps for the autoencoder.
        batch_size:  Mini-batch size per gradient step.
        lr:          Adam learning rate.
        n_frames:    Number of random frames to collect for training.
        device:      Torch device string.

    Returns:
        Pretrained ImageEncoder in eval mode.
    """
    env_name = _get_env_name(env)
    os.makedirs(f"model/{env_name}", exist_ok=True)
    encoder_dir = f"model/{env_name}/encoders"
    os.makedirs(encoder_dir, exist_ok=True)
    model_path = f"{encoder_dir}/encoder_{seed}.pth"
    legacy_model_path = f"model/{env_name}/encoder_{seed}.pth"

    # Determine image shape from the environment observation space
    obs_shape = env.observation_space.shape  # e.g. (210, 160) grayscale
    if len(obs_shape) == 2:
        chw_shape = (1, obs_shape[0], obs_shape[1])   # (H, W) → (1, H, W)
    elif len(obs_shape) == 3 and obs_shape[2] in (1, 3):
        chw_shape = (obs_shape[2], obs_shape[0], obs_shape[1])  # (H, W, C) → (C, H, W)
    else:
        raise ValueError(f"Unsupported observation shape: {obs_shape}")

    encoder = ImageEncoder(chw_shape, encoder_dim, device)

    if os.path.exists(model_path):
        print(f"[INFO] Loading pretrained encoder from {model_path}")
        encoder.load_state_dict(torch.load(model_path, map_location=device))
        encoder.eval()
        return encoder
    if os.path.exists(legacy_model_path):
        print(
            f"[INFO] Loading pretrained encoder from legacy path {legacy_model_path}"
        )
        encoder.load_state_dict(torch.load(legacy_model_path, map_location=device))
        encoder.eval()
        return encoder

    print(f"[INFO] Pretraining image encoder → {model_path}")
    print(f"[INFO] Collecting {n_frames} random frames …")
    frames = _collect_frames(env, n_frames, seed)
    frames_tensor = _preprocess_frames(frames, chw_shape)  # (N, C, H, W) float32 in [0,1]
    N = frames_tensor.shape[0]
    print(f"[INFO] Collected {N} frames. Training autoencoder for {epochs} steps …")

    autoencoder = ConvAutoEncoder(chw_shape, encoder_dim, device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    best_loss = float("inf")

    with tqdm(total=epochs, desc="Encoder Pretraining") as pbar:
        for _ in range(epochs):
            idx = torch.randint(0, N, (batch_size,))
            batch = frames_tensor[idx].to(device)

            loss = autoencoder.reconstruction_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 1.0)
            optimizer.step()

            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(autoencoder.encoder.state_dict(), model_path)

    print(f"[INFO] Encoder saved to {model_path}  (best recon loss: {best_loss:.4f})")
    encoder.load_state_dict(torch.load(model_path, map_location=device))
    encoder.eval()
    return encoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_env_name(env) -> str:
    spec = getattr(env, "spec", None)
    if spec is not None and hasattr(spec, "id"):
        raw = spec.id.lower()
        # strip version suffix, e.g. "ale/pacman-v5" → "pacman"
        name = raw.split("/")[-1].split("-")[0]
        return name
    # fallback: class name
    return type(env).__name__.lower()


def _collect_frames(env, n_frames: int, seed: int):
    """Collect n_frames observations using a random policy."""
    frames = []
    obs, _ = env.reset(seed=seed)
    frames.append(obs)

    rng = np.random.default_rng(seed)
    while len(frames) < n_frames:
        action = env.action_space.sample()
        obs, _, term, trunc, _ = env.step(action)
        frames.append(obs)
        if term or trunc:
            obs, _ = env.reset()
            frames.append(obs)

    return frames[:n_frames]


def _preprocess_frames(frames, chw_shape):
    """Stack raw frames into (N, C, H, W) float32 tensor in [0, 1]."""
    arr = np.stack(frames, axis=0)  # (N, H, W) or (N, H, W, C)

    if arr.ndim == 3:                   # grayscale (N, H, W)
        arr = arr[:, np.newaxis, :, :]  # → (N, 1, H, W)
    elif arr.ndim == 4:                 # colour (N, H, W, C)
        arr = arr.transpose(0, 3, 1, 2) # → (N, C, H, W)

    return torch.from_numpy(arr.astype(np.float32)) / 255.0
