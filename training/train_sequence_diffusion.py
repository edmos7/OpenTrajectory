#!/usr/bin/env python3
"""Train a compact trajectory-conditioned Diffusion Policy.

This is the main sequence-conditioned Open Trajectory trainer. It keeps the
trajectory/DINO/depth/state fusion architecture, uses compact tiny-style
capacity defaults, and can train either a stock diffusion U-Net or a local
DiT-style action denoiser via ``--denoiser-backbone``.

The trajectory preprocessing treats ``observation.global_trajectory`` as an
active manipulation trajectory:
the action channel is treated as already-active when it is > 0, then converted
to a first-active through last-active binary interval. No threshold is used for
that training-time trajectory preprocessing.

Typical usage:
    huggingface-cli login
    python training/train_sequence_diffusion.py

Common overrides:
    python training/train_sequence_diffusion.py \
        --denoiser-backbone dit \
        --output-dir outputs/train/sequence_dit_test \
        --num-steps 2000 \
        --batch-size 8
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import signal
import sys
import textwrap
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration - mirrors train_diffusion_tracks.ipynb defaults
# ---------------------------------------------------------------------------

# Dataset
DATASET_REPO_ID = "edmos7/opentrajectory_dataset"
DATASET_ROOT = None
EPISODES = None
# Episode 70 has an invalid global trajectory caused by a false-positive
# tracker detection and must not be used for training or validation.
EXCLUDED_EPISODES = {}
EXCLUDED_TASK_IDS: set[int] = set()

# Output
OUTPUT_DIR = SCRIPT_DIR.parent / "outputs/train/sequence_diffusion"

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2
SEED = 42

# Capacity preset: "tinier", "tiny", "small", "medium", "large", or "custom"
CAPACITY_PRESET = "tiny"

# Input features
USE_GLOBAL_TRAJ = True
USE_MAIN_TRACKS = False
USE_SEC_TRACKS = False
USE_MAIN_CAM = False
USE_SEC_CAM = False
USE_JOINT_STATE = False
GLOBAL_TRAJ_LENGTH = 224

MAIN_CAM_KEY = "observation.images.side"
SEC_CAM_KEY = "observation.images.wrist"

# Diffusion action horizon
HORIZON = 32
N_ACTION_STEPS = 16
N_OBS_STEPS = 2
# Observation frames relative to the current frame. Keep length equal to
# N_OBS_STEPS and include 0 as the final/current observation.
OBSERVATION_FRAME_OFFSETS = (-8,0)

# UNet architecture
DOWN_DIMS = (256, 512, 1024)
KERNEL_SIZE = 5
N_GROUPS = 8
DIFFUSION_STEP_EMBED_DIM = 128
USE_FILM_SCALE_MODULATION = True

# Action denoiser backbone. "dit" keeps all conditioning/preprocessing the same
# but replaces the temporal U-Net with a DiT-style transformer over action tokens.
DENOISER_BACKBONE = "dit"
DIT_DIM = 256
DIT_DEPTH = 6
DIT_NHEAD = 8
DIT_FFN_DIM = 1024
DIT_DROPOUT = 0.05

# Vision backbone for LeRobot's built-in ResNet path
VISION_BACKBONE = "resnet18"
PRETRAINED_BACKBONE = "ResNet18_Weights.IMAGENET1K_V1"
RESIZE_SHAPE = (96, 96)
CROP_RATIO = 0.84
SPATIAL_SOFTMAX_KP = 32

# Noise scheduler
NOISE_SCHEDULER_TYPE = "DDPM"
NUM_TRAIN_TIMESTEPS = 100
NUM_INFERENCE_STEPS = None
BETA_SCHEDULE = "squaredcos_cap_v2"
PREDICTION_TYPE = "epsilon"
CLIP_SAMPLE = True
CLIP_SAMPLE_RANGE = 1.0

# Optimizer
LR = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 10.0

# LR scheduler: "cosine", "plateau", or "none"
LR_SCHEDULER_TYPE = "cosine"
LR_WARMUP_STEPS = 1000
LR_DECAY_STEPS = 75_000
LR_PEAK = LR
LR_FLOOR = 1e-8
PLATEAU_FACTOR = 0.1
PLATEAU_PATIENCE = 0
PLATEAU_MIN_LR = 1e-7
VAL_EPISODE_STRIDE = 5
VAL_FREQ_STEPS = None
VAL_SPLIT_SEED = 42

# Training budget
BATCH_SIZE = 64
NUM_TRAIN_STEPS = 100_000
NUM_EPOCHS = 300
STRATIFY_BATCHES_BY_TASK_ID = True

# Logging and checkpointing
LOG_FREQ = 200
SAVE_FREQ = None
SAVE_EVERY_N_EPOCHS = 50
PLOT_FREQ = 1000
RUN_SUBDIRS = True

# Trajectory encoder
USE_TRAJ_ENCODER = True
TRAJ_ENC_TYPE = "transformer"
TRAJ_ENC_DIM = 16
FOURIER_BANDS = 8
TRAJ_USE_FOURIER_FEATURES = False
TRAJ_TEMPORAL_DIM = 16
TRAJ_ENC_HIDDEN = 32
TRAJ_ENC_LAYERS = 3
TRAJ_CONV_KERNEL_SIZES = None
TRAJ_CONV_DILATIONS = None
TRAJ_TF_D_MODEL = 16
TRAJ_TF_NHEAD = 4
TRAJ_TF_FFN_DIM = 32
TRAJ_TF_LAYERS = 2
TRAJ_ACTION_ENCODING = "none"
TRAJ_ACTION_BINARY_THRESHOLD = 0.25
TRAJ_ACTIVE_ACTION_SIGNAL = True
TRAJ_ACTIVE_WINDOW_ONLY = True
TRAJ_NUM_QUERY_TOKENS = 4

# Conditioning fusion
USE_CONDITIONING_FUSION = True
FUSION_TOKEN_DIM = 32
FUSION_OUT_DIM = 64
FUSION_NHEAD = 4
FUSION_FFN_DIM = 128
FUSION_LAYERS = 2
FUSION_NUM_OUTPUT_TOKENS = 1

# Trajectory augmentation
TRAJ_PAST_MASK_PROB = 0.10
TRAJ_NOISE_STD = 0.01
TRAJ_NOISE_SMOOTH_SIGMA = 30.0
TRAJ_RETURN_CUTOFF = True
TRAJ_RETURN_CUTOFF_SMOOTH_WINDOW = 15
TRAJ_RETURN_CUTOFF_SMOOTH_POLY = 2
TRAJ_RETURN_CUTOFF_GRADIENT_THRESHOLD = 0.001

# DINO visual encoder
USE_DINO_CAM = False
DINO_MODEL = "dinov2_vits14"
DINO_OUT_DIM = 32
DINO_PROJ_HIDDEN_DIM = None
DINO_FROZEN = True
DINO_RESIZE = (224, 224)
DINO_VIS_AUG = True
DINO_AUG_BRIGHTNESS = 0.30
DINO_AUG_CONTRAST = 0.30
DINO_AUG_SATURATION = 0.25
DINO_AUG_HUE = 0.06
DINO_AUG_NONE_PROB = 0.25
DINO_AUG_GRAYSCALE_PROB = 0.25
DINO_AUG_SOLARIZE_PROB = 0.25
DINO_AUG_SOLARIZE_THRESHOLD = 0.85
DINO_AUG_CROP_SCALE = (1.00, 1.00)
DINO_AUG_CROP_RATIO = (1.00, 1.00)
VISION_DROPOUT_PROB = 0.20

# External torchvision ResNet visual encoder. When enabled, selected camera
# frames are encoded into ENV features before the policy, like the Colab
# working notebook. This is separate from LeRobot's built-in VISUAL path.
USE_RESNET_ENV_ENCODER = True
RESNET_OUT_DIM = 32
RESNET_PROJ_HIDDEN_DIM = None
RESNET_FROZEN = False
RESNET_RESIZE = (224, 224)
RESNET_LR = 1e-5

USE_DEPTH_ENCODER = False
DEPTH_PREFIX = "observation.depth"
DEPTH_OUT_DIM = 32
DEPTH_RESIZE = (224, 224)
DEPTH_DROPOUT_PROB = 0.10
STATE_DROPOUT_PROB = 0.05
STATE_NOISE_STD = 0.01

# Cheap conditioning-usage diagnostics on one fixed training batch.
CONDITIONING_DIAG_FREQ = 500
CONDITIONING_DIAG_SEED = 12345

CAPACITY_PRESETS = {
    "tinier": {
        "DOWN_DIMS": (24, 48, 96),
        "DIFFUSION_STEP_EMBED_DIM": 64,
        "TRAJ_ENC_DIM": 64,
        "TRAJ_TF_D_MODEL": 56,
        "TRAJ_TF_NHEAD": 4,
        "TRAJ_TF_FFN_DIM": 288,
        "TRAJ_TF_LAYERS": 2,
        "DINO_OUT_DIM": 260,
        "DEPTH_OUT_DIM": 32,
        "FUSION_TOKEN_DIM": 40,
        "FUSION_OUT_DIM": 64,
        "FUSION_NHEAD": 4,
        "FUSION_FFN_DIM": 96,
        "FUSION_LAYERS": 3,
        "DIT_DIM": 128,
        "DIT_DEPTH": 4,
        "DIT_NHEAD": 4,
        "DIT_FFN_DIM": 512,
    },
    "tiny": {
        "DOWN_DIMS": (56, 112, 224),
        "TRAJ_ENC_TYPE": "conv1d",
        "TRAJ_ENC_DIM": 96,
        "TRAJ_ENC_HIDDEN": 128,
        "TRAJ_ENC_LAYERS": 4,
        "TRAJ_CONV_KERNEL_SIZES": (7, 5, 3, 3),
        "TRAJ_CONV_DILATIONS": (1, 2, 4, 8),
        "TRAJ_TF_D_MODEL": 96,
        "TRAJ_TF_NHEAD": 4,
        "TRAJ_TF_FFN_DIM": 320,
        "TRAJ_TF_LAYERS": 2,
        "TRAJ_NUM_QUERY_TOKENS": 0,
        "DINO_OUT_DIM": 520,
        "DINO_PROJ_HIDDEN_DIM": 220,
        "DEPTH_OUT_DIM": 812,
        "FUSION_TOKEN_DIM": 64,
        "FUSION_OUT_DIM": 96,
        "FUSION_NHEAD": 4,
        "FUSION_FFN_DIM": 256,
        "FUSION_LAYERS": 2,
        "DIT_DIM": 256,
        "DIT_DEPTH": 6,
        "DIT_NHEAD": 8,
        "DIT_FFN_DIM": 1024,
    },
    "small": {
        "DOWN_DIMS": (128, 256, 512),
        "TRAJ_ENC_DIM": 16,
        "TRAJ_TF_D_MODEL": 16,
        "TRAJ_TF_NHEAD": 4,
        "TRAJ_TF_FFN_DIM": 32,
        "TRAJ_TF_LAYERS": 2,
        "DINO_OUT_DIM": 32,
        "DEPTH_OUT_DIM": 32,
        "DIT_DIM": 384,
        "DIT_DEPTH": 8,
        "DIT_NHEAD": 8,
        "DIT_FFN_DIM": 1536,
    },
    "medium": {
        "DOWN_DIMS": (256, 512, 1024),
        "TRAJ_ENC_DIM": 16,
        "TRAJ_TF_D_MODEL": 16,
        "TRAJ_TF_NHEAD": 4,
        "TRAJ_TF_FFN_DIM": 32,
        "TRAJ_TF_LAYERS": 2,
        "DINO_OUT_DIM": 32,
        "DEPTH_OUT_DIM": 32,
        "DIT_DIM": 512,
        "DIT_DEPTH": 10,
        "DIT_NHEAD": 8,
        "DIT_FFN_DIM": 2048,
    },
    "large": {
        "DOWN_DIMS": (512, 1024, 2048),
        "TRAJ_ENC_DIM": 16,
        "TRAJ_TF_D_MODEL": 16,
        "TRAJ_TF_NHEAD": 4,
        "TRAJ_TF_FFN_DIM": 32,
        "TRAJ_TF_LAYERS": 2,
        "DINO_OUT_DIM": 32,
        "DEPTH_OUT_DIM": 32,
        "DIT_DIM": 768,
        "DIT_DEPTH": 12,
        "DIT_NHEAD": 12,
        "DIT_FFN_DIM": 3072,
    },
}


# ---------------------------------------------------------------------------
# Model helpers from the notebook
# ---------------------------------------------------------------------------


class FlattenTracksTransform:
    """Flatten selected track tensors into observation.environment_state."""

    def __init__(self, track_keys: list[str], use_joint_state: bool = True):
        self.track_keys = track_keys
        self.use_joint_state = use_joint_state

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if not self.use_joint_state and "observation.state" in sample:
            s = sample["observation.state"]
            sample["observation.state"] = torch.zeros(
                *s.shape[:-1], 1, dtype=s.dtype, device=s.device
            )

        parts = []
        for key in self.track_keys:
            if key in sample:
                parts.append(sample.pop(key).flatten(start_dim=2))

        if parts:
            sample["observation.environment_state"] = torch.cat(parts, dim=-1)
        return sample


class FourierFeatures(nn.Module):
    """Map normalized (x, y) coordinates to sinusoidal Fourier features."""

    def __init__(self, bands: int = 8):
        super().__init__()
        freqs = (2.0 ** torch.arange(bands)) * math.pi
        self.register_buffer("freqs", freqs)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        f = xy.unsqueeze(-1) * self.freqs
        return torch.cat([f.sin(), f.cos()], dim=-1).flatten(-2)


class TrajectoryEncoder(nn.Module):
    """Encode global_trajectory [B, n_obs_steps, max_T, 3] to env features."""

    def __init__(
        self,
        max_T: int,
        fourier_bands: int = 8,
        temporal_dim: int = 16,
        out_dim: int = 64,
        enc_type: str = "transformer",
        hidden: int = 128,
        n_layers: int = 3,
        conv_kernel_sizes: tuple[int, ...] | None = None,
        conv_dilations: tuple[int, ...] | None = None,
        d_model: int = 64,
        nhead: int = 4,
        ffn_dim: int = 128,
        tf_layers: int = 2,
        action_encoding: str = "none",
        action_binary_threshold: float = 0.25,
        num_query_tokens: int = 0,
        use_fourier_features: bool = False,
    ):
        super().__init__()
        self.max_T = max_T
        self.enc_type = enc_type
        self.action_encoding = action_encoding
        self.action_binary_threshold = action_binary_threshold
        self.num_query_tokens = num_query_tokens
        self.use_fourier_features = use_fourier_features
        self.fourier = FourierFeatures(fourier_bands) if use_fourier_features else nn.Identity()
        self.temporal_emb = nn.Embedding(max_T, temporal_dim)

        xy_dim = 4 * fourier_bands if use_fourier_features else 2
        action_dim = 0 if action_encoding == "none" else 1
        in_ch = xy_dim + action_dim + temporal_dim

        if enc_type == "conv1d":
            kernel_sizes = tuple(conv_kernel_sizes or (3,) * n_layers)
            dilations = tuple(conv_dilations or (1,) * n_layers)
            if len(kernel_sizes) != n_layers:
                raise ValueError("conv_kernel_sizes length must equal n_layers")
            if len(dilations) != n_layers:
                raise ValueError("conv_dilations length must equal n_layers")
            layers = []
            ch = in_ch
            for i in range(n_layers):
                out_ch = hidden if i < n_layers - 1 else out_dim
                kernel_size = kernel_sizes[i]
                dilation = dilations[i]
                layers += [
                    nn.Conv1d(
                        ch,
                        out_ch,
                        kernel_size=kernel_size,
                        padding=dilation * (kernel_size // 2),
                        dilation=dilation,
                    ),
                    nn.GroupNorm(min(8, out_ch), out_ch),
                    nn.Mish(),
                ]
                ch = out_ch
            self.backbone = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.out_proj = nn.Identity()
        elif enc_type == "transformer":
            self.input_proj = nn.Linear(in_ch, d_model)
            self.query_tokens = (
                nn.Parameter(torch.randn(num_query_tokens, d_model) * 0.02)
                if num_query_tokens > 0
                else None
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=ffn_dim,
                batch_first=True,
                norm_first=True,
            )
            self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=tf_layers)
            self.out_proj = nn.Linear(d_model, out_dim) if d_model != out_dim else nn.Identity()
        else:
            raise ValueError(f"Unknown trajectory encoder type: {enc_type!r}")

    def forward(self, x: torch.Tensor, return_tokens: bool = False) -> torch.Tensor:
        B, T_obs, max_T, _ = x.shape
        x_flat = x.reshape(B * T_obs, max_T, 3)

        xy = x_flat[..., :2]
        action_feat = None
        action = x_flat[..., 2:3]
        delta_a = torch.diff(action, dim=1, prepend=action[:, :1, :])
        if self.action_encoding == "delta":
            action_feat = delta_a
        elif self.action_encoding == "binary":
            tolerance = torch.finfo(action.dtype).eps * 4
            action_feat = (
                action >= self.action_binary_threshold - tolerance
            ).to(action.dtype)
        elif self.action_encoding == "none":
            action_feat = None
        else:
            raise ValueError(f"Unknown trajectory action encoding: {self.action_encoding!r}")

        xy_enc = self.fourier(xy)
        t_idx = torch.arange(max_T, device=x.device)
        t_emb = self.temporal_emb(t_idx).unsqueeze(0).expand(x_flat.shape[0], -1, -1)
        feat_parts = [xy_enc, t_emb] if action_feat is None else [xy_enc, action_feat, t_emb]
        feat = torch.cat(feat_parts, dim=-1)

        if self.enc_type == "conv1d":
            out = self.backbone(feat.transpose(1, 2))
            out = self.pool(out).squeeze(-1)
        else:
            feat = self.input_proj(feat)
            if self.query_tokens is not None:
                queries = self.query_tokens.unsqueeze(0).expand(feat.shape[0], -1, -1)
                feat = torch.cat([queries, feat], dim=1)
            feat = self.backbone(feat)
            if self.query_tokens is not None:
                out = self.out_proj(feat[:, : self.num_query_tokens])
                if not return_tokens:
                    out = out.flatten(start_dim=1)
            else:
                out = self.out_proj(feat.mean(dim=1))
        if return_tokens and out.ndim == 3:
            return out.reshape(B, T_obs, out.shape[1], out.shape[2])
        return out.reshape(B, T_obs, -1)


class TrajAugTransform:
    """Training-only path shaping, noise, and past masking for global trajectories.

    Tip-tracked/redrawn datasets store the active manipulation interval directly
    in the trajectory action channel. Preprocessing therefore finds active
    samples with ``action > 0`` instead of re-thresholding a continuous gripper
    signal.
    """

    def __init__(
        self,
        p_mask: float = 0.4,
        noise_std: float = 0.01,
        smooth_sigma: float = 30.0,
        active_action_signal: bool = True,
        active_window_only: bool = True,
        action_binary_threshold: float = 0.25,
        return_cutoff: bool = True,
        return_cutoff_smooth_window: int = 15,
        return_cutoff_smooth_poly: int = 2,
        return_cutoff_gradient_threshold: float = 0.001,
        traj_key: str = "observation.global_trajectory",
    ):
        self.p_mask = p_mask
        self.noise_std = noise_std
        self.smooth_sigma = smooth_sigma
        self.active_action_signal = active_action_signal
        self.active_window_only = active_window_only
        self.action_binary_threshold = action_binary_threshold
        self.return_cutoff = return_cutoff
        self.return_cutoff_smooth_window = return_cutoff_smooth_window
        self.return_cutoff_smooth_poly = return_cutoff_smooth_poly
        self.return_cutoff_gradient_threshold = return_cutoff_gradient_threshold
        self.traj_key = traj_key

    def _smooth_noise(self, traj: torch.Tensor) -> torch.Tensor:
        """Generate long-wavelength xy deformations while preserving path continuity."""
        path_length = traj.shape[-2]
        if path_length < 2 or self.smooth_sigma <= 0:
            return torch.randn_like(traj[..., :2]) * self.noise_std

        half_width = min(int(3 * self.smooth_sigma), path_length - 1)
        positions = torch.arange(
            -half_width,
            half_width + 1,
            dtype=traj.dtype,
            device=traj.device,
        )
        kernel = torch.exp(-(positions**2) / (2 * self.smooth_sigma**2))
        kernel = (kernel / kernel.sum()).view(1, 1, -1)

        raw = torch.randn_like(traj[..., :2]).movedim(-2, -1)
        leading_shape = raw.shape[:-1]
        raw = raw.reshape(-1, 1, path_length)
        smoothed = F.conv1d(
            F.pad(raw, (half_width, half_width), mode="reflect"),
            kernel,
        ).reshape(*leading_shape, path_length)

        std = smoothed.std(dim=-1, keepdim=True).clamp_min(1e-8)
        return (smoothed / std * self.noise_std).movedim(-1, -2)

    @staticmethod
    def _active_mask(action: torch.Tensor) -> torch.Tensor:
        return action > 0

    def _first_active_index(self, action: torch.Tensor) -> int | None:
        active = self._active_mask(action)
        idx = torch.nonzero(active, as_tuple=False)
        return int(idx[0].item()) if idx.numel() else None

    def _last_active_index(self, action: torch.Tensor) -> int | None:
        active = self._active_mask(action)
        idx = torch.nonzero(active, as_tuple=False)
        return int(idx[-1].item()) if idx.numel() else None

    def _apply_active_action_signal(self, sequence: torch.Tensor) -> None:
        first = self._first_active_index(sequence[:, 2])
        last = self._last_active_index(sequence[:, 2])
        sequence[:, 2] = 0.0
        if first is not None and last is not None and last >= first:
            sequence[first : last + 1, 2] = 1.0

    def _apply_active_window_only(self, sequence: torch.Tensor) -> None:
        first = self._first_active_index(sequence[:, 2])
        last = self._last_active_index(sequence[:, 2])
        if first is None or last is None or last < first:
            return

        sequence[:first] = sequence[first]
        sequence[last + 1 :] = sequence[last]
        sequence[:first, 2] = 0.0
        sequence[first : last + 1, 2] = 1.0
        sequence[last + 1 :, 2] = 0.0

    def _sharpest_return_index(self, sequence: torch.Tensor) -> int | None:
        path_length = sequence.shape[0]
        if path_length < 4:
            return None

        start_xy = sequence[0, :2]
        d_raw = torch.linalg.vector_norm(sequence[:, :2] - start_xy, dim=-1)
        win = self.return_cutoff_smooth_window
        if path_length <= win:
            win = max(3, (path_length // 5) | 1)
        if win % 2 == 0:
            win += 1
        if win <= self.return_cutoff_smooth_poly:
            return None

        half = win // 2
        kernel = torch.ones(1, 1, win, dtype=sequence.dtype, device=sequence.device) / win
        smooth = F.conv1d(
            F.pad(d_raw.view(1, 1, -1), (half, half), mode="replicate"),
            kernel,
        ).view(-1)
        gradient = torch.diff(smooth)
        decreasing = gradient < -self.return_cutoff_gradient_threshold
        idx = torch.nonzero(decreasing, as_tuple=False)
        if not idx.numel():
            return None

        padded = torch.cat([
            torch.zeros(1, dtype=torch.bool, device=sequence.device),
            decreasing,
            torch.zeros(1, dtype=torch.bool, device=sequence.device),
        ])
        edges = torch.diff(padded.to(torch.int8))
        starts = torch.nonzero(edges == 1, as_tuple=False).flatten()
        ends = torch.nonzero(edges == -1, as_tuple=False).flatten()
        if not starts.numel():
            return None
        longest = torch.argmax(ends - starts)
        return int(starts[longest].item())

    def _apply_return_cutoff(self, sequence: torch.Tensor) -> None:
        last_active = self._last_active_index(sequence[:, 2])
        if last_active is not None:
            cutoff = last_active + 1
        else:
            cutoff = self._sharpest_return_index(sequence)

        if cutoff is None or cutoff <= 0 or cutoff >= sequence.shape[0]:
            return
        sequence[cutoff:] = sequence[cutoff - 1]
        sequence[cutoff:, 2] = 0.0

    def _apply_path_transforms(self, traj: torch.Tensor) -> torch.Tensor:
        flat = traj.reshape(-1, traj.shape[-2], traj.shape[-1])
        for sequence in flat:
            if self.active_window_only:
                self._apply_active_window_only(sequence)
            elif self.return_cutoff:
                self._apply_return_cutoff(sequence)
            if self.active_action_signal:
                self._apply_active_action_signal(sequence)
        return traj

    def __call__(
        self,
        sample: dict[str, Any],
        mask_indices: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if self.traj_key not in sample:
            return sample

        traj = sample[self.traj_key].clone()
        traj = self._apply_path_transforms(traj)
        if self.noise_std > 0:
            traj[..., :2] = (
                traj[..., :2] + self._smooth_noise(traj)
            ).clamp(0.0, 1.0)

        if self.p_mask > 0 and random.random() < self.p_mask and (
            mask_indices is not None or "frame_index" in sample
        ):
            frame_index = mask_indices if mask_indices is not None else sample["frame_index"]
            if traj.ndim == 4:
                latest = frame_index[:, -1] if frame_index.ndim > 1 else frame_index
                for b, t in enumerate(latest.tolist()):
                    t = int(t)
                    if t > 0:
                        traj[b, :, :t, :] = 0.0
            else:
                t = int(frame_index[-1] if frame_index.ndim > 0 else frame_index)
                if t > 0:
                    traj[:, :t, :] = 0.0

        sample[self.traj_key] = traj
        return sample


def resample_global_trajectory(
    traj: torch.Tensor,
    target_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove repeated terminal padding and linearly resample each trajectory."""
    if traj.ndim == 3:
        traj = traj.unsqueeze(1)
    if traj.ndim != 4 or traj.shape[-1] != 3:
        raise ValueError(f"Expected trajectory [B, T_obs, T, 3], got {tuple(traj.shape)}")
    if target_length <= 1:
        raise ValueError(f"target_length must be greater than one, got {target_length}")

    B, T_obs, padded_length, _ = traj.shape
    output = torch.empty(
        B, T_obs, target_length, 3, dtype=traj.dtype, device=traj.device
    )
    source_lengths = torch.empty(B, T_obs, dtype=torch.long, device=traj.device)

    for b in range(B):
        for t in range(T_obs):
            sequence = traj[b, t]
            differs_from_final = torch.any(sequence != sequence[-1], dim=-1)
            differing = torch.nonzero(differs_from_final, as_tuple=False)
            if differing.numel() == 0:
                source_length = 1
            else:
                source_length = min(int(differing[-1].item()) + 2, padded_length)
            source_lengths[b, t] = source_length

            valid = sequence[:source_length].transpose(0, 1).unsqueeze(0)
            output[b, t] = torch.nn.functional.interpolate(
                valid,
                size=target_length,
                mode="linear",
                align_corners=True,
            ).squeeze(0).transpose(0, 1)

    return output, source_lengths


def rescale_frame_indices(
    frame_index: torch.Tensor,
    source_lengths: torch.Tensor,
    target_length: int,
) -> torch.Tensor:
    """Map original episode frame indices onto the resampled trajectory timeline."""
    if frame_index.ndim == 1:
        frame_index = frame_index.unsqueeze(1)
    lengths = source_lengths
    if lengths.shape[1] != frame_index.shape[1]:
        lengths = lengths[:, :1].expand(-1, frame_index.shape[1])
    scaled = frame_index.float() / (lengths.float() - 1).clamp_min(1)
    return (scaled * (target_length - 1)).round().long().clamp(0, target_length - 1)


class DinoImageAugTransform(nn.Module):
    """Training-only image augmentation before DINO normalization."""

    def __init__(
        self,
        brightness: float = 0.15,
        contrast: float = 0.15,
        saturation: float = 0.10,
        hue: float = 0.03,
        none_prob: float = 0.0,
        grayscale_prob: float = 0.0,
        solarize_prob: float = 0.0,
        solarize_threshold: float = 0.85,
        crop_scale: tuple[float, float] = (0.90, 1.00),
        crop_ratio: tuple[float, float] = (0.95, 1.05),
    ):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.none_prob = none_prob
        self.grayscale_prob = grayscale_prob
        self.solarize_prob = solarize_prob
        self.solarize_threshold = solarize_threshold
        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio

    def _random_resized_crop(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one sampled crop consistently to all T frames of each batch entry."""
        min_scale, max_scale = self.crop_scale
        if min_scale >= 1.0 and max_scale >= 1.0:
            return x

        B, T, _, h, w = x.shape
        area = h * w
        min_ratio, max_ratio = self.crop_ratio
        log_min_ratio = math.log(min_ratio)
        log_max_ratio = math.log(max_ratio)
        out = []

        for i in range(B):
            crop_h, crop_w = h, w
            for _ in range(10):
                target_area = area * float(torch.empty((), device=x.device).uniform_(min_scale, max_scale))
                aspect = math.exp(float(torch.empty((), device=x.device).uniform_(log_min_ratio, log_max_ratio)))
                crop_w = int(round(math.sqrt(target_area * aspect)))
                crop_h = int(round(math.sqrt(target_area / aspect)))
                if 0 < crop_h <= h and 0 < crop_w <= w:
                    break
            else:
                crop_h, crop_w = h, w

            top = int(torch.randint(0, h - crop_h + 1, (), device=x.device).item()) if crop_h < h else 0
            left = int(torch.randint(0, w - crop_w + 1, (), device=x.device).item()) if crop_w < w else 0
            crop = x[i, :, :, top : top + crop_h, left : left + crop_w]
            resized = torch.nn.functional.interpolate(
                crop, size=(h, w), mode="bilinear", align_corners=False
            )
            out.append(resized.unsqueeze(0))

        return torch.cat(out, dim=0)

    def _color_jitter(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        if self.brightness > 0:
            factor = torch.empty(B, 1, 1, 1, 1, device=x.device).uniform_(
                1 - self.brightness, 1 + self.brightness
            )
            x = x * factor

        if self.contrast > 0:
            mean = x.mean(dim=(-2, -1), keepdim=True)
            factor = torch.empty(B, 1, 1, 1, 1, device=x.device).uniform_(
                1 - self.contrast, 1 + self.contrast
            )
            x = (x - mean) * factor + mean

        if self.saturation > 0 and x.shape[2] == 3:
            gray = x[:, :, 0:1] * 0.299 + x[:, :, 1:2] * 0.587 + x[:, :, 2:3] * 0.114
            factor = torch.empty(B, 1, 1, 1, 1, device=x.device).uniform_(
                1 - self.saturation, 1 + self.saturation
            )
            x = (x - gray) * factor + gray

        x = x.clamp(0.0, 1.0)
        if self.hue > 0 and x.shape[2] == 3:
            from torchvision.transforms.functional import adjust_hue

            hue_factors = torch.empty(B, device=x.device).uniform_(-self.hue, self.hue)
            x = torch.stack(
                [adjust_hue(x[i], float(hue_factors[i])) for i in range(B)],
                dim=0,
            )
        return x.clamp(0.0, 1.0)

    def _grayscale(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] != 3:
            return x
        gray = x[:, :, 0:1] * 0.299 + x[:, :, 1:2] * 0.587 + x[:, :, 2:3] * 0.114
        return gray.expand_as(x)

    def _solarize(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= self.solarize_threshold, 1.0 - x, x)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        if imgs.ndim == 4:
            imgs = imgs.unsqueeze(1)
        elif imgs.ndim != 5:
            raise ValueError(
                f"DinoImageAugTransform expected [B, T, C, H, W] or [B, C, H, W], got {tuple(imgs.shape)}"
            )

        B, T = imgs.shape[:2]
        x = imgs.float()
        x = self._random_resized_crop(x)

        jittered = self._color_jitter(x)
        grayscaled = self._grayscale(x)
        solarized = self._solarize(x)

        draw = torch.rand(B, 1, 1, 1, 1, device=x.device)
        none_cutoff = self.none_prob
        gray_cutoff = none_cutoff + self.grayscale_prob
        solarize_cutoff = gray_cutoff + self.solarize_prob
        none_mask = draw < none_cutoff
        gray_mask = (draw >= none_cutoff) & (draw < gray_cutoff)
        solarize_mask = (draw >= gray_cutoff) & (draw < solarize_cutoff)

        out = torch.where(none_mask, x, jittered)
        out = torch.where(gray_mask, grayscaled, out)
        out = torch.where(solarize_mask, solarized, out)
        return out.clamp(0.0, 1.0)


class DinoVisualEncoder(nn.Module):
    """DINOv2 CLS-token encoder projected into environment-state features."""

    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        model_name: str,
        out_dim: int,
        proj_hidden_dim: int | None = None,
        frozen: bool = True,
        resize: tuple[int, int] = (224, 224),
    ):
        super().__init__()
        self.resize = resize
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        feat_dim = self.backbone.embed_dim
        self.proj = (
            nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, proj_hidden_dim),
                nn.Mish(),
                nn.Linear(proj_hidden_dim, out_dim),
            )
            if proj_hidden_dim is not None and proj_hidden_dim > 0
            else nn.Linear(feat_dim, out_dim)
        )
        self.out_dim = out_dim
        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        self.register_buffer("_mean", torch.tensor(self._MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(self._STD).view(1, 3, 1, 1))

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        # LeRobot usually provides [B, T_obs, C, H, W], but with n_obs_steps=1
        # some paths can collapse the time dimension to [B, C, H, W].
        if imgs.ndim == 4:
            imgs = imgs.unsqueeze(1)
        elif imgs.ndim != 5:
            raise ValueError(
                f"DinoVisualEncoder expected [B, T, C, H, W] or [B, C, H, W], got {tuple(imgs.shape)}"
            )

        B, T = imgs.shape[:2]
        x = imgs.flatten(0, 1)
        if x.shape[-2:] != tuple(self.resize):
            x = torch.nn.functional.interpolate(
                x, size=self.resize, mode="bilinear", align_corners=False
            )
        x = (x - self._mean) / self._std
        cls = self.backbone(x)
        out = self.proj(cls)
        return out.view(B, T, -1)


class ResNetEnvVisualEncoder(nn.Module):
    """External torchvision ResNet encoder projected into ENV features."""

    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        backbone: str = "resnet18",
        out_dim: int = 32,
        proj_hidden_dim: int | None = None,
        pretrained_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1",
        frozen: bool = False,
        resize: tuple[int, int] = (224, 224),
    ):
        super().__init__()
        from torchvision import models

        self.resize = resize
        constructor = getattr(models, backbone)
        weights = None
        if pretrained_weights:
            weights_enum_name, weights_value = pretrained_weights.split(".", 1)
            weights_enum = getattr(models, weights_enum_name)
            weights = getattr(weights_enum, weights_value)
        model = constructor(weights=weights)
        feat_dim = model.fc.in_features
        model.fc = nn.Identity()
        self.backbone = model
        self.proj = (
            nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, proj_hidden_dim),
                nn.Mish(),
                nn.Linear(proj_hidden_dim, out_dim),
            )
            if proj_hidden_dim is not None and proj_hidden_dim > 0
            else nn.Linear(feat_dim, out_dim)
        )
        self.out_dim = out_dim
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad_(False)
        self.register_buffer("_mean", torch.tensor(self._MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(self._STD).view(1, 3, 1, 1))

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        if imgs.ndim == 4:
            imgs = imgs.unsqueeze(1)
        elif imgs.ndim != 5:
            raise ValueError(
                f"ResNetEnvVisualEncoder expected [B, T, C, H, W] or [B, C, H, W], got {tuple(imgs.shape)}"
            )
        B, T = imgs.shape[:2]
        x = imgs.flatten(0, 1).float()
        if x.shape[-2:] != tuple(self.resize):
            x = torch.nn.functional.interpolate(
                x, size=self.resize, mode="bilinear", align_corners=False
            )
        x = (x - self._mean) / self._std
        out = self.proj(self.backbone(x))
        return out.view(B, T, -1)


class DepthVisualEncoder(nn.Module):
    """Small CNN for precomputed depth video frames."""

    def __init__(self, out_dim: int, resize: tuple[int, int] = (224, 224)):
        super().__init__()
        self.resize = resize
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.Mish(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.Mish(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.Mish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
        )

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        if imgs.ndim == 4:
            imgs = imgs.unsqueeze(1)
        elif imgs.ndim != 5:
            raise ValueError(
                f"DepthVisualEncoder expected [B, T, C, H, W] or [B, C, H, W], got {tuple(imgs.shape)}"
            )

        B, T = imgs.shape[:2]
        x = imgs.flatten(0, 1).float()
        if x.shape[-2:] != tuple(self.resize):
            x = torch.nn.functional.interpolate(
                x, size=self.resize, mode="bilinear", align_corners=False
            )
        out = self.net(x)
        return out.view(B, T, -1)


class ConditioningFusion(nn.Module):
    """Fuse compact modality tokens into one wider FiLM condition."""

    def __init__(
        self,
        *,
        state_dim: int,
        dino_dim: int,
        depth_dim: int,
        traj_dim: int,
        n_obs_steps: int,
        token_dim: int = 64,
        out_dim: int = 128,
        nhead: int = 4,
        ffn_dim: int = 256,
        n_layers: int = 2,
        num_output_tokens: int = 1,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.out_dim = out_dim
        self.num_output_tokens = num_output_tokens
        self.state_proj = nn.Linear(state_dim, token_dim) if state_dim > 0 else None
        self.dino_proj = nn.Linear(dino_dim, token_dim) if dino_dim > 0 else None
        self.depth_proj = nn.Linear(depth_dim, token_dim) if depth_dim > 0 else None
        self.traj_proj = nn.Linear(traj_dim, token_dim) if traj_dim > 0 else None
        self.output_tokens = nn.Parameter(torch.randn(num_output_tokens, out_dim) * 0.02)
        self.output_emb = nn.Parameter(torch.randn(1, out_dim) * 0.02)
        self.modality_emb = nn.Parameter(torch.randn(4, token_dim) * 0.02)
        self.obs_time_emb = nn.Parameter(torch.randn(n_obs_steps, token_dim) * 0.02)
        self.query_norms = nn.ModuleList([nn.LayerNorm(out_dim) for _ in range(n_layers)])
        self.context_norms = nn.ModuleList([nn.LayerNorm(token_dim) for _ in range(n_layers)])
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=out_dim,
                num_heads=nhead,
                kdim=token_dim,
                vdim=token_dim,
                batch_first=True,
            )
            for _ in range(n_layers)
        ])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(out_dim) for _ in range(n_layers)])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(out_dim, ffn_dim),
                nn.Mish(),
                nn.Linear(ffn_dim, out_dim),
            )
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(out_dim)

    @property
    def output_dim(self) -> int:
        return self.out_dim * self.num_output_tokens

    def _add_temporal_tokens(
        self,
        tokens: list[torch.Tensor],
        x: torch.Tensor | None,
        proj: nn.Module | None,
        modality_index: int,
    ) -> None:
        if x is None or proj is None:
            return
        if x.ndim != 3:
            raise ValueError(f"Expected temporal feature [B, T, D], got {tuple(x.shape)}")
        projected = proj(x)
        T = projected.shape[1]
        projected = (
            projected
            + self.modality_emb[modality_index - 1].view(1, 1, -1)
            + self.obs_time_emb[:T].view(1, T, -1)
        )
        tokens.append(projected)

    def _add_set_tokens(
        self,
        tokens: list[torch.Tensor],
        x: torch.Tensor | None,
        proj: nn.Module | None,
        modality_index: int,
    ) -> None:
        if x is None or proj is None:
            return
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"Expected token feature [B, N, D], got {tuple(x.shape)}")
        tokens.append(proj(x) + self.modality_emb[modality_index - 1].view(1, 1, -1))

    def forward(
        self,
        *,
        state: torch.Tensor | None,
        dino: torch.Tensor | None,
        depth: torch.Tensor | None,
        trajectory: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size = None
        for item in (state, dino, depth, trajectory):
            if item is not None:
                batch_size = item.shape[0]
                break
        if batch_size is None:
            raise ValueError("ConditioningFusion received no input tokens.")

        tokens: list[torch.Tensor] = []
        self._add_temporal_tokens(tokens, state, self.state_proj, 1)
        self._add_temporal_tokens(tokens, dino, self.dino_proj, 2)
        self._add_temporal_tokens(tokens, depth, self.depth_proj, 3)
        self._add_set_tokens(tokens, trajectory, self.traj_proj, 4)

        output = (
            self.output_tokens.unsqueeze(0).expand(batch_size, -1, -1)
            + self.output_emb.view(1, 1, -1)
        )
        context = torch.cat(tokens, dim=1)
        fused = output
        for query_norm, context_norm, attn, ffn_norm, ffn in zip(
            self.query_norms,
            self.context_norms,
            self.attn_layers,
            self.ffn_norms,
            self.ffn_layers,
        ):
            attn_out, _ = attn(
                query=query_norm(fused),
                key=context_norm(context),
                value=context_norm(context),
                need_weights=False,
            )
            fused = fused + attn_out
            fused = fused + ffn(ffn_norm(fused))
        fused = self.out_norm(fused)
        return fused.flatten(start_dim=1)


class TaskStratifiedBatchSampler:
    """Build near-balanced mini-batches from a scalar per-frame task_id feature."""

    def __init__(self, task_ids: list[int], batch_size: int, drop_last: bool = True, seed: int = 0):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if not task_ids:
            raise ValueError("Cannot stratify an empty dataset.")

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.task_to_indices: dict[int, list[int]] = {}
        for index, task_id in enumerate(task_ids):
            self.task_to_indices.setdefault(int(task_id), []).append(index)
        self.task_ids = sorted(self.task_to_indices)
        self.num_samples = len(task_ids)

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        return math.ceil(self.num_samples / self.batch_size)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        pools = {task_id: indices.copy() for task_id, indices in self.task_to_indices.items()}
        positions = {task_id: 0 for task_id in self.task_ids}
        for indices in pools.values():
            rng.shuffle(indices)

        def draw(task_id: int, count: int) -> list[int]:
            selected = []
            pool = pools[task_id]
            while len(selected) < count:
                position = positions[task_id]
                if position == len(pool):
                    rng.shuffle(pool)
                    position = 0
                take = min(count - len(selected), len(pool) - position)
                selected.extend(pool[position : position + take])
                positions[task_id] = position + take
            return selected

        num_tasks = len(self.task_ids)
        for batch_index in range(len(self)):
            current_batch_size = self.batch_size
            if not self.drop_last and batch_index == len(self) - 1:
                current_batch_size = self.num_samples - batch_index * self.batch_size

            per_task, remainder = divmod(current_batch_size, num_tasks)
            extra_start = (batch_index * remainder) % num_tasks if remainder else 0
            extra_tasks = {
                self.task_ids[(extra_start + offset) % num_tasks]
                for offset in range(remainder)
            }
            batch = []
            for task_id in self.task_ids:
                batch.extend(draw(task_id, per_task + int(task_id in extra_tasks)))
            rng.shuffle(batch)
            yield batch


def scalar_task_id(value: Any) -> int:
    """Extract a scalar task id from HF/LeRobot scalar or length-1 values."""
    if torch.is_tensor(value):
        return int(value.flatten()[0].item())
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("Encountered an empty task_id value.")
        value = value[0]
    return int(value)


def depth_key_for(camera_key: str, depth_prefix: str = DEPTH_PREFIX) -> str:
    suffix = camera_key.removeprefix("observation.images.")
    return f"{depth_prefix}.{suffix}"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class MetricsTracker:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.train_steps: list[int] = []
        self.train_losses: list[float] = []
        self.train_lrs: list[float] = []
        self.val_steps: list[int] = []
        self.val_losses: list[float] = []
        self._train_csv = output_dir / "metrics_train.csv"
        self._val_csv = output_dir / "metrics_val.csv"
        self._conditioning_csv = output_dir / "metrics_conditioning.csv"
        self._init_csv()

    def _init_csv(self) -> None:
        if not self._train_csv.exists():
            with open(self._train_csv, "w", newline="") as f:
                csv.writer(f).writerow(["step", "avg_loss", "lr"])
        if not self._val_csv.exists():
            with open(self._val_csv, "w", newline="") as f:
                csv.writer(f).writerow(["step", "val_loss"])
        if not self._conditioning_csv.exists():
            with open(self._conditioning_csv, "w", newline="") as f:
                csv.writer(f).writerow([
                    "step",
                    "correct_loss",
                    "shuffled_trajectory_loss",
                "zero_vision_loss",
                "zero_depth_loss",
                "zero_state_loss",
                "shuffle_loss_delta",
                "zero_vision_loss_delta",
                "zero_depth_loss_delta",
                "zero_state_loss_delta",
                ])

    def log_train(self, step: int, avg_loss: float, lr: float) -> None:
        self.train_steps.append(step)
        self.train_losses.append(avg_loss)
        self.train_lrs.append(lr)
        with open(self._train_csv, "a", newline="") as f:
            csv.writer(f).writerow([step, avg_loss, lr])

    def log_val(self, step: int, val_loss: float) -> None:
        self.val_steps.append(step)
        self.val_losses.append(val_loss)
        with open(self._val_csv, "a", newline="") as f:
            csv.writer(f).writerow([step, val_loss])

    def log_conditioning(
        self,
        step: int,
        correct_loss: float,
        shuffled_trajectory_loss: float,
        zero_vision_loss: float,
        zero_depth_loss: float,
        zero_state_loss: float,
    ) -> None:
        with open(self._conditioning_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                step,
                correct_loss,
                shuffled_trajectory_loss,
                zero_vision_loss,
                zero_depth_loss,
                zero_state_loss,
                shuffled_trajectory_loss - correct_loss,
                zero_vision_loss - correct_loss,
                zero_depth_loss - correct_loss,
                zero_state_loss - correct_loss,
            ])

    def save_plots(self, step: int) -> None:
        if not self.train_steps:
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.train_steps, self.train_losses, label="train_loss", linewidth=1)
        if self.val_steps:
            ax.plot(self.val_steps, self.val_losses, "o-", label="val_loss", linewidth=1)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(f"Diffusion Policy Loss (step {step:,})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "loss.png", dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.semilogy(self.train_steps, self.train_lrs, linewidth=1)
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning rate")
        ax.set_title(f"Learning Rate (step {step:,})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "lr.png", dpi=120)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Script helpers
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a trajectory-conditioned LeRobot Diffusion Policy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            HuggingFace authentication:
              1. --hf-token TOKEN
              2. HUGGINGFACE_TOKEN environment variable
              3. cached token from `huggingface-cli login`
            """
        ),
    )
    p.add_argument("--repo-id", default=None, help=f"Dataset repo id (default: {DATASET_REPO_ID})")
    p.add_argument("--dataset-root", type=Path, default=None, help="Local dataset root override")
    p.add_argument("--output-dir", type=Path, default=None, help=f"Output directory (default: {OUTPUT_DIR})")
    p.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Checkpoint directory to resume from. Restores weights and available optimizer/scheduler/trainer state.",
    )
    p.add_argument("--hf-token", default=None, help="HuggingFace token")
    p.add_argument("--device", default=None, help=f"Device override (default: {DEVICE})")
    p.add_argument("--num-workers", type=int, default=None, help=f"DataLoader workers (default: {NUM_WORKERS})")
    p.add_argument("--batch-size", type=int, default=None, help=f"Batch size (default: {BATCH_SIZE})")
    p.add_argument(
        "--global-traj-length",
        type=int,
        default=None,
        help=f"Unpadded trajectory resampling length (default: {GLOBAL_TRAJ_LENGTH})",
    )
    p.add_argument(
        "--n-obs-steps",
        type=int,
        default=None,
        help=f"Number of observation steps for the policy (default: {N_OBS_STEPS})",
    )
    p.add_argument(
        "--observation-frame-offsets",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Frame offsets for temporal observations, sorted and ending at 0 "
            f"(default: {' '.join(map(str, OBSERVATION_FRAME_OFFSETS))})"
        ),
    )
    p.add_argument(
        "--traj-action-encoding",
        choices=["none", "delta", "binary"],
        default=None,
        help=f"Trajectory action-channel encoding (default: {TRAJ_ACTION_ENCODING})",
    )
    p.add_argument(
        "--traj-action-binary-threshold",
        type=float,
        default=None,
        help=f"Action threshold for binary encoding (default: {TRAJ_ACTION_BINARY_THRESHOLD})",
    )
    p.add_argument(
        "--use-traj-fourier-features",
        action="store_true",
        help="Use sinusoidal Fourier features for trajectory xy coordinates.",
    )
    p.add_argument(
        "--no-traj-active-action-signal",
        action="store_true",
        help="Keep the original action channel instead of converting it to an active-manipulation window.",
    )
    p.add_argument(
        "--no-traj-active-window-only",
        action="store_true",
        help="Keep approach trajectory points instead of clamping to the active manipulation window.",
    )
    p.add_argument(
        "--no-traj-return-cutoff",
        action="store_true",
        help="Disable training-time cutoff after active manipulation or return-to-start fallback.",
    )
    p.add_argument(
        "--traj-return-cutoff-gradient-threshold",
        type=float,
        default=None,
        help=f"Gradient threshold for fallback return cutoff (default: {TRAJ_RETURN_CUTOFF_GRADIENT_THRESHOLD})",
    )
    p.add_argument(
        "--traj-return-cutoff-smooth-window",
        type=int,
        default=None,
        help=f"Smoothing window for fallback return cutoff (default: {TRAJ_RETURN_CUTOFF_SMOOTH_WINDOW})",
    )
    p.add_argument(
        "--stratify-batches-by-task-id",
        action="store_true",
        help="Build near-balanced training mini-batches using the dataset's task_id feature.",
    )
    p.add_argument(
        "--no-stratify-batches-by-task-id",
        dest="stratify_batches_by_task_id",
        action="store_false",
        help="Use shuffled batches with the dataset's natural task frequencies.",
    )
    p.set_defaults(stratify_batches_by_task_id=None)
    p.add_argument(
        "--exclude-task-ids",
        type=int,
        nargs="+",
        default=None,
        help="Exclude all frames whose scalar task_id is in this list.",
    )
    p.add_argument(
        "--capacity-preset",
        choices=["tinier", "tiny", "small", "medium", "large", "custom"],
        default=None,
        help=f"Model capacity preset (default: {CAPACITY_PRESET})",
    )
    p.add_argument(
        "--denoiser-backbone",
        choices=["dit", "unet"],
        default=None,
        help=f"Action denoiser backbone (default: {DENOISER_BACKBONE})",
    )
    p.add_argument("--dit-dim", type=int, default=None, help=f"DiT token width (default: {DIT_DIM})")
    p.add_argument("--dit-depth", type=int, default=None, help=f"DiT block count (default: {DIT_DEPTH})")
    p.add_argument("--dit-nhead", type=int, default=None, help=f"DiT attention heads (default: {DIT_NHEAD})")
    p.add_argument("--dit-ffn-dim", type=int, default=None, help=f"DiT feedforward width (default: {DIT_FFN_DIM})")
    p.add_argument("--dit-dropout", type=float, default=None, help=f"DiT dropout (default: {DIT_DROPOUT})")
    budget = p.add_mutually_exclusive_group()
    budget.add_argument("--num-steps", type=int, default=None, help=f"Training steps (default: {NUM_TRAIN_STEPS})")
    budget.add_argument("--num-epochs", type=float, default=None, help="Training epochs; converted after dataset load")
    p.add_argument("--lr", type=float, default=None, help=f"Learning rate (default: {LR})")
    p.add_argument("--scheduler", choices=["cosine", "plateau", "none"], default=None)
    p.add_argument("--log-freq", type=int, default=None, help=f"Log every N steps (default: {LOG_FREQ})")
    p.add_argument("--save-freq", type=int, default=None, help=f"Checkpoint every N steps (default: {SAVE_FREQ})")
    p.add_argument(
        "--conditioning-diag-freq",
        type=int,
        default=None,
        help=(
            "Run conditioning perturbation diagnostics every N training steps; "
            f"0 disables it (default: {CONDITIONING_DIAG_FREQ})"
        ),
    )
    p.add_argument(
        "--save-every-n-epochs",
        type=float,
        default=None,
        help="Checkpoint every N epochs, based on the actual dataloader length. Allows fractions, e.g. 0.5.",
    )
    p.add_argument("--plot-freq", type=int, default=None, help=f"Plot every N steps (default: {PLOT_FREQ})")
    p.add_argument(
        "--no-run-subdirs",
        action="store_true",
        help="Write directly into OUTPUT_DIR instead of appending run0, run1, ...",
    )
    p.add_argument("--no-dino", action="store_true", help="Disable DINO visual encoder")
    p.add_argument(
        "--use-resnet-env-encoder",
        action="store_true",
        help="Encode selected camera observations with an external torchvision ResNet into ENV features.",
    )
    p.add_argument("--resnet-backbone", default=None, help=f"External ResNet backbone (default: {VISION_BACKBONE})")
    p.add_argument(
        "--resnet-pretrained-weights",
        default=None,
        help=f"torchvision weights enum path, or 'none' (default: {PRETRAINED_BACKBONE})",
    )
    p.add_argument("--resnet-out-dim", type=int, default=None, help=f"External ResNet output dim (default: {RESNET_OUT_DIM})")
    p.add_argument(
        "--resnet-proj-hidden-dim",
        type=int,
        default=None,
        help=f"External ResNet projection hidden dim (default: {RESNET_PROJ_HIDDEN_DIM})",
    )
    p.add_argument("--resnet-frozen", action="store_true", help="Freeze the external ResNet backbone.")
    p.add_argument(
        "--resnet-resize",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help=f"External ResNet input resize (default: {RESNET_RESIZE})",
    )
    p.add_argument("--resnet-lr", type=float, default=None, help="Separate LR for external ResNet encoder.")
    p.add_argument("--use-depth", dest="use_depth", action="store_true", help="Enable precomputed depth encoder")
    p.add_argument("--no-depth", dest="use_depth", action="store_false", help="Disable precomputed depth encoder")
    p.set_defaults(use_depth=None)
    p.add_argument(
        "--no-state",
        action="store_true",
        help="Do not condition on robot joint state; feed a zero placeholder state instead.",
    )
    p.add_argument("--no-traj-encoder", action="store_true", help="Use raw flattened global trajectory")
    p.add_argument("--dry-run", action="store_true", help="Build everything and run one forward pass only")
    return p.parse_args()


def apply_overrides(args: argparse.Namespace) -> None:
    global DATASET_REPO_ID, DATASET_ROOT, OUTPUT_DIR, DEVICE, NUM_WORKERS
    global BATCH_SIZE, NUM_TRAIN_STEPS, NUM_EPOCHS, LR, LR_PEAK
    global GLOBAL_TRAJ_LENGTH, N_OBS_STEPS, OBSERVATION_FRAME_OFFSETS
    global TRAJ_ACTION_ENCODING, TRAJ_ACTION_BINARY_THRESHOLD
    global TRAJ_USE_FOURIER_FEATURES, TRAJ_ACTIVE_ACTION_SIGNAL
    global TRAJ_ACTIVE_WINDOW_ONLY
    global TRAJ_RETURN_CUTOFF, TRAJ_RETURN_CUTOFF_GRADIENT_THRESHOLD
    global TRAJ_RETURN_CUTOFF_SMOOTH_WINDOW
    global CONDITIONING_DIAG_FREQ
    global STRATIFY_BATCHES_BY_TASK_ID
    global EXCLUDED_TASK_IDS
    global LR_SCHEDULER_TYPE, LOG_FREQ, SAVE_FREQ, SAVE_EVERY_N_EPOCHS, PLOT_FREQ
    global USE_DINO_CAM, USE_RESNET_ENV_ENCODER, RESNET_OUT_DIM, RESNET_PROJ_HIDDEN_DIM
    global RESNET_FROZEN, RESNET_RESIZE, RESNET_LR, VISION_BACKBONE, PRETRAINED_BACKBONE
    global USE_DEPTH_ENCODER, USE_TRAJ_ENCODER, RUN_SUBDIRS, CAPACITY_PRESET
    global USE_JOINT_STATE

    if args.repo_id is not None:
        DATASET_REPO_ID = args.repo_id
    if args.dataset_root is not None:
        DATASET_ROOT = args.dataset_root
    if args.output_dir is not None:
        OUTPUT_DIR = args.output_dir
    if args.capacity_preset is not None:
        CAPACITY_PRESET = args.capacity_preset
    if args.device is not None:
        DEVICE = args.device
    if args.num_workers is not None:
        NUM_WORKERS = args.num_workers
    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size
    if args.global_traj_length is not None:
        GLOBAL_TRAJ_LENGTH = args.global_traj_length
    if args.n_obs_steps is not None:
        N_OBS_STEPS = args.n_obs_steps
    if args.observation_frame_offsets is not None:
        OBSERVATION_FRAME_OFFSETS = tuple(args.observation_frame_offsets)
    if args.traj_action_encoding is not None:
        TRAJ_ACTION_ENCODING = args.traj_action_encoding
    if args.traj_action_binary_threshold is not None:
        TRAJ_ACTION_BINARY_THRESHOLD = args.traj_action_binary_threshold
    if args.use_traj_fourier_features:
        TRAJ_USE_FOURIER_FEATURES = True
    if args.no_traj_active_action_signal:
        TRAJ_ACTIVE_ACTION_SIGNAL = False
    if args.no_traj_active_window_only:
        TRAJ_ACTIVE_WINDOW_ONLY = False
    if args.no_traj_return_cutoff:
        TRAJ_RETURN_CUTOFF = False
    if args.traj_return_cutoff_gradient_threshold is not None:
        TRAJ_RETURN_CUTOFF_GRADIENT_THRESHOLD = args.traj_return_cutoff_gradient_threshold
    if args.traj_return_cutoff_smooth_window is not None:
        TRAJ_RETURN_CUTOFF_SMOOTH_WINDOW = args.traj_return_cutoff_smooth_window
    if args.stratify_batches_by_task_id is not None:
        STRATIFY_BATCHES_BY_TASK_ID = args.stratify_batches_by_task_id
    if args.exclude_task_ids is not None:
        EXCLUDED_TASK_IDS = set(args.exclude_task_ids)
    if args.num_steps is not None:
        NUM_TRAIN_STEPS = args.num_steps
        NUM_EPOCHS = None
    if args.num_epochs is not None:
        NUM_EPOCHS = args.num_epochs
    if args.lr is not None:
        LR = args.lr
        LR_PEAK = args.lr
    if args.scheduler is not None:
        LR_SCHEDULER_TYPE = args.scheduler
    if args.log_freq is not None:
        LOG_FREQ = args.log_freq
    if args.save_freq is not None:
        SAVE_FREQ = args.save_freq
    if args.conditioning_diag_freq is not None:
        CONDITIONING_DIAG_FREQ = args.conditioning_diag_freq
    if args.save_every_n_epochs is not None:
        SAVE_EVERY_N_EPOCHS = args.save_every_n_epochs
    if args.plot_freq is not None:
        PLOT_FREQ = args.plot_freq
    if args.no_dino:
        USE_DINO_CAM = False
    if args.use_resnet_env_encoder:
        USE_RESNET_ENV_ENCODER = True
        USE_DINO_CAM = False
    if args.resnet_backbone is not None:
        VISION_BACKBONE = args.resnet_backbone
    if args.resnet_pretrained_weights is not None:
        PRETRAINED_BACKBONE = None if args.resnet_pretrained_weights.lower() == "none" else args.resnet_pretrained_weights
    if args.resnet_out_dim is not None:
        RESNET_OUT_DIM = args.resnet_out_dim
    if args.resnet_proj_hidden_dim is not None:
        RESNET_PROJ_HIDDEN_DIM = args.resnet_proj_hidden_dim
    if args.resnet_frozen:
        RESNET_FROZEN = True
    if args.resnet_resize is not None:
        RESNET_RESIZE = tuple(args.resnet_resize)
    if args.resnet_lr is not None:
        RESNET_LR = args.resnet_lr
    if args.use_depth is not None:
        USE_DEPTH_ENCODER = args.use_depth
    if args.no_state:
        USE_JOINT_STATE = False
    if args.no_traj_encoder:
        USE_TRAJ_ENCODER = False
    if args.no_run_subdirs:
        RUN_SUBDIRS = False


def apply_denoiser_overrides(args: argparse.Namespace) -> None:
    global DENOISER_BACKBONE, DIT_DIM, DIT_DEPTH, DIT_NHEAD, DIT_FFN_DIM, DIT_DROPOUT

    if args.denoiser_backbone is not None:
        DENOISER_BACKBONE = args.denoiser_backbone
    if args.dit_dim is not None:
        DIT_DIM = args.dit_dim
    if args.dit_depth is not None:
        DIT_DEPTH = args.dit_depth
    if args.dit_nhead is not None:
        DIT_NHEAD = args.dit_nhead
    if args.dit_ffn_dim is not None:
        DIT_FFN_DIM = args.dit_ffn_dim
    if args.dit_dropout is not None:
        DIT_DROPOUT = args.dit_dropout


def apply_capacity_preset() -> None:
    global DOWN_DIMS, DIFFUSION_STEP_EMBED_DIM, TRAJ_ENC_TYPE, TRAJ_ENC_DIM
    global TRAJ_ENC_HIDDEN, TRAJ_ENC_LAYERS, TRAJ_CONV_KERNEL_SIZES
    global TRAJ_CONV_DILATIONS, TRAJ_TF_D_MODEL
    global TRAJ_TF_NHEAD, TRAJ_TF_FFN_DIM, TRAJ_TF_LAYERS, TRAJ_NUM_QUERY_TOKENS
    global DINO_OUT_DIM
    global DINO_PROJ_HIDDEN_DIM, DEPTH_OUT_DIM
    global FUSION_TOKEN_DIM, FUSION_OUT_DIM, FUSION_NHEAD, FUSION_FFN_DIM
    global FUSION_LAYERS
    global DIT_DIM, DIT_DEPTH, DIT_NHEAD, DIT_FFN_DIM

    if CAPACITY_PRESET == "custom":
        return
    if CAPACITY_PRESET not in CAPACITY_PRESETS:
        raise ValueError(f"Unknown capacity preset: {CAPACITY_PRESET!r}")

    preset = CAPACITY_PRESETS[CAPACITY_PRESET]
    DOWN_DIMS = preset["DOWN_DIMS"]
    if "DIFFUSION_STEP_EMBED_DIM" in preset:
        DIFFUSION_STEP_EMBED_DIM = preset["DIFFUSION_STEP_EMBED_DIM"]
    if "TRAJ_ENC_TYPE" in preset:
        TRAJ_ENC_TYPE = preset["TRAJ_ENC_TYPE"]
    TRAJ_ENC_DIM = preset["TRAJ_ENC_DIM"]
    if "TRAJ_ENC_HIDDEN" in preset:
        TRAJ_ENC_HIDDEN = preset["TRAJ_ENC_HIDDEN"]
    if "TRAJ_ENC_LAYERS" in preset:
        TRAJ_ENC_LAYERS = preset["TRAJ_ENC_LAYERS"]
    if "TRAJ_CONV_KERNEL_SIZES" in preset:
        TRAJ_CONV_KERNEL_SIZES = preset["TRAJ_CONV_KERNEL_SIZES"]
    if "TRAJ_CONV_DILATIONS" in preset:
        TRAJ_CONV_DILATIONS = preset["TRAJ_CONV_DILATIONS"]
    TRAJ_TF_D_MODEL = preset["TRAJ_TF_D_MODEL"]
    TRAJ_TF_NHEAD = preset["TRAJ_TF_NHEAD"]
    TRAJ_TF_FFN_DIM = preset["TRAJ_TF_FFN_DIM"]
    TRAJ_TF_LAYERS = preset["TRAJ_TF_LAYERS"]
    if "TRAJ_NUM_QUERY_TOKENS" in preset:
        TRAJ_NUM_QUERY_TOKENS = preset["TRAJ_NUM_QUERY_TOKENS"]
    DINO_OUT_DIM = preset["DINO_OUT_DIM"]
    DINO_PROJ_HIDDEN_DIM = preset.get("DINO_PROJ_HIDDEN_DIM", None)
    DEPTH_OUT_DIM = preset["DEPTH_OUT_DIM"]
    if "FUSION_TOKEN_DIM" in preset:
        FUSION_TOKEN_DIM = preset["FUSION_TOKEN_DIM"]
        FUSION_OUT_DIM = preset["FUSION_OUT_DIM"]
        FUSION_NHEAD = preset["FUSION_NHEAD"]
        FUSION_FFN_DIM = preset["FUSION_FFN_DIM"]
        FUSION_LAYERS = preset["FUSION_LAYERS"]
    if "DIT_DIM" in preset:
        DIT_DIM = preset["DIT_DIM"]
        DIT_DEPTH = preset["DIT_DEPTH"]
        DIT_NHEAD = preset["DIT_NHEAD"]
        DIT_FFN_DIM = preset["DIT_FFN_DIM"]


def apply_resume_config_overrides(resume_from: Path | None) -> None:
    if resume_from is None:
        return

    global USE_TRAJ_ENCODER, TRAJ_ENC_TYPE, TRAJ_ENC_DIM, TRAJ_ENC_HIDDEN, TRAJ_ENC_LAYERS
    global TRAJ_CONV_KERNEL_SIZES, TRAJ_CONV_DILATIONS
    global TRAJ_TF_D_MODEL, TRAJ_TF_NHEAD
    global TRAJ_TF_FFN_DIM, TRAJ_TF_LAYERS, DINO_OUT_DIM, DINO_FROZEN
    global DINO_RESIZE, DINO_PROJ_HIDDEN_DIM, USE_DEPTH_ENCODER, DEPTH_OUT_DIM, DEPTH_RESIZE
    global USE_RESNET_ENV_ENCODER, RESNET_OUT_DIM, RESNET_PROJ_HIDDEN_DIM
    global RESNET_FROZEN, RESNET_RESIZE, VISION_BACKBONE, PRETRAINED_BACKBONE
    global FUSION_TOKEN_DIM, FUSION_OUT_DIM, FUSION_NHEAD, FUSION_FFN_DIM
    global FUSION_LAYERS, FUSION_NUM_OUTPUT_TOKENS, N_OBS_STEPS
    global OBSERVATION_FRAME_OFFSETS, DOWN_DIMS, DIFFUSION_STEP_EMBED_DIM
    global USE_JOINT_STATE
    global DENOISER_BACKBONE, DIT_DIM, DIT_DEPTH, DIT_NHEAD, DIT_FFN_DIM, DIT_DROPOUT

    encoder_config = load_json_if_exists(resume_from / "encoder_config.json") or {}
    policy_config = load_json_if_exists(resume_from / "config.json") or {}
    if not encoder_config and not policy_config:
        log.warning("Resume checkpoint has no encoder_config.json or config.json; using current architecture settings.")
        return

    USE_TRAJ_ENCODER = bool(encoder_config.get("USE_TRAJ_ENCODER", USE_TRAJ_ENCODER))
    TRAJ_ENC_TYPE = encoder_config.get("TRAJ_ENC_TYPE", TRAJ_ENC_TYPE)
    TRAJ_ENC_DIM = int(encoder_config.get("TRAJ_ENC_DIM", TRAJ_ENC_DIM))
    TRAJ_ENC_HIDDEN = int(encoder_config.get("TRAJ_ENC_HIDDEN", TRAJ_ENC_HIDDEN))
    TRAJ_ENC_LAYERS = int(encoder_config.get("TRAJ_ENC_LAYERS", TRAJ_ENC_LAYERS))
    if "TRAJ_CONV_KERNEL_SIZES" in encoder_config:
        TRAJ_CONV_KERNEL_SIZES = tuple(encoder_config["TRAJ_CONV_KERNEL_SIZES"])
    if "TRAJ_CONV_DILATIONS" in encoder_config:
        TRAJ_CONV_DILATIONS = tuple(encoder_config["TRAJ_CONV_DILATIONS"])
    TRAJ_TF_D_MODEL = int(encoder_config.get("TRAJ_TF_D_MODEL", TRAJ_TF_D_MODEL))
    TRAJ_TF_NHEAD = int(encoder_config.get("TRAJ_TF_NHEAD", TRAJ_TF_NHEAD))
    TRAJ_TF_FFN_DIM = int(encoder_config.get("TRAJ_TF_FFN_DIM", TRAJ_TF_FFN_DIM))
    TRAJ_TF_LAYERS = int(encoder_config.get("TRAJ_TF_LAYERS", TRAJ_TF_LAYERS))
    DINO_OUT_DIM = int(encoder_config.get("DINO_OUT_DIM", DINO_OUT_DIM))
    DINO_PROJ_HIDDEN_DIM = encoder_config.get("DINO_PROJ_HIDDEN_DIM", DINO_PROJ_HIDDEN_DIM)
    if DINO_PROJ_HIDDEN_DIM is not None:
        DINO_PROJ_HIDDEN_DIM = int(DINO_PROJ_HIDDEN_DIM)
    DINO_FROZEN = bool(encoder_config.get("DINO_FROZEN", DINO_FROZEN))
    DINO_RESIZE = tuple(encoder_config.get("DINO_RESIZE", DINO_RESIZE))
    USE_RESNET_ENV_ENCODER = bool(encoder_config.get("USE_RESNET_ENV_ENCODER", USE_RESNET_ENV_ENCODER))
    RESNET_OUT_DIM = int(encoder_config.get("RESNET_OUT_DIM", RESNET_OUT_DIM))
    RESNET_PROJ_HIDDEN_DIM = encoder_config.get("RESNET_PROJ_HIDDEN_DIM", RESNET_PROJ_HIDDEN_DIM)
    if RESNET_PROJ_HIDDEN_DIM is not None:
        RESNET_PROJ_HIDDEN_DIM = int(RESNET_PROJ_HIDDEN_DIM)
    RESNET_FROZEN = bool(encoder_config.get("RESNET_FROZEN", RESNET_FROZEN))
    RESNET_RESIZE = tuple(encoder_config.get("RESNET_RESIZE", RESNET_RESIZE))
    VISION_BACKBONE = encoder_config.get("VISION_BACKBONE", VISION_BACKBONE)
    PRETRAINED_BACKBONE = encoder_config.get("PRETRAINED_BACKBONE", PRETRAINED_BACKBONE)
    USE_DEPTH_ENCODER = bool(encoder_config.get("USE_DEPTH_ENCODER", USE_DEPTH_ENCODER))
    DEPTH_OUT_DIM = int(encoder_config.get("DEPTH_OUT_DIM", DEPTH_OUT_DIM))
    DEPTH_RESIZE = tuple(encoder_config.get("DEPTH_RESIZE", DEPTH_RESIZE))
    FUSION_TOKEN_DIM = int(encoder_config.get("FUSION_TOKEN_DIM", FUSION_TOKEN_DIM))
    FUSION_OUT_DIM = int(encoder_config.get("FUSION_OUT_DIM", FUSION_OUT_DIM))
    FUSION_NHEAD = int(encoder_config.get("FUSION_NHEAD", FUSION_NHEAD))
    FUSION_FFN_DIM = int(encoder_config.get("FUSION_FFN_DIM", FUSION_FFN_DIM))
    FUSION_LAYERS = int(encoder_config.get("FUSION_LAYERS", FUSION_LAYERS))
    FUSION_NUM_OUTPUT_TOKENS = int(encoder_config.get("FUSION_NUM_OUTPUT_TOKENS", FUSION_NUM_OUTPUT_TOKENS))
    DENOISER_BACKBONE = encoder_config.get("DENOISER_BACKBONE", DENOISER_BACKBONE)
    DIT_DIM = int(encoder_config.get("DIT_DIM", DIT_DIM))
    DIT_DEPTH = int(encoder_config.get("DIT_DEPTH", DIT_DEPTH))
    DIT_NHEAD = int(encoder_config.get("DIT_NHEAD", DIT_NHEAD))
    DIT_FFN_DIM = int(encoder_config.get("DIT_FFN_DIM", DIT_FFN_DIM))
    DIT_DROPOUT = float(encoder_config.get("DIT_DROPOUT", DIT_DROPOUT))
    if "OBSERVATION_FRAME_OFFSETS" in encoder_config:
        OBSERVATION_FRAME_OFFSETS = tuple(encoder_config["OBSERVATION_FRAME_OFFSETS"])

    if "n_obs_steps" in policy_config:
        N_OBS_STEPS = int(policy_config["n_obs_steps"])
    if "down_dims" in policy_config:
        DOWN_DIMS = tuple(policy_config["down_dims"])
    if "diffusion_step_embed_dim" in policy_config:
        DIFFUSION_STEP_EMBED_DIM = int(policy_config["diffusion_step_embed_dim"])
    input_features = policy_config.get("input_features", {})
    state_shape = input_features.get("observation.state", {}).get("shape")
    if state_shape is not None:
        USE_JOINT_STATE = tuple(state_shape) != (1,)

    log.info("Applied architecture settings from resume checkpoint: %s", resume_from)


def add_file_logger(output_dir: Path) -> None:
    file_handler = logging.FileHandler(output_dir / "train.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)


def resolve_output_dir() -> None:
    global OUTPUT_DIR
    if RUN_SUBDIRS:
        base = OUTPUT_DIR
        run_n = 0
        while (base / f"run{run_n}").exists():
            run_n += 1
        OUTPUT_DIR = base / f"run{run_n}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def validate_static_config() -> None:
    downsample_factor = 2 ** len(DOWN_DIMS)
    if HORIZON % downsample_factor != 0:
        raise ValueError(f"HORIZON={HORIZON} must be divisible by {downsample_factor}")
    if DENOISER_BACKBONE not in {"unet", "dit"}:
        raise ValueError("DENOISER_BACKBONE must be one of: unet, dit")
    if DIT_DIM <= 0:
        raise ValueError("DIT_DIM must be positive")
    if DIT_DEPTH <= 0:
        raise ValueError("DIT_DEPTH must be positive")
    if DIT_NHEAD <= 0:
        raise ValueError("DIT_NHEAD must be positive")
    if DIT_DIM % DIT_NHEAD != 0:
        raise ValueError("DIT_DIM must be divisible by DIT_NHEAD")
    if DIT_FFN_DIM <= 0:
        raise ValueError("DIT_FFN_DIM must be positive")
    if not 0.0 <= DIT_DROPOUT < 1.0:
        raise ValueError("DIT_DROPOUT must be in [0, 1)")
    if USE_DINO_CAM and USE_MAIN_CAM:
        raise ValueError("USE_DINO_CAM and USE_MAIN_CAM are mutually exclusive")
    if USE_DINO_CAM and USE_RESNET_ENV_ENCODER:
        raise ValueError("USE_DINO_CAM and USE_RESNET_ENV_ENCODER are mutually exclusive in this script")
    if USE_RESNET_ENV_ENCODER and USE_MAIN_CAM:
        raise ValueError("USE_RESNET_ENV_ENCODER and USE_MAIN_CAM are mutually exclusive")
    if USE_RESNET_ENV_ENCODER and PRETRAINED_BACKBONE is not None and "." not in PRETRAINED_BACKBONE:
        raise ValueError("PRETRAINED_BACKBONE must look like 'ResNet18_Weights.IMAGENET1K_V1' or be None")
    if RESNET_OUT_DIM <= 0:
        raise ValueError("RESNET_OUT_DIM must be positive")
    if RESNET_PROJ_HIDDEN_DIM is not None and RESNET_PROJ_HIDDEN_DIM <= 0:
        raise ValueError("RESNET_PROJ_HIDDEN_DIM must be positive when set")
    if len(RESNET_RESIZE) != 2 or min(RESNET_RESIZE) <= 0:
        raise ValueError("RESNET_RESIZE must be two positive integers")
    if N_ACTION_STEPS > HORIZON - N_OBS_STEPS + 1:
        raise ValueError("N_ACTION_STEPS must be <= HORIZON - N_OBS_STEPS + 1")
    if len(OBSERVATION_FRAME_OFFSETS) != N_OBS_STEPS:
        raise ValueError(
            "OBSERVATION_FRAME_OFFSETS length must equal N_OBS_STEPS"
        )
    if tuple(sorted(OBSERVATION_FRAME_OFFSETS)) != tuple(OBSERVATION_FRAME_OFFSETS):
        raise ValueError("OBSERVATION_FRAME_OFFSETS must be sorted")
    if OBSERVATION_FRAME_OFFSETS[-1] != 0:
        raise ValueError("OBSERVATION_FRAME_OFFSETS must end at current frame 0")
    if len(set(OBSERVATION_FRAME_OFFSETS)) != len(OBSERVATION_FRAME_OFFSETS):
        raise ValueError("OBSERVATION_FRAME_OFFSETS must not contain duplicates")
    if GLOBAL_TRAJ_LENGTH <= 1:
        raise ValueError("GLOBAL_TRAJ_LENGTH must be greater than one")
    if TRAJ_ACTION_BINARY_THRESHOLD < 0:
        raise ValueError("TRAJ_ACTION_BINARY_THRESHOLD must be non-negative")
    if TRAJ_ACTION_ENCODING not in {"none", "delta", "binary"}:
        raise ValueError("TRAJ_ACTION_ENCODING must be one of: none, delta, binary")
    if TRAJ_TF_D_MODEL % TRAJ_TF_NHEAD != 0:
        raise ValueError("TRAJ_TF_D_MODEL must be divisible by TRAJ_TF_NHEAD")
    if FUSION_OUT_DIM % FUSION_NHEAD != 0:
        raise ValueError("FUSION_OUT_DIM must be divisible by FUSION_NHEAD")
    if FUSION_NUM_OUTPUT_TOKENS <= 0:
        raise ValueError("FUSION_NUM_OUTPUT_TOKENS must be positive")
    if TRAJ_RETURN_CUTOFF_GRADIENT_THRESHOLD < 0:
        raise ValueError("TRAJ_RETURN_CUTOFF_GRADIENT_THRESHOLD must be non-negative")
    if TRAJ_RETURN_CUTOFF_SMOOTH_WINDOW < 3:
        raise ValueError("TRAJ_RETURN_CUTOFF_SMOOTH_WINDOW must be >= 3")
    if TRAJ_NUM_QUERY_TOKENS < 0:
        raise ValueError("TRAJ_NUM_QUERY_TOKENS must be non-negative")
    if TRAJ_NUM_QUERY_TOKENS > 0 and TRAJ_ENC_TYPE != "transformer":
        raise ValueError("Trajectory query tokens require TRAJ_ENC_TYPE='transformer'")
    if TRAJ_ENC_TYPE == "conv1d":
        if TRAJ_CONV_KERNEL_SIZES is not None and len(TRAJ_CONV_KERNEL_SIZES) != TRAJ_ENC_LAYERS:
            raise ValueError("TRAJ_CONV_KERNEL_SIZES length must equal TRAJ_ENC_LAYERS")
        if TRAJ_CONV_DILATIONS is not None and len(TRAJ_CONV_DILATIONS) != TRAJ_ENC_LAYERS:
            raise ValueError("TRAJ_CONV_DILATIONS length must equal TRAJ_ENC_LAYERS")
    if not 0.0 <= VISION_DROPOUT_PROB <= 1.0:
        raise ValueError("VISION_DROPOUT_PROB must be in [0, 1]")
    if min(DINO_AUG_NONE_PROB, DINO_AUG_GRAYSCALE_PROB, DINO_AUG_SOLARIZE_PROB) < 0:
        raise ValueError("DINO augmentation branch probabilities must be non-negative")
    if DINO_AUG_NONE_PROB + DINO_AUG_GRAYSCALE_PROB + DINO_AUG_SOLARIZE_PROB > 1.0:
        raise ValueError("DINO none, grayscale, and solarize probabilities must sum to <= 1")
    if not 0.0 <= DEPTH_DROPOUT_PROB <= 1.0:
        raise ValueError("DEPTH_DROPOUT_PROB must be in [0, 1]")
    if not 0.0 <= STATE_DROPOUT_PROB <= 1.0:
        raise ValueError("STATE_DROPOUT_PROB must be in [0, 1]")
    if VISION_DROPOUT_PROB + DEPTH_DROPOUT_PROB + STATE_DROPOUT_PROB > 1.0:
        raise ValueError("Vision, depth, and state dropout probabilities must sum to <= 1")
    if STATE_NOISE_STD < 0:
        raise ValueError("STATE_NOISE_STD must be non-negative")
    if TRAJ_NOISE_STD < 0:
        raise ValueError("TRAJ_NOISE_STD must be non-negative")
    if TRAJ_NOISE_SMOOTH_SIGMA < 0:
        raise ValueError("TRAJ_NOISE_SMOOTH_SIGMA must be non-negative")
    if SAVE_FREQ is not None and SAVE_FREQ <= 0:
        raise ValueError("SAVE_FREQ must be positive when enabled")
    if SAVE_EVERY_N_EPOCHS is not None and SAVE_EVERY_N_EPOCHS <= 0:
        raise ValueError("SAVE_EVERY_N_EPOCHS must be positive when enabled")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    def _default(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, tuple):
            return list(obj)
        return str(obj)

    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_default)


def hparams_dict() -> dict[str, Any]:
    names = [
        "DATASET_REPO_ID", "DATASET_ROOT", "EPISODES", "EXCLUDED_EPISODES", "EXCLUDED_TASK_IDS",
        "DEVICE", "NUM_WORKERS",
        "CAPACITY_PRESET",
        "USE_GLOBAL_TRAJ", "USE_MAIN_TRACKS", "USE_SEC_TRACKS", "USE_MAIN_CAM",
        "USE_SEC_CAM", "USE_JOINT_STATE", "GLOBAL_TRAJ_LENGTH",
        "MAIN_CAM_KEY", "SEC_CAM_KEY",
        "HORIZON", "N_ACTION_STEPS", "N_OBS_STEPS", "OBSERVATION_FRAME_OFFSETS",
        "DOWN_DIMS", "KERNEL_SIZE",
        "N_GROUPS", "DIFFUSION_STEP_EMBED_DIM", "USE_FILM_SCALE_MODULATION",
        "DENOISER_BACKBONE", "DIT_DIM", "DIT_DEPTH", "DIT_NHEAD",
        "DIT_FFN_DIM", "DIT_DROPOUT",
        "VISION_BACKBONE", "PRETRAINED_BACKBONE", "RESIZE_SHAPE", "CROP_RATIO",
        "SPATIAL_SOFTMAX_KP", "NOISE_SCHEDULER_TYPE", "NUM_TRAIN_TIMESTEPS",
        "NUM_INFERENCE_STEPS", "BETA_SCHEDULE", "PREDICTION_TYPE", "CLIP_SAMPLE",
        "CLIP_SAMPLE_RANGE", "LR", "WEIGHT_DECAY", "GRAD_CLIP_NORM",
        "LR_SCHEDULER_TYPE", "LR_WARMUP_STEPS", "LR_DECAY_STEPS", "LR_PEAK",
        "LR_FLOOR", "PLATEAU_FACTOR", "PLATEAU_PATIENCE", "PLATEAU_MIN_LR",
        "VAL_EPISODE_STRIDE", "VAL_FREQ_STEPS", "VAL_SPLIT_SEED", "BATCH_SIZE",
        "STRATIFY_BATCHES_BY_TASK_ID",
        "NUM_TRAIN_STEPS", "NUM_EPOCHS", "LOG_FREQ", "SAVE_FREQ",
        "SAVE_EVERY_N_EPOCHS", "PLOT_FREQ",
        "USE_TRAJ_ENCODER", "TRAJ_ENC_TYPE", "TRAJ_ENC_DIM", "FOURIER_BANDS",
        "TRAJ_USE_FOURIER_FEATURES",
        "TRAJ_TEMPORAL_DIM", "TRAJ_ENC_HIDDEN", "TRAJ_ENC_LAYERS",
        "TRAJ_CONV_KERNEL_SIZES", "TRAJ_CONV_DILATIONS",
        "TRAJ_TF_D_MODEL", "TRAJ_TF_NHEAD", "TRAJ_TF_FFN_DIM", "TRAJ_TF_LAYERS",
        "TRAJ_ACTION_ENCODING", "TRAJ_ACTION_BINARY_THRESHOLD",
        "TRAJ_ACTIVE_ACTION_SIGNAL", "TRAJ_ACTIVE_WINDOW_ONLY",
        "TRAJ_NUM_QUERY_TOKENS",
        "TRAJ_PAST_MASK_PROB", "TRAJ_NOISE_STD", "TRAJ_NOISE_SMOOTH_SIGMA",
        "TRAJ_RETURN_CUTOFF", "TRAJ_RETURN_CUTOFF_SMOOTH_WINDOW",
        "TRAJ_RETURN_CUTOFF_SMOOTH_POLY", "TRAJ_RETURN_CUTOFF_GRADIENT_THRESHOLD",
        "USE_CONDITIONING_FUSION", "FUSION_TOKEN_DIM", "FUSION_OUT_DIM",
        "FUSION_NHEAD", "FUSION_FFN_DIM", "FUSION_LAYERS",
        "FUSION_NUM_OUTPUT_TOKENS",
        "USE_DINO_CAM", "DINO_MODEL",
        "DINO_OUT_DIM", "DINO_PROJ_HIDDEN_DIM", "DINO_FROZEN", "DINO_RESIZE", "DINO_VIS_AUG",
        "DINO_AUG_BRIGHTNESS", "DINO_AUG_CONTRAST", "DINO_AUG_SATURATION",
        "DINO_AUG_HUE", "DINO_AUG_NONE_PROB", "DINO_AUG_GRAYSCALE_PROB",
        "DINO_AUG_SOLARIZE_PROB", "DINO_AUG_SOLARIZE_THRESHOLD",
        "DINO_AUG_CROP_SCALE", "DINO_AUG_CROP_RATIO", "VISION_DROPOUT_PROB",
        "USE_RESNET_ENV_ENCODER", "RESNET_OUT_DIM", "RESNET_PROJ_HIDDEN_DIM",
        "RESNET_FROZEN", "RESNET_RESIZE", "RESNET_LR",
        "USE_DEPTH_ENCODER", "DEPTH_PREFIX", "DEPTH_OUT_DIM", "DEPTH_RESIZE",
        "DEPTH_DROPOUT_PROB", "STATE_DROPOUT_PROB", "STATE_NOISE_STD",
        "CONDITIONING_DIAG_FREQ", "CONDITIONING_DIAG_SEED",
    ]
    return {name: globals()[name] for name in names}


def trajectory_conditioning_dim() -> int:
    if not (USE_TRAJ_ENCODER and USE_GLOBAL_TRAJ):
        return 0
    if TRAJ_ENC_TYPE == "transformer":
        return TRAJ_ENC_DIM * max(TRAJ_NUM_QUERY_TOKENS, 1)
    return TRAJ_ENC_DIM


def encoder_config_dict(
    max_T: int | None,
    dino_cam_keys: list[str],
    depth_cam_keys: list[str],
    resnet_cam_keys: list[str],
) -> dict[str, Any]:
    return {
        "USE_TRAJ_ENCODER": USE_TRAJ_ENCODER,
        "MAX_T": max_T,
        "TRAJ_ENC_TYPE": TRAJ_ENC_TYPE,
        "TRAJ_ENC_DIM": TRAJ_ENC_DIM,
        "FOURIER_BANDS": FOURIER_BANDS,
        "TRAJ_USE_FOURIER_FEATURES": TRAJ_USE_FOURIER_FEATURES,
        "TRAJ_TEMPORAL_DIM": TRAJ_TEMPORAL_DIM,
        "TRAJ_ENC_HIDDEN": TRAJ_ENC_HIDDEN,
        "TRAJ_ENC_LAYERS": TRAJ_ENC_LAYERS,
        "TRAJ_CONV_KERNEL_SIZES": list(TRAJ_CONV_KERNEL_SIZES) if TRAJ_CONV_KERNEL_SIZES is not None else None,
        "TRAJ_CONV_DILATIONS": list(TRAJ_CONV_DILATIONS) if TRAJ_CONV_DILATIONS is not None else None,
        "TRAJ_TF_D_MODEL": TRAJ_TF_D_MODEL,
        "TRAJ_TF_NHEAD": TRAJ_TF_NHEAD,
        "TRAJ_TF_FFN_DIM": TRAJ_TF_FFN_DIM,
        "TRAJ_TF_LAYERS": TRAJ_TF_LAYERS,
        "TRAJ_ACTION_ENCODING": TRAJ_ACTION_ENCODING,
        "TRAJ_ACTION_BINARY_THRESHOLD": TRAJ_ACTION_BINARY_THRESHOLD,
        "TRAJ_ACTIVE_ACTION_SIGNAL": TRAJ_ACTIVE_ACTION_SIGNAL,
        "TRAJ_ACTIVE_WINDOW_ONLY": TRAJ_ACTIVE_WINDOW_ONLY,
        "TRAJ_NUM_QUERY_TOKENS": TRAJ_NUM_QUERY_TOKENS,
        "TRAJ_RETURN_CUTOFF": TRAJ_RETURN_CUTOFF,
        "TRAJ_RETURN_CUTOFF_SMOOTH_WINDOW": TRAJ_RETURN_CUTOFF_SMOOTH_WINDOW,
        "TRAJ_RETURN_CUTOFF_SMOOTH_POLY": TRAJ_RETURN_CUTOFF_SMOOTH_POLY,
        "TRAJ_RETURN_CUTOFF_GRADIENT_THRESHOLD": TRAJ_RETURN_CUTOFF_GRADIENT_THRESHOLD,
        "SHARED_TRAJECTORY_CONDITIONING": True,
        "TRAJECTORY_CONDITIONING_DIM": trajectory_conditioning_dim(),
        "USE_CONDITIONING_FUSION": USE_CONDITIONING_FUSION,
        "FUSION_TOKEN_DIM": FUSION_TOKEN_DIM,
        "FUSION_OUT_DIM": FUSION_OUT_DIM,
        "FUSION_NHEAD": FUSION_NHEAD,
        "FUSION_FFN_DIM": FUSION_FFN_DIM,
        "FUSION_LAYERS": FUSION_LAYERS,
        "FUSION_NUM_OUTPUT_TOKENS": FUSION_NUM_OUTPUT_TOKENS,
        "FUSION_CONDITIONING_DIM": (
            FUSION_OUT_DIM * FUSION_NUM_OUTPUT_TOKENS
            if USE_CONDITIONING_FUSION
            else None
        ),
        "DENOISER_BACKBONE": DENOISER_BACKBONE,
        "DIT_DIM": DIT_DIM,
        "DIT_DEPTH": DIT_DEPTH,
        "DIT_NHEAD": DIT_NHEAD,
        "DIT_FFN_DIM": DIT_FFN_DIM,
        "DIT_DROPOUT": DIT_DROPOUT,
        "USE_DINO_CAM": USE_DINO_CAM,
        "DINO_MODEL": DINO_MODEL,
        "DINO_OUT_DIM": DINO_OUT_DIM,
        "DINO_PROJ_HIDDEN_DIM": DINO_PROJ_HIDDEN_DIM,
        "DINO_FROZEN": DINO_FROZEN,
        "DINO_RESIZE": list(DINO_RESIZE),
        "DINO_CAM_KEYS": dino_cam_keys,
        "USE_RESNET_ENV_ENCODER": USE_RESNET_ENV_ENCODER,
        "VISION_BACKBONE": VISION_BACKBONE,
        "PRETRAINED_BACKBONE": PRETRAINED_BACKBONE,
        "RESNET_OUT_DIM": RESNET_OUT_DIM,
        "RESNET_PROJ_HIDDEN_DIM": RESNET_PROJ_HIDDEN_DIM,
        "RESNET_FROZEN": RESNET_FROZEN,
        "RESNET_RESIZE": list(RESNET_RESIZE),
        "RESNET_CAM_KEYS": resnet_cam_keys,
        "USE_DEPTH_ENCODER": USE_DEPTH_ENCODER and bool(depth_cam_keys),
        "DEPTH_PREFIX": DEPTH_PREFIX,
        "DEPTH_OUT_DIM": DEPTH_OUT_DIM,
        "DEPTH_RESIZE": list(DEPTH_RESIZE),
        "DEPTH_CAM_KEYS": depth_cam_keys,
        "MAIN_CAM_KEY": MAIN_CAM_KEY,
        "SEC_CAM_KEY": SEC_CAM_KEY,
        "OBSERVATION_FRAME_OFFSETS": list(OBSERVATION_FRAME_OFFSETS),
    }


def save_training_state(
    ckpt_dir: Path,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    traj_encoder: nn.Module | None,
    dino_encoder: nn.Module | None,
    resnet_encoder: nn.Module | None,
    depth_encoder: nn.Module | None,
    optimizer: Any,
    lr_scheduler: Any,
    encoder_config: dict[str, Any],
    step: int,
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(ckpt_dir)
    preprocessor.save_pretrained(ckpt_dir)
    postprocessor.save_pretrained(ckpt_dir)
    if traj_encoder is not None:
        torch.save(traj_encoder.state_dict(), ckpt_dir / "traj_encoder.pt")
    if dino_encoder is not None:
        torch.save(dino_encoder.state_dict(), ckpt_dir / "dino_encoder.pt")
    if resnet_encoder is not None:
        torch.save(resnet_encoder.state_dict(), ckpt_dir / "resnet_env_encoder.pt")
    if depth_encoder is not None:
        torch.save(depth_encoder.state_dict(), ckpt_dir / "depth_encoder.pt")
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    if lr_scheduler is not None:
        try:
            from lerobot.optim.schedulers import save_scheduler_state

            save_scheduler_state(lr_scheduler, ckpt_dir)
        except Exception as exc:
            log.warning("Could not save scheduler state: %s", exc)
    save_json(ckpt_dir / "encoder_config.json", encoder_config)
    save_json(ckpt_dir / "trainer_state.json", {"step": step})


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_module_state(module: nn.Module | None, path: Path, name: str) -> None:
    if module is None:
        return
    if not path.exists():
        log.warning("Resume checkpoint has no %s state at %s.", name, path)
        return
    module.load_state_dict(torch.load(path, map_location=DEVICE))
    log.info("Restored %s from %s.", name, path)


def load_policy_state(policy: Any, ckpt_dir: Path) -> None:
    safetensors_path = ckpt_dir / "model.safetensors"
    pytorch_path = ckpt_dir / "pytorch_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file

        policy.load_state_dict(load_file(safetensors_path), strict=True)
        log.info("Restored policy from %s.", safetensors_path)
    elif pytorch_path.exists():
        policy.load_state_dict(torch.load(pytorch_path, map_location=DEVICE), strict=True)
        log.info("Restored policy from %s.", pytorch_path)
    else:
        raise FileNotFoundError(f"No policy weights found in {ckpt_dir}.")


def load_scheduler_state_if_exists(lr_scheduler: Any, ckpt_dir: Path) -> bool:
    if lr_scheduler is None:
        return False
    try:
        from lerobot.optim.schedulers import load_scheduler_state

        load_scheduler_state(lr_scheduler, ckpt_dir)
        log.info("Restored scheduler state from %s.", ckpt_dir)
        return True
    except Exception:
        pass
    for name in ("scheduler.pt", "scheduler_state.pt", "lr_scheduler.pt"):
        path = ckpt_dir / name
        if path.exists():
            lr_scheduler.load_state_dict(torch.load(path, map_location=DEVICE))
            log.info("Restored scheduler state from %s.", path)
            return True
    log.warning("Resume checkpoint has no scheduler state; scheduler will continue from current initialization.")
    return False


def resume_training_state(
    ckpt_dir: Path,
    policy: Any,
    traj_encoder: nn.Module | None,
    dino_encoder: nn.Module | None,
    resnet_encoder: nn.Module | None,
    depth_encoder: nn.Module | None,
    optimizer: Any,
    lr_scheduler: Any,
) -> int:
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {ckpt_dir}")
    load_policy_state(policy, ckpt_dir)
    load_module_state(traj_encoder, ckpt_dir / "traj_encoder.pt", "trajectory encoder")
    load_module_state(dino_encoder, ckpt_dir / "dino_encoder.pt", "DINO encoder")
    load_module_state(resnet_encoder, ckpt_dir / "resnet_env_encoder.pt", "ResNet ENV encoder")
    load_module_state(depth_encoder, ckpt_dir / "depth_encoder.pt", "depth encoder")

    optimizer_path = ckpt_dir / "optimizer.pt"
    if optimizer_path.exists():
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=DEVICE))
        log.info("Restored optimizer from %s.", optimizer_path)
    else:
        log.warning("Resume checkpoint has no optimizer.pt; continuing with a fresh optimizer.")

    trainer_state = load_json_if_exists(ckpt_dir / "trainer_state.json") or {}
    if "step" in trainer_state:
        start_step = int(trainer_state["step"])
    else:
        try:
            start_step = int(ckpt_dir.name.removeprefix("checkpoint_"))
        except ValueError:
            log.warning("Could not infer resume step from checkpoint name; starting loop at step 0.")
            start_step = 0

    scheduler_restored = load_scheduler_state_if_exists(lr_scheduler, ckpt_dir)
    if lr_scheduler is not None and not scheduler_restored and LR_SCHEDULER_TYPE == "cosine":
        for _ in range(start_step):
            lr_scheduler.step()
        log.info("Advanced fresh cosine scheduler by %s step(s).", f"{start_step:,}")
    return start_step


def main() -> None:
    global NUM_TRAIN_STEPS, LR_DECAY_STEPS

    args = parse_args()
    apply_overrides(args)
    apply_capacity_preset()
    apply_denoiser_overrides(args)
    apply_resume_config_overrides(args.resume_from)
    validate_static_config()
    resolve_output_dir()
    add_file_logger(OUTPUT_DIR)

    random.seed(SEED)
    torch.manual_seed(SEED)

    token = args.hf_token or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
        log.info("HuggingFace login successful (token provided).")
    else:
        log.info("No explicit HF token; relying on cached HuggingFace credentials.")

    log.info("Output directory : %s", OUTPUT_DIR.resolve())
    log.info("Device           : %s", DEVICE)
    log.info("Dataset          : %s", DATASET_REPO_ID)
    log.info("Batch size       : %s", BATCH_SIZE)
    log.info("Trajectory length: %s", GLOBAL_TRAJ_LENGTH)
    log.info(
        "Trajectory action: %s | binary threshold=%.3f",
        TRAJ_ACTION_ENCODING,
        TRAJ_ACTION_BINARY_THRESHOLD,
    )
    log.info("Trajectory preprocessing: RT active manipulation uses action > 0, no threshold")
    log.info("Trajectory Fourier features: %s", TRAJ_USE_FOURIER_FEATURES)
    log.info("Trajectory active action signal: %s", TRAJ_ACTIVE_ACTION_SIGNAL)
    log.info("Trajectory active window only: %s", TRAJ_ACTIVE_WINDOW_ONLY)
    log.info(
        "Conditioning fusion: %s | token_dim=%s | out_dim=%s | heads=%s | layers=%s | output_tokens=%s",
        USE_CONDITIONING_FUSION,
        FUSION_TOKEN_DIM,
        FUSION_OUT_DIM,
        FUSION_NHEAD,
        FUSION_LAYERS,
        FUSION_NUM_OUTPUT_TOKENS,
    )
    log.info(
        "Trajectory return cutoff: %s | smooth_window=%d | grad_threshold=%.4f",
        TRAJ_RETURN_CUTOFF,
        TRAJ_RETURN_CUTOFF_SMOOTH_WINDOW,
        TRAJ_RETURN_CUTOFF_GRADIENT_THRESHOLD,
    )
    log.info("Task stratified  : %s", STRATIFY_BATCHES_BY_TASK_ID)
    log.info(
        "Checkpointing    : every %s steps | every %s epochs",
        SAVE_FREQ,
        SAVE_EVERY_N_EPOCHS,
    )
    log.info("Capacity preset  : %s", CAPACITY_PRESET)
    log.info("Joint state input: %s", USE_JOINT_STATE)
    log.info("Observation frame offsets: %s", OBSERVATION_FRAME_OFFSETS)
    log.info("Trajectory queries: %s", TRAJ_NUM_QUERY_TOKENS)
    log.info("Past-mask probability: %.3f", TRAJ_PAST_MASK_PROB)
    log.info("Vision dropout probability: %.3f", VISION_DROPOUT_PROB)
    log.info(
        "ResNet ENV encoder: %s | backbone=%s | weights=%s | out_dim=%s | frozen=%s | resize=%s | lr=%s",
        USE_RESNET_ENV_ENCODER,
        VISION_BACKBONE,
        PRETRAINED_BACKBONE,
        RESNET_OUT_DIM,
        RESNET_FROZEN,
        RESNET_RESIZE,
        RESNET_LR,
    )
    log.info("Depth encoder     : %s", USE_DEPTH_ENCODER)
    log.info("Depth dropout probability: %.3f", DEPTH_DROPOUT_PROB)
    log.info("State dropout probability: %.3f", STATE_DROPOUT_PROB)
    log.info("Normalized state noise std: %.3f", STATE_NOISE_STD)
    log.info("Conditioning diagnostic frequency: %s", CONDITIONING_DIAG_FREQ)
    log.info(
        "Action denoiser  : %s | dit_dim=%s | depth=%s | heads=%s | ffn_dim=%s | dropout=%.3f",
        DENOISER_BACKBONE,
        DIT_DIM,
        DIT_DEPTH,
        DIT_NHEAD,
        DIT_FFN_DIM,
        DIT_DROPOUT,
    )
    save_json(OUTPUT_DIR / "hparams.json", hparams_dict())

    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.datasets.factory import IMAGENET_STATS, resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.utils import cycle
    from lerobot.optim.optimizers import AdamWConfig
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from torch.utils.data import DataLoader, Subset
    from tqdm.auto import tqdm

    from diffusion_transformer import DenoiserConfig, make_action_denoiser

    log.info("Loading dataset metadata...")
    ds_meta = LeRobotDatasetMetadata(repo_id=DATASET_REPO_ID, root=DATASET_ROOT)
    for key, ft in ds_meta.features.items():
        log.info("feature %-50s dtype=%-10s shape=%s", key, ft["dtype"], ft["shape"])

    track_keys_to_flatten = []
    if USE_GLOBAL_TRAJ and "observation.global_trajectory" in ds_meta.features:
        track_keys_to_flatten.append("observation.global_trajectory")
    if USE_MAIN_TRACKS and "observation.main_cam_point_tracks" in ds_meta.features:
        track_keys_to_flatten.append("observation.main_cam_point_tracks")
    if USE_SEC_TRACKS and "observation.sec_cam_point_tracks" in ds_meta.features:
        track_keys_to_flatten.append("observation.sec_cam_point_tracks")

    track_flat_dims = {
        key: int(torch.tensor(ds_meta.features[key]["shape"]).prod().item())
        for key in track_keys_to_flatten
    }
    if "observation.global_trajectory" in track_flat_dims:
        track_flat_dims["observation.global_trajectory"] = GLOBAL_TRAJ_LENGTH * 3
    total_track_flat_dim = sum(track_flat_dims.values())
    state_dim = ds_meta.features["observation.state"]["shape"][0]
    if total_track_flat_dim <= 0 and not USE_MAIN_CAM and not USE_SEC_CAM and not USE_DINO_CAM and not USE_RESNET_ENV_ENCODER:
        raise ValueError("Enable at least one conditioning input: tracks, camera, DINO camera, or ResNet ENV camera.")

    log.info("Joint state dim      : %s", state_dim)
    log.info("Track keys selected  : %s", track_keys_to_flatten)
    log.info("Track flat dims      : %s", track_flat_dims)

    input_features: dict[str, PolicyFeature] = {}
    input_features["observation.state"] = PolicyFeature(
        type=FeatureType.STATE,
        shape=(state_dim,) if USE_JOINT_STATE else (1,),
    )
    if total_track_flat_dim > 0:
        input_features["observation.environment_state"] = PolicyFeature(
            type=FeatureType.ENV,
            shape=(total_track_flat_dim,),
        )

    if USE_MAIN_CAM and MAIN_CAM_KEY in ds_meta.features:
        cam_shape = tuple(ds_meta.features[MAIN_CAM_KEY]["shape"])
        if len(cam_shape) == 3 and cam_shape[2] in (1, 3):
            cam_shape = (cam_shape[2], cam_shape[0], cam_shape[1])
        input_features[MAIN_CAM_KEY] = PolicyFeature(type=FeatureType.VISUAL, shape=cam_shape)

    if USE_SEC_CAM and not USE_DINO_CAM and not USE_RESNET_ENV_ENCODER and SEC_CAM_KEY in ds_meta.features:
        sc_shape = tuple(ds_meta.features[SEC_CAM_KEY]["shape"])
        if len(sc_shape) == 3 and sc_shape[2] in (1, 3):
            sc_shape = (sc_shape[2], sc_shape[0], sc_shape[1])
        input_features[SEC_CAM_KEY] = PolicyFeature(type=FeatureType.VISUAL, shape=sc_shape)

    traj_aug = TrajAugTransform(
        TRAJ_PAST_MASK_PROB,
        TRAJ_NOISE_STD,
        TRAJ_NOISE_SMOOTH_SIGMA,
        active_action_signal=TRAJ_ACTIVE_ACTION_SIGNAL,
        active_window_only=TRAJ_ACTIVE_WINDOW_ONLY,
        action_binary_threshold=TRAJ_ACTION_BINARY_THRESHOLD,
        return_cutoff=TRAJ_RETURN_CUTOFF,
        return_cutoff_smooth_window=TRAJ_RETURN_CUTOFF_SMOOTH_WINDOW,
        return_cutoff_smooth_poly=TRAJ_RETURN_CUTOFF_SMOOTH_POLY,
        return_cutoff_gradient_threshold=TRAJ_RETURN_CUTOFF_GRADIENT_THRESHOLD,
    )
    flatten_transform = FlattenTracksTransform(track_keys_to_flatten, USE_JOINT_STATE)

    traj_encoder = None
    traj_key = "observation.global_trajectory"
    max_T = None
    total_track_flat_dim_enc = total_track_flat_dim
    if USE_TRAJ_ENCODER and USE_GLOBAL_TRAJ and traj_key in ds_meta.features:
        traj_shape = tuple(ds_meta.features[traj_key]["shape"])
        source_max_T = int(traj_shape[0])
        max_T = GLOBAL_TRAJ_LENGTH
        traj_encoder = TrajectoryEncoder(
            max_T=max_T,
            fourier_bands=FOURIER_BANDS,
            temporal_dim=TRAJ_TEMPORAL_DIM,
            out_dim=TRAJ_ENC_DIM,
            enc_type=TRAJ_ENC_TYPE,
            hidden=TRAJ_ENC_HIDDEN,
            n_layers=TRAJ_ENC_LAYERS,
            conv_kernel_sizes=TRAJ_CONV_KERNEL_SIZES,
            conv_dilations=TRAJ_CONV_DILATIONS,
            d_model=TRAJ_TF_D_MODEL,
            nhead=TRAJ_TF_NHEAD,
            ffn_dim=TRAJ_TF_FFN_DIM,
            tf_layers=TRAJ_TF_LAYERS,
            action_encoding=TRAJ_ACTION_ENCODING,
            action_binary_threshold=TRAJ_ACTION_BINARY_THRESHOLD,
            num_query_tokens=TRAJ_NUM_QUERY_TOKENS,
            use_fourier_features=TRAJ_USE_FOURIER_FEATURES,
        ).to(DEVICE)
        traj_output_dim = (
            TRAJ_ENC_DIM * max(TRAJ_NUM_QUERY_TOKENS, 1)
            if TRAJ_ENC_TYPE == "transformer"
            else TRAJ_ENC_DIM
        )
        raw_flat = GLOBAL_TRAJ_LENGTH * int(traj_shape[1])
        total_track_flat_dim_enc = total_track_flat_dim - raw_flat
        if total_track_flat_dim_enc > 0:
            input_features["observation.environment_state"] = PolicyFeature(
                type=FeatureType.ENV,
                shape=(total_track_flat_dim_enc,),
            )
        else:
            input_features.pop("observation.environment_state", None)
        log.info("TrajectoryEncoder: %s | fourier=%s | resample %s -> %s | queries=%s | shared dim=%s | params=%s",
                 TRAJ_ENC_TYPE, TRAJ_USE_FOURIER_FEATURES, source_max_T, max_T, TRAJ_NUM_QUERY_TOKENS,
                 traj_output_dim,
                 f"{sum(p.numel() for p in traj_encoder.parameters()):,}")
    else:
        log.info("TrajectoryEncoder disabled; using raw flattened trajectory when selected.")

    if USE_CONDITIONING_FUSION and total_track_flat_dim_enc > 0:
        raise ValueError(
            "Conditioning fusion does not support raw flattened ENV tracks yet; "
            "enable the trajectory encoder and disable point-track inputs."
        )

    dino_encoder = None
    dino_aug = None
    dino_cam_keys: list[str] = []
    resnet_encoder = None
    resnet_cam_keys: list[str] = []
    depth_encoder = None
    depth_cam_keys: list[str] = []
    if USE_DINO_CAM:
        if MAIN_CAM_KEY in ds_meta.features:
            dino_cam_keys.append(MAIN_CAM_KEY)
        if USE_SEC_CAM and SEC_CAM_KEY in ds_meta.features:
            dino_cam_keys.append(SEC_CAM_KEY)
        if not dino_cam_keys:
            raise ValueError(f"USE_DINO_CAM=True but neither {MAIN_CAM_KEY!r} nor {SEC_CAM_KEY!r} exists.")

        dino_encoder = DinoVisualEncoder(
            model_name=DINO_MODEL,
            out_dim=DINO_OUT_DIM,
            proj_hidden_dim=DINO_PROJ_HIDDEN_DIM,
            frozen=DINO_FROZEN,
            resize=DINO_RESIZE,
        ).to(DEVICE)
        new_env_dim = total_track_flat_dim_enc + DINO_OUT_DIM * len(dino_cam_keys)
        input_features["observation.environment_state"] = PolicyFeature(
            type=FeatureType.ENV,
            shape=(new_env_dim,),
        )
        log.info("DINO encoder: %s | head_hidden=%s | cameras=%s | env dim %s -> %s | trainable params=%s",
                 DINO_MODEL, DINO_PROJ_HIDDEN_DIM, dino_cam_keys, total_track_flat_dim_enc, new_env_dim,
                 f"{sum(p.numel() for p in dino_encoder.parameters() if p.requires_grad):,}")
        if DINO_VIS_AUG:
            dino_aug = DinoImageAugTransform(
                brightness=DINO_AUG_BRIGHTNESS,
                contrast=DINO_AUG_CONTRAST,
                saturation=DINO_AUG_SATURATION,
                hue=DINO_AUG_HUE,
                none_prob=DINO_AUG_NONE_PROB,
                grayscale_prob=DINO_AUG_GRAYSCALE_PROB,
                solarize_prob=DINO_AUG_SOLARIZE_PROB,
                solarize_threshold=DINO_AUG_SOLARIZE_THRESHOLD,
                crop_scale=DINO_AUG_CROP_SCALE,
                crop_ratio=DINO_AUG_CROP_RATIO,
            ).to(DEVICE)
            log.info(
                "DINO visual aug: brightness=%.3f contrast=%.3f saturation=%.3f hue=%.3f "
                "none_prob=%.3f grayscale_prob=%.3f solarize_prob=%.3f solarize_threshold=%.3f "
                "crop_scale=%s crop_ratio=%s",
                DINO_AUG_BRIGHTNESS,
                DINO_AUG_CONTRAST,
                DINO_AUG_SATURATION,
                DINO_AUG_HUE,
                DINO_AUG_NONE_PROB,
                DINO_AUG_GRAYSCALE_PROB,
                DINO_AUG_SOLARIZE_PROB,
                DINO_AUG_SOLARIZE_THRESHOLD,
                DINO_AUG_CROP_SCALE,
                DINO_AUG_CROP_RATIO,
            )
        else:
            log.info("DINO visual aug: disabled")

    if USE_RESNET_ENV_ENCODER:
        if MAIN_CAM_KEY in ds_meta.features:
            resnet_cam_keys.append(MAIN_CAM_KEY)
        if USE_SEC_CAM and SEC_CAM_KEY in ds_meta.features:
            resnet_cam_keys.append(SEC_CAM_KEY)
        if not resnet_cam_keys:
            raise ValueError(
                f"USE_RESNET_ENV_ENCODER=True but neither {MAIN_CAM_KEY!r} nor {SEC_CAM_KEY!r} exists."
            )
        resnet_encoder = ResNetEnvVisualEncoder(
            backbone=VISION_BACKBONE,
            out_dim=RESNET_OUT_DIM,
            proj_hidden_dim=RESNET_PROJ_HIDDEN_DIM,
            pretrained_weights=PRETRAINED_BACKBONE,
            frozen=RESNET_FROZEN,
            resize=RESNET_RESIZE,
        ).to(DEVICE)
        current_env_dim = (
            input_features["observation.environment_state"].shape[0]
            if "observation.environment_state" in input_features
            else 0
        )
        input_features["observation.environment_state"] = PolicyFeature(
            type=FeatureType.ENV,
            shape=(current_env_dim + RESNET_OUT_DIM * len(resnet_cam_keys),),
        )
        log.info(
            "ResNet ENV encoder: %s | weights=%s | cameras=%s | env dim %s -> %s | trainable params=%s",
            VISION_BACKBONE,
            PRETRAINED_BACKBONE,
            resnet_cam_keys,
            current_env_dim,
            current_env_dim + RESNET_OUT_DIM * len(resnet_cam_keys),
            f"{sum(p.numel() for p in resnet_encoder.parameters() if p.requires_grad):,}",
        )
    else:
        log.info("ResNet ENV encoder disabled")

    visual_depth_source_cam_keys = dino_cam_keys or resnet_cam_keys
    if USE_DEPTH_ENCODER and visual_depth_source_cam_keys:
        for camera_key in visual_depth_source_cam_keys:
            depth_key = depth_key_for(camera_key)
            if depth_key in ds_meta.features:
                depth_cam_keys.append(depth_key)
        if depth_cam_keys:
            depth_encoder = DepthVisualEncoder(DEPTH_OUT_DIM, DEPTH_RESIZE).to(DEVICE)
            current_env_dim = (
                input_features["observation.environment_state"].shape[0]
                if "observation.environment_state" in input_features
                else 0
            )
            input_features["observation.environment_state"] = PolicyFeature(
                type=FeatureType.ENV,
                shape=(current_env_dim + DEPTH_OUT_DIM * len(depth_cam_keys),),
            )
            log.info(
                "Depth encoder: cameras=%s | env dim %s -> %s | params=%s",
                depth_cam_keys,
                current_env_dim,
                current_env_dim + DEPTH_OUT_DIM * len(depth_cam_keys),
                f"{sum(p.numel() for p in depth_encoder.parameters()):,}",
            )
        else:
            log.info("Depth encoder requested but no matching %s.* features found.", DEPTH_PREFIX)
    else:
        log.info("Depth encoder disabled")

    for key, feature in input_features.items():
        log.info("input_feature %-50s type=%s shape=%s", key, feature.type.value, feature.shape)

    drop_n_last_frames = HORIZON - N_ACTION_STEPS - N_OBS_STEPS + 1
    dp_cfg = DiffusionConfig(
        horizon=HORIZON,
        n_obs_steps=N_OBS_STEPS,
        n_action_steps=N_ACTION_STEPS,
        drop_n_last_frames=drop_n_last_frames,
        down_dims=DOWN_DIMS,
        kernel_size=KERNEL_SIZE,
        n_groups=N_GROUPS,
        diffusion_step_embed_dim=DIFFUSION_STEP_EMBED_DIM,
        use_film_scale_modulation=USE_FILM_SCALE_MODULATION,
        vision_backbone=VISION_BACKBONE,
        pretrained_backbone_weights=PRETRAINED_BACKBONE if ((USE_MAIN_CAM or (USE_SEC_CAM and not USE_DINO_CAM)) and not USE_RESNET_ENV_ENCODER) else None,
        use_group_norm=not ((USE_MAIN_CAM or (USE_SEC_CAM and not USE_DINO_CAM)) and not USE_RESNET_ENV_ENCODER),
        resize_shape=RESIZE_SHAPE if ((USE_MAIN_CAM or (USE_SEC_CAM and not USE_DINO_CAM)) and not USE_RESNET_ENV_ENCODER) else None,
        crop_ratio=CROP_RATIO,
        spatial_softmax_num_keypoints=SPATIAL_SOFTMAX_KP,
        noise_scheduler_type=NOISE_SCHEDULER_TYPE,
        num_train_timesteps=NUM_TRAIN_TIMESTEPS,
        num_inference_steps=NUM_INFERENCE_STEPS,
        beta_schedule=BETA_SCHEDULE,
        prediction_type=PREDICTION_TYPE,
        clip_sample=CLIP_SAMPLE,
        clip_sample_range=CLIP_SAMPLE_RANGE,
        do_mask_loss_for_padding=False,
        normalization_mapping={
            "STATE": NormalizationMode.MIN_MAX if USE_JOINT_STATE else NormalizationMode.IDENTITY,
            "ENV": NormalizationMode.IDENTITY,
            "VISUAL": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MIN_MAX,
        },
        optimizer_lr=LR,
        optimizer_weight_decay=WEIGHT_DECAY,
        device=DEVICE,
        input_features=input_features,
    )

    delta_timestamps = resolve_delta_timestamps(dp_cfg, ds_meta)
    observation_timestamps = [
        offset / ds_meta.fps for offset in OBSERVATION_FRAME_OFFSETS
    ]
    for key in ds_meta.features:
        if key.startswith("observation."):
            delta_timestamps[key] = observation_timestamps
    log.info(
        "Observation timestamps: %s seconds at %s fps",
        observation_timestamps,
        ds_meta.fps,
    )

    requested_episodes = (
        list(EPISODES)
        if EPISODES is not None
        else list(range(ds_meta.total_episodes))
    )
    invalid_exclusions = {
        ep for ep in EXCLUDED_EPISODES
        if ep < 0 or ep >= ds_meta.total_episodes
    }
    if invalid_exclusions:
        raise ValueError(
            f"EXCLUDED_EPISODES contains invalid ids: {sorted(invalid_exclusions)}"
        )
    ep_pool = [
        ep for ep in requested_episodes
        if ep not in EXCLUDED_EPISODES
    ]
    if not ep_pool:
        raise ValueError("No episodes remain after applying EXCLUDED_EPISODES.")
    log.info(
        "Episodes selected: %d/%d; excluded_episodes=%s; excluded_task_ids=%s",
        len(ep_pool),
        ds_meta.total_episodes,
        sorted(EXCLUDED_EPISODES),
        sorted(EXCLUDED_TASK_IDS),
    )
    val_ep: list[int] = []
    if LR_SCHEDULER_TYPE == "plateau":
        rng = random.Random(VAL_SPLIT_SEED)
        val_ep = [rng.choice(ep_pool[i:i + VAL_EPISODE_STRIDE]) for i in range(0, len(ep_pool), VAL_EPISODE_STRIDE)]
        train_ep = [e for e in ep_pool if e not in set(val_ep)]
        if not train_ep:
            raise ValueError("No training episodes after validation split.")
        dataset_ep = train_ep
        log.info("Episode split: %s train / %s val", len(train_ep), len(val_ep))
    else:
        dataset_ep = ep_pool

    log.info("Building training dataset...")
    dataset = LeRobotDataset(
        repo_id=DATASET_REPO_ID,
        root=DATASET_ROOT,
        episodes=dataset_ep,
        delta_timestamps=delta_timestamps,
        download_videos=True,
    )
    active_dataset = dataset
    active_task_ids: list[int] | None = None
    if EXCLUDED_TASK_IDS:
        if "task_id" not in dataset.hf_dataset.column_names:
            raise ValueError("--exclude-task-ids requires a scalar 'task_id' dataset feature.")
        all_task_ids = [scalar_task_id(value) for value in dataset.hf_dataset["task_id"]]
        keep_indices = [
            index
            for index, task_id in enumerate(all_task_ids)
            if task_id not in EXCLUDED_TASK_IDS
        ]
        if not keep_indices:
            raise ValueError(f"No frames remain after excluding task ids {sorted(EXCLUDED_TASK_IDS)}.")
        dropped = len(all_task_ids) - len(keep_indices)
        active_task_ids = [all_task_ids[index] for index in keep_indices]
        active_dataset = Subset(dataset, keep_indices)
        log.info(
            "Task-id exclusion enabled | excluded=%s | dropped=%s/%s frames | remaining_task_counts=%s",
            sorted(EXCLUDED_TASK_IDS),
            f"{dropped:,}",
            f"{len(all_task_ids):,}",
            {
                task_id: active_task_ids.count(task_id)
                for task_id in sorted(set(active_task_ids))
            },
        )

    if USE_MAIN_CAM or USE_SEC_CAM or USE_DINO_CAM or USE_RESNET_ENV_ENCODER:
        for key in dataset.meta.camera_keys:
            for stat_type, stat_val in IMAGENET_STATS.items():
                dataset.meta.stats.setdefault(key, {})[stat_type] = torch.tensor(stat_val, dtype=torch.float32)

    loader_kwargs = {
        "num_workers": NUM_WORKERS,
        "pin_memory": DEVICE == "cuda",
        "prefetch_factor": 2 if NUM_WORKERS > 0 else None,
    }
    if STRATIFY_BATCHES_BY_TASK_ID:
        if "task_id" not in dataset.hf_dataset.column_names:
            raise ValueError(
                "STRATIFY_BATCHES_BY_TASK_ID=True requires a scalar 'task_id' dataset feature."
            )
        task_ids = active_task_ids
        if task_ids is None:
            task_ids = [scalar_task_id(value) for value in dataset.hf_dataset["task_id"]]
        batch_sampler = TaskStratifiedBatchSampler(
            task_ids=task_ids,
            batch_size=BATCH_SIZE,
            drop_last=True,
            seed=SEED,
        )
        dataloader = DataLoader(active_dataset, batch_sampler=batch_sampler, **loader_kwargs)
        task_counts = {
            task_id: len(indices)
            for task_id, indices in batch_sampler.task_to_indices.items()
        }
        log.info("Task-stratified batches enabled | frame counts=%s", task_counts)
    else:
        dataloader = DataLoader(
            active_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        )
    steps_per_epoch = len(dataloader)
    if NUM_EPOCHS is not None:
        NUM_TRAIN_STEPS = math.ceil(NUM_EPOCHS * steps_per_epoch)
        log.info("NUM_EPOCHS=%s -> NUM_TRAIN_STEPS=%s", NUM_EPOCHS, f"{NUM_TRAIN_STEPS:,}")
        if LR_SCHEDULER_TYPE == "cosine":
            LR_DECAY_STEPS = math.ceil(NUM_TRAIN_STEPS * 0.75)
            log.info("Cosine decay steps set to 3/4 of training: %s", f"{LR_DECAY_STEPS:,}")
    log.info("Train dataset: %s frames | %s steps/epoch | %.1f effective epochs",
             f"{len(active_dataset):,}", f"{steps_per_epoch:,}", NUM_TRAIN_STEPS / steps_per_epoch)

    val_loader = None
    if LR_SCHEDULER_TYPE == "plateau":
        val_dataset = LeRobotDataset(
            repo_id=DATASET_REPO_ID,
            root=DATASET_ROOT,
            episodes=val_ep,
            delta_timestamps=delta_timestamps,
            download_videos=False,
        )
        val_dataset.meta.stats = dataset.meta.stats
        active_val_dataset = val_dataset
        if EXCLUDED_TASK_IDS:
            if "task_id" not in val_dataset.hf_dataset.column_names:
                raise ValueError("--exclude-task-ids requires a scalar 'task_id' validation dataset feature.")
            val_task_ids = [scalar_task_id(value) for value in val_dataset.hf_dataset["task_id"]]
            val_keep_indices = [
                index
                for index, task_id in enumerate(val_task_ids)
                if task_id not in EXCLUDED_TASK_IDS
            ]
            if not val_keep_indices:
                raise ValueError(
                    f"No validation frames remain after excluding task ids {sorted(EXCLUDED_TASK_IDS)}."
                )
            active_val_dataset = Subset(val_dataset, val_keep_indices)
            log.info(
                "Validation task-id exclusion dropped %s/%s frames.",
                f"{len(val_task_ids) - len(val_keep_indices):,}",
                f"{len(val_task_ids):,}",
            )
        val_loader = DataLoader(
            active_val_dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            shuffle=False,
            pin_memory=(DEVICE == "cuda"),
            drop_last=False,
            prefetch_factor=2 if NUM_WORKERS > 0 else None,
        )
        log.info("Val dataset: %s frames | %s batches", f"{len(active_val_dataset):,}", f"{len(val_loader):,}")

    log.info("Building policy...")
    policy = make_policy(cfg=dp_cfg, ds_meta=ds_meta)
    shared_traj_key = "observation.shared_trajectory"
    denoiser_cfg = DenoiserConfig(
        backbone=DENOISER_BACKBONE,
        dit_dim=DIT_DIM,
        dit_depth=DIT_DEPTH,
        dit_nhead=DIT_NHEAD,
        dit_ffn_dim=DIT_FFN_DIM,
        dit_dropout=DIT_DROPOUT,
    )

    class FusedConditioningDiffusionModel(DiffusionModel):
        """Fuse conditioning tokens before passing one global vector to the U-Net."""

        def __init__(
            self,
            config: DiffusionConfig,
            shared_traj_dim: int,
            *,
            state_dim: int,
            dino_dim: int,
            depth_dim: int,
        ):
            super().__init__(config)
            self.shared_traj_dim = shared_traj_dim
            self.dino_dim = dino_dim
            self.depth_dim = depth_dim
            if config.image_features:
                raise ValueError("Conditioning fusion expects custom visual features in ENV, not LeRobot image features.")
            self.fusion = ConditioningFusion(
                state_dim=state_dim,
                dino_dim=dino_dim,
                depth_dim=depth_dim,
                traj_dim=TRAJ_ENC_DIM if shared_traj_dim > 0 else 0,
                n_obs_steps=config.n_obs_steps,
                token_dim=FUSION_TOKEN_DIM,
                out_dim=FUSION_OUT_DIM,
                nhead=FUSION_NHEAD,
                ffn_dim=FUSION_FFN_DIM,
                n_layers=FUSION_LAYERS,
                num_output_tokens=FUSION_NUM_OUTPUT_TOKENS,
            )
            self.unet = make_action_denoiser(
                config,
                global_cond_dim=self.fusion.output_dim,
                denoiser_cfg=denoiser_cfg,
            )
            if config.compile_model:
                self.unet = torch.compile(self.unet, mode=config.compile_mode)

        def _prepare_global_conditioning(
            self,
            batch: dict[str, torch.Tensor],
        ) -> torch.Tensor:
            state = batch.get("observation.state")
            env = batch.get("observation.environment_state")
            dino = None
            depth = None
            if env is not None:
                offset = 0
                if self.dino_dim > 0:
                    dino = env[..., offset : offset + self.dino_dim]
                    offset += self.dino_dim
                if self.depth_dim > 0:
                    depth = env[..., offset : offset + self.depth_dim]
                    offset += self.depth_dim
                if offset != env.shape[-1]:
                    raise ValueError(
                        f"ENV feature dim mismatch: consumed {offset}, got {env.shape[-1]}"
                    )
            shared = batch.get(shared_traj_key)
            if shared is not None and (
                shared.ndim != 3 or shared.shape[-1] != TRAJ_ENC_DIM
            ):
                raise ValueError(
                    f"{shared_traj_key} must have shape [B, N, {TRAJ_ENC_DIM}], "
                    f"got {tuple(shared.shape)}"
                )
            return self.fusion(
                state=state,
                dino=dino,
                depth=depth,
                trajectory=shared,
            )

    class SharedTrajectoryDiffusionModel(DiffusionModel):
        """Append one episode-level trajectory embedding after temporal features."""

        def __init__(self, config: DiffusionConfig, shared_traj_dim: int):
            super().__init__(config)
            self.shared_traj_dim = shared_traj_dim

            temporal_cond_dim = config.robot_state_feature.shape[0]
            if config.image_features:
                num_images = len(config.image_features)
                if config.use_separate_rgb_encoder_per_camera:
                    temporal_cond_dim += self.rgb_encoder[0].feature_dim * num_images
                else:
                    temporal_cond_dim += self.rgb_encoder.feature_dim * num_images
            if config.env_state_feature:
                temporal_cond_dim += config.env_state_feature.shape[0]

            total_cond_dim = temporal_cond_dim * config.n_obs_steps + shared_traj_dim
            self.unet = make_action_denoiser(
                config,
                global_cond_dim=total_cond_dim,
                denoiser_cfg=denoiser_cfg,
            )
            if config.compile_model:
                self.unet = torch.compile(self.unet, mode=config.compile_mode)

        def _prepare_global_conditioning(
            self,
            batch: dict[str, torch.Tensor],
        ) -> torch.Tensor:
            temporal = super()._prepare_global_conditioning(batch)
            shared = batch[shared_traj_key]
            if shared.ndim != 2 or shared.shape[-1] != self.shared_traj_dim:
                raise ValueError(
                    f"{shared_traj_key} must have shape [B, {self.shared_traj_dim}], "
                    f"got {tuple(shared.shape)}"
                )
            return torch.cat([temporal, shared], dim=-1)

    class BackboneOverrideDiffusionModel(DiffusionModel):
        """Use the selected denoiser with LeRobot's standard global conditioning."""

        def __init__(self, config: DiffusionConfig):
            super().__init__(config)
            global_cond_dim = config.robot_state_feature.shape[0]
            if config.image_features:
                num_images = len(config.image_features)
                if config.use_separate_rgb_encoder_per_camera:
                    global_cond_dim += self.rgb_encoder[0].feature_dim * num_images
                else:
                    global_cond_dim += self.rgb_encoder.feature_dim * num_images
            if config.env_state_feature:
                global_cond_dim += config.env_state_feature.shape[0]
            self.unet = make_action_denoiser(
                config,
                global_cond_dim=global_cond_dim * config.n_obs_steps,
                denoiser_cfg=denoiser_cfg,
            )
            if config.compile_model:
                self.unet = torch.compile(self.unet, mode=config.compile_mode)

    dino_env_dim = DINO_OUT_DIM * len(dino_cam_keys) if dino_encoder is not None else 0
    resnet_env_dim = RESNET_OUT_DIM * len(resnet_cam_keys) if resnet_encoder is not None else 0
    visual_env_dim = dino_env_dim + resnet_env_dim
    depth_env_dim = DEPTH_OUT_DIM * len(depth_cam_keys) if depth_encoder is not None else 0
    if USE_CONDITIONING_FUSION:
        policy.diffusion = FusedConditioningDiffusionModel(
            dp_cfg,
            traj_output_dim if traj_encoder is not None else 0,
            state_dim=(state_dim if USE_JOINT_STATE else 1),
            dino_dim=visual_env_dim,
            depth_dim=depth_env_dim,
        ).to(DEVICE)
    elif traj_encoder is not None:
        policy.diffusion = SharedTrajectoryDiffusionModel(
            dp_cfg,
            traj_output_dim,
        ).to(DEVICE)
    elif DENOISER_BACKBONE != "unet":
        policy.diffusion = BackboneOverrideDiffusionModel(dp_cfg).to(DEVICE)
    policy.train()
    log.info("Policy trainable params: %s", f"{sum(p.numel() for p in policy.parameters() if p.requires_grad):,}")
    log.info("Policy inputs : %s", list(dp_cfg.input_features.keys()))
    log.info("Policy outputs: %s", list(dp_cfg.output_features.keys()))

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=dp_cfg,
        dataset_stats=dataset.meta.stats,
    )

    optimizer_cfg = AdamWConfig(
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        grad_clip_norm=GRAD_CLIP_NORM if GRAD_CLIP_NORM is not None else float("inf"),
    )
    all_params = list(policy.get_optim_params())
    if traj_encoder is not None:
        all_params += list(traj_encoder.parameters())
    if dino_encoder is not None:
        all_params += [p for p in dino_encoder.parameters() if p.requires_grad]
    resnet_params = []
    if resnet_encoder is not None:
        resnet_params = [p for p in resnet_encoder.parameters() if p.requires_grad]
        if RESNET_LR is None:
            all_params += resnet_params
    if depth_encoder is not None:
        all_params += list(depth_encoder.parameters())
    optimizer_params = (
        [{"params": all_params}, {"params": resnet_params, "lr": RESNET_LR}]
        if resnet_params and RESNET_LR is not None
        else all_params
    )
    optimizer = optimizer_cfg.build(optimizer_params)
    log.info("Optimizer: AdamW | lr=%s | resnet_lr=%s | weight_decay=%s", LR, RESNET_LR, WEIGHT_DECAY)

    lr_scheduler = None
    val_freq = None
    if LR_SCHEDULER_TYPE == "cosine":
        from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

        scheduler_cfg = CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=LR_WARMUP_STEPS,
            num_decay_steps=LR_DECAY_STEPS,
            peak_lr=LR_PEAK,
            decay_lr=LR_FLOOR,
        )
        lr_scheduler = scheduler_cfg.build(optimizer, num_training_steps=NUM_TRAIN_STEPS)
        log.info("LR scheduler: cosine | warmup=%s | decay=%s", f"{LR_WARMUP_STEPS:,}", f"{LR_DECAY_STEPS:,}")
    elif LR_SCHEDULER_TYPE == "plateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=PLATEAU_FACTOR,
            patience=PLATEAU_PATIENCE,
            min_lr=PLATEAU_MIN_LR,
        )
        val_freq = VAL_FREQ_STEPS if VAL_FREQ_STEPS is not None else steps_per_epoch
        log.info("LR scheduler: plateau | val every %s steps", f"{val_freq:,}")
    else:
        log.info("LR scheduler: disabled")

    encoder_config = encoder_config_dict(max_T, dino_cam_keys, depth_cam_keys, resnet_cam_keys)
    save_json(OUTPUT_DIR / "encoder_config.json", encoder_config)
    start_step = 0
    if args.resume_from is not None:
        start_step = resume_training_state(
            args.resume_from,
            policy,
            traj_encoder,
            dino_encoder,
            resnet_encoder,
            depth_encoder,
            optimizer,
            lr_scheduler,
        )
        if start_step >= NUM_TRAIN_STEPS:
            raise ValueError(
                f"Resume step {start_step:,} is already >= NUM_TRAIN_STEPS={NUM_TRAIN_STEPS:,}."
            )
        log.info("Resuming training from step %s.", f"{start_step:,}")

    def prepare_batch(
        batch: dict[str, Any],
        train: bool,
        shuffle_trajectory: bool = False,
        zero_vision: bool = False,
        zero_depth: bool = False,
        zero_state: bool = False,
    ) -> dict[str, Any]:
        dropout_draw = None
        if train and (
            VISION_DROPOUT_PROB > 0
            or DEPTH_DROPOUT_PROB > 0
            or STATE_DROPOUT_PROB > 0
        ):
            batch_size = batch["observation.state"].shape[0]
            dropout_draw = torch.rand(batch_size, device=DEVICE)

        dino_imgs_list = []
        if dino_encoder is not None:
            for dino_key in dino_cam_keys:
                if dino_key in batch:
                    imgs = batch.pop(dino_key).to(DEVICE)
                    if imgs.ndim == 4:
                        imgs = imgs.unsqueeze(1)
                    dino_imgs_list.append(imgs)
            if train and dino_aug is not None and dino_imgs_list:
                obs_steps_per_camera = [imgs.shape[1] for imgs in dino_imgs_list]
                if len({tuple(imgs.shape[2:]) for imgs in dino_imgs_list}) != 1:
                    raise ValueError(
                        "Temporally and cross-view consistent DINO augmentation requires "
                        "all camera tensors to have the same [C, H, W] shape."
                    )
                all_views = torch.cat(dino_imgs_list, dim=1)
                all_views = dino_aug(all_views)
                dino_imgs_list = list(
                    torch.split(all_views, obs_steps_per_camera, dim=1)
                )

        resnet_imgs_list = []
        if resnet_encoder is not None:
            for resnet_key in resnet_cam_keys:
                if resnet_key in batch:
                    imgs = batch.pop(resnet_key).to(DEVICE)
                    if imgs.ndim == 4:
                        imgs = imgs.unsqueeze(1)
                    resnet_imgs_list.append(imgs)

        depth_imgs_list = []
        if depth_encoder is not None:
            for depth_key in depth_cam_keys:
                if depth_key in batch:
                    imgs = batch.pop(depth_key).to(DEVICE)
                    if imgs.ndim == 4:
                        imgs = imgs.unsqueeze(1)
                    depth_imgs_list.append(imgs)

        raw_traj = None
        if USE_GLOBAL_TRAJ and traj_key in batch:
            raw_traj = batch[traj_key].to(DEVICE)
            if raw_traj.ndim == 3:
                raw_traj = raw_traj.unsqueeze(1)
            # The path specifies the episode-level task, so retain only one
            # copy instead of encoding it independently for each observation.
            raw_traj = raw_traj[:, -1:, ...]
            raw_traj, source_lengths = resample_global_trajectory(
                raw_traj, GLOBAL_TRAJ_LENGTH
            )
            if shuffle_trajectory and raw_traj.shape[0] > 1:
                raw_traj = raw_traj.roll(shifts=1, dims=0)
            if train:
                mask_indices = None
                if "frame_index" in batch:
                    frame_index = batch["frame_index"].to(DEVICE)
                    if frame_index.ndim > 1:
                        frame_index = frame_index[:, -1:]
                    mask_indices = rescale_frame_indices(
                        frame_index,
                        source_lengths,
                        GLOBAL_TRAJ_LENGTH,
                    )
                batch[traj_key] = raw_traj
                batch = traj_aug(batch, mask_indices=mask_indices)
                raw_traj = batch[traj_key]
            if traj_encoder is not None:
                batch.pop(traj_key)
            else:
                batch[traj_key] = raw_traj

        batch = flatten_transform(batch)

        shared_traj = None
        if traj_encoder is not None:
            if raw_traj is None:
                raise KeyError(f"{traj_key!r} missing from batch; cannot run trajectory encoder.")
            if USE_CONDITIONING_FUSION and TRAJ_ENC_TYPE == "transformer":
                shared_traj = traj_encoder(raw_traj, return_tokens=True).squeeze(1)
            else:
                shared_traj = traj_encoder(raw_traj).squeeze(1)
                if USE_CONDITIONING_FUSION:
                    shared_traj = shared_traj.unsqueeze(1)

        if dino_encoder is not None and dino_imgs_list:
            dino_feat = torch.cat([dino_encoder(imgs) for imgs in dino_imgs_list], dim=-1)
            if zero_vision:
                dino_feat = torch.zeros_like(dino_feat)
            elif dropout_draw is not None and VISION_DROPOUT_PROB > 0:
                drop_vision = dropout_draw < VISION_DROPOUT_PROB
                dino_feat = dino_feat.masked_fill(drop_vision[:, None, None], 0.0)
            env = batch.get("observation.environment_state")
            batch["observation.environment_state"] = (
                torch.cat([env.to(DEVICE), dino_feat], dim=-1) if env is not None else dino_feat
            )

        if resnet_encoder is not None and resnet_imgs_list:
            resnet_feat = torch.cat([resnet_encoder(imgs) for imgs in resnet_imgs_list], dim=-1)
            if zero_vision:
                resnet_feat = torch.zeros_like(resnet_feat)
            elif dropout_draw is not None and VISION_DROPOUT_PROB > 0:
                drop_vision = dropout_draw < VISION_DROPOUT_PROB
                resnet_feat = resnet_feat.masked_fill(drop_vision[:, None, None], 0.0)
            env = batch.get("observation.environment_state")
            batch["observation.environment_state"] = (
                torch.cat([env.to(DEVICE), resnet_feat], dim=-1) if env is not None else resnet_feat
            )

        if depth_encoder is not None and depth_imgs_list:
            depth_feat = torch.cat(
                [depth_encoder(imgs) for imgs in depth_imgs_list],
                dim=-1,
            )
            if zero_depth:
                depth_feat = torch.zeros_like(depth_feat)
            elif dropout_draw is not None and DEPTH_DROPOUT_PROB > 0:
                drop_depth = (
                    (dropout_draw >= VISION_DROPOUT_PROB)
                    & (
                        dropout_draw
                        < VISION_DROPOUT_PROB + DEPTH_DROPOUT_PROB
                    )
                )
                depth_feat = depth_feat.masked_fill(drop_depth[:, None, None], 0.0)
            env = batch.get("observation.environment_state")
            batch["observation.environment_state"] = (
                torch.cat([env.to(DEVICE), depth_feat], dim=-1)
                if env is not None
                else depth_feat
            )

        processed = preprocessor(batch)
        if shared_traj is not None:
            processed[shared_traj_key] = shared_traj
        state = processed.get("observation.state")
        if state is not None:
            if zero_state:
                state = torch.zeros_like(state)
            elif train:
                if STATE_NOISE_STD > 0:
                    state = state + torch.randn_like(state) * STATE_NOISE_STD
                if dropout_draw is not None and STATE_DROPOUT_PROB > 0:
                    drop_state = (
                        (dropout_draw >= VISION_DROPOUT_PROB + DEPTH_DROPOUT_PROB)
                        & (
                            dropout_draw
                            < VISION_DROPOUT_PROB
                            + DEPTH_DROPOUT_PROB
                            + STATE_DROPOUT_PROB
                        )
                    )
                    state_mask = drop_state.view(
                        drop_state.shape[0], *([1] * (state.ndim - 1))
                    )
                    state = state.masked_fill(state_mask, 0.0)
            processed["observation.state"] = state

        return processed

    def clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    fixed_diagnostic_batch = clone_batch(next(iter(dataloader)))

    log.info("Running one-batch sanity check...")
    with torch.no_grad():
        sanity_batch = prepare_batch(clone_batch(fixed_diagnostic_batch), train=False)
        sanity_loss, _ = policy.forward(sanity_batch)
    log.info("Sanity forward loss: %.6f", sanity_loss.item())
    if not torch.isfinite(sanity_loss):
        raise RuntimeError("Sanity forward produced non-finite loss.")

    def conditioning_diagnostics(
        step: int,
        metrics_tracker: MetricsTracker | None = None,
    ) -> None:
        """Measure whether the policy loss reacts to each conditioning source.

        This is deliberately not a generalization estimate: all forwards use
        the same fixed training examples and action targets. The trajectory,
        vision, depth, and state conditions are changed one at a time, while
        the diffusion RNG is reset so every forward receives identical noise
        and timesteps.
        """
        was_training = policy.training
        policy.eval()
        if traj_encoder is not None:
            traj_encoder.eval()
        if dino_encoder is not None:
            dino_encoder.eval()
        if resnet_encoder is not None:
            resnet_encoder.eval()
        if depth_encoder is not None:
            depth_encoder.eval()

        def evaluate(**prepare_kwargs) -> float:
            torch.manual_seed(CONDITIONING_DIAG_SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(CONDITIONING_DIAG_SEED)
            prepared = prepare_batch(
                clone_batch(fixed_diagnostic_batch),
                train=False,
                **prepare_kwargs,
            )
            value, _ = policy.forward(prepared)
            return float(value.item())

        with torch.no_grad():
            correct = evaluate()
            shuffled = evaluate(shuffle_trajectory=True)
            no_vision = evaluate(zero_vision=True)
            no_depth = evaluate(zero_depth=True)
            no_state = evaluate(zero_state=True)

        if metrics_tracker is not None:
            metrics_tracker.log_conditioning(
                step,
                correct,
                shuffled,
                no_vision,
                no_depth,
                no_state,
            )
        log.info(
            "[step %s] conditioning diagnostic | correct=%.5f | "
            "shuffled_traj=%.5f (delta=%+.5f) | zero_vision=%.5f (delta=%+.5f) | "
            "zero_depth=%.5f (delta=%+.5f) | zero_state=%.5f (delta=%+.5f)",
            f"{step:,}",
            correct,
            shuffled,
            shuffled - correct,
            no_vision,
            no_vision - correct,
            no_depth,
            no_depth - correct,
            no_state,
            no_state - correct,
        )

        if was_training:
            policy.train()
            if traj_encoder is not None:
                traj_encoder.train()
            if dino_encoder is not None:
                dino_encoder.train()
            if resnet_encoder is not None:
                resnet_encoder.train()
            if depth_encoder is not None:
                depth_encoder.train()

    if args.dry_run:
        conditioning_diagnostics(0)
        log.info("--dry-run requested; exiting after sanity check.")
        return

    metrics = MetricsTracker(OUTPUT_DIR)
    interrupt_flag = {"requested": False}

    def signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        log.warning("Received %s; will checkpoint and exit after current step.", sig_name)
        interrupt_flag["requested"] = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    log.info("Starting training for %s steps...", f"{NUM_TRAIN_STEPS:,}")
    dl_iter = cycle(dataloader)
    total_loss = 0.0
    log_count = 0
    next_epoch_checkpoint = SAVE_EVERY_N_EPOCHS
    if next_epoch_checkpoint is not None:
        resume_epoch = start_step / steps_per_epoch
        while next_epoch_checkpoint <= resume_epoch + 1e-12:
            next_epoch_checkpoint += SAVE_EVERY_N_EPOCHS
    pbar = tqdm(range(start_step, NUM_TRAIN_STEPS), desc="Training", unit="step", dynamic_ncols=True)

    for step in pbar:
        batch = prepare_batch(next(dl_iter), train=True)
        loss, _ = policy.forward(batch)

        if not torch.isfinite(loss):
            msg = f"[step {step + 1:,}] WARNING: non-finite loss ({loss.item():.4g}), skipping"
            tqdm.write(msg)
            log.warning(msg)
            optimizer.zero_grad()
            continue

        loss.backward()
        if GRAD_CLIP_NORM is not None:
            clip_params = list(policy.parameters())
            if traj_encoder is not None:
                clip_params += list(traj_encoder.parameters())
            if dino_encoder is not None:
                clip_params += [p for p in dino_encoder.parameters() if p.requires_grad]
            if resnet_encoder is not None:
                clip_params += [p for p in resnet_encoder.parameters() if p.requires_grad]
            if depth_encoder is not None:
                clip_params += list(depth_encoder.parameters())
            torch.nn.utils.clip_grad_norm_(clip_params, GRAD_CLIP_NORM)
        optimizer.step()
        optimizer.zero_grad()

        if lr_scheduler is not None and LR_SCHEDULER_TYPE == "cosine":
            lr_scheduler.step()

        if (
            LR_SCHEDULER_TYPE == "plateau"
            and lr_scheduler is not None
            and val_loader is not None
            and val_freq is not None
            and (step + 1) % val_freq == 0
        ):
            policy.eval()
            if traj_encoder is not None:
                traj_encoder.eval()
            if dino_encoder is not None:
                dino_encoder.eval()
            if resnet_encoder is not None:
                resnet_encoder.eval()
            if depth_encoder is not None:
                depth_encoder.eval()
            val_total = 0.0
            val_n = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    n = val_batch[next(iter(val_batch))].shape[0]
                    val_batch = prepare_batch(val_batch, train=False)
                    v_loss, _ = policy.forward(val_batch)
                    val_total += v_loss.item() * n
                    val_n += n
            mean_val = val_total / val_n
            lr_scheduler.step(mean_val)
            metrics.log_val(step + 1, mean_val)
            msg = f"[step {step + 1:,}] val_loss={mean_val:.4f} lr={optimizer.param_groups[0]['lr']:.2e}"
            tqdm.write(msg)
            log.info(msg)
            policy.train()
            if traj_encoder is not None:
                traj_encoder.train()
            if dino_encoder is not None:
                dino_encoder.train()
            if resnet_encoder is not None:
                resnet_encoder.train()
            if depth_encoder is not None:
                depth_encoder.train()

        total_loss += loss.item()
        log_count += 1
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_frac = (step + 1) / steps_per_epoch
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}", "epoch": f"{epoch_frac:.1f}"})

        if (step + 1) % LOG_FREQ == 0 and log_count > 0:
            avg_loss = total_loss / log_count
            msg = (
                f"step={step + 1:>7,} | epoch~{epoch_frac:>6.1f} | "
                f"avg_loss={avg_loss:.4f} | lr={current_lr:.2e}"
            )
            tqdm.write(msg)
            log.info(msg)
            metrics.log_train(step + 1, avg_loss, current_lr)
            total_loss = 0.0
            log_count = 0

        if (step + 1) % PLOT_FREQ == 0:
            metrics.save_plots(step + 1)

        if CONDITIONING_DIAG_FREQ > 0 and (step + 1) % CONDITIONING_DIAG_FREQ == 0:
            conditioning_diagnostics(step + 1, metrics)

        step_checkpoint = SAVE_FREQ is not None and (step + 1) % SAVE_FREQ == 0
        epoch_checkpoint = False
        epoch_label = None
        if next_epoch_checkpoint is not None and epoch_frac + 1e-12 >= next_epoch_checkpoint:
            epoch_checkpoint = True
            epoch_label = next_epoch_checkpoint
            while next_epoch_checkpoint <= epoch_frac + 1e-12:
                next_epoch_checkpoint += SAVE_EVERY_N_EPOCHS

        save_now = step_checkpoint or epoch_checkpoint or interrupt_flag["requested"]
        if save_now:
            ckpt_dir = OUTPUT_DIR / f"checkpoint_{step + 1:07d}"
            save_training_state(
                ckpt_dir, policy, preprocessor, postprocessor,
                traj_encoder, dino_encoder, resnet_encoder, depth_encoder, optimizer, lr_scheduler, encoder_config,
                step + 1,
            )
            reasons = []
            if step_checkpoint:
                reasons.append("step")
            if epoch_checkpoint:
                reasons.append(f"epoch~{epoch_label:.3g}")
            if interrupt_flag["requested"]:
                reasons.append("interrupt")
            msg = (
                f"[step {step + 1:,}] Checkpoint saved "
                f"({', '.join(reasons)}) -> {ckpt_dir}"
            )
            tqdm.write(msg)
            log.info(msg)

        if interrupt_flag["requested"]:
            log.info("Interrupt flag set; exiting training loop.")
            break

    pbar.close()
    final_step = step + 1
    metrics.save_plots(final_step)

    final_dir = OUTPUT_DIR / "final"
    save_training_state(
        final_dir, policy, preprocessor, postprocessor,
        traj_encoder, dino_encoder, resnet_encoder, depth_encoder, optimizer, lr_scheduler, encoder_config,
        final_step,
    )
    log.info("Training complete. Final model saved to %s", final_dir)


if __name__ == "__main__":
    main()
