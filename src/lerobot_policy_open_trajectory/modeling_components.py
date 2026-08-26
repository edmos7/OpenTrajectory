"""Trajectory-specific encoders and modality fusion."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _ensure_temporal_images(images: torch.Tensor, name: str) -> torch.Tensor:
    if images.ndim == 4:
        return images.unsqueeze(1)
    if images.ndim != 5:
        raise ValueError(f"{name} expected [B,T,C,H,W] or [B,C,H,W], got {tuple(images.shape)}")
    return images


class TrajectoryEncoder(nn.Module):
    """Capacity-matched Conv1D encoder for `[B,Tobs,L,3]` trajectories."""

    def __init__(
        self,
        length: int = 224,
        temporal_dim: int = 16,
        out_dim: int = 96,
        hidden_dim: int = 128,
        layers: int = 4,
        kernel_sizes: tuple[int, ...] = (7, 5, 3, 3),
        dilations: tuple[int, ...] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        if len(kernel_sizes) != layers or len(dilations) != layers:
            raise ValueError("Trajectory Conv1D kernel and dilation counts must equal layers")
        self.length = length
        self.temporal_embedding = nn.Embedding(length, temporal_dim)
        modules: list[nn.Module] = []
        in_channels = 2 + temporal_dim
        for index in range(layers):
            out_channels = hidden_dim if index < layers - 1 else out_dim
            kernel = kernel_sizes[index]
            dilation = dilations[index]
            modules.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel,
                        padding=dilation * (kernel // 2),
                        dilation=dilation,
                    ),
                    nn.GroupNorm(min(8, out_channels), out_channels),
                    nn.Mish(),
                ]
            )
            in_channels = out_channels
        self.backbone = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = out_dim

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        if trajectory.ndim == 3:
            trajectory = trajectory.unsqueeze(1)
        if trajectory.ndim != 4 or trajectory.shape[-1] != 3:
            raise ValueError(f"Expected trajectory [B,Tobs,L,3], got {tuple(trajectory.shape)}")
        batch, obs_steps, source_length, _ = trajectory.shape
        value = trajectory.reshape(batch * obs_steps, source_length, 3).float()
        if source_length != self.length:
            value = F.interpolate(
                value.transpose(1, 2), size=self.length, mode="linear", align_corners=True
            ).transpose(1, 2)
        time = self.temporal_embedding(torch.arange(self.length, device=value.device))
        time = time.unsqueeze(0).expand(value.shape[0], -1, -1).clone()
        time[..., 0] = time[..., 0] + value[..., 2]
        encoded = self.backbone(torch.cat((value[..., :2], time), dim=-1).transpose(1, 2))
        return self.pool(encoded).squeeze(-1).reshape(batch, obs_steps, self.out_dim)


class ResNetEnvironmentEncoder(nn.Module):
    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        backbone: str,
        out_dim: int,
        pretrained_weights: str | None,
        frozen: bool,
        resize: tuple[int, int],
    ) -> None:
        super().__init__()
        from torchvision import models

        constructor = getattr(models, backbone)
        weights = None
        if pretrained_weights:
            enum_name, value_name = pretrained_weights.split(".", 1)
            weights = getattr(getattr(models, enum_name), value_name)
        model = constructor(weights=weights)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        self.backbone = model
        self.projection = nn.Linear(feature_dim, out_dim)
        self.out_dim = out_dim
        self.resize = resize
        if frozen:
            self.backbone.requires_grad_(False)
        self.register_buffer("_mean", torch.tensor(self._MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(self._STD).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = _ensure_temporal_images(images, self.__class__.__name__)
        batch, obs_steps = images.shape[:2]
        value = images.flatten(0, 1).float()
        if value.shape[-2:] != self.resize:
            value = F.interpolate(value, size=self.resize, mode="bilinear", align_corners=False)
        value = (value - self._mean) / self._std
        return self.projection(self.backbone(value)).reshape(batch, obs_steps, self.out_dim)


class DinoEnvironmentEncoder(nn.Module):
    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)

    def __init__(self, model_name: str, out_dim: int, frozen: bool, resize: tuple[int, int]) -> None:
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        self.projection = nn.Linear(self.backbone.embed_dim, out_dim)
        self.out_dim = out_dim
        self.resize = resize
        if frozen:
            self.backbone.requires_grad_(False)
        self.register_buffer("_mean", torch.tensor(self._MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(self._STD).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = _ensure_temporal_images(images, self.__class__.__name__)
        batch, obs_steps = images.shape[:2]
        value = images.flatten(0, 1).float()
        if value.shape[-2:] != self.resize:
            value = F.interpolate(value, size=self.resize, mode="bilinear", align_corners=False)
        value = (value - self._mean) / self._std
        return self.projection(self.backbone(value)).reshape(batch, obs_steps, self.out_dim)


class RTTrajectoryImageEncoder(nn.Module):
    """Dedicated 184,904-parameter CNN for rendered RT-Trajectory images."""

    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        out_dim: int = 96,
        channels: tuple[int, ...] = (24, 48, 96, 136),
        resize: tuple[int, int] = (224, 224),
    ) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        in_channels = 3
        for index, out_channels in enumerate(channels):
            kernel = 5 if index == 0 else 3
            modules.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel,
                        stride=2,
                        padding=kernel // 2,
                        bias=False,
                    ),
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                ]
            )
            in_channels = out_channels
        self.backbone = nn.Sequential(*modules, nn.AdaptiveAvgPool2d(1))
        self.projection = nn.Linear(in_channels, out_dim)
        self.out_dim = out_dim
        self.resize = resize
        self.register_buffer("_mean", torch.tensor(self._MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(self._STD).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = _ensure_temporal_images(images, self.__class__.__name__)
        batch, obs_steps = images.shape[:2]
        value = images.flatten(0, 1).float()
        if value.shape[-2:] != self.resize:
            value = F.interpolate(value, size=self.resize, mode="bilinear", align_corners=False)
        value = (value - self._mean) / self._std
        encoded = self.projection(self.backbone(value).flatten(1))
        return encoded.reshape(batch, obs_steps, self.out_dim)


class DepthEncoder(nn.Module):
    def __init__(self, out_dim: int, resize: tuple[int, int]) -> None:
        super().__init__()
        self.resize = resize
        self.out_dim = out_dim
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.Mish(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.Mish(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.Mish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = _ensure_temporal_images(images, self.__class__.__name__)
        batch, obs_steps = images.shape[:2]
        value = images.flatten(0, 1).float()
        if value.shape[-2:] != self.resize:
            value = F.interpolate(value, size=self.resize, mode="bilinear", align_corners=False)
        return self.network(value).reshape(batch, obs_steps, self.out_dim)


class ConditioningFusion(nn.Module):
    """Cross-attention fusion of temporal and trajectory modality tokens."""

    def __init__(
        self,
        *,
        state_dim: int,
        vision_dim: int,
        depth_dim: int,
        trajectory_dim: int,
        n_obs_steps: int,
        token_dim: int,
        out_dim: int,
        nhead: int,
        ffn_dim: int,
        layers: int,
    ) -> None:
        super().__init__()
        self.state_projection = nn.Linear(state_dim, token_dim) if state_dim else None
        self.vision_projection = nn.Linear(vision_dim, token_dim) if vision_dim else None
        self.depth_projection = nn.Linear(depth_dim, token_dim) if depth_dim else None
        self.trajectory_projection = nn.Linear(trajectory_dim, token_dim)
        self.output_token = nn.Parameter(torch.randn(1, out_dim) * 0.02)
        self.modality_embedding = nn.Parameter(torch.randn(4, token_dim) * 0.02)
        self.time_embedding = nn.Parameter(torch.randn(n_obs_steps, token_dim) * 0.02)
        self.query_norms = nn.ModuleList(nn.LayerNorm(out_dim) for _ in range(layers))
        self.context_norms = nn.ModuleList(nn.LayerNorm(token_dim) for _ in range(layers))
        self.attention = nn.ModuleList(
            nn.MultiheadAttention(out_dim, nhead, kdim=token_dim, vdim=token_dim, batch_first=True)
            for _ in range(layers)
        )
        self.ffn_norms = nn.ModuleList(nn.LayerNorm(out_dim) for _ in range(layers))
        self.ffns = nn.ModuleList(
            nn.Sequential(nn.Linear(out_dim, ffn_dim), nn.Mish(), nn.Linear(ffn_dim, out_dim))
            for _ in range(layers)
        )
        self.out_norm = nn.LayerNorm(out_dim)
        self.output_dim = out_dim
        self.n_obs_steps = n_obs_steps

    def _temporal(
        self, value: torch.Tensor | None, projection: nn.Module | None, index: int
    ) -> torch.Tensor | None:
        if value is None or projection is None:
            return None
        projected = projection(value)
        return projected + self.modality_embedding[index] + self.time_embedding[: projected.shape[1]]

    def forward(
        self,
        *,
        state: torch.Tensor | None,
        vision: torch.Tensor | None,
        depth: torch.Tensor | None,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        for name, value in (
            ("state", state),
            ("vision", vision),
            ("depth", depth),
            ("trajectory", trajectory),
        ):
            if value is None:
                continue
            if value.ndim != 3 or value.shape[1] != self.n_obs_steps:
                raise ValueError(
                    f"{name} conditioning must have n_obs_steps={self.n_obs_steps}, got {tuple(value.shape)}"
                )
        tokens = [
            item
            for item in (
                self._temporal(state, self.state_projection, 0),
                self._temporal(vision, self.vision_projection, 1),
                self._temporal(depth, self.depth_projection, 2),
                self.trajectory_projection(trajectory) + self.modality_embedding[3],
            )
            if item is not None
        ]
        context = torch.cat(tokens, dim=1)
        output = self.output_token.unsqueeze(0).expand(context.shape[0], -1, -1)
        for query_norm, context_norm, attention, ffn_norm, ffn in zip(
            self.query_norms, self.context_norms, self.attention, self.ffn_norms, self.ffns, strict=True
        ):
            normalized_context = context_norm(context)
            attended, _ = attention(
                query_norm(output), normalized_context, normalized_context, need_weights=False
            )
            output = output + attended
            output = output + ffn(ffn_norm(output))
        return self.out_norm(output).flatten(start_dim=1)
