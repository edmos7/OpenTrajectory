"""Camera-keyed predictor API with explicit inputs and portable checkpoints."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..data.schema import SpatialMetadata


@dataclass(frozen=True)
class TrackPredictorConfig:
    camera_keys: tuple[str, ...] = ("observation.images.side",)
    horizon: int = 32  # future positions; current position is added by predict()
    horizon_stride: int = 1
    image_size: int = 224
    trajectory_length: int = 224
    backbone_name: str = "efficientnet_b0"
    dim_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    pretrained: bool = False

    def __post_init__(self):
        object.__setattr__(self, "camera_keys", tuple(self.camera_keys))
        if not 1 <= len(self.camera_keys) <= 2 or len(set(self.camera_keys)) != len(self.camera_keys):
            raise ValueError("Track Transformer requires one or two distinct camera keys")
        if min(self.horizon, self.horizon_stride, self.trajectory_length, self.n_layers, self.n_heads) < 1:
            raise ValueError("Horizon, strides, lengths, and transformer dimensions must be positive")
        if self.image_size < 32 or self.dim_model < 1 or self.dim_model % self.n_heads:
            raise ValueError("image_size must be >=32 and dim_model divisible by n_heads")


@dataclass(frozen=True)
class TrackPrediction:
    tracks: dict[str, Tensor]  # camera -> [B, N, current+future, XY]
    metadata: dict[str, SpatialMetadata]

    def policy_features(self, camera_to_feature: dict[str, str]) -> dict[str, Tensor]:
        """Map predictions into explicit policy input keys; no automatic camera guessing."""
        if set(camera_to_feature) != set(self.tracks):
            raise ValueError("Feature mapping must cover exactly the predicted cameras")
        if len(set(camera_to_feature.values())) != len(camera_to_feature):
            raise ValueError("Predicted cameras cannot share a policy feature key")
        return {camera_to_feature[camera]: tracks for camera, tracks in self.tracks.items()}


class TrackPredictor(nn.Module):
    """Predict from current RGB, current query points, and a supplied task sketch.

    Future observed frames/tracks are deliberately absent from this interface.
    Coordinates are image-plane XY divided by original width/height. Outputs may
    leave [0, 1] as points move outside the image; they are not metric 3-D paths.
    """

    def __init__(self, config: TrackPredictorConfig):
        super().__init__()
        from .track_transformer import TrackTransformer

        self.config = config
        self.model = TrackTransformer(
            n_points=0,
            horizon=config.horizon,
            backbone_name=config.backbone_name,
            d_model=config.dim_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            traj_length=config.trajectory_length,
            img_size=config.image_size,
            pretrained=config.pretrained,
            traj_action_encoding="binary",
            traj_action_binary_threshold=0.5,
        )

    def forward(
        self, images: dict[str, Tensor], query_points: dict[str, Tensor], trajectory: Tensor
    ) -> dict[str, Tensor]:
        keys = self.config.camera_keys
        if set(images) != set(keys) or set(query_points) != set(keys):
            raise ValueError("Images and query points must match configured camera keys exactly")
        if trajectory.ndim != 3 or trajectory.shape[-1] != 3 or trajectory.shape[1] < 2:
            raise ValueError("Trajectory must be [B, L>=2, 3] normalized XY + active")
        if not torch.isfinite(trajectory).all():
            raise ValueError("Trajectory contains non-finite values")
        frames = []
        for key in keys:
            frame, points = images[key], query_points[key]
            if frame.ndim != 4 or frame.shape[:2] != (trajectory.shape[0], 3):
                raise ValueError(f"{key}: images must be [B, 3, H, W]")
            if (
                points.ndim != 3
                or points.shape[0] != trajectory.shape[0]
                or points.shape[1] < 1
                or points.shape[-1] != 2
            ):
                raise ValueError(f"{key}: query points must be [B, N>=1, 2]")
            if not torch.isfinite(points).all() or not torch.isfinite(frame).all():
                raise ValueError(f"{key}: inputs must be finite")
            if (points < 0).any() or (points > 1).any() or (frame < 0).any() or (frame > 1).any():
                raise ValueError(f"{key}: RGB and current query coordinates must be normalized to [0, 1]")
            frames.append(
                F.interpolate(frame, (self.config.image_size,) * 2, mode="bilinear", align_corners=False)
            )
        secondary = {} if len(keys) == 1 else {"sec_frame": frames[1], "sec_cur_pos": query_points[keys[1]]}
        offsets = self.model(frames[0], trajectory, query_points[keys[0]], **secondary)
        offsets = (offsets,) if isinstance(offsets, Tensor) else offsets
        return {
            key: query_points[key].unsqueeze(2) + offset for key, offset in zip(keys, offsets, strict=True)
        }

    def loss(self, images, query_points, trajectory, targets, validity=None):
        """Masked MSE against future-only absolute tracks; targets never enter forward."""
        predictions = self(images, query_points, trajectory)
        if set(targets) != set(predictions) or (validity is not None and set(validity) != set(predictions)):
            raise ValueError("Targets and validity must match configured cameras")
        numerator, denominator = trajectory.new_zeros(()), trajectory.new_zeros(())
        for key, prediction in predictions.items():
            target = targets[key]
            if target.shape != prediction.shape:
                raise ValueError(f"{key}: targets must be future-only [B, N, horizon, 2]")
            mask = torch.ones_like(target[..., 0], dtype=torch.bool) if validity is None else validity[key]
            if mask.dtype != torch.bool or mask.shape != target.shape[:-1]:
                raise ValueError(f"{key}: validity must be boolean [B, N, horizon]")
            if not torch.isfinite(target[mask]).all():
                raise ValueError("Valid targets must be finite")
            error = prediction[mask] - target[mask]
            numerator = numerator + error.square().sum()
            denominator = denominator + mask.sum() * 2
        if denominator == 0:
            raise ValueError("No valid future positions in batch")
        return numerator / denominator

    @torch.no_grad()
    def predict(self, images, query_points, trajectory, *, fps: float | None = None) -> TrackPrediction:
        self.eval()
        futures = self(images, query_points, trajectory)
        tracks = {
            key: torch.cat([query_points[key].unsqueeze(2), future], dim=2) for key, future in futures.items()
        }
        metadata = {
            key: SpatialMetadata(
                camera_key=key,
                image_size=tuple(images[key].shape[-2:]),
                fps=fps,
                horizon_stride=self.config.horizon_stride,
                source="predicted",
                generator="track_transformer",
                padding="none",
            )
            for key in futures
        }
        return TrackPrediction(tracks, metadata)

    def save_pretrained(self, directory):
        from safetensors.torch import save_file

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "predictor_config.json").write_text(json.dumps(asdict(self.config), indent=2) + "\n")
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()},
            directory / "model.safetensors",
        )

    @classmethod
    def from_pretrained(cls, directory, *, device="cpu"):
        from safetensors.torch import load_file

        directory = Path(directory)
        values = json.loads((directory / "predictor_config.json").read_text())
        values["pretrained"] = False  # checkpoint loading must not download backbone weights
        model = cls(TrackPredictorConfig(**values))
        model.load_state_dict(load_file(directory / "model.safetensors"), strict=True)
        return model.to(device).eval()
