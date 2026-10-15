"""Supervised track prediction from LeRobot episodes (not an action-policy trainer)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from ..data.schema import SpatialMetadata, read_dataset_metadata
from .predictor import TrackPredictor, TrackPredictorConfig


def split_episodes(episode_ids, validation_fraction=0.2, seed=0):
    ids = sorted(set(int(i) for i in episode_ids))
    if len(ids) < 2 or not 0 < validation_fraction < 1:
        raise ValueError("Need at least two episodes and a validation fraction in (0, 1)")
    random.Random(seed).shuffle(ids)
    count = min(len(ids) - 1, max(1, round(len(ids) * validation_fraction)))
    return sorted(ids[count:]), sorted(ids[:count])


class TrackTrainingDataset(Dataset):
    """Separate current-only model inputs from future supervision at the boundary."""

    def __init__(
        self,
        dataset,
        camera_tracks,
        episode_ids,
        *,
        image_size=224,
        trajectory_key="observation.global_trajectory",
        validity_keys=None,
    ):
        self.dataset, self.camera_tracks = dataset, camera_tracks
        self.image_size, self.trajectory_key = image_size, trajectory_key
        self.validity_keys = validity_keys or {}
        if not set(self.validity_keys) <= set(camera_tracks):
            raise ValueError("Validity mapping contains an unknown camera")
        self.indices = [
            i for i, ep in enumerate(dataset.hf_dataset["episode_index"]) if int(ep) in set(episode_ids)
        ]
        if not self.indices:
            raise ValueError("No frames selected")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.dataset[self.indices[index]]
        images, queries, targets, masks = {}, {}, {}, {}
        for camera, track_key in self.camera_tracks.items():
            image = row[camera].float()
            images[camera] = F.interpolate(
                image.unsqueeze(0), (self.image_size,) * 2, mode="bilinear", align_corners=False
            )[0]
            tracks = row[track_key].float()
            if tracks.ndim != 3 or tracks.shape[1] < 2 or tracks.shape[-1] != 2:
                raise ValueError(f"{track_key} must be [N, current+future, XY]")
            queries[camera], targets[camera] = tracks[:, 0].clone(), tracks[:, 1:].clone()
            masks[camera] = (
                row[self.validity_keys[camera]][:, 1:].bool()
                if camera in self.validity_keys
                else torch.ones_like(targets[camera][..., 0], dtype=torch.bool)
            )
        return images, queries, row[self.trajectory_key].float(), targets, masks


def _device(value, device):
    if isinstance(value, dict):
        return {key: _device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_device(item, device) for item in value)
    return value.to(device)


def main(argv=None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    parser = argparse.ArgumentParser(description="Train a future-track predictor with held-out episodes")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root")
    parser.add_argument(
        "--camera-tracks",
        required=True,
        type=json.loads,
        help="JSON mapping from source camera key to point-track feature key",
    )
    parser.add_argument("--validity-keys", type=json.loads, default={})
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectory-key", default="observation.global_trajectory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--trajectory-length", type=int, default=224)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if min(args.epochs, args.batch_size) < 1 or args.lr <= 0:
        raise ValueError("Epochs, batch size and learning rate must be positive")
    torch.manual_seed(args.seed)
    dataset = LeRobotDataset(args.repo_id, root=args.root)
    metadata = read_dataset_metadata(dataset.root)
    horizon, stride = None, None
    for camera, key in args.camera_tracks.items():
        descriptor = metadata["features"][key]
        spatial = SpatialMetadata.from_dict(descriptor["spatial"])
        if (
            spatial.camera_key != camera
            or spatial.coordinate_frame != "image_plane"
            or spatial.normalization != "image_dimensions"
        ):
            raise ValueError(f"{key}: predictor requires camera-matched normalized image-plane tracks")
        if not descriptor.get("horizon_includes_current"):
            raise ValueError(f"{key}: track horizon must explicitly include the current position")
        if descriptor.get("anchor_stride", 1) != 1:
            raise ValueError(
                f"{key}: resample episodes to tracker FPS before training on strided query anchors"
            )
        current_horizon = dataset.features[key]["shape"][1] - 1
        if horizon is not None and (horizon != current_horizon or stride != spatial.horizon_stride):
            raise ValueError("All cameras must use the same future horizon and stride")
        horizon, stride = current_horizon, spatial.horizon_stride
    train_ids, val_ids = split_episodes(
        dataset.hf_dataset["episode_index"], args.validation_fraction, args.seed
    )
    cfg = TrackPredictorConfig(
        camera_keys=tuple(args.camera_tracks),
        horizon=horizon,
        horizon_stride=stride,
        image_size=args.image_size,
        trajectory_length=args.trajectory_length,
    )
    model = TrackPredictor(cfg).to(args.device)
    loaders = [
        DataLoader(
            TrackTrainingDataset(
                dataset,
                args.camera_tracks,
                ids,
                image_size=args.image_size,
                trajectory_key=args.trajectory_key,
                validity_keys=args.validity_keys,
            ),
            batch_size=args.batch_size,
            shuffle=i == 0,
            num_workers=0,
        )
        for i, ids in enumerate((train_ids, val_ids))
    ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "training.json").write_text(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "seed": args.seed,
                "train_episodes": train_ids,
                "validation_episodes": val_ids,
                "camera_tracks": args.camera_tracks,
                "validity_keys": args.validity_keys,
                "trajectory_key": args.trajectory_key,
                "spatial_metadata": metadata,
                "target_padding": "included unless excluded by supplied validity masks",
            },
            indent=2,
        )
        + "\n"
    )
    history = []
    for epoch in range(args.epochs):
        metrics = {"epoch": epoch + 1}
        for training, loader, label in (
            (True, loaders[0], "train_mse"),
            (False, loaders[1], "validation_mse"),
        ):
            model.train(training)
            total, positions = 0.0, 0
            with torch.set_grad_enabled(training):
                for batch in loader:
                    batch = _device(batch, args.device)
                    loss = model.loss(*batch)
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                    count = sum(int(mask.sum()) for mask in batch[4].values())
                    total += float(loss.detach()) * count
                    positions += count
            metrics[label] = total / positions
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        model.save_pretrained(args.output_dir)
        (args.output_dir / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")
