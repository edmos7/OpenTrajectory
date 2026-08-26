"""Add Open Trajectory features with LeRobot's native dataset editor."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import EpisodeArtifactStore

GLOBAL_TRAJECTORY = "observation.global_trajectory"
RT_TRAJECTORY_IMAGE = "observation.images.rt_trajectory"


def _slug(value: str) -> str:
    value = value.removeprefix("observation.images.")
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_")


def _episode_indices(dataset: Any) -> list[int]:
    total = getattr(dataset.meta, "total_episodes", None)
    if total is not None:
        return list(range(int(total)))
    return sorted(int(index) for index in dataset.meta.episodes)


def build_feature_callbacks(
    store: EpisodeArtifactStore,
    *,
    features: tuple[str, ...],
    trajectory_length: int,
    camera_keys: tuple[str, ...],
) -> dict[str, tuple[Any, dict[str, Any]]]:
    """Build lazy callbacks accepted by LeRobot ``modify_features``."""
    additions: dict[str, tuple[Any, dict[str, Any]]] = {}

    if "tracks" in features and not camera_keys:
        raise ValueError("camera_keys is required when features includes 'tracks'")

    if "sequence" in features:

        def sequence(_row: dict, episode_index: int, _frame_index: int) -> np.ndarray:
            return store.get(episode_index).condition.to_sequence(trajectory_length)

        additions[GLOBAL_TRAJECTORY] = (
            sequence,
            {"dtype": "float32", "shape": [trajectory_length, 3], "names": None},
        )

    if "rt-image" in features:
        height, width, channels = store.rt_image_shape

        def rt_image(_row: dict, episode_index: int, _frame_index: int) -> np.ndarray:
            artifact = store.get(episode_index)
            if artifact.image_size != (height, width):
                raise ValueError(
                    f"Episode {episode_index} image_size {artifact.image_size} does not match "
                    f"declared RT-image shape {(height, width)}"
                )
            return artifact.condition.to_rt_image()

        additions[RT_TRAJECTORY_IMAGE] = (
            rt_image,
            {
                "dtype": "image",
                "shape": [height, width, channels],
                "names": ["height", "width", "channels"],
            },
        )

    if "tracks" in features:
        for camera_key in camera_keys:
            camera = _slug(camera_key)
            shape = list(store.point_track_shape(camera))

            def tracks(_row: dict, episode_index: int, frame_index: int, *, name: str = camera) -> np.ndarray:
                if frame_index < 0:
                    raise ValueError("frame_index must be non-negative")
                artifact = store.get(episode_index)
                values = artifact.point_tracks[name]
                track_index = min(int(frame_index) // artifact.frame_stride, len(values) - 1)
                return values[track_index]

            additions[f"observation.point_tracks.{camera}"] = (
                tracks,
                {"dtype": "float32", "shape": shape, "names": ["query", "horizon", "xy"]},
            )

    return additions


def augment_dataset(
    *,
    repo_id: str,
    artifact_dir: str | Path,
    output_repo_id: str,
    output_dir: str | Path | None = None,
    root: str | Path | None = None,
    revision: str | None = None,
    features: tuple[str, ...] = ("sequence", "rt-image"),
    trajectory_length: int = 224,
    camera_keys: tuple[str, ...] = (),
    drop_legacy_task_id: bool = True,
):
    """Create an augmented derivative without mutating the source dataset."""
    from lerobot.datasets.dataset_tools import modify_features
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    unknown = set(features) - {"sequence", "rt-image", "tracks"}
    if unknown:
        raise ValueError(f"Unknown features: {sorted(unknown)}")
    if not features:
        raise ValueError("features must contain at least one augmentation")
    dataset = LeRobotDataset(repo_id=repo_id, root=root, revision=revision)
    store = EpisodeArtifactStore(artifact_dir)
    store.validate_complete(_episode_indices(dataset))
    additions = build_feature_callbacks(
        store,
        features=features,
        trajectory_length=trajectory_length,
        camera_keys=camera_keys,
    )
    collisions = sorted(set(additions) & set(dataset.meta.features))
    if collisions:
        raise ValueError(f"Output features already exist: {collisions}")
    remove = ["task_id"] if drop_legacy_task_id and "task_id" in dataset.meta.features else None
    return modify_features(
        dataset,
        add_features=additions,
        remove_features=remove,
        output_dir=output_dir,
        repo_id=output_repo_id,
    )
