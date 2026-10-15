"""Portable, validated intermediate artifacts for trajectory augmentation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..conditioning import TrajectoryCondition
from .schema import SCHEMA_VERSION, SpatialMetadata


@dataclass(frozen=True)
class EpisodeArtifact:
    """Tracking output for one episode, independent of LeRobot storage details."""

    condition: TrajectoryCondition | None
    image_size: tuple[int, int]
    point_tracks: dict[str, np.ndarray]
    frame_stride: int = 1
    spatial_metadata: dict[str, SpatialMetadata] = field(default_factory=dict)
    validity: dict[str, np.ndarray] = field(default_factory=dict)
    anchor_strides: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        image_size = tuple(int(value) for value in self.image_size)
        if self.condition is not None and image_size != self.condition.image_shape:
            raise ValueError("image_size must match condition.image_shape")
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be positive")
        if set(self.anchor_strides) - set(self.point_tracks):
            raise ValueError("Anchor strides supplied for unknown camera")
        if any(not isinstance(value, int) or value < 1 for value in self.anchor_strides.values()):
            raise ValueError("Anchor strides must be positive integers")
        if len(image_size) != 2 or min(image_size) < 1:
            raise ValueError("image_size must contain positive height and width")
        point_tracks: dict[str, np.ndarray] = {}
        for camera, values in self.point_tracks.items():
            if not camera:
                raise ValueError("point-track camera names must not be empty")
            tracks = np.asarray(values, dtype=np.float32)
            if tracks.ndim != 4 or any(size < 1 for size in tracks.shape[:-1]) or tracks.shape[-1] != 2:
                raise ValueError(f"point_tracks[{camera!r}] must have shape [T, N, H, 2], got {tracks.shape}")
            if not np.isfinite(tracks).all():
                raise ValueError(f"point_tracks[{camera!r}] contains non-finite values")
            point_tracks[camera] = tracks
            if camera in self.validity:
                mask = np.asarray(self.validity[camera])
                if mask.dtype != np.bool_ or mask.shape != tracks.shape[:-1]:
                    raise ValueError(f"validity[{camera!r}] must be boolean [T, N, H]")
        if set(self.validity) - set(point_tracks):
            raise ValueError("Validity supplied for unknown camera")
        if self.condition is None and not point_tracks:
            raise ValueError("An artifact must contain a trajectory or point tracks")
        object.__setattr__(self, "image_size", image_size)
        object.__setattr__(self, "point_tracks", point_tracks)

    def require_condition(self) -> TrajectoryCondition:
        if self.condition is None:
            raise ValueError("This operation requires a trajectory; artifact contains tracks only")
        return self.condition

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "schema_version": np.asarray(SCHEMA_VERSION),
            "metadata_json": np.asarray(
                json.dumps({k: v.to_dict() for k, v in self.spatial_metadata.items()})
            ),
            "points_xy": self.condition.path_xy.astype(np.float32) if self.condition else np.empty((0, 2)),
            "grasp_index": np.asarray(
                -1
                if self.condition is None or self.condition.grasp_index is None
                else self.condition.grasp_index,
                dtype=np.int64,
            ),
            "release_index": np.asarray(
                -1
                if self.condition is None or self.condition.release_index is None
                else self.condition.release_index,
                dtype=np.int64,
            ),
            "image_size": np.asarray(self.image_size, dtype=np.int64),
            "frame_stride": np.asarray(self.frame_stride, dtype=np.int64),
            "anchor_strides_json": np.asarray(json.dumps(self.anchor_strides)),
        }
        for camera, tracks in self.point_tracks.items():
            payload[f"tracks__{camera}"] = np.asarray(tracks, dtype=np.float32)
        for camera, mask in self.validity.items():
            payload[f"valid__{camera}"] = mask
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> EpisodeArtifact:
        with np.load(path, allow_pickle=False) as data:
            if "schema_version" in data.files and int(data["schema_version"]) != SCHEMA_VERSION:
                raise ValueError("Unsupported episode artifact schema version")
            grasp = int(data["grasp_index"])
            release = int(data["release_index"])
            frame_stride = int(data["frame_stride"]) if "frame_stride" in data.files else 1
            image_size = tuple(int(x) for x in data["image_size"])
            condition = (
                TrajectoryCondition(
                    np.asarray(data["points_xy"], dtype=np.float32),
                    image_shape=image_size,
                    grasp_index=None if grasp < 0 else grasp,
                    release_index=None if release < 0 else release,
                )
                if len(data["points_xy"])
                else None
            )
            tracks = {
                key.removeprefix("tracks__"): np.asarray(data[key], dtype=np.float32)
                for key in data.files
                if key.startswith("tracks__")
            }
            metadata = json.loads(str(data["metadata_json"])) if "metadata_json" in data.files else {}
            validity = {
                k.removeprefix("valid__"): data[k].copy() for k in data.files if k.startswith("valid__")
            }
            return cls(
                condition,
                image_size,
                tracks,
                frame_stride=frame_stride,
                spatial_metadata={k: SpatialMetadata.from_dict(v) for k, v in metadata.items()},
                validity=validity,
                anchor_strides=json.loads(str(data["anchor_strides_json"]))
                if "anchor_strides_json" in data.files
                else {},
            )


class EpisodeArtifactStore:
    """Lazy episode-keyed artifact reader suitable for LeRobot callbacks."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._cache: dict[int, EpisodeArtifact] = {}

    def path_for(self, episode_index: int) -> Path:
        return self.root / f"episode_{int(episode_index):06d}.npz"

    @property
    def episode_indices(self) -> list[int]:
        return [int(path.stem.split("_")[-1]) for path in sorted(self.root.glob("episode_*.npz"))]

    def first(self) -> EpisodeArtifact:
        indices = self.episode_indices
        if not indices:
            raise FileNotFoundError(f"No episode artifacts found in {self.root}")
        return self.get(indices[0])

    @property
    def rt_image_shape(self) -> tuple[int, int, int]:
        """Return the shared HWC RT-image shape after validating every artifact."""
        indices = self.episode_indices
        if not indices:
            raise FileNotFoundError(f"No episode artifacts found in {self.root}")
        expected = self.get(indices[0]).image_size
        for episode_index in indices[1:]:
            image_size = self.get(episode_index).image_size
            if image_size != expected:
                raise ValueError(
                    f"Episode {episode_index} image_size {image_size} is inconsistent with {expected}"
                )
        return (*expected, 3)

    def point_track_shape(self, camera: str) -> tuple[int, int, int]:
        """Return the shared per-frame ``[query, horizon, xy]`` shape for a camera."""
        indices = self.episode_indices
        if not indices:
            raise FileNotFoundError(f"No episode artifacts found in {self.root}")
        expected: tuple[int, int, int] | None = None
        for episode_index in indices:
            artifact = self.get(episode_index)
            if camera not in artifact.point_tracks:
                raise KeyError(f"Point tracks {camera!r} are missing from episode {episode_index}")
            shape = tuple(artifact.point_tracks[camera].shape[1:])
            if expected is None:
                expected = shape
            elif shape != expected:
                raise ValueError(
                    f"Point tracks {camera!r} have inconsistent per-frame shape in episode "
                    f"{episode_index}: {shape} != {expected}"
                )
        assert expected is not None
        return expected

    def __contains__(self, episode_index: int) -> bool:
        return self.path_for(episode_index).is_file()

    def get(self, episode_index: int) -> EpisodeArtifact:
        episode_index = int(episode_index)
        if episode_index not in self._cache:
            path = self.path_for(episode_index)
            if not path.is_file():
                raise FileNotFoundError(f"Missing episode artifact: {path}")
            self._cache[episode_index] = EpisodeArtifact.load(path)
        return self._cache[episode_index]

    def validate_complete(self, episode_indices: list[int]) -> None:
        missing = [index for index in episode_indices if index not in self]
        if missing:
            preview = ", ".join(str(x) for x in missing[:10])
            suffix = "..." if len(missing) > 10 else ""
            raise FileNotFoundError(f"Missing artifacts for episodes {preview}{suffix}")
