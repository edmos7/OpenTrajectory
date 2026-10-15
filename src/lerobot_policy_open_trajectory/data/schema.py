"""Portable spatial provenance, independent of model and dataset storage."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
METADATA_FILE = "meta/open_trajectory.json"


@dataclass(frozen=True)
class SpatialMetadata:
    camera_key: str
    image_size: tuple[int, int]  # height, width before normalization
    coordinate_frame: str = "image_plane"
    normalization: str = "image_dimensions"
    normalization_scope: str = "camera"
    units: str = "normalized"
    axis_order: tuple[str, ...] = ("x", "y")
    frame_stride: int = 1
    horizon_stride: int = 1
    fps: float | None = None
    padding: str = "repeat_terminal"
    source: str = "tracked"
    generator: str | None = None
    checkpoint: str | None = None
    calibration_ref: str | None = None
    depth_kind: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_size", tuple(self.image_size))
        object.__setattr__(self, "axis_order", tuple(self.axis_order))
        if not self.camera_key or len(self.image_size) != 2 or min(self.image_size) < 1:
            raise ValueError("camera_key and positive image_size (height, width) are required")
        if self.coordinate_frame not in {"image_plane", "camera", "robot_base", "world"}:
            raise ValueError("Unknown coordinate_frame")
        if self.normalization not in {"image_dimensions", "none"}:
            raise ValueError("Unsupported normalization")
        if self.normalization_scope not in {"camera", "episode", "dataset"}:
            raise ValueError("Unknown normalization_scope")
        if (
            self.frame_stride < 1
            or self.horizon_stride < 1
            or (self.fps is not None and (not math.isfinite(self.fps) or self.fps <= 0))
        ):
            raise ValueError("Temporal strides and fps must be positive")
        if self.coordinate_frame != "image_plane" and not self.calibration_ref:
            raise ValueError("Metric coordinate frames require a calibration reference")
        if self.coordinate_frame == "image_plane" and self.axis_order != ("x", "y"):
            raise ValueError("Image-plane coordinates require XY axis order")
        if self.coordinate_frame != "image_plane" and (
            self.normalization != "none" or self.units != "meters" or self.axis_order != ("x", "y", "z")
        ):
            raise ValueError("Metric coordinates require unnormalized XYZ in meters")
        if self.normalization == "image_dimensions" and self.units != "normalized":
            raise ValueError("Image-dimension normalization requires normalized units")
        if self.depth_kind not in {None, "relative", "metric", "colorized_relative"}:
            raise ValueError("Unknown depth_kind")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SpatialMetadata:
        return cls(**value)


def write_dataset_metadata(root: str | Path, features: dict, *, source: dict | None = None) -> None:
    """Write a Hub-portable sidecar; unknown legacy provenance is explicit."""
    path = Path(root) / METADATA_FILE
    previous = read_dataset_metadata(root) if path.exists() else {}
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "features": {**previous.get("features", {}), **features},
        "source": source if source is not None else previous.get("source", {}),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def read_dataset_metadata(root: str | Path) -> dict:
    payload = json.loads((Path(root) / METADATA_FILE).read_text())
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported spatial metadata schema version")
    return payload
