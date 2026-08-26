"""Trajectory drawing, sequence encoding, RT-image rendering, and overlays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw


def _as_rgb_uint8(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"Expected an HWC RGB image, got {value.shape}")
    if np.issubdtype(value.dtype, np.floating):
        value = value * 255.0 if value.max(initial=0) <= 1.0 else value
    return np.clip(value, 0, 255).astype(np.uint8)


@dataclass(frozen=True)
class TrajectoryCondition:
    """A normalized path with no markers or one grasp/release pair."""

    path_xy: np.ndarray
    image_shape: tuple[int, int]
    grasp_index: int | None = None
    release_index: int | None = None

    def __post_init__(self) -> None:
        path = np.asarray(self.path_xy, dtype=np.float32)
        if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
            raise ValueError("path_xy must have shape [N, 2] with at least two points")
        if not np.isfinite(path).all():
            raise ValueError("path_xy contains non-finite coordinates")
        if np.any(path < 0) or np.any(path > 1):
            raise ValueError("path_xy coordinates must be normalized to [0, 1]")
        height, width = self.image_shape
        if height <= 0 or width <= 0:
            raise ValueError("image_shape must contain positive height and width")
        markers = (self.grasp_index, self.release_index)
        if (markers[0] is None) != (markers[1] is None):
            raise ValueError("grasp and release markers must be both absent or both present")
        if markers[0] is not None:
            grasp = int(markers[0])
            release = int(markers[1])
            if not (0 <= grasp < release < len(path)):
                raise ValueError("markers must satisfy 0 <= grasp < release < path length")
        object.__setattr__(self, "path_xy", path)

    @property
    def has_interaction(self) -> bool:
        return self.grasp_index is not None

    @classmethod
    def from_pixels(
        cls,
        points_xy: np.ndarray,
        image_shape: tuple[int, int],
        grasp_index: int | None = None,
        release_index: int | None = None,
    ) -> TrajectoryCondition:
        height, width = image_shape
        points = np.asarray(points_xy, dtype=np.float32).copy()
        points[:, 0] = np.clip(points[:, 0] / width, 0.0, 1.0)
        points[:, 1] = np.clip(points[:, 1] / height, 0.0, 1.0)
        return cls(points, image_shape, grasp_index, release_index)

    def _resample(self, length: int) -> tuple[np.ndarray, np.ndarray]:
        if length < 2:
            raise ValueError("length must be at least 2")
        delta = np.linalg.norm(np.diff(self.path_xy, axis=0), axis=1)
        distance = np.concatenate(([0.0], np.cumsum(delta))).astype(np.float32)
        total = float(distance[-1])
        if total <= 1e-8:
            points = np.repeat(self.path_xy[:1], length, axis=0)
            index_map = np.linspace(0, length - 1, len(self.path_xy), dtype=np.float32)
            return points, index_map
        target = np.linspace(0.0, total, length, dtype=np.float32)
        points = np.stack(
            [np.interp(target, distance, self.path_xy[:, axis]) for axis in range(2)], axis=1
        ).astype(np.float32)
        return points, distance / total * (length - 1)

    def to_sequence(self, length: int = 224) -> np.ndarray:
        """Return `[length, 3]` as normalized XY plus a binary active interval."""
        points, index_map = self._resample(length)
        action = np.zeros(length, dtype=np.float32)
        if self.has_interaction:
            grasp = round(float(index_map[self.grasp_index]))
            release = round(float(index_map[self.release_index]))
            action[grasp : release + 1] = 1.0
            action[release + 1 :] = 0.0
        return np.column_stack((points, action)).astype(np.float32)

    def to_rt_image(
        self,
        line_width: int = 5,
        marker_radius: int = 15,
        marker_width: int = 8,
    ) -> np.ndarray:
        """Render the RT-Trajectory image contract on a black RGB canvas."""
        height, width = self.image_shape
        canvas = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        final_index = self.release_index if self.has_interaction else len(self.path_xy) - 1
        points = self.path_xy[: final_index + 1].copy()
        points[:, 0] *= width
        points[:, 1] *= height
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        denominator = max(len(points) - 1, 1)
        for index in range(len(points) - 1):
            red = round(255 * index / denominator)
            p0 = tuple(np.round(points[index]).astype(int))
            p1 = tuple(np.round(points[index + 1]).astype(int))
            draw.line((p0, p1), fill=(red, 0, 0), width=line_width)

        def ring(index: int | None, color: tuple[int, int, int]) -> None:
            if index is None:
                return
            local_index = min(index, len(points) - 1)
            x, y = np.round(points[local_index]).astype(int)
            outer = (x - marker_radius, y - marker_radius, x + marker_radius, y + marker_radius)
            inner_radius = max(marker_radius - marker_width, 1)
            inner = (x - inner_radius, y - inner_radius, x + inner_radius, y + inner_radius)
            draw.ellipse(outer, fill=color)
            draw.ellipse(inner, fill=(0, 0, 0))

        ring(self.grasp_index, (0, 255, 0))
        ring(self.release_index, (0, 0, 255))
        return np.asarray(canvas)

    def overlay(
        self, frame_rgb: np.ndarray, mode: Literal["sequence", "rt_image"], alpha: float = 0.75
    ) -> np.ndarray:
        frame = _as_rgb_uint8(frame_rgb)
        if frame.shape[:2] != self.image_shape:
            raise ValueError("Overlay frame shape does not match trajectory image_shape")
        if mode == "rt_image":
            layer = self.to_rt_image()
        elif mode == "sequence":
            points = self.path_xy.copy()
            points[:, 0] *= self.image_shape[1]
            points[:, 1] *= self.image_shape[0]
            canvas = Image.new("RGB", self.image_shape[::-1], (0, 0, 0))
            draw = ImageDraw.Draw(canvas)
            denominator = max(len(points) - 1, 1)
            for index in range(len(points) - 1):
                active = bool(self.has_interaction and self.grasp_index <= index <= self.release_index)
                color = (int(np.clip((index / denominator) * 255, 0, 255)), 0, 255 if active else 0)
                p0 = tuple(
                    np.clip(
                        np.round(points[index]).astype(int),
                        [0, 0],
                        [self.image_shape[1] - 1, self.image_shape[0] - 1],
                    )
                )
                p1 = tuple(
                    np.clip(
                        np.round(points[index + 1]).astype(int),
                        [0, 0],
                        [self.image_shape[1] - 1, self.image_shape[0] - 1],
                    )
                )
                draw.line([p0, p1], fill=color, width=6)
            layer = np.asarray(canvas)
        else:
            raise ValueError(f"Unknown overlay mode: {mode!r}")
        mask = (layer.max(axis=2, keepdims=True) > 0).astype(np.float32) * alpha
        return (frame * (1.0 - mask) + layer * mask).clip(0, 255).astype(np.uint8)


class _OpenCVTrajectoryDrawer:
    def __init__(self, frame_rgb: np.ndarray):
        self.frame = _as_rgb_uint8(frame_rgb)
        self.height, self.width = self.frame.shape[:2]
        self.points: list[tuple[int, int]] = []
        self.markers: list[int] = []
        self.started = False
        self.finished = False
        self.window = "Open Trajectory | left: start/grasp/release | right/space: finish"

    def _mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.started:
                self.started = True
                self.points.append((x, y))
            elif len(self.markers) < 2:
                self.points.append((x, y))
                self.markers.append(len(self.points) - 1)
            self._show()
        elif event == cv2.EVENT_MOUSEMOVE and self.started and not self.finished:
            self.points.append((x, y))
            self._show()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.finished = True

    def _show(self) -> None:
        display = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
        for index in range(len(self.points) - 1):
            cv2.line(display, self.points[index], self.points[index + 1], (0, 0, 255), 2)
        for marker_number, point_index in enumerate(self.markers):
            color = (0, 255, 0) if marker_number == 0 else (255, 0, 0)
            cv2.circle(display, self.points[point_index], 8, color, 2)
        cv2.imshow(self.window, display)

    def run(self) -> TrajectoryCondition:
        self._show()
        cv2.setMouseCallback(self.window, self._mouse)
        while not self.finished:
            key = cv2.waitKey(10) & 0xFF
            if key == ord(" "):
                self.finished = True
            elif key == 27:
                cv2.destroyAllWindows()
                raise RuntimeError("Trajectory drawing aborted")
            if cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyAllWindows()
                raise RuntimeError("Trajectory drawing window was closed")
        if len(self.points) < 2:
            cv2.destroyAllWindows()
            raise ValueError("Draw a path with at least two points")
        if len(self.markers) == 1:
            cv2.destroyAllWindows()
            raise ValueError("A grasp marker requires a later release marker")
        condition = TrajectoryCondition.from_pixels(
            np.asarray(self.points),
            (self.height, self.width),
            self.markers[0] if self.markers else None,
            self.markers[1] if self.markers else None,
        )
        preview = condition.overlay(self.frame, "rt_image")
        cv2.imshow(self.window, cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (13, 10, ord(" ")):
                cv2.destroyAllWindows()
                return condition
            if key in (ord("r"), ord("R")):
                cv2.destroyAllWindows()
                return _OpenCVTrajectoryDrawer(self.frame).run()
            if key == 27:
                cv2.destroyAllWindows()
                raise RuntimeError("Trajectory drawing aborted")


def draw_trajectory_condition(frame_rgb: np.ndarray) -> TrajectoryCondition:
    """Open the blocking trajectory drawer on a live RGB frame."""
    return _OpenCVTrajectoryDrawer(frame_rgb).run()
