"""Render reproducible examples from canonical episode artifacts."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .data.artifacts import EpisodeArtifact


def _rgb_uint8(frame: np.ndarray) -> np.ndarray:
    value = np.asarray(frame)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"Expected an HWC RGB frame, got {value.shape}")
    if np.issubdtype(value.dtype, np.floating):
        value = value * 255.0 if value.max(initial=0) <= 1.0 else value
    return np.clip(value, 0, 255).astype(np.uint8)


def _normalized_pixels(points_xy: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    height, width = image_size
    points = np.asarray(points_xy, dtype=np.float32).copy()
    points[..., 0] = np.clip(points[..., 0] * width, 0, width - 1)
    points[..., 1] = np.clip(points[..., 1] * height, 0, height - 1)
    return points


def render_gripper_tip_frame(
    frame_rgb: np.ndarray,
    path_xy: np.ndarray,
    path_index: int,
    *,
    line_width: int = 5,
) -> np.ndarray:
    """Draw the gripper-tip trail and its current episode position."""
    frame = _rgb_uint8(frame_rgb)
    path = np.asarray(path_xy, dtype=np.float32)
    if path.ndim != 2 or path.shape[1] != 2 or not len(path):
        raise ValueError(f"Expected path_xy [T, 2], got {path.shape}")
    path_index = int(np.clip(path_index, 0, len(path) - 1))
    points = _normalized_pixels(path[: path_index + 1], frame.shape[:2])

    base = Image.fromarray(frame).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    denominator = max(len(path) - 1, 1)
    for index in range(len(points) - 1):
        red = round(64 + 191 * index / denominator)
        draw.line(
            (tuple(points[index]), tuple(points[index + 1])),
            fill=(red, 32, 224, 220),
            width=line_width,
        )
    x, y = points[-1]
    radius = max(line_width + 2, 6)
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=(255, 255, 255, 245),
        outline=(25, 25, 25, 245),
        width=2,
    )
    return np.asarray(Image.alpha_composite(base, layer).convert("RGB"))


def render_point_tracks_frame(
    frame_rgb: np.ndarray,
    point_tracks: np.ndarray,
    *,
    persistent_point_count: int = 0,
    line_width: int = 2,
) -> np.ndarray:
    """Draw every stored current and forecast point, without visibility filtering.

    The first ``persistent_point_count`` rows are colored cyan. Remaining rows,
    which can be freshly seeded grid tracks, are colored amber. A visibility mask
    is deliberately not accepted: predicted positions remain rendered while the
    corresponding scene point is occluded.
    """
    frame = _rgb_uint8(frame_rgb)
    tracks = np.asarray(point_tracks, dtype=np.float32)
    if tracks.ndim != 3 or tracks.shape[-1] != 2 or any(size < 1 for size in tracks.shape[:-1]):
        raise ValueError(f"Expected point_tracks [N, H, 2], got {tracks.shape}")
    if not np.isfinite(tracks).all():
        raise ValueError("point_tracks contains non-finite coordinates")
    if not 0 <= persistent_point_count <= tracks.shape[0]:
        raise ValueError("persistent_point_count must be between zero and the query count")

    pixels = _normalized_pixels(tracks, frame.shape[:2])
    base = Image.fromarray(frame).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    horizon = tracks.shape[1]
    for query_index, query in enumerate(pixels):
        rgb = (35, 211, 238) if query_index < persistent_point_count else (251, 146, 60)
        for step in range(horizon - 1):
            opacity = round(210 - 150 * step / max(horizon - 1, 1))
            draw.line(
                (tuple(query[step]), tuple(query[step + 1])),
                fill=(*rgb, opacity),
                width=line_width,
            )
        for step, (x, y) in enumerate(query):
            opacity = round(225 - 145 * step / max(horizon - 1, 1))
            radius = 4 if step == 0 else 1
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*rgb, opacity))
    return np.asarray(Image.alpha_composite(base, layer).convert("RGB"))


def _video_frames(path: str | Path, frame_step: int) -> tuple[Iterator[tuple[int, np.ndarray]], float, int]:
    if frame_step < 1:
        raise ValueError("frame_step must be positive")
    try:
        import av
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Episode rendering requires the 'av' package") from exc

    container = av.open(str(path))
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate is not None else 30.0
    total_frames = int(stream.frames)

    def selected() -> Iterator[tuple[int, np.ndarray]]:
        try:
            for index, frame in enumerate(container.decode(stream)):
                if index % frame_step == 0:
                    yield index, frame.to_ndarray(format="rgb24")
        finally:
            container.close()

    return selected(), fps, total_frames


def _save_gif(frames: list[np.ndarray], path: Path, duration_ms: int) -> None:
    if not frames:
        raise ValueError("Cannot save an empty animation")
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=max(duration_ms, 1),
        loop=0,
        disposal=2,
    )


def render_episode_visuals(
    artifact_path: str | Path,
    video_path: str | Path,
    output_dir: str | Path,
    *,
    camera: str | None = None,
    persistent_point_count: int = 0,
    frame_step: int = 4,
) -> list[Path]:
    """Render the four canonical visuals for a complete episode."""
    artifact = EpisodeArtifact.load(artifact_path)
    if camera is None:
        if len(artifact.point_tracks) != 1:
            raise ValueError("camera is required when an artifact contains multiple point-track views")
        camera = next(iter(artifact.point_tracks))
    if camera not in artifact.point_tracks:
        raise KeyError(f"Point tracks {camera!r} are not present in the artifact")

    decoded, fps, total_video_frames = _video_frames(video_path, frame_step)
    gripper_frames: list[np.ndarray] = []
    point_track_frames: list[np.ndarray] = []
    first_frame: np.ndarray | None = None
    tracks = artifact.point_tracks[camera]
    for frame_index, frame in decoded:
        if frame.shape[:2] != artifact.image_size:
            frame = cv2.resize(frame, artifact.image_size[::-1], interpolation=cv2.INTER_AREA)
        if first_frame is None:
            first_frame = frame
        denominator = max(total_video_frames - 1, frame_index, 1)
        path_index = round(frame_index / denominator * (len(artifact.condition.path_xy) - 1))
        track_index = min(frame_index // artifact.frame_stride, tracks.shape[0] - 1)
        gripper_frames.append(render_gripper_tip_frame(frame, artifact.condition.path_xy, path_index))
        point_track_frames.append(
            render_point_tracks_frame(
                frame,
                tracks[track_index],
                persistent_point_count=persistent_point_count,
            )
        )

    if first_frame is None:
        raise ValueError(f"No frames could be decoded from {video_path}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rendered = {
        "rt_image.png": artifact.condition.to_rt_image(),
        "sequence_overlay.png": artifact.condition.overlay(first_frame, "sequence"),
    }
    paths: list[Path] = []
    for name, image in rendered.items():
        path = output / name
        Image.fromarray(image).save(path)
        paths.append(path)

    duration_ms = round(1000 * frame_step / max(fps, 1e-6))
    gripper_path = output / "gripper_tip_tracking.gif"
    point_tracks_path = output / "point_tracks.gif"
    _save_gif(gripper_frames, gripper_path, duration_ms)
    _save_gif(point_track_frames, point_tracks_path, duration_ms)
    paths.extend((gripper_path, point_tracks_path))
    return paths


def render_artifact_visuals(
    artifact_path: str | Path,
    frame_path: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Render sequence and RT-image panels through the production encoders."""
    artifact = EpisodeArtifact.load(artifact_path)
    frame = np.asarray(Image.open(frame_path).convert("RGB"))
    if frame.shape[:2] != artifact.image_size:
        frame = np.asarray(Image.fromarray(frame).resize(artifact.image_size[::-1]))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rendered = {
        "sequence_overlay.png": artifact.condition.overlay(frame, "sequence"),
        "rt_image.png": artifact.condition.to_rt_image(),
        "rt_image_overlay.png": artifact.condition.overlay(frame, "rt_image"),
    }
    paths = []
    for name, image in rendered.items():
        path = output / name
        Image.fromarray(image).save(path)
        paths.append(path)
    return paths
