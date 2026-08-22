#!/usr/bin/env python3
"""Visualize the trajectory-only gripper-tip data generation process.

This is a sanity-check companion for the curated trajectory-only generator.
It does not write a dataset. It renders the current pipeline for one LeRobot
episode:

  demo frames -> YOLO robot masks -> gripper-tip seed points -> TAPNext++ point
  tracks -> median gripper-tip path -> observation.global_trajectory rendering.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SOURCE_REPO_ID = "edmos7/opentrajectory_dataset"
DEFAULT_CAMERA_KEY = "observation.images.side"
DEFAULT_YOLO_PATH = REPO_ROOT / "models" / "segmentso101.pt"
DEFAULT_TAPNEXT_PATH = REPO_ROOT / "models" / "tapnextpp_ckpt.pt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "assets" / "visuals" / "trajectory_only_generation"

log = logging.getLogger(__name__)


def _ensure_lerobot() -> None:
    try:
        import lerobot  # noqa: F401
    except ImportError:
        python = Path.home() / "miniforge3" / "envs" / "lerobotclone" / "bin" / "python"
        if not python.exists():
            sys.exit("ERROR: lerobot is unavailable and lerobotclone was not found.")
        result = subprocess.run([str(python), __file__, *sys.argv[1:]])
        raise SystemExit(result.returncode)


_ensure_lerobot()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from data_generation.generate_global_trajectory_dataset import (  # noqa: E402
    find_seed,
    load_episode_arrays,
    parse_task_event_thresholds,
)
from utils.rt_trajectory_utils import (  # noqa: E402
    best_gripper_mask,
    detect_grasp_release,
    find_gripper_action_index,
    infer_tip_anchors,
    normalize_action_signal,
    render_debug_overlay,
    resolve_camera_key,
    rgb_to_bgr,
    rgb_to_gray_bgr,
    sample_bottom_band_tip_points,
    sample_tip_points,
    trajectory_is_usable,
    yolo_tip_trajectory,
)
from utils.tracking import compute_median_trajectory, load_tapnext_model, run_tapnext  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render trajectory-only data-generation walkthrough artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source-repo-id", default=DEFAULT_SOURCE_REPO_ID)
    p.add_argument("--source-root", default=None)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--camera-key", default=DEFAULT_CAMERA_KEY)
    p.add_argument("--trajectory-key", default="observation.global_trajectory")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    p.add_argument("--yolo-path", default=str(DEFAULT_YOLO_PATH))
    p.add_argument("--tapnext-checkpoint", default=str(DEFAULT_TAPNEXT_PATH))
    p.add_argument(
        "--track-mode",
        choices=("existing", "recompute"),
        default="existing",
        help=(
            "Use the dataset's existing global trajectory by default. "
            "Use 'recompute' only when the visualization should rerun TAPNext++."
        ),
    )
    p.add_argument("--tapnext-size", type=int, default=256)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--num-tip-points", type=int, default=16)
    p.add_argument("--tip-conf", type=float, default=0.85)
    p.add_argument("--tip-band-quantile", type=float, default=0.80)
    p.add_argument("--tip-sample-mode", choices=("morphology", "band"), default="morphology")
    p.add_argument("--tip-radius", type=float, default=5.0)
    p.add_argument(
        "--seed-max-distance-px",
        type=float,
        default=80.0,
        help="Maximum image-space distance for choosing the first detection near the stored trajectory.",
    )
    p.add_argument("--yolo-grayscale", action="store_true")
    p.add_argument("--event-threshold", type=float, default=0.25)
    p.add_argument(
        "--task-event-threshold",
        action="append",
        default=[],
        metavar="TASK_ID:THRESHOLD",
        help="Override event threshold for task ids, e.g. 0:0.20,5:0.35.",
    )

    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--max-video-frames", type=int, default=180)
    p.add_argument("--line-width", type=int, default=5)
    p.add_argument("--track-line-width", type=int, default=2)
    p.add_argument("--point-radius", type=int, default=5)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def fit_text_pixels(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        return text
    ellipsis = "..."
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        candidate = text[:mid].rstrip() + ellipsis
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            lo = mid + 1
        else:
            hi = mid
    return text[: max(1, lo - 1)].rstrip() + ellipsis


def add_title(image_rgb: np.ndarray, title: str) -> np.ndarray:
    image = Image.fromarray(to_uint8_rgb(image_rgb)).convert("RGB")
    width, _height = image.size
    bar_h = 44
    canvas = Image.new("RGB", (width, image.height + bar_h), (18, 18, 18))
    canvas.paste(image, (0, bar_h))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    draw.text((14, 10), fit_text_pixels(draw, title, font, width - 28), fill=(245, 245, 245), font=font)
    return np.asarray(canvas)


def wrap_text_pixels(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def add_explanation(image_rgb: np.ndarray, title: str, body: str) -> np.ndarray:
    base = Image.fromarray(to_uint8_rgb(image_rgb)).convert("RGB")
    draw_base = ImageDraw.Draw(base, "RGBA")
    width, height = base.size
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        body_font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

    panel_h = 128
    draw_base.rectangle([0, height - panel_h, width, height], fill=(0, 0, 0, 188))
    draw = ImageDraw.Draw(base)
    draw.text((18, height - panel_h + 14), fit_text_pixels(draw, title, title_font, width - 36), fill=(255, 255, 255), font=title_font)
    y = height - panel_h + 48
    for line in wrap_text_pixels(draw, body, body_font, width - 36)[:4]:
        draw.text((18, y), line, fill=(230, 230, 230), font=body_font)
        y += 22
    return np.asarray(base)


def hold_frame(frame_rgb: np.ndarray, title: str, body: str, fps: float, seconds: float) -> list[np.ndarray]:
    frame = add_explanation(frame_rgb, title, body)
    return [frame] * max(1, int(round(fps * seconds)))


def side_panel_frame(
    image_rgb: np.ndarray,
    title: str,
    body: str,
    *,
    panel_width: int = 330,
) -> np.ndarray:
    image = Image.fromarray(to_uint8_rgb(image_rgb)).convert("RGB")
    width, height = image.size
    canvas = Image.new("RGB", (width + panel_width, height), (18, 18, 18))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
        body_font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

    x = width + 24
    max_w = panel_width - 48
    y = 34
    for line in wrap_text_pixels(draw, title, title_font, max_w)[:3]:
        draw.text((x, y), line, fill=(255, 255, 255), font=title_font)
        y += 30
    y += 14
    for line in wrap_text_pixels(draw, body, body_font, max_w)[:12]:
        draw.text((x, y), line, fill=(226, 226, 226), font=body_font)
        y += 24
    return np.asarray(canvas)


def repeat_side_panel(
    image_rgb: np.ndarray,
    title: str,
    body: str,
    fps: float,
    seconds: float,
) -> list[np.ndarray]:
    frame = side_panel_frame(image_rgb, title, body)
    return [frame] * max(1, int(round(fps * seconds)))


def trajectory_index_for_frame(frame_idx: int, frame_stride: int, trajectory_len: int) -> int:
    return int(np.clip(frame_idx // max(frame_stride, 1), 0, max(trajectory_len - 1, 0)))


def write_video(path: Path, frames_rgb: list[np.ndarray], fps: float, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists. Use --overwrite to replace it.")
    if not frames_rgb:
        raise ValueError(f"No frames to write for {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    first = frames_rgb[0]
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}.")
    for frame in frames_rgb:
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(cv2.cvtColor(to_uint8_rgb(frame), cv2.COLOR_RGB2BGR))
    writer.release()


def save_png(path: Path, image_rgb: np.ndarray, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists. Use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8_rgb(image_rgb)).save(path)


def sampled_indices(n_frames: int, max_frames: int) -> np.ndarray:
    if n_frames <= max_frames:
        return np.arange(n_frames, dtype=int)
    return np.linspace(0, n_frames - 1, max_frames, dtype=int)


def mask_color(idx: int) -> tuple[int, int, int]:
    palette = [
        (235, 64, 52),
        (52, 152, 219),
        (46, 204, 113),
        (241, 196, 15),
        (155, 89, 182),
        (230, 126, 34),
        (26, 188, 156),
    ]
    return palette[idx % len(palette)]


def render_all_robot_masks(
    yolo_model,
    frame_rgb: np.ndarray,
    *,
    conf: float,
    grayscale: bool,
    alpha: float = 0.48,
    detect_frame_rgb: np.ndarray | None = None,
) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    detect_rgb = frame_rgb if detect_frame_rgb is None else detect_frame_rgb
    bgr = rgb_to_gray_bgr(detect_rgb) if grayscale else rgb_to_bgr(detect_rgb)
    results = yolo_model(bgr, conf=conf, verbose=False)
    overlay = frame_rgb.astype(np.float32).copy()
    if results[0].masks is None or results[0].boxes is None:
        return frame_rgb.copy()

    masks = results[0].masks.data
    classes = results[0].boxes.cls
    for i in range(len(classes)):
        class_id = int(classes[i].item())
        name = str(yolo_model.names.get(class_id, class_id))
        if "so101" not in name.lower():
            continue
        raw = masks[i].cpu().numpy()
        mask = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
        color = np.array(mask_color(i), dtype=np.float32)
        overlay[mask] = overlay[mask] * (1.0 - alpha) + color * alpha

    return overlay.clip(0, 255).astype(np.uint8)


def render_gripper_mask_and_tip_points(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    query_points: np.ndarray,
    *,
    band_quantile: float,
    tip_radius: float,
    point_radius: int,
) -> np.ndarray:
    out = frame_rgb.copy()
    mask_bool = mask > 0
    tint = np.zeros_like(out)
    tint[..., 1] = 220
    out[mask_bool] = (out[mask_bool].astype(np.float32) * 0.48 + tint[mask_bool].astype(np.float32) * 0.52).astype(np.uint8)

    ys, _xs = np.where(mask_bool)
    if len(ys):
        y_cut = int(np.quantile(ys, np.clip(band_quantile, 0.0, 1.0)))
        band = mask_bool.copy()
        band[:y_cut, :] = False
        blue_tint = np.zeros_like(out)
        blue_tint[..., 2] = 255
        out[band] = (out[band].astype(np.float32) * 0.35 + blue_tint[band].astype(np.float32) * 0.65).astype(np.uint8)

    pil = Image.fromarray(out)
    draw = ImageDraw.Draw(pil)
    for ax, ay in infer_tip_anchors(mask, band_quantile):
        r = max(2, int(round(tip_radius)))
        draw.ellipse([ax - r, ay - r, ax + r, ay + r], outline=(255, 255, 255), width=3)

    for _t, y, x in query_points:
        x_i, y_i = int(round(float(x))), int(round(float(y)))
        draw.ellipse(
            [x_i - point_radius, y_i - point_radius, x_i + point_radius, y_i + point_radius],
            fill=(255, 230, 0),
            outline=(0, 0, 0),
            width=2,
        )
    return np.asarray(pil)


def render_tracks_frame(
    frame_rgb: np.ndarray,
    tracks: np.ndarray,
    t: int,
    *,
    line_width: int,
    point_radius: int,
) -> np.ndarray:
    out = Image.fromarray(frame_rgb.copy())
    draw = ImageDraw.Draw(out)
    colors = [(255, 230, 0), (0, 190, 255), (255, 80, 80), (160, 255, 120)]
    for i in range(tracks.shape[0]):
        color = colors[i % len(colors)]
        pts = tracks[i, : t + 1]
        if len(pts) >= 2:
            xy = [tuple(np.round(p).astype(int)) for p in pts]
            draw.line(xy, fill=color, width=line_width)
        x, y = np.round(tracks[i, t]).astype(int)
        draw.ellipse(
            [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
            fill=color,
            outline=(0, 0, 0),
            width=1,
        )
    return np.asarray(out)


def render_median_frame(
    frame_rgb: np.ndarray,
    tracks: np.ndarray,
    trajectory_xy: np.ndarray,
    t: int,
    *,
    line_width: int,
    point_radius: int,
) -> np.ndarray:
    out = Image.fromarray(frame_rgb.copy())
    draw = ImageDraw.Draw(out)
    for i in range(tracks.shape[0]):
        pts = [tuple(np.round(p).astype(int)) for p in tracks[i, : t + 1]]
        if len(pts) >= 2:
            draw.line(pts, fill=(0, 190, 255), width=1)

    med = trajectory_xy[: t + 1]
    if len(med) >= 2:
        med_pts = [tuple(np.round(p).astype(int)) for p in med]
        draw.line(med_pts, fill=(255, 255, 255), width=line_width + 2)
        draw.line(med_pts, fill=(255, 0, 0), width=line_width)
    x, y = np.round(trajectory_xy[t]).astype(int)
    draw.ellipse(
        [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
        fill=(255, 0, 0),
        outline=(255, 255, 255),
        width=2,
    )
    return np.asarray(out)


def render_existing_trajectory_progress_frame(
    frame_rgb: np.ndarray,
    global_trajectory: np.ndarray,
    t: int,
    *,
    line_width: int,
    point_radius: int,
) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    traj = np.asarray(global_trajectory, dtype=np.float32)
    pts = traj[:, :2].copy()
    pts[:, 0] *= w - 1
    pts[:, 1] *= h - 1
    t = int(np.clip(t, 0, len(pts) - 1))

    out = Image.fromarray(frame_rgb.copy())
    draw = ImageDraw.Draw(out)
    for i in range(t):
        p0 = tuple(np.clip(np.round(pts[i]).astype(int), [0, 0], [w - 1, h - 1]))
        p1 = tuple(np.clip(np.round(pts[i + 1]).astype(int), [0, 0], [w - 1, h - 1]))
        draw.line([p0, p1], fill=(230, 30, 30), width=line_width)
    x, y = np.clip(np.round(pts[t]).astype(int), [0, 0], [w - 1, h - 1])
    draw.ellipse(
        [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
        fill=(230, 30, 30),
        outline=(255, 255, 255),
        width=2,
    )
    return np.asarray(out)


def render_global_trajectory_blank(
    trajectory: np.ndarray,
    image_shape: tuple[int, int],
    *,
    line_width: int,
) -> np.ndarray:
    h, w = image_shape
    pts = np.asarray(trajectory[:, :2], dtype=np.float32).copy()
    pts[:, 0] *= w - 1
    pts[:, 1] *= h - 1
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for i in range(len(pts) - 1):
        p0 = tuple(np.clip(np.round(pts[i]).astype(int), [0, 0], [w - 1, h - 1]))
        p1 = tuple(np.clip(np.round(pts[i + 1]).astype(int), [0, 0], [w - 1, h - 1]))
        draw.line([p0, p1], fill=(230, 30, 30), width=line_width)
    return np.asarray(canvas)


def build_global_trajectory(
    dataset: LeRobotDataset,
    actions: np.ndarray,
    trajectory_xy: np.ndarray,
    frame_shape: tuple[int, int],
    frame_stride: int,
    threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = frame_shape
    grip_idx = find_gripper_action_index(dataset)
    action_norm = normalize_action_signal(dataset, actions, grip_idx)[::frame_stride][: len(trajectory_xy)]
    grasp_idx, release_idx, peak_starts = detect_grasp_release(action_norm, threshold=threshold)
    binary_action = np.zeros(len(trajectory_xy), dtype=np.float32)
    if grasp_idx is not None and release_idx is not None and release_idx >= grasp_idx:
        binary_action[grasp_idx : release_idx + 1] = 1.0
    global_trajectory = np.stack(
        [
            np.clip(trajectory_xy[:, 0] / w, 0.0, 1.0),
            np.clip(trajectory_xy[:, 1] / h, 0.0, 1.0),
            binary_action,
        ],
        axis=1,
    ).astype(np.float32)
    return global_trajectory, {
        "grasp_idx": grasp_idx,
        "release_idx": release_idx,
        "peak_starts": peak_starts,
        "gripper_action_index": grip_idx,
    }


def load_source_episode(
    repo_id: str,
    root: str | None,
    episode: int,
    camera_key: str,
) -> tuple[LeRobotDataset, str, np.ndarray, np.ndarray, int | None]:
    ds = LeRobotDataset(repo_id=repo_id, root=root, episodes=[episode], download_videos=True)
    resolved_camera_key = resolve_camera_key(ds, camera_key)
    frames_rgb, actions, task_id = load_episode_arrays(ds, resolved_camera_key)
    return ds, resolved_camera_key, frames_rgb, actions, task_id


def load_existing_global_trajectory(dataset: LeRobotDataset, trajectory_key: str) -> np.ndarray:
    if trajectory_key not in dataset.features:
        raise KeyError(
            f"Dataset does not contain {trajectory_key!r}. "
            "Use --track-mode recompute to rerun trajectory tracking."
        )
    value = dataset[0][trajectory_key]
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    trajectory = np.asarray(value, dtype=np.float32)
    if trajectory.ndim != 2 or trajectory.shape[1] < 3:
        raise ValueError(f"Expected {trajectory_key} with shape [T, 3], got {trajectory.shape}.")
    return trajectory[:, :3].astype(np.float32)


def trajectory_point_px(
    global_trajectory: np.ndarray,
    frame_idx: int,
    frame_stride: int,
    frame_shape: tuple[int, int],
) -> np.ndarray:
    h, w = frame_shape
    traj_idx = trajectory_index_for_frame(frame_idx, frame_stride, len(global_trajectory))
    point = np.asarray(global_trajectory[traj_idx, :2], dtype=np.float32).copy()
    point[0] *= w - 1
    point[1] *= h - 1
    return point


def mask_tip_xy(mask: np.ndarray, band_quantile: float) -> np.ndarray | None:
    anchors = infer_tip_anchors(mask, band_quantile)
    if anchors:
        return np.asarray(anchors, dtype=np.float32).mean(axis=0)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    idx = int(np.argmax(ys))
    return np.asarray([xs[idx], ys[idx]], dtype=np.float32)


def sample_query_points_for_mask(
    mask: np.ndarray,
    n_points: int,
    rng: np.random.Generator,
    band_quantile: float,
    tip_radius: float,
    sample_mode: str,
) -> np.ndarray:
    if sample_mode == "morphology":
        return sample_tip_points(mask, n_points, rng, band_quantile, tip_radius)
    return sample_bottom_band_tip_points(mask, n_points, rng, band_quantile)


def find_seed_near_global_trajectory(
    yolo_model,
    frames_rgb: np.ndarray,
    global_trajectory: np.ndarray,
    *,
    tip_conf: float,
    frame_stride: int,
    n_points: int,
    band_quantile: float,
    tip_radius: float,
    sample_mode: str,
    yolo_grayscale: bool,
    max_distance_px: float,
    rng: np.random.Generator,
) -> tuple[int, np.ndarray, dict[str, Any]]:
    frame_h, frame_w = frames_rgb.shape[1:3]
    closest: tuple[float, int, np.ndarray, dict[str, Any]] | None = None
    for frame_idx in range(0, len(frames_rgb), max(frame_stride, 1)):
        frame_rgb = frames_rgb[frame_idx]
        bgr = rgb_to_gray_bgr(frame_rgb) if yolo_grayscale else rgb_to_bgr(frame_rgb)
        results = yolo_model(bgr, conf=tip_conf, verbose=False)
        candidate = best_gripper_mask(results, frame_h, frame_w, yolo_model.names)
        if candidate is None:
            continue
        tip_xy = mask_tip_xy(candidate["mask"], band_quantile)
        if tip_xy is None:
            continue
        traj_xy = trajectory_point_px(global_trajectory, frame_idx, frame_stride, (frame_h, frame_w))
        distance = float(np.linalg.norm(tip_xy - traj_xy))
        query_points = sample_query_points_for_mask(
            candidate["mask"],
            n_points,
            rng,
            band_quantile,
            tip_radius,
            sample_mode,
        )
        candidate = dict(candidate)
        candidate.update(
            {
                "frame_idx": frame_idx,
                "trajectory_match_distance_px": distance,
                "trajectory_match_xy": traj_xy.tolist(),
                "tip_xy": tip_xy.tolist(),
                "seed_selection": "first_detection_near_global_trajectory",
            }
        )
        if distance <= max_distance_px:
            return frame_idx, query_points, candidate
        if closest is None or distance < closest[0]:
            closest = (distance, frame_idx, query_points, candidate)

    if closest is not None:
        distance, frame_idx, query_points, candidate = closest
        candidate["seed_selection"] = "closest_detection_to_global_trajectory_fallback"
        log.warning(
            "No gripper detection within %.1f px of the stored trajectory; using closest detection at frame %d (%.1f px).",
            max_distance_px,
            frame_idx,
            distance,
        )
        return frame_idx, query_points, candidate
    raise RuntimeError("No gripper mask found while matching detections to the stored trajectory.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        stale_by_mode = {
            "existing": [
                "trajectory_only_04_tapnext_tip_tracks.mp4",
                "trajectory_only_05_median_tip_trajectory.mp4",
            ],
            "recompute": [
                "trajectory_only_04_existing_global_trajectory_progress.mp4",
            ],
        }
        for name in stale_by_mode[args.track_mode]:
            path = out_dir / name
            if path.exists():
                path.unlink()

    if not Path(args.yolo_path).exists():
        raise FileNotFoundError(f"YOLO weights not found: {args.yolo_path}")
    if args.track_mode == "recompute" and not Path(args.tapnext_checkpoint).exists():
        raise FileNotFoundError(f"TAPNext++ checkpoint not found: {args.tapnext_checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tapnext_size = (args.tapnext_size, args.tapnext_size)
    task_event_thresholds = parse_task_event_thresholds(args.task_event_threshold)

    log.info("Loading episode %s from %s.", args.episode, args.source_repo_id)
    episode_ds, camera_key, frames_rgb, actions, task_id = load_source_episode(
        args.source_repo_id,
        args.source_root,
        args.episode,
        args.camera_key,
    )
    frame_h, frame_w = frames_rgb.shape[1:3]
    threshold = task_event_thresholds.get(task_id, args.event_threshold) if task_id is not None else args.event_threshold

    log.info("Loading YOLO on %s.", device)
    yolo_model = YOLO(args.yolo_path, verbose=False).to(device)

    rng = np.random.default_rng([args.episode, 2026])

    tracks = None
    tracking_frames = frames_rgb[:: args.frame_stride]
    if args.track_mode == "existing":
        log.info("Using existing %s from the dataset; TAPNext++ is not run.", args.trajectory_key)
        global_trajectory = load_existing_global_trajectory(episode_ds, args.trajectory_key)
        seed_frame_idx, query_points, seed_meta = find_seed_near_global_trajectory(
            yolo_model,
            frames_rgb,
            global_trajectory,
            tip_conf=args.tip_conf,
            frame_stride=args.frame_stride,
            n_points=args.num_tip_points,
            band_quantile=args.tip_band_quantile,
            tip_radius=args.tip_radius,
            sample_mode=args.tip_sample_mode,
            yolo_grayscale=args.yolo_grayscale,
            max_distance_px=args.seed_max_distance_px,
            rng=rng,
        )
        trajectory_xy = global_trajectory[:, :2].copy()
        trajectory_xy[:, 0] *= frame_w
        trajectory_xy[:, 1] *= frame_h
        trajectory_source = "existing_dataset_feature"
        action_meta = {"source": args.trajectory_key}
    else:
        seed_frame_idx, query_points, seed_meta = find_seed(
            yolo_model,
            frames_rgb,
            tip_conf=args.tip_conf,
            frame_stride=args.frame_stride,
            n_points=args.num_tip_points,
            band_quantile=args.tip_band_quantile,
            tip_radius=args.tip_radius,
            sample_mode=args.tip_sample_mode,
            yolo_grayscale=args.yolo_grayscale,
            rng=rng,
        )
        tapnext_model = load_tapnext_model(args.tapnext_checkpoint, device=device, image_size=tapnext_size)
        seed_t = seed_frame_idx // args.frame_stride
        log.info("Seed frame %d; running TAPNext++.", seed_frame_idx)
        if seed_t == 0:
            tracks, visibles = run_tapnext(tapnext_model, tracking_frames, query_points, tapnext_size, device)
        else:
            tracks_bwd, vis_bwd = run_tapnext(
                tapnext_model,
                tracking_frames[seed_t::-1].copy(),
                query_points,
                tapnext_size,
                device,
            )
            tracks_bwd = tracks_bwd[:, ::-1, :].copy()
            vis_bwd = vis_bwd[:, ::-1].copy()
            tracks_fwd, vis_fwd = run_tapnext(
                tapnext_model,
                tracking_frames[seed_t:],
                query_points,
                tapnext_size,
                device,
            )
            tracks = np.concatenate([tracks_bwd[:, :seed_t, :], tracks_fwd], axis=1)
            visibles = np.concatenate([vis_bwd[:, :seed_t], vis_fwd], axis=1)

        trajectory_xy = compute_median_trajectory(tracks, visibles)
        trajectory_source = "tapnext"
        if not trajectory_is_usable(trajectory_xy):
            log.warning("TAPNext++ trajectory is degenerate; using YOLO tip fallback.")
            trajectory_xy, fallback_meta = yolo_tip_trajectory(
                yolo_model,
                frames_rgb,
                conf=args.tip_conf,
                frame_stride=args.frame_stride,
                grayscale=args.yolo_grayscale,
            )
            trajectory_source = fallback_meta["source"]
            seed_meta.update(fallback_meta)

        global_trajectory, action_meta = build_global_trajectory(
            episode_ds,
            actions,
            trajectory_xy,
            (frame_h, frame_w),
            args.frame_stride,
            threshold,
        )

    demo_indices = sampled_indices(len(frames_rgb), args.max_video_frames)
    track_indices = sampled_indices(len(tracking_frames), args.max_video_frames)

    demo_frames = [
        add_title(frames_rgb[i], f"Trajectory-only input demo | episode {args.episode} | frame {i}")
        for i in demo_indices
    ]
    write_video(out_dir / "trajectory_only_01_demo.mp4", demo_frames, args.fps, args.overwrite)

    seg_frames = [
        add_title(
            render_all_robot_masks(
                yolo_model,
                frames_rgb[i],
                conf=args.tip_conf,
                grayscale=args.yolo_grayscale,
            ),
            f"Trajectory-only YOLO robot-part segmentation | frame {i}",
        )
        for i in demo_indices
    ]
    write_video(out_dir / "trajectory_only_02_segmentation.mp4", seg_frames, args.fps, args.overwrite)

    seed_bgr = rgb_to_gray_bgr(frames_rgb[seed_frame_idx]) if args.yolo_grayscale else rgb_to_bgr(frames_rgb[seed_frame_idx])
    seed_results = yolo_model(seed_bgr, conf=args.tip_conf, verbose=False)
    seed_candidate = best_gripper_mask(seed_results, frame_h, frame_w, yolo_model.names)
    if seed_candidate is None:
        raise RuntimeError("Seed mask disappeared when re-rendering the seed frame.")
    seed_traj_t = trajectory_index_for_frame(seed_frame_idx, args.frame_stride, len(global_trajectory))
    seed_frame_with_traj = render_existing_trajectory_progress_frame(
        frames_rgb[seed_frame_idx],
        global_trajectory,
        seed_traj_t,
        line_width=args.line_width,
        point_radius=args.point_radius,
    )
    tip_selection = render_gripper_mask_and_tip_points(
        seed_frame_with_traj,
        seed_candidate["mask"],
        query_points,
        band_quantile=args.tip_band_quantile,
        tip_radius=args.tip_radius,
        point_radius=args.point_radius,
    )
    save_png(
        out_dir / "trajectory_only_03_gripper_tip_selection.png",
        add_title(tip_selection, "Trajectory-only gripper mask filtered to tip-local TAPNext++ query points"),
        args.overwrite,
    )

    if tracks is not None:
        track_video = [
            add_title(
                render_tracks_frame(
                    tracking_frames[t],
                    tracks,
                    t,
                    line_width=args.track_line_width,
                    point_radius=max(2, args.point_radius - 1),
                ),
                f"Trajectory-only TAPNext++ tip-point tracks | tracking frame {t}",
            )
            for t in track_indices
            if t < tracks.shape[1]
        ]
        write_video(out_dir / "trajectory_only_04_tapnext_tip_tracks.mp4", track_video, args.fps, args.overwrite)

        trajectory_progress_video = [
            add_title(
                render_median_frame(
                    tracking_frames[t],
                    tracks,
                    trajectory_xy,
                    t,
                    line_width=args.line_width,
                    point_radius=args.point_radius,
                ),
                f"Trajectory-only point tracks collapse to median tip trajectory | tracking frame {t}",
            )
            for t in track_indices
            if t < len(trajectory_xy)
        ]
        write_video(
            out_dir / "trajectory_only_05_median_tip_trajectory.mp4",
            trajectory_progress_video,
            args.fps,
            args.overwrite,
        )
    else:
        existing_indices = sampled_indices(len(global_trajectory), args.max_video_frames)
        trajectory_progress_video = [
            add_title(
                render_existing_trajectory_progress_frame(
                    frames_rgb[min(i * args.frame_stride, len(frames_rgb) - 1)],
                    global_trajectory,
                    i,
                    line_width=args.line_width,
                    point_radius=args.point_radius,
                ),
                f"Trajectory-only existing observation.global_trajectory | trajectory step {i}",
            )
            for i in existing_indices
        ]
        write_video(
            out_dir / "trajectory_only_04_existing_global_trajectory_progress.mp4",
            trajectory_progress_video,
            args.fps,
            args.overwrite,
        )

    blank = render_global_trajectory_blank(
        global_trajectory,
        (frame_h, frame_w),
        line_width=args.line_width,
    )
    overlay = render_debug_overlay(frames_rgb[0], blank, alpha=0.78)
    save_png(
        out_dir / "trajectory_only_06_global_trajectory_blank.png",
        add_title(blank, "Trajectory-only observation.global_trajectory | uniform global path"),
        args.overwrite,
    )
    save_png(
        out_dir / "trajectory_only_07_global_trajectory_overlay.png",
        add_title(overlay, "Trajectory-only final global trajectory over first frame"),
        args.overwrite,
    )

    combined_frames: list[np.ndarray] = []
    continuous_indices = sampled_indices(len(frames_rgb), args.max_video_frames)
    pre_seed = [int(i) for i in continuous_indices if i <= seed_frame_idx]
    if not pre_seed or pre_seed[-1] != seed_frame_idx:
        pre_seed.append(seed_frame_idx)
    pre_seed = sorted(set(pre_seed))
    post_seed = [int(i) for i in continuous_indices if i >= seed_frame_idx]
    if not post_seed or post_seed[0] != seed_frame_idx:
        post_seed.insert(0, seed_frame_idx)
    post_seed = sorted(set(post_seed))

    for frame_idx in pre_seed:
        traj_t = trajectory_index_for_frame(frame_idx, args.frame_stride, len(global_trajectory))
        frame_with_traj = render_existing_trajectory_progress_frame(
            frames_rgb[frame_idx],
            global_trajectory,
            traj_t,
            line_width=args.line_width,
            point_radius=args.point_radius,
        )
        combined_frames.append(
            side_panel_frame(
                render_all_robot_masks(
                    yolo_model,
                    frame_with_traj,
                    conf=args.tip_conf,
                    grayscale=args.yolo_grayscale,
                    detect_frame_rgb=frames_rgb[frame_idx],
                ),
                "Mask detection",
                "Text placeholder.",
            )
        )

    combined_frames.extend(
        repeat_side_panel(
            render_all_robot_masks(
                yolo_model,
                seed_frame_with_traj,
                conf=args.tip_conf,
                grayscale=args.yolo_grayscale,
                detect_frame_rgb=frames_rgb[seed_frame_idx],
            ),
            "Mask detection",
            "Text placeholder.",
            args.fps,
            1.3,
        )
    )

    combined_frames.extend(
        repeat_side_panel(
            tip_selection,
            "Gripper filtering",
            "Text placeholder.",
            args.fps,
            1.6,
        )
    )

    for frame_idx in post_seed:
        traj_t = trajectory_index_for_frame(frame_idx, args.frame_stride, len(global_trajectory))
        combined_frames.append(
            side_panel_frame(
                render_existing_trajectory_progress_frame(
                    frames_rgb[frame_idx],
                    global_trajectory,
                    traj_t,
                    line_width=args.line_width,
                    point_radius=args.point_radius,
                ),
                "Global trajectory",
                "Text placeholder.",
            )
        )

    combined_frames.extend(
        repeat_side_panel(
            render_existing_trajectory_progress_frame(
                frames_rgb[-1],
                global_trajectory,
                trajectory_index_for_frame(len(frames_rgb) - 1, args.frame_stride, len(global_trajectory)),
                line_width=args.line_width,
                point_radius=args.point_radius,
            ),
            "Global trajectory",
            "Text placeholder.",
            args.fps,
            1.6,
        )
    )
    write_video(
        out_dir / "trajectory_only_generation_walkthrough.mp4",
        combined_frames,
        args.fps,
        args.overwrite,
    )
    write_video(
        out_dir / "trajectory_only_generation_explainer.mp4",
        combined_frames,
        args.fps,
        args.overwrite,
    )

    meta = {
        "source_repo_id": args.source_repo_id,
        "source_root": args.source_root,
        "episode": args.episode,
        "camera_key": camera_key,
        "task_id": task_id,
        "frame_count": int(len(frames_rgb)),
        "frame_shape": [int(frame_h), int(frame_w), 3],
        "frame_stride": args.frame_stride,
        "track_mode": args.track_mode,
        "seed_frame_idx": seed_frame_idx,
        "seed_meta": {
            key: (float(value) if isinstance(value, np.floating) else int(value) if isinstance(value, np.integer) else value)
            for key, value in seed_meta.items()
            if key != "mask"
        },
        "trajectory_source": trajectory_source,
        "trajectory_shape": list(global_trajectory.shape),
        "action_meta": action_meta,
        "artifacts": sorted(path.name for path in out_dir.iterdir() if path.is_file()),
    }
    (out_dir / "trajectory_only_generation_metadata.json").write_text(json.dumps(meta, indent=2))
    log.info("Wrote trajectory-only walkthrough artifacts to %s.", out_dir)


if __name__ == "__main__":
    main()
