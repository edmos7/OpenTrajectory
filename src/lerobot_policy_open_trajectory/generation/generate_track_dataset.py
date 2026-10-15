#!/usr/bin/env python3
"""
generate_track_dataset.py
--------------------------
Generate a track dataset for training a Track Transformer and add it as new
features to a LeRobot dataset.

For each episode this script:

  1. Samples main-camera query points in two groups:
       - ``--num-robot-points // 2`` robot points from YOLO detection
         (``so101_gripper`` + ``so101_wrist`` classes) — tracked once in a
         single TAPNext++ pass for the full episode.
       - ``--num-robot-points // 2`` grid points sampled **fresh at every
         trajectory timestep t** from the central 75 % crop of frame *t*,
         then tracked forward for exactly H frames via a per-timestep
         TAPNext++ call.  This ensures grid tracks at timestep t represent
         objects visible at t, not objects that happened to be at those
         grid positions at t=0.
     The per-episode global median trajectory is computed from the robot
     (YOLO) points only.
  2. When building dataset rows, robot tracks are the **same for every
     timestep** of the episode; grid tracks are **freshly tracked from
     each timestep** and require no further sub-sampling.  The final
     per-row feature shape is ``[--num-robot-points, 1+H, 2]``.
  3. For the secondary camera (optional): samples ``--num-sec-points`` fresh
     random grid points at every trajectory timestep and tracks them forward
     for exactly H frames.
  4. Normalises all 2-D coordinates to [0, 1] (pixel space → normalised form).
  5. Builds per-timestep features:
       observation.global_trajectory        [max_T, 3]            - padded median + action
       observation.main_cam_point_tracks    [N_main, 1+H, 2]      - current + H future
       observation.sec_cam_point_tracks     [N_sec,  1+H, 2]      - current + H future
  6. Writes a new LeRobot dataset that contains all original features plus the
     three new ones, using ``lerobot.datasets.dataset_tools.add_features``.
  7. Optionally pushes the new dataset to the Hugging Face Hub.

Intermediate per-episode results are saved to a temp directory as ``.npz``
files so the entire dataset never needs to be in memory at once.  The temp
files are deleted automatically after the new dataset is written (unless
``--keep-temp`` is set).

Usage
-----
    python generate_track_dataset.py \\
        --source-repo-id edmos7/pickplace_20 \\
        --main-cam-key   observation.images.MAIN_CAM \\
        --output-repo-id edmos7/pickplace_20-tracks  \\
        [--no-sec-cam] [--no-push]

Added dataset features
-----------------------
    observation.global_trajectory        float32  [max_T, 3]
        Per-episode median trajectory of YOLO gripper query points in the main
        camera, normalised to [0, 1], with a third channel encoding the
        min-max-normalised gripper action signal.  Layout per row: (x, y, action).
        Padded with the last row to a fixed length of max_T (the longest episode
        in the dataset).  Same value broadcast to every frame of the same episode.

    observation.main_cam_point_tracks    float32  [N_main, 1+H, 2]
        N_main = num_robot_pts.  First N_main // 2 rows are robot points (same
        across all timesteps of the episode); remaining rows are grid tracks
        freshly sampled and tracked from each individual timestep.  Tracks are
        normalised to [0, 1].  At the end of an episode the last known position
        is repeated.

    observation.sec_cam_point_tracks     float32  [N_sec, 1+H, 2]
        N_sec fresh random grid points sampled from each secondary-camera
        timestep and tracked for the current plus H future frames. Tracks near
        the episode end repeat the terminal frame. Only added when
        --sec-cam-key is provided.
"""

from __future__ import annotations

import argparse
import logging
import math
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot_policy_open_trajectory.generation.preprocessing import (
    add_release_trim_args,
    release_trim_generated_dataset,
)
from lerobot_policy_open_trajectory.generation.utils.events import get_frame_at_sharpest_drop
from lerobot_policy_open_trajectory.generation.utils.preprocessing import preprocess_frame
from lerobot_policy_open_trajectory.generation.utils.segmentation import (
    get_category_center_by_priority,
    sample_points_from_class,
)
from lerobot_policy_open_trajectory.generation.utils.tracking import (
    compute_median_trajectory,
    load_tapnext_model,
    run_tapnext,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# Robot-class configuration
ROBOT_CLASSES = ["so101_gripper", "so101_wrist", "so101_upper_arm"]
MEDIAN_CLASSES = frozenset(["so101_gripper"])
# Confidence threshold for YOLO robot detection — must match the probe threshold
# used in detect_occlusion_center (get_category_center_by_priority default = 0.5)
ROBOT_YOLO_CONF = 0.75

# How many strided frames to scan for a gripper detection before giving up
N_SCAN_FRAMES = 150


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate track-transformer features and add them to a LeRobot dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset
    p.add_argument(
        "--source-repo-id", required=True, help="Source LeRobot dataset repo id, e.g. edmos7/pickplace_20."
    )
    p.add_argument(
        "--output-repo-id",
        default=None,
        help="Repo id for the new dataset (default: <source_repo_id>-tracks).",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Local directory for the new dataset (default: $HF_LEROBOT_HOME/<output-repo-id>).",
    )
    p.add_argument(
        "--main-cam-key", default="observation.images.MAIN_CAM", help="Main-camera observation key."
    )
    p.add_argument(
        "--sec-cam-key", default=None, help="Secondary-camera key. Auto-detected from the dataset by default."
    )
    p.add_argument(
        "--no-sec-cam",
        action="store_true",
        help="Disable secondary-camera tracking even if a second camera is found.",
    )
    # Models
    p.add_argument("--model-path", required=True, help="Path to YOLO segmentation model weights.")
    p.add_argument("--tapnext-checkpoint", required=True, help="Path to TAPNext++ checkpoint (.pt).")
    # Tracking parameters
    p.add_argument(
        "--num-robot-points",
        type=int,
        default=16,
        help="Total query points per row from the main camera "
        "(half from YOLO robot classes, half from the grid pool).",
    )
    p.add_argument(
        "--grid-pool-multiplier",
        type=int,
        default=4,
        help="Grid-pool size = multiplier x (num-robot-points // 2). "
        "More pool points → more per-timestep diversity. Must be ≥ 1.",
    )
    p.add_argument(
        "--num-sec-points", type=int, default=16, help="Secondary-camera query points (grid-sampled)."
    )
    p.add_argument(
        "--horizon",
        type=int,
        default=32,  # change to 64
        help="Future-position horizon H.",
    )
    p.add_argument(
        "--tapir-target-size", type=int, default=256, help="Frame size (H=W) for TAPNext++ inference."
    )
    p.add_argument("--frame-stride", type=int, default=1, help="Feed every Nth frame to TAPNext++.")
    # Run control
    p.add_argument("--episodes", default=None, help="Subset to process, e.g. '0-9,15'. All by default.")
    p.add_argument("--resume", action="store_true", help="Skip episodes whose temp .npz already exists.")
    p.add_argument(
        "--keep-temp",
        action="store_true",
        help="Do not delete the temp .npz directory after writing the dataset.",
    )
    p.add_argument(
        "--temp-dir",
        default=None,
        help="Directory for intermediate .npz files (default: a new temp dir inside --output-dir).",
    )
    # Push
    p.add_argument(
        "--push",
        dest="no_push",
        action="store_false",
        help="Explicitly upload the generated dataset to the Hub.",
    )
    p.add_argument(
        "--no-push",
        action="store_true",
        default=True,
        help="Build the dataset locally without pushing to the Hub.",
    )
    p.add_argument("--public", action="store_true", help="Push the new dataset as public (default: private).")
    p.add_argument(
        "--single-episode-output",
        action="store_true",
        help="Output dataset contains only the episodes in --episodes "
        "(instead of the full source dataset). "
        "Auto-enabled when --episodes resolves to exactly one episode.",
    )
    p.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable early-stop truncation (don't cut trajectory at return-to-start).",
    )
    p.add_argument(
        "--overlay-dir",
        default=None,
        help="Directory to save per-episode trajectory overlay PNGs. "
        "Defaults to data_generation/overlays/<output-repo-id>. "
        "Pass '--no-overlays' to disable.",
    )
    p.add_argument("--no-overlays", action="store_true", help="Disable saving trajectory overlay images.")
    p.add_argument(
        "--no-global-trajectory",
        action="store_true",
        default=True,
        help="Do not add observation.global_trajectory to the output dataset.",
    )
    add_release_trim_args(p)
    p.add_argument("--artifact-dir", help="Also export portable per-episode annotations.")
    return p.parse_args(argv)


def parse_episode_list(spec: str, max_ep: int) -> list[int]:
    indices: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            indices.update(range(int(lo), int(hi) + 1))
        else:
            indices.add(int(part))
    return sorted(i for i in indices if 0 <= i < max_ep)


# ── Point sampling ─────────────────────────────────────────────────────────────


def _covering_grid_dims(n_points: int) -> tuple[int, int]:
    """Smallest RxC grid with R*C >= n_points, R and C close to sqrt(n_points)."""
    cols = math.ceil(math.sqrt(n_points))
    rows = math.ceil(n_points / cols)
    return rows, cols


def sample_grid_points(
    frame_h: int,
    frame_w: int,
    n_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample *n_points* query points from a covering grid.

    Constructs the smallest RxC grid covering *n_points* cells, randomly
    selects *n_points* cells (without replacement), then draws one uniformly
    random point from each selected cell.

    Returns ``[n_points, 3]`` float32 rows ``(t=0, y, x)`` in pixel space.
    """
    rows, cols = _covering_grid_dims(n_points)
    selected = rng.choice(rows * cols, size=n_points, replace=False)
    cell_h, cell_w = frame_h / rows, frame_w / cols

    pts: list[list[float]] = []
    for idx in selected:
        r, c = divmod(int(idx), cols)
        y = float(np.clip(rng.uniform(r * cell_h, (r + 1) * cell_h), 0, frame_h - 1))
        x = float(np.clip(rng.uniform(c * cell_w, (c + 1) * cell_w), 0, frame_w - 1))
        pts.append([0.0, y, x])
    return np.array(pts, dtype=np.float32)


def sample_grid_points_cropped(
    frame_h: int,
    frame_w: int,
    n_points: int,
    rng: np.random.Generator,
    crop_frac: float = 0.75,
) -> np.ndarray:
    """Sample *n_points* from a covering grid on the central *crop_frac* region.

    Constructs the bounding box ``[y0, y1) x [x0, x1)`` centred in the frame:

        y0 = margin * H,  y1 = (1 - margin) * H
        x0 = margin * W,  x1 = (1 - margin) * W
        margin = (1 - crop_frac) / 2

    For ``crop_frac=0.75`` this keeps the inner 75 % of the frame (outer 12.5 %
    margin on each side).  A covering RxC grid is laid over this region,
    *n_points* cells are selected at random without replacement, and one
    uniformly random point is drawn from each selected cell.

    Returns ``[n_points, 3]`` float32 ``(t=0, y, x)`` in pixel space.
    """
    margin = (1.0 - crop_frac) / 2.0
    y0 = margin * frame_h
    y1 = (1.0 - margin) * frame_h
    x0 = margin * frame_w
    x1 = (1.0 - margin) * frame_w
    crop_h = y1 - y0
    crop_w = x1 - x0

    rows, cols = _covering_grid_dims(n_points)
    selected = rng.choice(rows * cols, size=n_points, replace=False)
    cell_h = crop_h / rows
    cell_w = crop_w / cols

    pts: list[list[float]] = []
    for idx in selected:
        r, c = divmod(int(idx), cols)
        y = float(np.clip(rng.uniform(y0 + r * cell_h, y0 + (r + 1) * cell_h), 0, frame_h - 1))
        x = float(np.clip(rng.uniform(x0 + c * cell_w, x0 + (c + 1) * cell_w), 0, frame_w - 1))
        pts.append([0.0, y, x])
    return np.array(pts, dtype=np.float32)


def _fallback_cluster(
    cx: float,
    cy: float,
    n: int,
    frame_h: int,
    frame_w: int,
    rng: np.random.Generator,
    spread_fraction: float = 0.02,
) -> np.ndarray:
    """Gaussian cluster around *(cx, cy)* at t=0 — used when YOLO has no detection."""
    std = max(frame_h, frame_w) * spread_fraction
    ys = np.clip(rng.normal(cy, std, n), 0, frame_h - 1).astype(np.float32)
    xs = np.clip(rng.normal(cx, std, n), 0, frame_w - 1).astype(np.float32)
    return np.stack([np.zeros(n, dtype=np.float32), ys, xs], axis=1)


def _cluster_around(
    center_xy: np.ndarray,
    n: int,
    frame_h: int,
    frame_w: int,
    spread: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample *n* points in a Gaussian cluster around *center_xy = (x, y)* at t=0."""
    cx, cy = float(center_xy[0]), float(center_xy[1])
    ys = np.clip(rng.normal(cy, spread, n), 0, frame_h - 1).astype(np.float32)
    xs = np.clip(rng.normal(cx, spread, n), 0, frame_w - 1).astype(np.float32)
    return np.stack([np.zeros(n, dtype=np.float32), ys, xs], axis=1)


def detect_robot_points(
    model,
    frame_bgr: np.ndarray,
    n_total: int,
    classes: list[str],
    median_classes: frozenset,
    rng: np.random.Generator,
    conf: float = ROBOT_YOLO_CONF,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample query points from multiple robot YOLO classes.

    Points are distributed equally across *classes* initially.  Any class that
    is not detected at *conf* threshold has its point budget redistributed
    evenly to the detected classes (no fallback cluster).  Detected classes are
    re-sampled with their enlarged budget.

    *median_classes* points are only marked True in the returned mask when
    that class was actually detected (not redistributed-to).

    Returns
    -------
    qp           [n_total, 3] float32  ``(t=0, y, x)`` in original pixel space
    median_mask  [n_total]    bool     True for points from a detected median class
    """
    n_cls = len(classes)
    base = n_total // n_cls
    extras = n_total - base * n_cls
    init_counts = [base + (1 if i < extras else 0) for i in range(n_cls)]

    # ── First pass: find which classes are detectable ──────────────────────────
    detected_classes: list[str] = []
    undetected_budget = 0
    for cls_name, n_pts in zip(classes, init_counts, strict=False):
        probe = sample_points_from_class(model, frame_bgr, cls_name, 1, rng, conf=conf)
        if probe is not None:
            detected_classes.append(cls_name)
        else:
            undetected_budget += n_pts
            logger.info(
                "    [YOLO] '%s': NOT detected (conf≥%.2f) — %d pts redistributed.",
                cls_name,
                conf,
                n_pts,
            )

    if not detected_classes:
        raise ValueError("No robot class detected; refusing to label arbitrary image points as robot tracks")

    # ── Redistribute undetected budget evenly to detected classes ──────────────
    n_det = len(detected_classes)
    budgets: dict[str, int] = {}
    for cls_name in detected_classes:
        idx = classes.index(cls_name)
        budgets[cls_name] = init_counts[idx]
    if undetected_budget > 0:
        extra_base = undetected_budget // n_det
        extra_extras = undetected_budget - extra_base * n_det
        for i, cls_name in enumerate(detected_classes):
            budgets[cls_name] += extra_base + (1 if i < extra_extras else 0)

    # ── Second pass: sample full budgets from detected classes ─────────────────
    all_qp: list[np.ndarray] = []
    all_mask: list[np.ndarray] = []
    for cls_name in detected_classes:
        n_pts = budgets[cls_name]
        pts = sample_points_from_class(model, frame_bgr, cls_name, n_pts, rng, conf=conf)
        if pts is None:
            # Race between probe and sample — extremely unlikely; skip silently
            logger.warning("    [YOLO] '%s': vanished between probe and sample, skipping.", cls_name)
            continue
        logger.info(
            "    [YOLO] '%s': detected (conf≥%.2f) — %d points sampled.",
            cls_name,
            conf,
            n_pts,
        )
        all_qp.append(pts)
        all_mask.append(np.full(n_pts, cls_name in median_classes, dtype=bool))

    return np.concatenate(all_qp, axis=0), np.concatenate(all_mask, axis=0)


def sample_task_points_per_grasp(
    median_traj: np.ndarray,
    grasp_frames: list[int],
    frame_stride: int,
    n_total: int,
    frame_h: int,
    frame_w: int,
    rng: np.random.Generator,
    spread_fraction: float = 0.08,
) -> list[tuple[int, np.ndarray]]:
    """Distribute *n_total* task points across all grasp events.

    Returns a list of ``(gi_traj, qp_i)`` where *gi_traj* is the grasp frame
    index in trajectory time (post-striding) and *qp_i* is ``[n_i, 3]``
    float32 ``(t=0, y, x)`` in pixel space.

    If no grasps are detected, returns a single group centred at the trajectory
    midpoint.
    """
    spread = max(frame_h, frame_w) * spread_fraction
    T_traj = len(median_traj)

    if not grasp_frames:
        gi_traj = T_traj // 2
        center = median_traj[gi_traj]
        return [(gi_traj, _cluster_around(center, n_total, frame_h, frame_w, spread, rng))]

    n_g = len(grasp_frames)
    base = n_total // n_g
    extras = n_total - base * n_g
    counts = [base + (1 if i < extras else 0) for i in range(n_g)]

    result: list[tuple[int, np.ndarray]] = []
    for g_frame, n_g_pts in zip(grasp_frames, counts, strict=False):
        gi_traj = min(g_frame // max(frame_stride, 1), T_traj - 1)
        center = median_traj[gi_traj]
        result.append((gi_traj, _cluster_around(center, n_g_pts, frame_h, frame_w, spread, rng)))
    return result


# ── Track helpers ──────────────────────────────────────────────────────────────


def normalize_tracks(tracks: np.ndarray, h: int, w: int) -> np.ndarray:
    """Normalise ``[N, T, 2]`` (x, y) pixel tracks → [0, 1] and clamp."""
    norm = tracks.copy().astype(np.float32)
    norm[..., 0] = np.clip(norm[..., 0] / w, 0.0, 1.0)
    norm[..., 1] = np.clip(norm[..., 1] / h, 0.0, 1.0)
    return norm


def build_point_tracks(tracks_norm: np.ndarray, horizon: int) -> np.ndarray:
    """Build ``[N, T, 1+H, 2]`` windows from ``[N, T, 2]`` normalised tracks.

    Window at timestep *t*: ``[pos(t), pos(t+1), …, pos(t+H)]``.
    Past the episode end the last known position is repeated.
    """
    N, T, _ = tracks_norm.shape
    out = np.empty((N, T, 1 + horizon, 2), dtype=np.float32)
    for h in range(1 + horizon):
        t_idx = np.minimum(np.arange(T) + h, T - 1)
        out[:, :, h, :] = tracks_norm[:, t_idx, :]
    return out


# ── Occlusion-center detection ─────────────────────────────────────────────────


def detect_occlusion_center(
    dataset,
    yolo_model,
    cam_key: str,
    n_probe: int = 10,
) -> tuple[float, float]:
    """Median YOLO-detected gripper centroid over the first *n_probe* episodes."""
    ep_starts = list(dataset.meta.episodes["dataset_from_index"])
    detections: list[tuple[float, float]] = []
    for ep_idx in range(min(n_probe, len(ep_starts))):
        bgr = preprocess_frame(dataset[ep_starts[ep_idx]][cam_key])
        pt = get_category_center_by_priority(yolo_model, bgr, conf=0.5)
        if pt is not None:
            detections.append(pt)
            logger.info("  Probe ep %03d: detection %s", ep_idx, pt)
        else:
            logger.warning("  Probe ep %03d: no YOLO detection.", ep_idx)

    if detections:
        cx = float(np.median([d[0] for d in detections]))
        cy = float(np.median([d[1] for d in detections]))
        logger.info("Occlusion center: (%.1f, %.1f) from %d detections.", cx, cy, len(detections))
        return cx, cy

    h, w = dataset[ep_starts[0]][cam_key].shape[1:]
    logger.warning("No YOLO detections; using image centre.")
    return w / 2.0, h / 2.0


# ── Trajectory overlay renderer ───────────────────────────────────────────────


def _render_traj_overlay(
    first_bgr: np.ndarray,
    median_traj: np.ndarray,
    action_norm: np.ndarray,
    frame_stride: int,
    out_path: Path,
    line_width: int = 6,
    alpha: float = 0.75,
) -> None:
    """Draw the 2-D trajectory onto a black canvas with time→red / action→blue
    color grading, blend it over *first_bgr*, and save to *out_path*.

    Args:
        first_bgr:    HW3 BGR uint8 first frame of the episode.
        median_traj:  [T, 2] float array of (x, y) pixel positions.
        action_norm:  [T] float32 array of normalised gripper values in [0, 1].
        frame_stride: Stride used when building *median_traj* (for time encoding).
        out_path:     Destination PNG path.
        line_width:   Stroke width in pixels.
        alpha:        Trajectory opacity blended over the frame (0-1).
    """
    frame_h, frame_w = first_bgr.shape[:2]
    traj_img = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    line_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)

    T = len(median_traj)
    # Total full-rate frames represented (for accurate time colour)
    time_denom = max(T * frame_stride - 1, 1)

    for i in range(T - 1):
        pt1 = (
            int(np.clip(median_traj[i, 0], 0, frame_w - 1)),
            int(np.clip(median_traj[i, 1], 0, frame_h - 1)),
        )
        pt2 = (
            int(np.clip(median_traj[i + 1, 0], 0, frame_w - 1)),
            int(np.clip(median_traj[i + 1, 1], 0, frame_h - 1)),
        )

        # Subdivide strided segment so colour matches sub-frame granularity
        for s in range(frame_stride):
            alpha_a = s / frame_stride
            alpha_b = (s + 1) / frame_stride
            sp1 = (
                int(np.clip(pt1[0] + alpha_a * (pt2[0] - pt1[0]), 0, frame_w - 1)),
                int(np.clip(pt1[1] + alpha_a * (pt2[1] - pt1[1]), 0, frame_h - 1)),
            )
            sp2 = (
                int(np.clip(pt1[0] + alpha_b * (pt2[0] - pt1[0]), 0, frame_w - 1)),
                int(np.clip(pt1[1] + alpha_b * (pt2[1] - pt1[1]), 0, frame_h - 1)),
            )

            full_frame = i * frame_stride + s
            t = full_frame / time_denom
            red_val = int(np.clip(t * 255, 0, 255))
            blue_val = int(np.clip(action_norm[min(i, len(action_norm) - 1)] * 255, 0, 255))

            line_mask[:] = 0
            cv2.line(line_mask, sp1, sp2, 255, line_width)
            traj_img[line_mask > 0, 0] = red_val
            traj_img[line_mask > 0, 2] = blue_val

    # Convert first frame to RGB for blending
    frame_rgb = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    traj_f = traj_img.astype(np.float32)
    weight = (traj_f.max(axis=2, keepdims=True) / 255.0) * alpha
    overlay = (frame_rgb * (1.0 - weight) + traj_f * weight).clip(0, 255).astype(np.uint8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


# ── Per-episode computation ────────────────────────────────────────────────────


def process_episode(
    episode_id: int,
    yolo_model,
    tapnext_model,
    repo_id: str,
    main_cam_key: str,
    sec_cam_key: str | None,
    occlusion_center: tuple[float, float],
    horizon: int,
    num_robot_pts: int,
    num_sec_pts: int,
    tapir_target_size: tuple[int, int],
    frame_stride: int,
    device: str,
    out_path: Path,
    early_stop: bool = True,
    overlay_dir: Path | None = None,
) -> bool:
    """Process one episode and save results to *out_path* as a compressed NPZ.

    Returns True on success, False on failure.

    Main-camera query points:
    - ``n_robot_seed = num_robot_pts // 2`` robot points from YOLO detection
      (distributed across ROBOT_CLASSES) — tracked in a single pass for the
      full episode, then windowed into ``[1+H]``-step slices.
    - ``n_grid_per_row = num_robot_pts - n_robot_seed`` fresh grid points sampled
      independently at every trajectory timestep *t* from the central 75 % crop
      of frame *t*, then tracked forward for exactly *H* frames via a dedicated
      per-timestep TAPNext++ call.

    The global median trajectory is computed from the YOLO (robot seed) points
    only and stored with a third channel encoding the normalised gripper action.

    Saved arrays
    ------------
    global_trajectory    [T_traj, 3]                          normalised (x, y, action)
    robot_tracks         [n_robot_seed, T_traj, 1+H, 2]       same across all timesteps
    grid_tracks          [T_traj, n_grid_per_row, 1+H, 2]     fresh per-timestep tracks
    sec_cam_point_tracks [T_traj, N_sec, 1+H, 2]              (if sec cam)
    n_robot_seed_pts     scalar int
    n_grid_per_row_pts   scalar int
    frame_stride         scalar int
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, episodes=[episode_id])
    n_frames = len(dataset)
    if n_frames == 0:
        logger.warning("Episode %d: no frames — skipped.", episode_id)
        return False

    tgt_H, tgt_W = tapir_target_size
    rng_main = np.random.default_rng(42)

    # ── Query point counts ─────────────────────────────────────────────────────
    n_robot_seed = max(1, num_robot_pts // 2)
    n_grid_per_row = num_robot_pts - n_robot_seed

    # ── Find gripper action dimension ──────────────────────────────────────────
    grip_idx: int | None = None
    for candidate in ("so101_gripper", "gripper"):
        for i, name in enumerate(dataset.meta.names.get("action", [])):
            if candidate.lower() in name.lower():
                grip_idx = i
                break
        if grip_idx is not None:
            break

    # ── Single-pass frame + action loading ────────────────────────────────────
    main_frames_list: list[np.ndarray] = []
    sec_frames_list: list[np.ndarray] = []
    actions_list: list[torch.Tensor] = []
    scan_bgrs: list[np.ndarray] = []  # original-size BGRs for YOLO scan
    frame_h = frame_w = None
    sec_fh = sec_fw = None
    first_bgr_orig: np.ndarray | None = None

    for i, entry in enumerate(dataset):
        actions_list.append(entry["action"])
        if i % frame_stride != 0:
            continue

        bgr = preprocess_frame(entry[main_cam_key])
        if frame_h is None:
            frame_h, frame_w = bgr.shape[:2]
            first_bgr_orig = bgr
        if len(scan_bgrs) < N_SCAN_FRAMES:
            scan_bgrs.append(bgr)
        main_frames_list.append(cv2.cvtColor(cv2.resize(bgr, (tgt_W, tgt_H)), cv2.COLOR_BGR2RGB))

        if sec_cam_key is not None:
            bgr_s = preprocess_frame(entry[sec_cam_key])
            if sec_fh is None:
                sec_fh, sec_fw = bgr_s.shape[:2]
            sec_frames_list.append(cv2.cvtColor(cv2.resize(bgr_s, (tgt_W, tgt_H)), cv2.COLOR_BGR2RGB))

    main_frames = np.stack(main_frames_list, axis=0)
    T_traj = main_frames.shape[0]
    del main_frames_list
    if not sec_frames_list:
        del sec_frames_list
    actions = torch.stack(actions_list)

    def _scale_qp(qp_orig, fh, fw):
        qp = qp_orig.copy()
        qp[:, 1] *= tgt_H / fh
        qp[:, 2] *= tgt_W / fw
        return qp

    def _scale_tracks(tr, fh, fw):
        tr = tr.copy()
        tr[:, :, 0] *= fw / tgt_W
        tr[:, :, 1] *= fh / tgt_H
        return tr

    # ── Scan for first gripper detection (up to N_SCAN_FRAMES strided frames) ────
    # Query points are sampled from the first frame where a gripper is detected so
    # that fallback-cluster points don't pollute the global median trajectory.
    t_detect = 0
    first_detect_bgr = scan_bgrs[0]
    gripper_scan_succeeded = False
    logger.info(
        "Episode %d — scanning up to %d strided frames for gripper detection.",
        episode_id,
        len(scan_bgrs),
    )
    for scan_t, bgr_s in enumerate(scan_bgrs):
        gripper_found = any(
            sample_points_from_class(
                yolo_model, bgr_s, cls, 1, np.random.default_rng(0), conf=ROBOT_YOLO_CONF
            )
            is not None
            for cls in MEDIAN_CLASSES
        )
        logger.info(
            "  scan frame %d/%d: gripper %s",
            scan_t,
            len(scan_bgrs) - 1,
            "DETECTED ✓" if gripper_found else "not detected",
        )
        if gripper_found:
            t_detect = scan_t
            first_detect_bgr = bgr_s
            gripper_scan_succeeded = True
            break
    else:
        logger.warning(
            "Episode %d: gripper not detected in any of %d scanned frames — fallback cluster will be used.",
            episode_id,
            len(scan_bgrs),
        )

    if t_detect > 0:
        logger.info(
            "Episode %d: gripper first detected at strided frame %d → "
            "backward pass [0→%d] + forward pass [%d→end].",
            episode_id,
            t_detect,
            t_detect,
            t_detect,
        )
    elif gripper_scan_succeeded:
        logger.info("Episode %d: gripper detected at frame 0 → single forward pass.", episode_id)
    else:
        logger.info(
            "Episode %d: gripper not detected in scan — using frame 0 fallback, single forward pass.",
            episode_id,
        )

    # ── Sample robot query points (YOLO at t_detect) ───────────────────────────
    logger.info(
        "Episode %d — sampling %d robot main-cam query points.",
        episode_id,
        n_robot_seed,
    )
    qp_robot_orig, median_mask = detect_robot_points(
        yolo_model,
        first_detect_bgr,
        n_robot_seed,
        ROBOT_CLASSES,
        MEDIAN_CLASSES,
        rng_main,
    )
    median_mask_all = median_mask
    n_gripper_pts = int(median_mask_all.sum())
    logger.info(
        "Episode %d: %d/%d robot-seed points will contribute to the median trajectory "
        "(gripper-detected only).",
        episode_id,
        n_gripper_pts,
        n_robot_seed,
    )

    # ── Robot tracking pass(es) — full episode ────────────────────────────────
    qp_scaled = _scale_qp(qp_robot_orig, frame_h, frame_w)

    if t_detect == 0:
        logger.info(
            "Episode %d — robot single forward pass: %d points.",
            episode_id,
            n_robot_seed,
        )
        tr_main, vi_main = run_tapnext(
            tapnext_model,
            main_frames,
            qp_scaled,
            tapir_target_size,
            device,
        )
    else:
        # Backward pass covers frames 0..t_detect (reversed), forward covers t_detect..end.
        logger.info(
            "Episode %d — robot backward pass [0→%d] + forward pass [%d→end], %d points.",
            episode_id,
            t_detect,
            t_detect,
            n_robot_seed,
        )
        tr_bwd, vi_bwd = run_tapnext(
            tapnext_model,
            main_frames[t_detect::-1].copy(),
            qp_scaled,
            tapir_target_size,
            device,
        )  # [N, t_detect+1, 2/bool]; index 0 = original frame t_detect
        # Reverse temporal axis so index 0 = original frame 0
        tr_bwd = tr_bwd[:, ::-1, :].copy()
        vi_bwd = vi_bwd[:, ::-1].copy()

        tr_fwd, vi_fwd = run_tapnext(
            tapnext_model,
            main_frames[t_detect:],
            qp_scaled,
            tapir_target_size,
            device,
        )  # [N, T-t_detect, 2/bool]; index 0 = original frame t_detect

        # Concatenate: use backward for frames 0..t_detect-1, forward for t_detect..T-1
        tr_main = np.concatenate([tr_bwd[:, :t_detect, :], tr_fwd], axis=1)
        vi_main = np.concatenate([vi_bwd[:, :t_detect], vi_fwd], axis=1)

    del vi_main
    tr_main = _scale_tracks(tr_main, frame_h, frame_w)  # [n_robot_seed, T_traj, 2]

    # Use gripper tracks for trajectory; fall back to all robot-seed tracks if none detected
    if median_mask_all.any():
        tr_for_median = tr_main[median_mask_all]
    else:
        logger.warning(
            "Episode %d: no gripper detected anywhere — using all %d robot-seed tracks "
            "for trajectory (quality may be reduced).",
            episode_id,
            n_robot_seed,
        )
        tr_for_median = tr_main[:n_robot_seed]

    median_traj = compute_median_trajectory(tr_for_median, None)  # visibles ignored
    xy_range = median_traj.max(axis=0) - median_traj.min(axis=0)
    logger.info(
        "Episode %d: median trajectory computed from %d points — x_range=%.1f px, y_range=%.1f px.",
        episode_id,
        len(tr_for_median),
        xy_range[0],
        xy_range[1],
    )

    # Keep the resized secondary video for per-timestep tracking below.
    sec_frames: np.ndarray | None = None
    if sec_cam_key is not None and sec_frames_list:
        sec_frames = np.stack(sec_frames_list, axis=0)
        del sec_frames_list

    # ── Early stop: truncate trajectory when robot returns to start pose ────────
    if early_stop:
        try:
            _, sharpest_drop_idx = get_frame_at_sharpest_drop(actions.numpy())
            traj_cut = min(sharpest_drop_idx // max(frame_stride, 1) + 1, T_traj)
            logger.info(
                "Episode %d: early stop — full-rate frame %d → traj frame %d/%d.",
                episode_id,
                sharpest_drop_idx,
                traj_cut - 1,
                T_traj - 1,
            )
            T_traj = traj_cut
            tr_main = tr_main[:, :traj_cut, :]
            median_traj = median_traj[:traj_cut]
        except Exception as exc:
            logger.warning(
                "Episode %d: early stop failed (%s) — using full trajectory.",
                episode_id,
                exc,
            )

    # ── Per-timestep grid tracking ─────────────────────────────────────────────
    # For each trajectory timestep t, sample n_grid_per_row fresh grid points
    # directly on frame t and track them forward for exactly H frames.  This
    # ensures grid tracks at timestep t represent objects visible *at* t, not
    # objects that happened to be at grid positions at t=0.
    #
    # Query points are sampled in TAPNext-resolution space (tgt_H x tgt_W)
    # because main_frames is already resized; run_tapnext's internal scale
    # factor is therefore 1 and the returned tracks are in the same space.
    # _scale_tracks then converts from tgt → original pixel space.
    logger.info(
        "Episode %d — per-timestep grid tracking: %d timesteps x %d pts x H=%d …",
        episode_id,
        T_traj,
        n_grid_per_row,
        horizon,
    )
    tgt_H_int, tgt_W_int = int(tapir_target_size[0]), int(tapir_target_size[1])
    grid_tracks_norm = np.empty((T_traj, n_grid_per_row, 1 + horizon, 2), dtype=np.float32)

    for t in range(T_traj):
        # Fresh reproducible RNG per (episode, timestep) so results are stable
        # across --resume runs and independent of robot-point sampling order.
        rng_t = np.random.default_rng([episode_id, t])
        qp_t = sample_grid_points_cropped(tgt_H_int, tgt_W_int, n_grid_per_row, rng_t)

        # Build an (H+1)-frame clip starting at t; repeat last frame if needed.
        end = min(t + 1 + horizon, T_traj)
        clip = main_frames[t:end]
        if clip.shape[0] < 1 + horizon:
            pad = np.tile(clip[-1:], (1 + horizon - clip.shape[0], 1, 1, 1))
            clip = np.concatenate([clip, pad], axis=0)

        tr_t, _ = run_tapnext(tapnext_model, clip, qp_t, tapir_target_size, device)
        # tr_t: [n_grid_per_row, 1+H, 2] in tgt pixel space
        grid_tracks_norm[t] = normalize_tracks(
            _scale_tracks(tr_t, frame_h, frame_w),
            frame_h,
            frame_w,
        )

        if (t + 1) % 20 == 0 or t == T_traj - 1:
            logger.info("  Episode %d: grid tracking %d/%d timesteps done.", episode_id, t + 1, T_traj)

    del main_frames

    # Secondary-camera tracks follow the same strategy as main-camera grid
    # tracks: sample fresh random points at each timestep and track them over a
    # dedicated H+1 frame clip.
    sec_tracks_norm: np.ndarray | None = None
    if sec_frames is not None:
        logger.info(
            "Episode %d — secondary per-timestep grid tracking: %d timesteps x %d pts x H=%d …",
            episode_id,
            T_traj,
            num_sec_pts,
            horizon,
        )
        sec_tracks_norm = np.empty((T_traj, num_sec_pts, 1 + horizon, 2), dtype=np.float32)
        for t in range(T_traj):
            # Separate deterministic random stream from the main camera.
            rng_t = np.random.default_rng([episode_id, t, 1])
            # Secondary views benefit from full-frame spatial coverage, so no
            # central crop is applied here.
            qp_t = sample_grid_points(tgt_H_int, tgt_W_int, num_sec_pts, rng_t)

            end = min(t + 1 + horizon, T_traj)
            clip = sec_frames[t:end]
            if clip.shape[0] < 1 + horizon:
                pad = np.tile(clip[-1:], (1 + horizon - clip.shape[0], 1, 1, 1))
                clip = np.concatenate([clip, pad], axis=0)

            tr_t, _ = run_tapnext(tapnext_model, clip, qp_t, tapir_target_size, device)
            sec_tracks_norm[t] = normalize_tracks(_scale_tracks(tr_t, sec_fh, sec_fw), sec_fh, sec_fw)

            if (t + 1) % 20 == 0 or t == T_traj - 1:
                logger.info(
                    "  Episode %d: secondary grid tracking %d/%d timesteps done.",
                    episode_id,
                    t + 1,
                    T_traj,
                )
        del sec_frames

    # ── Normalise global trajectory: (x, y, action) ───────────────────────────
    x_norm = np.clip(median_traj[:, 0] / frame_w, 0.0, 1.0).astype(np.float32)
    y_norm = np.clip(median_traj[:, 1] / frame_h, 0.0, 1.0).astype(np.float32)

    if grip_idx is not None:
        grip_signal_full = actions[:, grip_idx].numpy().astype(np.float32)  # [T_full]
        # Sample at traj rate and trim/pad to T_traj
        grip_signal_traj = grip_signal_full[::frame_stride][:T_traj]
        if len(grip_signal_traj) < T_traj:
            pad = np.full(T_traj - len(grip_signal_traj), grip_signal_traj[-1], dtype=np.float32)
            grip_signal_traj = np.concatenate([grip_signal_traj, pad])
        # Normalise using dataset stats (required — no episode-range fallback)
        if "action" not in dataset.meta.stats:
            raise RuntimeError(
                f"Episode {episode_id}: dataset.meta.stats has no 'action' key. "
                "Compute dataset statistics first (e.g. lerobot-compute-stats)."
            )
        action_stats = dataset.meta.stats["action"]
        a_min = float(action_stats["min"][grip_idx])
        a_max = float(action_stats["max"][grip_idx])
        grip_range = max(float(a_max - a_min), 1e-6)
        action_norm = np.clip(
            (grip_signal_traj - a_min) / grip_range,
            0.0,
            1.0,
        ).astype(np.float32)
    else:
        action_norm = np.zeros(T_traj, dtype=np.float32)
        logger.warning(
            "Episode %d: gripper action dim not found; action channel will be zero.",
            episode_id,
        )

    global_traj_norm = np.stack([x_norm, y_norm, action_norm], axis=1)  # [T_traj, 3]

    # ── Trajectory overlay image ───────────────────────────────────────────────
    if overlay_dir is not None and first_bgr_orig is not None:
        try:
            _render_traj_overlay(
                first_bgr=first_bgr_orig,
                median_traj=median_traj,
                action_norm=action_norm,
                frame_stride=frame_stride,
                out_path=overlay_dir / f"episode_{episode_id:06d}_overlay.png",
            )
            logger.info(
                "Episode %d: overlay saved → %s",
                episode_id,
                overlay_dir / f"episode_{episode_id:06d}_overlay.png",
            )
        except Exception as exc:
            logger.warning("Episode %d: overlay render failed — %s", episode_id, exc)

    # ── Build per-timestep windows and save ────────────────────────────────────
    robot_pt = build_point_tracks(normalize_tracks(tr_main, frame_h, frame_w), horizon)
    # robot_pt: [n_robot_seed, T_traj, 1+H, 2]

    arrays: dict[str, np.ndarray] = {
        "global_trajectory": global_traj_norm,
        "robot_tracks": robot_pt,
        "grid_tracks": grid_tracks_norm,
        "n_robot_seed_pts": np.array(n_robot_seed, dtype=np.int32),
        "n_grid_per_row_pts": np.array(n_grid_per_row, dtype=np.int32),
        "sec_tracking_mode": np.array("per_timestep_grid"),
        "frame_stride": np.array(frame_stride, dtype=np.int32),
    }

    if sec_tracks_norm is not None:
        arrays["sec_cam_point_tracks"] = sec_tracks_norm

    np.savez_compressed(str(out_path), **arrays)
    logger.info("Episode %d: saved → %s", episode_id, out_path)
    return True


# ── Lazy loader for temp NPZ files ────────────────────────────────────────────


class _EpisodeLoader:
    """LRU-cached loader for per-episode NPZ files.

    Keeps at most *max_cached* episodes in memory so the callable functions
    passed to ``add_features`` do not reload the same file for every frame.
    """

    def __init__(self, temp_dir: Path, max_cached: int = 4):
        self._dir = temp_dir
        self._max = max_cached
        self._cache: dict[int, dict[str, np.ndarray]] = {}

    def get(self, ep_id: int) -> dict[str, np.ndarray]:
        if ep_id not in self._cache:
            path = self._dir / f"episode_{ep_id:06d}.npz"
            with np.load(str(path)) as f:
                data = {k: f[k] for k in f.files}
            if len(self._cache) >= self._max:
                self._cache.pop(next(iter(self._cache)))
            self._cache[ep_id] = data
        return self._cache[ep_id]


def main(argv=None) -> None:
    args = parse_args(argv)
    from functools import partial

    from .writer import add_spatial_features

    add_features = partial(add_spatial_features, options=vars(args))
    lerobot_add_features = add_features
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    if args.grid_pool_multiplier != 4:
        logger.warning(
            "--grid-pool-multiplier is no longer used (per-timestep grid tracking "
            "replaces the pool approach); the value %d will be ignored.",
            args.grid_pool_multiplier,
        )

    horizon = args.horizon
    num_robot_pts = args.num_robot_points
    n_main = num_robot_pts
    n_robot_per_row = num_robot_pts // 2
    n_grid_per_row = num_robot_pts - n_robot_per_row
    num_sec_pts = args.num_sec_points
    tapir_target_size = (args.tapir_target_size, args.tapir_target_size)

    # ── Load models ────────────────────────────────────────────────────────────
    from ultralytics import YOLO

    logger.info("Loading YOLO from %s …", args.model_path)
    yolo_model = YOLO(args.model_path, verbose=False).to(device)

    logger.info("Loading TAPNext++ from %s …", args.tapnext_checkpoint)
    tapnext_model = load_tapnext_model(
        args.tapnext_checkpoint,
        device=device,
        image_size=tapir_target_size,
    )

    # ── Load source dataset ────────────────────────────────────────────────────
    logger.info("Loading dataset %s …", args.source_repo_id)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(args.source_repo_id)
    num_episodes = dataset.meta.total_episodes
    logger.info("Total episodes: %d  |  total frames: %d", num_episodes, len(dataset))

    episodes = parse_episode_list(args.episodes, num_episodes) if args.episodes else list(range(num_episodes))

    # Auto-enable single-episode-output when exactly one episode is requested.
    single_ep_output = args.single_episode_output or (args.episodes is not None and len(episodes) == 1)
    if single_ep_output:
        logger.info(
            "Single-episode-output mode: output dataset will contain only episode(s) %s.",
            episodes,
        )

    # ── Resolve camera keys (support short names like "side") ─────────────────
    all_keys = list(dataset.features.keys())

    img_video_keys = [k for k in all_keys if dataset.features[k].get("dtype") in ("video", "image")]

    def resolve_cam_key(key: str) -> str:
        if key in all_keys:
            return key
        matches = [k for k in all_keys if key in k]
        if len(matches) == 1:
            logger.info("Camera key '%s' resolved to '%s'.", key, matches[0])
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous camera key '{key}'. Matches: {matches}. Please pass the full key.")
        # Fall back to auto-detecting the first image/video key so that datasets
        # with non-canonical camera names still work without --main-cam-key.
        if img_video_keys:
            logger.info(
                "Camera key '%s' not found; auto-selecting '%s' as main camera.",
                key,
                img_video_keys[0],
            )
            return img_video_keys[0]
        raise KeyError(
            f"Camera key '{key}' not found in dataset and no image/video keys exist. "
            f"Available keys: {all_keys}"
        )

    args.main_cam_key = resolve_cam_key(args.main_cam_key)

    # ── Auto-detect secondary camera if not explicitly set / disabled ──────────
    if args.no_sec_cam:
        args.sec_cam_key = None
    elif args.sec_cam_key is not None:
        args.sec_cam_key = resolve_cam_key(args.sec_cam_key)
    else:
        # Identify image/video keys that are not the main camera.
        img_keys = [
            k
            for k in all_keys
            if k != args.main_cam_key and dataset.features[k].get("dtype") in ("video", "image")
        ]
        if img_keys:
            args.sec_cam_key = img_keys[0]
            if len(img_keys) > 1:
                logger.warning(
                    "Multiple secondary cameras found %s — using '%s'. "
                    "Pass --sec-cam-key to choose a different one.",
                    img_keys,
                    args.sec_cam_key,
                )
            else:
                logger.info("Secondary camera auto-detected: '%s'.", args.sec_cam_key)
        else:
            logger.info("No secondary camera found; skipping secondary-cam tracking.")

    # ── Paths ──────────────────────────────────────────────────────────────────
    output_repo_id = args.output_repo_id or f"{args.source_repo_id}-tracks"
    output_dir = Path(args.output_dir) if args.output_dir else None

    # Overlay directory (None disables saving overlays)
    if args.no_overlays:
        overlay_dir: Path | None = None
    elif args.overlay_dir:
        overlay_dir = Path(args.overlay_dir)
    else:
        # Default: data_generation/overlays/<dataset_name> — one folder per output dataset,
        # always co-located with the script regardless of --output-dir.
        dataset_slug = output_repo_id.replace("/", "__")
        overlay_dir = Path.cwd() / "outputs" / "overlays" / dataset_slug
    if overlay_dir is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Overlay directory: %s", overlay_dir)

    # Temp directory for intermediate NPZ files.
    if args.temp_dir:
        temp_dir = Path(args.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        _own_temp = False
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="lerobot_tracks_"))
        _own_temp = True
    logger.info("Temp NPZ directory: %s", temp_dir)

    # ── Detect stable occlusion center ────────────────────────────────────────
    logger.info("Detecting occlusion center …")
    occlusion_center = detect_occlusion_center(dataset, yolo_model, args.main_cam_key)

    # ── Phase 1: compute tracks per episode ───────────────────────────────────
    logger.info("=== Phase 1: computing tracks ===")
    ok_episodes: list[int] = []
    fail_episodes: list[int] = []
    skip_episodes: list[int] = []

    for ep_id in episodes:
        out_path = temp_dir / f"episode_{ep_id:06d}.npz"
        if args.resume and out_path.exists():
            try:
                with np.load(str(out_path)) as f:
                    cached_n_grid_per_row = int(f["n_grid_per_row_pts"]) if "n_grid_per_row_pts" in f else -1
                    cached_sec_mode = str(f["sec_tracking_mode"]) if "sec_tracking_mode" in f else ""
            except Exception:
                cached_n_grid_per_row = -1
                cached_sec_mode = ""
            expected_sec_mode = "per_timestep_grid" if args.sec_cam_key else ""
            if cached_n_grid_per_row == n_grid_per_row and (
                not args.sec_cam_key or cached_sec_mode == expected_sec_mode
            ):
                logger.info(
                    "Episode %d: temp file exists (n_grid_per_row=%d) — skipped.",
                    ep_id,
                    cached_n_grid_per_row,
                )
                ok_episodes.append(ep_id)
                skip_episodes.append(ep_id)
                continue
            else:
                logger.info(
                    "Episode %d: temp file config mismatch "
                    "(cached n_grid_per_row=%d ≠ expected %d) — regenerating.",
                    ep_id,
                    cached_n_grid_per_row,
                    n_grid_per_row,
                )

        try:
            success = process_episode(
                episode_id=ep_id,
                yolo_model=yolo_model,
                tapnext_model=tapnext_model,
                repo_id=args.source_repo_id,
                main_cam_key=args.main_cam_key,
                sec_cam_key=args.sec_cam_key,
                occlusion_center=occlusion_center,
                horizon=horizon,
                num_robot_pts=num_robot_pts,
                num_sec_pts=num_sec_pts,
                tapir_target_size=tapir_target_size,
                frame_stride=args.frame_stride,
                device=device,
                out_path=out_path,
                early_stop=not args.no_early_stop,
                overlay_dir=overlay_dir,
            )
        except Exception as exc:
            logger.error("Episode %d: failed — %s", ep_id, exc, exc_info=True)
            fail_episodes.append(ep_id)
            continue

        (ok_episodes if success else fail_episodes).append(ep_id)

    logger.info(
        "Phase 1 done: %d ok (%d skipped), %d failed.",
        len(ok_episodes),
        len(skip_episodes),
        len(fail_episodes),
    )
    if fail_episodes or not ok_episodes:
        raise RuntimeError(
            f"Requested episodes failed: {fail_episodes}; intermediate results retained for resume"
        )
    args.selected_episode_ids = sorted(ok_episodes)

    add_global_trajectory = not args.no_global_trajectory

    # ── Phase 2: compute max_T for global_trajectory padding ──────────────────
    max_T = 0
    if add_global_trajectory:
        logger.info("=== Phase 2: computing max episode length for padding ===")
        for ep_id in ok_episodes:
            with np.load(str(temp_dir / f"episode_{ep_id:06d}.npz")) as f:
                t = f["global_trajectory"].shape[0]
            if t > max_T:
                max_T = t
        logger.info("max_T (traj frames) = %d", max_T)
    else:
        logger.info("=== Phase 2: global_trajectory disabled; skipping padding setup ===")

    # ── Phase 3: build add_features inputs ────────────────────────────────────
    logger.info("=== Phase 3: building LeRobot feature inputs ===")

    loader = _EpisodeLoader(temp_dir)

    # Zero arrays for missing-episode fallback in the point-track callables.
    zero_main = np.zeros((n_main, 1 + horizon, 2), dtype=np.float32)

    # Per-frame callables for point tracks.
    # Each callable receives (row_dict, ep_idx, frame_idx) and returns the
    # [N, 1+H, 2] array for that specific frame.
    # Episodes without an NPZ (not processed) receive an all-zero array so the
    # callable never crashes when _copy_data_with_feature_changes iterates over
    # the full parquet.
    def _main_tracks_fn(row: dict, ep_idx: int, frame_idx: int) -> np.ndarray:
        npz_path = temp_dir / f"episode_{ep_idx:06d}.npz"
        if not npz_path.exists():
            return zero_main
        data = loader.get(ep_idx)
        robot_tracks = data["robot_tracks"]  # [n_robot_seed, T_traj, 1+H, 2]
        grid_tracks = data["grid_tracks"]  # [T_traj, n_grid_per_row, 1+H, 2]
        stride = int(data["frame_stride"])
        t_traj = min(frame_idx // max(stride, 1), robot_tracks.shape[1] - 1)
        t_grid = min(frame_idx // max(stride, 1), grid_tracks.shape[0] - 1)

        robot = robot_tracks[:, t_traj, :, :]  # [n_robot, 1+H, 2]
        grid = grid_tracks[t_grid]  # [n_grid, 1+H, 2]
        return np.concatenate([robot, grid], axis=0).astype(np.float32)

    features_to_add: dict = {
        "observation.main_cam_point_tracks": (
            _main_tracks_fn,
            {
                "dtype": "float32",
                "shape": [n_main, 1 + horizon, 2],
                "names": ["query", "horizon", "xy"],
            },
        ),
    }

    if add_global_trajectory:
        # global_trajectory — episode-keyed dict (broadcast to all frames of episode)
        # Shorter episodes are padded by repeating the last row (including action channel).
        global_traj_dict: dict[int, np.ndarray] = {}
        for ep_id in ok_episodes:
            traj = loader.get(ep_id)["global_trajectory"]  # [T_ep, 3]
            T_ep = traj.shape[0]
            if T_ep < max_T:
                pad = np.tile(traj[-1:], (max_T - T_ep, 1))
                traj = np.concatenate([traj, pad], axis=0)
            global_traj_dict[ep_id] = traj.astype(np.float32)

        # Fill zero-placeholders for every episode in the source dataset.
        # lerobot's _copy_data_with_feature_changes reads ALL parquet files from
        # dataset.root regardless of any episodes= filter on the LeRobotDataset
        # object, so the dict must cover all episode indices that appear in the
        # parquet — not just the ones we processed.
        zero_traj = np.zeros((max_T, 3), dtype=np.float32)
        for ep_id in range(num_episodes):
            if ep_id not in global_traj_dict:
                global_traj_dict[ep_id] = zero_traj

        features_to_add["observation.global_trajectory"] = (
            global_traj_dict,
            {
                "dtype": "float32",
                "shape": [max_T, 3],
                "names": ["t", "xya"],
            },
        )

    # Secondary-cam feature — only add if at least one episode has sec tracks.
    has_sec = any("sec_cam_point_tracks" in loader.get(ep_id) for ep_id in ok_episodes)
    if has_sec:
        zero_sec = np.zeros((num_sec_pts, 1 + horizon, 2), dtype=np.float32)

        def _sec_tracks_fn(row: dict, ep_idx: int, frame_idx: int) -> np.ndarray:
            npz_path = temp_dir / f"episode_{ep_idx:06d}.npz"
            if not npz_path.exists():
                return zero_sec
            data = loader.get(ep_idx)
            if "sec_cam_point_tracks" not in data:
                return zero_sec
            tracks = data["sec_cam_point_tracks"]  # [T_traj, N_sec, 1+H, 2]
            stride = int(data["frame_stride"])
            t_traj = min(frame_idx // max(stride, 1), tracks.shape[0] - 1)
            return tracks[t_traj].astype(np.float32)

        features_to_add["observation.sec_cam_point_tracks"] = (
            _sec_tracks_fn,
            {
                "dtype": "float32",
                "shape": [num_sec_pts, 1 + horizon, 2],
                "names": ["query", "horizon", "xy"],
            },
        )

    # ── Phase 4: write new dataset ─────────────────────────────────────────────
    logger.info("=== Phase 4: writing new LeRobot dataset '%s' ===", output_repo_id)
    new_dataset = lerobot_add_features(
        dataset=dataset,
        features=features_to_add,
        repo_id=output_repo_id,
        output_dir=output_dir,
    )

    logger.info("Finalizing dataset …")
    new_dataset.finalize()

    # Preserve original camera identities: track metadata refers to these keys.

    # ── Phase 5: optional release trim, then push to Hub ──────────────────────
    final_dataset = new_dataset
    if args.trim_after_action_release:
        logger.info("Creating release-trimmed derivative of %s.", output_repo_id)
        final_dataset = release_trim_generated_dataset(
            repo_id=output_repo_id,
            root=Path(new_dataset.root),
            output_repo_id=args.trim_output_repo_id,
            output_dir=Path(args.trim_output_dir) if args.trim_output_dir else None,
            overwrite_output_dir=False,
            no_push=args.no_push,
            public=args.public,
            keep_after_frames=args.trim_keep_after_frames,
            event_threshold=args.trim_event_threshold,
        )
    elif not args.no_push:
        logger.info("Pushing '%s' to Hub …", output_repo_id)
        new_dataset.push_to_hub(private=not args.public)
        logger.info("✓ Available at https://huggingface.co/datasets/%s", output_repo_id)
    else:
        logger.info("✓ Dataset saved locally at %s (not pushed).", new_dataset.root)

    # ── Cleanup ────────────────────────────────────────────────────────────────
    if not args.keep_temp:
        if _own_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info("Temp directory deleted: %s", temp_dir)
        else:
            for ep_id in ok_episodes:
                p = temp_dir / f"episode_{ep_id:06d}.npz"
                if p.exists():
                    p.unlink()
            logger.info("Temp NPZ files deleted from %s", temp_dir)

    # ── Summary ────────────────────────────────────────────────────────────────
    grid_r, grid_c = _covering_grid_dims(num_sec_pts)
    global_feature_summary = (
        f"                          observation.global_trajectory  [{max_T}, 3]\n"
        if add_global_trajectory
        else "                          observation.global_trajectory  disabled\n"
    )
    logger.info(
        "Done.\n"
        "  Output dataset        : %s\n"
        "  Main-cam tracking     : %d robot pts (single pass) + %d grid pts (per-timestep)\n"
        "  Features added        :%s"
        "                          observation.main_cam_point_tracks  [%d, %d, 2]\n"
        "%s"
        "  Episodes ok           : %d  (%d skipped phase 1)\n"
        "  Episodes failed       : %d",
        output_repo_id,
        n_robot_per_row,
        n_grid_per_row,
        global_feature_summary,
        n_main,
        1 + horizon,
        f"                          observation.sec_cam_point_tracks  [{num_sec_pts}, {1 + horizon}, 2]"
        f"  (grid {grid_r}x{grid_c})\n"
        if has_sec
        else "",
        len(ok_episodes),
        len(skip_episodes),
        len(fail_episodes),
    )
    if final_dataset is not new_dataset:
        logger.info("Final trimmed dataset: %s", final_dataset.root)


if __name__ == "__main__":
    main()
