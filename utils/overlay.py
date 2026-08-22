"""Overlay trajectory images on dataset camera frames."""

from __future__ import annotations

import os

import numpy as np
from PIL import Image


def visualise_trajectory_overlay(
    episode_id: int,
    trajectory_folder: str,
    ds,
    cam_key: str,
    alpha: float = 0.7,
    save_dir: str | None = None,
) -> np.ndarray | None:
    """
    Blend a pre-generated trajectory image over the first camera frame of an episode.

    The blend is additive and weighted by per-pixel trajectory brightness so that
    black (empty) areas of the trajectory image are fully transparent.

    Args:
        episode_id:        Episode index.
        trajectory_folder: Directory containing ``episode_NNN.png`` trajectory files.
        ds:                Loaded LeRobotDataset instance.
        cam_key:           Dataset key for the camera observation to overlay on.
        alpha:             Trajectory opacity in [0, 1]. Default 0.7.
        save_dir:          If given, saves ``episode_NNN_overlay.png`` to this directory.

    Returns:
        HWC uint8 RGB numpy array of the blended image, or None if the
        trajectory file does not exist.
    """
    traj_path = os.path.join(trajectory_folder, f"episode_{episode_id:03d}.png")
    if not os.path.exists(traj_path):
        print(f"  Skipped episode {episode_id}: trajectory not found at {traj_path}")
        return None

    traj_rgb = np.array(Image.open(traj_path))

    first_global = int(ds.meta.episodes["dataset_from_index"][episode_id])
    sample = ds[first_global]
    cam_rgb = (sample[cam_key].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

    h_cam, w_cam = cam_rgb.shape[:2]
    if traj_rgb.shape[:2] != (h_cam, w_cam):
        traj_rgb = np.array(Image.fromarray(traj_rgb).resize((w_cam, h_cam), Image.BILINEAR))

    cam_f = cam_rgb.astype(np.float32)
    traj_f = traj_rgb.astype(np.float32)
    traj_weight = (traj_f.max(axis=2, keepdims=True) / 255.0) * alpha
    overlay_rgb = ((cam_f * (1.0 - traj_weight) + traj_f * traj_weight)
                   .clip(0, 255).astype(np.uint8))

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"episode_{episode_id:03d}_overlay.png")
        Image.fromarray(overlay_rgb).save(out_path)
        print(f"  Saved overlay → {out_path}")

    return overlay_rgb
