#!/usr/bin/env python3
"""Replace observation.global_trajectory using gripper-tip tracking."""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

TRAJECTORY_KEY = "observation.global_trajectory"
DEFAULT_SOURCE_REPO_ID = "edmos7/so100_multitask_quality_checked_tip_v1-dav2small-depth-release-trimmed-tip-tracked-trajectory"
DEFAULT_CAMERA_KEY = "observation.images.side"
DEFAULT_YOLO_PATH = REPO_ROOT / "models" / "segmentso101.pt"
DEFAULT_TAPNEXT_PATH = REPO_ROOT / "models" / "tapnextpp_ckpt.pt"

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

import torch  # noqa: E402
from lerobot.datasets.dataset_tools import add_features, modify_features  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.utils.constants import HF_LEROBOT_HOME  # noqa: E402
from utils.rt_trajectory_utils import (  # noqa: E402
    best_gripper_mask,
    detect_grasp_release,
    find_gripper_action_index,
    normalize_action_signal,
    render_debug_overlay,
    resolve_camera_key,
    rgb_to_bgr,
    rgb_to_gray_bgr,
    sample_bottom_band_tip_points,
    sample_tip_points,
    sample_to_rgb,
    to_numpy,
    trajectory_is_usable,
    yolo_tip_trajectory,
)
from utils.tracking import compute_median_trajectory, load_tapnext_model, run_tapnext  # noqa: E402
from data_generation.preprocessing import add_release_trim_args, release_trim_generated_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replace observation.global_trajectory with gripper-tip tracking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source-repo-id", default=DEFAULT_SOURCE_REPO_ID)
    p.add_argument("--source-root", default=None)
    p.add_argument(
        "--output-repo-id",
        default=None,
        help="Default: <source>-tip-tracked-trajectory, or <source> with --redraw-only.",
    )
    p.add_argument("--output-dir", default=None)
    p.add_argument("--camera-key", default=DEFAULT_CAMERA_KEY)
    p.add_argument("--episodes", default=None, help="Subset like '0-9,15'. Full dataset by default.")
    p.add_argument("--skip-episodes", default=None, help="Episodes to skip, like '0-9,15'.")

    p.add_argument("--yolo-path", default=str(DEFAULT_YOLO_PATH))
    p.add_argument("--tapnext-checkpoint", default=str(DEFAULT_TAPNEXT_PATH))
    p.add_argument("--tapnext-size", type=int, default=256)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--num-tip-points", type=int, default=16)
    p.add_argument("--tip-conf", type=float, default=0.85)
    p.add_argument("--tip-band-quantile", type=float, default=0.80)
    p.add_argument("--tip-sample-mode", choices=("morphology", "band"), default="morphology")
    p.add_argument("--tip-radius", type=float, default=5.0)
    p.add_argument("--yolo-grayscale", action="store_true")
    p.add_argument("--event-threshold", type=float, default=0.25)
    p.add_argument(
        "--task-event-threshold",
        action="append",
        default=[],
        metavar="TASK_ID:THRESHOLD",
        help="Override --event-threshold for task ids, e.g. 0:0.20,5:0.35.",
    )

    p.add_argument(
        "--target-length",
        default="existing",
        help="Use 'existing' to preserve the current feature length, 'max' for max generated length, or an integer.",
    )
    p.add_argument("--temp-dir", default=None)
    p.add_argument("--keep-temp", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite-output-dir", action="store_true")
    p.add_argument("--overlay-dir", default=None)
    p.add_argument("--no-overlays", action="store_true")
    p.add_argument(
        "--redraw-episodes",
        default=None,
        help="Episode ids/ranges to manually redraw after tip tracking, e.g. '58,68'.",
    )
    p.add_argument(
        "--redraw-only",
        action="store_true",
        help=(
            "Skip automatic tracking and only redraw --redraw-episodes, preserving existing "
            "global trajectories for all other episodes. If --output-repo-id is not "
            "set, pushes back to --source-repo-id."
        ),
    )
    p.add_argument("--drawn-trajectory-time-sec", type=float, default=15.0)
    p.add_argument("--episode-time-sec", type=float, default=90.0)
    p.add_argument("--action-interaction-sec", type=float, default=1.0)
    p.add_argument("--action-timing-mode", choices=("time", "arclength"), default="time")
    p.add_argument(
        "--opencv-qt-platform",
        default=None,
        help=(
            "Set QT_QPA_PLATFORM before importing OpenCV for redraw UI, e.g. xcb or wayland. "
            "Defaults to xcb when running under a Wayland session."
        ),
    )
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--public", action="store_true")
    add_release_trim_args(p)
    return p.parse_args()


def parse_episode_list(spec: str, max_ep: int) -> list[int]:
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                lo, hi = hi, lo
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return sorted(i for i in out if 0 <= i < max_ep)


def parse_task_event_thresholds(specs: list[str] | None) -> dict[int, float]:
    thresholds: dict[int, float] = {}
    for spec in specs or []:
        for part in str(spec).split(","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                raise ValueError(f"Expected TASK_ID:THRESHOLD, got {part!r}.")
            task_id_s, threshold_s = part.split(":", 1)
            thresholds[int(task_id_s.strip())] = float(threshold_s.strip())
    return thresholds


def scalar_task_id(value: Any) -> int | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return int(value.flatten()[0].item())
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    return int(value)


def safe_repo_name(repo_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", repo_id).strip("_")


def load_episode_arrays(dataset: LeRobotDataset, camera_key: str) -> tuple[np.ndarray, np.ndarray, int | None]:
    frames = []
    actions = []
    task_id = None
    for sample in dataset:
        if task_id is None and "task_id" in sample:
            task_id = scalar_task_id(sample["task_id"])
        frames.append(sample_to_rgb(sample, camera_key))
        actions.append(to_numpy(sample["action"]))
    if not frames:
        raise ValueError("episode has no frames")
    return np.stack(frames), np.stack(actions), task_id


def find_seed(
    yolo_model,
    frames_rgb: np.ndarray,
    *,
    tip_conf: float,
    frame_stride: int,
    n_points: int,
    band_quantile: float,
    tip_radius: float,
    sample_mode: str,
    yolo_grayscale: bool,
    rng: np.random.Generator,
) -> tuple[int, np.ndarray, dict]:
    frame_h, frame_w = frames_rgb.shape[1:3]
    scan_limit = max(1, len(frames_rgb) // 2)
    best = None
    for frame_idx in range(0, scan_limit, frame_stride):
        bgr = rgb_to_gray_bgr(frames_rgb[frame_idx]) if yolo_grayscale else rgb_to_bgr(frames_rgb[frame_idx])
        results = yolo_model(bgr, conf=tip_conf, verbose=False)
        candidate = best_gripper_mask(results, frame_h, frame_w, yolo_model.names)
        if candidate is None:
            continue
        candidate["frame_idx"] = frame_idx
        if best is None or candidate["max_y"] > best["max_y"] or (
            candidate["max_y"] == best["max_y"] and candidate["area"] > best["area"]
        ):
            best = candidate
    if best is None:
        raise RuntimeError("No gripper mask found in scan window.")
    if sample_mode == "morphology":
        query_points = sample_tip_points(best["mask"], n_points, rng, band_quantile, tip_radius)
    else:
        query_points = sample_bottom_band_tip_points(best["mask"], n_points, rng, band_quantile)
    return int(best["frame_idx"]), query_points, best


def fit_trajectory_length(traj: np.ndarray, target_len: int) -> np.ndarray:
    traj = np.asarray(traj, dtype=np.float32)
    if len(traj) == target_len:
        return traj
    if len(traj) == 0:
        return np.zeros((target_len, 3), dtype=np.float32)
    if len(traj) > target_len:
        return traj[:target_len].astype(np.float32)
    return np.concatenate([traj, np.tile(traj[-1:], (target_len - len(traj), 1))], axis=0).astype(np.float32)


def render_overlay(frame_rgb: np.ndarray, trajectory: np.ndarray, line_width: int = 6, alpha: float = 0.75) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    pts = np.asarray(trajectory[:, :2], dtype=np.float32).copy()
    pts[:, 0] *= w
    pts[:, 1] *= h
    rt = np.zeros((h, w, 3), dtype=np.uint8)
    canvas = Image.fromarray(rt)
    from PIL import ImageDraw

    draw = ImageDraw.Draw(canvas)
    action = np.clip(trajectory[:, 2], 0.0, 1.0)
    denom = max(len(pts) - 1, 1)
    for i in range(len(pts) - 1):
        red = int(np.clip((i / denom) * 255, 0, 255))
        blue = int(np.clip(action[i] * 255, 0, 255))
        p0 = tuple(np.clip(np.round(pts[i]).astype(int), [0, 0], [w - 1, h - 1]))
        p1 = tuple(np.clip(np.round(pts[i + 1]).astype(int), [0, 0], [w - 1, h - 1]))
        draw.line([p0, p1], fill=(red, 0, blue), width=line_width)
    rt = np.asarray(canvas)
    return render_debug_overlay(frame_rgb, rt, alpha=alpha)


def generate_episode_global_trajectory(
    *,
    episode_id: int,
    repo_id: str,
    root: str | None,
    camera_key: str,
    yolo_model,
    tapnext_model,
    device: str,
    tapnext_size: tuple[int, int],
    frame_stride: int,
    num_tip_points: int,
    tip_conf: float,
    tip_band_quantile: float,
    tip_sample_mode: str,
    tip_radius: float,
    yolo_grayscale: bool,
    event_threshold: float,
    task_event_thresholds: dict[int, float],
) -> tuple[np.ndarray, np.ndarray, dict]:
    ds = LeRobotDataset(repo_id=repo_id, root=root, episodes=[episode_id], download_videos=True)
    camera_key = resolve_camera_key(ds, camera_key)
    frames_rgb, actions, task_id = load_episode_arrays(ds, camera_key)
    threshold = task_event_thresholds.get(task_id, event_threshold) if task_id is not None else event_threshold
    frame_h, frame_w = frames_rgb.shape[1:3]
    rng = np.random.default_rng([episode_id, 2026])

    seed_frame_idx = None
    seed_meta: dict = {}
    trajectory_xy = None
    trajectory_source = "tapnext"
    try:
        seed_frame_idx, query_points, seed_meta = find_seed(
            yolo_model,
            frames_rgb,
            tip_conf=tip_conf,
            frame_stride=frame_stride,
            n_points=num_tip_points,
            band_quantile=tip_band_quantile,
            tip_radius=tip_radius,
            sample_mode=tip_sample_mode,
            yolo_grayscale=yolo_grayscale,
            rng=rng,
        )
        tracking_frames = frames_rgb[::frame_stride]
        seed_t = seed_frame_idx // frame_stride
        if seed_t == 0:
            tracks, visibles = run_tapnext(tapnext_model, tracking_frames, query_points, tapnext_size, device)
        else:
            tracks_bwd, vis_bwd = run_tapnext(
                tapnext_model, tracking_frames[seed_t::-1].copy(), query_points, tapnext_size, device
            )
            tracks_bwd = tracks_bwd[:, ::-1, :].copy()
            vis_bwd = vis_bwd[:, ::-1].copy()
            tracks_fwd, vis_fwd = run_tapnext(
                tapnext_model, tracking_frames[seed_t:], query_points, tapnext_size, device
            )
            tracks = np.concatenate([tracks_bwd[:, :seed_t, :], tracks_fwd], axis=1)
            visibles = np.concatenate([vis_bwd[:, :seed_t], vis_fwd], axis=1)
        trajectory_xy = compute_median_trajectory(tracks, visibles)
        if not trajectory_is_usable(trajectory_xy):
            log.warning("Episode %06d: TAPNext trajectory is degenerate; falling back to YOLO tip trajectory.", episode_id)
            trajectory_xy = None
    except Exception as exc:
        log.warning("Episode %06d: TAPNext path failed (%s); falling back to YOLO tip trajectory.", episode_id, exc)

    if trajectory_xy is None:
        trajectory_xy, fallback_meta = yolo_tip_trajectory(
            yolo_model,
            frames_rgb,
            conf=tip_conf,
            frame_stride=frame_stride,
            grayscale=yolo_grayscale,
        )
        trajectory_source = fallback_meta["source"]
        seed_meta.update(fallback_meta)

    grip_idx = find_gripper_action_index(ds)
    action_norm = normalize_action_signal(ds, actions, grip_idx)[::frame_stride][: len(trajectory_xy)]
    grasp_idx, release_idx, peak_starts = detect_grasp_release(action_norm, threshold=threshold)
    binary_action = np.zeros(len(trajectory_xy), dtype=np.float32)
    if grasp_idx is not None and release_idx is not None and release_idx >= grasp_idx:
        binary_action[grasp_idx : release_idx + 1] = 1.0

    global_traj = np.stack(
        [
            np.clip(trajectory_xy[:, 0] / frame_w, 0.0, 1.0),
            np.clip(trajectory_xy[:, 1] / frame_h, 0.0, 1.0),
            binary_action,
        ],
        axis=1,
    ).astype(np.float32)
    meta = {
        "seed_frame_idx": seed_frame_idx,
        "seed_class": seed_meta.get("class_name"),
        "seed_conf": seed_meta.get("conf"),
        "seed_area": seed_meta.get("area"),
        "trajectory_source": trajectory_source,
        "task_id": task_id,
        "event_threshold": threshold,
        "grasp_idx": grasp_idx,
        "release_idx": release_idx,
        "peak_starts": peak_starts,
        "trajectory_len": int(len(global_traj)),
    }
    return global_traj, frames_rgb[0], meta


def existing_trajectory_length(dataset: LeRobotDataset) -> int:
    shape = dataset.features[TRAJECTORY_KEY].get("shape")
    if not shape:
        raise ValueError(f"{TRAJECTORY_KEY} has no shape metadata.")
    return int(shape[0])


def resolve_target_length(spec: str, existing_len: int, generated: dict[int, np.ndarray]) -> int:
    if spec == "existing":
        return existing_len
    if spec == "max":
        if not generated:
            return existing_len
        return max(len(traj) for traj in generated.values())
    return int(spec)


def episode_start_index(dataset: LeRobotDataset, episode_id: int) -> int:
    ep = dataset.meta.episodes[episode_id]
    return int(ep["dataset_from_index"])


def load_existing_global_trajectory(dataset: LeRobotDataset, episode_id: int) -> np.ndarray:
    dataset._ensure_hf_dataset_loaded()
    start = episode_start_index(dataset, episode_id)
    return np.asarray(dataset.hf_dataset[start][TRAJECTORY_KEY], dtype=np.float32)


def load_episode_first_frame(repo_id: str, root: str | None, episode_id: int, camera_key: str) -> np.ndarray:
    ds = LeRobotDataset(repo_id=repo_id, root=root, episodes=[episode_id], download_videos=True)
    camera_key = resolve_camera_key(ds, camera_key)
    return sample_to_rgb(ds[0], camera_key)


def configure_opencv_qt_platform(args: argparse.Namespace) -> None:
    if args.opencv_qt_platform:
        platform = args.opencv_qt_platform
        if platform == "xcb" and os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland":
            platform = "wayland"
            log.info(
                "Wayland session detected; using QT_QPA_PLATFORM=wayland "
                "instead of requested xcb for OpenCV redraw UI."
            )
        os.environ["QT_QPA_PLATFORM"] = platform
        log.info("Set QT_QPA_PLATFORM=%s for OpenCV redraw UI.", platform)
        return
    if os.environ.get("QT_QPA_PLATFORM"):
        log.info("Using existing QT_QPA_PLATFORM=%s for OpenCV redraw UI.", os.environ["QT_QPA_PLATFORM"])
        return
    log.info("Leaving QT_QPA_PLATFORM unset for OpenCV redraw UI.")


def show_image_and_wait(cv2, window_name: str, image: np.ndarray, prompt: str) -> None:
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, image)
        cv2.waitKey(100)
        visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
    except cv2.error as exc:
        cv2.destroyAllWindows()
        raise RuntimeError(
            "OpenCV could not create a GUI window. Try rerunning with "
            "--opencv-qt-platform xcb, or run from a terminal with access to your desktop display."
        ) from exc
    if visible < 1:
        cv2.destroyAllWindows()
        raise RuntimeError(
            "OpenCV created the redraw window but it is not visible. Try rerunning with "
            "--opencv-qt-platform xcb, or check your desktop/display forwarding setup."
        )
    print(prompt)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def render_global_trajectory_overlay(
    frame_rgb: np.ndarray,
    global_trajectory: np.ndarray,
    line_width: int = 6,
    alpha: float = 0.75,
) -> np.ndarray:
    """Render a global trajectory over an RGB frame; returns BGR for OpenCV."""
    out = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    overlay = out.copy()
    h, w = frame_rgb.shape[:2]
    traj = np.asarray(global_trajectory, dtype=np.float32)
    if traj.ndim != 2 or traj.shape[1] != 3:
        raise ValueError(f"Expected global trajectory [T, 3], got {traj.shape}.")

    pts = np.column_stack(
        [
            np.clip(traj[:, 0], 0.0, 1.0) * (w - 1),
            np.clip(traj[:, 1], 0.0, 1.0) * (h - 1),
        ]
    ).astype(np.int32)
    action = np.clip(traj[:, 2], 0.0, 1.0)

    for i in range(len(pts) - 1):
        r = int(255 * i / max(len(pts) - 2, 1))
        b = int(255 * float(action[i]))
        cv2.line(overlay, tuple(pts[i]), tuple(pts[i + 1]), (b, 0, r), line_width)

    cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0.0, out)
    return out


class TrajectoryDrawer:
    """Interactive tool to draw a trajectory on a camera frame.

    This mirrors the inference redraw UI: mouse movement records the path after
    the first click, later left-clicks mark active manipulation boundaries, and
    the final tensor preview is shown in the same OpenCV window.
    """

    ACTION_PEAK = 1.0
    LINE_THICKNESS = 3

    def __init__(
        self,
        image_array: np.ndarray,
        *,
        fps: int,
        drawn_trajectory_time_sec: float | None,
        episode_time_sec: float,
        action_interaction_sec: float,
        action_timing_mode: str,
    ):
        self.bg_rgb = image_array.copy()
        self.image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        self.height, self.width = image_array.shape[:2]
        self.fps = fps
        self.drawn_trajectory_time_sec = drawn_trajectory_time_sec
        self.episode_time_sec = episode_time_sec
        self.action_interaction_sec = action_interaction_sec
        self.action_timing_mode = action_timing_mode

        self.trajectory: list[tuple[int, int]] = []
        self.click_indices: list[int] = []
        self.started = False
        self.stop_collection = False
        self.display_image = self.image.copy()
        self.window_name = (
            "Draw Trajectory  |  Left-click to start / add action event  |  "
            "Right-click or SPACE to finish"
        )

    def _interaction_steps(self, n: int) -> int:
        if n < 2:
            return 0
        duration_s = self.drawn_trajectory_time_sec
        if duration_s is None:
            duration_s = self.episode_time_sec
        duration_s = max(float(duration_s), 1e-6)
        steps_per_sec = (n - 1) / duration_s
        return max(1, int(round(float(self.action_interaction_sec) * steps_per_sec)))

    def _build_action_signal(self, n: int) -> np.ndarray:
        action = np.zeros(n, dtype=np.float32)
        if n < 2:
            return action
        if len(self.click_indices) >= 2:
            first_idx = int(np.clip(min(self.click_indices), 0, n - 1))
            last_idx = int(np.clip(max(self.click_indices), 0, n - 1))
            action[first_idx : last_idx + 1] = self.ACTION_PEAK
            return action
        window_steps = self._interaction_steps(n)
        for idx in self.click_indices:
            click_idx = int(np.clip(idx, 0, n - 1))
            start_idx = max(0, click_idx - window_steps + 1)
            action[start_idx : click_idx + 1] = self.ACTION_PEAK
        return action

    def mouse_callback(self, event, x: int, y: int, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.started:
                self.started = True
                self.trajectory.append((x, y))
                print(f"Trajectory started at ({x}, {y})")
            else:
                self.trajectory.append((x, y))
                idx = len(self.trajectory) - 1
                self.click_indices.append(idx)
                if len(self.click_indices) == 1:
                    detail = f"{self.action_interaction_sec:.2f}s fallback window"
                else:
                    detail = "active interval boundary"
                print(f"Action event at ({x}, {y})  [traj index {idx}, {detail}]")
            self._update_display()
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.started and not self.stop_collection:
                self.trajectory.append((x, y))
                self._update_display()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.stop_collection = True

    def _render_overlay(self, show_action: bool = False) -> np.ndarray:
        out = self.image.copy()
        pts = self.trajectory
        n = len(pts)
        if n < 2:
            return out

        action = self._build_action_signal(n) if show_action else np.zeros(n, dtype=np.float32)
        for i in range(n - 1):
            t_norm = i / (n - 1)
            r = int(255 * t_norm)
            b = int(255 * float(action[i])) if show_action else 0
            cv2.line(
                out,
                tuple(map(int, pts[i])),
                tuple(map(int, pts[i + 1])),
                (b, 0, r),
                self.LINE_THICKNESS,
            )

        for idx in self.click_indices:
            if idx < n:
                cv2.circle(out, tuple(map(int, pts[idx])), 7, (255, 255, 255), 2)
        return out

    def _update_display(self) -> None:
        self.display_image = self._render_overlay(show_action=False)
        cv2.imshow(self.window_name, self.display_image)

    def run(self) -> None:
        cv2.imshow(self.window_name, self.display_image)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\nDrawing controls:")
        print("  First LEFT CLICK   -> start trajectory")
        print("  Move cursor        -> draw path")
        print("  LEFT CLICK         -> mark active manipulation boundary")
        print("  RIGHT CLICK / SPACE -> finish\n")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") or self.stop_collection:
                break
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        if len(self.trajectory) < 2:
            print("Warning: no trajectory drawn; using a dummy point at centre.")
            cx, cy = self.width // 2, self.height // 2
            self.trajectory = [(cx, cy), (cx, cy)]

        preview_bgr = self._render_overlay(show_action=True)
        cv2.imshow(self.window_name, preview_bgr)
        print(
            f"Trajectory complete: {len(self.trajectory)} points, "
            f"{len(self.click_indices)} action boundary event(s). Press any key to continue."
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def _resample_series(values: np.ndarray, out_n: int) -> tuple[np.ndarray, np.ndarray]:
        in_n = len(values)
        if out_n <= 1:
            return values[:1].copy(), np.array([0.0], dtype=np.float32)
        if in_n <= 1:
            return np.repeat(values[:1], out_n, axis=0), np.zeros(in_n, dtype=np.float32)

        src_t = np.linspace(0.0, 1.0, in_n, dtype=np.float32)
        dst_t = np.linspace(0.0, 1.0, out_n, dtype=np.float32)
        if values.ndim == 1:
            out = np.interp(dst_t, src_t, values).astype(np.float32)
        else:
            cols = [np.interp(dst_t, src_t, values[:, c]) for c in range(values.shape[1])]
            out = np.stack(cols, axis=1).astype(np.float32)
        index_scale = (out_n - 1) / max(in_n - 1, 1)
        return out, np.arange(in_n, dtype=np.float32) * index_scale

    @staticmethod
    def _resample_path_arclength(pts: np.ndarray, out_n: int) -> tuple[np.ndarray, np.ndarray]:
        in_n = len(pts)
        if out_n <= 1:
            return pts[:1].copy(), np.zeros(in_n, dtype=np.float32)
        if in_n <= 1:
            return np.repeat(pts[:1], out_n, axis=0), np.zeros(in_n, dtype=np.float32)

        deltas = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        dist = np.concatenate([[0.0], np.cumsum(deltas)]).astype(np.float32)
        total = float(dist[-1])
        if total < 1e-6:
            return TrajectoryDrawer._resample_series(pts, out_n)

        dst_dist = np.linspace(0.0, total, out_n, dtype=np.float32)
        x = np.interp(dst_dist, dist, pts[:, 0])
        y = np.interp(dst_dist, dist, pts[:, 1])
        pts_out = np.stack([x, y], axis=1).astype(np.float32)
        click_index_map = dist / total * (out_n - 1)
        return pts_out, click_index_map.astype(np.float32)

    @staticmethod
    def _build_action_signal_from_indices(
        n: int,
        click_indices: list[float],
        interaction_steps: float,
        peak: float,
    ) -> np.ndarray:
        action = np.zeros(n, dtype=np.float32)
        if n < 2:
            return action
        if len(click_indices) >= 2:
            first_idx = int(np.clip(round(float(min(click_indices))), 0, n - 1))
            last_idx = int(np.clip(round(float(max(click_indices))), 0, n - 1))
            action[first_idx : last_idx + 1] = peak
            return action
        window_steps = max(1, int(round(float(interaction_steps))))
        for idx in click_indices:
            click_idx = int(np.clip(round(float(idx)), 0, n - 1))
            start_idx = max(0, click_idx - window_steps + 1)
            action[start_idx : click_idx + 1] = peak
        return action

    def to_global_trajectory(self, max_T: int) -> np.ndarray:
        pts = np.array(self.trajectory, dtype=np.float32)
        n = len(pts)
        pts[:, 0] /= self.width
        pts[:, 1] /= self.height
        pts = np.clip(pts, 0.0, 1.0)

        duration_s = self.drawn_trajectory_time_sec
        if duration_s is None:
            duration_s = self.episode_time_sec
        duration_s = max(float(duration_s), 1e-6)
        effective_T = int(round(duration_s * self.fps))
        effective_T = max(2, min(max_T, effective_T))

        pts_eff, original_to_eff = self._resample_path_arclength(pts, effective_T)
        action_timing_mode = self.action_timing_mode.strip().lower()
        if action_timing_mode == "arclength":
            click_eff = [float(original_to_eff[min(max(idx, 0), n - 1)]) for idx in self.click_indices]
        elif action_timing_mode == "time":
            click_eff = [
                (float(min(max(idx, 0), n - 1)) / max(n - 1, 1)) * (effective_T - 1)
                for idx in self.click_indices
            ]
        else:
            raise ValueError(
                f"Unknown action timing mode {self.action_timing_mode!r}; expected 'time' or 'arclength'."
            )

        steps_per_sec = (effective_T - 1) / duration_s
        action = self._build_action_signal_from_indices(
            effective_T,
            click_eff,
            interaction_steps=max(1.0, float(self.action_interaction_sec) * steps_per_sec),
            peak=self.ACTION_PEAK,
        )

        pad = max_T - effective_T
        if pad > 0:
            pts_out = np.vstack([pts_eff, np.tile(pts_eff[-1:], (pad, 1))])
            action_out = np.concatenate([action, np.full(pad, action[-1], dtype=np.float32)])
        else:
            pts_out = pts_eff
            action_out = action

        return np.column_stack([pts_out, action_out]).astype(np.float32)


def maybe_redraw_trajectories(
    *,
    args: argparse.Namespace,
    source: LeRobotDataset,
    episodes_to_redraw: list[int],
    generated: dict[int, np.ndarray],
    target_len: int,
    camera_key: str,
    overlay_dir: Path | None,
) -> None:
    if not episodes_to_redraw:
        return

    fps = int(getattr(source.meta, "fps", 30) or 30)
    for ep in episodes_to_redraw:
        log.info("Episode %06d: opening manual trajectory redraw UI.", ep)
        frame_rgb = load_episode_first_frame(args.source_repo_id, args.source_root, ep, camera_key)
        current = fit_trajectory_length(
            generated.get(ep, load_existing_global_trajectory(source, ep)),
            target_len,
        )
        input(
            f"Episode {ep:06d}: loaded current trajectory with shape {current.shape}. "
            "Press Enter to start drawing a replacement..."
        )

        while True:
            print("Please draw the replacement trajectory in the window that opens.")
            drawer = TrajectoryDrawer(
                frame_rgb,
                fps=fps,
                drawn_trajectory_time_sec=args.drawn_trajectory_time_sec,
                episode_time_sec=args.episode_time_sec,
                action_interaction_sec=args.action_interaction_sec,
                action_timing_mode=args.action_timing_mode,
            )
            drawer.run()
            replacement = drawer.to_global_trajectory(target_len)
            choice = input(
                f"\nEpisode {ep:06d} replacement has shape {replacement.shape}.\n"
                "  [Enter] -> accept replacement\n"
                "  [r]     -> redraw\n"
                "> "
            ).strip().lower()
            if choice != "r":
                break

        generated[ep] = replacement.astype(np.float32)
        if overlay_dir is not None:
            Image.fromarray(render_overlay(frame_rgb, replacement)).save(
                overlay_dir / f"episode_{ep:06d}_overlay.png"
            )
        log.info("Episode %06d: manual replacement accepted.", ep)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    task_event_thresholds = parse_task_event_thresholds(args.task_event_threshold)

    source = LeRobotDataset(args.source_repo_id, root=args.source_root)
    if TRAJECTORY_KEY not in source.features:
        raise KeyError(f"Source dataset must already contain {TRAJECTORY_KEY!r}.")
    camera_key = resolve_camera_key(source, args.camera_key)
    num_episodes = source.meta.total_episodes
    episodes = parse_episode_list(args.episodes, num_episodes) if args.episodes else list(range(num_episodes))
    skipped = set(parse_episode_list(args.skip_episodes, num_episodes)) if args.skip_episodes else set()
    episodes = [ep for ep in episodes if ep not in skipped]
    redraw_episodes = parse_episode_list(args.redraw_episodes, num_episodes) if args.redraw_episodes else []
    if args.redraw_only:
        if not redraw_episodes:
            raise ValueError("--redraw-only requires --redraw-episodes.")
        episodes = []
    if not episodes:
        if not args.redraw_only:
            raise ValueError("No episodes selected.")

    output_repo_id = args.output_repo_id or (
        args.source_repo_id if args.redraw_only else f"{args.source_repo_id}-tip-tracked-trajectory"
    )

    temp_root = Path(args.temp_dir) if args.temp_dir else Path(tempfile.mkdtemp(prefix="rt_tracked_global_traj_"))
    own_temp = args.temp_dir is None
    temp_root.mkdir(parents=True, exist_ok=True)
    npz_dir = temp_root / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)
    stripped_dir = temp_root / "source_without_global_trajectory"
    output_dir = Path(args.output_dir) if args.output_dir else HF_LEROBOT_HOME / output_repo_id
    same_repo_output = output_repo_id == args.source_repo_id
    if same_repo_output and args.output_dir is None:
        output_dir = temp_root / "output_dataset"
    if output_dir.exists():
        if not args.overwrite_output_dir:
            raise FileExistsError(f"Output directory exists: {output_dir}. Use --overwrite-output-dir.")
        source_root = Path(source.root).resolve()
        output_root = output_dir.resolve()
        if output_root == source_root or source_root in output_root.parents or output_root in source_root.parents:
            raise ValueError(
                f"Refusing to remove output directory because it overlaps the source dataset root: {output_dir}"
            )
        shutil.rmtree(output_dir)
    overlay_dir = None if args.no_overlays else (
        Path(args.overlay_dir) if args.overlay_dir else HERE / "overlays" / safe_repo_name(output_repo_id)
    )
    if overlay_dir is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    generated: dict[int, np.ndarray] = {}
    failed: list[int] = []
    if args.redraw_only:
        log.info("--redraw-only set: skipping automatic tracking and preserving existing trajectories.")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tapnext_size = (args.tapnext_size, args.tapnext_size)
        from ultralytics import YOLO

        log.info("Loading YOLO: %s", args.yolo_path)
        yolo_model = YOLO(args.yolo_path, verbose=False).to(device)
        log.info("Loading TAPNext++: %s", args.tapnext_checkpoint)
        tapnext_model = load_tapnext_model(args.tapnext_checkpoint, device=device, image_size=tapnext_size)

        for ep in episodes:
            npz_path = npz_dir / f"episode_{ep:06d}.npz"
            if args.resume and npz_path.exists():
                with np.load(str(npz_path), allow_pickle=False) as f:
                    generated[ep] = f["global_trajectory"].astype(np.float32)
                log.info("Episode %06d: loaded existing %s", ep, npz_path)
                continue
            try:
                traj, first_frame, meta = generate_episode_global_trajectory(
                    episode_id=ep,
                    repo_id=args.source_repo_id,
                    root=args.source_root,
                    camera_key=camera_key,
                    yolo_model=yolo_model,
                    tapnext_model=tapnext_model,
                    device=device,
                    tapnext_size=tapnext_size,
                    frame_stride=args.frame_stride,
                    num_tip_points=args.num_tip_points,
                    tip_conf=args.tip_conf,
                    tip_band_quantile=args.tip_band_quantile,
                    tip_sample_mode=args.tip_sample_mode,
                    tip_radius=args.tip_radius,
                    yolo_grayscale=args.yolo_grayscale,
                    event_threshold=args.event_threshold,
                    task_event_thresholds=task_event_thresholds,
                )
                np.savez_compressed(str(npz_path), global_trajectory=traj)
                generated[ep] = traj
                if overlay_dir is not None:
                    Image.fromarray(render_overlay(first_frame, traj)).save(
                        overlay_dir / f"episode_{ep:06d}_overlay.png"
                    )
                log.info("Episode %06d: saved | %s", ep, meta)
            except Exception as exc:
                log.exception("Episode %06d failed: %s", ep, exc)
                failed.append(ep)

    if failed:
        raise RuntimeError(f"Failed episodes: {failed}")

    existing_len = existing_trajectory_length(source)
    target_len = resolve_target_length(args.target_length, existing_len, generated)
    maybe_redraw_trajectories(
        args=args,
        source=source,
        episodes_to_redraw=redraw_episodes,
        generated=generated,
        target_len=target_len,
        camera_key=camera_key,
        overlay_dir=overlay_dir,
    )

    feature_values: dict[int, np.ndarray] = {}
    for ep in range(num_episodes):
        if ep in generated:
            feature_values[ep] = fit_trajectory_length(generated[ep], target_len)
        else:
            feature_values[ep] = fit_trajectory_length(load_existing_global_trajectory(source, ep), target_len)

    log.info("Removing existing %s into temporary source: %s", TRAJECTORY_KEY, stripped_dir)
    if stripped_dir.exists():
        shutil.rmtree(stripped_dir)
    stripped = modify_features(
        dataset=source,
        remove_features=TRAJECTORY_KEY,
        output_dir=stripped_dir,
        repo_id=f"{args.source_repo_id}-without-global-trajectory-tmp",
    )
    stripped.finalize()

    log.info("Adding tip-tracked %s to %s.", TRAJECTORY_KEY, output_repo_id)
    new_dataset = add_features(
        dataset=stripped,
        features={
            TRAJECTORY_KEY: (
                feature_values,
                {"dtype": "float32", "shape": [target_len, 3], "names": ["t", "xya"]},
            )
        },
        output_dir=output_dir,
        repo_id=output_repo_id,
    )
    new_dataset.finalize()

    final_dataset = new_dataset
    if args.trim_after_action_release:
        log.info("Creating release-trimmed derivative of %s.", output_repo_id)
        final_dataset = release_trim_generated_dataset(
            repo_id=output_repo_id,
            root=Path(new_dataset.root),
            output_repo_id=args.trim_output_repo_id,
            output_dir=Path(args.trim_output_dir) if args.trim_output_dir else None,
            overwrite_output_dir=args.overwrite_output_dir,
            no_push=args.no_push,
            public=args.public,
            keep_after_frames=args.trim_keep_after_frames,
            event_threshold=args.trim_event_threshold,
        )
    elif args.no_push:
        log.info("--no-push set: dataset finalized locally only at %s.", new_dataset.root)
    else:
        log.info("Pushing to Hub: %s", output_repo_id)
        new_dataset.push_to_hub(private=not args.public)

    if not args.keep_temp and own_temp:
        shutil.rmtree(temp_root, ignore_errors=True)
    changed_episodes = redraw_episodes if args.redraw_only else episodes
    log.info("Done: %d episode(s), target_len=%d, output=%s", len(changed_episodes), target_len, output_repo_id)
    if final_dataset is not new_dataset:
        log.info("Final trimmed dataset: %s", final_dataset.root)


if __name__ == "__main__":
    main()
