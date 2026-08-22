#!/usr/bin/env python3
"""Generate RT trajectory images and add them as a LeRobot image feature.

TAPNext point tracks are used only as temporary evidence to render one
episode-level conditioning image. The output dataset contains one new feature:

    observation.images.rt_trajectory

No point tracks or trajectory time-series features are added.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.rt_trajectory_utils import (  # noqa: E402
    best_gripper_mask,
    detect_grasp_release,
    find_gripper_action_index,
    normalize_action_signal,
    render_debug_overlay,
    render_rt_trajectory_image,
    resolve_camera_key,
    rgb_to_bgr,
    rgb_to_gray_bgr,
    sample_bottom_band_tip_points,
    sample_tip_points,
    sample_to_rgb,
    save_png,
    trajectory_is_usable,
    to_numpy,
    yolo_tip_trajectory,
)
from utils.tracking import compute_median_trajectory, load_tapnext_model, run_tapnext  # noqa: E402
from data_generation.preprocessing import add_release_trim_args, release_trim_generated_dataset  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_SOURCE_REPO_ID = "edmos7/so100_multitask_quality_checked_tip_v1-dav2small-depth"
DEFAULT_YOLO_PATH = REPO_ROOT / "models" / "segmentso101.pt"
DEFAULT_TAPNEXT_PATH = REPO_ROOT / "models" / "tapnextpp_ckpt.pt"


def _ensure_lerobot() -> None:
    try:
        import lerobot  # noqa: F401
    except ImportError:
        python = Path.home() / "miniforge3/envs/lerobotclone/bin/python"
        if not python.exists():
            sys.exit("ERROR: lerobot unavailable and ~/miniforge3/envs/lerobotclone/bin/python not found.")
        result = subprocess.run([str(python), __file__, *sys.argv[1:]])
        raise SystemExit(result.returncode)


_ensure_lerobot()

from lerobot.datasets.dataset_tools import add_features  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.utils.constants import HF_LEROBOT_HOME  # noqa: E402


def parse_episode_list(spec: str, max_ep: int) -> list[int]:
    indices: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            indices.update(range(int(lo), int(hi) + 1))
        else:
            indices.add(int(part))
    return sorted(i for i in indices if 0 <= i < max_ep)


def parse_task_event_thresholds(specs: list[str] | None) -> dict[int, float]:
    thresholds: dict[int, float] = {}
    for spec in specs or []:
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                raise ValueError(f"Expected TASK_ID:THRESHOLD, got {part!r}.")
            task_id_str, threshold_str = part.split(":", 1)
            task_id = int(task_id_str.strip())
            threshold = float(threshold_str.strip())
            thresholds[task_id] = threshold
    return thresholds


def scalar_task_id(value: Any) -> int:
    """Extract a scalar task id from HF/LeRobot scalar or length-1 values."""
    if torch.is_tensor(value):
        return int(value.flatten()[0].item())
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("Encountered an empty task_id value.")
        value = value[0]
    return int(value)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Add RT trajectory image conditioning to a new LeRobot dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source-repo-id", default=DEFAULT_SOURCE_REPO_ID)
    p.add_argument("--source-root", default=None)
    p.add_argument("--output-repo-id", default=None, help="Default: <source>-rt-trajectory-images")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--feature-key", default="observation.images.rt_trajectory")
    p.add_argument("--camera-key", default="observation.images.side")
    p.add_argument("--episode", type=int, default=None, help="Generate a single episode by index.")
    p.add_argument("--episodes", default=None, help="Subset like '0-9,15'. Full dataset by default.")
    p.add_argument("--skip-episodes", default=None, help="Episodes to skip, like '0-9,15'. Applied after --episode/--episodes.")

    p.add_argument("--yolo-path", default=str(DEFAULT_YOLO_PATH))
    p.add_argument("--tapnext-checkpoint", default=str(DEFAULT_TAPNEXT_PATH))
    p.add_argument("--tapnext-size", type=int, default=256)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--num-tip-points", type=int, default=16)
    p.add_argument("--tip-conf", type=float, default=0.85)
    p.add_argument("--tip-band-quantile", type=float, default=0.80)
    p.add_argument("--tip-sample-mode", choices=("morphology", "band"), default="morphology")
    p.add_argument("--tip-radius", type=float, default=5.0)
    p.add_argument("--yolo-grayscale", action="store_true", help="Use grayscale YOLO preprocessing instead of RGB/BGR.")
    p.add_argument("--event-threshold", type=float, default=0.25)
    p.add_argument(
        "--task-event-threshold",
        action="append",
        default=[],
        metavar="TASK_ID:THRESHOLD",
        help=(
            "Override --event-threshold for a task_id. May be repeated or comma-separated, "
            "for example: --task-event-threshold 0:0.20,5:0.35"
        ),
    )

    p.add_argument("--line-width", type=int, default=5)
    p.add_argument("--marker-radius", type=int, default=15)
    p.add_argument("--marker-width", type=int, default=8)

    p.add_argument("--image-dir", default=None, help="Cache directory for generated episode PNGs.")
    p.add_argument("--debug-overlay-dir", default=None)
    p.add_argument("--no-debug-overlays", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite-output-dir", action="store_true")
    p.add_argument("--images-only", action="store_true", help="Only generate/cache PNGs and debug overlays; do not build or push a LeRobot dataset.")
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--public", action="store_true")
    add_release_trim_args(p)
    return p.parse_args()


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


def generate_episode_image(
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
    line_width: int,
    marker_radius: int,
    marker_width: int,
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
    rt_img = render_rt_trajectory_image(
        trajectory_xy,
        image_shape=(frame_h, frame_w),
        grasp_idx=grasp_idx,
        release_idx=release_idx,
        line_width=line_width,
        marker_radius=marker_radius,
        marker_width=marker_width,
    )
    debug = render_debug_overlay(frames_rgb[0], rt_img)
    meta = {
        "seed_frame_idx": seed_frame_idx,
        "seed_class": seed_meta.get("class_name"),
        "seed_conf": seed_meta.get("conf"),
        "seed_area": seed_meta.get("area"),
        "trajectory_source": trajectory_source,
        "grasp_idx": grasp_idx,
        "release_idx": release_idx,
        "peak_starts": peak_starts,
        "task_id": task_id,
        "event_threshold": threshold,
        "camera_key": camera_key,
    }
    return rt_img, debug, meta


def load_trajectory_images(image_dir: Path, num_episodes: int, selected: set[int]) -> dict[int, np.ndarray]:
    images: dict[int, np.ndarray] = {}
    missing = []
    for ep_id in range(num_episodes):
        path = image_dir / f"episode_{ep_id:06d}.png"
        if path.exists():
            images[ep_id] = np.asarray(Image.open(path).convert("RGB"))
        elif ep_id in selected:
            missing.append(ep_id)
    if missing:
        raise FileNotFoundError(f"Missing generated images for selected episodes: {missing[:20]}")
    if not images:
        raise ValueError(f"No generated images found in {image_dir}")
    sample = next(iter(images.values()))
    zero = np.zeros_like(sample)
    return {ep_id: images.get(ep_id, zero) for ep_id in range(num_episodes)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    task_event_thresholds = parse_task_event_thresholds(args.task_event_threshold)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tapnext_size = (args.tapnext_size, args.tapnext_size)

    log.info("Loading source metadata: %s", args.source_repo_id)
    source = LeRobotDataset(args.source_repo_id, root=args.source_root)
    camera_key = resolve_camera_key(source, args.camera_key)
    if args.feature_key in source.features:
        raise ValueError(f"Source dataset already contains {args.feature_key!r}.")
    num_episodes = source.meta.total_episodes
    if args.episode is not None and args.episodes:
        raise ValueError("Use either --episode or --episodes, not both.")
    if args.episode is not None:
        if not (0 <= args.episode < num_episodes):
            raise ValueError(f"--episode must be in [0, {num_episodes - 1}], got {args.episode}.")
        episodes = [args.episode]
    else:
        episodes = parse_episode_list(args.episodes, num_episodes) if args.episodes else list(range(num_episodes))
    skipped = set(parse_episode_list(args.skip_episodes, num_episodes)) if args.skip_episodes else set()
    episodes = [ep_id for ep_id in episodes if ep_id not in skipped]
    if not episodes:
        raise ValueError("No episodes selected after applying --skip-episodes.")
    if skipped:
        log.info("Skipping episodes: %s", sorted(skipped))

    output_repo_id = args.output_repo_id or f"{args.source_repo_id}-rt-trajectory-images"
    image_dir = Path(args.image_dir) if args.image_dir else HERE / "images" / output_repo_id.replace("/", "__")
    debug_dir = None if args.no_debug_overlays else (
        Path(args.debug_overlay_dir) if args.debug_overlay_dir else HERE / "debug_overlays" / output_repo_id.replace("/", "__")
    )
    image_dir.mkdir(parents=True, exist_ok=True)
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    log.info("Loading YOLO: %s", args.yolo_path)
    yolo_model = YOLO(args.yolo_path, verbose=False).to(device)
    log.info("Loading TAPNext++: %s", args.tapnext_checkpoint)
    tapnext_model = load_tapnext_model(args.tapnext_checkpoint, device=device, image_size=tapnext_size)

    ok, failed = [], []
    for ep_id in episodes:
        out_path = image_dir / f"episode_{ep_id:06d}.png"
        if args.resume and out_path.exists():
            log.info("Episode %06d: image exists, skipped.", ep_id)
            ok.append(ep_id)
            continue
        try:
            rt_img, debug_img, meta = generate_episode_image(
                episode_id=ep_id,
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
                line_width=args.line_width,
                marker_radius=args.marker_radius,
                marker_width=args.marker_width,
            )
            save_png(out_path, rt_img)
            if debug_dir is not None:
                save_png(debug_dir / f"episode_{ep_id:06d}_overlay.png", debug_img)
            log.info("Episode %06d: saved | %s", ep_id, meta)
            ok.append(ep_id)
        except Exception as exc:
            log.exception("Episode %06d failed: %s", ep_id, exc)
            failed.append(ep_id)

    if failed:
        raise RuntimeError(f"Failed episodes: {failed}")

    if args.images_only:
        log.info("Images-only mode: generated/cached %d episode image(s) in %s.", len(ok), image_dir)
        return

    log.info("Adding %s to new dataset %s.", args.feature_key, output_repo_id)
    output_dir = Path(args.output_dir) if args.output_dir else HF_LEROBOT_HOME / output_repo_id
    if output_dir.exists():
        if not args.overwrite_output_dir:
            raise FileExistsError(f"Output directory exists: {output_dir}. Use --overwrite-output-dir.")
        shutil.rmtree(output_dir)

    feature_dict = load_trajectory_images(image_dir, num_episodes, set(episodes))
    h, w, c = next(iter(feature_dict.values())).shape
    new_dataset = add_features(
        dataset=source,
        features={
            args.feature_key: (
                feature_dict,
                {"dtype": "image", "shape": [h, w, c], "names": ["height", "width", "channel"]},
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
        from huggingface_hub import HfApi

        HfApi().create_tag(output_repo_id, tag=str(new_dataset.meta._version), repo_type="dataset", exist_ok=True)
    log.info("Done. Local dataset: %s", new_dataset.root)
    if final_dataset is not new_dataset:
        log.info("Final trimmed dataset: %s", final_dataset.root)


if __name__ == "__main__":
    main()
