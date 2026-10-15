#!/usr/bin/env python3
"""Generate gripper-tip global trajectories and point-track features.

This combines:

* gripper-tip trajectory generation from ``generate_global_trajectory_dataset.py``;
* main- and secondary-camera point-track generation from ``generate_track_dataset.py``.

Selected feature families are written in a single ``add_features`` operation.
The ordinary body-point global trajectory produced by the track processor is
intentionally ignored; when trajectory generation is enabled, the output global
trajectory comes from the newer gripper-tip tracking strategy.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

from lerobot_policy_open_trajectory.generation.generate_global_trajectory_dataset import (
    generate_episode_global_trajectory,
    parse_task_event_thresholds,
    render_overlay,
)
from lerobot_policy_open_trajectory.generation.generate_track_dataset import (
    _EpisodeLoader,
    detect_occlusion_center,
    parse_episode_list,
)
from lerobot_policy_open_trajectory.generation.generate_track_dataset import (
    process_episode as process_track_episode,
)
from lerobot_policy_open_trajectory.generation.preprocessing import (
    add_release_trim_args,
    release_trim_generated_dataset,
)
from lerobot_policy_open_trajectory.generation.utils.tracking import load_tapnext_model

from .preprocessing import remove_output_dataset

DEFAULT_TIP_CONF = 0.85
DEFAULT_TIP_BAND_QUANTILE = 0.80
log = logging.getLogger(__name__)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Add gripper-tip global trajectories and/or point tracks to a new LeRobot dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source-repo-id", required=True, help="Source LeRobot dataset repo ID.")
    p.add_argument(
        "--output-repo-id",
        default=None,
        help="Output repo ID (default: <source-repo-id>-tracks-and-trajectory).",
    )
    p.add_argument(
        "--features",
        choices=("both", "tracks", "trajectory"),
        default="both",
        help="Feature families to generate.",
    )
    p.add_argument("--output-dir", default=None, help="Local output directory.")
    p.add_argument("--main-cam-key", default="observation.images.MAIN_CAM")
    p.add_argument("--sec-cam-key", default=None)
    p.add_argument("--no-sec-cam", action="store_true")

    p.add_argument("--model-path", required=True)
    p.add_argument("--tapnext-checkpoint", required=True)
    p.add_argument("--tapir-target-size", type=int, default=256)
    p.add_argument("--frame-stride", type=int, default=1)

    p.add_argument("--num-tip-points", type=int, default=16)
    p.add_argument("--tip-conf", type=float, default=DEFAULT_TIP_CONF)
    p.add_argument("--tip-band-quantile", type=float, default=DEFAULT_TIP_BAND_QUANTILE)
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

    p.add_argument("--num-robot-points", type=int, default=32)
    p.add_argument("--num-sec-points", type=int, default=32)
    p.add_argument("--horizon", type=int, default=64)

    p.add_argument("--episodes", default=None, help="Subset such as '0-9,15'.")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--temp-dir", default=None)
    p.add_argument("--keep-temp", action="store_true")
    p.add_argument("--no-early-stop", action="store_true")
    p.add_argument("--overlay-dir", default=None)
    p.add_argument("--no-overlays", action="store_true")

    p.add_argument("--overwrite-output-dir", action="store_true")
    p.add_argument(
        "--push",
        dest="no_push",
        action="store_false",
        help="Explicitly upload the generated dataset to the Hub.",
    )
    p.add_argument("--no-push", action="store_true", default=True)
    p.add_argument("--public", action="store_true")
    add_release_trim_args(p)
    p.add_argument("--artifact-dir", help="Also export portable per-episode annotations.")
    return p.parse_args(argv)


def _resolve_camera_key(dataset: LeRobotDataset, requested: str) -> str:
    keys = list(dataset.features)
    if requested in keys:
        return requested
    matches = [key for key in keys if requested in key]
    if len(matches) == 1:
        log.info("Camera key %r resolved to %r.", requested, matches[0])
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous camera key {requested!r}: {matches}")
    image_keys = [key for key in keys if dataset.features[key].get("dtype") in ("video", "image")]
    if image_keys:
        log.warning("Camera key %r not found; using %r.", requested, image_keys[0])
        return image_keys[0]
    raise KeyError(f"No image/video feature is available. Dataset keys: {keys}")


def _build_global_trajectories(
    tip_loader: _EpisodeLoader,
    ok_episodes: list[int],
    num_episodes: int,
) -> tuple[dict[int, np.ndarray], int]:
    max_t = max(tip_loader.get(ep_id)["global_trajectory"].shape[0] for ep_id in ok_episodes)
    zero = np.zeros((max_t, 3), dtype=np.float32)
    trajectories: dict[int, np.ndarray] = {}
    for ep_id in ok_episodes:
        trajectory = tip_loader.get(ep_id)["global_trajectory"]
        if trajectory.shape[0] < max_t:
            padding = np.tile(trajectory[-1:], (max_t - trajectory.shape[0], 1))
            trajectory = np.concatenate([trajectory, padding], axis=0)
        trajectories[ep_id] = trajectory.astype(np.float32)
    for ep_id in range(num_episodes):
        if ep_id not in trajectories:
            trajectories[ep_id] = zero
    return trajectories, max_t


def main(argv=None) -> None:
    args = parse_args(argv)
    from functools import partial

    from .writer import add_spatial_features

    add_features = partial(add_spatial_features, options=vars(args))
    lerobot_add_features = add_features
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_size = (args.tapir_target_size, args.tapir_target_size)
    generate_tracks = args.features in ("both", "tracks")
    generate_trajectory = args.features in ("both", "trajectory")
    task_event_thresholds = parse_task_event_thresholds(args.task_event_threshold)

    log.info("Loading source dataset %s.", args.source_repo_id)
    dataset = LeRobotDataset(args.source_repo_id)
    num_episodes = dataset.meta.total_episodes
    episodes = parse_episode_list(args.episodes, num_episodes) if args.episodes else list(range(num_episodes))
    if not episodes:
        raise ValueError("No valid episodes selected.")
    expected_new_features = set()
    if generate_trajectory:
        expected_new_features.add("observation.global_trajectory")
    if generate_tracks:
        expected_new_features.add("observation.main_cam_point_tracks")
        expected_new_features.add("observation.sec_cam_point_tracks")
    feature_collisions = expected_new_features.intersection(dataset.features)
    if feature_collisions:
        raise ValueError(
            "Source dataset already contains output feature(s): "
            f"{sorted(feature_collisions)}. Use a source dataset without these features."
        )

    args.main_cam_key = _resolve_camera_key(dataset, args.main_cam_key)
    image_keys = [key for key in dataset.features if dataset.features[key].get("dtype") in ("video", "image")]
    if args.no_sec_cam:
        args.sec_cam_key = None
    elif args.sec_cam_key:
        args.sec_cam_key = _resolve_camera_key(dataset, args.sec_cam_key)
    else:
        candidates = [key for key in image_keys if key != args.main_cam_key]
        args.sec_cam_key = candidates[0] if candidates else None
        log.info("Secondary camera: %s", args.sec_cam_key or "disabled")

    output_suffix = {
        "both": "tracks-and-trajectory",
        "tracks": "tracks",
        "trajectory": "tip-tracked-trajectory",
    }[args.features]
    output_repo_id = args.output_repo_id or f"{args.source_repo_id}-{output_suffix}"
    output_dir = Path(args.output_dir) if args.output_dir else None
    effective_output = output_dir or HF_LEROBOT_HOME / output_repo_id
    if Path(effective_output).resolve() == Path(dataset.root).resolve():
        raise ValueError("Output directory resolves to the source dataset.")
    if Path(effective_output).exists():
        if not args.overwrite_output_dir:
            raise FileExistsError(
                f"Output directory exists: {effective_output}. Use --overwrite-output-dir to replace it."
            )
        remove_output_dataset(effective_output, source=dataset.root)

    if args.temp_dir:
        temp_root = Path(args.temp_dir)
        temp_root.mkdir(parents=True, exist_ok=True)
        own_temp = False
    else:
        temp_root = Path(tempfile.mkdtemp(prefix="lerobot_tip_tracks_"))
        own_temp = True
    track_dir = temp_root / "tracks"
    tip_dir = temp_root / "tip_trajectories"
    track_dir.mkdir(parents=True, exist_ok=True)
    tip_dir.mkdir(parents=True, exist_ok=True)

    if args.no_overlays:
        track_overlay = tip_overlay = None
    else:
        overlay_root = (
            Path(args.overlay_dir)
            if args.overlay_dir
            else Path.cwd() / "outputs" / "overlays" / output_repo_id.replace("/", "__")
        )
        track_overlay = overlay_root / "tracks" if generate_tracks else None
        tip_overlay = overlay_root / "tip_trajectories" if generate_trajectory else None
        if track_overlay is not None:
            track_overlay.mkdir(parents=True, exist_ok=True)
        if tip_overlay is not None:
            tip_overlay.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    log.info("Loading YOLO and TAPNext++ on %s.", device)
    yolo_model = YOLO(args.model_path, verbose=False).to(device)
    tapnext_model = load_tapnext_model(
        args.tapnext_checkpoint,
        device=device,
        image_size=target_size,
    )
    occlusion_center = (
        detect_occlusion_center(dataset, yolo_model, args.main_cam_key) if generate_tracks else (0.0, 0.0)
    )

    n_robot = args.num_robot_points // 2
    n_grid = args.num_robot_points - n_robot
    ok_episodes: list[int] = []
    failed_episodes: list[int] = []

    for ep_id in episodes:
        track_path = track_dir / f"episode_{ep_id:06d}.npz"
        tip_path = tip_dir / f"episode_{ep_id:06d}.npz"
        tracks_ready = (not generate_tracks) or (args.resume and track_path.exists())
        tip_ready = (not generate_trajectory) or (args.resume and tip_path.exists())
        try:
            if generate_tracks and not tracks_ready:
                tracks_ready = process_track_episode(
                    episode_id=ep_id,
                    yolo_model=yolo_model,
                    tapnext_model=tapnext_model,
                    repo_id=args.source_repo_id,
                    main_cam_key=args.main_cam_key,
                    sec_cam_key=args.sec_cam_key,
                    occlusion_center=occlusion_center,
                    horizon=args.horizon,
                    num_robot_pts=args.num_robot_points,
                    num_sec_pts=args.num_sec_points,
                    tapir_target_size=target_size,
                    frame_stride=args.frame_stride,
                    device=device,
                    out_path=track_path,
                    early_stop=not args.no_early_stop,
                    overlay_dir=track_overlay,
                )
            if generate_trajectory and not tip_ready:
                trajectory, first_frame, meta = generate_episode_global_trajectory(
                    episode_id=ep_id,
                    repo_id=args.source_repo_id,
                    root=None,
                    camera_key=args.main_cam_key,
                    yolo_model=yolo_model,
                    tapnext_model=tapnext_model,
                    device=device,
                    tapnext_size=target_size,
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
                np.savez_compressed(str(tip_path), global_trajectory=trajectory)
                tip_ready = True
                if tip_overlay is not None:
                    from PIL import Image

                    Image.fromarray(render_overlay(first_frame, trajectory)).save(
                        tip_overlay / f"episode_{ep_id:06d}_overlay.png"
                    )
                log.info("Episode %06d tip trajectory: %s", ep_id, meta)
        except Exception as exc:
            log.error("Episode %d failed: %s", ep_id, exc, exc_info=True)
            failed_episodes.append(ep_id)
            continue
        if tracks_ready and tip_ready:
            ok_episodes.append(ep_id)
        else:
            failed_episodes.append(ep_id)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if failed_episodes or not ok_episodes:
        raise RuntimeError(
            f"Requested episodes failed: {failed_episodes}; intermediate results retained for resume"
        )
    args.selected_episode_ids = sorted(ok_episodes)

    track_loader = _EpisodeLoader(track_dir) if generate_tracks else None
    tip_loader = _EpisodeLoader(tip_dir) if generate_trajectory else None
    trajectories: dict[int, np.ndarray] = {}
    max_t = 0
    if generate_trajectory and tip_loader is not None:
        trajectories, max_t = _build_global_trajectories(tip_loader, ok_episodes, num_episodes)

    zero_main = np.zeros((args.num_robot_points, 1 + args.horizon, 2), dtype=np.float32)

    def main_tracks(row: dict, ep_idx: int, frame_idx: int) -> np.ndarray:
        path = track_dir / f"episode_{ep_idx:06d}.npz"
        if not path.exists():
            return zero_main
        assert track_loader is not None
        data = track_loader.get(ep_idx)
        robot = data["robot_tracks"]
        grid = data["grid_tracks"]
        stride = max(int(data["frame_stride"]), 1)
        robot_t = min(frame_idx // stride, robot.shape[1] - 1)
        grid_t = min(frame_idx // stride, grid.shape[0] - 1)
        return np.concatenate([robot[:, robot_t], grid[grid_t]], axis=0).astype(np.float32)

    features: dict = {}
    if generate_trajectory:
        features["observation.global_trajectory"] = (
            trajectories,
            {"dtype": "float32", "shape": [max_t, 3], "names": ["t", "xya"]},
        )
    if generate_tracks:
        features["observation.main_cam_point_tracks"] = (
            main_tracks,
            {
                "dtype": "float32",
                "shape": [args.num_robot_points, 1 + args.horizon, 2],
                "names": ["query", "horizon", "xy"],
            },
        )

    has_sec = bool(
        generate_tracks
        and track_loader is not None
        and any("sec_cam_point_tracks" in track_loader.get(ep_id) for ep_id in ok_episodes)
    )
    if has_sec:
        zero_sec = np.zeros((args.num_sec_points, 1 + args.horizon, 2), dtype=np.float32)

        def sec_tracks(row: dict, ep_idx: int, frame_idx: int) -> np.ndarray:
            path = track_dir / f"episode_{ep_idx:06d}.npz"
            if not path.exists():
                return zero_sec
            assert track_loader is not None
            data = track_loader.get(ep_idx)
            if "sec_cam_point_tracks" not in data:
                return zero_sec
            tracks = data["sec_cam_point_tracks"]
            stride = max(int(data["frame_stride"]), 1)
            track_t = min(frame_idx // stride, tracks.shape[0] - 1)
            return tracks[track_t].astype(np.float32)

        features["observation.sec_cam_point_tracks"] = (
            sec_tracks,
            {
                "dtype": "float32",
                "shape": [args.num_sec_points, 1 + args.horizon, 2],
                "names": ["query", "horizon", "xy"],
            },
        )

    log.info("Writing output dataset %s.", output_repo_id)
    new_dataset = lerobot_add_features(
        dataset=dataset,
        features=features,
        repo_id=output_repo_id,
        output_dir=output_dir,
    )
    new_dataset.finalize()

    # Preserve original camera identities: track metadata refers to these keys.

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
    elif not args.no_push:
        new_dataset.push_to_hub(private=not args.public)
        log.info("Available at https://huggingface.co/datasets/%s", output_repo_id)
    else:
        log.info("Dataset saved locally at %s.", new_dataset.root)

    if not args.keep_temp and own_temp:
        shutil.rmtree(temp_root, ignore_errors=True)
        # User-supplied temporary directories remain available for inspection.

    log.info(
        "Done: %d successful, %d failed. Added tip trajectory [%d,3], "
        "main tracks [%d,%d,2]%s. Main split: %d robot + %d grid points.",
        len(ok_episodes),
        len(failed_episodes),
        max_t,
        args.num_robot_points,
        1 + args.horizon,
        (f", secondary tracks [{args.num_sec_points},{1 + args.horizon},2]" if has_sec else ""),
        n_robot,
        n_grid,
    )
    if final_dataset is not new_dataset:
        log.info("Final trimmed dataset: %s", final_dataset.root)


if __name__ == "__main__":
    main()
