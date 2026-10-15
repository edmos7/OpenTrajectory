#!/usr/bin/env python3
"""Create camera-keyed colorized relative-depth video features with Depth Anything V2."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot_policy_open_trajectory.generation.preprocessing import (
    add_release_trim_args,
    release_trim_generated_dataset,
)

from .demo_depth_anything_v2_small import load_model as load_dav2_model
from .demo_depth_anything_v2_small import tensor_to_rgb
from .preprocessing import remove_output_dataset

CAMERA_KEYS = ["auto"]
DEPTH_PREFIX = "observation.depth"
AUTO_FEATURES = {"episode_index", "frame_index", "timestamp", "index", "task_index"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add DA V2 Small depth videos to a LeRobot dataset.")
    parser.add_argument("--source-repo-id", required=True)
    parser.add_argument("--output-repo-id", required=True)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--episodes", type=int, nargs="+", default=None)
    parser.add_argument(
        "--camera-keys",
        nargs="+",
        default=CAMERA_KEYS,
        help="Camera keys to process, or 'auto' to use all observation.images.* features.",
    )
    parser.add_argument("--depth-prefix", default=DEPTH_PREFIX)
    parser.add_argument("--depth-anything-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--device", default=None)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument(
        "--push",
        dest="no_push",
        action="store_false",
        help="Explicitly upload the generated dataset to the Hub.",
    )
    parser.add_argument("--no-push", action="store_true", default=True)
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--download-videos", action="store_true")
    add_release_trim_args(parser)
    return parser.parse_args(argv)


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def output_dir_for(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    return Path.home() / ".cache" / "huggingface" / "lerobot" / args.output_repo_id


def prepare_output_dir(output_dir: Path, overwrite: bool, source: Path) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory exists: {output_dir}. Use --overwrite-output-dir to replace it."
            )
        logger.info("Removing existing output directory: %s", output_dir)
        remove_output_dataset(output_dir, source=source)


def to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def frame_value_for_write(value: Any, feature: dict[str, Any]) -> Any:
    if feature.get("dtype") in {"image", "video"}:
        array = to_numpy(value)
        expected = tuple(feature["shape"])
        if array.ndim == 3 and array.shape[0] in (1, 3) and expected[-1] in (1, 3):
            array = np.moveaxis(array, 0, -1)
        if np.issubdtype(array.dtype, np.floating) and array.max(initial=0) <= 1.0:
            array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
        return array

    array = to_numpy(value)
    expected = tuple(feature.get("shape", ()))
    if expected and array.shape != expected and array.size == int(np.prod(expected)):
        array = array.reshape(expected)
    return array


def depth_key_for(camera_key: str, depth_prefix: str) -> str:
    suffix = camera_key.removeprefix("observation.images.")
    return f"{depth_prefix}.{suffix}"


def build_output_features(
    source_features: dict[str, dict[str, Any]], camera_keys: list[str], depth_prefix: str
):
    features = {key: dict(value) for key, value in source_features.items()}
    for camera_key in camera_keys:
        if camera_key not in source_features:
            raise KeyError(f"Camera key {camera_key!r} missing from source dataset.")
        shape = tuple(source_features[camera_key]["shape"])
        if len(shape) != 3:
            raise ValueError(f"Expected image/video shape [H,W,C] for {camera_key}, got {shape}")
        h, w = int(shape[0]), int(shape[1])
        features[depth_key_for(camera_key, depth_prefix)] = {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channel"],
        }
    return features


def resolve_camera_keys(source_features: dict[str, dict[str, Any]], camera_keys: list[str]) -> list[str]:
    if camera_keys == ["auto"]:
        keys = [
            key
            for key, feature in source_features.items()
            if key.startswith("observation.images.") and feature.get("dtype") in {"image", "video"}
        ]
        if not keys:
            raise ValueError("No observation.images.* features found in source dataset.")
        return keys
    return camera_keys


def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    lo, hi = np.percentile(depth, [2.0, 98.0])
    norm = (depth - lo) / max(float(hi - lo), 1e-6)
    gray = (norm.clip(0.0, 1.0) * 255.0).astype(np.uint8)
    try:
        import cv2

        colored = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    except Exception:
        return np.repeat(gray[:, :, None], 3, axis=2)


def infer_depth_rgb(model: Any, rgb: np.ndarray, input_size: int) -> np.ndarray:
    with torch.inference_mode():
        depth = model.infer_image(rgb[:, :, ::-1].copy(), input_size=input_size)
    return depth_to_rgb(np.asarray(depth))


def task_name_for_sample(source, sample: dict[str, Any]) -> str:
    if "task_index" not in sample:
        return ""
    task_index = int(to_numpy(sample["task_index"]).reshape(-1)[0])
    try:
        return str(source.meta.tasks.iloc[task_index].name)
    except Exception:
        return ""


def main(argv=None) -> None:
    args = parse_args(argv)
    output_dir = output_dir_for(args)
    device = resolve_device(args.device)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    logger.info("Loading source dataset %s", args.source_repo_id)
    source = LeRobotDataset(
        repo_id=args.source_repo_id,
        root=args.source_root,
        download_videos=args.download_videos,
    )

    camera_keys = resolve_camera_keys(source.features, args.camera_keys)
    prepare_output_dir(output_dir, args.overwrite_output_dir, source.root)
    logger.info("Depth camera keys: %s", camera_keys)
    output_features = build_output_features(source.features, camera_keys, args.depth_prefix)
    logger.info("Loading DA V2 Small on %s", device)
    depth_model = load_dav2_model(args.depth_anything_dir, args.checkpoint, device)

    target = LeRobotDataset.create(
        repo_id=args.output_repo_id,
        root=output_dir,
        fps=source.fps,
        features=output_features,
        robot_type=source.meta.robot_type,
        use_videos=True,
        image_writer_threads=args.image_writer_threads,
    )

    episodes = (
        sorted(set(args.episodes)) if args.episodes is not None else list(range(source.meta.total_episodes))
    )
    if not episodes or min(episodes) < 0 or max(episodes) >= source.meta.total_episodes:
        raise ValueError("Requested episode indices are outside the source dataset")
    total_frames = 0
    for episode in episodes:
        ep_row = source.meta.episodes[episode]
        start = int(ep_row["dataset_from_index"])
        end = int(ep_row["dataset_to_index"])
        first_sample = source[start]
        task_name = task_name_for_sample(source, first_sample)
        logger.info("Writing episode %03d (%d frames)", episode, end - start)

        for idx in range(start, end):
            sample = source[idx]
            frame = {
                key: frame_value_for_write(sample[key], source.features[key])
                for key in source.features
                if key not in AUTO_FEATURES
            }
            for camera_key in camera_keys:
                rgb = tensor_to_rgb(sample[camera_key])
                frame[depth_key_for(camera_key, args.depth_prefix)] = infer_depth_rgb(
                    depth_model,
                    rgb,
                    args.input_size,
                )
            frame["task"] = task_name
            target.add_frame(frame)
            total_frames += 1

        target.save_episode()

    logger.info("Finalizing %s", args.output_repo_id)
    target.finalize()
    from ..data.schema import SpatialMetadata, write_dataset_metadata
    from .writer import image_size, inherit_metadata

    inherit_metadata(source, target, episode_ids=episodes)
    write_dataset_metadata(
        target.root,
        {
            depth_key_for(camera, args.depth_prefix): {
                "spatial": SpatialMetadata(
                    camera_key=camera,
                    image_size=image_size(source.features, camera),
                    fps=float(source.fps),
                    source="estimated",
                    generator="depth_anything_v2_small",
                    checkpoint=args.checkpoint.name
                    if args.checkpoint
                    else "depth-anything/Depth-Anything-V2-Small",
                    depth_kind="colorized_relative",
                ).to_dict(),
                "granularity": "frame",
                "depth_normalization": "per_frame_percentile_2_98",
                "metric_depth": False,
            }
            for camera in camera_keys
        },
        source={
            "repo_id": source.repo_id,
            "episode_mapping": {str(new): old for new, old in enumerate(episodes)},
        },
    )

    final_dataset = target
    if args.trim_after_action_release:
        logger.info("Creating release-trimmed derivative of %s.", args.output_repo_id)
        final_dataset = release_trim_generated_dataset(
            repo_id=args.output_repo_id,
            root=Path(target.root),
            output_repo_id=args.trim_output_repo_id,
            output_dir=Path(args.trim_output_dir) if args.trim_output_dir else None,
            overwrite_output_dir=args.overwrite_output_dir,
            no_push=args.no_push,
            public=args.public,
            keep_after_frames=args.trim_keep_after_frames,
            event_threshold=args.trim_event_threshold,
        )
    elif not args.no_push:
        logger.info("Pushing %s to Hugging Face", args.output_repo_id)
        target.push_to_hub(private=not args.public)

    logger.info("Done: %s (%d frames)", target.root, total_frames)
    if final_dataset is not target:
        logger.info("Final trimmed dataset: %s", final_dataset.root)


if __name__ == "__main__":
    main()
