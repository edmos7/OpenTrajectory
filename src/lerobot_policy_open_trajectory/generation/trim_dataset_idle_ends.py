#!/usr/bin/env python3
"""Trim stationary prefixes and suffixes from LeRobot episodes.

The script creates a new dataset; it never modifies the source. Trimming is
based only on contiguous idle regions at episode boundaries. Interior pauses
and low-motion observations near grasp/place goals are retained.

For datasets containing ``observation.global_trajectory``, each episode's
trajectory is sliced with the same frame bounds and then padded to the longest
trimmed episode so the feature remains fixed-shape.

Example:
    python trim_dataset_idle_ends.py \
        --source-repo-id edmos7/so100_multitask_80-gt \
        --output-repo-id edmos7/so100_multitask_80-gt-trimmed
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .preprocessing import remove_output_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

AUTO_FEATURES = {
    "index",
    "frame_index",
    "episode_index",
    "task_index",
    "timestamp",
}
TRAJECTORY_KEY = "observation.global_trajectory"


import torch  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.utils.constants import HF_LEROBOT_HOME  # noqa: E402


@dataclass
class TrimBounds:
    start: int
    end: int
    original_length: int

    @property
    def length(self) -> int:
        return self.end - self.start


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a LeRobot dataset with stationary episode boundaries trimmed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-repo-id", required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument(
        "--output-repo-id",
        default=None,
        help="Defaults to <source-repo-id>-trimmed.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=0.012,
        help="Normalized per-frame action-change norm considered active.",
    )
    parser.add_argument(
        "--state-motion-threshold",
        type=float,
        default=0.008,
        help="Normalized per-frame measured-state change considered active.",
    )
    parser.add_argument(
        "--activity-dims",
        default="all",
        help=(
            "Action/state dimensions used for boundary activity detection. Use "
            "'all', a comma list like '0,2,4', or a half-open slice like '0:5'. "
            "For SO100, '0:5' uses arm joints and ignores the gripper."
        ),
    )
    parser.add_argument(
        "--no-action-threshold",
        type=float,
        default=0.01,
        help=(
            "Normalized norm(action - observation.state[:action_dim]) below which "
            "a frame is considered a no-op command for trimming trailing idle suffixes."
        ),
    )
    parser.add_argument(
        "--sustain-frames",
        type=int,
        default=4,
        help="Active frames required inside the detection window.",
    )
    parser.add_argument(
        "--detection-window",
        type=int,
        default=8,
        help="Window used to reject isolated command jitter.",
    )
    parser.add_argument(
        "--keep-before-frames",
        type=int,
        default=15,
        help="Context retained before the first sustained active region.",
    )
    parser.add_argument(
        "--keep-after-frames",
        type=int,
        default=45,
        help="Context retained after the last sustained active region.",
    )
    parser.add_argument(
        "--min-episode-frames",
        type=int,
        default=32,
        help="Never produce an episode shorter than this.",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Print proposed bounds without creating a dataset.",
    )
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument(
        "--push",
        dest="no_push",
        action="store_false",
        help="Explicitly upload the generated dataset to the Hub.",
    )
    parser.add_argument("--no-push", action="store_true", default=True)
    parser.add_argument("--public", action="store_true")
    return parser.parse_args(argv)


def _to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _frame_value_for_write(value, feature: dict):
    """Convert decoded dataset values back to the recorder's expected layout."""
    if feature.get("dtype") in {"image", "video"}:
        array = _to_numpy(value)
        expected = tuple(feature["shape"])
        if array.ndim == 3 and array.shape[0] in (1, 3) and expected[-1] in (1, 3):
            array = np.moveaxis(array, 0, -1)
        if np.issubdtype(array.dtype, np.floating) and array.max(initial=0) <= 1.0:
            array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
        return array
    array = _to_numpy(value)
    expected = tuple(feature.get("shape", ()))
    if expected and array.shape != expected and array.size == int(np.prod(expected)):
        array = array.reshape(expected)
    return array


def _episode_column(dataset: LeRobotDataset, ep_id: int, key: str) -> np.ndarray:
    ep = dataset.meta.episodes[ep_id]
    start = int(ep["dataset_from_index"])
    end = int(ep["dataset_to_index"])
    values = dataset.hf_dataset.select(range(start, end))[key]
    return np.stack([_to_numpy(value) for value in values]).astype(np.float32)


def _feature_scale(dataset: LeRobotDataset, dim: int) -> np.ndarray:
    stats = dataset.meta.stats.get("action", {})
    if "min" in stats and "max" in stats:
        scale = _to_numpy(stats["max"]).reshape(-1) - _to_numpy(stats["min"]).reshape(-1)
        if len(scale) == dim:
            return np.maximum(scale.astype(np.float32), 1.0)
    return np.ones(dim, dtype=np.float32)


def _parse_activity_dims(spec: str, dim: int) -> np.ndarray:
    spec = spec.strip().lower()
    if spec == "all":
        return np.arange(dim)
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid --activity-dims slice: {spec!r}")
        start = int(parts[0]) if parts[0] else 0
        stop = int(parts[1]) if parts[1] else dim
        dims = np.arange(start, stop)
    else:
        dims = np.array([int(part.strip()) for part in spec.split(",") if part.strip()])
    if len(dims) == 0 or np.any(dims < 0) or np.any(dims >= dim):
        raise ValueError(f"--activity-dims must select dimensions in [0, {dim}): {spec!r}")
    return dims


def _sustained_activity(active: np.ndarray, window: int, required: int) -> np.ndarray:
    if len(active) == 0:
        return active
    window = max(1, min(window, len(active)))
    required = max(1, min(required, window))
    counts = np.convolve(active.astype(np.int32), np.ones(window, dtype=np.int32), mode="same")
    return counts >= required


def _trailing_no_action_start(
    actions: np.ndarray,
    states: np.ndarray,
    scale: np.ndarray,
    threshold: float,
) -> int:
    """Return the first frame of a contiguous trailing no-op command suffix."""
    action_dim = actions.shape[1]
    if states.ndim != 2 or states.shape[1] < action_dim:
        return len(actions)

    command_delta = actions - states[:, :action_dim]
    normalized_command_delta = np.linalg.norm(command_delta / scale, axis=1)
    no_action = normalized_command_delta <= threshold

    suffix_start = len(no_action)
    while suffix_start > 0 and bool(no_action[suffix_start - 1]):
        suffix_start -= 1
    return suffix_start


def compute_trim_bounds(
    actions: np.ndarray,
    states: np.ndarray,
    scale: np.ndarray,
    args: argparse.Namespace,
) -> TrimBounds:
    length = len(actions)
    if length <= args.min_episode_frames:
        return TrimBounds(0, length, length)

    action_step = np.zeros_like(actions)
    action_step[1:] = actions[1:] - actions[:-1]
    state_step = np.zeros_like(states)
    state_step[1:] = states[1:] - states[:-1]
    activity_dims = _parse_activity_dims(args.activity_dims, actions.shape[1])
    normalized_step = np.linalg.norm(
        action_step[:, activity_dims] / scale[activity_dims],
        axis=1,
    )
    normalized_state_step = np.linalg.norm(
        state_step[:, activity_dims] / scale[activity_dims],
        axis=1,
    )
    active = (normalized_step >= args.motion_threshold) | (
        normalized_state_step >= args.state_motion_threshold
    )
    sustained = _sustained_activity(active, args.detection_window, args.sustain_frames)
    indices = np.flatnonzero(sustained)
    if len(indices) == 0:
        logger.warning("No sustained activity detected; keeping complete episode.")
        return TrimBounds(0, length, length)

    start = max(0, int(indices[0]) - args.keep_before_frames)
    end = min(length, int(indices[-1]) + 1 + args.keep_after_frames)

    trailing_no_action_start = _trailing_no_action_start(
        actions,
        states,
        scale,
        args.no_action_threshold,
    )
    if 0 < trailing_no_action_start < end:
        end = trailing_no_action_start

    if end - start < args.min_episode_frames:
        missing = args.min_episode_frames - (end - start)
        start = max(0, start - missing // 2)
        end = min(length, end + missing - missing // 2)
        start = max(0, end - args.min_episode_frames)

    return TrimBounds(start, end, length)


def analyze_bounds(dataset: LeRobotDataset, args: argparse.Namespace) -> list[TrimBounds]:
    if "action" not in dataset.features or "observation.state" not in dataset.features:
        raise KeyError("Dataset must contain 'action' and 'observation.state'.")

    action_dim = int(dataset.features["action"]["shape"][0])
    scale = _feature_scale(dataset, action_dim)
    bounds = []
    for ep_id in range(dataset.meta.total_episodes):
        actions = _episode_column(dataset, ep_id, "action")
        states = _episode_column(dataset, ep_id, "observation.state")
        bound = compute_trim_bounds(actions, states, scale, args)
        bounds.append(bound)
        logger.info(
            "episode %03d: %4d -> %4d frames | remove start=%3d end=%3d | keep [%d:%d]",
            ep_id,
            bound.original_length,
            bound.length,
            bound.start,
            bound.original_length - bound.end,
            bound.start,
            bound.end,
        )

    original = sum(bound.original_length for bound in bounds)
    retained = sum(bound.length for bound in bounds)
    logger.info(
        "Total: %d -> %d frames; removed %d (%.1f%%). Length range after trim: %d-%d",
        original,
        retained,
        original - retained,
        100.0 * (original - retained) / max(original, 1),
        min(bound.length for bound in bounds),
        max(bound.length for bound in bounds),
    )
    return bounds


def build_trimmed_trajectories(
    dataset: LeRobotDataset,
    bounds: list[TrimBounds],
) -> tuple[dict[int, np.ndarray], int]:
    trajectories = {}
    max_length = max(bound.length for bound in bounds)
    for ep_id, bound in enumerate(bounds):
        original = _episode_column(dataset, ep_id, TRAJECTORY_KEY)[0]
        valid_length = min(bound.original_length, len(original))
        sliced = original[bound.start : min(bound.end, valid_length)]
        if len(sliced) == 0:
            sliced = original[:1]
        if len(sliced) < max_length:
            sliced = np.concatenate(
                [sliced, np.repeat(sliced[-1:], max_length - len(sliced), axis=0)],
                axis=0,
            )
        trajectories[ep_id] = sliced.astype(np.float32)
    return trajectories, max_length


def copy_trimmed_dataset(
    source: LeRobotDataset,
    bounds: list[TrimBounds],
    output_repo_id: str,
    output_dir: Path,
) -> LeRobotDataset:
    features = {key: dict(value) for key, value in source.features.items()}
    trajectories = None
    if TRAJECTORY_KEY in features:
        trajectories, max_length = build_trimmed_trajectories(source, bounds)
        features[TRAJECTORY_KEY]["shape"] = (max_length, 3)
        logger.info("Trimmed global trajectory fixed length: %d", max_length)

    target = LeRobotDataset.create(
        repo_id=output_repo_id,
        root=output_dir,
        fps=source.fps,
        features=features,
        robot_type=source.meta.robot_type,
        use_videos=bool(source.meta.video_keys),
        image_writer_threads=4,
    )

    for ep_id, bound in enumerate(bounds):
        ep_meta = source.meta.episodes[ep_id]
        global_start = int(ep_meta["dataset_from_index"])
        task_name = source.meta.tasks.iloc[int(source.hf_dataset[global_start]["task_index"])].name

        logger.info("Writing episode %03d (%d frames)...", ep_id, bound.length)
        for local_idx in range(bound.start, bound.end):
            sample = source[global_start + local_idx]
            frame = {
                key: _frame_value_for_write(sample[key], source.features[key])
                for key in source.features
                if key not in AUTO_FEATURES
            }
            if trajectories is not None:
                frame[TRAJECTORY_KEY] = trajectories[ep_id]
            frame["task"] = task_name
            target.add_frame(frame)
        target.save_episode()

    target.finalize()
    from .writer import inherit_trimmed_metadata

    inherit_trimmed_metadata(source, target, bounds)
    return LeRobotDataset(repo_id=output_repo_id, root=output_dir)


def main(argv=None) -> None:
    args = parse_args(argv)
    output_repo_id = args.output_repo_id or f"{args.source_repo_id}-trimmed"
    output_dir = Path(args.output_dir) if args.output_dir else HF_LEROBOT_HOME / output_repo_id

    if output_repo_id == args.source_repo_id:
        sys.exit("ERROR: output repo id must differ from source repo id.")

    logger.info("Loading %s ...", args.source_repo_id)
    source = LeRobotDataset(args.source_repo_id, root=args.source_root)
    logger.info(
        "Source: %d episodes, %d frames, %d fps",
        source.meta.total_episodes,
        len(source),
        source.fps,
    )

    bounds = analyze_bounds(source, args)
    if args.analysis_only:
        return

    if output_dir.exists():
        if not args.overwrite_output_dir:
            sys.exit(
                f"ERROR: output directory exists: {output_dir}\nUse --overwrite-output-dir to replace it."
            )
        remove_output_dataset(output_dir, source=source.root)

    target = copy_trimmed_dataset(source, bounds, output_repo_id, output_dir)
    logger.info(
        "Created %s: %d episodes, %d frames",
        output_repo_id,
        target.meta.total_episodes,
        len(target),
    )

    if not args.no_push:
        logger.info("Pushing %s to Hub ...", output_repo_id)
        target.push_to_hub(private=not args.public)
        logger.info("Done: https://huggingface.co/datasets/%s", output_repo_id)
    else:
        logger.info("Done locally: %s", target.root)


if __name__ == "__main__":
    main()
