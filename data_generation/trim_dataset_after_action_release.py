#!/usr/bin/env python3
"""Trim LeRobot episode tails after the final action release.

This creates a new dataset from an existing one. For each episode it detects
the final grasp/release action bump adaptively, then keeps frames through the
first sustained return to the baseline/minimum action after that final bump.

Example:
    python trim_dataset_after_action_release.py \
        --source-repo-id edmos7/so100_multitask_80-gt \
        --output-repo-id edmos7/so100_multitask_80-gt-release-trimmed
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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


def _ensure_lerobot() -> None:
    try:
        import lerobot  # noqa: F401
    except ImportError:
        python = Path.home() / "miniforge3" / "envs" / "lerobotclone" / "bin" / "python3.12"
        if not python.exists():
            sys.exit("ERROR: lerobot is unavailable and lerobotclone was not found.")
        sys.exit(subprocess.run([str(python), __file__, *sys.argv[1:]]).returncode)


_ensure_lerobot()

import torch  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.utils.constants import HF_LEROBOT_HOME  # noqa: E402


@dataclass
class TrimBounds:
    start: int
    end: int
    original_length: int
    min_value: float
    peak_value: float
    cutoff_index: int | None

    @property
    def length(self) -> int:
        return self.end - self.start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a LeRobot dataset with episode tails trimmed after final action release.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-repo-id", required=True)
    parser.add_argument(
        "--output-repo-id",
        default=None,
        help="Defaults to <source-repo-id>-release-trimmed.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--action-dims",
        default="auto",
        help=(
            "Action dimensions used for the release signal. Use 'auto', 'all', a comma "
            "list like '-1,5', or a half-open slice like '0:5'. Negative indices "
            "are resolved from the end. Auto uses the named gripper action if present, "
            "otherwise the last action dim."
        ),
    )
    parser.add_argument(
        "--combine",
        choices=("mean", "max", "norm"),
        default="mean",
        help="How to combine multiple selected action dimensions into one signal.",
    )
    parser.add_argument(
        "--min-tolerance-frac",
        type=float,
        default=0.05,
        help="Frames within this fraction of the episode action range count as returned to minimum.",
    )
    parser.add_argument(
        "--min-tolerance-abs",
        type=float,
        default=0.0,
        help="Absolute floor for the minimum-return tolerance.",
    )
    parser.add_argument(
        "--event-threshold",
        type=float,
        default=0.25,
        help=(
            "Normalized threshold passed to the adaptive action-bump detector. "
            "The detector adapts downward for smaller per-episode action magnitudes."
        ),
    )
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--min-gap", type=int, default=8, help="Merge action bumps separated by at most this many frames.")
    parser.add_argument("--min-prominence", type=float, default=0.12, help="Minimum normalized bump prominence above baseline.")
    parser.add_argument("--baseline-quantile", type=float, default=10.0, help="Percentile used as the release baseline.")
    parser.add_argument(
        "--baseline-tolerance",
        type=float,
        default=0.05,
        help="Normalized margin above baseline considered returned to minimum.",
    )
    parser.add_argument(
        "--return-sustain-frames",
        type=int,
        default=3,
        help="Near-baseline frames required after the final bump to count as closed/released.",
    )
    parser.add_argument(
        "--cutoff-mode",
        choices=("first-return", "bump-end"),
        default="first-return",
        help=(
            "Where to cut after the final bump. 'first-return' waits until the "
            "gripper/action signal has returned to minimum; 'bump-end' cuts at "
            "the detector's final active frame."
        ),
    )
    parser.add_argument(
        "--search-after-peak",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only accept minimum returns occurring after the final bump peak.",
    )
    parser.add_argument(
        "--keep-after-frames",
        type=int,
        default=0,
        help="Extra frames retained after the detected return-to-minimum timestep.",
    )
    parser.add_argument(
        "--min-episode-frames",
        type=int,
        default=8,
        help="Never produce an episode shorter than this.",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Print proposed bounds without creating a dataset.",
    )
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--public", action="store_true")
    return parser.parse_args()


def _to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def find_gripper_action_index(dataset: LeRobotDataset) -> int | None:
    names = dataset.meta.names.get("action", [])
    for candidate in ("so101_gripper", "so100_gripper", "gripper"):
        for i, name in enumerate(names):
            if candidate.lower() in name.lower():
                return i
    return None


def normalize_action_signal(dataset: LeRobotDataset, actions: np.ndarray, action_idx: int) -> np.ndarray:
    signal = actions[:, action_idx].astype(np.float32)
    stats = dataset.meta.stats.get("action")
    if stats is None:
        lo = float(signal.min(initial=0.0))
        hi = float(signal.max(initial=1.0))
    else:
        lo = float(_to_numpy(stats["min"])[action_idx])
        hi = float(_to_numpy(stats["max"])[action_idx])
    return np.clip((signal - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def smooth_1d(signal: np.ndarray, window: int = 5) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    if window <= 1 or len(signal) < 3:
        return signal.copy()
    window = min(window, len(signal))
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return signal.copy()
    pad = window // 2
    padded = np.pad(signal, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def action_bump_segments(
    action_norm: np.ndarray,
    threshold: float = 0.25,
    smooth_window: int = 5,
    min_gap: int = 8,
    min_prominence: float = 0.12,
) -> list[dict[str, int | float]]:
    """Return distinct action bumps with rise, peak, fall, and end indices."""
    y = smooth_1d(action_norm, smooth_window)
    if len(y) == 0:
        return []

    baseline = float(np.percentile(y, 10))
    amplitude = float(y.max(initial=baseline) - baseline)
    if amplitude <= 1e-6:
        return []

    cutoff = max(
        baseline + min_prominence,
        min(float(threshold), baseline + 0.35 * amplitude),
    )
    active = y >= cutoff
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    ends = np.flatnonzero(active & ~np.r_[active[1:], False])

    if len(starts) == 0:
        maxima = np.flatnonzero((y[1:-1] >= y[:-2]) & (y[1:-1] > y[2:])) + 1
        maxima = maxima[y[maxima] >= baseline + min_prominence]
        return [
            {
                "rise": int(max(0, i - 1)),
                "peak": int(i),
                "fall": int(min(len(y) - 1, i + 1)),
                "end": int(min(len(y) - 1, i + 1)),
                "height": float(y[i]),
            }
            for i in maxima
        ]

    merged: list[tuple[int, int]] = []
    for start, end in zip(starts, ends):
        start_i, end_i = int(start), int(end)
        if merged and start_i - merged[-1][1] <= min_gap:
            merged[-1] = (merged[-1][0], end_i)
        else:
            merged.append((start_i, end_i))

    bumps: list[dict[str, int | float]] = []
    dy = np.diff(y, prepend=y[0])
    slope_eps = max(0.005, 0.02 * amplitude)
    for start, end in merged:
        segment = y[start : end + 1]
        peak = start + int(np.argmax(segment))
        fall_candidates = np.flatnonzero(dy[peak + 1 : end + 1] < -slope_eps)
        fall = (
            int(peak + 1 + fall_candidates[0])
            if len(fall_candidates)
            else int(min(end, peak + 1))
        )
        bumps.append(
            {
                "rise": int(start),
                "peak": int(peak),
                "fall": fall,
                "end": int(end),
                "height": float(y[peak]),
            }
        )
    return [b for b in bumps if float(b["height"]) >= baseline + min_prominence]


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


def _parse_action_dims(spec: str, dim: int, auto_dim: int | None = None) -> np.ndarray:
    spec = spec.strip().lower()
    if spec == "auto":
        dims = np.array([auto_dim if auto_dim is not None else dim - 1])
    elif spec == "all":
        dims = np.arange(dim)
    elif ":" in spec:
        parts = spec.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid --action-dims slice: {spec!r}")
        start = int(parts[0]) if parts[0] else 0
        stop = int(parts[1]) if parts[1] else dim
        if start < 0:
            start += dim
        if stop < 0:
            stop += dim
        dims = np.arange(start, stop)
    else:
        dims = np.array([int(part.strip()) for part in spec.split(",") if part.strip()])
        dims = np.where(dims < 0, dims + dim, dims)
    if len(dims) == 0 or np.any(dims < 0) or np.any(dims >= dim):
        raise ValueError(f"--action-dims must select dimensions in [0, {dim}): {spec!r}")
    return dims.astype(np.int64)


def _release_signal(
    dataset: LeRobotDataset,
    actions: np.ndarray,
    dims: np.ndarray,
    combine: str,
) -> np.ndarray:
    if len(dims) == 1:
        return normalize_action_signal(dataset, actions, int(dims[0]))

    selected = actions[:, dims]
    stats = dataset.meta.stats.get("action")
    if stats is not None:
        lo = np.asarray(stats["min"], dtype=np.float32).reshape(-1)[dims]
        hi = np.asarray(stats["max"], dtype=np.float32).reshape(-1)[dims]
    else:
        lo = selected.min(axis=0)
        hi = selected.max(axis=0)
    selected = np.clip((selected - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)
    if combine == "mean":
        return selected.mean(axis=1)
    if combine == "max":
        return selected.max(axis=1)
    return np.linalg.norm(selected, axis=1)


def compute_trim_bounds(
    dataset: LeRobotDataset,
    actions: np.ndarray,
    args: argparse.Namespace,
) -> TrimBounds:
    length = len(actions)
    grip_idx = find_gripper_action_index(dataset)
    dims = _parse_action_dims(args.action_dims, actions.shape[1], grip_idx)
    signal = _release_signal(dataset, actions, dims, args.combine)
    if length <= args.min_episode_frames:
        return TrimBounds(0, length, length, float(signal.min()), float(signal.max()), None)

    min_value = float(signal.min())
    peak_value = float(signal.max())
    bumps = action_bump_segments(
        signal,
        threshold=args.event_threshold,
        smooth_window=args.smooth_window,
        min_gap=args.min_gap,
        min_prominence=args.min_prominence,
    )
    if not bumps:
        logger.warning("No clear release/action bump detected; keeping complete episode.")
        return TrimBounds(0, length, length, min_value, peak_value, None)

    last_bump = bumps[-1]
    if args.cutoff_mode == "bump-end":
        cutoff_index = int(last_bump["end"])
        end = min(length, cutoff_index + 1 + max(0, args.keep_after_frames))
        if end < args.min_episode_frames:
            end = min(length, args.min_episode_frames)
        return TrimBounds(0, end, length, min_value, peak_value, cutoff_index)

    search_start = int(last_bump["peak"] if args.search_after_peak else last_bump["rise"])
    baseline = float(np.percentile(signal, args.baseline_quantile))
    tolerance = max(args.min_tolerance_abs, args.min_tolerance_frac * (peak_value - baseline), args.baseline_tolerance)
    near_baseline = signal <= baseline + tolerance
    sustain = max(1, args.return_sustain_frames)
    candidates = []
    for idx in range(search_start, length):
        if bool(near_baseline[idx : min(length, idx + sustain)].all()) and idx + sustain <= length:
            candidates.append(idx)

    if len(candidates) == 0:
        logger.warning("Action signal never returned to baseline after final bump; keeping complete episode.")
        return TrimBounds(0, length, length, min_value, peak_value, None)

    cutoff_index = int(candidates[0])
    end = min(length, cutoff_index + 1 + max(0, args.keep_after_frames))
    if end < args.min_episode_frames:
        end = min(length, args.min_episode_frames)

    return TrimBounds(0, end, length, min_value, peak_value, cutoff_index)


def analyze_bounds(dataset: LeRobotDataset, args: argparse.Namespace) -> list[TrimBounds]:
    if "action" not in dataset.features:
        raise KeyError("Dataset must contain 'action'.")

    bounds = []
    for ep_id in range(dataset.meta.total_episodes):
        actions = _episode_column(dataset, ep_id, "action")
        bound = compute_trim_bounds(dataset, actions, args)
        bounds.append(bound)
        logger.info(
            "episode %03d: %4d -> %4d frames | remove end=%3d | cutoff=%s | min=%.4f peak=%.4f",
            ep_id,
            bound.original_length,
            bound.length,
            bound.original_length - bound.end,
            "none" if bound.cutoff_index is None else str(bound.cutoff_index),
            bound.min_value,
            bound.peak_value,
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
    return LeRobotDataset(repo_id=output_repo_id, root=output_dir)


def main() -> None:
    args = parse_args()
    output_repo_id = args.output_repo_id or f"{args.source_repo_id}-release-trimmed"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else HF_LEROBOT_HOME / output_repo_id
    )

    if output_repo_id == args.source_repo_id:
        sys.exit("ERROR: output repo id must differ from source repo id.")

    logger.info("Loading %s ...", args.source_repo_id)
    source = LeRobotDataset(args.source_repo_id)
    source._ensure_hf_dataset_loaded()
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
                f"ERROR: output directory exists: {output_dir}\n"
                "Use --overwrite-output-dir to replace it."
            )
        shutil.rmtree(output_dir)

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
