"""Shared dataset preprocessing helpers for generation scripts."""

from __future__ import annotations

import argparse
import logging
import uuid
from pathlib import Path


def remove_output_dataset(path: Path, *, source: Path) -> None:
    """Make room for explicitly requested replacement, retaining a recoverable backup."""
    path, source = Path(path).resolve(), Path(source).resolve()
    if path == source or path in source.parents or source in path.parents:
        raise ValueError("Output must not overlap the source dataset")
    if not (path / "meta/info.json").is_file():
        raise ValueError(f"Refusing to replace a directory without LeRobot metadata: {path}")
    backup = path.with_name(f"{path.name}.backup-{uuid.uuid4().hex[:12]}")
    path.rename(backup)
    logging.getLogger(__name__).warning("Previous output retained at %s", backup)


def add_release_trim_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("release trimming")
    group.add_argument(
        "--trim-after-action-release",
        action="store_true",
        help=(
            "After generating the dataset, create a release-trimmed derivative "
            "that removes post-completion no-action tails."
        ),
    )
    group.add_argument(
        "--trim-output-repo-id",
        default=None,
        help="Repo id for the release-trimmed dataset. Defaults to <output-repo-id>-release-trimmed.",
    )
    group.add_argument(
        "--trim-output-dir",
        default=None,
        help="Local output dir for the release-trimmed dataset.",
    )
    group.add_argument(
        "--trim-keep-after-frames",
        type=int,
        default=0,
        help="Extra frames retained after the detected final release/action return.",
    )
    group.add_argument(
        "--trim-event-threshold",
        type=float,
        default=0.25,
        help="Normalized threshold passed to the adaptive release/action-bump detector.",
    )


def release_trim_generated_dataset(
    *,
    repo_id: str,
    root: Path,
    output_repo_id: str | None,
    output_dir: Path | None,
    overwrite_output_dir: bool,
    no_push: bool,
    public: bool,
    keep_after_frames: int,
    event_threshold: float,
):
    """Create a release-trimmed derivative of a freshly generated LeRobot dataset."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME

    from lerobot_policy_open_trajectory.generation.trim_dataset_after_action_release import (
        analyze_bounds,
        copy_trimmed_dataset,
    )

    trim_repo_id = output_repo_id or f"{repo_id}-release-trimmed"
    trim_dir = output_dir if output_dir is not None else HF_LEROBOT_HOME / trim_repo_id

    if trim_dir.exists():
        if not overwrite_output_dir:
            raise FileExistsError(
                f"Trim output directory exists: {trim_dir}. Use --overwrite-output-dir to replace it."
            )
        source_root = Path(root).resolve()
        trim_root = trim_dir.resolve()
        if trim_root == source_root or source_root in trim_root.parents or trim_root in source_root.parents:
            raise ValueError(
                f"Refusing to remove trim output directory because it overlaps source root: {trim_dir}"
            )
        remove_output_dataset(trim_dir, source=root)

    source = LeRobotDataset(repo_id=repo_id, root=root)
    trim_args = argparse.Namespace(
        action_dims="auto",
        combine="mean",
        min_tolerance_frac=0.05,
        min_tolerance_abs=0.0,
        event_threshold=event_threshold,
        smooth_window=5,
        min_gap=8,
        min_prominence=0.12,
        baseline_quantile=10.0,
        baseline_tolerance=0.05,
        return_sustain_frames=3,
        cutoff_mode="first-return",
        search_after_peak=True,
        keep_after_frames=keep_after_frames,
        min_episode_frames=8,
    )
    bounds = analyze_bounds(source, trim_args)
    trimmed = copy_trimmed_dataset(source, bounds, trim_repo_id, trim_dir)
    if not no_push:
        trimmed.push_to_hub(private=not public)
    return trimmed
