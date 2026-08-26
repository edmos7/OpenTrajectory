from __future__ import annotations

from lerobot_policy_open_trajectory.cli import build_parser


def test_augment_cli_defaults() -> None:
    args = build_parser().parse_args(
        [
            "dataset",
            "augment",
            "--repo-id",
            "source/dataset",
            "--artifact-dir",
            "artifacts",
            "--output-repo-id",
            "output/dataset",
        ]
    )

    assert args.features == ("sequence", "rt-image")
    assert args.trajectory_length == 224


def test_generate_cli_forwards_backend_arguments() -> None:
    args = build_parser().parse_args(["dataset", "generate", "--kind", "tracks", "--", "--episodes", "0"])

    assert args.kind == "tracks"
    assert args.arguments == ["--", "--episodes", "0"]


def test_render_episode_cli_is_object_name_agnostic() -> None:
    args = build_parser().parse_args(
        [
            "visuals",
            "render-episode",
            "--artifact",
            "episode.npz",
            "--video",
            "camera.mp4",
            "--output-dir",
            "visuals",
            "--camera",
            "main-view",
            "--persistent-point-count",
            "16",
        ]
    )

    assert args.camera == "main-view"
    assert args.persistent_point_count == 16
    assert args.frame_step == 4
