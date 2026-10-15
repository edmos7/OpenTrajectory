"""Command-line entry point for dataset and visualization workflows."""

from __future__ import annotations

import argparse
import logging
from importlib import import_module

from .data import augment_dataset
from .visuals import render_artifact_visuals, render_episode_visuals

LOG = logging.getLogger("open-trajectory")


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _generation_command(name: str, arguments: list[str]) -> int:
    """Load an installed backend without relying on the working directory."""
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    module = import_module(f".generation.{name.removesuffix('.py')}", __package__)
    module.main(arguments)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="open-trajectory")
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    predictor = commands.add_parser("predictor", help="Learned future-track models")
    predictor_commands = predictor.add_subparsers(dest="predictor_command", required=True)
    train_predictor = predictor_commands.add_parser("train", help="Train with episode-disjoint validation")
    train_predictor.add_argument("arguments", nargs=argparse.REMAINDER)

    dataset = commands.add_parser("dataset", help="Generate or modify LeRobot datasets")
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)

    augment = dataset_commands.add_parser("augment", help="Add validated trajectory features")
    augment.add_argument("--repo-id", required=True)
    augment.add_argument("--artifact-dir", required=True)
    augment.add_argument("--output-repo-id", required=True)
    augment.add_argument("--root")
    augment.add_argument("--revision")
    augment.add_argument("--output-dir")
    augment.add_argument("--features", type=_csv, default=("sequence", "rt-image"))
    augment.add_argument("--camera-keys", type=_csv, default=())
    augment.add_argument("--trajectory-length", type=int, default=224)
    augment.add_argument("--keep-legacy-task-id", action="store_true")

    generate = dataset_commands.add_parser(
        "generate",
        help="Generate trajectory conditioning and point tracks",
    )
    generate.add_argument(
        "--kind",
        choices=("sequence", "rt-image", "tracks", "tracks-and-sequence"),
        default="sequence",
    )
    generate.add_argument("arguments", nargs=argparse.REMAINDER)

    trim = dataset_commands.add_parser("trim", help="Trim post-release episode tails")
    trim.add_argument("--mode", choices=("release", "idle"), default="release")
    trim.add_argument("arguments", nargs=argparse.REMAINDER)

    depth = dataset_commands.add_parser("add-depth", help="Add Depth Anything V2 features")
    depth.add_argument("arguments", nargs=argparse.REMAINDER)

    visuals = commands.add_parser("visuals", help="Render README/debug visuals")
    visual_commands = visuals.add_subparsers(dest="visual_command", required=True)
    render = visual_commands.add_parser("render", help="Render an episode artifact")
    render.add_argument("--artifact", required=True)
    render.add_argument("--frame", required=True)
    render.add_argument("--output-dir", required=True)
    render_episode = visual_commands.add_parser(
        "render-episode", help="Render trajectory and point-track visuals for a complete episode"
    )
    render_episode.add_argument("--artifact", required=True)
    render_episode.add_argument("--video", required=True)
    render_episode.add_argument("--output-dir", required=True)
    render_episode.add_argument("--camera")
    render_episode.add_argument("--persistent-point-count", type=int, default=0)
    render_episode.add_argument("--frame-step", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    if args.command == "predictor":
        from .predictors.training import main as train

        train(args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments)
        return 0
    if args.command == "dataset" and args.dataset_command == "augment":
        output = augment_dataset(
            repo_id=args.repo_id,
            artifact_dir=args.artifact_dir,
            output_repo_id=args.output_repo_id,
            output_dir=args.output_dir,
            root=args.root,
            revision=args.revision,
            features=args.features,
            trajectory_length=args.trajectory_length,
            camera_keys=args.camera_keys,
            drop_legacy_task_id=not args.keep_legacy_task_id,
        )
        LOG.info("Created %s at %s", output.repo_id, output.root)
        return 0
    if args.command == "dataset" and args.dataset_command == "generate":
        scripts = {
            "sequence": "generate_global_trajectory_dataset.py",
            "rt-image": "generate_trajectory_image_dataset.py",
            "tracks": "generate_track_dataset.py",
            "tracks-and-sequence": "generate_tracks_and_trajectory_dataset.py",
        }
        return _generation_command(scripts[args.kind], args.arguments)
    if args.command == "dataset" and args.dataset_command == "trim":
        module = "trim_dataset_after_action_release" if args.mode == "release" else "trim_dataset_idle_ends"
        return _generation_command(module, args.arguments)
    if args.command == "dataset" and args.dataset_command == "add-depth":
        return _generation_command("add_depth_features", args.arguments)
    if args.command == "visuals" and args.visual_command == "render":
        for path in render_artifact_visuals(args.artifact, args.frame, args.output_dir):
            LOG.info("Wrote %s", path)
        return 0
    if args.command == "visuals" and args.visual_command == "render-episode":
        for path in render_episode_visuals(
            args.artifact,
            args.video,
            args.output_dir,
            camera=args.camera,
            persistent_point_count=args.persistent_point_count,
            frame_step=args.frame_step,
        ):
            LOG.info("Wrote %s", path)
        return 0
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
