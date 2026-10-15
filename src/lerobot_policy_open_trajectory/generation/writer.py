"""One feature writer for installed generators and stock LeRobot."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ..data.augment import _slug
from ..data.schema import METADATA_FILE, SpatialMetadata, read_dataset_metadata, write_dataset_metadata


def image_size(features: dict, camera: str) -> tuple[int, int]:
    feature = features[camera]
    shape = feature["shape"]
    names = feature.get("names")
    if isinstance(names, (list, tuple)) and "height" in names and "width" in names:
        return int(shape[names.index("height")]), int(shape[names.index("width")])
    raise ValueError(f"Camera {camera!r} must describe height/width axes; refusing to guess normalization")


def inherit_metadata(source, target, *, episode_ids=None) -> None:
    path = Path(source.root) / METADATA_FILE
    if path.exists():
        payload = read_dataset_metadata(source.root)
        if episode_ids is not None:
            for descriptor in payload["features"].values():
                if "episodes" in descriptor:
                    descriptor["episodes"] = {
                        str(new): descriptor["episodes"].get(str(old))
                        for new, old in enumerate(sorted(episode_ids))
                    }
        write_dataset_metadata(target.root, payload["features"], source=payload["source"])


def inherit_trimmed_metadata(source, target, bounds) -> None:
    if not (Path(source.root) / METADATA_FILE).exists():
        return
    payload = read_dataset_metadata(source.root)
    features = payload["features"]
    for key, descriptor in features.items():
        if key in target.features:
            descriptor["shape"] = list(target.features[key]["shape"])
            descriptor["temporal_transform"] = "trimmed; original horizon annotations retained"
    write_dataset_metadata(
        target.root,
        features,
        source={
            "repo_id": source.repo_id,
            "trim_bounds": {
                str(i): {"start": bound.start, "end_exclusive": bound.end} for i, bound in enumerate(bounds)
            },
            "parent": payload["source"],
        },
    )


def copy_selected_features(dataset, features, episode_ids, *, repo_id, output_dir=None):
    """Copy selected episodes using LeRobot's frame API, preserving multidimensional arrays."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from .add_depth_features import AUTO_FEATURES, frame_value_for_write

    selected = sorted(set(episode_ids))
    if not selected or selected[0] < 0 or selected[-1] >= dataset.meta.total_episodes:
        raise ValueError("Selected episodes are outside the source dataset")
    feature_specs = {**dataset.features, **{key: spec for key, (_, spec) in features.items()}}
    target = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_dir,
        fps=dataset.fps,
        features=feature_specs,
        robot_type=dataset.meta.robot_type,
        use_videos=bool(dataset.meta.video_keys),
        image_writer_threads=4,
    )
    indices = {}
    for index, ep in enumerate(dataset.hf_dataset["episode_index"]):
        if int(ep) in selected:
            indices.setdefault(int(ep), []).append(index)
    for episode in selected:
        for index in indices[episode]:
            row = dataset[index]
            frame = {
                key: frame_value_for_write(row[key], spec)
                for key, spec in dataset.features.items()
                if key not in AUTO_FEATURES
            }
            for key, (values, _spec) in features.items():
                frame[key] = values(row, episode, int(row["frame_index"]))
            frame["task"] = row["task"]
            target.add_frame(frame)
        target.save_episode()
    target.finalize()
    return LeRobotDataset(repo_id=repo_id, root=target.root)


def add_spatial_features(dataset, features, *, options, **kwargs):
    from lerobot.datasets.dataset_tools import add_features

    main_camera = options.get("main_cam_key") or options.get("camera_key")
    secondary = options.get("sec_cam_key")
    cameras = {
        "observation.main_cam_point_tracks": main_camera,
        "observation.sec_cam_point_tracks": secondary,
    }
    converted, metadata = {}, {}
    stride = options.get("frame_stride", 1)
    for old_key, (values, spec) in features.items():
        camera = cameras.get(old_key, main_camera)
        if camera not in dataset.meta.features:
            raise ValueError(f"Source camera {camera!r} is not present in dataset metadata")
        key = f"observation.point_tracks.{_slug(camera)}" if old_key in cameras else old_key
        if key in converted:
            raise ValueError(f"Camera feature name collision: {key}")
        if isinstance(values, Mapping):
            # Episode-keyed mappings are not yet supported by released LeRobot.
            def callback(row, episode_index, frame_index, *, mapping=values):
                if episode_index not in mapping:
                    raise ValueError(f"Missing annotation for episode {episode_index}")
                return mapping[episode_index]

            values = callback
        converted[key] = (values, spec)
        is_track = key.startswith("observation.point_tracks.")
        spatial = SpatialMetadata(
            camera_key=camera,
            image_size=image_size(dataset.meta.features, camera),
            frame_stride=1,  # stored rows remain at source FPS, even when tracker input is strided
            horizon_stride=stride if is_track else 1,
            fps=float(dataset.fps),
            generator="segmentation_tracking_pipeline",
            checkpoint=Path(options["tapnext_checkpoint"]).name
            if options.get("tapnext_checkpoint")
            else None,
        )
        metadata[key] = {
            "spatial": spatial.to_dict(),
            "granularity": "frame" if is_track else "episode",
            "shape": list(spec["shape"]),
            "legacy_alias": old_key if key != old_key else None,
            "horizon_includes_current": is_track,
            "anchor_stride": stride if is_track else 1,
            "anchor_rule": "floor(frame_index / anchor_stride) * anchor_stride" if is_track else None,
            "active_semantics": None if is_track else "grasp_release_interval",
        }
        if is_track:
            count = int(spec["shape"][0])
            robot_count = int(options.get("num_robot_points", count)) // 2 if camera == main_camera else 0
            metadata[key]["point_groups"] = [
                {"kind": "robot", "start": 0, "end_exclusive": robot_count, "query_source": "segmentation"},
                {
                    "kind": "environment",
                    "start": robot_count,
                    "end_exclusive": count,
                    "query_source": "per_frame_grid",
                },
            ]
            metadata[key]["visibility"] = "not_exported"
    collisions = set(converted) & set(dataset.meta.features)
    if collisions:
        raise ValueError(f"Output features already exist: {sorted(collisions)}")
    episode_ids = options.get("selected_episode_ids")
    if episode_ids is not None and set(episode_ids) != set(range(dataset.meta.total_episodes)):
        output = copy_selected_features(dataset, converted, episode_ids, **kwargs)
    else:
        output = add_features(dataset=dataset, features=converted, **kwargs)
    inherit_metadata(dataset, output, episode_ids=episode_ids)
    source = {"repo_id": options.get("source_repo_id", dataset.repo_id)}
    if episode_ids is not None:
        source["episode_mapping"] = {str(new): old for new, old in enumerate(sorted(episode_ids))}
    write_dataset_metadata(output.root, metadata, source=source)
    if options.get("artifact_dir"):
        export_artifacts(dataset, converted, metadata, options["artifact_dir"], episode_ids=episode_ids)
    return output


def export_artifacts(dataset, features, metadata, output_dir, *, episode_ids=None):
    """Export annotation artifacts without copying source videos."""
    from ..conditioning import TrajectoryCondition
    from ..data.artifacts import EpisodeArtifact

    episodes: dict[int, list[dict]] = {}
    # LeRobot may have non-contiguous episode IDs after selection.
    for row in dataset:
        ep = int(row["episode_index"])
        if episode_ids is not None and ep not in episode_ids:
            continue
        episodes.setdefault(ep, []).append({"frame_index": int(row["frame_index"])})
    for output_episode, ep in enumerate(sorted(episodes)):
        rows = episodes[ep]
        tracks, condition, spatial, anchor_strides = {}, None, {}, {}
        size = None
        for key, (values, _spec) in features.items():
            descriptor = SpatialMetadata.from_dict(metadata[key]["spatial"])
            size = size or descriptor.image_size
            if key.startswith("observation.point_tracks."):
                camera = _slug(descriptor.camera_key)
                tracks[camera] = np.stack([values(row, ep, row["frame_index"]) for row in rows])
                spatial[camera] = descriptor
                anchor_strides[camera] = metadata[key]["anchor_stride"]
            elif key == "observation.global_trajectory":
                sequence = np.asarray(values(rows[0], ep, rows[0]["frame_index"]))
                active = np.flatnonzero(sequence[:, 2] > 0.5)
                if len(active) == 1:
                    raise ValueError(f"Episode {ep} has a one-point interaction interval")
                condition = TrajectoryCondition(
                    sequence[:, :2],
                    descriptor.image_size,
                    int(active[0]) if len(active) else None,
                    int(active[-1]) if len(active) else None,
                )
                spatial["trajectory"] = descriptor
        if condition is None and not tracks:
            raise ValueError(
                "Rendered images cannot reconstruct a trajectory; generate sequence artifacts first"
            )
        EpisodeArtifact(
            condition, size, tracks, spatial_metadata=spatial, anchor_strides=anchor_strides
        ).save(Path(output_dir) / f"episode_{output_episode:06d}.npz")
