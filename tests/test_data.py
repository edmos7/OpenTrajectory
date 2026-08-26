from __future__ import annotations

import numpy as np
import pytest

from lerobot_policy_open_trajectory.conditioning import TrajectoryCondition
from lerobot_policy_open_trajectory.data.artifacts import EpisodeArtifact, EpisodeArtifactStore
from lerobot_policy_open_trajectory.data.augment import (
    GLOBAL_TRAJECTORY,
    RT_TRAJECTORY_IMAGE,
    augment_dataset,
    build_feature_callbacks,
)


def _artifact(frame_stride: int = 2) -> EpisodeArtifact:
    condition = TrajectoryCondition(
        np.asarray([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]], dtype=np.float32),
        image_shape=(32, 48),
        grasp_index=1,
        release_index=2,
    )
    tracks = {"side": np.arange(24, dtype=np.float32).reshape(2, 2, 3, 2)}
    return EpisodeArtifact(condition, (32, 48), tracks, frame_stride=frame_stride)


def test_feature_callbacks_match_declared_shapes_and_stride(tmp_path) -> None:
    _artifact().save(tmp_path / "episode_000000.npz")
    store = EpisodeArtifactStore(tmp_path)

    callbacks = build_feature_callbacks(
        store,
        features=("sequence", "rt-image", "tracks"),
        trajectory_length=16,
        camera_keys=("observation.images.side",),
    )

    sequence, sequence_info = callbacks[GLOBAL_TRAJECTORY]
    rt_image, rt_info = callbacks[RT_TRAJECTORY_IMAGE]
    tracks, tracks_info = callbacks["observation.point_tracks.side"]
    assert sequence({}, 0, 3).shape == tuple(sequence_info["shape"])
    assert rt_image({}, 0, 3).shape == tuple(rt_info["shape"])
    np.testing.assert_array_equal(tracks({}, 0, 3), _artifact().point_tracks["side"][1])
    assert tracks({}, 0, 3).shape == tuple(tracks_info["shape"])
    assert tracks_info["names"] == ["query", "horizon", "xy"]


def test_store_exposes_consistent_feature_shapes(tmp_path) -> None:
    _artifact().save(tmp_path / "episode_000000.npz")
    EpisodeArtifact(
        _artifact().condition,
        (32, 48),
        {"side": np.zeros((4, 2, 3, 2), dtype=np.float32)},
    ).save(tmp_path / "episode_000001.npz")
    store = EpisodeArtifactStore(tmp_path)

    assert store.rt_image_shape == (32, 48, 3)
    assert store.point_track_shape("side") == (2, 3, 2)


def test_store_rejects_inconsistent_point_track_shapes(tmp_path) -> None:
    _artifact().save(tmp_path / "episode_000000.npz")
    EpisodeArtifact(
        _artifact().condition,
        (32, 48),
        {"side": np.zeros((2, 4, 3, 2), dtype=np.float32)},
    ).save(tmp_path / "episode_000001.npz")

    with pytest.raises(ValueError, match="inconsistent per-frame shape"):
        EpisodeArtifactStore(tmp_path).point_track_shape("side")


def test_store_requires_requested_tracks_in_every_episode(tmp_path) -> None:
    _artifact().save(tmp_path / "episode_000000.npz")
    EpisodeArtifact(_artifact().condition, (32, 48), {}).save(tmp_path / "episode_000001.npz")

    with pytest.raises(KeyError, match="episode 1"):
        EpisodeArtifactStore(tmp_path).point_track_shape("side")


def test_tracks_require_camera_keys(tmp_path) -> None:
    _artifact().save(tmp_path / "episode_000000.npz")

    with pytest.raises(ValueError, match="camera_keys"):
        build_feature_callbacks(
            EpisodeArtifactStore(tmp_path),
            features=("tracks",),
            trajectory_length=16,
            camera_keys=(),
        )


def test_artifact_rejects_inconsistent_image_size() -> None:
    artifact = _artifact()

    with pytest.raises(ValueError, match="image_size"):
        EpisodeArtifact(artifact.condition, (64, 64), artifact.point_tracks)


def test_augment_dataset_copies_one_episode_without_mutating_source(tmp_path) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    artifact_root = tmp_path / "artifacts"
    features = {
        "action": {"dtype": "float32", "shape": (3,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
        "task_id": {"dtype": "int64", "shape": (1,), "names": None},
    }
    source = LeRobotDataset.create(
        repo_id="test/source",
        fps=10,
        features=features,
        root=source_root,
        use_videos=False,
    )
    for frame_index in range(2):
        source.add_frame(
            {
                "action": np.full(3, frame_index, dtype=np.float32),
                "observation.state": np.full(2, frame_index, dtype=np.float32),
                "task_id": np.asarray([0], dtype=np.int64),
                "task": "test task",
            }
        )
    source.save_episode()
    source.finalize()
    EpisodeArtifact(_artifact().condition, (32, 48), {}).save(artifact_root / "episode_000000.npz")

    output = augment_dataset(
        repo_id="test/source",
        root=source_root,
        artifact_dir=artifact_root,
        output_repo_id="test/output",
        output_dir=output_root,
        trajectory_length=16,
    )

    assert GLOBAL_TRAJECTORY in output.meta.features
    assert RT_TRAJECTORY_IMAGE in output.meta.features
    assert "task_id" not in output.meta.features
    assert "task_id" in LeRobotDataset(repo_id="test/source", root=source_root).meta.features
    assert len(output) == 2
