from __future__ import annotations

import numpy as np
import pytest

from lerobot_policy_open_trajectory.conditioning import TrajectoryCondition
from lerobot_policy_open_trajectory.data.artifacts import EpisodeArtifact


def test_episode_artifact_round_trip_preserves_normalized_trajectory(tmp_path) -> None:
    condition = TrajectoryCondition(
        path_xy=np.asarray([[0.25, 0.5], [0.5, 0.75], [0.9, 1.0]], dtype=np.float32),
        image_shape=(100, 200),
        grasp_index=1,
        release_index=2,
    )
    tracks = {"side": np.arange(48, dtype=np.float32).reshape(2, 3, 4, 2)}
    artifact = EpisodeArtifact(condition=condition, image_size=(100, 200), point_tracks=tracks)

    path = tmp_path / "episode_000000.npz"
    artifact.save(path)
    loaded = EpisodeArtifact.load(path)

    np.testing.assert_array_equal(loaded.condition.path_xy, condition.path_xy)
    assert loaded.condition.image_shape == condition.image_shape
    assert loaded.condition.grasp_index == condition.grasp_index
    assert loaded.condition.release_index == condition.release_index
    assert loaded.image_size == artifact.image_size
    np.testing.assert_array_equal(loaded.point_tracks["side"], tracks["side"])


def test_episode_artifact_requires_timestep_query_horizon_xy_tracks() -> None:
    condition = TrajectoryCondition(
        path_xy=np.asarray([[0.25, 0.5], [0.5, 0.75]], dtype=np.float32),
        image_shape=(100, 200),
    )

    with pytest.raises(ValueError, match=r"\[T, N, H, 2\]"):
        EpisodeArtifact(
            condition=condition,
            image_size=(100, 200),
            point_tracks={"side": np.zeros((2, 4, 2), dtype=np.float32)},
        )
