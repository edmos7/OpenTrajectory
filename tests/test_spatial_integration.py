from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot_policy_open_trajectory.data.artifacts import EpisodeArtifact
from lerobot_policy_open_trajectory.data.schema import SpatialMetadata, read_dataset_metadata
from lerobot_policy_open_trajectory.generation.writer import add_spatial_features, inherit_trimmed_metadata


def test_tracks_only_roundtrip_with_metadata_and_validity(tmp_path):
    descriptor = SpatialMetadata("observation.images.side", (32, 48), horizon_stride=4, fps=30)
    tracks = np.zeros((2, 3, 4, 2), dtype=np.float32)
    mask = np.ones((2, 3, 4), dtype=bool)
    artifact = EpisodeArtifact(
        None, (32, 48), {"side": tracks}, spatial_metadata={"side": descriptor}, validity={"side": mask}
    )
    artifact.save(tmp_path / "episode.npz")
    restored = EpisodeArtifact.load(tmp_path / "episode.npz")
    assert restored.condition is None
    assert restored.spatial_metadata == artifact.spatial_metadata
    np.testing.assert_array_equal(restored.validity["side"], mask)
    with pytest.raises(ValueError, match="tracks only"):
        restored.require_condition()


def test_metric_frame_requires_calibration():
    with pytest.raises(ValueError, match="calibration"):
        SpatialMetadata("side", (32, 48), coordinate_frame="robot_base")


def test_artifact_validity_becomes_dataset_feature(tmp_path):
    from lerobot_policy_open_trajectory.data.artifacts import EpisodeArtifactStore
    from lerobot_policy_open_trajectory.data.augment import build_feature_callbacks

    tracks = np.zeros((2, 3, 4, 2), dtype=np.float32)
    mask = np.ones((2, 3, 4), dtype=bool)
    mask[1, :, -1] = False
    EpisodeArtifact(None, (32, 48), {"side": tracks}, validity={"side": mask}).save(
        tmp_path / "episode_000000.npz"
    )
    features = build_feature_callbacks(
        EpisodeArtifactStore(tmp_path),
        features=("tracks",),
        trajectory_length=8,
        camera_keys=("observation.images.side",),
    )
    callback, spec = features["observation.point_tracks_validity.side"]
    assert spec["dtype"] == "bool"
    np.testing.assert_array_equal(callback({}, 0, 1), mask[1])


def test_output_replacement_preserves_backup_and_rejects_source(tmp_path):
    from lerobot_policy_open_trajectory.generation.preprocessing import remove_output_dataset

    output = tmp_path / "output"
    (output / "meta").mkdir(parents=True)
    (output / "meta/info.json").write_text("{}")
    with pytest.raises(ValueError, match="overlap"):
        remove_output_dataset(output, source=output)
    remove_output_dataset(output, source=tmp_path / "source")
    assert not output.exists()
    assert len(list(tmp_path.glob("output.backup-*/meta/info.json"))) == 1


def test_writer_callbacks_and_camera_metadata(monkeypatch, tmp_path):
    source = SimpleNamespace(
        root=tmp_path / "source",
        repo_id="test/source",
        fps=30,
        meta=SimpleNamespace(
            features={
                "observation.images.side": {"shape": (32, 48, 3), "names": ["height", "width", "channels"]}
            }
        ),
    )
    target = SimpleNamespace(root=tmp_path / "target")
    captured = {}

    def add_features(**kwargs):
        captured.update(kwargs)
        return target

    monkeypatch.setattr("lerobot.datasets.dataset_tools.add_features", add_features)
    values = {0: np.zeros((3, 4, 2), dtype=np.float32)}
    result = add_spatial_features(
        source,
        {"observation.main_cam_point_tracks": (values, {"dtype": "float32", "shape": (3, 4, 2)})},
        options={"main_cam_key": "observation.images.side", "frame_stride": 4},
        repo_id="test/output",
    )
    key = "observation.point_tracks.side"
    callback, _ = captured["features"][key]
    assert callable(callback)
    np.testing.assert_array_equal(callback({}, 0, 0), values[0])
    with pytest.raises(ValueError, match="Missing annotation"):
        callback({}, 1, 0)
    descriptor = read_dataset_metadata(result.root)["features"][key]
    assert descriptor["spatial"]["camera_key"] == "observation.images.side"
    assert descriptor["spatial"]["image_size"] == [32, 48]
    assert descriptor["spatial"]["frame_stride"] == 1
    assert descriptor["spatial"]["horizon_stride"] == 4
    assert descriptor["horizon_includes_current"] is True
    result.repo_id = "test/output"
    trimmed = SimpleNamespace(root=tmp_path / "trimmed", features={key: {"shape": (3, 4, 2)}})
    inherit_trimmed_metadata(result, trimmed, [SimpleNamespace(start=2, end=8)])
    assert read_dataset_metadata(trimmed.root)["source"]["trim_bounds"]["0"] == {
        "start": 2,
        "end_exclusive": 8,
    }


@pytest.mark.parametrize(
    "module",
    [
        "generate_global_trajectory_dataset",
        "generate_track_dataset",
        "generate_tracks_and_trajectory_dataset",
        "generate_trajectory_image_dataset",
        "trim_dataset_idle_ends",
        "trim_dataset_after_action_release",
        "add_depth_features",
    ],
)
def test_installed_backend_help_does_not_load_weights(module, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    backend = import_module(f"lerobot_policy_open_trajectory.generation.{module}")
    with pytest.raises(SystemExit) as result:
        backend.main(["--help"])
    assert result.value.code == 0


def test_predictor_split_is_episode_disjoint():
    from lerobot_policy_open_trajectory.predictors.training import split_episodes

    train, validation = split_episodes([0, 0, 1, 2, 3, 3], seed=42)
    assert set(train).isdisjoint(validation)
    assert sorted(train + validation) == [0, 1, 2, 3]
    assert (train, validation) == split_episodes([0, 1, 2, 3], seed=42)


def test_cli_calls_installed_backend(monkeypatch):
    from lerobot_policy_open_trajectory import cli

    arguments = []
    monkeypatch.setattr(
        cli, "import_module", lambda *a: SimpleNamespace(main=lambda argv: arguments.extend(argv))
    )
    assert cli.main(["dataset", "generate", "--kind", "tracks", "--", "--source-repo-id", "test/source"]) == 0
    assert arguments == ["--source-repo-id", "test/source"]


@pytest.mark.parametrize("anchor_stride", [1, 2])
def test_writer_selected_episodes_and_artifacts(tmp_path, anchor_stride):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    camera = "observation.images.side"
    source = LeRobotDataset.create(
        repo_id="test/source",
        root=tmp_path / "source",
        fps=10,
        use_videos=False,
        features={
            camera: {"dtype": "image", "shape": (16, 24, 3), "names": ["height", "width", "channels"]},
            "action": {"dtype": "float32", "shape": (2,), "names": None},
        },
    )
    for ep in range(2):
        for _ in range(2):
            source.add_frame(
                {
                    camera: np.zeros((16, 24, 3), dtype=np.uint8),
                    "action": np.zeros(2, dtype=np.float32),
                    "task": f"task {ep}",
                }
            )
        source.save_episode()
    source.finalize()
    source = LeRobotDataset("test/source", root=tmp_path / "source")

    def values(row, episode, frame):
        return np.full((2, 3, 2), 0.1 * episode + 0.01 * (frame // anchor_stride), dtype=np.float32)

    output = add_spatial_features(
        source,
        {
            "observation.main_cam_point_tracks": (
                values,
                {"dtype": "float32", "shape": (2, 3, 2), "names": None},
            )
        },
        options={
            "main_cam_key": camera,
            "selected_episode_ids": [1],
            "artifact_dir": tmp_path / "artifacts",
            "frame_stride": anchor_stride,
        },
        repo_id="test/output",
        output_dir=tmp_path / "output",
    )
    assert output.meta.total_episodes == 1
    assert source.meta.total_episodes == 2
    np.testing.assert_allclose(output[0]["observation.point_tracks.side"], 0.1)
    artifact = EpisodeArtifact.load(tmp_path / "artifacts/episode_000000.npz")
    assert artifact.condition is None
    assert artifact.frame_stride == 1
    assert artifact.anchor_strides == {"side": anchor_stride}
    assert read_dataset_metadata(output.root)["source"]["episode_mapping"] == {"0": 1}

    from lerobot_policy_open_trajectory.data.augment import augment_dataset
    from lerobot_policy_open_trajectory.generation.writer import export_artifacts

    key = "observation.point_tracks.side"
    export_artifacts(
        source,
        {key: (values, {"shape": (2, 3, 2)})},
        read_dataset_metadata(output.root)["features"],
        tmp_path / "all_artifacts",
    )
    augmented = augment_dataset(
        repo_id="test/source",
        root=source.root,
        artifact_dir=tmp_path / "all_artifacts",
        output_repo_id="test/augmented",
        output_dir=tmp_path / "augmented",
        features=("tracks",),
        camera_keys=(camera,),
    )
    assert read_dataset_metadata(augmented.root)["features"][key]["anchor_stride"] == anchor_stride
    for index in range(len(source)):
        row = source[index]
        np.testing.assert_allclose(
            augmented[index][key], values(row, int(row["episode_index"]), int(row["frame_index"]))
        )
