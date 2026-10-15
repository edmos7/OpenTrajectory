import pytest
import torch

from lerobot_policy_open_trajectory.predictors import TrackPredictor, TrackPredictorConfig


@pytest.mark.parametrize("camera_keys", [("side",), ("side", "wrist")])
def test_predictor_train_predict_reload(camera_keys, tmp_path):
    cfg = TrackPredictorConfig(
        camera_keys=camera_keys,
        image_size=32,
        horizon=2,
        trajectory_length=8,
        dim_model=16,
        n_heads=2,
        n_layers=1,
    )
    predictor = TrackPredictor(cfg)
    images = {key: torch.rand(2, 3, 32, 40) for key in camera_keys}
    queries = {key: torch.rand(2, 3, 2) for key in camera_keys}
    sketch = torch.rand(2, 8, 3)
    targets = {key: torch.rand(2, 3, 2, 2) for key in camera_keys}
    masks = {key: torch.ones(2, 3, 2, dtype=torch.bool) for key in camera_keys}
    masks[camera_keys[0]][0, 0] = False
    targets[camera_keys[0]][0, 0] = float("nan")
    loss = predictor.loss(images, queries, sketch, targets, masks)
    loss.backward()
    assert torch.isfinite(loss)
    torch.optim.AdamW(predictor.parameters()).step()
    result = predictor.predict(images, queries, sketch, fps=10)
    for key in camera_keys:
        assert result.tracks[key].shape == (2, 3, 3, 2)
        torch.testing.assert_close(result.tracks[key][:, :, 0], queries[key])
        assert result.metadata[key].image_size == (32, 40)
        assert result.metadata[key].source == "predicted"
    predictor.save_pretrained(tmp_path)
    restored = TrackPredictor.from_pretrained(tmp_path)
    restored_result = restored.predict(images, queries, sketch)
    for key in camera_keys:
        torch.testing.assert_close(result.tracks[key], restored_result.tracks[key])
    with pytest.raises(ValueError, match="query points"):
        predictor(images, {key: targets[key] for key in camera_keys}, sketch)


def test_predictor_requires_valid_target():
    cfg = TrackPredictorConfig(
        camera_keys=("side",),
        image_size=32,
        horizon=2,
        trajectory_length=8,
        dim_model=16,
        n_heads=2,
        n_layers=1,
    )
    predictor = TrackPredictor(cfg).eval()
    with pytest.raises(ValueError, match="No valid"):
        predictor.loss(
            {"side": torch.rand(1, 3, 32, 32)},
            {"side": torch.rand(1, 2, 2)},
            torch.rand(1, 8, 3),
            {"side": torch.rand(1, 2, 2, 2)},
            {"side": torch.zeros(1, 2, 2, dtype=torch.bool)},
        )


def test_predictor_training_command_and_anchor_gate(tmp_path):
    import json

    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from lerobot_policy_open_trajectory.data.schema import SpatialMetadata, write_dataset_metadata
    from lerobot_policy_open_trajectory.predictors.training import main

    camera = "observation.images.side"
    track_key = "observation.point_tracks.side"
    trajectory_key = "observation.global_trajectory"
    dataset = LeRobotDataset.create(
        repo_id="test/predictor",
        root=tmp_path / "data",
        fps=10,
        use_videos=False,
        features={
            camera: {"dtype": "image", "shape": (32, 32, 3), "names": ["height", "width", "channels"]},
            track_key: {"dtype": "float32", "shape": (2, 3, 2), "names": None},
            trajectory_key: {"dtype": "float32", "shape": (8, 3), "names": None},
        },
    )
    for _ in range(2):
        for _ in range(2):
            dataset.add_frame(
                {
                    camera: np.zeros((32, 32, 3), dtype=np.uint8),
                    track_key: np.full((2, 3, 2), 0.5, dtype=np.float32),
                    trajectory_key: np.full((8, 3), 0.5, dtype=np.float32),
                    "task": "move",
                }
            )
        dataset.save_episode()
    dataset.finalize()
    descriptor = {
        "spatial": SpatialMetadata(camera, (32, 32), fps=10).to_dict(),
        "horizon_includes_current": True,
        "anchor_stride": 1,
    }
    write_dataset_metadata(dataset.root, {track_key: descriptor})
    arguments = [
        "--repo-id",
        "test/predictor",
        "--root",
        str(dataset.root),
        "--camera-tracks",
        json.dumps({camera: track_key}),
        "--output-dir",
        str(tmp_path / "predictor"),
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--image-size",
        "32",
        "--trajectory-length",
        "8",
    ]
    main(arguments)
    history = json.loads((tmp_path / "predictor/metrics.json").read_text())
    assert len(history) == 1
    assert np.isfinite(history[0]["validation_mse"])
    split = json.loads((tmp_path / "predictor/training.json").read_text())
    assert set(split["train_episodes"]).isdisjoint(split["validation_episodes"])
    restored = TrackPredictor.from_pretrained(tmp_path / "predictor")
    assert restored.config.horizon == 2
    for stride in (2, None):
        descriptor["anchor_stride"] = stride
        write_dataset_metadata(dataset.root, {track_key: descriptor})
        arguments[arguments.index("--output-dir") + 1] = str(tmp_path / "rejected")
        with pytest.raises(ValueError, match="strided query anchors"):
            main(arguments)
        assert not (tmp_path / "rejected").exists()
