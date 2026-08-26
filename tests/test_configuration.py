from __future__ import annotations

from pathlib import Path

import draccus
import pytest
from lerobot.configs.train import TrainPipelineConfig

from lerobot_policy_open_trajectory.configuration_open_trajectory import OpenTrajectoryConfig


def test_action_deltas_match_lerobot_diffusion_alignment() -> None:
    config = OpenTrajectoryConfig()

    assert config.action_delta_indices == list(range(-1, 31))
    assert config.drop_n_last_frames == 15


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("noise_scheduler_type", "other", "noise_scheduler_type"),
        ("prediction_type", "velocity", "prediction_type"),
        ("observation_frame_offsets", (0, -8), "sorted"),
        ("observation_frame_offsets", (-8, -1), "end at 0"),
    ],
)
def test_invalid_configuration_fails_loudly(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        OpenTrajectoryConfig(**{field: value})


def test_unet_horizon_must_match_downsampling() -> None:
    with pytest.raises(ValueError, match="downsampling factor"):
        OpenTrajectoryConfig(denoiser_backbone="unet", horizon=30)


@pytest.mark.parametrize(
    ("filename", "conditioning_mode"),
    [
        ("train_sequence_dit.yaml", "sequence"),
        ("train_rt_image_dit.yaml", "rt_image"),
    ],
)
def test_native_training_example_parses(filename: str, conditioning_mode: str) -> None:
    path = Path(__file__).parents[1] / "configs" / filename

    config = draccus.parse(TrainPipelineConfig, config_path=path, args=[])
    config.validate()

    assert config.policy is not None
    assert config.policy.type == "open_trajectory"
    assert config.policy.conditioning_mode == conditioning_mode
    assert config.policy.denoiser_backbone == "dit"
    assert config.policy.environment_image_keys == (
        "observation.images.side",
        "observation.images.wrist",
    )
    assert not config.policy.push_to_hub
    assert not config.wandb.enable
