from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from lerobot.configs.types import FeatureType, PolicyFeature

from lerobot_policy_open_trajectory.conditioning import TrajectoryCondition
from lerobot_policy_open_trajectory.configuration_open_trajectory import OpenTrajectoryConfig
from lerobot_policy_open_trajectory.modeling_components import ConditioningFusion
from lerobot_policy_open_trajectory.modeling_open_trajectory import OpenTrajectoryPolicy
from lerobot_policy_open_trajectory.processor_open_trajectory import make_open_trajectory_pre_post_processors


def _config(mode: str = "sequence") -> OpenTrajectoryConfig:
    condition_key = (
        "observation.global_trajectory" if mode == "sequence" else "observation.images.rt_trajectory"
    )
    condition_feature = (
        PolicyFeature(type=FeatureType.STATE, shape=(16, 3))
        if mode == "sequence"
        else PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32))
    )
    return OpenTrajectoryConfig(
        conditioning_mode=mode,
        input_features={
            "observation.images.side": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
            condition_key: condition_feature,
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        observation_frame_offsets=(-1, 0),
        trajectory_length=16,
        horizon=8,
        n_action_steps=4,
        num_train_timesteps=2,
        environment_encoder_pretrained_weights=None,
        environment_image_resize=(32, 32),
        condition_image_resize=(32, 32),
        dit_dim=32,
        dit_depth=1,
        dit_nhead=4,
        dit_ffn_dim=64,
        fusion_token_dim=32,
        fusion_out_dim=32,
        fusion_nhead=4,
        fusion_ffn_dim=64,
        fusion_layers=1,
    )


def _training_batch(mode: str) -> dict[str, torch.Tensor]:
    batch = {
        "observation.images.side": torch.rand(1, 2, 3, 32, 32),
        "action": torch.rand(1, 8, 3),
    }
    if mode == "sequence":
        batch["observation.global_trajectory"] = torch.rand(1, 2, 16, 3)
    else:
        batch["observation.images.rt_trajectory"] = torch.rand(1, 2, 3, 32, 32)
    return batch


@pytest.mark.parametrize("mode", ["sequence", "rt_image"])
def test_forward_loss_is_finite(mode: str) -> None:
    policy = OpenTrajectoryPolicy(_config(mode))

    loss, metrics = policy(_training_batch(mode))

    assert torch.isfinite(loss)
    assert metrics["diffusion_loss"] == pytest.approx(float(loss.detach()))


def test_diffusion_accepts_predicted_track_features():
    config = _config()
    key = "observation.point_tracks.side"
    config.track_feature_keys = (key,)
    config.input_features[key] = PolicyFeature(type=FeatureType.STATE, shape=(3, 4, 2))
    policy = OpenTrajectoryPolicy(config)
    batch = _training_batch("sequence")
    batch[key] = torch.rand(1, 2, 3, 4, 2)
    loss, _ = policy(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert next(policy.model.track_encoders.parameters()).grad is not None
    policy.eval()
    observations = {k: v[:, -1] for k, v in batch.items() if k != "action"}
    assert policy.predict_action_chunk(observations).shape == (1, 4, 3)


def test_processor_normalizes_only_joint_state_observations() -> None:
    config = _config()
    preprocessor, postprocessor = make_open_trajectory_pre_post_processors(config)

    assert preprocessor.steps[3].normalize_observation_keys == set()
    assert len(preprocessor.steps) == 4
    assert len(postprocessor.steps) == 2


def test_draw_rollout_populates_temporal_history(monkeypatch) -> None:
    condition = TrajectoryCondition(
        np.asarray([[0.1, 0.1], [0.9, 0.9]], dtype=np.float32),
        image_shape=(32, 32),
    )
    monkeypatch.setattr(
        "lerobot_policy_open_trajectory.modeling_open_trajectory.draw_trajectory_condition",
        lambda _frame: condition,
    )
    policy = OpenTrajectoryPolicy(_config())
    policy.eval()
    observation = {"observation.images.side": torch.rand(1, 3, 32, 32)}

    actions = policy.predict_action_chunk(observation, noise=torch.randn(1, 8, 3))

    assert actions.shape == (1, 4, 3)
    assert policy._condition_cache is not None


def test_training_batch_requires_exact_observation_history() -> None:
    policy = OpenTrajectoryPolicy(_config())
    batch = _training_batch("sequence")
    batch["observation.images.side"] = batch["observation.images.side"][:, :1]

    with pytest.raises(ValueError, match="n_obs_steps"):
        policy(batch)


def test_unet_forward_loss_is_finite() -> None:
    config = replace(_config(), denoiser_backbone="unet", down_dims=(16, 32))
    policy = OpenTrajectoryPolicy(config)

    loss, _ = policy(_training_batch("sequence"))

    assert torch.isfinite(loss)


def test_dataset_condition_rollout_uses_cached_condition(monkeypatch) -> None:
    config = replace(_config(), conditioning_source="dataset")
    policy = OpenTrajectoryPolicy(config)
    loaded = torch.rand(16, 3)
    monkeypatch.setattr(policy, "_load_dataset_condition", lambda: loaded)
    policy.eval()

    actions = policy.predict_action_chunk(
        {"observation.images.side": torch.rand(1, 3, 32, 32)},
        noise=torch.randn(1, 8, 3),
    )

    assert actions.shape == (1, 4, 3)
    torch.testing.assert_close(policy._condition_cache.squeeze(0), loaded)


def test_fusion_accepts_all_modalities() -> None:
    fusion = ConditioningFusion(
        state_dim=3,
        vision_dim=8,
        depth_dim=4,
        trajectory_dim=6,
        n_obs_steps=2,
        token_dim=16,
        out_dim=16,
        nhead=4,
        ffn_dim=32,
        layers=1,
    )

    output = fusion(
        state=torch.rand(2, 2, 3),
        vision=torch.rand(2, 2, 8),
        depth=torch.rand(2, 2, 4),
        trajectory=torch.rand(2, 2, 6),
    )

    assert output.shape == (2, 16)


def test_checkpoint_round_trip_preserves_config_and_weights(tmp_path) -> None:
    policy = OpenTrajectoryPolicy(_config())
    with torch.no_grad():
        next(policy.parameters()).fill_(0.125)

    policy.save_pretrained(tmp_path)
    restored = OpenTrajectoryPolicy.from_pretrained(
        tmp_path,
        local_files_only=True,
        strict=True,
    )

    assert restored.config.conditioning_mode == policy.config.conditioning_mode
    assert restored.config.denoiser_backbone == policy.config.denoiser_backbone
    assert not restored.training
    for expected, actual in zip(policy.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
