import pytest
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies import get_policy_class, make_policy_config, make_pre_post_processors

from lerobot_policy_open_trajectory.configuration_open_trajectory_act import OpenTrajectoryACTConfig
from lerobot_policy_open_trajectory.modeling_open_trajectory_act import OpenTrajectoryACTPolicy


def config(mode):
    key = "observation.global_trajectory" if mode == "sequence" else "observation.images.rt_trajectory"
    feature = (
        PolicyFeature(FeatureType.STATE, (8, 3))
        if mode == "sequence"
        else PolicyFeature(FeatureType.VISUAL, (3, 32, 32))
    )
    return OpenTrajectoryACTConfig(
        conditioning_mode=mode,
        trajectory_length=8,
        input_features={key: feature, "observation.environment_state": PolicyFeature(FeatureType.ENV, (3,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
        chunk_size=4,
        n_action_steps=2,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        latent_dim=8,
        pretrained_backbone_weights=None,
        task_encoder_pretrained_weights=None,
    )


@pytest.mark.parametrize("mode", ["sequence", "rt_image"])
def test_act_train_infer_checkpoint(mode, tmp_path):
    cfg = config(mode)
    policy = OpenTrajectoryACTPolicy(cfg)
    batch = {
        "observation.environment_state": torch.rand(2, 3),
        "action": torch.rand(2, 4, 2),
        "action_is_pad": torch.zeros(2, 4, dtype=torch.bool),
        cfg.conditioning_feature_key: torch.rand(2, *cfg.input_features[cfg.conditioning_feature_key].shape),
    }
    loss, _ = policy(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for n, p in policy.named_parameters() if "task_encoder" in n)
    optimizer = torch.optim.AdamW(policy.get_optim_params())
    optimizer.step()
    observations = {k: v for k, v in batch.items() if not k.startswith("action")}
    assert policy.select_action(observations).shape == (2, 2)
    policy.reset()
    policy.save_pretrained(tmp_path)
    restored = OpenTrajectoryACTPolicy.from_pretrained(tmp_path, local_files_only=True)
    torch.testing.assert_close(
        policy.predict_action_chunk(observations), restored.predict_action_chunk(observations)
    )


def test_act_native_factory_and_processors():
    cfg = make_policy_config("open_trajectory_act")
    assert isinstance(cfg, OpenTrajectoryACTConfig)
    assert get_policy_class("open_trajectory_act") is OpenTrajectoryACTPolicy
    pre, post = make_pre_post_processors(config("sequence"))
    assert "observation.global_trajectory" not in pre.steps[-1].normalize_observation_keys
    assert len(post.steps) == 2


@pytest.mark.parametrize("mode", ["sequence", "rt_image"])
def test_act_camera_temporal_ensemble_without_vae(mode):
    cfg = config(mode)
    cfg.input_features["observation.images.side"] = PolicyFeature(FeatureType.VISUAL, (3, 32, 32))
    cfg.use_vae = False
    cfg.n_action_steps = 1
    cfg.temporal_ensemble_coeff = 0.01
    policy = OpenTrajectoryACTPolicy(cfg)
    batch = {key: torch.rand(2, *feature.shape) for key, feature in cfg.input_features.items()}
    batch["action"] = torch.rand(2, 4, 2)
    batch["action_is_pad"] = torch.zeros(2, 4, dtype=torch.bool)
    loss, metrics = policy(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert "kld_loss" not in metrics
    observations = {key: value for key, value in batch.items() if not key.startswith("action")}
    first = policy.select_action(observations)
    second = policy.select_action(observations)
    assert first.shape == second.shape == (2, 2)
    assert torch.isfinite(second).all()
    policy.reset()
    torch.testing.assert_close(policy.select_action(observations), first)
