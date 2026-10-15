"""Native LeRobot Open Trajectory policy."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from diffusers import DDIMScheduler, DDPMScheduler
from lerobot.policies import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE
from torch import Tensor, nn

from .conditioning import draw_trajectory_condition
from .configuration_open_trajectory import OpenTrajectoryConfig
from .modeling_components import (
    ConditioningFusion,
    DepthEncoder,
    DinoEnvironmentEncoder,
    ResNetEnvironmentEncoder,
    RTTrajectoryImageEncoder,
    TrajectoryEncoder,
)
from .modeling_dit import DiffusionTransformer1d


def _drop_modality(value: Tensor | None, probability: float, training: bool) -> Tensor | None:
    if value is None or not training or probability <= 0:
        return value
    keep = (torch.rand(value.shape[0], device=value.device) >= probability).to(value.dtype)
    return value * keep.view(value.shape[0], *([1] * (value.ndim - 1)))


class OpenTrajectoryModel(nn.Module):
    def __init__(self, config: OpenTrajectoryConfig) -> None:
        super().__init__()
        self.config = config
        self.environment_image_keys = config.resolved_environment_image_keys()

        if config.environment_encoder == "resnet18":
            self.environment_encoder = ResNetEnvironmentEncoder(
                backbone="resnet18",
                out_dim=config.environment_encoder_out_dim,
                pretrained_weights=config.environment_encoder_pretrained_weights,
                frozen=config.environment_encoder_frozen,
                resize=config.environment_image_resize,
            )
        else:
            self.environment_encoder = DinoEnvironmentEncoder(
                model_name=config.dino_model,
                out_dim=config.environment_encoder_out_dim,
                frozen=config.environment_encoder_frozen,
                resize=config.environment_image_resize,
            )

        if config.conditioning_mode == "sequence":
            self.condition_encoder = TrajectoryEncoder(
                length=config.trajectory_length,
                temporal_dim=config.trajectory_temporal_dim,
                out_dim=config.trajectory_encoder_out_dim,
                hidden_dim=config.trajectory_encoder_hidden_dim,
                layers=config.trajectory_encoder_layers,
                kernel_sizes=config.trajectory_encoder_kernel_sizes,
                dilations=config.trajectory_encoder_dilations,
            )
            condition_dim = config.trajectory_encoder_out_dim
        else:
            self.condition_encoder = RTTrajectoryImageEncoder(
                out_dim=config.rt_image_encoder_out_dim,
                channels=config.rt_image_encoder_channels,
                resize=config.condition_image_resize,
            )
            condition_dim = config.rt_image_encoder_out_dim

        self.track_encoders = nn.ModuleList(
            nn.Sequential(
                nn.Linear(config.input_features[key].shape[1] * 2, config.track_encoder_out_dim),
                nn.GELU(),
                nn.Linear(config.track_encoder_out_dim, config.track_encoder_out_dim),
            )
            for key in config.track_feature_keys
        )
        condition_dim += len(config.track_feature_keys) * config.track_encoder_out_dim
        self.depth_encoders = nn.ModuleList(
            DepthEncoder(config.depth_encoder_out_dim, config.environment_image_resize)
            for _ in config.depth_feature_keys
        )
        state_dim = 0
        if config.use_joint_state:
            state_feature = config.input_features[OBS_STATE]
            state_dim = state_feature.shape[0]
        vision_dim = config.environment_encoder_out_dim * len(self.environment_image_keys)
        depth_dim = config.depth_encoder_out_dim * len(config.depth_feature_keys)
        self.fusion = ConditioningFusion(
            state_dim=state_dim,
            vision_dim=vision_dim,
            depth_dim=depth_dim,
            trajectory_dim=condition_dim,
            n_obs_steps=config.n_obs_steps,
            token_dim=config.fusion_token_dim,
            out_dim=config.fusion_out_dim,
            nhead=config.fusion_nhead,
            ffn_dim=config.fusion_ffn_dim,
            layers=config.fusion_layers,
        )

        if config.denoiser_backbone == "dit":
            self.denoiser = DiffusionTransformer1d(
                action_dim=config.action_feature.shape[0],
                horizon=config.horizon,
                global_cond_dim=self.fusion.output_dim,
                dim=config.dit_dim,
                depth=config.dit_depth,
                nhead=config.dit_nhead,
                ffn_dim=config.dit_ffn_dim,
                timestep_embed_dim=config.diffusion_step_embed_dim,
                dropout=config.dit_dropout,
            )
        else:
            from lerobot.policies.diffusion.modeling_diffusion import DiffusionConditionalUnet1d

            self.denoiser = DiffusionConditionalUnet1d(config, global_cond_dim=self.fusion.output_dim)

        scheduler_class = DDPMScheduler if config.noise_scheduler_type == "DDPM" else DDIMScheduler
        self.noise_scheduler = scheduler_class(
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )
        self.num_inference_steps = config.num_inference_steps or config.num_train_timesteps

    def encode_condition(self, batch: dict[str, Tensor]) -> Tensor:
        temporal_inputs = {
            **{key: 5 for key in self.environment_image_keys},
            **{key: 5 for key in self.config.depth_feature_keys},
            **{key: 5 for key in self.config.track_feature_keys},
            self.config.conditioning_feature_key: 4 if self.config.conditioning_mode == "sequence" else 5,
            **({OBS_STATE: 3} if self.config.use_joint_state else {}),
        }
        for key, expected_ndim in temporal_inputs.items():
            value = batch[key]
            if value.ndim != expected_ndim or value.shape[1] != self.config.n_obs_steps:
                raise ValueError(
                    f"{key!r} must have n_obs_steps={self.config.n_obs_steps}, got {tuple(value.shape)}"
                )
        image_features = [self.environment_encoder(batch[key]) for key in self.environment_image_keys]
        vision = torch.cat(image_features, dim=-1)
        vision = _drop_modality(vision, self.config.vision_dropout_prob, self.training)

        state = batch.get(OBS_STATE) if self.config.use_joint_state else None
        if state is not None:
            if state.ndim == 2:
                state = state.unsqueeze(1)
            if self.training and self.config.state_noise_std > 0:
                state = state + torch.randn_like(state) * self.config.state_noise_std
            state = _drop_modality(state, self.config.state_dropout_prob, self.training)

        depth_features = [
            encoder(batch[key])
            for key, encoder in zip(self.config.depth_feature_keys, self.depth_encoders, strict=True)
        ]
        depth = torch.cat(depth_features, dim=-1) if depth_features else None
        depth = _drop_modality(depth, self.config.depth_dropout_prob, self.training)

        condition = self.condition_encoder(batch[self.config.conditioning_feature_key])
        track_features = []
        for key, encoder in zip(self.config.track_feature_keys, self.track_encoders, strict=True):
            tracks = batch[key]
            if (
                tuple(tracks.shape[2:]) != tuple(self.config.input_features[key].shape)
                or not torch.isfinite(tracks).all()
            ):
                raise ValueError(f"{key}: tracks must be finite and match the configured [N, horizon, XY]")
            # Encode each path, then pool points without depending on query order.
            track_features.append(encoder(tracks.flatten(-2)).mean(dim=2))
        if track_features:
            condition = torch.cat([condition, *track_features], dim=-1)
        return self.fusion(state=state, vision=vision, depth=depth, trajectory=condition)

    def compute_loss(self, batch: dict[str, Tensor], reduction: str = "mean") -> Tensor:
        actions = batch[ACTION]
        if actions.ndim != 3:
            raise ValueError(f"Expected actions [B,H,D], got {tuple(actions.shape)}")
        noise = torch.randn_like(actions)
        timesteps = torch.randint(
            0, self.config.num_train_timesteps, (actions.shape[0],), device=actions.device, dtype=torch.long
        )
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        prediction = self.denoiser(noisy_actions, timesteps, global_cond=self.encode_condition(batch))
        target = noise if self.config.prediction_type == "epsilon" else actions
        loss = (prediction - target).square().mean(dim=-1)
        if self.config.do_mask_loss_for_padding and "action_is_pad" in batch:
            valid = ~batch["action_is_pad"]
            loss = (loss * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        else:
            loss = loss.mean(dim=1)
        if reduction == "none":
            return loss
        if reduction != "mean":
            raise ValueError("reduction must be 'mean' or 'none'")
        return loss.mean()

    @torch.no_grad()
    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        condition = self.encode_condition(batch)
        action_dim = self.config.action_feature.shape[0]
        sample = noise
        if sample is None:
            sample = torch.randn(
                condition.shape[0],
                self.config.horizon,
                action_dim,
                device=condition.device,
                dtype=condition.dtype,
            )
        self.noise_scheduler.set_timesteps(self.num_inference_steps, device=condition.device)
        for timestep in self.noise_scheduler.timesteps:
            prediction = self.denoiser(sample, timestep, global_cond=condition)
            sample = self.noise_scheduler.step(prediction, timestep, sample).prev_sample
        start = self.config.n_obs_steps - 1
        return sample[:, start : start + self.config.n_action_steps]


class OpenTrajectoryPolicy(PreTrainedPolicy):
    config_class = OpenTrajectoryConfig
    name = "open_trajectory"
    _fsdp_wrap_modules: ClassVar[list[str]] = ["DiTBlock"]

    def __init__(
        self,
        config: OpenTrajectoryConfig,
        dataset_stats: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        del dataset_stats
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = OpenTrajectoryModel(config)
        self._condition_cache: Tensor | None = None
        self.reset()

    def get_optim_params(self) -> list[dict[str, Any]]:
        environment_ids = {id(parameter) for parameter in self.model.environment_encoder.parameters()}
        environment = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) in environment_ids
        ]
        policy = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in environment_ids
        ]
        groups: list[dict[str, Any]] = [{"params": policy}]
        if environment:
            groups.append({"params": environment, "lr": self.config.environment_encoder_lr})
        return groups

    def reset(self) -> None:
        self._observation_queues: dict[str, deque[Tensor]] = {
            key: deque(maxlen=self.config.n_obs_steps)
            for key in (
                *self.config.resolved_environment_image_keys(),
                *self.config.depth_feature_keys,
                *self.config.track_feature_keys,
                self.config.conditioning_feature_key,
                *((OBS_STATE,) if self.config.use_joint_state else ()),
            )
        }
        self._action_queue: deque[Tensor] = deque(maxlen=self.config.n_action_steps)
        self._condition_cache = None

    @staticmethod
    def _camera_to_rgb(camera: Tensor) -> np.ndarray:
        value = camera.detach().cpu()
        if value.ndim == 4:
            if value.shape[0] != 1:
                raise ValueError("Interactive trajectory drawing requires rollout batch size 1")
            value = value[0]
        if value.ndim != 3:
            raise ValueError(f"Expected rollout camera [B,C,H,W] or [C,H,W], got {tuple(value.shape)}")
        if value.shape[0] == 3:
            value = value.permute(1, 2, 0)
        array = value.numpy()
        if array.max(initial=0) <= 1.0:
            array = array * 255.0
        return np.clip(array, 0, 255).astype(np.uint8)

    def _load_dataset_condition(self) -> Tensor:
        if self.config.conditioning_dataset_repo_id is None or self.config.conditioning_episode is None:
            raise ValueError(
                "conditioning_source='dataset' requires conditioning_dataset_repo_id and conditioning_episode"
            )
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(
            repo_id=self.config.conditioning_dataset_repo_id,
            root=Path(self.config.conditioning_dataset_root)
            if self.config.conditioning_dataset_root
            else None,
            episodes=[self.config.conditioning_episode],
            download_videos=True,
        )
        sample = dataset[0]
        key = self.config.conditioning_feature_key
        if key not in sample:
            raise KeyError(f"Conditioning dataset does not contain {key!r}")
        return torch.as_tensor(sample[key])

    def _resolve_rollout_condition(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        key = self.config.conditioning_feature_key
        if key in batch:
            return batch
        batch = dict(batch)
        if self._condition_cache is None:
            if self.config.conditioning_source == "dataset":
                value = self._load_dataset_condition()
            else:
                camera_key = (
                    self.config.trajectory_camera_key or self.config.resolved_environment_image_keys()[0]
                )
                if camera_key not in batch:
                    raise KeyError(
                        f"Trajectory drawing camera {camera_key!r} is missing from the observation"
                    )
                condition = draw_trajectory_condition(self._camera_to_rgb(batch[camera_key]))
                if self.config.conditioning_mode == "sequence":
                    value = torch.from_numpy(condition.to_sequence(self.config.trajectory_length))
                else:
                    value = torch.from_numpy(condition.to_rt_image()).permute(2, 0, 1).float() / 255.0
            if value.ndim in {2, 3}:
                value = value.unsqueeze(0)
            self._condition_cache = value.to(device=next(self.parameters()).device, dtype=torch.float32)
        batch_size = next(iter(batch.values())).shape[0]
        batch[key] = self._condition_cache.expand(batch_size, *self._condition_cache.shape[1:])
        return batch

    def _queue_observations(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        for key, queue in self._observation_queues.items():
            value = batch[key]
            while not queue:
                queue.append(value)
            queue.append(value)
            while len(queue) < self.config.n_obs_steps:
                queue.appendleft(queue[0])
        return {key: torch.stack(tuple(queue), dim=1) for key, queue in self._observation_queues.items()}

    def _expand_rollout_history(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        image_keys = {
            *self.config.resolved_environment_image_keys(),
            *self.config.depth_feature_keys,
            *((self.config.rt_image_feature_key,) if self.config.conditioning_mode == "rt_image" else ()),
        }
        prepared: dict[str, Tensor] = {}
        for key in self._observation_queues:
            value = batch[key]
            single_ndim = (
                4
                if key in image_keys or key in self.config.track_feature_keys
                else 3
                if key == self.config.trajectory_feature_key
                else 2
            )
            if value.ndim == single_ndim:
                value = value.unsqueeze(1).expand(-1, self.config.n_obs_steps, *value.shape[1:])
            elif value.ndim != single_ndim + 1 or value.shape[1] != self.config.n_obs_steps:
                raise ValueError(
                    f"{key!r} must be a single rollout observation or have "
                    f"n_obs_steps={self.config.n_obs_steps}, got {tuple(value.shape)}"
                )
            prepared[key] = value
        return prepared

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        **_: Any,
    ) -> Tensor:
        batch = self._resolve_rollout_condition(batch)
        if any(self._observation_queues[key] for key in self._observation_queues):
            prepared = {
                key: torch.stack(tuple(queue), dim=1) for key, queue in self._observation_queues.items()
            }
        else:
            prepared = self._expand_rollout_history(batch)
        return self.model.generate_actions(prepared, noise=noise)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        batch = {key: value for key, value in batch.items() if key != ACTION}
        batch = self._resolve_rollout_condition(batch)
        prepared = self._queue_observations(batch)
        if not self._action_queue:
            actions = self.model.generate_actions(prepared, noise=noise)
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict[str, float]]:
        loss = self.model.compute_loss(batch, reduction=reduction)
        metrics = {"diffusion_loss": float(loss.detach().mean().cpu())}
        return loss, metrics
