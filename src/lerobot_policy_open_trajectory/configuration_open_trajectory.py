"""Serializable LeRobot configuration for Open Trajectory policies."""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE


@PreTrainedConfig.register_subclass("open_trajectory")
@dataclass
class OpenTrajectoryConfig(PreTrainedConfig):
    """One policy configuration for sequence and RT-Trajectory image conditioning."""

    conditioning_mode: str = "sequence"
    denoiser_backbone: str = "dit"

    n_obs_steps: int = 2
    observation_frame_offsets: tuple[int, ...] = (-8, 0)
    horizon: int = 32
    n_action_steps: int = 16
    drop_n_last_frames: int | None = None

    trajectory_feature_key: str = "observation.global_trajectory"
    rt_image_feature_key: str = "observation.images.rt_trajectory"
    trajectory_camera_key: str | None = None
    environment_image_keys: tuple[str, ...] = ()
    depth_feature_keys: tuple[str, ...] = ()
    track_feature_keys: tuple[str, ...] = ()
    track_encoder_out_dim: int = 32
    use_joint_state: bool = False

    trajectory_length: int = 224
    trajectory_encoder_out_dim: int = 96
    trajectory_encoder_hidden_dim: int = 128
    trajectory_encoder_layers: int = 4
    trajectory_encoder_kernel_sizes: tuple[int, ...] = (7, 5, 3, 3)
    trajectory_encoder_dilations: tuple[int, ...] = (1, 2, 4, 8)
    trajectory_temporal_dim: int = 16

    rt_image_encoder_out_dim: int = 96
    rt_image_encoder_channels: tuple[int, ...] = (24, 48, 96, 136)
    condition_image_resize: tuple[int, int] = (224, 224)

    environment_encoder: str = "resnet18"
    environment_encoder_out_dim: int = 32
    environment_encoder_pretrained_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    environment_encoder_frozen: bool = False
    environment_encoder_lr: float = 1e-5
    environment_image_resize: tuple[int, int] = (224, 224)
    dino_model: str = "dinov2_vits14"

    depth_encoder_out_dim: int = 32
    state_dropout_prob: float = 0.05
    state_noise_std: float = 0.01
    vision_dropout_prob: float = 0.20
    depth_dropout_prob: float = 0.10

    fusion_token_dim: int = 64
    fusion_out_dim: int = 96
    fusion_nhead: int = 4
    fusion_ffn_dim: int = 256
    fusion_layers: int = 2

    down_dims: tuple[int, ...] = (56, 112, 224)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True
    gradient_checkpointing: bool = False

    dit_dim: int = 256
    dit_depth: int = 6
    dit_nhead: int = 8
    dit_ffn_dim: int = 1024
    dit_dropout: float = 0.05

    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    num_inference_steps: int | None = None
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0
    do_mask_loss_for_padding: bool = False

    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-4
    scheduler_warmup_steps: int = 1000

    conditioning_source: str = "draw"
    conditioning_dataset_repo_id: str | None = None
    conditioning_dataset_root: str | None = None
    conditioning_episode: int | None = None

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.conditioning_mode not in {"sequence", "rt_image"}:
            raise ValueError("conditioning_mode must be 'sequence' or 'rt_image'")
        if self.denoiser_backbone not in {"dit", "unet"}:
            raise ValueError("denoiser_backbone must be 'dit' or 'unet'")
        if self.environment_encoder not in {"resnet18", "dinov2"}:
            raise ValueError("environment_encoder must be 'resnet18' or 'dinov2'")
        if self.conditioning_source not in {"draw", "dataset"}:
            raise ValueError("conditioning_source must be 'draw' or 'dataset'")
        if self.noise_scheduler_type not in {"DDPM", "DDIM"}:
            raise ValueError("noise_scheduler_type must be 'DDPM' or 'DDIM'")
        if self.prediction_type not in {"epsilon", "sample"}:
            raise ValueError("prediction_type must be 'epsilon' or 'sample'")
        if self.n_obs_steps < 1 or self.horizon < 1 or self.n_action_steps < 1:
            raise ValueError("n_obs_steps, horizon, and n_action_steps must be positive")
        if self.n_action_steps > self.horizon - self.n_obs_steps + 1:
            raise ValueError("n_action_steps exceeds the usable action horizon")
        if len(self.observation_frame_offsets) != self.n_obs_steps:
            raise ValueError("observation_frame_offsets must contain n_obs_steps entries")
        if tuple(sorted(self.observation_frame_offsets)) != self.observation_frame_offsets:
            raise ValueError("observation_frame_offsets must be sorted")
        if self.observation_frame_offsets[-1] != 0:
            raise ValueError("observation_frame_offsets must end at 0")
        if self.trajectory_length < 2:
            raise ValueError("trajectory_length must be at least 2")
        if self.denoiser_backbone == "unet":
            downsampling_factor = 2 ** len(self.down_dims)
            if self.horizon % downsampling_factor:
                raise ValueError("horizon must be divisible by the U-Net downsampling factor")
        derived_drop = self.horizon - self.n_action_steps - self.n_obs_steps + 1
        if self.drop_n_last_frames is None:
            self.drop_n_last_frames = derived_drop
        elif self.drop_n_last_frames < 0:
            raise ValueError("drop_n_last_frames must be non-negative")

    @property
    def conditioning_feature_key(self) -> str:
        return (
            self.trajectory_feature_key if self.conditioning_mode == "sequence" else self.rt_image_feature_key
        )

    def resolved_environment_image_keys(self) -> tuple[str, ...]:
        if self.environment_image_keys:
            return self.environment_image_keys
        return tuple(key for key in self.image_features if key != self.rt_image_feature_key)

    def validate_features(self) -> None:
        if not self.input_features:
            raise ValueError("Open Trajectory input features have not been resolved")
        if self.action_feature is None:
            raise ValueError(f"Open Trajectory requires {ACTION!r} in output_features")
        condition_key = self.conditioning_feature_key
        if condition_key not in self.input_features:
            raise ValueError(
                f"conditioning_mode={self.conditioning_mode!r} requires feature {condition_key!r}"
            )
        if self.conditioning_mode == "sequence":
            shape = tuple(self.input_features[condition_key].shape)
            if len(shape) != 2 or shape[-1] != 3:
                raise ValueError(f"{condition_key!r} must have shape [T, 3], got {shape}")
        elif len(self.input_features[condition_key].shape) != 3:
            raise ValueError(f"{condition_key!r} must be an HWC or CHW RGB image feature")
        image_keys = self.resolved_environment_image_keys()
        if not image_keys:
            raise ValueError("Open Trajectory requires at least one environment RGB camera")
        missing = [key for key in image_keys if key not in self.input_features]
        if missing:
            raise ValueError(f"Missing configured environment images: {missing}")
        if self.use_joint_state and OBS_STATE not in self.input_features:
            raise ValueError(f"use_joint_state=True requires {OBS_STATE!r}")
        missing_depth = [key for key in self.depth_feature_keys if key not in self.input_features]
        if missing_depth:
            raise ValueError(f"Missing configured depth features: {missing_depth}")
        if len(set(self.track_feature_keys)) != len(self.track_feature_keys):
            raise ValueError("track_feature_keys must be distinct")
        if self.track_encoder_out_dim < 1:
            raise ValueError("track_encoder_out_dim must be positive")
        for key in self.track_feature_keys:
            if key not in self.input_features:
                raise ValueError(f"Missing configured track feature: {key}")
            shape = self.input_features[key].shape
            if len(shape) != 3 or min(shape[:2]) < 1 or shape[-1] != 2:
                raise ValueError(f"{key}: tracks must be [N, horizon, XY]")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=10.0,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=1e-8,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=75_000,
        )

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(self.observation_frame_offsets)

    @property
    def action_delta_indices(self) -> list[int]:
        start = 1 - self.n_obs_steps
        return list(range(start, start + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
