#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("open_trajectory_act")
@dataclass
class OpenTrajectoryACTConfig(PreTrainedConfig):
    """Configuration class for ACT with a dedicated trajectory-image task encoder.

    This is a spinoff of the standard ACT architecture that adds a separate "task encoder"
    backbone (ResNet18 truncated at layer3) for a trajectory/conditioning image.  The
    trajectory image is processed through the task encoder and its feature tokens are
    appended to the transformer encoder input alongside the regular camera tokens.

    The key difference from vanilla ACT:
      - One additional input key (``trajectory_image_key``) is routed through a dedicated
        lightweight backbone rather than the shared camera backbone.
      - That backbone is a ResNet18 **stripped of layer4** (stops at layer3, 256 ch,
        stride 16 → 2x larger spatial feature map, ~8.4 M fewer parameters).
      - All other architecture parameters are identical to ACT.

    Notes on the inputs and outputs:
        - Either:
            - At least one key starting with ``observation.image`` that is NOT the
              trajectory key is required as a camera input.  AND/OR
            - The key ``observation.environment_state`` is required as input.
        - ``trajectory_image_key`` must be present in ``input_features`` as a VISUAL
          feature; it is excluded from the regular camera backbone processing.
        - May optionally work without an ``observation.state`` key.
        - ``action`` is required as an output key.
        - The trajectory image should be normalised independently from the regular camera
          images (e.g. identity / [0, 1] range).  This is handled at the dataset /
          preprocessor level, not inside the policy.

    Args:
        trajectory_image_key: The ``input_features`` key for the trajectory/conditioning
            image.  This image is routed through the task encoder backbone.
        task_encoder_backbone: Name of the torchvision ResNet backbone to build the task
            encoder from.  layer4 is always stripped, so only layers up to layer3 are kept.
        task_encoder_pretrained_weights: Pretrained weights from torchvision to initialise
            the task encoder backbone.  ``None`` means no pretrained weights.
        optimizer_lr_task_encoder: Learning rate for the task encoder backbone parameters.
            Defaults to the same value as ``optimizer_lr_backbone``.

        All other arguments are identical to ``ACTConfig``.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Architecture — camera backbone.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False

    # Architecture — task encoder (trajectory backbone, truncated at layer3).
    conditioning_mode: str = "rt_image"
    trajectory_image_key: str = "observation.images.rt_trajectory"
    trajectory_feature_key: str = "observation.global_trajectory"
    trajectory_length: int = 224
    task_encoder_backbone: str = "resnet18"
    task_encoder_pretrained_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"

    # Architecture — transformer.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    # See ACTConfig note about n_decoder_layers.
    n_decoder_layers: int = 1

    # VAE.
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Inference.
    temporal_ensemble_coeff: float | None = None

    # Training and loss computation.
    dropout: float = 0.1
    kl_weight: float = 10.0

    # Optimizer presets.
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5
    optimizer_lr_task_encoder: float | None = None  # None → uses optimizer_lr_backbone

    def __post_init__(self):
        super().__post_init__()
        if self.conditioning_mode not in {"rt_image", "sequence"}:
            raise ValueError("conditioning_mode must be rt_image or sequence")
        if self.trajectory_length < 2:
            raise ValueError("trajectory_length must be at least two")

        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if not self.task_encoder_backbone.startswith("resnet"):
            raise ValueError(
                f"`task_encoder_backbone` must be one of the ResNet variants. "
                f"Got {self.task_encoder_backbone}."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError("`n_action_steps` must be 1 when using temporal ensembling.")
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. "
                f"Got {self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )

    # ── Feature properties ────────────────────────────────────────────────────

    @property
    def conditioning_feature_key(self) -> str:
        return (
            self.trajectory_image_key if self.conditioning_mode == "rt_image" else self.trajectory_feature_key
        )

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        """Camera image features — excludes the trajectory key."""
        if not self.input_features:
            return {}
        return {
            key: ft
            for key, ft in self.input_features.items()
            if ft.type is FeatureType.VISUAL and key != self.trajectory_image_key
        }

    @property
    def trajectory_image_feature(self) -> PolicyFeature | None:
        """The PolicyFeature for the trajectory/conditioning image, or None if not present."""
        if not self.input_features:
            return None
        return self.input_features.get(self.trajectory_image_key)

    # ── Optimizer / scheduler presets ─────────────────────────────────────────

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    # ── Validation ────────────────────────────────────────────────────────────

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError(
                "You must provide at least one non-trajectory camera image or the "
                "environment state among the inputs."
            )
        feature = self.input_features.get(self.conditioning_feature_key)
        if feature is None:
            raise ValueError(f"Missing trajectory condition: {self.conditioning_feature_key}")
        if self.conditioning_mode == "sequence" and tuple(feature.shape) != (self.trajectory_length, 3):
            raise ValueError("Sequence condition must match [trajectory_length, 3]")
        if self.conditioning_mode == "rt_image" and (
            feature.type != FeatureType.VISUAL or len(feature.shape) != 3
        ):
            raise ValueError("Rendered condition must be a VISUAL image feature")
        if self.action_feature is None:
            raise ValueError("ACT requires an action feature")

    # ── Delta / reward indices ─────────────────────────────────────────────────

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
