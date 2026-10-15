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
"""Action Chunking Transformer with Trajectory Conditioning (ACTTrajectory)

A spinoff of ACT that adds a dedicated lightweight "task encoder" backbone for a
trajectory/conditioning image.  The task encoder is a ResNet18 truncated at layer3
(256 channels, stride 16), saving ~8.4 M parameters compared to a full ResNet18.

Architecture overview
─────────────────────

                                  Transformer
                                  (VAE decoder during training)
                                 ┌───────────────────────────────┐
                                 │               Outputs         │
                                 │                  ▲            │
                                 │       ┌────────► │            │
                    ┌──────┐     │       │         Transf.       │
                    │      │     │       ├────────► decoder      │
               ┌────┴────┐ │     │       │                       │
               │         │ │     │   ┌───┴────┬──────────────►   │
               │ VAE     │ │     │   │        │  Transf.         │
               │ encoder │ │     │   │Transf. │  encoder         │
               │         │ │     │   │encoder │                  │
               └───▲─────┘ │     │   └▲──▲──▲─┘                  │
                   │       │     │    │  │  │                    │
                 inputs    └─────┼────┘  │  │                    │
                                 │    state  cam. emb.           │
                                 │    emb.   (regular cameras)   │
                                 │           + traj. emb.        │
                                 │           (task encoder)      │
                                 └───────────────────────────────┘

The trajectory image tokens are appended to the transformer encoder input sequence
after the regular camera tokens.  The transformer encoder performs self-attention
over all tokens jointly, and the decoder cross-attends to the full encoder output.
"""

from collections import deque
from itertools import chain

import einops
import torch
import torch.nn.functional as F
import torchvision
from lerobot.policies.act.modeling_act import (
    ACTDecoder,
    ACTEncoder,
    ACTSinusoidalPositionEmbedding2d,
    ACTTemporalEnsembler,
    create_sinusoidal_pos_embedding,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot_policy_open_trajectory.configuration_open_trajectory_act import OpenTrajectoryACTConfig

# Key used inside the batch dict to pass the trajectory image to the model forward.
OBS_TRAJECTORY = "obs_trajectory"

# ResNet18 layer3 output channels (256).  ResNet50/101 layer3 = 1024.
_RESNET18_LAYER3_CHANNELS = 256


class _Layer3Backbone(nn.ModuleDict):
    """ResNet backbone truncated to layer3 (inclusive).

    Built from an ``nn.ModuleDict`` so that each layer is a named sub-module
    (enabling ``named_parameters()`` to work naturally) and the forward pass
    simply applies them in order, returning ``{"feature_map": x}`` to match
    the interface expected by the rest of the model.
    """

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        for module in self.values():
            x = module(x)
        return {"feature_map": x}


class ACTTrajectory(nn.Module):
    """ACT neural network with a dedicated task encoder for a trajectory image.

    Parameters
    ----------
    config:
        An ``OpenTrajectoryACTConfig`` instance.
    """

    def __init__(self, config: OpenTrajectoryACTConfig):
        super().__init__()
        self.config = config

        # ── VAE encoder ───────────────────────────────────────────────────────
        if config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            if config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    config.robot_state_feature.shape[0], config.dim_model
                )
            self.vae_encoder_action_input_proj = nn.Linear(config.action_feature.shape[0], config.dim_model)
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            num_input_token_encoder = 1 + config.chunk_size
            if config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        # ── Regular camera backbone (ResNet, full) ────────────────────────────
        if config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # ── Task encoder backbone (ResNet truncated at layer3) ────────────────
        if config.conditioning_mode == "rt_image":
            _te_full = getattr(torchvision.models, config.task_encoder_backbone)(
                weights=config.task_encoder_pretrained_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            _te_layers: dict[str, nn.Module] = {}
            for name, module in _te_full.named_children():
                _te_layers[name] = module
                if name == "layer3":
                    break
            self.task_encoder = _Layer3Backbone(_te_layers)
            with torch.no_grad():
                _ch = self.task_encoder(torch.zeros(1, 3, 32, 32))["feature_map"].shape[1]
            self.task_encoder_img_feat_input_proj = nn.Conv2d(_ch, config.dim_model, kernel_size=1)
            self.task_encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)
        else:
            self.task_encoder = nn.Sequential(
                nn.Linear(3, config.dim_model), nn.GELU(), nn.Linear(config.dim_model, config.dim_model)
            )
            self.register_buffer(
                "trajectory_pos_embed",
                create_sinusoidal_pos_embedding(config.trajectory_length, config.dim_model).unsqueeze(1),
            )

        # ── Transformer encoder + decoder ─────────────────────────────────────
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        # ── Transformer encoder input projections ─────────────────────────────
        if config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                config.robot_state_feature.shape[0], config.dim_model
            )
        if config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(config.env_state_feature.shape[0], config.dim_model)
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )

        # ── Transformer encoder positional embeddings ─────────────────────────
        n_1d_tokens = 1  # for the latent
        if config.robot_state_feature:
            n_1d_tokens += 1
        if config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # ── Decoder positional embedding + action head ────────────────────────
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)
        self.action_head = nn.Linear(config.dim_model, config.action_feature.shape[0])

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialisation of transformer parameters."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """Forward pass through ACTTrajectory.

        ``batch`` should have the following structure::

            {
                OBS_IMAGES (optional): list of (B, C, H, W) tensors — regular cameras.
                OBS_TRAJECTORY:        (B, C, H, W) tensor — trajectory/conditioning image.
                OBS_STATE  (optional): (B, state_dim) robot state.
                OBS_ENV_STATE (optional): (B, env_dim) environment state.
                ACTION (optional, training + VAE): (B, chunk_size, action_dim).
            }

        Returns
        -------
        actions:
            (B, chunk_size, action_dim) predicted action sequence.
        (mu, log_sigma_x2):
            VAE latent distribution parameters (each (B, latent_dim)), or (None, None).
        """
        if self.config.use_vae and self.training:
            assert ACTION in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_TRAJECTORY].shape[0]

        # ── VAE encoder (training only) ───────────────────────────────────────
        if self.config.use_vae and ACTION in batch and self.training:
            cls_embed = einops.repeat(self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size)
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE])
                robot_state_embed = robot_state_embed.unsqueeze(1)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])

            if self.config.robot_state_feature:
                vae_encoder_input = torch.cat([cls_embed, robot_state_embed, action_embed], dim=1)
            else:
                vae_encoder_input = torch.cat([cls_embed, action_embed], dim=1)

            pos_embed = self.vae_encoder_pos_enc.clone().detach()

            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch[OBS_TRAJECTORY].device,
            )
            key_padding_mask = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], dim=1)

            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                batch[OBS_TRAJECTORY].device
            )

        # ── Build transformer encoder input tokens ────────────────────────────
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        # Regular camera tokens.
        if self.config.image_features:
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        # Trajectory / task encoder tokens.
        if self.config.conditioning_mode == "rt_image":
            traj_features = self.task_encoder(batch[OBS_TRAJECTORY])["feature_map"]
            traj_pos_embed = self.task_encoder_cam_feat_pos_embed(traj_features).to(dtype=traj_features.dtype)
            traj_features = self.task_encoder_img_feat_input_proj(traj_features)
            traj_features = einops.rearrange(traj_features, "b c h w -> (h w) b c")
            traj_pos_embed = einops.rearrange(traj_pos_embed, "b c h w -> (h w) b c")
        else:
            traj_features = self.task_encoder(batch[OBS_TRAJECTORY]).transpose(0, 1)
            traj_pos_embed = self.trajectory_pos_embed.to(dtype=traj_features.dtype)
        encoder_in_tokens.extend(list(traj_features))
        encoder_in_pos_embed.extend(list(traj_pos_embed))

        # ── Transformer forward ───────────────────────────────────────────────
        encoder_in_tokens = torch.stack(encoder_in_tokens, dim=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, dim=0)

        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        actions = self.action_head(decoder_out.transpose(0, 1))
        return actions, (mu, log_sigma_x2)


class OpenTrajectoryACTPolicy(PreTrainedPolicy):
    """ACT policy with a dedicated trajectory-image task encoder.

    See ``ACTTrajectory`` for the full architecture description.
    """

    config_class = OpenTrajectoryACTConfig
    name = "open_trajectory_act"

    def __init__(self, config: OpenTrajectoryACTConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = ACTTrajectory(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> list[dict]:
        task_encoder_lr = (
            self.config.optimizer_lr_task_encoder
            if self.config.optimizer_lr_task_encoder is not None
            else self.config.optimizer_lr_backbone
        )
        return [
            {
                # All non-backbone, non-task-encoder parameters.
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone")
                    and not n.startswith("model.task_encoder")
                    and p.requires_grad
                ]
            },
            {
                # Regular camera backbone.
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
            {
                # Task encoder backbone + projection.
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.task_encoder") and p.requires_grad
                ],
                "lr": task_encoder_lr,
            },
        ]

    def reset(self):
        """Call whenever the environment is reset."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque(maxlen=self.config.n_action_steps)

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Separate the trajectory image from the regular camera images."""
        batch = dict(batch)  # shallow copy
        # Collect regular camera images (all VISUAL inputs except the trajectory key).
        if self.config.image_features:
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        # Move the trajectory image under its internal key.
        condition = batch[self.config.conditioning_feature_key]
        if self.config.conditioning_mode == "sequence":
            if condition.ndim != 3 or condition.shape[1:] != (self.config.trajectory_length, 3):
                raise ValueError("ACT sequence batch must be [B, trajectory_length, 3]")
        elif condition.ndim != 4 or condition.shape[1] != 3:
            raise ValueError("ACT rendered-image batch must be [B, 3, H, W]")
        batch[OBS_TRAJECTORY] = condition
        return batch

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        self.eval()

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            return self.temporal_ensembler.update(actions)

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a full action chunk given environment observations."""
        self.eval()
        batch = self._prepare_batch(batch)
        return self.model(batch)[0]

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the training loss."""
        batch = self._prepare_batch(batch)
        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.detach()}
        if self.config.use_vae:
            mean_kld = (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp())).sum(-1).mean()
            loss_dict["kld_loss"] = mean_kld.detach()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        return loss, loss_dict
