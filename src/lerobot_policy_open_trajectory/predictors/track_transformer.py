"""Track Transformer architecture, ported from the OpenTrajectory research implementation."""

import math

import timm
import torch
from torch import nn
from torch.nn import functional as F


class EfficientNetBackbone(nn.Module):
    """EfficientNet last-stage features → ``[B, h*w, d_model]`` tokens with 2-D pos encoding."""

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        d_model: int = 256,
        img_size: int = 224,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, features_only=True, out_indices=(-1,)
        )
        # Probe feature-map shape at init time so we know grid size and channels.
        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            feat = self.backbone(dummy)[-1]  # [1, C, h, w]
            in_ch = feat.shape[1]
            self._h = feat.shape[2]
            self._w = feat.shape[3]

        self.backbone.train(was_training)
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=1)

        # Learned 2-D positional embeddings, one per spatial location.
        n_tokens = self._h * self._w
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @property
    def feat_h(self) -> int:
        return self._h

    @property
    def feat_w(self) -> int:
        return self._w

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : [B, 3, H, W]  in [0, 1]

        Returns
        -------
        tokens   : [B, h*w, d_model]   (with positional encoding)
        feat_map : [B, d_model, h, w]  (for bilinear point sampling)
        """
        x = (x - self.mean) / self.std
        feat_map = self.proj(self.backbone(x)[-1])  # [B, d_model, h, w]
        tokens = feat_map.flatten(2).transpose(1, 2)  # [B, h*w, d_model]
        tokens = tokens + self.pos_embed
        return tokens, feat_map


# ── Trajectory encoder ─────────────────────────────────────────────────────────


class FourierFeatures(nn.Module):
    def __init__(self, bands: int = 8):
        super().__init__()
        self.register_buffer("freqs", (2.0 ** torch.arange(bands)) * math.pi)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        f = xy.unsqueeze(-1) * self.freqs
        return torch.cat([f.sin(), f.cos()], dim=-1).flatten(-2)


class TrajectoryEncoder(nn.Module):
    """Same Fourier-feature trajectory Transformer used by Diffusion Tracks."""

    def __init__(
        self,
        max_T: int = 550,
        fourier_bands: int = 8,
        temporal_dim: int = 16,
        out_dim: int = 64,
        d_model: int = 64,
        nhead: int = 4,
        ffn_dim: int = 128,
        tf_layers: int = 2,
        action_encoding: str = "delta",
        action_binary_threshold: float = 0.25,
    ):
        super().__init__()
        self.max_T = max_T
        self.action_encoding = action_encoding
        self.action_binary_threshold = action_binary_threshold
        self.fourier = FourierFeatures(fourier_bands)
        self.temporal_emb = nn.Embedding(max_T, temporal_dim)
        in_ch = 4 * fourier_bands + 1 + temporal_dim
        self.input_proj = nn.Linear(in_ch, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers=tf_layers)
        self.out_proj = nn.Linear(d_model, out_dim) if d_model != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xy = x[..., :2]
        action = x[..., 2:3]
        delta_a = torch.diff(action, dim=1, prepend=action[:, :1])
        if self.action_encoding == "delta":
            action_feat = delta_a
        elif self.action_encoding == "binary":
            tolerance = torch.finfo(action.dtype).eps * 4
            action_feat = (action >= self.action_binary_threshold - tolerance).to(action.dtype)
        else:
            raise ValueError(f"Unknown action encoding: {self.action_encoding!r}")

        t_idx = torch.arange(self.max_T, device=x.device)
        t_emb = self.temporal_emb(t_idx).unsqueeze(0).expand(x.shape[0], -1, -1)
        feat = torch.cat([self.fourier(xy), action_feat, t_emb], dim=-1)
        feat = self.backbone(self.input_proj(feat))
        return self.out_proj(feat.mean(dim=1))


def resample_global_trajectory(traj: torch.Tensor, target_length: int) -> torch.Tensor:
    """Remove repeated terminal padding and resample each trajectory."""
    output = torch.empty(
        traj.shape[0],
        target_length,
        traj.shape[-1],
        dtype=traj.dtype,
        device=traj.device,
    )
    for b, sequence in enumerate(traj):
        differs = torch.any(sequence != sequence[-1], dim=-1)
        indices = torch.nonzero(differs, as_tuple=False)
        source_length = 1 if indices.numel() == 0 else min(int(indices[-1].item()) + 2, sequence.shape[0])
        valid = sequence[:source_length].transpose(0, 1).unsqueeze(0)
        output[b] = (
            F.interpolate(valid, size=target_length, mode="linear", align_corners=True)
            .squeeze(0)
            .transpose(0, 1)
        )
    return output


# ── Track token encoder ────────────────────────────────────────────────────────


class TrackTokenEncoder(nn.Module):
    """Build one permutation-equivariant context token per tracked point."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # Encode the 2-D current position
        self.pos_proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # Fuse visual feature sampled from backbone (same dim because of 1x1 conv proj)
        self.vis_fuse = nn.Linear(d_model * 2, d_model)

    def forward(
        self,
        cur_pos: torch.Tensor,  # [B, N, 2] in [0, 1]
        feat_map: torch.Tensor,  # [B, d_model, h, w]
    ) -> torch.Tensor:
        """Returns [B, N, d_model]."""
        coord_val = self.pos_proj(cur_pos)  # [B, N, d]

        # grid_sample expects grid in [-1, 1], shape [B, 1, N, 2], xy order
        grid = (cur_pos * 2.0 - 1.0).unsqueeze(1)  # [B, 1, N, 2]  (x, y)
        vis_feat = F.grid_sample(
            feat_map, grid, mode="bilinear", align_corners=True, padding_mode="border"
        )  # [B, d, 1, N]
        vis_feat = vis_feat.squeeze(2).transpose(1, 2)  # [B, N, d]

        return self.vis_fuse(torch.cat([coord_val, vis_feat], dim=-1))


# ── Full model ────────────────────────────────────────────────────────────────


class TrackTransformer(nn.Module):
    """Track Transformer: image + trajectory + masked tracks → future offsets."""

    def __init__(
        self,
        n_points: int,
        horizon: int,
        backbone_name: str = "efficientnet_b0",
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        traj_length: int = 550,
        traj_enc_dim: int = 64,
        traj_fourier_bands: int = 8,
        traj_temporal_dim: int = 16,
        traj_tf_d_model: int = 64,
        traj_tf_nhead: int = 4,
        traj_tf_ffn_dim: int = 128,
        traj_tf_layers: int = 2,
        traj_action_encoding: str = "delta",
        traj_action_binary_threshold: float = 0.25,
        img_size: int = 224,
        pretrained: bool = True,
    ):
        super().__init__()
        self.n_points = n_points
        self.horizon = horizon
        self.d_model = d_model
        self.traj_length = traj_length

        self.img_encoder = EfficientNetBackbone(backbone_name, d_model, img_size, pretrained)
        self.traj_encoder = TrajectoryEncoder(
            max_T=traj_length,
            fourier_bands=traj_fourier_bands,
            temporal_dim=traj_temporal_dim,
            out_dim=traj_enc_dim,
            d_model=traj_tf_d_model,
            nhead=traj_tf_nhead,
            ffn_dim=traj_tf_ffn_dim,
            tf_layers=traj_tf_layers,
            action_encoding=traj_action_encoding,
            action_binary_threshold=traj_action_binary_threshold,
        )
        self.traj_token_proj = nn.Linear(traj_enc_dim, d_model)
        self.track_encoder = TrackTokenEncoder(d_model)
        self.view_embed = nn.Embedding(2, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,  # pre-norm (more stable training)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        self.future_time_embed = nn.Embedding(horizon, d_model)
        # The transformer attends once per point. This shared decoder expands
        # each transformed point token over learned future-time embeddings.
        self.temporal_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

    def forward(
        self,
        frame: torch.Tensor,  # [B, 3, H, W]   in [0, 1]
        traj: torch.Tensor,  # [B, max_T, 3]
        cur_pos: torch.Tensor,  # [B, N, 2]       in [0, 1]
        sec_frame: torch.Tensor | None = None,
        sec_cur_pos: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        frame.shape[0]
        img_tokens, feat_map = self.img_encoder(frame)
        img_tokens = img_tokens + self.view_embed.weight[0].view(1, 1, -1)
        traj = resample_global_trajectory(traj, self.traj_length)
        traj_tokens = self.traj_token_proj(self.traj_encoder(traj)).unsqueeze(1)  # [B, 1, d]
        track_tokens = self.track_encoder(cur_pos, feat_map)
        track_tokens = track_tokens + self.view_embed.weight[0].view(1, 1, -1)

        L_img = img_tokens.shape[1]
        L_traj = traj_tokens.shape[1]
        token_parts = [img_tokens]
        sec_img_len = 0
        sec_track_len = 0
        if sec_frame is not None or sec_cur_pos is not None:
            if sec_frame is None or sec_cur_pos is None:
                raise ValueError("sec_frame and sec_cur_pos must be provided together.")
            sec_img_tokens, sec_feat_map = self.img_encoder(sec_frame)
            sec_img_tokens = sec_img_tokens + self.view_embed.weight[1].view(1, 1, -1)
            sec_track_tokens = self.track_encoder(sec_cur_pos, sec_feat_map)
            sec_track_tokens = sec_track_tokens + self.view_embed.weight[1].view(1, 1, -1)
            sec_img_len = sec_img_tokens.shape[1]
            sec_track_len = sec_track_tokens.shape[1]
            token_parts.append(sec_img_tokens)
        token_parts.extend([traj_tokens, track_tokens])
        if sec_track_len:
            token_parts.append(sec_track_tokens)

        all_tokens = torch.cat(token_parts, dim=1)
        all_tokens = self.transformer(all_tokens)  # [B, L_total, d]

        # Extract future track token outputs.
        # Track layout: [N*T] with time varying fastest.
        # Future slots: t=1..H → indices {i*T + t | i in 0..N-1, t in 1..H}
        track_start = L_img + sec_img_len + L_traj
        main_track_len = cur_pos.shape[1]
        track_out = all_tokens[:, track_start : track_start + main_track_len]
        main_pred = self.decode_future(track_out)
        if not sec_track_len:
            return main_pred

        sec_start = track_start + main_track_len
        sec_out = all_tokens[:, sec_start : sec_start + sec_track_len]
        return main_pred, self.decode_future(sec_out)

    def decode_future(self, point_tokens: torch.Tensor) -> torch.Tensor:
        time = self.future_time_embed.weight.view(1, 1, self.horizon, self.d_model)
        pair_features = point_tokens.unsqueeze(2) + time
        return self.temporal_head(pair_features)


# ── Dataset ────────────────────────────────────────────────────────────────────
