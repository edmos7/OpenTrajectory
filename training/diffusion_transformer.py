"""Transformer denoiser for 1D action diffusion policies.

The module intentionally matches LeRobot's diffusion denoiser call signature:
``forward(x, timestep, global_cond=None)`` where ``x`` is ``[B, T, action_dim]``.
This lets training keep LeRobot's scheduler, loss, normalization, and sampling
code while swapping only the U-Net backbone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalPosEmb(nn.Module):
    """1D sinusoidal timestep embeddings."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        half_dim = self.dim // 2
        if half_dim < 1:
            raise ValueError("SinusoidalPosEmb dim must be at least 2")
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """AdaLN-Zero transformer block conditioned by one global vector."""

    def __init__(
        self,
        dim: int,
        nhead: int,
        ffn_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(cond).chunk(6, dim=-1)
        )
        attn_in = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out
        mlp_out = self.ffn(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x + gate_mlp.unsqueeze(1) * mlp_out


class FinalLayer(nn.Module):
    """AdaLN-modulated output projection."""

    def __init__(self, dim: int, action_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, action_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


class DiffusionTransformer1d(nn.Module):
    """DiT-style denoiser over action-horizon tokens."""

    def __init__(
        self,
        *,
        action_dim: int,
        horizon: int,
        global_cond_dim: int,
        dim: int = 256,
        depth: int = 6,
        nhead: int = 8,
        ffn_dim: int = 1024,
        timestep_embed_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        if dim % nhead != 0:
            raise ValueError(f"dim={dim} must be divisible by nhead={nhead}")
        self.action_dim = action_dim
        self.horizon = horizon
        self.global_cond_dim = global_cond_dim
        self.dim = dim

        self.x_embedder = nn.Linear(action_dim, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, horizon, dim))
        self.t_embedder = nn.Sequential(
            SinusoidalPosEmb(timestep_embed_dim),
            nn.Linear(timestep_embed_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.cond_embedder = (
            nn.Sequential(
                nn.LayerNorm(global_cond_dim),
                nn.Linear(global_cond_dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
            )
            if global_cond_dim > 0
            else None
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(dim=dim, nhead=nhead, ffn_dim=ffn_dim, dropout=dropout) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(dim=dim, action_dim=action_dim)

        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.zeros_(self.x_embedder.bias)

    def forward(
        self,
        x: Tensor,
        timestep: Tensor | int,
        global_cond: Tensor | None = None,
    ) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x [B, T, D], got {tuple(x.shape)}")
        if x.shape[1] != self.horizon:
            raise ValueError(f"Expected horizon {self.horizon}, got {x.shape[1]}")
        if isinstance(timestep, int):
            timestep = torch.full((x.shape[0],), timestep, dtype=torch.long, device=x.device)
        elif timestep.ndim == 0:
            timestep = timestep.expand(x.shape[0]).to(device=x.device)

        cond = self.t_embedder(timestep)
        if self.cond_embedder is not None:
            if global_cond is None:
                raise ValueError("global_cond is required for this DiffusionTransformer1d")
            cond = cond + self.cond_embedder(global_cond)

        h = self.x_embedder(x) + self.pos_embed[:, : x.shape[1]].to(dtype=x.dtype)
        for block in self.blocks:
            h = block(h, cond)
        return self.final_layer(h, cond)


@dataclass(frozen=True)
class DenoiserConfig:
    backbone: str = "unet"
    dit_dim: int = 256
    dit_depth: int = 6
    dit_nhead: int = 8
    dit_ffn_dim: int = 1024
    dit_dropout: float = 0.0


def make_action_denoiser(
    config,
    *,
    global_cond_dim: int,
    denoiser_cfg: DenoiserConfig,
):
    """Build either the stock LeRobot U-Net or the local DiT denoiser."""
    if denoiser_cfg.backbone == "unet":
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionConditionalUnet1d

        return DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim)
    if denoiser_cfg.backbone == "dit":
        return DiffusionTransformer1d(
            action_dim=config.action_feature.shape[0],
            horizon=config.horizon,
            global_cond_dim=global_cond_dim,
            dim=denoiser_cfg.dit_dim,
            depth=denoiser_cfg.dit_depth,
            nhead=denoiser_cfg.dit_nhead,
            ffn_dim=denoiser_cfg.dit_ffn_dim,
            timestep_embed_dim=config.diffusion_step_embed_dim,
            dropout=denoiser_cfg.dit_dropout,
        )
    raise ValueError(f"Unknown denoiser backbone: {denoiser_cfg.backbone!r}")
