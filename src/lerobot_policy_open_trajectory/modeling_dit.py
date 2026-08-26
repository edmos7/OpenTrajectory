"""DiT denoiser with LeRobot diffusion's `(x, timestep, global_cond)` signature."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, value: Tensor) -> Tensor:
        half = self.dim // 2
        frequency = torch.exp(torch.arange(half, device=value.device) * (-math.log(10000) / max(half - 1, 1)))
        embedding = value.float().unsqueeze(-1) * frequency
        embedding = torch.cat((embedding.sin(), embedding.cos()), dim=-1)
        return torch.nn.functional.pad(embedding, (0, self.dim - embedding.shape[-1]))


def _modulate(value: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return value * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, dim: int, nhead: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attention = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Dropout(dropout), nn.Linear(ffn_dim, dim)
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, value: Tensor, condition: Tensor) -> Tensor:
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = self.modulation(condition).chunk(
            6, -1
        )
        normalized = _modulate(self.norm1(value), shift_attn, scale_attn)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        value = value + gate_attn.unsqueeze(1) * attended
        return value + gate_ffn.unsqueeze(1) * self.ffn(_modulate(self.norm2(value), shift_ffn, scale_ffn))


class DiffusionTransformer1d(nn.Module):
    def __init__(
        self,
        *,
        action_dim: int,
        horizon: int,
        global_cond_dim: int,
        dim: int,
        depth: int,
        nhead: int,
        ffn_dim: int,
        timestep_embed_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if dim % nhead:
            raise ValueError("dit_dim must be divisible by dit_nhead")
        self.horizon = horizon
        self.action_embedding = nn.Linear(action_dim, dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, horizon, dim))
        self.timestep_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(timestep_embed_dim),
            nn.Linear(timestep_embed_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.condition_embedding = nn.Sequential(
            nn.LayerNorm(global_cond_dim), nn.Linear(global_cond_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.blocks = nn.ModuleList(DiTBlock(dim, nhead, ffn_dim, dropout) for _ in range(depth))
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.output = nn.Linear(dim, action_dim)
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, value: Tensor, timestep: Tensor | int, global_cond: Tensor | None = None) -> Tensor:
        if value.ndim != 3 or value.shape[1] != self.horizon:
            raise ValueError(f"Expected action tensor [B,{self.horizon},D], got {tuple(value.shape)}")
        if global_cond is None:
            raise ValueError("global_cond is required")
        if isinstance(timestep, int):
            timestep = torch.full((value.shape[0],), timestep, device=value.device, dtype=torch.long)
        elif timestep.ndim == 0:
            timestep = timestep.expand(value.shape[0])
        condition = self.timestep_embedding(timestep) + self.condition_embedding(global_cond)
        hidden = self.action_embedding(value) + self.position_embedding.to(dtype=value.dtype)
        for block in self.blocks:
            hidden = block(hidden, condition)
        shift, scale = self.final_modulation(condition).chunk(2, -1)
        return self.output(_modulate(self.final_norm(hidden), shift, scale))
