#!/usr/bin/env python3
"""Train Diffusion Policy with RT trajectory image conditioning.

This trainer expects a derived LeRobot dataset with an episode-level image
feature such as:

    observation.images.rt_trajectory

It keeps the same DINO-style visual conditioning used by the current tracks
trainer for camera inputs, but replaces trajectory time-series conditioning
with a dedicated CNN encoding of the RT trajectory image. The defaults match the current
``train_diffusion_tracks.py`` small preset where it matters for DP capacity:
horizon, action steps, observation steps, UNet widths, DINO output dims, and
fusion dims. It deliberately does not use TAPNext point tracks or trajectory
time-series features.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def _ensure_lerobot() -> None:
    try:
        import lerobot  # noqa: F401
    except ImportError:
        python = Path.home() / "miniforge3/envs/lerobotclone/bin/python"
        if not python.exists():
            sys.exit("ERROR: lerobot unavailable and ~/miniforge3/envs/lerobotclone/bin/python not found.")
        result = subprocess.run([str(python), __file__, *sys.argv[1:]])
        raise SystemExit(result.returncode)


_ensure_lerobot()

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.datasets.factory import IMAGENET_STATS, resolve_delta_timestamps  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.utils import cycle  # noqa: E402
from lerobot.optim.optimizers import AdamWConfig  # noqa: E402
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig  # noqa: E402
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig  # noqa: E402
from lerobot.policies.diffusion.modeling_diffusion import (  # noqa: E402
    DiffusionConditionalUnet1d,
    DiffusionModel,
)
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402


DEFAULT_REPO_ID = "edmos7/so100_multitask_quality_checked_tip_v1-dav2small-depth-rt-trajectory-images"
DEFAULT_OUTPUT_DIR = HERE / "outputs/train/dp_rt_trajectory_image"

CAPACITY_PRESETS = {
    "tinier": {
        "down_dims": [24, 48, 96],
        "diffusion_step_embed_dim": 64,
        "dino_out_dim": 260,
        "rt_out_dim": 32,
        "depth_out_dim": 32,
        "fusion_token_dim": 40,
        "fusion_out_dim": 64,
        "fusion_nhead": 4,
        "fusion_ffn_dim": 96,
        "fusion_layers": 3,
    },
    "tiny": {
        "down_dims": [56, 112, 224],
        "dino_out_dim": 32,
        "rt_out_dim": 32,
        "depth_out_dim": 32,
        "fusion_token_dim": 24,
        "fusion_out_dim": 48,
        "fusion_nhead": 4,
        "fusion_ffn_dim": 96,
        "fusion_layers": 2,
    },
    "small": {
        "down_dims": [128, 256, 512],
        "dino_out_dim": 32,
        "rt_out_dim": 32,
        "depth_out_dim": 32,
    },
    "medium": {
        "down_dims": [256, 512, 1024],
        "dino_out_dim": 32,
        "rt_out_dim": 32,
        "depth_out_dim": 32,
    },
    "large": {
        "down_dims": [512, 1024, 2048],
        "dino_out_dim": 32,
        "rt_out_dim": 32,
        "depth_out_dim": 32,
    },
}


def parse_episode_list(spec: str, max_ep: int) -> list[int]:
    indices: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            indices.update(range(int(lo), int(hi) + 1))
        else:
            indices.add(int(part))
    return sorted(i for i in indices if 0 <= i < max_ep)


def scalar_task_id(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.flatten()[0].item())
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("Encountered an empty task_id value.")
        value = value[0]
    return int(value)


def depth_key_for(camera_key: str, depth_prefix: str) -> str:
    suffix = camera_key.removeprefix("observation.images.")
    return f"{depth_prefix}.{suffix}"


class TaskStratifiedBatchSampler:
    """Build near-balanced mini-batches from a scalar per-frame task_id feature."""

    def __init__(self, task_ids: list[int], batch_size: int, drop_last: bool = True, seed: int = 0):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if not task_ids:
            raise ValueError("Cannot stratify an empty dataset.")
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.task_to_indices: dict[int, list[int]] = {}
        for index, task_id in enumerate(task_ids):
            self.task_to_indices.setdefault(int(task_id), []).append(index)
        self.task_ids = sorted(self.task_to_indices)
        self.num_samples = len(task_ids)

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        return math.ceil(self.num_samples / self.batch_size)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        pools = {task: indices.copy() for task, indices in self.task_to_indices.items()}
        positions = {task: 0 for task in self.task_ids}
        for indices in pools.values():
            rng.shuffle(indices)

        def draw(task_id: int, count: int) -> list[int]:
            selected = []
            pool = pools[task_id]
            while len(selected) < count:
                position = positions[task_id]
                if position == len(pool):
                    rng.shuffle(pool)
                    position = 0
                take = min(count - len(selected), len(pool) - position)
                selected.extend(pool[position : position + take])
                positions[task_id] = position + take
            return selected

        num_tasks = len(self.task_ids)
        for batch_index in range(len(self)):
            current_batch_size = self.batch_size
            if not self.drop_last and batch_index == len(self) - 1:
                current_batch_size = self.num_samples - batch_index * self.batch_size
            per_task, remainder = divmod(current_batch_size, num_tasks)
            extra_start = (batch_index * remainder) % num_tasks if remainder else 0
            extra_tasks = {self.task_ids[(extra_start + offset) % num_tasks] for offset in range(remainder)}
            batch = []
            for task_id in self.task_ids:
                batch.extend(draw(task_id, per_task + int(task_id in extra_tasks)))
            rng.shuffle(batch)
            yield batch


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str))


def save_checkpoint_extras(
    out_dir: Path,
    *,
    dino_encoder: nn.Module,
    rt_encoder: nn.Module,
    depth_encoder: nn.Module | None,
    dino_keys: list[str],
    depth_keys: list[str],
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(dino_encoder.state_dict(), out_dir / "dino_encoder.pt")
    torch.save(rt_encoder.state_dict(), out_dir / "rt_trajectory_encoder.pt")
    if depth_encoder is not None:
        torch.save(depth_encoder.state_dict(), out_dir / "depth_encoder.pt")
    save_json(
        out_dir / "rt_conditioning_config.json",
        {
            "dino_keys": dino_keys,
            "rt_trajectory_key": args.rt_trajectory_key,
            "depth_keys": depth_keys,
            "dino_model": args.dino_model,
            "dino_out_dim": args.dino_out_dim,
            "dino_resize": args.dino_resize,
            "dino_frozen": args.dino_frozen,
            "rt_out_dim": args.rt_out_dim,
            "rt_resize": args.rt_resize,
            "depth_out_dim": args.depth_out_dim,
            "depth_resize": args.depth_resize,
            "fusion_token_dim": args.fusion_token_dim,
            "fusion_out_dim": args.fusion_out_dim,
            "fusion_nhead": args.fusion_nhead,
            "fusion_ffn_dim": args.fusion_ffn_dim,
            "fusion_layers": args.fusion_layers,
            "fusion_num_output_tokens": args.fusion_num_output_tokens,
        },
    )


class Metrics:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.train_csv = out_dir / "train_metrics.csv"
        if not self.train_csv.exists():
            with self.train_csv.open("w", newline="") as f:
                csv.writer(f).writerow(["step", "avg_loss", "lr"])

    def log_train(self, step: int, avg_loss: float, lr: float) -> None:
        with self.train_csv.open("a", newline="") as f:
            csv.writer(f).writerow([step, avg_loss, lr])


class DinoImageAugTransform(nn.Module):
    """Training-only image augmentation before DINO normalization."""

    def __init__(
        self,
        brightness: float = 0.30,
        contrast: float = 0.30,
        saturation: float = 0.25,
        hue: float = 0.06,
        none_prob: float = 0.25,
        grayscale_prob: float = 0.25,
        solarize_prob: float = 0.25,
        solarize_threshold: float = 0.85,
        crop_scale: tuple[float, float] = (1.0, 1.0),
        crop_ratio: tuple[float, float] = (1.0, 1.0),
    ):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.none_prob = none_prob
        self.grayscale_prob = grayscale_prob
        self.solarize_prob = solarize_prob
        self.solarize_threshold = solarize_threshold
        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio

    def _random_resized_crop(self, x: torch.Tensor) -> torch.Tensor:
        min_scale, max_scale = self.crop_scale
        if min_scale >= 1.0 and max_scale >= 1.0:
            return x
        batch_size, _, _, h, w = x.shape
        area = h * w
        min_ratio, max_ratio = self.crop_ratio
        log_min_ratio = math.log(min_ratio)
        log_max_ratio = math.log(max_ratio)
        out = []
        for i in range(batch_size):
            crop_h, crop_w = h, w
            for _ in range(10):
                target_area = area * float(torch.empty((), device=x.device).uniform_(min_scale, max_scale))
                aspect = math.exp(float(torch.empty((), device=x.device).uniform_(log_min_ratio, log_max_ratio)))
                crop_w = int(round(math.sqrt(target_area * aspect)))
                crop_h = int(round(math.sqrt(target_area / aspect)))
                if 0 < crop_h <= h and 0 < crop_w <= w:
                    break
            else:
                crop_h, crop_w = h, w
            top = int(torch.randint(0, h - crop_h + 1, (), device=x.device).item()) if crop_h < h else 0
            left = int(torch.randint(0, w - crop_w + 1, (), device=x.device).item()) if crop_w < w else 0
            crop = x[i, :, :, top : top + crop_h, left : left + crop_w]
            out.append(torch.nn.functional.interpolate(crop, size=(h, w), mode="bilinear", align_corners=False).unsqueeze(0))
        return torch.cat(out, dim=0)

    def _color_jitter(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        if self.brightness > 0:
            factor = torch.empty(batch_size, 1, 1, 1, 1, device=x.device).uniform_(1 - self.brightness, 1 + self.brightness)
            x = x * factor
        if self.contrast > 0:
            mean = x.mean(dim=(-2, -1), keepdim=True)
            factor = torch.empty(batch_size, 1, 1, 1, 1, device=x.device).uniform_(1 - self.contrast, 1 + self.contrast)
            x = (x - mean) * factor + mean
        if self.saturation > 0 and x.shape[2] == 3:
            gray = x[:, :, 0:1] * 0.299 + x[:, :, 1:2] * 0.587 + x[:, :, 2:3] * 0.114
            factor = torch.empty(batch_size, 1, 1, 1, 1, device=x.device).uniform_(1 - self.saturation, 1 + self.saturation)
            x = (x - gray) * factor + gray
        x = x.clamp(0.0, 1.0)
        if self.hue > 0 and x.shape[2] == 3:
            from torchvision.transforms.functional import adjust_hue

            hue_factors = torch.empty(batch_size, device=x.device).uniform_(-self.hue, self.hue)
            x = torch.stack([adjust_hue(x[i], float(hue_factors[i])) for i in range(batch_size)], dim=0)
        return x.clamp(0.0, 1.0)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        if imgs.ndim == 4:
            imgs = imgs.unsqueeze(1)
        elif imgs.ndim != 5:
            raise ValueError(f"DinoImageAugTransform expected [B,T,C,H,W] or [B,C,H,W], got {tuple(imgs.shape)}")
        x = self._random_resized_crop(imgs.float())
        jittered = self._color_jitter(x)
        gray = x[:, :, 0:1] * 0.299 + x[:, :, 1:2] * 0.587 + x[:, :, 2:3] * 0.114
        grayscaled = gray.expand_as(x)
        solarized = torch.where(x >= self.solarize_threshold, 1.0 - x, x)
        draw = torch.rand(x.shape[0], 1, 1, 1, 1, device=x.device)
        none_cutoff = self.none_prob
        gray_cutoff = none_cutoff + self.grayscale_prob
        solarize_cutoff = gray_cutoff + self.solarize_prob
        out = torch.where(draw < none_cutoff, x, jittered)
        out = torch.where((draw >= none_cutoff) & (draw < gray_cutoff), grayscaled, out)
        out = torch.where((draw >= gray_cutoff) & (draw < solarize_cutoff), solarized, out)
        return out.clamp(0.0, 1.0)


class DinoVisualEncoder(nn.Module):
    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)

    def __init__(self, model_name: str, out_dim: int, frozen: bool = True, resize: tuple[int, int] = (224, 224)):
        super().__init__()
        self.resize = resize
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        feat_dim = self.backbone.embed_dim
        self.proj = nn.Linear(feat_dim, out_dim)
        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        self.register_buffer("_mean", torch.tensor(self._MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(self._STD).view(1, 3, 1, 1))

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        if imgs.ndim == 4:
            imgs = imgs.unsqueeze(1)
        elif imgs.ndim != 5:
            raise ValueError(f"DinoVisualEncoder expected [B,T,C,H,W] or [B,C,H,W], got {tuple(imgs.shape)}")
        b, t = imgs.shape[:2]
        x = imgs.flatten(0, 1).float()
        if x.shape[-2:] != tuple(self.resize):
            x = torch.nn.functional.interpolate(x, size=self.resize, mode="bilinear", align_corners=False)
        x = (x - self._mean) / self._std
        cls = self.backbone(x)
        return self.proj(cls).view(b, t, -1)


class DepthVisualEncoder(nn.Module):
    def __init__(self, out_dim: int, resize: tuple[int, int] = (224, 224)):
        super().__init__()
        self.resize = resize
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.Mish(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.Mish(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.Mish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
        )

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        if imgs.ndim == 4:
            imgs = imgs.unsqueeze(1)
        elif imgs.ndim != 5:
            raise ValueError(f"DepthVisualEncoder expected [B,T,C,H,W] or [B,C,H,W], got {tuple(imgs.shape)}")
        b, t = imgs.shape[:2]
        x = imgs.flatten(0, 1).float()
        if x.shape[-2:] != tuple(self.resize):
            x = torch.nn.functional.interpolate(x, size=self.resize, mode="bilinear", align_corners=False)
        return self.net(x).view(b, t, -1)


class RtTrajectoryImageEncoder(DepthVisualEncoder):
    """Small CNN for rendered RT trajectory images."""


class ConditioningFusion(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int,
        dino_dim: int,
        rt_dim: int,
        depth_dim: int,
        n_obs_steps: int,
        token_dim: int = 32,
        out_dim: int = 64,
        nhead: int = 4,
        ffn_dim: int = 128,
        n_layers: int = 2,
        num_output_tokens: int = 1,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.num_output_tokens = num_output_tokens
        self.state_proj = nn.Linear(state_dim, token_dim) if state_dim > 0 else None
        self.dino_proj = nn.Linear(dino_dim, token_dim) if dino_dim > 0 else None
        self.rt_proj = nn.Linear(rt_dim, token_dim) if rt_dim > 0 else None
        self.depth_proj = nn.Linear(depth_dim, token_dim) if depth_dim > 0 else None
        self.output_tokens = nn.Parameter(torch.randn(num_output_tokens, out_dim) * 0.02)
        self.output_emb = nn.Parameter(torch.randn(1, out_dim) * 0.02)
        self.modality_emb = nn.Parameter(torch.randn(4, token_dim) * 0.02)
        self.obs_time_emb = nn.Parameter(torch.randn(n_obs_steps, token_dim) * 0.02)
        self.query_norms = nn.ModuleList([nn.LayerNorm(out_dim) for _ in range(n_layers)])
        self.context_norms = nn.ModuleList([nn.LayerNorm(token_dim) for _ in range(n_layers)])
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(out_dim, nhead, kdim=token_dim, vdim=token_dim, batch_first=True)
            for _ in range(n_layers)
        ])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(out_dim) for _ in range(n_layers)])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(out_dim, ffn_dim), nn.Mish(), nn.Linear(ffn_dim, out_dim))
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(out_dim)

    @property
    def output_dim(self) -> int:
        return self.out_dim * self.num_output_tokens

    def _add_temporal(self, tokens: list[torch.Tensor], x: torch.Tensor | None, proj, modality_index: int) -> None:
        if x is None or proj is None:
            return
        if x.ndim != 3:
            raise ValueError(f"Expected temporal feature [B,T,D], got {tuple(x.shape)}")
        projected = proj(x)
        t = projected.shape[1]
        tokens.append(projected + self.modality_emb[modality_index].view(1, 1, -1) + self.obs_time_emb[:t].view(1, t, -1))

    def forward(
        self,
        *,
        state: torch.Tensor | None,
        dino: torch.Tensor | None,
        rt: torch.Tensor | None,
        depth: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size = next(x.shape[0] for x in (state, dino, rt, depth) if x is not None)
        tokens: list[torch.Tensor] = []
        self._add_temporal(tokens, state, self.state_proj, 0)
        self._add_temporal(tokens, dino, self.dino_proj, 1)
        self._add_temporal(tokens, rt, self.rt_proj, 2)
        self._add_temporal(tokens, depth, self.depth_proj, 3)
        context = torch.cat(tokens, dim=1)
        fused = self.output_tokens.unsqueeze(0).expand(batch_size, -1, -1) + self.output_emb.view(1, 1, -1)
        for query_norm, context_norm, attn, ffn_norm, ffn in zip(
            self.query_norms, self.context_norms, self.attn_layers, self.ffn_norms, self.ffn_layers
        ):
            attn_out, _ = attn(query=query_norm(fused), key=context_norm(context), value=context_norm(context), need_weights=False)
            fused = fused + attn_out
            fused = fused + ffn(ffn_norm(fused))
        return self.out_norm(fused).flatten(start_dim=1)


class FusedDinoConditioningDiffusionModel(DiffusionModel):
    def __init__(
        self,
        config: DiffusionConfig,
        *,
        state_dim: int,
        dino_dim: int,
        rt_dim: int,
        depth_dim: int,
        args: argparse.Namespace,
    ):
        super().__init__(config)
        if config.image_features:
            raise ValueError("DINO trainer expects images encoded into ENV, not native VISUAL features.")
        self.dino_dim = dino_dim
        self.rt_dim = rt_dim
        self.depth_dim = depth_dim
        self.fusion = ConditioningFusion(
            state_dim=state_dim,
            dino_dim=dino_dim,
            rt_dim=rt_dim,
            depth_dim=depth_dim,
            n_obs_steps=config.n_obs_steps,
            token_dim=args.fusion_token_dim,
            out_dim=args.fusion_out_dim,
            nhead=args.fusion_nhead,
            ffn_dim=args.fusion_ffn_dim,
            n_layers=args.fusion_layers,
            num_output_tokens=args.fusion_num_output_tokens,
        )
        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=self.fusion.output_dim)

    def _prepare_global_conditioning(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        state = batch.get("observation.state")
        env = batch.get("observation.environment_state")
        dino = rt = depth = None
        if env is not None:
            offset = 0
            if self.dino_dim > 0:
                dino = env[..., offset : offset + self.dino_dim]
                offset += self.dino_dim
            if self.rt_dim > 0:
                rt = env[..., offset : offset + self.rt_dim]
                offset += self.rt_dim
            if self.depth_dim > 0:
                depth = env[..., offset : offset + self.depth_dim]
                offset += self.depth_dim
            if offset != env.shape[-1]:
                raise ValueError(f"ENV dim mismatch: consumed {offset}, got {env.shape[-1]}")
        return self.fusion(state=state, dino=dino, rt=rt, depth=depth)


def camera_keys_for_args(args: argparse.Namespace) -> list[str]:
    keys = list(args.camera_keys)
    if args.camera_key:
        keys.append(args.camera_key)
    deduped = []
    for key in keys:
        if key and key not in deduped:
            deduped.append(key)
    return deduped


def build_config(args: argparse.Namespace, ds_meta: LeRobotDatasetMetadata, env_dim: int) -> DiffusionConfig:
    input_features: dict[str, PolicyFeature] = {}
    output_features: dict[str, PolicyFeature] = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=tuple(ds_meta.features["action"]["shape"]))
    }

    state_dim = tuple(ds_meta.features["observation.state"]["shape"])[0]
    input_features["observation.state"] = PolicyFeature(
        type=FeatureType.STATE,
        shape=(state_dim,) if args.use_joint_state else (1,),
    )
    if env_dim > 0:
        input_features["observation.environment_state"] = PolicyFeature(
            type=FeatureType.ENV,
            shape=(env_dim,),
        )

    drop_n_last_frames = args.horizon - args.n_action_steps - args.n_obs_steps + 1
    return DiffusionConfig(
        horizon=args.horizon,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        drop_n_last_frames=drop_n_last_frames,
        down_dims=tuple(args.down_dims),
        kernel_size=args.kernel_size,
        n_groups=args.n_groups,
        diffusion_step_embed_dim=args.diffusion_step_embed_dim,
        use_film_scale_modulation=True,
        vision_backbone=args.vision_backbone,
        pretrained_backbone_weights=None,
        use_group_norm=True,
        resize_shape=None,
        crop_ratio=args.crop_ratio,
        spatial_softmax_num_keypoints=args.spatial_softmax_kp,
        noise_scheduler_type="DDPM",
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=True,
        clip_sample_range=1.0,
        do_mask_loss_for_padding=False,
        normalization_mapping={
            "STATE": NormalizationMode.MIN_MAX if args.use_joint_state else NormalizationMode.IDENTITY,
            "ENV": NormalizationMode.IDENTITY,
            "VISUAL": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MIN_MAX,
        },
        optimizer_lr=args.lr,
        optimizer_weight_decay=args.weight_decay,
        device=args.device,
        input_features=input_features,
        output_features=output_features,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train DP with RT trajectory image conditioning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset-repo-id", default=DEFAULT_REPO_ID)
    p.add_argument("--dataset-root", default=None)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--episodes", default=None)
    p.add_argument("--exclude-episodes", default="24,33,45,70,78,82")
    p.add_argument("--rt-trajectory-key", default="observation.images.rt_trajectory")
    p.add_argument("--capacity-preset", choices=["tinier", "tiny", "small", "medium", "large", "custom"], default="tinier")
    p.add_argument(
        "--camera-keys",
        nargs="+",
        default=["observation.images.side", "observation.images.wrist"],
        help="RGB camera keys encoded with DINO.",
    )
    p.add_argument("--camera-key", default=None, help="Deprecated: extra single RGB camera key encoded with DINO.")
    p.add_argument("--depth-key", default="", help="Optional explicit depth image/video key. Empty auto-selects matching depth keys.")
    p.add_argument("--depth-prefix", default="observation.depth", help="Prefix used to auto-select depth keys for DINO cameras.")
    p.add_argument("--use-depth", dest="no_depth", action="store_false", help="Enable automatic or explicit depth conditioning.")
    p.add_argument("--no-depth", dest="no_depth", action="store_true", help="Disable depth conditioning.")
    p.set_defaults(no_depth=True)
    p.add_argument("--use-joint-state", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--n-action-steps", type=int, default=16)
    p.add_argument("--n-obs-steps", type=int, default=2)
    p.add_argument("--observation-frame-offsets", type=int, nargs="+", default=[-8, 0])
    p.add_argument("--down-dims", type=int, nargs="+", default=None)
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--n-groups", type=int, default=8)
    p.add_argument("--diffusion-step-embed-dim", type=int, default=None)
    p.add_argument("--vision-backbone", default="resnet18")
    p.add_argument("--resize-shape", type=int, nargs=2, default=[96, 96])
    p.add_argument("--crop-ratio", type=float, default=0.84)
    p.add_argument("--spatial-softmax-kp", type=int, default=32)
    p.add_argument("--num-train-timesteps", type=int, default=100)
    p.add_argument("--dino-model", default="dinov2_vits14")
    p.add_argument("--dino-out-dim", type=int, default=None)
    p.add_argument("--dino-resize", type=int, nargs=2, default=[224, 224])
    p.add_argument("--dino-frozen", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dino-vis-aug", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--rt-out-dim", type=int, default=None)
    p.add_argument("--rt-resize", type=int, nargs=2, default=[224, 224])
    p.add_argument("--depth-out-dim", type=int, default=None)
    p.add_argument("--depth-resize", type=int, nargs=2, default=[224, 224])
    p.add_argument("--fusion-token-dim", type=int, default=None)
    p.add_argument("--fusion-out-dim", type=int, default=None)
    p.add_argument("--fusion-nhead", type=int, default=None)
    p.add_argument("--fusion-ffn-dim", type=int, default=None)
    p.add_argument("--fusion-layers", type=int, default=None)
    p.add_argument("--fusion-num-output-tokens", type=int, default=1)
    p.add_argument("--vision-dropout-prob", type=float, default=0.20)
    p.add_argument("--depth-dropout-prob", type=float, default=0.10)
    p.add_argument("--state-dropout-prob", type=float, default=0.05)
    p.add_argument("--state-noise-std", type=float, default=0.01)

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-epochs", type=float, default=200)
    p.add_argument("--num-steps", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip-norm", type=float, default=10.0)
    p.add_argument("--lr-warmup-steps", type=int, default=500)
    p.add_argument("--lr-decay-steps", type=int, default=None, help="Cosine decay steps. Default: 75% of resolved training steps.")
    p.add_argument("--lr-floor", type=float, default=1e-7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stratify-batches-by-task-id", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--log-freq", type=int, default=200)
    p.add_argument("--save-freq", type=int, default=50001)
    p.add_argument("--save-every-n-epochs", type=float, default=50)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def apply_capacity_preset(args: argparse.Namespace) -> None:
    defaults = {
        "down_dims": [128, 256, 512],
        "diffusion_step_embed_dim": 128,
        "dino_out_dim": 32,
        "rt_out_dim": 32,
        "depth_out_dim": 32,
        "fusion_token_dim": 32,
        "fusion_out_dim": 64,
        "fusion_nhead": 4,
        "fusion_ffn_dim": 128,
        "fusion_layers": 2,
    }
    preset = defaults if args.capacity_preset == "custom" else {**defaults, **CAPACITY_PRESETS[args.capacity_preset]}
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value.copy() if isinstance(value, list) else value)


def main() -> None:
    args = parse_args()
    apply_capacity_preset(args)
    args.camera_key = args.camera_key or None
    args.depth_key = args.depth_key or None
    if len(args.observation_frame_offsets) != args.n_obs_steps:
        raise ValueError("--observation-frame-offsets length must equal --n-obs-steps.")
    if tuple(sorted(args.observation_frame_offsets)) != tuple(args.observation_frame_offsets):
        raise ValueError("--observation-frame-offsets must be sorted.")
    if args.observation_frame_offsets[-1] != 0:
        raise ValueError("--observation-frame-offsets must end at current frame 0.")
    if args.horizon % (2 ** len(args.down_dims)) != 0:
        raise ValueError("--horizon must be divisible by 2 ** len(--down-dims).")
    if args.fusion_out_dim % args.fusion_nhead != 0:
        raise ValueError("--fusion-out-dim must be divisible by --fusion-nhead.")
    if not 0.0 <= args.vision_dropout_prob <= 1.0:
        raise ValueError("--vision-dropout-prob must be in [0, 1].")
    if not 0.0 <= args.depth_dropout_prob <= 1.0:
        raise ValueError("--depth-dropout-prob must be in [0, 1].")
    if not 0.0 <= args.state_dropout_prob <= 1.0:
        raise ValueError("--state-dropout-prob must be in [0, 1].")
    if args.vision_dropout_prob + args.depth_dropout_prob + args.state_dropout_prob > 1.0:
        raise ValueError("Vision, depth, and state dropout probabilities must sum to <= 1.")
    if args.state_noise_std < 0:
        raise ValueError("--state-noise-std must be non-negative.")
    if args.save_every_n_epochs is not None and args.save_every_n_epochs <= 0:
        raise ValueError("--save-every-n-epochs must be positive when set.")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "hparams.json", vars(args))

    log.info("Loading dataset metadata: %s", args.dataset_repo_id)
    ds_meta = LeRobotDatasetMetadata(repo_id=args.dataset_repo_id, root=args.dataset_root)
    dino_keys = camera_keys_for_args(args)
    for key in dino_keys:
        if key not in ds_meta.features:
            raise KeyError(f"DINO visual key {key!r} missing from dataset.")
    if args.rt_trajectory_key not in ds_meta.features:
        raise KeyError(f"RT trajectory image key {args.rt_trajectory_key!r} missing from dataset.")
    if args.no_depth:
        depth_keys = []
    elif args.depth_key:
        depth_keys = [args.depth_key]
    else:
        depth_keys = [depth_key_for(key, args.depth_prefix) for key in dino_keys if depth_key_for(key, args.depth_prefix) in ds_meta.features]
    for key in depth_keys:
        if key not in ds_meta.features:
            raise KeyError(f"Depth key {key!r} missing from dataset.")
    dino_env_dim = args.dino_out_dim * len(dino_keys)
    rt_env_dim = args.rt_out_dim
    depth_env_dim = args.depth_out_dim * len(depth_keys)
    env_dim = dino_env_dim + rt_env_dim + depth_env_dim
    dp_cfg = build_config(args, ds_meta, env_dim=env_dim)
    log.info("DP inputs: %s", list(dp_cfg.input_features))
    log.info("DP down_dims: %s", dp_cfg.down_dims)
    log.info("DINO keys: %s | env dim=%s", dino_keys, dino_env_dim)
    log.info("RT trajectory key: %s | env dim=%s", args.rt_trajectory_key, rt_env_dim)
    log.info("Depth keys: %s | env dim=%s", depth_keys, depth_env_dim)
    log.info("Capacity preset: %s | DINO aug=%s", args.capacity_preset, args.dino_vis_aug)
    log.info(
        "Dropout: vision=%.3f depth=%.3f state=%.3f | state_noise=%.3f",
        args.vision_dropout_prob,
        args.depth_dropout_prob,
        args.state_dropout_prob,
        args.state_noise_std,
    )

    delta_timestamps = resolve_delta_timestamps(dp_cfg, ds_meta)
    obs_ts = [offset / ds_meta.fps for offset in args.observation_frame_offsets]
    for key in ds_meta.features:
        if key.startswith("observation."):
            delta_timestamps[key] = obs_ts

    requested = parse_episode_list(args.episodes, ds_meta.total_episodes) if args.episodes else list(range(ds_meta.total_episodes))
    excluded = set(parse_episode_list(args.exclude_episodes, ds_meta.total_episodes)) if args.exclude_episodes else set()
    train_episodes = [ep for ep in requested if ep not in excluded]
    if not train_episodes:
        raise ValueError("No training episodes after exclusions.")
    log.info("Episodes selected: %d/%d | excluded=%s", len(train_episodes), ds_meta.total_episodes, sorted(excluded))

    dataset = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        episodes=train_episodes,
        delta_timestamps=delta_timestamps,
        download_videos=True,
    )
    for key in dataset.meta.camera_keys:
        for stat_type, stat_val in IMAGENET_STATS.items():
            dataset.meta.stats.setdefault(key, {})[stat_type] = torch.tensor(stat_val, dtype=torch.float32)

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": args.device == "cuda",
        "prefetch_factor": 2 if args.num_workers > 0 else None,
    }
    if args.stratify_batches_by_task_id:
        if "task_id" not in dataset.hf_dataset.column_names:
            raise ValueError("Task stratification requires scalar task_id feature.")
        task_ids = [scalar_task_id(value) for value in dataset.hf_dataset["task_id"]]
        batch_sampler = TaskStratifiedBatchSampler(task_ids, args.batch_size, drop_last=True, seed=args.seed)
        dataloader = DataLoader(dataset, batch_sampler=batch_sampler, **loader_kwargs)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kwargs)

    steps_per_epoch = len(dataloader)
    num_steps = args.num_steps if args.num_epochs is None else math.ceil(args.num_epochs * steps_per_epoch)
    if args.lr_decay_steps is None:
        args.lr_decay_steps = math.ceil(num_steps * 0.75)
    log.info(
        "Train frames=%s | steps/epoch=%s | steps=%s | cosine_decay=%s",
        f"{len(dataset):,}",
        f"{steps_per_epoch:,}",
        f"{num_steps:,}",
        f"{args.lr_decay_steps:,}",
    )

    log.info("Loading DINO encoder: %s", args.dino_model)
    dino_encoder = DinoVisualEncoder(
        args.dino_model,
        out_dim=args.dino_out_dim,
        frozen=args.dino_frozen,
        resize=tuple(args.dino_resize),
    ).to(args.device)
    dino_aug = DinoImageAugTransform().to(args.device) if args.dino_vis_aug else None
    rt_encoder = RtTrajectoryImageEncoder(args.rt_out_dim, resize=tuple(args.rt_resize)).to(args.device)
    depth_encoder = (
        DepthVisualEncoder(args.depth_out_dim, resize=tuple(args.depth_resize)).to(args.device)
        if depth_keys
        else None
    )

    policy = make_policy(cfg=dp_cfg, ds_meta=ds_meta)
    state_dim = tuple(ds_meta.features["observation.state"]["shape"])[0] if args.use_joint_state else 1
    policy.diffusion = FusedDinoConditioningDiffusionModel(
        dp_cfg,
        state_dim=state_dim,
        dino_dim=dino_env_dim,
        rt_dim=rt_env_dim,
        depth_dim=depth_env_dim,
        args=args,
    ).to(args.device)
    policy.to(args.device)
    policy.train()
    dino_encoder.train()
    rt_encoder.train()
    if depth_encoder is not None:
        depth_encoder.train()
    log.info("Policy trainable params: %s", f"{sum(p.numel() for p in policy.parameters() if p.requires_grad):,}")
    log.info("DINO trainable params: %s", f"{sum(p.numel() for p in dino_encoder.parameters() if p.requires_grad):,}")
    log.info("RT image trainable params: %s", f"{sum(p.numel() for p in rt_encoder.parameters() if p.requires_grad):,}")
    if depth_encoder is not None:
        log.info("Depth trainable params: %s", f"{sum(p.numel() for p in depth_encoder.parameters() if p.requires_grad):,}")

    preprocessor, _postprocessor = make_pre_post_processors(policy_cfg=dp_cfg, dataset_stats=dataset.meta.stats)
    optimizer_cfg = AdamWConfig(lr=args.lr, weight_decay=args.weight_decay, grad_clip_norm=args.grad_clip_norm)
    optim_params = list(policy.get_optim_params())
    optim_params += [p for p in dino_encoder.parameters() if p.requires_grad]
    optim_params += list(rt_encoder.parameters())
    if depth_encoder is not None:
        optim_params += list(depth_encoder.parameters())
    optimizer = optimizer_cfg.build(optim_params)
    scheduler_cfg = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=args.lr_warmup_steps,
        num_decay_steps=args.lr_decay_steps,
        peak_lr=args.lr,
        decay_lr=args.lr_floor,
    )
    lr_scheduler = scheduler_cfg.build(optimizer, num_training_steps=num_steps)

    def prepare_batch(batch: dict[str, Any], train: bool) -> dict[str, Any]:
        dropout_draw = None
        if train and (args.vision_dropout_prob > 0 or args.depth_dropout_prob > 0 or args.state_dropout_prob > 0):
            dropout_draw = torch.rand(batch["observation.state"].shape[0], device=args.device)

        dino_imgs = []
        for key in dino_keys:
            imgs = batch.pop(key).to(args.device)
            if imgs.ndim == 4:
                imgs = imgs.unsqueeze(1)
            dino_imgs.append(imgs)
        if train and dino_aug is not None and dino_imgs:
            obs_steps_per_camera = [imgs.shape[1] for imgs in dino_imgs]
            if len({tuple(imgs.shape[2:]) for imgs in dino_imgs}) != 1:
                raise ValueError("DINO augmentation requires all camera tensors to share [C,H,W].")
            all_views = dino_aug(torch.cat(dino_imgs, dim=1))
            dino_imgs = list(torch.split(all_views, obs_steps_per_camera, dim=1))

        rt_imgs = batch.pop(args.rt_trajectory_key).to(args.device)
        if rt_imgs.ndim == 4:
            rt_imgs = rt_imgs.unsqueeze(1)
        depth_imgs = []
        for key in depth_keys:
            imgs = batch.pop(key).to(args.device)
            if imgs.ndim == 4:
                imgs = imgs.unsqueeze(1)
            depth_imgs.append(imgs)

        if not args.use_joint_state and "observation.state" in batch:
            state = batch["observation.state"].to(args.device)
            batch["observation.state"] = torch.zeros(*state.shape[:-1], 1, dtype=state.dtype, device=state.device)

        batch = preprocessor(batch)
        dino_feat = torch.cat([dino_encoder(imgs) for imgs in dino_imgs], dim=-1)
        if dropout_draw is not None and args.vision_dropout_prob > 0:
            drop_vision = dropout_draw < args.vision_dropout_prob
            dino_feat = dino_feat.masked_fill(drop_vision[:, None, None], 0.0)
        rt_feat = rt_encoder(rt_imgs)
        env_parts = [dino_feat, rt_feat]
        if depth_encoder is not None and depth_imgs:
            depth_feat = torch.cat([depth_encoder(imgs) for imgs in depth_imgs], dim=-1)
            if dropout_draw is not None and args.depth_dropout_prob > 0:
                drop_depth = (
                    (dropout_draw >= args.vision_dropout_prob)
                    & (dropout_draw < args.vision_dropout_prob + args.depth_dropout_prob)
                )
                depth_feat = depth_feat.masked_fill(drop_depth[:, None, None], 0.0)
            env_parts.append(depth_feat)
        batch["observation.environment_state"] = torch.cat(env_parts, dim=-1)
        state = batch.get("observation.state")
        if state is not None:
            if args.use_joint_state and train:
                if args.state_noise_std > 0:
                    state = state + torch.randn_like(state) * args.state_noise_std
                if dropout_draw is not None and args.state_dropout_prob > 0:
                    drop_state = (
                        (dropout_draw >= args.vision_dropout_prob + args.depth_dropout_prob)
                        & (
                            dropout_draw
                            < args.vision_dropout_prob + args.depth_dropout_prob + args.state_dropout_prob
                        )
                    )
                    state = state.masked_fill(drop_state.view(drop_state.shape[0], *([1] * (state.ndim - 1))), 0.0)
            batch["observation.state"] = state
        return batch

    sanity = prepare_batch(next(iter(dataloader)), train=False)
    with torch.no_grad():
        sanity_loss, _ = policy.forward(sanity)
    log.info("Sanity forward loss: %.6f", sanity_loss.item())
    if not torch.isfinite(sanity_loss):
        raise RuntimeError("Sanity forward produced non-finite loss.")
    if args.dry_run:
        return

    metrics = Metrics(args.output_dir)
    interrupt = {"stop": False}

    def signal_handler(signum, _frame):
        log.warning("Received %s; checkpointing after current step.", signal.Signals(signum).name)
        interrupt["stop"] = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    total_loss = 0.0
    log_count = 0
    next_epoch_checkpoint = args.save_every_n_epochs
    dl_iter = cycle(dataloader)
    pbar = tqdm(range(num_steps), desc="Training", unit="step", dynamic_ncols=True)
    for step in pbar:
        batch = prepare_batch(next(dl_iter), train=True)
        loss, _ = policy.forward(batch)
        if not torch.isfinite(loss):
            log.warning("[step %s] non-finite loss %.4g; skipping", step + 1, loss.item())
            optimizer.zero_grad()
            continue
        loss.backward()
        clip_params = list(policy.parameters())
        clip_params += [p for p in dino_encoder.parameters() if p.requires_grad]
        clip_params += list(rt_encoder.parameters())
        if depth_encoder is not None:
            clip_params += list(depth_encoder.parameters())
        torch.nn.utils.clip_grad_norm_(clip_params, args.grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        total_loss += float(loss.item())
        log_count += 1
        lr = optimizer.param_groups[0]["lr"]
        epoch_frac = (step + 1) / steps_per_epoch
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.2e}", "epoch": f"{epoch_frac:.1f}"})

        if (step + 1) % args.log_freq == 0:
            avg = total_loss / max(log_count, 1)
            log.info("[step %s] epoch~%.1f avg_loss=%.5f lr=%.2e", f"{step + 1:,}", epoch_frac, avg, lr)
            metrics.log_train(step + 1, avg, lr)
            total_loss = 0.0
            log_count = 0

        step_checkpoint = args.save_freq is not None and (step + 1) % args.save_freq == 0
        epoch_checkpoint = False
        if next_epoch_checkpoint is not None and epoch_frac + 1e-12 >= next_epoch_checkpoint:
            epoch_checkpoint = True
            while next_epoch_checkpoint <= epoch_frac + 1e-12:
                next_epoch_checkpoint += args.save_every_n_epochs

        if step_checkpoint or epoch_checkpoint or interrupt["stop"]:
            ckpt_dir = args.output_dir / f"checkpoint_{step + 1:07d}"
            policy.save_pretrained(ckpt_dir)
            save_checkpoint_extras(
                ckpt_dir,
                dino_encoder=dino_encoder,
                rt_encoder=rt_encoder,
                depth_encoder=depth_encoder,
                dino_keys=dino_keys,
                depth_keys=depth_keys,
                args=args,
            )
            reasons = []
            if step_checkpoint:
                reasons.append("step")
            if epoch_checkpoint:
                reasons.append("epoch")
            if interrupt["stop"]:
                reasons.append("interrupt")
            log.info("Saved checkpoint (%s): %s", ", ".join(reasons), ckpt_dir)
            if interrupt["stop"]:
                pbar.close()
                return
    pbar.close()
    final_dir = args.output_dir / "final"
    policy.save_pretrained(final_dir)
    save_checkpoint_extras(
        final_dir,
        dino_encoder=dino_encoder,
        rt_encoder=rt_encoder,
        depth_encoder=depth_encoder,
        dino_keys=dino_keys,
        depth_keys=depth_keys,
        args=args,
    )
    log.info("Saved final policy: %s", final_dir)


if __name__ == "__main__":
    main()
