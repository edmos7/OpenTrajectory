#!/usr/bin/env python3
"""Benchmark Depth Anything V2 Small on LeRobot dataset episodes.

This script saves per-frame RGB/depth visualizations, optional per-episode
videos, and a CSV latency summary.

Expected setup:
    git clone https://github.com/DepthAnything/Depth-Anything-V2 /path/to/Depth-Anything-V2
    conda run -n lerobotclone python data_generation/demo_depth_anything_v2_small.py \
        --depth-anything-dir /path/to/Depth-Anything-V2 \
        --episodes 0 5 10 \
        --video \
        --frame-stride 5

By default the Small checkpoint is downloaded from Hugging Face:
    depth-anything/Depth-Anything-V2-Small/depth_anything_v2_vits.pth
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from PIL import Image, ImageDraw, ImageFont

DEFAULT_REPO_ID = "edmos7/so100_multitask_quality_checked_tip_v1"
DEFAULT_CAMERA_KEYS = ["observation.images.side", "observation.images.wrist"]
DEFAULT_OUT_DIR = Path.cwd() / "outputs" / "depth_anything_v2_small_demo"

MODEL_CONFIG = {
    "encoder": "vits",
    "features": 64,
    "out_channels": [48, 96, 192, 384],
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Depth Anything V2 Small on selected LeRobot episode frames."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 5, 10])
    parser.add_argument(
        "--camera-keys",
        nargs="+",
        default=DEFAULT_CAMERA_KEYS,
        help="Camera observation keys to process. Defaults to the training-script side and wrist views.",
    )
    parser.add_argument(
        "--frames-per-episode",
        type=int,
        default=5,
        help="Number of evenly-spaced still visualizations to save per episode.",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Also render a depth visualization video for each selected episode.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=5,
        help="Use every Nth episode frame when rendering videos.",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help="Output video FPS. Defaults to dataset fps divided by --frame-stride.",
    )
    parser.add_argument("--depth-anything-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default=None)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames per episode/camera video. Stills are still controlled by --frames-per-episode.",
    )
    parser.add_argument("--save-npy", action="store_true")
    parser.add_argument("--download-videos", action="store_true")
    return parser.parse_args(argv)


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def add_depth_anything_to_path(depth_anything_dir: Path) -> None:
    dpt_file = depth_anything_dir / "depth_anything_v2" / "dpt.py"
    if not dpt_file.exists():
        raise FileNotFoundError(
            f"Could not find {dpt_file}. Clone https://github.com/DepthAnything/Depth-Anything-V2 "
            "and pass it via --depth-anything-dir."
        )
    sys.path.insert(0, str(depth_anything_dir.resolve()))


def resolve_checkpoint(checkpoint: Path | None) -> Path:
    if checkpoint is not None:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        return checkpoint

    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id="depth-anything/Depth-Anything-V2-Small",
            filename="depth_anything_v2_vits.pth",
        )
    )


def load_model(depth_anything_dir: Path, checkpoint: Path | None, device: torch.device):
    add_depth_anything_to_path(depth_anything_dir)
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(**MODEL_CONFIG)
    state = torch.load(resolve_checkpoint(checkpoint), map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model


def tensor_to_rgb(sample_value: Any) -> np.ndarray:
    if torch.is_tensor(sample_value):
        arr = sample_value.detach().cpu()
        if arr.ndim != 3:
            raise ValueError(f"Expected image tensor [C,H,W], got {tuple(arr.shape)}")
        arr = arr.permute(1, 2, 0).numpy()
    else:
        arr = np.asarray(sample_value)

    if arr.dtype != np.uint8:
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    if arr.ndim != 3:
        raise ValueError(f"Expected RGB image array [H,W,C], got {arr.shape}")
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.shape[2] != 3:
        raise ValueError(f"Expected 1 or 3 image channels, got {arr.shape[2]}")
    return arr


def depth_to_uint8(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    lo, hi = np.percentile(depth, [2.0, 98.0])
    if hi <= lo:
        hi = float(depth.max())
        lo = float(depth.min())
    norm = (depth - lo) / max(hi - lo, 1e-6)
    return (norm.clip(0.0, 1.0) * 255.0).astype(np.uint8)


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    gray = depth_to_uint8(depth)
    try:
        import cv2

        colored = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    except Exception:
        r = gray
        g = np.sqrt(gray.astype(np.float32) / 255.0) * 255.0
        b = 255 - gray
        return np.stack([r, g.astype(np.uint8), b], axis=-1)


def add_label(image: Image.Image, text: str) -> Image.Image:
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    margin = 8
    bbox = draw.textbbox((margin, margin), text, font=font)
    pad = 4
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        fill=(0, 0, 0),
    )
    draw.text((margin, margin), text, fill=(255, 255, 255), font=font)
    return out


def save_visualization(
    rgb: np.ndarray,
    depth: np.ndarray,
    out_path: Path,
    label: str,
) -> None:
    rgb_img = Image.fromarray(rgb, "RGB")
    depth_rgb = colorize_depth(depth)
    depth_img = Image.fromarray(depth_rgb, "RGB").resize(rgb_img.size, Image.BILINEAR)
    overlay = Image.blend(rgb_img, depth_img, alpha=0.45)

    panels = [
        add_label(rgb_img, f"{label} | RGB"),
        add_label(depth_img, "Depth Anything V2 Small"),
        add_label(overlay, "Overlay"),
    ]
    w, h = rgb_img.size
    strip = Image.new("RGB", (w * len(panels), h), (24, 24, 24))
    for i, panel in enumerate(panels):
        strip.paste(panel, (i * w, 0))
    strip.save(out_path)


def scalar_int(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.flatten()[0].item())
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def episode_positions(dataset: LeRobotDataset, episode: int) -> list[tuple[int, int]]:
    """Return (local_frame, dataset_row) pairs for an episode in the loaded subset."""
    episodes = dataset.hf_dataset["episode_index"]
    frames = dataset.hf_dataset["frame_index"]
    positions = [
        (scalar_int(frame), row)
        for row, (ep, frame) in enumerate(zip(episodes, frames, strict=True))
        if scalar_int(ep) == episode
    ]
    positions.sort(key=lambda item: item[0])
    return positions


def episode_frame_indices(
    dataset: LeRobotDataset, episode: int, frames_per_episode: int
) -> list[tuple[int, int]]:
    positions = episode_positions(dataset, episode)
    if not positions:
        return []
    selected = np.linspace(0, len(positions) - 1, max(1, frames_per_episode), dtype=int)
    return [positions[int(i)] for i in selected]


def all_episode_frame_indices(
    dataset: LeRobotDataset, episode: int, frame_stride: int
) -> list[tuple[int, int]]:
    return episode_positions(dataset, episode)[:: max(1, frame_stride)]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def infer_depth(
    model: Any, rgb: np.ndarray, device: torch.device, input_size: int
) -> tuple[np.ndarray, float]:
    # Official DA V2 expects OpenCV-style BGR uint8 input.
    raw_bgr = rgb[:, :, ::-1].copy()
    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        depth = model.infer_image(raw_bgr, input_size=input_size)
    synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return np.asarray(depth), elapsed_ms


def make_visualization_array(rgb: np.ndarray, depth: np.ndarray, label: str) -> np.ndarray:
    rgb_img = Image.fromarray(rgb, "RGB")
    depth_rgb = colorize_depth(depth)
    depth_img = Image.fromarray(depth_rgb, "RGB").resize(rgb_img.size, Image.BILINEAR)
    overlay = Image.blend(rgb_img, depth_img, alpha=0.45)

    panels = [
        add_label(rgb_img, f"{label} | RGB"),
        add_label(depth_img, "Depth Anything V2 Small"),
        add_label(overlay, "Overlay"),
    ]
    w, h = rgb_img.size
    strip = Image.new("RGB", (w * len(panels), h), (24, 24, 24))
    for i, panel in enumerate(panels):
        strip.paste(panel, (i * w, 0))
    return np.asarray(strip)


def write_summary(out_dir: Path, args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    latencies = [float(row["latency_ms"]) for row in rows]
    summary = {
        "repo_id": args.repo_id,
        "camera_keys": args.camera_keys,
        "episodes": args.episodes,
        "frames": len(rows),
        "input_size": args.input_size,
        "device": args.device,
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "latency.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "camera_key",
                "local_frame",
                "dataset_row",
                "latency_ms",
                "depth_min",
                "depth_max",
                "path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> None:
    args = parse_args(argv)
    device = resolve_device(args.device)
    args.device = str(device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset metadata: {args.repo_id}")
    meta = LeRobotDatasetMetadata(repo_id=args.repo_id, root=args.dataset_root)
    missing_camera_keys = [key for key in args.camera_keys if key not in meta.features]
    if missing_camera_keys:
        camera_keys = [key for key in meta.features if key.startswith("observation.images.")]
        raise KeyError(f"Missing camera keys {missing_camera_keys}. Available image keys: {camera_keys}")

    bad_eps = [ep for ep in args.episodes if ep < 0 or ep >= meta.total_episodes]
    if bad_eps:
        raise ValueError(f"Episode ids out of range 0..{meta.total_episodes - 1}: {bad_eps}")

    print(f"Loading dataset episodes: {args.episodes}")
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.dataset_root,
        episodes=args.episodes,
        download_videos=args.download_videos,
    )

    print(f"Loading Depth Anything V2 Small on {device}")
    model = load_model(args.depth_anything_dir, args.checkpoint, device)

    first_ep = args.episodes[0]
    first_row = episode_frame_indices(dataset, first_ep, 1)[0][1]
    first_rgb = tensor_to_rgb(dataset[first_row][args.camera_keys[0]])
    for _ in range(max(0, args.warmup)):
        infer_depth(model, first_rgb, device, args.input_size)

    rows: list[dict[str, Any]] = []
    for episode in args.episodes:
        for camera_key in args.camera_keys:
            camera_name = camera_key.rsplit(".", 1)[-1]
            video_saved = 0
            if args.video:
                import imageio.v2 as imageio

                video_path = args.out_dir / f"episode_{episode:03d}_{camera_name}_dav2_small.mp4"
                fps = args.video_fps
                if fps is None:
                    fps = max(1.0, float(meta.fps) / max(1, args.frame_stride))
                video_rows = []
                print(f"rendering {video_path} at {fps:.2f} fps")
                with imageio.get_writer(video_path, fps=fps, quality=8, macro_block_size=1) as writer:
                    for local_frame, dataset_row in all_episode_frame_indices(
                        dataset, episode, args.frame_stride
                    ):
                        if args.max_frames is not None and video_saved >= args.max_frames:
                            break
                        sample = dataset[dataset_row]
                        rgb = tensor_to_rgb(sample[camera_key])
                        depth, latency_ms = infer_depth(model, rgb, device, args.input_size)
                        writer.append_data(
                            make_visualization_array(
                                rgb,
                                depth,
                                label=(
                                    f"ep {episode} {camera_name} frame {local_frame} | {latency_ms:.1f} ms"
                                ),
                            )
                        )
                        row = {
                            "episode": episode,
                            "camera_key": camera_key,
                            "local_frame": local_frame,
                            "dataset_row": dataset_row,
                            "latency_ms": f"{latency_ms:.3f}",
                            "depth_min": f"{float(np.min(depth)):.6f}",
                            "depth_max": f"{float(np.max(depth)):.6f}",
                            "path": str(video_path),
                        }
                        rows.append(row)
                        video_rows.append(row)
                        video_saved += 1
                if video_rows:
                    print(f"saved {video_path} ({len(video_rows)} frames)")

            for local_frame, dataset_row in episode_frame_indices(dataset, episode, args.frames_per_episode):
                sample = dataset[dataset_row]
                rgb = tensor_to_rgb(sample[camera_key])
                depth, latency_ms = infer_depth(model, rgb, device, args.input_size)

                stem = f"episode_{episode:03d}_{camera_name}_frame_{local_frame:05d}"
                vis_path = args.out_dir / f"{stem}_dav2_small.png"
                save_visualization(
                    rgb,
                    depth,
                    vis_path,
                    label=f"ep {episode} {camera_name} frame {local_frame} | {latency_ms:.1f} ms",
                )
                if args.save_npy:
                    np.save(args.out_dir / f"{stem}_depth.npy", depth.astype(np.float32))

                rows.append(
                    {
                        "episode": episode,
                        "camera_key": camera_key,
                        "local_frame": local_frame,
                        "dataset_row": dataset_row,
                        "latency_ms": f"{latency_ms:.3f}",
                        "depth_min": f"{float(np.min(depth)):.6f}",
                        "depth_max": f"{float(np.max(depth)):.6f}",
                        "path": str(vis_path),
                    }
                )
                print(f"saved {vis_path} ({latency_ms:.1f} ms)")

    write_summary(args.out_dir, args, rows)
    print(f"Wrote {len(rows)} visualizations to {args.out_dir}")
    if rows:
        latencies = [float(row["latency_ms"]) for row in rows]
        print(
            "Latency ms: "
            f"mean={statistics.fmean(latencies):.1f}, "
            f"median={statistics.median(latencies):.1f}, "
            f"min={min(latencies):.1f}, "
            f"max={max(latencies):.1f}"
        )


if __name__ == "__main__":
    main()
