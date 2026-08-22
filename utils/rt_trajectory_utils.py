from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

log = logging.getLogger(__name__)

GRIPPER_CLASSES = ("so101_gripper", "so101_moving_jaw")


def to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def sample_to_rgb(sample: dict[str, Any], camera_key: str) -> np.ndarray:
    image = to_numpy(sample[camera_key])
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.moveaxis(image, 0, -1)
    if np.issubdtype(image.dtype, np.floating):
        if image.max(initial=0.0) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image for {camera_key}, got {image.shape}")
    return image


def rgb_to_gray_bgr(image_rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB uint8 image to 3-channel grayscale BGR for YOLO."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def rgb_to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB uint8 image to BGR for YOLO."""
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def resolve_camera_key(dataset, requested: str) -> str:
    keys = list(dataset.features)
    if requested in keys:
        return requested
    matches = [key for key in keys if requested in key]
    if len(matches) == 1:
        log.info("Camera key %r resolved to %r.", requested, matches[0])
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous camera key {requested!r}: {matches}")
    image_keys = [key for key in keys if dataset.features[key].get("dtype") in ("image", "video")]
    if image_keys:
        log.warning("Camera key %r not found; using %r.", requested, image_keys[0])
        return image_keys[0]
    raise KeyError(f"No image/video feature available. Dataset keys: {keys}")


def find_gripper_action_index(dataset) -> int | None:
    names = dataset.meta.names.get("action", [])
    for candidate in ("so101_gripper", "gripper"):
        for i, name in enumerate(names):
            if candidate.lower() in name.lower():
                return i
    return None


def normalize_action_signal(dataset, actions: np.ndarray, grip_idx: int | None) -> np.ndarray:
    if grip_idx is None:
        return np.zeros(len(actions), dtype=np.float32)
    signal = actions[:, grip_idx].astype(np.float32)
    stats = dataset.meta.stats.get("action")
    if stats is None:
        lo = float(signal.min(initial=0.0))
        hi = float(signal.max(initial=1.0))
    else:
        lo = float(stats["min"][grip_idx])
        hi = float(stats["max"][grip_idx])
    return np.clip((signal - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def smooth_1d(signal: np.ndarray, window: int = 5) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    if window <= 1 or len(signal) < 3:
        return signal.copy()
    window = min(window, len(signal))
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return signal.copy()
    pad = window // 2
    padded = np.pad(signal, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def action_peak_starts(
    action_norm: np.ndarray,
    threshold: float = 0.45,
    smooth_window: int = 5,
    min_gap: int = 8,
    min_prominence: float = 0.05,
) -> list[int]:
    y = smooth_1d(action_norm, smooth_window)
    cutoff = max(float(threshold), float(y.min(initial=0.0) + min_prominence))
    active = y >= cutoff
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    ends = np.flatnonzero(active & ~np.r_[active[1:], False])
    if len(starts) == 0:
        maxima = np.flatnonzero((y[1:-1] >= y[:-2]) & (y[1:-1] > y[2:])) + 1
        maxima = maxima[y[maxima] >= y.min(initial=0.0) + min_prominence]
        return [int(i) for i in maxima]

    peaks: list[tuple[int, int, float]] = []
    for start, end in zip(starts, ends):
        if peaks and int(start) - peaks[-1][0] < min_gap:
            prev_start, _prev_end, prev_peak = peaks[-1]
            peak = float(y[start : end + 1].max(initial=prev_peak))
            peaks[-1] = (prev_start, int(end), max(prev_peak, peak))
        else:
            peak = float(y[start : end + 1].max(initial=0.0))
            peaks.append((int(start), int(end), peak))
    floor = float(y.min(initial=0.0) + min_prominence)
    return [start for start, _end, peak in peaks if peak >= floor]


def _action_bump_segments(
    action_norm: np.ndarray,
    threshold: float = 0.45,
    smooth_window: int = 5,
    min_gap: int = 8,
    min_prominence: float = 0.05,
) -> list[dict[str, int | float]]:
    """Return distinct gripper-action bumps with rise, peak, and fall indices."""
    y = smooth_1d(action_norm, smooth_window)
    if len(y) == 0:
        return []

    baseline = float(np.percentile(y, 10))
    amplitude = float(y.max(initial=baseline) - baseline)
    if amplitude <= 1e-6:
        return []

    # Keep the user threshold meaningful, but adapt downward for episodes where
    # the release command is much lighter than the close/grasp command.
    cutoff = max(
        baseline + min_prominence,
        min(float(threshold), baseline + 0.20 * amplitude),
    )
    active = y >= cutoff
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    ends = np.flatnonzero(active & ~np.r_[active[1:], False])

    if len(starts) == 0:
        maxima = np.flatnonzero((y[1:-1] >= y[:-2]) & (y[1:-1] > y[2:])) + 1
        maxima = maxima[y[maxima] >= baseline + min_prominence]
        return [
            {
                "rise": int(max(0, i - 1)),
                "peak": int(i),
                "fall": int(min(len(y) - 1, i + 1)),
                "end": int(min(len(y) - 1, i + 1)),
                "height": float(y[i]),
            }
            for i in maxima
        ]

    merged: list[tuple[int, int]] = []
    for start, end in zip(starts, ends):
        start_i, end_i = int(start), int(end)
        if merged and start_i - merged[-1][1] <= min_gap:
            merged[-1] = (merged[-1][0], end_i)
        else:
            merged.append((start_i, end_i))

    bumps: list[dict[str, int | float]] = []
    dy = np.diff(y, prepend=y[0])
    slope_eps = max(0.005, 0.02 * amplitude)
    for start, end in merged:
        segment = y[start : end + 1]
        peak = start + int(np.argmax(segment))
        fall_candidates = np.flatnonzero(dy[peak + 1 : end + 1] < -slope_eps)
        fall = (
            int(peak + 1 + fall_candidates[0])
            if len(fall_candidates)
            else int(min(end, peak + 1))
        )
        bumps.append(
            {
                "rise": int(start),
                "peak": int(peak),
                "fall": fall,
                "end": int(end),
                "height": float(y[peak]),
            }
        )
    return [b for b in bumps if float(b["height"]) >= baseline + min_prominence]


def detect_grasp_release(action_norm: np.ndarray, threshold: float = 0.45) -> tuple[int | None, int | None, list[int]]:
    """Return grasp/release marker frames from gripper-action bumps.

    The datasets contain a close/grasp bump and an open/release bump in every
    episode, but their magnitudes vary. The grasp marker is placed when the
    first bump starts dropping from its peak. The release marker is placed at
    the spike-up/rising edge of the second bump, or the last bump if extra
    command bumps are present.
    """
    action_norm = np.asarray(action_norm, dtype=np.float32)
    if len(action_norm) == 0:
        return None, None, []

    bumps = _action_bump_segments(action_norm, threshold=threshold)
    if not bumps:
        return None, None, []

    grasp = int(bumps[0]["fall"])
    release = int(bumps[-1]["rise"]) if len(bumps) >= 2 else None
    markers = [int(b["rise"]) for b in bumps]
    markers[0] = grasp
    if release is not None:
        markers[-1] = release
    return grasp, release, markers



def best_gripper_mask(results, frame_h: int, frame_w: int, model_names: dict[int, str]):
    if results[0].masks is None or results[0].boxes is None:
        return None
    masks_data = results[0].masks.data
    classes = results[0].boxes.cls
    confs = results[0].boxes.conf
    best = None
    for cls_name in GRIPPER_CLASSES:
        cls_id = next((idx for idx, name in model_names.items() if cls_name.lower() in name.lower()), None)
        if cls_id is None:
            continue
        for idx in range(len(classes)):
            if int(classes[idx].item()) != cls_id:
                continue
            raw = masks_data[idx].cpu().numpy()
            resized = cv2.resize(raw, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
            mask = (resized > 0.5).astype(np.uint8)
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue
            if len(xs) > 0.04 * frame_h * frame_w:
                continue
            border_margin = 3
            if (
                xs.min() <= border_margin
                or xs.max() >= frame_w - 1 - border_margin
                or ys.min() <= border_margin
                or ys.max() >= frame_h - 1 - border_margin
            ):
                continue
            candidate = {
                "mask": mask,
                "conf": float(confs[idx].item()),
                "max_y": int(ys.max()),
                "area": int(len(xs)),
                "class_name": cls_name,
            }
            if best is None or candidate["conf"] > best["conf"]:
                best = candidate
        if best is not None:
            return best
    return best


def gripper_tip_xy(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    best = int(np.argmax(ys))
    return float(xs[best]), float(ys[best])


def detect_gripper_tip_from_rgb(
    yolo_model,
    frame_rgb: np.ndarray,
    *,
    conf: float,
    grayscale: bool = False,
) -> tuple[float, float] | None:
    frame_h, frame_w = frame_rgb.shape[:2]
    frame_bgr = rgb_to_gray_bgr(frame_rgb) if grayscale else rgb_to_bgr(frame_rgb)
    results = yolo_model(frame_bgr, conf=conf, verbose=False)
    candidate = best_gripper_mask(results, frame_h, frame_w, yolo_model.names)
    if candidate is None:
        return None
    return gripper_tip_xy(candidate["mask"])


def interpolate_and_smooth_xy(points: np.ndarray, valid: np.ndarray, window: int = 11) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).copy()
    valid_idx = np.flatnonzero(valid)
    if len(valid_idx) == 0:
        return points
    all_idx = np.arange(len(points))
    for dim in (0, 1):
        values = points[valid_idx, dim]
        points[:, dim] = np.interp(all_idx, valid_idx, values).astype(np.float32)
    if len(points) >= 5 and window > 1:
        window = min(window, len(points))
        if window % 2 == 0:
            window -= 1
        if window >= 3:
            pad = window // 2
            kernel = np.ones(window, dtype=np.float32) / float(window)
            for dim in (0, 1):
                padded = np.pad(points[:, dim], (pad, pad), mode="edge")
                points[:, dim] = np.convolve(padded, kernel, mode="valid").astype(np.float32)
    return points


def yolo_tip_trajectory(
    yolo_model,
    frames_rgb: np.ndarray,
    *,
    conf: float,
    frame_stride: int,
    grayscale: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a robust gripper-tip trajectory using YOLO detections per frame."""
    indices = np.arange(0, len(frames_rgb), max(frame_stride, 1), dtype=int)
    if len(indices) == 0 or indices[-1] != len(frames_rgb) - 1:
        indices = np.r_[indices, len(frames_rgb) - 1]
    points = np.full((len(indices), 2), np.nan, dtype=np.float32)
    valid = np.zeros(len(indices), dtype=bool)
    for out_i, frame_i in enumerate(indices):
        tip = detect_gripper_tip_from_rgb(yolo_model, frames_rgb[frame_i], conf=conf, grayscale=grayscale)
        if tip is not None:
            points[out_i] = tip
            valid[out_i] = True
    if not valid.any():
        h, w = frames_rgb.shape[1:3]
        points[:] = (w / 2.0, h / 2.0)
    else:
        points = interpolate_and_smooth_xy(points, valid)
    return points.astype(np.float32), {
        "source": "yolo_tip_fallback",
        "frame_indices": indices.tolist(),
        "detections": int(valid.sum()),
        "frames": int(len(indices)),
    }


def trajectory_is_usable(trajectory_xy: np.ndarray, min_span_px: float = 8.0) -> bool:
    pts = np.asarray(trajectory_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        return False
    if not np.isfinite(pts).all():
        return False
    span = np.nanmax(pts, axis=0) - np.nanmin(pts, axis=0)
    return bool(float(np.linalg.norm(span)) >= min_span_px)


def infer_tip_anchors(mask: np.ndarray, band_quantile: float, min_separation_px: float = 8.0):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []
    y_cut = float(np.quantile(ys, np.clip(band_quantile, 0.0, 1.0)))
    bottom = ys >= y_cut
    bx, by = xs[bottom], ys[bottom]
    if len(bx) < 4:
        bx, by = xs, ys
    split = float(np.median(bx))
    anchors = []
    for side in (bx <= split, bx > split):
        if int(side.sum()) < 2:
            continue
        sx, sy = bx[side], by[side]
        near_tip = sy >= sy.max() - 2
        anchors.append((float(sx[near_tip].mean()), float(sy[near_tip].mean())))
    if len(anchors) == 2 and abs(anchors[0][0] - anchors[1][0]) >= min_separation_px:
        return anchors
    near_tip = by >= by.max() - 2
    return [(float(bx[near_tip].mean()), float(by[near_tip].mean()))]


def sample_tip_points(
    mask: np.ndarray,
    n_points: int,
    rng: np.random.Generator,
    band_quantile: float,
    radius_px: float,
) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("empty gripper mask")
    anchors = infer_tip_anchors(mask, band_quantile)
    if not anchors:
        idx = rng.choice(len(xs), size=n_points, replace=len(xs) < n_points)
        return np.stack(
            [np.zeros(n_points, dtype=np.float32), ys[idx].astype(np.float32), xs[idx].astype(np.float32)],
            axis=1,
        )
    counts = [n_points // len(anchors)] * len(anchors)
    for i in range(n_points % len(anchors)):
        counts[i] += 1
    out_x, out_y = [], []
    for (ax, ay), count in zip(anchors, counts):
        d2 = (xs.astype(np.float32) - ax) ** 2 + (ys.astype(np.float32) - ay) ** 2
        local = d2 <= radius_px**2
        if int(local.sum()) < max(2, count // 2):
            pool = np.argsort(d2)[: max(count, 4)]
            lx, ly = xs[pool], ys[pool]
        else:
            lx, ly = xs[local], ys[local]
        idx = rng.choice(len(lx), size=count, replace=len(lx) < count)
        out_x.append(lx[idx].astype(np.float32))
        out_y.append(ly[idx].astype(np.float32))
    return np.stack(
        [np.zeros(n_points, dtype=np.float32), np.concatenate(out_y), np.concatenate(out_x)],
        axis=1,
    )


def sample_bottom_band_tip_points(
    mask: np.ndarray,
    n_points: int,
    rng: np.random.Generator,
    band_quantile: float,
) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("empty gripper mask")
    y_cut = float(np.quantile(ys, np.clip(band_quantile, 0.0, 1.0)))
    tip_sel = ys >= y_cut
    if int(tip_sel.sum()) < max(1, n_points // 2):
        tip_sel = np.ones_like(ys, dtype=bool)
    tip_ys = ys[tip_sel]
    tip_xs = xs[tip_sel]
    n_avail = len(tip_xs)
    idx = rng.choice(n_avail, size=min(n_points, n_avail), replace=False)
    sampled_y = tip_ys[idx].astype(np.float32)
    sampled_x = tip_xs[idx].astype(np.float32)
    if n_avail < n_points:
        extra = rng.choice(n_avail, size=n_points - n_avail, replace=True)
        sampled_y = np.concatenate([sampled_y, tip_ys[extra].astype(np.float32)])
        sampled_x = np.concatenate([sampled_x, tip_xs[extra].astype(np.float32)])
    return np.stack([np.zeros(n_points, dtype=np.float32), sampled_y, sampled_x], axis=1)


def render_rt_trajectory_image(
    trajectory_xy: np.ndarray,
    image_shape: tuple[int, int],
    grasp_idx: int | None,
    release_idx: int | None,
    line_width: int = 5,
    marker_radius: int = 15,
    marker_width: int = 8,
) -> np.ndarray:
    h, w = image_shape
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    pts = np.asarray(trajectory_xy, dtype=np.float32).copy()
    if release_idx is not None and 0 <= release_idx < len(pts):
        pts = pts[: release_idx + 1]
    if len(pts) == 0:
        return np.asarray(canvas)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    denom = max(len(pts) - 1, 1)
    for i in range(len(pts) - 1):
        red = int(np.clip(round(255 * i / denom), 0, 255))
        p0 = tuple(np.round(pts[i]).astype(int))
        p1 = tuple(np.round(pts[i + 1]).astype(int))
        draw.line([p0, p1], fill=(red, 0, 0), width=line_width)

    def ring(idx: int | None, color: tuple[int, int, int]) -> None:
        if idx is None or not (0 <= idx < len(pts)):
            return
        x, y = np.round(pts[idx]).astype(int)
        outer = [x - marker_radius, y - marker_radius, x + marker_radius, y + marker_radius]
        inner_r = max(marker_radius - marker_width, 1)
        inner = [x - inner_r, y - inner_r, x + inner_r, y + inner_r]
        draw.ellipse(outer, fill=color)
        draw.ellipse(inner, fill=(0, 0, 0))

    ring(grasp_idx, (0, 255, 0))
    ring(release_idx, (0, 0, 255))
    return np.asarray(canvas)


def render_debug_overlay(frame_rgb: np.ndarray, rt_img: np.ndarray, alpha: float = 0.75) -> np.ndarray:
    frame = frame_rgb.astype(np.float32)
    rt = rt_img.astype(np.float32)
    mask = (rt.max(axis=2, keepdims=True) > 0).astype(np.float32) * alpha
    return (frame * (1.0 - mask) + rt * mask).clip(0, 255).astype(np.uint8)


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
