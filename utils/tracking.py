"""tracking.py — TAPNext++-based point tracking utilities.

Public API
----------
load_tapnext_model(checkpoint_path, device, image_size)
    Load and return a TAPNext++ model.

run_tapnext(model, frames_uint8, query_points, target_size, device)
    Run TAPNext++ online on a video clip and return tracks and visibility.
    TAPNext++ processes frames one at a time, so peak GPU memory is
    independent of episode length.

compute_median_trajectory(tracks, visibles)
    Compute per-frame median (x, y) over all visible tracked points.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
import torch

from tapnet.tapnext.tapnext_torch import TAPNext

logger = logging.getLogger(__name__)

# ── model loading ──────────────────────────────────────────────────────────────

def load_tapnext_model(
    checkpoint_path: str,
    device: Optional[torch.device | str] = None,
    image_size: tuple[int, int] = (256, 256),
) -> TAPNext:
    """Load TAPNext++ weights and return the model in eval mode.

    Parameters
    ----------
    checkpoint_path:
        Path to a ``tapnextpp_ckpt.pt`` checkpoint.
    device:
        Torch device string / object.  Defaults to CUDA if available.
    image_size:
        ``(H, W)`` the model was trained at.  Must match ``target_size``
        passed to :func:`run_tapnext`.  Default ``(256, 256)``.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    default_device = torch.get_default_device()
    torch.set_default_device(device)
    try:
        model = TAPNext(image_size=image_size)
    finally:
        torch.set_default_device(default_device)
    ckpt  = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(
        {k.replace("tapnext.", ""): v for k, v in ckpt["state_dict"].items()}
    )
    model.to(device)
    model.eval()
    logger.info("TAPNext++ model loaded from %s on %s.", checkpoint_path, device)
    return model


# ── inference ──────────────────────────────────────────────────────────────────

def run_tapnext(
    model: TAPNext,
    frames_uint8: np.ndarray,
    query_points: np.ndarray,
    target_size: tuple[int, int] = (256, 256),
    device: Optional[torch.device | str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Track *query_points* through *frames_uint8* using TAPNext++.

    TAPNext++ is an online model: it processes frames one at a time and
    maintains a recurrent state, so peak GPU memory is O(1) in sequence
    length regardless of episode duration.

    Parameters
    ----------
    model:
        A loaded TAPNext++ model (from :func:`load_tapnext_model`).
    frames_uint8:
        Video frames as ``[T, H, W, 3]`` uint8 (RGB order).
    query_points:
        Initial query points as ``[N, 3]`` float32 with rows ``(t, y, x)``
        in the *original* pixel space of *frames_uint8*.
    target_size:
        ``(height, width)`` to resize frames before inference.  Must match
        the ``image_size`` the model was loaded with.
    device:
        Torch device.  Defaults to CUDA if available.

    Returns
    -------
    tracks : np.ndarray
        Shape ``[N, T, 2]`` float32 — ``(x, y)`` in original pixel coordinates.
    visibles : np.ndarray
        Shape ``[N, T]`` bool — True where the point is visible/reliable.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    T, orig_H, orig_W, _ = frames_uint8.shape
    tgt_H, tgt_W = target_size

    # ── resize frames ──────────────────────────────────────────────────────────
    if orig_H == tgt_H and orig_W == tgt_W:
        resized = frames_uint8
    else:
        resized = np.stack(
            [cv2.resize(f, (tgt_W, tgt_H)) for f in frames_uint8], axis=0
        )  # [T, tgt_H, tgt_W, 3]

    # ── normalise to [-1, 1] → [1, T, H, W, 3] ────────────────────────────────
    video = (torch.from_numpy(resized).float() / 127.5 - 1.0).unsqueeze(0).to(device)

    # ── scale query points to target resolution ────────────────────────────────
    scale_y = tgt_H / orig_H
    scale_x = tgt_W / orig_W
    qp = query_points.copy()
    qp[:, 1] *= scale_y   # y
    qp[:, 2] *= scale_x   # x
    qp_t = torch.from_numpy(qp).float().unsqueeze(0).to(device)  # [1, N, 3]

    # ── online inference (frame by frame) ──────────────────────────────────────
    all_tracks: list[torch.Tensor] = []
    all_vis:    list[torch.Tensor] = []

    with torch.no_grad():
        pred_tracks, _, vis_logits, state = model(
            video=video[:, :1], query_points=qp_t
        )
        all_tracks.append(pred_tracks.cpu())        # [1, 1, N, 2]
        all_vis.append((vis_logits > 0).cpu())      # [1, 1, N, 1]

        for t in range(1, T):
            pred_tracks, _, vis_logits, state = model(
                video=video[:, t : t + 1], state=state
            )
            all_tracks.append(pred_tracks.cpu())
            all_vis.append((vis_logits > 0).cpu())

    # ── assemble: [1, T, N, 2] → [N, T, 2] ───────────────────────────────────
    tracks_t = torch.cat(all_tracks, dim=1).transpose(1, 2)[0]  # [N, T, 2]
    vis_t    = torch.cat(all_vis,    dim=1).transpose(1, 2)[0].squeeze(-1)  # [N, T]

    # TAPNext++ outputs (y, x) internally; swap to (x, y) convention.
    tracks_np = tracks_t.numpy()[..., ::-1].copy()   # [N, T, 2] now (x, y)
    visibles  = vis_t.numpy()                         # [N, T] bool

    # ── scale tracks back to original resolution ───────────────────────────────
    tracks_np[..., 0] /= scale_x   # x
    tracks_np[..., 1] /= scale_y   # y

    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    return tracks_np, visibles


# ── median trajectory ──────────────────────────────────────────────────────────

def compute_median_trajectory(
    tracks: np.ndarray,
    visibles: np.ndarray,
) -> np.ndarray:
    """Return the per-frame median (x, y) over all tracked points.

    TAPNext++ predicts point positions recurrently even during occlusion, so
    all N predicted positions are used at every timestep regardless of the
    visibility mask.

    Parameters
    ----------
    tracks:
        ``[N, T, 2]`` float32 tracks as returned by :func:`run_tapnext`.
    visibles:
        Unused.  Kept for API compatibility.

    Returns
    -------
    traj : np.ndarray
        Shape ``[T, 2]`` float32.  Each row is the median ``(x, y)`` at that
        timestep.
    """
    _, T, _ = tracks.shape
    return np.median(tracks, axis=0).astype(np.float32)  # [T, 2]
