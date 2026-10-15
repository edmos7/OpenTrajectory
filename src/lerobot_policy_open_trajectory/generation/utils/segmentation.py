"""segmentation.py — YOLO-based gripper detection utilities.

Only two public functions are needed in tapir_scale:
- ``get_category_center_by_priority``: finds the centroid of the highest-priority
  detected class in a single frame (used for the occlusion-anchor).
- ``detect_gripper_mask_and_points``: detects the gripper mask in the first frame
  and samples N random points as TAPIR query points.

The frame-by-frame gripper-tip trajectory that existed in scale has been removed;
tracking is now entirely handled by TAPIR (see ``utils/tracking.py``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from ultralytics import YOLO

logger = logging.getLogger(__name__)

# SO-101 class priority order: gripper area only — do NOT fall back to arm/base
GRIPPER_PRIORITY = [
    "so101_gripper",  # 5
    "so101_moving_jaw",  # 7
]


# ── internal helpers ───────────────────────────────────────────────────────────


def _run_yolo(model: YOLO, frame_bgr: np.ndarray, conf: float = 0.25) -> list:
    """Run YOLO on a single BGR uint8 frame and return results."""
    return model(frame_bgr, conf=conf, verbose=False)


def _class_names(model: YOLO) -> dict[int, str]:
    return model.names


def _best_mask_for_classes(
    results,
    class_ids: list[int],
    frame_h: int,
    frame_w: int,
) -> tuple[np.ndarray | None, int | None]:
    """Return the highest-confidence mask among the given class IDs.

    Returns ``(mask_binary_HW, class_id)`` or ``(None, None)`` if no match.
    """
    if results[0].masks is None:
        return None, None

    masks_data = results[0].masks.data  # [N, h, w] float32
    classes = results[0].boxes.cls  # [N]
    confs = results[0].boxes.conf  # [N]

    best_mask = None
    best_conf = -1.0
    best_cls = None

    for cls_id in class_ids:
        for idx in range(len(classes)):
            if int(classes[idx].item()) == cls_id:
                c = float(confs[idx].item())
                if c > best_conf:
                    raw = masks_data[idx].cpu().numpy()
                    resized = cv2.resize(raw, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
                    best_mask = (resized > 0.5).astype(np.uint8)
                    best_conf = c
                    best_cls = cls_id
    return best_mask, best_cls


# ── public API ─────────────────────────────────────────────────────────────────


def get_category_center_by_priority(
    model: YOLO,
    frame_bgr: np.ndarray,
    priority: list[str] = GRIPPER_PRIORITY,
    conf: float = 0.5,
) -> tuple[int, int] | None:
    """Return ``(cx, cy)`` centroid of the highest-priority detected class.

    Used to determine the occlusion-anchor point (pinned frame-0 position).
    Returns ``None`` if nothing is detected.
    """
    results = _run_yolo(model, frame_bgr, conf=conf)
    names = _class_names(model)
    frame_h, frame_w = frame_bgr.shape[:2]

    if results[0].masks is None:
        return None

    # build name→class_id map
    id_by_name = {v: k for k, v in names.items()}

    for cls_name in priority:
        cls_id = id_by_name.get(cls_name)
        if cls_id is None:
            continue
        mask, _ = _best_mask_for_classes(results, [cls_id], frame_h, frame_w)
        if mask is not None:
            ys, xs = np.where(mask > 0)
            if len(xs):
                return int(xs.mean()), int(ys.mean())
    return None


def detect_gripper_mask_and_points(
    model: YOLO,
    frame_bgr: np.ndarray,
    num_points: int = 16,
    priority: list[str] = GRIPPER_PRIORITY,
    conf: float = 0.5,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Detect the gripper in *frame_bgr* and sample TAPIR query points.

    Returns
    -------
    mask : np.ndarray | None
        Binary mask of shape ``[H, W]`` (uint8) for the best gripper detection,
        or ``None`` if nothing was detected.
    query_points : np.ndarray | None
        Float32 array of shape ``[N, 3]`` where each row is ``(t=0, y, x)``
        in *pixel* coordinates of the original frame.  ``None`` if no detection.
    """
    results = _run_yolo(model, frame_bgr, conf=conf)
    names = _class_names(model)
    frame_h, frame_w = frame_bgr.shape[:2]

    if results[0].masks is None:
        logger.warning("No masks found in frame; no query points sampled.")
        return None, None

    id_by_name = {v: k for k, v in names.items()}
    # collect all class IDs in priority order
    priority_ids = [id_by_name[n] for n in priority if n in id_by_name]

    mask, cls_id = _best_mask_for_classes(results, priority_ids, frame_h, frame_w)
    if mask is None:
        logger.warning("No gripper mask found; no query points sampled.")
        return None, None

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        logger.warning("Gripper mask is empty; no query points sampled.")
        return None, None

    if rng is None:
        rng = np.random.default_rng()

    n = min(num_points, len(xs))
    idx = rng.choice(len(xs), size=n, replace=False)
    sampled_y = ys[idx].astype(np.float32)
    sampled_x = xs[idx].astype(np.float32)

    # TAPIR query format: (timestep, y, x)
    query_points = np.stack([np.zeros(n, dtype=np.float32), sampled_y, sampled_x], axis=1)  # [N, 3]

    logger.debug(
        "Sampled %d TAPIR query points from class '%s' mask (%d px).",
        n,
        names.get(cls_id, str(cls_id)),
        len(xs),
    )
    return mask, query_points


def sample_points_from_class(
    model: YOLO,
    frame_bgr: np.ndarray,
    class_name: str,
    n_points: int,
    rng: np.random.Generator,
    conf: float = 0.25,
) -> np.ndarray | None:
    """Sample *n_points* TAPIR query points from the mask of *class_name*.

    Returns ``[n_points, 3]`` float32 rows ``(t=0, y, x)`` in original pixel
    space, or ``None`` if *class_name* is not detected in *frame_bgr*.

    If the mask has fewer pixels than *n_points*, the remainder is filled by
    repeating randomly-selected pixels from the mask (sampling with replacement).
    """
    results = _run_yolo(model, frame_bgr, conf=conf)
    names = _class_names(model)
    frame_h, frame_w = frame_bgr.shape[:2]

    if results[0].masks is None:
        return None

    id_by_name = {v: k for k, v in names.items()}
    cls_id = id_by_name.get(class_name)
    if cls_id is None:
        logger.debug("Class '%s' not in YOLO model vocabulary.", class_name)
        return None

    mask, _ = _best_mask_for_classes(results, [cls_id], frame_h, frame_w)
    if mask is None:
        return None

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    # Sample without replacement up to len(xs), then pad with replacement if needed.
    n_avail = len(xs)
    idx = rng.choice(n_avail, size=min(n_points, n_avail), replace=False)
    sampled_y = ys[idx].astype(np.float32)
    sampled_x = xs[idx].astype(np.float32)

    if n_avail < n_points:
        extra = rng.choice(n_avail, size=n_points - n_avail, replace=True)
        sampled_y = np.concatenate([sampled_y, ys[extra].astype(np.float32)])
        sampled_x = np.concatenate([sampled_x, xs[extra].astype(np.float32)])

    query_points = np.stack([np.zeros(n_points, dtype=np.float32), sampled_y, sampled_x], axis=1)
    logger.debug("Sampled %d points from class '%s'.", n_points, class_name)
    return query_points
