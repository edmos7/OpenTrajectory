"""Frame preprocessing utilities: tensor → OpenCV uint8 and occlusion masking."""

import cv2
import numpy as np
import torch


def preprocess_frame(frame_tensor: torch.Tensor, target_size=None) -> np.ndarray:
    """
    Convert a dataset frame tensor to an OpenCV-compatible uint8 BGR image.

    Steps:
        1. Permute CHW → HWC
        2. Convert to numpy (float32, [0, 1])
        3. Ensure contiguous memory
        4. RGB → BGR
        5. Scale to [0, 255] and cast to uint8
        6. Optional resize

    Args:
        frame_tensor: CHW float32 tensor with values in [0, 1].
        target_size:  Optional (width, height) for resizing.

    Returns:
        HWC uint8 BGR numpy array.
    """
    frame_np = np.ascontiguousarray(frame_tensor.permute(1, 2, 0).numpy())
    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    frame_uint8 = (frame_bgr * 255.0).astype(np.uint8)
    if target_size is not None:
        frame_uint8 = cv2.resize(frame_uint8, target_size)
    return frame_uint8


def occlude_except_circle(
    frame_tensor: torch.Tensor,
    center: tuple,
    radius: int,
) -> torch.Tensor:
    """
    Zero out all pixels outside a circle centred at *center* with the given *radius*.

    Args:
        frame_tensor: CHW float32 tensor with values in [0, 1].
        center:       (cx, cy) pixel coordinates.
        radius:       Circle radius in pixels.

    Returns:
        Occluded frame as a CHW float32 tensor with values in [0, 1].
    """
    frame_uint8 = preprocess_frame(frame_tensor)
    mask = np.zeros_like(frame_uint8, dtype=np.uint8)
    cv2.circle(mask, (int(center[0]), int(center[1])), radius, (255, 255, 255), thickness=-1)
    occluded = cv2.bitwise_and(frame_uint8, mask)
    occluded_rgb = cv2.cvtColor(occluded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(occluded_rgb).permute(2, 0, 1)
