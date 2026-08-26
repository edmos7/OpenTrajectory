from __future__ import annotations

import numpy as np

from lerobot_policy_open_trajectory.visuals import (
    render_gripper_tip_frame,
    render_point_tracks_frame,
)


def test_gripper_tip_frame_advances_the_trail() -> None:
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    path = np.asarray([[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]], dtype=np.float32)

    early = render_gripper_tip_frame(frame, path, 0)
    late = render_gripper_tip_frame(frame, path, 2)

    assert early.shape == frame.shape
    assert np.count_nonzero(late) > np.count_nonzero(early)


def test_point_track_frame_renders_every_forecast_without_visibility_filtering() -> None:
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    tracks = np.asarray(
        [
            [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]],
            [[0.8, 0.2], [0.7, 0.3], [0.6, 0.4]],
        ],
        dtype=np.float32,
    )

    rendered = render_point_tracks_frame(frame, tracks, persistent_point_count=1)

    assert rendered.shape == frame.shape
    for x_norm, y_norm in tracks.reshape(-1, 2):
        x = min(round(float(x_norm) * frame.shape[1]), frame.shape[1] - 1)
        y = min(round(float(y_norm) * frame.shape[0]), frame.shape[0] - 1)
        assert rendered[y, x].max() > 0
