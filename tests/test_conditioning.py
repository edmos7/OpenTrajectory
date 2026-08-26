from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from lerobot_policy_open_trajectory.conditioning import TrajectoryCondition
from lerobot_policy_open_trajectory.modeling_components import RTTrajectoryImageEncoder, TrajectoryEncoder


def test_marker_pair_is_required() -> None:
    with pytest.raises(ValueError, match="both absent or both present"):
        TrajectoryCondition(
            np.asarray([[0.1, 0.1], [0.9, 0.9]], dtype=np.float32),
            image_shape=(100, 100),
            grasp_index=0,
        )


def test_sequence_marks_only_grasp_release_interval() -> None:
    condition = TrajectoryCondition(
        np.asarray([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float32),
        image_shape=(100, 100),
        grasp_index=1,
        release_index=2,
    )

    sequence = condition.to_sequence(length=9)

    np.testing.assert_array_equal(sequence[:4, 2], np.zeros(4, dtype=np.float32))
    np.testing.assert_array_equal(sequence[4:, 2], np.ones(5, dtype=np.float32))


def test_rt_image_stops_at_release() -> None:
    condition = TrajectoryCondition(
        np.asarray([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]], dtype=np.float32),
        image_shape=(100, 100),
        grasp_index=0,
        release_index=1,
    )

    image = condition.to_rt_image(marker_radius=4, marker_width=2)

    assert image[45:56, 45:56, 2].max() > 0
    assert image[90, 90].sum() == 0


def test_rt_image_defaults_match_reference_marker_width() -> None:
    condition = TrajectoryCondition(
        np.asarray([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]], dtype=np.float32),
        image_shape=(100, 100),
        grasp_index=0,
        release_index=1,
    )

    np.testing.assert_array_equal(condition.to_rt_image(), condition.to_rt_image(marker_width=8))


def test_condition_encoder_parameter_counts_match_research_contract() -> None:
    trajectory = sum(parameter.numel() for parameter in TrajectoryEncoder().parameters())
    rt_image = sum(parameter.numel() for parameter in RTTrajectoryImageEncoder().parameters())

    assert trajectory == 189_088
    assert rt_image == 184_904


def test_sequence_encoder_uses_active_interval() -> None:
    encoder = TrajectoryEncoder().eval()
    inactive = torch.rand(1, 2, 224, 3)
    inactive[..., 2] = 0
    active = inactive.clone()
    active[..., 2] = 1

    with torch.no_grad():
        inactive_output = encoder(inactive)
        active_output = encoder(active)

    assert not torch.equal(inactive_output, active_output)


def test_sequence_overlay_uses_original_timesteps(monkeypatch) -> None:
    condition = TrajectoryCondition(
        np.asarray([[0.1, 0.1], [0.2, 0.9], [0.9, 0.9]], dtype=np.float32),
        image_shape=(100, 100),
        grasp_index=1,
        release_index=2,
    )
    monkeypatch.setattr(
        TrajectoryCondition,
        "_resample",
        lambda *_args, **_kwargs: pytest.fail("sequence overlay must not resample the path"),
    )

    overlay = condition.overlay(np.zeros((100, 100, 3), dtype=np.uint8), "sequence")

    assert overlay.shape == (100, 100, 3)


def test_sequence_overlay_matches_rt_tracking_reference_geometry() -> None:
    condition = TrajectoryCondition(
        np.asarray([[0.106, 0.104], [0.504, 0.506], [0.904, 0.902]], dtype=np.float32),
        image_shape=(100, 100),
        grasp_index=1,
        release_index=2,
    )
    expected = Image.new("RGB", (100, 100), (0, 0, 0))
    draw = ImageDraw.Draw(expected)
    draw.line([(11, 10), (50, 51)], fill=(0, 0, 0), width=6)
    draw.line([(50, 51), (90, 90)], fill=(127, 0, 255), width=6)

    overlay = condition.overlay(np.zeros((100, 100, 3), dtype=np.uint8), "sequence", alpha=1.0)

    np.testing.assert_array_equal(overlay, np.asarray(expected))
