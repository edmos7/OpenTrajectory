"""LeRobot pre/post processors for Open Trajectory."""

from __future__ import annotations

from typing import Any

import torch
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.utils.constants import OBS_STATE

from .configuration_open_trajectory import OpenTrajectoryConfig


def make_open_trajectory_pre_post_processors(
    config: OpenTrajectoryConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Normalize robot state/actions while leaving cameras and trajectory conditions untouched."""
    steps = make_default_policy_processor_steps(config, dataset_stats, normalizer_device=config.device)
    steps.normalize.normalize_observation_keys = {OBS_STATE} if config.use_joint_state else set()
    return make_policy_processor_pipelines(
        input_steps=[
            steps.rename_observations,
            steps.add_batch_dim,
            steps.to_device,
            steps.normalize,
        ],
        output_steps=[steps.unnormalize, steps.to_cpu],
    )
