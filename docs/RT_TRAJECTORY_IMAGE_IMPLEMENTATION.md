# RT Trajectory Image Implementation

This document describes the RT trajectory-image experiment path.

## Goal

Create an ablation/comparison against the current diffusion-tracks architecture where the main difference is the trajectory conditioning format:

- current tracks path: trajectory sequence/time-series conditioning
- RT-image path: one rendered trajectory image per episode, broadcast as a LeRobot image feature

TAPNext point tracks are used only to generate the rendered image. They are not added to the final dataset and are not consumed by the policy.

## Files

- `rt_trajectory_image_demo.ipynb`
  Notebook demo of the full image-generation process for a single episode. It loads a LeRobot episode, seeds gripper-tip points with YOLO, tracks them with TAPNext++, infers grasp/release events from gripper-action peaks, renders the blank RT trajectory image, and shows debug overlays.

- `rt_trajectory_utils.py`
  Shared utilities for the notebook and scripts:
  camera/key handling, RGB conversion, gripper action normalization, peak-based grasp/release detection, YOLO mask selection, tip-point sampling, RT image rendering, and debug overlay rendering.

- `generate_rt_trajectory_image_dataset.py`
  Production dataset-generation script. It generates/caches one PNG per episode, optionally writes debug overlays, then creates a new LeRobot dataset with only one added feature:
  `observation.images.rt_trajectory`.

- `train_rt_trajectory_image_diffusion.py`
  Standalone Diffusion Policy trainer for datasets containing `observation.images.rt_trajectory`. It keeps DINO-style visual conditioning for robot camera inputs, while the rendered RT image uses its own small CNN encoder.

## RT Image Semantics

Each episode gets one RGB image:

- black background
- red trajectory path, with red intensity encoding time/progress
- green donut marker at grasp
- blue donut marker at release
- trajectory is truncated at release, excluding the return-to-start motion

Grasp/release detection uses starts of separate gripper-action peaks. This is important because both grasp and release produce similar action spikes; release is not the falling edge of the grasp peak.

## Dataset Output

The generated dataset preserves the source dataset and adds:

```text
observation.images.rt_trajectory
```

It does not add:

```text
observation.global_trajectory
observation.main_cam_point_tracks
observation.sec_cam_point_tracks
```

## Training Comparison

The RT-image trainer uses the same broad DP capacity profile as the current small diffusion-tracks setup:

- `horizon=32`
- `n_action_steps=16`
- `n_obs_steps=2`
- `down_dims=(128, 256, 512)`
- DINOv2 visual encoder with `out_dim=32` for each default RGB camera input: `observation.images.side` and `observation.images.wrist`
- RT trajectory image CNN encoder with `out_dim=32`, similar to the depth-image encoder path used by the sequence-conditioned trainers.
- fusion output dim `64`

The key intended difference is trajectory conditioning:

- current path: trajectory sequence encoded as conditioning tokens
- RT-image path: rendered RT trajectory image encoded through its own CNN and fused with state plus DINO camera streams

## Example Commands

Generate the derived dataset:

```bash
python \
  data_generation/generate_trajectory_image_dataset.py \
  --source-repo-id edmos7/so100_multitask_quality_checked_tip_v1-dav2small-depth \
  --output-repo-id edmos7/so100_multitask_quality_checked_tip_v1-dav2small-depth-rt-trajectory-images \
  --resume
```

Smoke-test training:

```bash
python \
  training/train_trajectory_image_diffusion.py \
  --dataset-repo-id edmos7/so100_multitask_quality_checked_tip_v1-dav2small-depth-rt-trajectory-images \
  --dry-run
```
