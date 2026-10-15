# Changelog

## 0.2.0

- Package generation, depth, and release/idle trimming backends for use through the installed CLI.
- Add camera-keyed point tracks, spatial provenance, track-only artifacts, and optional validity masks.
- Add native LeRobot ACT policies with sequence or rendered-image conditioning.
- Add one/two-camera future-track predictors, portable checkpoints, and episode-disjoint training.
- Allow diffusion policies to consume explicitly configured point-track features.
- Preserve query-anchor stride through artifact export and augmentation; reject unaligned predictor training.
- Support published LeRobot 0.6.1 and install its dataset dependencies.
- Add a `training` extra for LeRobot's native training dependencies.

Tracking requires a separate upstream TAPNext source install and explicit model
checkpoints. Regenerate older generator-exported artifacts made with tracker stride
greater than one: they lack query-anchor metadata. Research-script checkpoints
remain distinct from native plugin checkpoints. Real-weight generation, training
quality, and robot rollout have not been validated for this release.
