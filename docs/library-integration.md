# Library interfaces

Use Python 3.12 and the LeRobot API required by `pyproject.toml`. Install extras
with `pip install '.[data,predictors]'` when building from this repository.
LeRobot `>=0.6.1,<0.7` and its dataset dependencies are installed by this package.

## Tracking dependency

Tracking additionally requires the official [TAPNext source package](https://github.com/google-deepmind/tapnet).
Install this pinned revision, which provides the `tapnet.tapnext` backend:

```bash
python -m pip install 'tapnet[torch] @ git+https://github.com/google-deepmind/tapnet.git@730cda1c730877cfedbe01bf87fb1cadb78a565d'
```

The `data` extra installs SciPy and Ultralytics. It does not install TAPNext:
the required upstream package is unavailable from the package index. Supply
segmentation and TAPNext checkpoints separately; no model weights are bundled.

## Data generation

The installed CLI forwards backend options after `--`:

```bash
open-trajectory dataset generate --kind tracks -- --help
open-trajectory dataset generate --kind tracks-and-sequence -- --help
open-trajectory dataset generate --kind sequence -- --help
open-trajectory dataset generate --kind rt-image -- --help
open-trajectory dataset trim --mode release -- --help
open-trajectory dataset trim --mode idle -- --help
open-trajectory dataset add-depth -- --help
```

Supply segmentation and TAPNext checkpoints explicitly. Track generation emits
robot and environment point tracks; `tracks-and-sequence` additionally generates
the gripper-tip task sequence. Sequence/track generators accept `--artifact-dir`
to export portable per-episode NPZ annotations alongside the augmented dataset.
Track-only artifacts do not need a gripper trajectory. Backend intermediates used
for `--resume` are distinct from these portable artifacts.

Depth uses an explicit `--depth-anything-dir` and optional `--checkpoint`; absent
a checkpoint it downloads the named Small model. Its output is colorized relative
depth, not metric depth. Generation stays local by default. `--push` is an explicit
upload request. Supported output replacement retains a sibling backup directory.

`dataset augment` consumes portable artifacts and uses LeRobot's feature editor.
Existing feature collisions are errors. Available validity masks are written as
`observation.point_tracks_validity.<camera>`; absent masks mean unknown validity,
not verified visibility. Historical artifacts without provenance remain readable
but are labeled `unknown_legacy` in metadata.

## Spatial metadata

`meta/open_trajectory.json` records camera identity, original height/width,
XY coordinate convention, normalization and its scope, coordinate frame,
frame/horizon stride, FPS, padding, and available generator/checkpoint provenance.
Tracks use `observation.point_tracks.<camera>`; legacy role-based names are recorded
as aliases, not used to rename source cameras. Episode selection records original
episode IDs; trimming retains provenance and records source frame bounds.

Image-plane XY divided by width/height is not robot-base XYZ. Converting to metric
coordinates requires calibrated geometry and suitable depth. The schema requires
an explicit calibration reference for metric frames; it does not perform that
conversion or certify a robot safety envelope.

Track horizons include the query position followed by future positions. Strided
tracking repeats annotations between tracker frames; `anchor_stride` and
`anchor_rule` identify the query frame. The predictor training command rejects
unaligned strided anchors. Portable artifacts preserve anchor stride independently
from stored-row spacing. Regenerate older generator-exported artifacts made with
a tracker stride greater than one; they did not preserve anchor stride.
Resample episodes to tracker FPS first. Terminal
padding contributes to training unless excluded by explicit validity masks.
Trimming does not recompute original track horizons.

## ACT

Install `lerobot_policy_open_trajectory[training]` (or `.[training]` from source)
to include LeRobot's Accelerate and training dependencies.

Use native `lerobot-train` with `--policy.type=open_trajectory_act` and
`--policy.conditioning_mode=sequence` or `rt_image`. Supply normal dataset,
output, and training options. Sequence input is `[trajectory_length, 3]` normalized
XY plus active interval. Image input is `[3, H, W]` under
`observation.images.rt_trajectory`. Override feature names in the policy config
when needed. Environment images or environment state are also required.

The policy retains ACT's variational objective, action chunks, temporal ensembling,
native processors, and checkpoint handling. Conditioning inputs bypass state
normalization. Pretrained visual backbone weights may download unless their config
fields are set to `None`. At rollout, supply the chosen condition in each batch;
the diffusion policy's interactive drawing cache is not part of ACT's interface.

## Learned track prediction

`TrackPredictor` ports the research Track Transformer: one/two camera backbones,
task-sequence encoding, point-sampled visual tokens, and future-offset decoding.
It accepts current RGB `[B,3,H,W]` in `[0,1]`, current query XY `[B,N,2]`, and a
task sequence `[B,L,3]`. Future frames/tracks are supervision, never model inputs.
It does not choose online query points; callers supply them from current images.

```python
from lerobot_policy_open_trajectory.predictors import TrackPredictor

predictor = TrackPredictor.from_pretrained(checkpoint_directory, device="cpu")
prediction = predictor.predict(images, current_query_points, task_sequence, fps=30)
features = prediction.policy_features({
    "observation.images.side": "observation.point_tracks.side",
})
observation.update(features)
```

`prediction.tracks` includes current + future normalized XY, with per-camera
metadata. Predictions can leave the image; no clamping or metric conversion is
performed. Existing `render_point_tracks_frame` can visualize a prediction after
moving it to CPU/NumPy. Save/load uses JSON configuration and safetensors.

Train a predictor separately from action policies:

```bash
open-trajectory predictor train -- \
  --repo-id user/annotated-data --root /path/to/data \
  --camera-tracks '{"observation.images.side":"observation.point_tracks.side"}' \
  --output-dir outputs/track-predictor
```

The command uses episode-disjoint training/validation splits, masked MSE, records
split/provenance metadata, and refuses existing output directories. Use
`--validity-keys` to map cameras to validity features. It requires known spatial
provenance; migrate legacy annotations explicitly before training. Checkpoints
are inference checkpoints, not exact optimizer/RNG-resume checkpoints.

For diffusion policy consumption, configure `track_feature_keys` and matching
`[N,current+future,2]` input shapes. Offline batches also include observation
history. Rollout callers map predictor outputs into those same feature keys.
Training a policy on hindsight tracks creates a distribution shift relative to
predicted rollout tracks; evaluate that shift explicitly, and never report
hindsight-track evaluation as online prediction performance.
