# Diffusion Transformer Implementation Notes

This note summarizes the local DiT training implementation used by the sequence-conditioned diffusion policy.

## Files

- `training/diffusion_transformer.py`
  - Defines `DiffusionTransformer1d`, a DiT-style action denoiser.
  - Defines `DenoiserConfig`.
  - Defines `make_action_denoiser(...)`, which can return either LeRobot's stock U-Net or the local DiT.

- `training/train_sequence_diffusion.py`
  - Mirrors `train_diffusion_tracks_tiny_rt_active.py`.
  - Keeps the same dataset loading, trajectory preprocessing, trajectory encoder, vision/depth encoders, conditioning fusion, optimizer, scheduler, logging, and checkpointing structure.
  - Replaces direct `DiffusionConditionalUnet1d(...)` construction with `make_action_denoiser(...)`.
  - Defaults to `DENOISER_BACKBONE = "dit"`.

No LeRobot source files were changed for this DiT experiment.

## Architecture

The existing conditioning pipeline is unchanged:

```text
dataset batch
  -> active-window global trajectory preprocessing/augmentation
  -> trajectory encoder
  -> optional ResNet/DINO/depth encoders
  -> optional ConditioningFusion
  -> global_cond vector
```

LeRobot's diffusion training loop is also unchanged:

```text
clean action chunk
  -> scheduler.add_noise(...)
  -> denoiser(noisy_actions, timestep, global_cond)
  -> MSE loss against epsilon or sample target
```

Only the denoising backbone changes:

```text
old: DiffusionConditionalUnet1d
new: DiffusionTransformer1d
```

`DiffusionTransformer1d` has the same call signature and tensor contract as the U-Net:

```python
x:           [B, horizon, action_dim]
timestep:    [B]
global_cond: [B, cond_dim]
output:      [B, horizon, action_dim]
```

Internally:

```text
noisy actions [B, H, A]
  -> Linear(A -> DIT_DIM)
  -> learned position embedding over H action tokens
  -> AdaLN-Zero transformer blocks
  -> AdaLN final projection
  -> predicted noise/action [B, H, A]
```

The timestep and conditioning vector are embedded separately and added:

```text
timestep -> sinusoidal embedding -> MLP -> [B, DIT_DIM]
global_cond -> LayerNorm -> MLP -> [B, DIT_DIM]
cond = timestep_emb + global_cond_emb
```

Each DiT block uses this `cond` vector for AdaLN shift/scale/gating around self-attention and MLP residuals. Attention is over action-horizon tokens only; modality/trajectory conditioning remains global, matching the previous U-Net FiLM strategy.

## Training Configuration

The DiT script adds these globals:

```python
DENOISER_BACKBONE = "dit"
DIT_DIM = 256
DIT_DEPTH = 6
DIT_NHEAD = 8
DIT_FFN_DIM = 1024
DIT_DROPOUT = 0.0
```

CLI overrides:

```bash
python training/train_sequence_diffusion.py \
  --dit-dim 256 \
  --dit-depth 6 \
  --dit-nhead 8 \
  --dit-ffn-dim 1024 \
  --dit-dropout 0.0
```

The capacity presets also set default DiT sizes:

- `tinier`: `dim=128`, `depth=4`, `heads=4`, `ffn=512`
- `tiny`: `dim=256`, `depth=6`, `heads=8`, `ffn=1024`
- `small`: `dim=384`, `depth=8`, `heads=8`, `ffn=1536`
- `medium`: `dim=512`, `depth=10`, `heads=8`, `ffn=2048`
- `large`: `dim=768`, `depth=12`, `heads=12`, `ffn=3072`

DiT CLI overrides are applied after capacity presets, so command-line DiT settings win.

## Checkpoint Metadata

The DiT script saves denoiser metadata to both `hparams.json` and `encoder_config.json`:

```json
{
  "DENOISER_BACKBONE": "dit",
  "DIT_DIM": 256,
  "DIT_DEPTH": 6,
  "DIT_NHEAD": 8,
  "DIT_FFN_DIM": 1024,
  "DIT_DROPOUT": 0.0
}
```

Resume logic reads these values from `encoder_config.json` before constructing the policy, so resumed training rebuilds the same model shape before loading weights.

## Validation Done

Syntax validation passed:

```bash
python -m py_compile \
  training/diffusion_transformer.py \
  training/train_sequence_diffusion.py
```

A runtime tensor forward check was not run in the current shell because `torch` was not installed in that Python environment.

## Future Inference Or Reference Script Considerations

A DiT checkpoint cannot be loaded through plain LeRobot `DiffusionPolicy.from_pretrained(...)` unless the loader constructs the DiT backbone before loading the state dict. Plain LeRobot will build the U-Net and state-dict loading will fail.

For a future inference script, mirror the existing custom-policy pattern from `run_inference_so100_diffusion_tracks_tiny_rt_active.py`:

1. Read `encoder_config.json` from the checkpoint.
2. Extract:
   - `DENOISER_BACKBONE`
   - `DIT_DIM`
   - `DIT_DEPTH`
   - `DIT_NHEAD`
   - `DIT_FFN_DIM`
   - `DIT_DROPOUT`
3. Build a `DenoiserConfig`.
4. In every custom `DiffusionModel` subclass, replace direct `DiffusionConditionalUnet1d(...)` construction with:

   ```python
   self.unet = make_action_denoiser(
       config,
       global_cond_dim=cond_dim,
       denoiser_cfg=denoiser_cfg,
   )
   ```

5. Ensure `diffusion_transformer.py` is importable on the inference machine.

The inference data path should stay unchanged:

```text
camera/state/global trajectory
  -> trajectory encoder and visual/depth encoders
  -> observation.environment_state / observation.shared_trajectory
  -> ConditioningFusion or shared-trajectory conditioning
  -> global_cond
  -> scheduler calls DiT denoiser repeatedly
```

For a reference script or lightweight smoke test, create a small script that:

1. Imports `DiffusionTransformer1d`.
2. Builds it with a tiny config.
3. Runs:

   ```python
   y = model(
       torch.randn(B, horizon, action_dim),
       torch.randint(0, num_train_timesteps, (B,)),
       torch.randn(B, global_cond_dim),
   )
   ```

4. Asserts:

   ```python
   y.shape == (B, horizon, action_dim)
   torch.isfinite(y).all()
   ```

For a full checkpoint reference loader, the script should additionally instantiate the same custom policy wrapper used in training, load a checkpoint, run one prepared batch through `policy.forward(...)`, and optionally run one short `generate_actions(...)` call with fixed noise.

## Deployment Notes

On another training machine, the minimum requirements are:

- the modified LeRobot repo installed/importable, because the broader training and robot scripts depend on local LeRobot changes;
- `training/diffusion_transformer.py`;
- `training/train_sequence_diffusion.py`;
- the checkpoint's `encoder_config.json` for resume or future inference reconstruction.

Recommended setup:

```bash
git clone <modified-lerobot-repo>
cd lerobot
pip install -e .
```

Then run the sequence diffusion trainer from the repository root.
