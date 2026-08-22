# Model Artifacts

Large weights are not tracked by git.

Default data-generation paths expect:

```text
models/segmentso101.pt
models/tapnextpp_ckpt.pt
```

You can also pass explicit paths:

```bash
--yolo-path /path/to/segmentso101.pt
--tapnext-checkpoint /path/to/tapnextpp_ckpt.pt
```

Training checkpoints should be written under `outputs/` or another ignored directory.
