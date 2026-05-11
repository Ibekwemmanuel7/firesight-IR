# Models

Trained checkpoints for FireSight-IR.

## Files

| File | Description |
|---|---|
| `firesight_pinn_best.pt` | Best-validation checkpoint (epoch 23, val_loss = 0.1149, val accuracy = 95.84%) |
| `firesight_pinn_final.pt` | Last-epoch checkpoint (epoch 30) |
| `training_log.json` | Per-epoch metrics over the full training run |
| `ablations/` | Checkpoints, logs, and metrics for the three ablation variants (no-physics, no-ERA5, no-surface) |

The `.pt` files are PyTorch state dictionaries (loadable with
`torch.load(...)`). They are gitignored from the repo because of their
size; distribute them via Hugging Face Hub or a release artifact.

## Loading

```python
import torch
ckpt = torch.load('models/firesight_pinn_best.pt', weights_only=False)
print(ckpt.keys())          # ['model_state_dict', 'epoch', 'val_loss', ...]
print(ckpt['epoch'])        # 23
print(ckpt['val_loss'])     # 0.1149
```

To load into a fresh model instance, use `firesight_ir.FireSightPINN.from_checkpoint(...)`
(once that helper has been lifted out of the training notebook).

## Reproducibility

Training was done on Google Colab free tier with an NVIDIA T4 16 GB.
Approximate wall-clock time per epoch: 26 minutes. Total training time
was approximately 13 hours, spread across multiple Colab sessions with
auto-resume.
