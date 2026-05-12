---
license: mit
library_name: pytorch
tags:
  - wildfire-detection
  - remote-sensing
  - physics-informed-neural-network
  - viirs
  - earth-observation
  - infrared
language:
  - en
pipeline_tag: image-classification
---

# FireSight-IR

A physics-informed neural network for wildfire detection and false-alarm
rejection on satellite multispectral infrared data. Trained on VIIRS as a
publicly available proxy for FireSat Protoflight's MWIR/LWIR instrument.

* Code: https://github.com/Ibekwemmanuel7/firesight-IR
* Technical report: in the GitHub repo under `docs/`
* Author: Emmanuel Ibekwe (M.Sc. Atmospheric Science, Texas A&M University)

## Model summary

| | |
|---|---|
| Architecture | Four-branch fusion PINN (CNN + 3 x MLP) |
| Parameters | 202,228 |
| Input | 4 x 32 x 32 IR patches + 16 ERA5 + 20 surface + 6 derived features |
| Output | 3-class softmax (Non-fire, Wildfire, False-alarm) + transmittance scalar |
| Training data | 1,149,722 VIIRS pixels, western CONUS, 2018-2022 |
| Held-out validation | 76,084 VIIRS pixels, 2023 (fully temporal split) |
| Framework | PyTorch 2.x |

## Headline results

On the 2023 held-out validation year:

| Metric | Value |
|---|---:|
| Overall accuracy | 95.84% |
| Wildfire recall | 95.41% |
| Wildfire precision | 99.87% |
| False-alarm recall | 99.93% |
| False-alarm AUC | 1.0000 |
| Non-fire recall | 97.55% |

The full per-class breakdown and confusion matrices are in the technical
report.

## How to use

Clone the GitHub repo for the `firesight_ir` package, then load the
checkpoint from this Hub repo:

```python
from huggingface_hub import hf_hub_download
from firesight_ir import FireSightPINN, predict
import torch, numpy as np

ckpt_path = hf_hub_download(
    repo_id="emmanuelibekwe5525/firesight-ir",
    filename="firesight_pinn_best.pt",
)
model = FireSightPINN.from_checkpoint(ckpt_path)

# Prepare your inputs (see "Input format" below)
preds, probs, transmittance = predict(model, patches, atm, srf, derived)
```

## Input format

The model expects four tensors per pixel, all pre-normalised to match
training:

| Input | Shape | Description |
|---|---|---|
| `patches` | `(N, 4, 32, 32)` float32 | Brightness-temperature patches with channels `[BT_I4, BT_I5, BTD, fire_mask]`. Normalisation: `BT_I4` zero-centred at 300 K and scaled by 50; `BT_I5` zero-centred at 290 K and scaled by 20; `BTD` divided by 40; fire mask divided by 9. |
| `atm` | `(N, 16)` float32 | ERA5 atmospheric features: T2m, PBL height, TCWV, surface pressure, and (T, q) profiles at 1000, 850, 700, 500, 300 hPa. Robust-scaled. |
| `srf` | `(N, 20)` float32 | Surface features: 8 binary land-cover indicators (MODIS MCD12Q1), 3 OSM proximity distances, urban/industrial/burn-scar flags, solar zenith, daytime flag, day-of-year sin/cos. |
| `derived` | `(N, 6)` float32 | Derived physics features: Beer-Lambert TCWV proxy, AOD proxy at 3700 nm, lifted index, atmospheric instability, normalised BT_I4 and BTD anomalies. |

The pre-processing pipeline that produces these inputs from raw VIIRS,
ERA5, MODIS, and OSM downloads lives in the GitHub repo notebooks.

## Output format

`predict(...)` returns a tuple:

| Output | Shape | Description |
|---|---|---|
| `preds` | `(N,)` int64 | Argmax class index: 0=Non-fire, 1=Wildfire, 2=False-alarm |
| `probs` | `(N, 3)` float32 | Softmax probabilities |
| `transmittance` | `(N,)` float32 | Predicted atmospheric transmittance from the physics head |

## Training

* **Optimiser:** AdamW, learning rate 3e-4, cosine schedule over 30 epochs
* **Batch size:** 1024
* **Loss:** Weighted cross-entropy with three physics regularisers
  (Beer-Lambert, dynamic-range, thermal-realism). Lambdas 0.10 / 0.05 / 0.05.
* **Class weights:** Non-fire 6.90, Wildfire 0.36, False-alarm 9.81
* **Hardware:** Google Colab T4 GPU, mixed-precision (FP16)
* **Best checkpoint:** Epoch 23, validation loss 0.1149
* **Total training time:** ~13 GPU-hours

## Ablation findings

An ablation study removed one branch or loss term per variant. Results on
the 2023 validation set:

| Variant | Val acc | WF recall | FA precision | FA AUC |
|---|---:|---:|---:|---:|
| Full model | 95.84% | 95.41% | 97.83% | 1.0000 |
| No physics loss | 95.41% | 94.91% | 97.43% | 0.99999 |
| No ERA5 (atm branch) | 96.26% | 95.88% | 98.30% | 0.99999 |
| **No surface (srf branch)** | **80.70%** | **78.55%** | **35.33%** | **0.9737** |

The dominant input is the surface branch (OSM proximity plus MODIS land
cover). The physics-informed loss contributes a fraction of a percentage
point and behaves as a regulariser rather than a primary signal carrier.
ERA5 atmospheric features contribute slightly negatively on this dataset
and region.

## Intended use

Research and educational use on satellite IR wildfire detection. Suitable
as a baseline for:

* Wildfire / false-alarm pixel classification on VIIRS-like multispectral
  inputs.
* Studying physics-informed loss design for remote-sensing problems.
* Benchmarking transfer-learning approaches to other satellite IR
  instruments.

## Out-of-scope use

Operational fire alerting that drives emergency response. The model has
not been validated against on-the-ground fire perimeters, detection
latency has not been measured, and it has not been integrated with any
operational system. Do not use as a sole detection source.

## Limitations

* **Geographic scope.** Trained on western CONUS only, fire season
  (May to October). Tropical, boreal, and year-round fire regimes are
  out of distribution.
* **VIIRS-to-FireSat transfer.** VIIRS has 375 m spatial resolution and
  different spectral response than FireSat's custom MWIR/LWIR instrument.
  Validation on actual FireSat data is required before any operational use.
* **False-alarm labelling and physics loss are partially circular.** The
  false-alarm class was constructed via a BTD threshold (BTD < 20 K)
  combined with OSM industrial proximity. The thermal-realism loss
  penalises wildfire predictions on pixels with BTD < 10 K. Both depend
  on BTD, so the model's strong FA discrimination partly recovers the
  labelling rule rather than discovering it from data.
* **Missing data sources.** VIIRS Day/Night Band and MTBS historical
  burn-scar data were specified in the project plan but failed to download
  during the data-build phase. They are zero-filled in the feature
  catalogue and contribute nothing to the model.
* **No baseline comparison.** The trained PINN has not been compared
  against a simple threshold rule (BTD > 20 K with not-is-industrial)
  or against VIIRS Collection 2 contextual algorithm. Reviewers should
  expect a substantial fraction of the discriminative performance to come
  from feature engineering rather than the learned model.

## Bias considerations

Class weights penalise missing wildfire pixels less heavily than missing
false-alarm pixels. This produces a conservative wildfire predictor that
sometimes labels a low-confidence wildfire detection as non-fire (~4.6%
wildfire miss rate at threshold 0.5). For a first-responder alerting
system this is the operationally acceptable error direction, but it does
mean some real fires get suppressed.

## Citation

```bibtex
@misc{ibekwe2026firesight,
  title  = {FireSight-IR: A physics-informed neural network pipeline
            for wildfire detection and false-alarm rejection},
  author = {Ibekwe, Emmanuel},
  year   = {2026},
  url    = {https://github.com/Ibekwemmanuel7/firesight-IR},
}
```

## Acknowledgments

VIIRS active fire data from NASA FIRMS. Atmospheric reanalysis from
ECMWF ERA5 (Copernicus CDS). Land cover from MODIS MCD12Q1 v6.1 via
NASA earthaccess. Infrastructure features from OpenStreetMap via the
Overpass API. The work was motivated by the FireSat Protoflight first-light
imagery release (Earth Fire Alliance, Muon Space, Google Research, July 2025).
