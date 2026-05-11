# FireSight-IR

A physics-informed neural network pipeline for wildfire detection and
false-alarm rejection on satellite multispectral infrared data. Built as a
software response to the false-alarm gap exposed in FireSat Protoflight
first-light imagery (July 2025), trained on VIIRS as the closest publicly
available proxy for FireSat's MWIR/LWIR instrument.

## Headline results

Trained on 1,149,722 VIIRS fire pixels across western CONUS for fire seasons
2018–2022, with 2023 held out as a strict temporal validation year.

| Metric | Validation (2023) | Test (2018–2022) |
|---|---:|---:|
| Wildfire recall | 95.4% | 95.9% |
| Wildfire precision | 99.87% | 99.98% |
| False-alarm recall | 99.9% | 99.9% |
| False-alarm AUC | 1.0000 | 1.0000 |
| Overall accuracy | 95.84% | 96.16% |

The full technical report is in [`docs/`](docs/) (PDF). The ablation study
finds that surface context features (OSM industrial proximity plus MODIS
land cover) carry most of the discriminative signal; the physics-informed
loss terms behave as soft regularizers rather than primary signal carriers.

## Repository layout

```
firesight-ir/
├── firesight_ir/      Python package: model, losses, dataset, inference
├── scripts/           One-shot scripts (download, train, evaluate)
├── notebooks/         Canonical pipeline notebooks (one per module)
│   └── legacy/        Earlier iterations, kept for reference
├── figures/
│   ├── scripts/       Figure-generation scripts (fig01-fig62)
│   └── publication/   Rendered PDF/PNG outputs
├── dashboard/         Streamlit operator-facing demo (app.py)
├── docs/              Technical report and supplementary docs
├── data/              Local data cache (gitignored, not committed)
└── models/            Trained checkpoints (large files gitignored)
```

## Quick start

### Inference on the trained model

The trained checkpoint is `models/firesight_pinn_best.pt` (epoch 23,
val_loss = 0.1149). It expects four inputs:

* `patches`: 4 × 32 × 32 brightness-temperature patches
  (BT_I4, BT_I5, BTD, fire mask)
* `atm`: 16 ERA5 atmospheric features
* `srf`: 20 surface features (MODIS land cover, OSM proximity, solar geometry)
* `derived`: 6 derived physics features

Output is a 3-class softmax over {Non-fire, Wildfire, False-alarm} plus a
sigmoid scalar for predicted atmospheric transmittance.

```python
# Pseudocode; see firesight_ir/inference.py for the working version once
# the package modules have been lifted from the notebooks.
from firesight_ir import FireSightPINN, predict
model = FireSightPINN.from_checkpoint('models/firesight_pinn_best.pt')
preds = predict(model, patches, atm, srf, derived)
```

### Reproducing the training pipeline

The notebooks in `notebooks/` run end-to-end on Google Colab with a T4 GPU.
Suggested order:

1. `01a_viirs_firms_ingest_colab_v3.ipynb`: VIIRS FIRMS ingest
2. `01b_download_era5_aod.ipynb`: ERA5 atmospheric co-location
3. `01c_surface_context_v2.ipynb`: MODIS land cover and OSM infrastructure
4. `02_feature_engineering_v2.ipynb`: Feature engineering and label construction
5. `03a_pinn_training_final.ipynb`: Train the four-branch PINN
6. `03b_analysis_ablation.ipynb`: Ablation study and analysis

Each notebook expects the project root to be mounted on Drive at
`/content/drive/MyDrive/firesight-ir/`.

### Regenerating the publication figures

```bash
pip install -r requirements.txt
python figures/scripts/fig00_intro_overview.py
python figures/scripts/fig01_study_area.py
# ... etc.
```

Each script reads from `/content/drive/MyDrive/firesight-ir/` by default;
edit the `BASE_DIR` constant at the top of each script for local execution.

## Dashboard

A Streamlit operator-facing demo lives in [`dashboard/app.py`](dashboard/app.py).
Local run:

```bash
pip install streamlit
streamlit run dashboard/app.py
```

A hosted version may be deployed on Streamlit Community Cloud; if so, the
URL will appear here.

## Data sources

| Source | Use | Access |
|---|---|---|
| NASA FIRMS (VIIRS Collection 2) | Fire pixels and FRP | FIRMS API |
| ERA5 reanalysis | Atmospheric state (T2m, TCWV, profiles) | Copernicus CDS |
| MODIS MCD12Q1 v6.1 | Land cover, 500 m | NASA earthaccess |
| OpenStreetMap | Industrial / urban / power-plant proximity | Overpass API |

All data used in this project is publicly available with free registration.
See `data/README.md` for download recipes.

## Citation

If you use this work, please cite the technical report:

```
Ibekwe, E. (2026). FireSight-IR: A physics-informed neural network pipeline
for wildfire detection and false-alarm rejection. Technical Report.
```

## Author

Emmanuel Ibekwe
M.Sc. Atmospheric Science, Texas A&M University
[github.com/Ibekwemmanuel7](https://github.com/Ibekwemmanuel7)

## License

MIT. See [LICENSE](LICENSE).
