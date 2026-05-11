# Notebooks

Canonical pipeline notebooks for FireSight-IR. One notebook per module in
the data/training/analysis pipeline. Older iterations have been moved to
[`legacy/`](legacy/) for reference.

## Module index

| Module | Notebook | Purpose |
|---|---|---|
| 1a | `01a_viirs_firms_ingest_colab_v3.ipynb` | Download VIIRS active-fire pixels from FIRMS, QA filtering, patch extraction |
| 1b | `01b_download_era5_aod.ipynb` | Co-locate ERA5 atmospheric features with fire pixels |
| 1c | `01c_surface_context_v2.ipynb` | MODIS land cover, OSM infrastructure, solar geometry |
| 1c-viz | `01c_visualization.ipynb` | Visual checks on the surface-context join |
| 2 | `02_feature_engineering_v2.ipynb` | Feature engineering, false-alarm label construction, train/val/test split |
| 3a | `03a_pinn_training_final.ipynb` | Train the four-branch fusion PINN (30 epochs on T4) |
| 3a-ext | `03a_extensions.ipynb` | Supplementary analysis on the trained model |
| 3b | `03b_analysis_ablation.ipynb` | Ablation study: no-physics, no-ERA5, no-surface variants |

## Running on Google Colab

Each notebook expects `/content/drive/MyDrive/firesight-ir/` to be mounted
and to contain `data/`, `models/`, and `figures/` subdirectories. The first
cell of each notebook handles the Drive mount and dependency install.

Suggested run order: 1a -> 1b -> 1c -> 2 -> 3a -> 3b. Modules 1a-2
together build the labelled feature dataset; 3a trains the model; 3b
analyses it.

## Running locally

For local execution, edit the `BASE_DIR` constant near the top of each
notebook to point at your local copy of the project, and skip the
`drive.mount(...)` cell.

## Legacy folder

`legacy/` contains earlier iterations: original 01a/01c/02 versions, the
retired Module 3b false-alarm discriminator (made obsolete by AUC = 1.0 on
the first-stage classifier), and miscellaneous LinkedIn/Untitled scratch
notebooks. They are not part of the canonical pipeline but are preserved
for traceability.

## Local cleanup pending

Two folders at the repo root start with `_DEPRECATED_`: empty
production-skeleton folders that were never populated. They cannot be
deleted from the cleanup script's sandbox. Delete them manually before
pushing:

    Remove-Item -Recurse -Force _DEPRECATED_firesat-ml-production
    Remove-Item -Recurse -Force _DEPRECATED_firesat-ml-production-1
    Remove-Item -Recurse -Force notebooks\_DEPRECATED_models
