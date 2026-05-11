# Scripts

One-shot scripts for end-to-end pipeline operations. The current state of
the pipeline lives in `notebooks/` (run interactively in Colab); this
folder is for the eventual lifted, runnable command-line versions.

## Planned

| Script | Purpose |
|---|---|
| `01_download_data.py` | Download VIIRS, ERA5, MODIS, OSM (calls the notebook code paths) |
| `02_build_features.py` | Co-locate inputs and engineer features |
| `03_train.py` | Train the four-branch PINN |
| `04_evaluate.py` | Compute test/val metrics and dump predictions |
| `05_export_to_huggingface.py` | Push the trained checkpoint to HF Hub |

Until these are written, run the notebooks in `notebooks/` directly.
