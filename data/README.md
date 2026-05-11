# Data

Raw and processed datasets used by FireSight-IR. The contents of this
folder are gitignored because they are large (tens of GB across all
modules) and are reproducible from public sources.

## Layout

```
data/
├── raw/                       Raw downloads, untouched
│   ├── firms/                 VIIRS active-fire pixels (parquet, by year)
│   ├── era5/                  ERA5 monthly NetCDF (surface variables)
│   └── surface/               MODIS LC GeoTIFF, OSM JSON, MTBS burn-scar JSON
├── processed/                 Co-located, feature-engineered data
│   ├── viirs_fp_v2/           VIIRS pixels with QA fields
│   ├── viirs_fp_atm_v2/       + ERA5 atmospheric features
│   ├── viirs_fp_srf_v2/       + surface features (final labelled dataset)
│   └── patches/               HDF5 archive of 32x32 IR patches
├── cache/                     Numpy memmaps for fast training on Colab
│   ├── patches.npy
│   ├── atm.npy
│   ├── srf.npy
│   ├── derived.npy
│   ├── labels.npy
│   └── aux.npy
├── splits/                    Train/val/test indices
├── scalers/                   Feature scalers (sklearn) and class weights
└── predictions/               Cached model outputs for figure generation
    ├── val_predictions.npz
    └── test_predictions.npz
```

## How to obtain the data

Each source has a notebook that downloads it from scratch. See the parent
`notebooks/` directory:

* VIIRS FIRMS: `01a_viirs_firms_ingest_colab_v3.ipynb`
  (NASA FIRMS API; free registration required for an API key)
* ERA5: `01b_download_era5_aod.ipynb`
  (Copernicus CDS API; free registration required)
* MODIS land cover: `01c_surface_context_v2.ipynb`
  (NASA earthaccess; free registration required)
* OSM infrastructure: `01c_surface_context_v2.ipynb`
  (Overpass API; no auth required, rate-limited)

Total runtime to rebuild the labelled dataset from scratch is about
8-12 hours on a Colab free-tier session, dominated by the ERA5 download
and per-pixel co-location.
