# Legacy notebooks

Earlier iterations of the pipeline notebooks, kept for reference and
traceability. None of these are part of the canonical pipeline; the
current versions live one level up in `notebooks/`.

## What lives here

* Earlier versions of Module 1a (VIIRS FIRMS ingest), pre-`_v3`.
* Earlier versions of Module 1c (surface context), pre-`_v2`.
* `01a_viirs_modis_cloud_ingest.ipynb`: an early combined-ingest experiment
  superseded by the current 1a + 1c split.
* Earlier `02_feature_engineering.ipynb` (pre-`_v2`).
* Earlier 3a training notebooks (`_fast`, base) before the `_final` version.
* `03b_false_alarm_discriminator.ipynb`: the planned Module 3b downstream
  discriminator. Retired because the first-stage classifier achieved
  AUC = 1.0 on false-alarm detection and the discriminator's ambiguous
  zone (P in [0.10, 0.90]) was effectively empty.
* `02_linkedin_visualization.ipynb`, `03a_linkedin_visualization.ipynb`:
  social-media visualisations.
* `Untitled.ipynb`: scratch notebook.
