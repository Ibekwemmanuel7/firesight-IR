# Dashboard

Streamlit-based operator-facing demo for FireSight-IR.

## Local run

```bash
pip install streamlit
streamlit run dashboard/app.py
```

The app loads the trained checkpoint from `models/firesight_pinn_best.pt`
and serves an interactive interface for exploring detections.

## Deploying

[Streamlit Community Cloud](https://streamlit.io/cloud) takes a public
GitHub repo and deploys the app to a free hosted URL. Connect your
`firesight-ir` repo, point the service at `dashboard/app.py`, and add the
`requirements.txt` from the repo root.

For the deployed version, the full training dataset is too large to
bundle. Either:

1. Bundle a small sample dataset (a few thousand pre-computed
   predictions cached as a parquet file) so the deployed app demonstrates
   inference without requiring the user to upload data, or
2. Fetch sample data at runtime from a public bucket (HuggingFace Hub,
   S3 with public read, etc.).

The cached-sample approach is simpler and doesn't depend on outbound
network access from the deployed instance.

## Source

The dashboard was migrated from `notebooks/firesight_dashboard.py` and is
preserved as `app.py` here.
