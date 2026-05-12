"""
05_export_to_huggingface.py
============================

Upload the trained FireSight-IR model to Hugging Face Hub.

What gets pushed:
  * firesight_pinn_best.pt        : main trained checkpoint (epoch 23)
  * firesight_pinn_final.pt       : last-epoch checkpoint
  * training_log.json             : per-epoch metrics for the full model
  * ablations/                    : all four ablation variants
                                    (no-physics, no-ERA5, no-surface)
                                    plus all_ablation_results.json
  * README.md                     : the model card from docs/MODEL_CARD.md

Prerequisites:
  1. Hugging Face account at https://huggingface.co
  2. Access token (write scope) from https://huggingface.co/settings/tokens
  3. `pip install huggingface_hub`
  4. `huggingface-cli login` (one-time; pastes the token)

Run from the repo root:
  python scripts/05_export_to_huggingface.py

The script will create the repo if it does not exist, then upload all
listed files. Subsequent runs replace files of the same name.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ─── CONFIG ──────────────────────────────────────────────────────────────────
# Hub repo identifier. The username portion must match your HF account.
HF_USERNAME = "emmanuelibekwe5525"
HF_REPO_NAME = "firesight-ir"
HF_REPO_ID = f"{HF_USERNAME}/{HF_REPO_NAME}"

# Local paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MAIN_CHECKPOINT = PROJECT_ROOT / "models" / "firesight_pinn_best.pt"
FINAL_CHECKPOINT = PROJECT_ROOT / "models" / "firesight_pinn_final.pt"
TRAINING_LOG = PROJECT_ROOT / "models" / "training_log.json"
MODEL_CARD = PROJECT_ROOT / "docs" / "MODEL_CARD.md"

# Ablation files: pick the deeper nested folder if present, else flatter one
ABLATION_DIR_CANDIDATES = [
    PROJECT_ROOT / "models" / "ablations" / "ablations",
    PROJECT_ROOT / "models" / "ablations",
]

# Files to upload from ablations/ (anything matching these patterns)
ABLATION_PATTERNS = ["*_best.pt", "*_log.json", "*_metrics.json",
                     "all_ablation_results.json"]


# ─── HELPERS ─────────────────────────────────────────────────────────────────
def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def collect_ablation_files(ablation_dir):
    files = []
    for pattern in ABLATION_PATTERNS:
        files.extend(sorted(ablation_dir.glob(pattern)))
    return files


def main():
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("ERROR: huggingface_hub not installed.")
        print("Run: pip install huggingface_hub")
        sys.exit(1)

    # Sanity-check that the required files exist
    required = {
        "Main checkpoint": MAIN_CHECKPOINT,
        "Training log": TRAINING_LOG,
        "Model card": MODEL_CARD,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        print("ERROR: missing required files:")
        for name in missing:
            print(f"  {name}: {required[name]}")
        sys.exit(1)

    ablation_dir = first_existing(ABLATION_DIR_CANDIDATES)
    ablation_files = collect_ablation_files(ablation_dir) if ablation_dir else []

    print(f"Project root      : {PROJECT_ROOT}")
    print(f"HF repo id        : {HF_REPO_ID}")
    print(f"Main checkpoint   : {MAIN_CHECKPOINT.name} "
          f"({MAIN_CHECKPOINT.stat().st_size / 1024:.0f} KB)")
    print(f"Training log      : {TRAINING_LOG.name}")
    print(f"Model card        : {MODEL_CARD}")
    print(f"Ablation files    : {len(ablation_files)} from {ablation_dir}")
    print()

    api = HfApi()

    # Step 1: create the repo if needed
    print(f"Step 1: ensuring repo {HF_REPO_ID} exists ...")
    try:
        create_repo(
            repo_id=HF_REPO_ID,
            repo_type="model",
            exist_ok=True,
            private=False,
        )
        print("  OK")
    except Exception as e:
        print(f"  FAILED: {e}")
        print("  Did you run `huggingface-cli login`?")
        sys.exit(1)

    # Step 2: upload the model card as README.md (HF expects this filename)
    print()
    print("Step 2: uploading model card (README.md) ...")
    api.upload_file(
        path_or_fileobj=str(MODEL_CARD),
        path_in_repo="README.md",
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message="Add model card",
    )
    print("  OK")

    # Step 3: upload the main checkpoint
    print()
    print("Step 3: uploading main checkpoint ...")
    api.upload_file(
        path_or_fileobj=str(MAIN_CHECKPOINT),
        path_in_repo=MAIN_CHECKPOINT.name,
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message="Add main FireSightPINN checkpoint",
    )
    print(f"  OK: {MAIN_CHECKPOINT.name}")

    # Step 4: upload the final-epoch checkpoint (optional)
    if FINAL_CHECKPOINT.exists():
        print()
        print("Step 4: uploading final-epoch checkpoint ...")
        api.upload_file(
            path_or_fileobj=str(FINAL_CHECKPOINT),
            path_in_repo=FINAL_CHECKPOINT.name,
            repo_id=HF_REPO_ID,
            repo_type="model",
            commit_message="Add final-epoch checkpoint",
        )
        print(f"  OK: {FINAL_CHECKPOINT.name}")

    # Step 5: upload training log
    print()
    print("Step 5: uploading training log ...")
    api.upload_file(
        path_or_fileobj=str(TRAINING_LOG),
        path_in_repo=TRAINING_LOG.name,
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message="Add training log",
    )
    print(f"  OK: {TRAINING_LOG.name}")

    # Step 6: upload ablation files
    if ablation_files:
        print()
        print(f"Step 6: uploading {len(ablation_files)} ablation files ...")
        for f in ablation_files:
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f"ablations/{f.name}",
                repo_id=HF_REPO_ID,
                repo_type="model",
                commit_message=f"Add ablation file: {f.name}",
            )
            print(f"  OK: ablations/{f.name}")

    print()
    print("=" * 60)
    print(f"Done. Visit https://huggingface.co/{HF_REPO_ID}")
    print("=" * 60)


if __name__ == "__main__":
    main()
