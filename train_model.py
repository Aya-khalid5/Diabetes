"""Train the Random Forest EXACTLY as in main.py / diabetes.ipynb and save it for the app.

    python train_model.py                 # writes the deployable files
    python train_model.py --save-sklearn  # additionally dump the full scikit-learn pickle (~225 MB, do NOT commit)

Writes:
    model/diabetes_rf_compact.npz   lossless compact copy of the trained forest (~21 MB)
    model/metadata.json             feature order, labels, derived-feature maps, evaluation numbers

Before saving, the compact copy is compared with the scikit-learn model on the whole
held-out test set; the script aborts if even one predicted label differs.
Requires scikit-learn (see requirements-train.txt). The deployed app does not.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from core import config
from core.training import train_and_evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-sklearn", action="store_true", help="also save the full sklearn pickle")
    args = ap.parse_args()

    if not config.DATA_PATH.exists():
        raise SystemExit(f"Dataset not found: {config.DATA_PATH}")

    df = pd.read_csv(config.DATA_PATH)
    print(f"dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")

    model, compact, metadata = train_and_evaluate(df)

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    compact.save(config.MODEL_PATH)
    config.METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.save_sklearn:
        import joblib

        joblib.dump(model, config.MODEL_DIR / "diabetes_rf_sklearn_full.joblib", compress=3)

    size_mb = config.MODEL_PATH.stat().st_size / 1024 / 1024
    print(f"saved {config.MODEL_PATH.relative_to(config.ROOT)}  ({size_mb:.1f} MB)")
    print(f"saved {config.METADATA_PATH.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
