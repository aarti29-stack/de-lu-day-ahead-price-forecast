"""I train a baseline model and report validation metrics."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.utils_io import load_parquet, save_parquet

LOGGER = logging.getLogger(__name__)

TRAIN_PATH = Path("data/processed/train.parquet")
VALID_PATH = Path("data/processed/valid.parquet")
PRED_PATH = Path("reports/baseline_valid_predictions.parquet")
METRICS_PATH = Path("reports/baseline_metrics.json")
TARGET_COL = "target_price_eur_mwh"


def configure_logging() -> None:
    """I configure consistent logging for baseline training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _validate_columns(df: pd.DataFrame) -> None:
    required = {"datetime", "hour", TARGET_COL}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")


def _build_pipeline(feature_columns: list[str]) -> Pipeline:
    # These are cyclical calendar categories where one-hot encoding works better than raw integers.
    categorical = [c for c in ["hour", "weekday", "month", "is_weekend"] if c in feature_columns]
    numeric = [c for c in feature_columns if c not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )


def _regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    # MAE is robust and interpretable in EUR/MWh; RMSE highlights large misses.
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"mae": mae, "rmse": rmse}


def main() -> None:
    """I train the baseline model and save predictions plus metrics."""
    configure_logging()

    try:
        train = load_parquet(TRAIN_PATH)
        valid = load_parquet(VALID_PATH)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load train/valid splits. Run src.build_features and src.split_data first."
        ) from exc

    _validate_columns(train)
    _validate_columns(valid)

    train = train.sort_values("datetime").reset_index(drop=True)
    valid = valid.sort_values("datetime").reset_index(drop=True)

    feature_columns = [c for c in train.columns if c not in {"datetime", TARGET_COL}]
    X_train = train[feature_columns]
    y_train = train[TARGET_COL]
    X_valid = valid[feature_columns]
    y_valid = valid[TARGET_COL]

    pipeline = _build_pipeline(feature_columns)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_valid)

    overall = _regression_metrics(y_valid, y_pred)

    pred_df = valid[["datetime", "hour", TARGET_COL]].copy()
    pred_df = pred_df.rename(columns={TARGET_COL: "y_true"})
    pred_df["y_pred"] = y_pred
    pred_df["abs_error"] = (pred_df["y_true"] - pred_df["y_pred"]).abs()
    save_parquet(pred_df, PRED_PATH)

    # Hour-level metrics show where the baseline is strong or weak across the day profile.
    by_hour_rows: list[dict[str, float | int]] = []
    for hour, group in pred_df.groupby("hour"):
        metrics = _regression_metrics(group["y_true"], group["y_pred"].to_numpy())
        by_hour_rows.append(
            {
                "hour": int(hour),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "count": int(len(group)),
            }
        )

    metrics_payload = {
        "overall": overall,
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "by_hour": sorted(by_hour_rows, key=lambda x: int(x["hour"])),
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    LOGGER.info("Saved validation predictions to %s (%d rows).", PRED_PATH, len(pred_df))
    LOGGER.info("Saved baseline metrics to %s.", METRICS_PATH)
    LOGGER.info(
        "Validation metrics | MAE: %.4f | RMSE: %.4f",
        overall["mae"],
        overall["rmse"],
    )


if __name__ == "__main__":
    main()
