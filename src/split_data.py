"""I create a time-based train/validation split for the model dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils_io import load_parquet, save_parquet

LOGGER = logging.getLogger(__name__)

INPUT_PATH = Path("data/processed/model_dataset.parquet")
TRAIN_PATH = Path("data/processed/train.parquet")
VALID_PATH = Path("data/processed/valid.parquet")
VALID_DAYS = 180


def configure_logging() -> None:
    """I configure consistent logging for split reporting."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def split_train_valid(df: pd.DataFrame, valid_days: int = VALID_DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """I split data into historical train and most-recent validation windows."""
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce", utc=False)
    if out["datetime"].isna().any():
        bad_count = int(out["datetime"].isna().sum())
        raise RuntimeError(f"Found {bad_count} invalid datetime values in model dataset.")

    out = out.sort_values("datetime").reset_index(drop=True)
    # I use a pure time cutoff (not random split) to simulate real forecasting deployment.
    cutoff = out["datetime"].max() - pd.Timedelta(days=valid_days)

    train = out[out["datetime"] < cutoff].copy()
    valid = out[out["datetime"] >= cutoff].copy()

    if train.empty or valid.empty:
        raise RuntimeError(
            "Train or validation split is empty. Reduce VALID_DAYS or check dataset size."
        )

    return train, valid


def main() -> None:
    """I build and save time-ordered train/validation splits."""
    configure_logging()
    try:
        dataset = load_parquet(INPUT_PATH)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model dataset from {INPUT_PATH}. Run src.build_features first."
        ) from exc

    train, valid = split_train_valid(dataset, valid_days=VALID_DAYS)
    save_parquet(train, TRAIN_PATH)
    save_parquet(valid, VALID_PATH)

    LOGGER.info("Saved train split to %s (%d rows).", TRAIN_PATH, len(train))
    LOGGER.info("Saved valid split to %s (%d rows).", VALID_PATH, len(valid))
    LOGGER.info(
        "Split ranges | train: [%s, %s] | valid: [%s, %s]",
        train["datetime"].min(),
        train["datetime"].max(),
        valid["datetime"].min(),
        valid["datetime"].max(),
    )


if __name__ == "__main__":
    main()
