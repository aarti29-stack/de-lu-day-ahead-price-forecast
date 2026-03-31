"""I validate the cleaned DE-LU hourly day-ahead price series."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils_io import load_parquet

LOGGER = logging.getLogger(__name__)
INPUT_PATH = Path("data/processed/prices_de_lu_hourly.parquet")


def configure_logging() -> None:
    """I configure consistent logging for validation output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _check_columns(df: pd.DataFrame) -> None:
    expected = {"datetime", "price_eur_mwh"}
    missing = expected.difference(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")


def _prepare_datetime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce", utc=False)
    if out["datetime"].isna().any():
        bad_count = int(out["datetime"].isna().sum())
        raise RuntimeError(f"Found {bad_count} invalid datetime values.")
    return out.sort_values("datetime").reset_index(drop=True)


def validate(df: pd.DataFrame) -> None:
    """I run core time-series checks and log what I find."""
    _check_columns(df)
    ts = _prepare_datetime(df)

    dt = ts["datetime"]
    is_monotonic = bool(dt.is_monotonic_increasing)
    duplicate_count = int(dt.duplicated(keep=False).sum())

    idx = pd.DatetimeIndex(dt)
    full_index = pd.date_range(start=idx.min(), end=idx.max(), freq="h", tz=idx.tz)
    missing_hours = full_index.difference(idx)

    LOGGER.info("Validation results for %s", INPUT_PATH)
    LOGGER.info("Rows: %d", len(ts))
    LOGGER.info("Datetime monotonic increasing: %s", is_monotonic)
    LOGGER.info("Duplicate timestamps: %d", duplicate_count)
    LOGGER.info("Missing hourly timestamps: %d", len(missing_hours))

    if len(missing_hours) > 0:
        preview = [str(t) for t in missing_hours[:10]]
        LOGGER.warning("First missing timestamps (up to 10): %s", preview)

    stats = ts["price_eur_mwh"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    LOGGER.info("Price summary statistics:\n%s", stats.to_string())


def main() -> None:
    """I load cleaned prices and execute validation checks."""
    configure_logging()
    try:
        df = load_parquet(INPUT_PATH)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load processed prices from {INPUT_PATH}. Run src.fetch_prices first."
        ) from exc

    validate(df)


if __name__ == "__main__":
    main()
