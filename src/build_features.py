"""I build leakage-safe features for day-ahead price forecasting."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils_io import load_parquet, save_parquet

LOGGER = logging.getLogger(__name__)

INPUT_PATH = Path("data/processed/prices_de_lu_hourly.parquet")
OUTPUT_PATH = Path("data/processed/model_dataset.parquet")


def configure_logging() -> None:
    """I configure a consistent log format for feature generation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _prepare_base(df: pd.DataFrame) -> pd.DataFrame:
    required = {"datetime", "price_eur_mwh"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce", utc=False)
    if out["datetime"].isna().any():
        bad_count = int(out["datetime"].isna().sum())
        raise RuntimeError(f"Found {bad_count} invalid datetime values.")

    # I sort and deduplicate first so all lag operations use a stable, unique hourly timeline.
    out = out.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
    out = out.set_index("datetime")
    out["price_eur_mwh"] = pd.to_numeric(out["price_eur_mwh"], errors="coerce")
    return out


def build_feature_table(prices: pd.DataFrame) -> pd.DataFrame:
    """I create leakage-safe features and a delivery-price target.

    I define the target at delivery timestamp t.
    I restrict features to information available at or before t-24h.
    """
    df = _prepare_base(prices)
    y = df["price_eur_mwh"]

    # I encode delivery-time calendar structure.
    feat = pd.DataFrame(index=df.index)
    feat["hour"] = feat.index.hour
    feat["weekday"] = feat.index.weekday
    feat["month"] = feat.index.month
    feat["dayofyear"] = feat.index.dayofyear
    feat["is_weekend"] = (feat["weekday"] >= 5).astype(int)

    # I add daily and weekly lag memory.
    feat["lag_24"] = y.shift(24)
    feat["lag_48"] = y.shift(48)
    feat["lag_72"] = y.shift(72)
    feat["lag_168"] = y.shift(168)

    # I compute rolling stats only from values known at auction time.
    past = y.shift(24)
    feat["roll_mean_24"] = past.rolling(window=24, min_periods=24).mean()
    feat["roll_std_24"] = past.rolling(window=24, min_periods=24).std()
    feat["roll_mean_168"] = past.rolling(window=168, min_periods=168).mean()
    feat["roll_std_168"] = past.rolling(window=168, min_periods=168).std()

    # This is the delivery-hour target we forecast.
    feat["target_price_eur_mwh"] = y
    feat = feat.dropna().reset_index().rename(columns={"index": "datetime"})

    if feat.empty:
        raise RuntimeError("Feature table is empty after dropna; check source data coverage.")

    return feat


def main() -> None:
    """I build and save the model-ready feature table."""
    configure_logging()
    try:
        raw = load_parquet(INPUT_PATH)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load hourly prices from {INPUT_PATH}. Run src.fetch_prices first."
        ) from exc

    dataset = build_feature_table(raw)
    save_parquet(dataset, OUTPUT_PATH)
    LOGGER.info("Saved feature table to %s (%d rows).", OUTPUT_PATH, len(dataset))


if __name__ == "__main__":
    main()
