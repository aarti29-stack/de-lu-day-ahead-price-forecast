"""I fetch and clean DE-LU day-ahead prices from ENTSO-E."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient

from src.config import get_entsoe_api_key
from src.utils_io import ensure_dir, save_parquet

LOGGER = logging.getLogger(__name__)
TARGET_TZ = "Europe/Berlin"
AREA_CODE = "DE_LU"
RAW_PATH = Path("data/raw/prices_de_lu_raw.parquet")
CLEAN_PATH = Path("data/processed/prices_de_lu_hourly.parquet")


def configure_logging() -> None:
    """I configure a consistent log format for pipeline runs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def compute_time_window(years: int = 5) -> tuple[pd.Timestamp, pd.Timestamp]:
    """I compute a [start, end) window for the last `years` in Berlin time."""
    # I floor to hour so downstream joins and resampling stay aligned on exact hour boundaries.
    end = pd.Timestamp.now(tz=TARGET_TZ).floor("h")
    start = end - pd.DateOffset(years=years)
    return start, end


def fetch_raw_prices(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """I fetch raw day-ahead prices for DE-LU from ENTSO-E."""
    api_key = get_entsoe_api_key()
    client = EntsoePandasClient(api_key=api_key)
    try:
        series = client.query_day_ahead_prices(country_code=AREA_CODE, start=start, end=end)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch ENTSO-E day-ahead prices for {AREA_CODE}: {exc}"
        ) from exc

    if series.empty:
        raise RuntimeError("ENTSO-E returned an empty price series for the requested window.")

    return series


def normalize_timezone(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """I normalize timestamps to Europe/Berlin while preserving true instants."""
    if index.tz is None:
        # ENTSO-E should return tz-aware data, but this fallback keeps behavior deterministic.
        LOGGER.warning("Received naive timestamps; assuming UTC before converting to Europe/Berlin.")
        return index.tz_localize("UTC").tz_convert(TARGET_TZ)
    return index.tz_convert(TARGET_TZ)


def clean_prices(series: pd.Series) -> pd.DataFrame:
    """I clean hourly prices and resolve duplicates deterministically."""
    if not isinstance(series.index, pd.DatetimeIndex):
        raise RuntimeError("Expected DatetimeIndex from ENTSO-E response.")

    # I coerce unexpected strings to NaN so bad values are explicitly handled below.
    df = pd.DataFrame({"price_eur_mwh": pd.to_numeric(series, errors="coerce")})
    df.index = normalize_timezone(df.index)
    df.index.name = "datetime"
    df = df.sort_index()

    raw_rows = len(df)
    duplicate_mask = df.index.duplicated(keep="last")
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count > 0:
        # At DST boundaries, repeated local clock times can appear after tz conversion.
        LOGGER.warning(
            "Detected %d duplicate timestamps after timezone normalization; "
            "resolving deterministically by keeping the last value.",
            duplicate_count,
        )
        df = df[~duplicate_mask]

    missing_price_count = int(df["price_eur_mwh"].isna().sum())
    if missing_price_count > 0:
        LOGGER.warning("Dropping %d rows with missing price values.", missing_price_count)
        df = df.dropna(subset=["price_eur_mwh"])

    LOGGER.info(
        "Cleaned price series from %d to %d rows.",
        raw_rows,
        len(df),
    )

    return df.reset_index()


def main() -> None:
    """I run fetch, cleaning, and save steps for DE-LU day-ahead prices."""
    configure_logging()
    try:
        start, end = compute_time_window(years=5)
        LOGGER.info("Fetching DE-LU day-ahead prices from %s to %s.", start, end)

        raw_series = fetch_raw_prices(start=start, end=end)

        raw_df = pd.DataFrame(
            {
                "datetime": normalize_timezone(raw_series.index),
                "price_eur_mwh": pd.to_numeric(raw_series.values, errors="coerce"),
            }
        )
        raw_df = raw_df.sort_values("datetime").reset_index(drop=True)

        # I persist both raw and cleaned views to keep data lineage auditable.
        ensure_dir(RAW_PATH.parent)
        ensure_dir(CLEAN_PATH.parent)
        save_parquet(raw_df, RAW_PATH)
        LOGGER.info("Saved raw series to %s (%d rows).", RAW_PATH, len(raw_df))

        clean_df = clean_prices(raw_series)
        save_parquet(clean_df, CLEAN_PATH)
        LOGGER.info("Saved cleaned hourly series to %s (%d rows).", CLEAN_PATH, len(clean_df))
    except Exception as exc:
        LOGGER.exception("Price fetch pipeline failed.")
        raise RuntimeError(f"Price fetch pipeline failed: {exc}") from exc


if __name__ == "__main__":
    main()
