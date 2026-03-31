"""I use these helpers for filesystem and Parquet reads/writes."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """I create a directory if needed and return it as a Path."""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """I save a DataFrame to Parquet and create parent folders first."""
    file_path = Path(path)
    ensure_dir(file_path.parent)
    df.to_parquet(file_path, index=False)


def load_parquet(path: str | Path) -> pd.DataFrame:
    """I load and return a DataFrame from a Parquet file."""
    file_path = Path(path)
    return pd.read_parquet(file_path)
