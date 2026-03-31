# Pipeline Notes (What, Where, Why)

I wrote this file in simple language, in my own voice.
I use it as side-by-side notes while reading the code in src/.

## 1) Configuration and Secrets
- File: src/config.py
- What I do: I load ENTSOE_API_KEY from .env.
- Why I do it: I keep secrets out of code and I stop early if the key is missing.

## 2) I/O Helpers
- File: src/utils_io.py
- What I do: I create folders, and I read/write parquet files.
- Why I do it: I keep file handling simple and consistent in every stage.

## 3) Data Fetch + Cleaning
- File: src/fetch_prices.py
- What I do: I download DE-LU day-ahead prices and normalize timezone.
- Why I do it: I need a clean hourly base table before modeling.
- Important choices I made:
  - I use Europe/Berlin time to match delivery market time.
  - I remove duplicate timestamps in a fixed way (keep last).
  - I drop invalid price values after numeric conversion.

## 4) Time-Series Validation
- File: src/validate_timeseries.py
- What I do: I check sort order, duplicates, and missing hours.
- Why I do it: lag and rolling features only work correctly on a clean hourly timeline.

## 5) Feature Engineering (Leakage-Safe)
- File: src/build_features.py
- What I do: I build calendar, lag, and rolling features.
- Why I do it: the model needs useful history without seeing future information.
- Leakage rule I follow:
  - My target is price at delivery time t.
  - My rolling stats use y.shift(24), so I only use information available before delivery auction time.

## 6) Train/Validation Split
- File: src/split_data.py
- What I do: I split using a time cutoff.
- Why I do it: random split can give overly optimistic results in forecasting.

## 7) Baseline Training + Metrics
- File: src/train_baseline.py
- What I do: I train a linear regression baseline and save predictions/metrics.
- Why I do it: I need a clear starting benchmark before trying advanced models.
- Metrics I save:
  - Overall MAE and RMSE.
  - By-hour MAE and RMSE to see which hours are easier or harder.

## 8) End-to-End Orchestration
- File: src/run_pipeline.py
- What I do: I run all stages in dependency order.
- Why I do it: one command gives reproducible runs for review and demos.

## Typical Data Flow
1. fetch_prices.py -> data/processed/prices_de_lu_hourly.parquet
2. build_features.py -> data/processed/model_dataset.parquet
3. split_data.py -> data/processed/train.parquet and data/processed/valid.parquet
4. train_baseline.py -> reports/baseline_valid_predictions.parquet and reports/baseline_metrics.json
