# de-lu-day-ahead-price-forecast
In this project, I forecast day-ahead electricity prices for the DE-LU bidding zone using ENTSO-E data (last 5 years).

## Code Notes

For my side-by-side "what/where/why" notes, I use:

- docs/pipeline_notes.md

## Quickstart

```bash
# 1) I create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2) I install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) I configure my ENTSO-E API key
cp .env.example .env
# edit .env and set ENTSOE_API_KEY=your_key_here

# 4) I run a smoke test
python -m src.smoke_test

# 5) I fetch and clean DE-LU prices (last 5 years)
python -m src.fetch_prices

# 6) I validate the cleaned hourly time series
python -m src.validate_timeseries

# 7) I build a leakage-safe feature table
python -m src.build_features

# 8) I create a time-based train/validation split
python -m src.split_data

# 9) I train the baseline model and save metrics
python -m src.train_baseline

# 10) I run everything in one command with live logs
python -u -m src.run_pipeline 2>&1 | tee reports/live_pipeline.log

# optional: in another terminal, I follow logs live
tail -f reports/live_pipeline.log
```
