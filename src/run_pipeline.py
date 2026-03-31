"""I run the full day-ahead pipeline end to end in one command."""

from __future__ import annotations

import logging

from src import build_features, fetch_prices, smoke_test, split_data, train_baseline, validate_timeseries

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """I configure top-level logging for pipeline orchestration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    """I execute all stages in the correct order with clear stage boundaries."""
    configure_logging()
    # Stage order is intentional: each step produces artifacts required by the next one.
    stages = [
        ("smoke_test", smoke_test.main),
        ("fetch_prices", fetch_prices.main),
        ("validate_timeseries", validate_timeseries.main),
        ("build_features", build_features.main),
        ("split_data", split_data.main),
        ("train_baseline", train_baseline.main),
    ]

    for name, func in stages:
        LOGGER.info("========== START %s =========", name)
        try:
            func()
        except Exception as exc:
            LOGGER.exception("Stage failed: %s", name)
            raise RuntimeError(f"Pipeline failed at stage '{name}': {exc}") from exc
        LOGGER.info("========== END %s =========", name)

    LOGGER.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
