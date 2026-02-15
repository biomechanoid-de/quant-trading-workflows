"""WF1: Data Ingestion Pipeline - Workflow.

Daily workflow that fetches, validates, and dual-writes market data
to PostgreSQL (hot) + Parquet/MinIO (cold), fetches dividend events,
then quality-checks.

Schedule: Daily 06:00 UTC (configured in launch_plans/)

Pipeline:
    fetch_market_data -> validate_ticks -> ┬─ store_to_database    ─┬─> check_data_quality
                                           ├─ store_to_parquet     ─┤
                                           └─ fetch_dividend_events ┘

The store + dividend tasks run IN PARALLEL (Flyte DAG: no dependency between them).

Example local run:
    pyflyte run src/wf1_data_ingestion/workflow.py data_ingestion_workflow \\
        --symbols '["AAPL", "MSFT", "GOOGL"]' --date 2026-02-07
"""

from typing import List

from flytekit import workflow

from src.shared.config import PHASE1_SYMBOLS
from src.wf1_data_ingestion.tasks import (
    fetch_market_data,
    validate_ticks,
    store_to_database,
    store_to_parquet,
    check_data_quality,
    fetch_dividend_events,
)


@workflow
def data_ingestion_workflow(
    symbols: List[str] = PHASE1_SYMBOLS,
    date: str = "",
) -> str:
    """WF1: Daily data ingestion pipeline with dual-write + dividends.

    Fetches EOD market data, validates per Brenndoerfer patterns,
    stores to PostgreSQL AND Parquet/MinIO in parallel, fetches
    dividend events, then generates a quality report.

    Storage strategy (Hybrid):
    - PostgreSQL: Hot data for WF2-WF5 real-time SQL queries (last 90 days)
    - Parquet/MinIO: Cold data for backtesting, ML, and archive (full history)

    Args:
        symbols: List of stock symbols to fetch. Default: 10 US Large Caps.
        date: Target date (YYYY-MM-DD). Empty string = today.

    Returns:
        Data quality report as formatted string.
    """
    batch = fetch_market_data(symbols=symbols, date=date)
    validated = validate_ticks(batch=batch)

    # Dual-write + dividends: all three tasks run in parallel
    store_db_result = store_to_database(batch=validated)
    store_s3_result = store_to_parquet(batch=validated)
    dividend_result = fetch_dividend_events(symbols=symbols, date=date)

    # Quality check runs after validation (doesn't need store/dividend results)
    quality_report = check_data_quality(batch=validated)
    return quality_report
