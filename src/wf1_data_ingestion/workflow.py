"""WF1: Data Ingestion Pipeline - Workflow.

Daily workflow that fetches, validates, stores, and quality-checks market data.

Schedule: Daily 06:00 UTC (configured in launch_plans/)
Pipeline: fetch_market_data -> validate_ticks -> store_to_database -> check_data_quality

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
    check_data_quality,
)


@workflow
def data_ingestion_workflow(
    symbols: List[str] = PHASE1_SYMBOLS,
    date: str = "",
) -> str:
    """WF1: Daily data ingestion pipeline.

    Fetches EOD market data, validates per Brenndoerfer patterns,
    stores to PostgreSQL, and generates a quality report.

    Args:
        symbols: List of stock symbols to fetch. Default: 10 US Large Caps.
        date: Target date (YYYY-MM-DD). Empty string = today.

    Returns:
        Data quality report as formatted string.
    """
    batch = fetch_market_data(symbols=symbols, date=date)
    validated = validate_ticks(batch=batch)
    store_result = store_to_database(batch=validated)
    quality_report = check_data_quality(batch=validated)
    return quality_report
