"""WF1: Data Ingestion Pipeline - Tasks.

Daily pipeline that fetches EOD market data, validates it,
stores it to PostgreSQL + Parquet (MinIO), and generates a quality report.

Schedule: Daily 06:00 UTC (before European market open)
Node: Any Pi 4 Worker

Task chain: fetch_market_data -> validate_ticks -> [store_to_database, store_to_parquet] -> check_data_quality
Dual-write: PostgreSQL (hot data, fast queries) + Parquet on MinIO (cold data, bulk analysis)
"""

from typing import List

from flytekit import task, Resources

from src.shared.models import MarketDataBatch


@task(
    requests=Resources(cpu="500m", mem="512Mi"),
    limits=Resources(cpu="1000m", mem="1024Mi"),
)
def fetch_market_data(symbols: List[str], date: str) -> MarketDataBatch:
    """Fetch end-of-day market data via provider abstraction.

    Phase 1: YFinanceProvider (free).
    Later: Switch to EODHD, Massive, or Alpha Vantage by changing
    the provider instantiation (1 line change).

    Args:
        symbols: List of stock ticker symbols to fetch.
        date: Target date (YYYY-MM-DD). Empty string means today.

    Returns:
        MarketDataBatch with prices, volumes, spreads, and market caps.
    """
    from src.shared.providers.yfinance_provider import YFinanceProvider

    provider = YFinanceProvider()
    return provider.fetch_eod(symbols=symbols, date=date)


@task(
    requests=Resources(cpu="200m", mem="256Mi"),
    limits=Resources(cpu="500m", mem="512Mi"),
)
def validate_ticks(batch: MarketDataBatch) -> MarketDataBatch:
    """Validate market data batch following Brenndoerfer patterns.

    Validation checks:
    - Zero or negative prices -> filtered out
    - Unreasonable spreads (>1000 bps) -> filtered out
    - Missing data (symbol not in prices) -> counted as issue
    - Stale data detection (future enhancement)

    Args:
        batch: Raw MarketDataBatch from fetch_market_data.

    Returns:
        Validated MarketDataBatch with filtered data and updated quality score.
    """
    validated_prices = {}
    validated_volumes = {}
    validated_spreads = {}
    issues = 0

    for symbol in batch.symbols:
        price = batch.prices.get(symbol)
        volume = batch.volumes.get(symbol)
        spread = batch.spreads.get(symbol)

        # Skip symbols with missing data
        if price is None:
            issues += 1
            continue

        # Zero or negative price check
        if price <= 0:
            issues += 1
            continue

        # Unreasonable spread check (>1000 bps = 10%)
        if spread is not None and spread > 1000:
            issues += 1
            continue

        validated_prices[symbol] = price
        validated_volumes[symbol] = volume if volume is not None else 0
        validated_spreads[symbol] = spread if spread is not None else 0.0

    # Recalculate quality score based on validated data
    quality = len(validated_prices) / len(batch.symbols) if batch.symbols else 0.0

    return MarketDataBatch(
        symbols=batch.symbols,
        date=batch.date,
        prices=validated_prices,
        volumes=validated_volumes,
        spreads=validated_spreads,
        market_caps=batch.market_caps,
        data_quality_score=round(quality, 4),
    )


@task(
    requests=Resources(cpu="200m", mem="256Mi"),
    limits=Resources(cpu="500m", mem="512Mi"),
)
def store_to_database(batch: MarketDataBatch) -> str:
    """Store validated market data to PostgreSQL (pi5-1tb SSD).

    Uses UPSERT (ON CONFLICT UPDATE) for idempotency.
    Flyte may retry failed tasks, so re-running is safe.

    Args:
        batch: Validated MarketDataBatch.

    Returns:
        Summary string with number of rows stored.
    """
    from src.shared.db import store_market_data

    rows = store_market_data(batch)
    return f"Stored {rows} rows for {batch.date}"


@task(
    requests=Resources(cpu="200m", mem="256Mi"),
    limits=Resources(cpu="500m", mem="512Mi"),
)
def store_to_parquet(batch: MarketDataBatch) -> str:
    """Store validated market data as Parquet to MinIO/S3 (Data Lake).

    Writes to Hive-style partitioned path:
        s3://quant-data/market_data/source=yfinance/year=YYYY/month=MM/day=DD/market_data.parquet

    This runs IN PARALLEL with store_to_database (dual-write pattern):
    - PostgreSQL: Hot data for WF2-WF5 real-time queries
    - Parquet/MinIO: Cold data for backtesting, ML training, archive

    Overwrites existing file for the same partition (idempotent).

    Args:
        batch: Validated MarketDataBatch.

    Returns:
        Summary string with S3 path and row count.
    """
    from src.shared.storage import store_parquet_to_s3

    return store_parquet_to_s3(batch)


@task(
    requests=Resources(cpu="100m", mem="128Mi"),
    limits=Resources(cpu="200m", mem="256Mi"),
)
def check_data_quality(batch: MarketDataBatch) -> str:
    """Generate a data quality report.

    Reports:
    - Number of symbols requested vs fetched
    - Quality score as percentage
    - Missing symbols list
    - Zero-volume symbols (potential stale data)

    Args:
        batch: Validated MarketDataBatch.

    Returns:
        Quality report as formatted string.
    """
    total = len(batch.symbols)
    fetched = len(batch.prices)
    missing = [s for s in batch.symbols if s not in batch.prices]

    lines = [
        f"=== Data Quality Report for {batch.date} ===",
        f"Symbols requested: {total}",
        f"Symbols fetched:   {fetched}",
        f"Quality score:     {batch.data_quality_score:.2%}",
    ]

    if missing:
        lines.append(f"Missing symbols:   {', '.join(missing)}")

    # Check for volume outliers (volume = 0 might indicate stale data)
    zero_volume = [s for s, v in batch.volumes.items() if v == 0]
    if zero_volume:
        lines.append(f"Zero volume:       {', '.join(zero_volume)}")

    # Price summary
    if batch.prices:
        avg_price = sum(batch.prices.values()) / len(batch.prices)
        lines.append(f"Average price:     ${avg_price:.2f}")

    lines.append("=" * 45)
    return "\n".join(lines)


@task(
    requests=Resources(cpu="200m", mem="256Mi"),
    limits=Resources(cpu="500m", mem="512Mi"),
)
def fetch_dividend_events(symbols: List[str], date: str) -> str:
    """Fetch recent dividend events for all symbols and store to DB.

    Queries each symbol for dividends with ex-dates in the last 90 days
    via the provider abstraction, then stores new events to the dividends
    table with processed=FALSE for later matching in WF4.

    Args:
        symbols: List of stock ticker symbols to check.
        date: Reference date (YYYY-MM-DD). Empty string = today.

    Returns:
        Summary string with count of new dividend events found.
    """
    from src.shared.providers.yfinance_provider import YFinanceProvider
    from src.shared.db import store_dividends

    provider = YFinanceProvider()
    all_dividends = []

    for symbol in symbols:
        divs = provider.fetch_dividends(symbol)
        all_dividends.extend(divs)

    if all_dividends:
        rows = store_dividends(all_dividends)
        return f"Found {len(all_dividends)} dividend events, stored {rows} rows"

    return "No dividend events found"
