"""PostgreSQL database helpers for Quant Trading Workflows.

Target: PostgreSQL on pi5-1tb (192.168.178.45) with NVMe SSD.
Uses psycopg2 with lazy imports to avoid import errors in test environments.
"""

from src.shared.config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
from src.shared.models import MarketDataBatch


def get_connection():
    """Get a PostgreSQL database connection.

    Connection parameters are read from environment variables via config.py.
    """
    import psycopg2

    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def store_market_data(batch: MarketDataBatch) -> int:
    """Store a MarketDataBatch into the market_data table.

    Uses UPSERT (ON CONFLICT UPDATE) for idempotency.
    Flyte may retry failed tasks, so this ensures no duplicates.

    Returns:
        Number of rows inserted/updated.
    """
    conn = get_connection()
    cursor = conn.cursor()
    rows_inserted = 0

    try:
        for symbol in batch.symbols:
            if symbol in batch.prices:
                cursor.execute(
                    """INSERT INTO market_data
                           (symbol, date, close, volume, spread_bps, data_source)
                       VALUES (%s, %s, %s, %s, %s, %s)
                       ON CONFLICT (symbol, date) DO UPDATE SET
                           close = EXCLUDED.close,
                           volume = EXCLUDED.volume,
                           spread_bps = EXCLUDED.spread_bps,
                           data_source = EXCLUDED.data_source,
                           created_at = NOW()""",
                    (
                        symbol,
                        batch.date,
                        batch.prices.get(symbol),
                        batch.volumes.get(symbol),
                        batch.spreads.get(symbol),
                        "yfinance",
                    ),
                )
                rows_inserted += 1
        conn.commit()
    finally:
        cursor.close()
        conn.close()

    return rows_inserted


def get_latest_market_data(symbol: str, days: int = 30) -> list:
    """Get the latest N days of market data for a symbol.

    Args:
        symbol: Stock ticker symbol.
        days: Number of days to retrieve.

    Returns:
        List of tuples (symbol, date, close, volume, spread_bps).
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """SELECT symbol, date, close, volume, spread_bps
               FROM market_data
               WHERE symbol = %s
               ORDER BY date DESC
               LIMIT %s""",
            (symbol, days),
        )
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()
