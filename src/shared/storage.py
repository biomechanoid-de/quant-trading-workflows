"""S3/MinIO Parquet storage helpers for Quant Trading Workflows.

Writes market data as Parquet files to MinIO with Hive-style partitioning:
    s3://quant-data/market_data/source=yfinance/year=2026/month=02/day=08/market_data.parquet

Uses boto3 for S3 operations and pyarrow for Parquet serialization.
Lazy imports to avoid issues in test environments without S3 access.
"""

import io
from typing import Optional

from src.shared.config import (
    MINIO_ENDPOINT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    S3_DATA_BUCKET,
)
from src.shared.models import MarketDataBatch


def get_s3_client():
    """Get a boto3 S3 client configured for MinIO.

    Uses in-cluster endpoint by default (minio.flyte.svc.cluster.local:9000).
    Override via MINIO_ENDPOINT env var for local development.
    """
    import boto3
    from botocore.client import Config

    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def _build_s3_key(date: str, source: str = "yfinance") -> str:
    """Build Hive-style partitioned S3 key from date string.

    Args:
        date: Date string in YYYY-MM-DD format.
        source: Data source name (default: yfinance).

    Returns:
        S3 key like: market_data/source=yfinance/year=2026/month=02/day=08/market_data.parquet
    """
    parts = date.split("-")
    if len(parts) != 3:
        raise ValueError(f"Invalid date format '{date}', expected YYYY-MM-DD")

    year, month, day = parts
    return (
        f"market_data/source={source}/"
        f"year={year}/month={month}/day={day}/"
        f"market_data.parquet"
    )


def batch_to_parquet_bytes(batch: MarketDataBatch) -> bytes:
    """Convert a MarketDataBatch to Parquet bytes in memory.

    Creates a columnar Parquet file with one row per symbol.
    Schema: symbol, date, close, volume, spread_bps, market_cap, data_source

    Args:
        batch: Validated MarketDataBatch.

    Returns:
        Parquet file contents as bytes.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Build rows only for symbols that have price data
    symbols = [s for s in batch.symbols if s in batch.prices]

    table = pa.table({
        "symbol": pa.array(symbols, type=pa.string()),
        "date": pa.array([batch.date] * len(symbols), type=pa.string()),
        "close": pa.array(
            [batch.prices[s] for s in symbols], type=pa.float64()
        ),
        "volume": pa.array(
            [batch.volumes.get(s, 0) for s in symbols], type=pa.int64()
        ),
        "spread_bps": pa.array(
            [batch.spreads.get(s, 0.0) for s in symbols], type=pa.float64()
        ),
        "market_cap": pa.array(
            [batch.market_caps.get(s, 0.0) for s in symbols], type=pa.float64()
        ),
        "data_source": pa.array(
            ["yfinance"] * len(symbols), type=pa.string()
        ),
    })

    # Write to in-memory buffer
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    return buf.getvalue()


def store_parquet_to_s3(
    batch: MarketDataBatch,
    source: str = "yfinance",
    bucket: Optional[str] = None,
) -> str:
    """Store a MarketDataBatch as Parquet to MinIO/S3.

    Writes to Hive-style partitioned path:
        s3://{bucket}/market_data/source={source}/year=YYYY/month=MM/day=DD/market_data.parquet

    Overwrites existing file for the same partition (idempotent for Flyte retries).

    Args:
        batch: Validated MarketDataBatch with price data.
        source: Data source name for partitioning.
        bucket: S3 bucket name. Default: from config.

    Returns:
        Summary string with S3 path and row count.
    """
    bucket = bucket or S3_DATA_BUCKET
    s3_key = _build_s3_key(date=batch.date, source=source)
    parquet_bytes = batch_to_parquet_bytes(batch)

    client = get_s3_client()
    client.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=parquet_bytes,
        ContentType="application/octet-stream",
    )

    rows = len([s for s in batch.symbols if s in batch.prices])
    s3_path = f"s3://{bucket}/{s3_key}"
    return f"Stored {rows} rows to {s3_path} ({len(parquet_bytes)} bytes)"
