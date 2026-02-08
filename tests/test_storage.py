"""Tests for S3/MinIO Parquet storage.

Tests the pure logic functions (Parquet serialization, S3 key building).
store_parquet_to_s3 is mocked since it requires MinIO access.
"""

import io

import pyarrow.parquet as pq

from src.shared.models import MarketDataBatch
from src.shared.storage import _build_s3_key, batch_to_parquet_bytes


# ============================================================
# _build_s3_key tests
# ============================================================

def test_build_s3_key_standard_date():
    """Standard date produces correct Hive-style path."""
    key = _build_s3_key("2026-02-08", source="yfinance")
    assert key == "market_data/source=yfinance/year=2026/month=02/day=08/market_data.parquet"


def test_build_s3_key_different_source():
    """Different data source is reflected in the partition path."""
    key = _build_s3_key("2026-12-31", source="eodhd")
    assert key == "market_data/source=eodhd/year=2026/month=12/day=31/market_data.parquet"


def test_build_s3_key_first_of_year():
    """First day of year edge case."""
    key = _build_s3_key("2026-01-01", source="yfinance")
    assert key == "market_data/source=yfinance/year=2026/month=01/day=01/market_data.parquet"


def test_build_s3_key_invalid_date():
    """Invalid date format raises ValueError."""
    import pytest
    with pytest.raises(ValueError, match="Invalid date format"):
        _build_s3_key("20260208")


def test_build_s3_key_invalid_date_partial():
    """Partial date format raises ValueError."""
    import pytest
    with pytest.raises(ValueError, match="Invalid date format"):
        _build_s3_key("2026-02")


# ============================================================
# batch_to_parquet_bytes tests
# ============================================================

def test_batch_to_parquet_bytes_basic(sample_market_data_batch):
    """Parquet bytes contain all symbols with prices."""
    parquet_bytes = batch_to_parquet_bytes(sample_market_data_batch)

    # Should produce non-empty bytes
    assert isinstance(parquet_bytes, bytes)
    assert len(parquet_bytes) > 0

    # Read back and verify contents
    table = pq.read_table(io.BytesIO(parquet_bytes))
    assert table.num_rows == 3  # AAPL, MSFT, GOOGL
    assert set(table.column_names) == {
        "symbol", "date", "close", "volume", "spread_bps", "market_cap", "data_source"
    }


def test_batch_to_parquet_bytes_values(sample_market_data_batch):
    """Parquet values match the batch data."""
    parquet_bytes = batch_to_parquet_bytes(sample_market_data_batch)
    table = pq.read_table(io.BytesIO(parquet_bytes))

    # Convert to pandas for easier testing
    df = table.to_pandas()
    aapl_row = df[df["symbol"] == "AAPL"].iloc[0]

    assert aapl_row["close"] == 195.50
    assert aapl_row["volume"] == 50_000_000
    assert aapl_row["spread_bps"] == 5.2
    assert aapl_row["date"] == "2026-01-15"
    assert aapl_row["data_source"] == "yfinance"


def test_batch_to_parquet_bytes_filters_missing_prices():
    """Symbols without price data are excluded from Parquet."""
    batch = MarketDataBatch(
        symbols=["AAPL", "MISSING"],
        date="2026-02-08",
        prices={"AAPL": 280.0},  # MISSING has no price
        volumes={"AAPL": 1_000_000},
        spreads={"AAPL": 5.0},
        market_caps={"AAPL": 3.0e12},
        data_quality_score=0.5,
    )
    parquet_bytes = batch_to_parquet_bytes(batch)
    table = pq.read_table(io.BytesIO(parquet_bytes))

    assert table.num_rows == 1
    assert table.column("symbol").to_pylist() == ["AAPL"]


def test_batch_to_parquet_bytes_empty_batch():
    """Empty batch produces valid Parquet with zero rows."""
    batch = MarketDataBatch(
        symbols=[], date="2026-02-08",
        prices={}, volumes={}, spreads={}, market_caps={},
        data_quality_score=0.0,
    )
    parquet_bytes = batch_to_parquet_bytes(batch)
    table = pq.read_table(io.BytesIO(parquet_bytes))

    assert table.num_rows == 0
    assert "symbol" in table.column_names


def test_batch_to_parquet_bytes_snappy_compression(sample_market_data_batch):
    """Parquet file uses Snappy compression."""
    parquet_bytes = batch_to_parquet_bytes(sample_market_data_batch)
    pf = pq.ParquetFile(io.BytesIO(parquet_bytes))

    # Check that Snappy compression is used
    metadata = pf.schema_arrow
    row_group = pf.metadata.row_group(0)
    col_meta = row_group.column(0)
    assert col_meta.compression == "SNAPPY"


def test_batch_to_parquet_bytes_default_values():
    """Missing volume/spread/market_cap defaults to 0."""
    batch = MarketDataBatch(
        symbols=["TSLA"],
        date="2026-02-08",
        prices={"TSLA": 400.0},
        volumes={},  # No volume data
        spreads={},  # No spread data
        market_caps={},  # No market cap data
        data_quality_score=1.0,
    )
    parquet_bytes = batch_to_parquet_bytes(batch)
    table = pq.read_table(io.BytesIO(parquet_bytes))
    df = table.to_pandas()

    assert df.iloc[0]["volume"] == 0
    assert df.iloc[0]["spread_bps"] == 0.0
    assert df.iloc[0]["market_cap"] == 0.0


# ============================================================
# store_parquet_to_s3 tests (mocked S3)
# ============================================================

def test_store_parquet_to_s3_calls_s3(sample_market_data_batch, mocker):
    """store_parquet_to_s3 calls S3 put_object with correct params."""
    from src.shared.storage import store_parquet_to_s3

    mock_client = mocker.MagicMock()
    mocker.patch("src.shared.storage.get_s3_client", return_value=mock_client)

    result = store_parquet_to_s3(sample_market_data_batch, bucket="test-bucket")

    # Verify S3 put_object was called
    mock_client.put_object.assert_called_once()
    call_kwargs = mock_client.put_object.call_args
    assert call_kwargs[1]["Bucket"] == "test-bucket"
    assert "market_data/source=yfinance/year=2026/month=01/day=15/" in call_kwargs[1]["Key"]
    assert call_kwargs[1]["ContentType"] == "application/octet-stream"

    # Verify the body is valid Parquet bytes
    body_bytes = call_kwargs[1]["Body"]
    table = pq.read_table(io.BytesIO(body_bytes))
    assert table.num_rows == 3

    # Verify result message
    assert "3 rows" in result
    assert "s3://test-bucket/" in result


def test_store_parquet_to_s3_default_bucket(sample_market_data_batch, mocker):
    """store_parquet_to_s3 uses default bucket from config."""
    from src.shared.storage import store_parquet_to_s3

    mock_client = mocker.MagicMock()
    mocker.patch("src.shared.storage.get_s3_client", return_value=mock_client)

    result = store_parquet_to_s3(sample_market_data_batch)

    call_kwargs = mock_client.put_object.call_args
    assert call_kwargs[1]["Bucket"] == "quant-data"
    assert "s3://quant-data/" in result
