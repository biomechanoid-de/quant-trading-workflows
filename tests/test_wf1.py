"""Tests for WF1: Data Ingestion Pipeline.

Tests validate the pure logic tasks (validate_ticks, check_data_quality).
Tasks that require network (fetch_market_data) or database (store_to_database)
are not tested here - those are integration tests.
"""

from src.shared.models import MarketDataBatch
from src.wf1_data_ingestion.tasks import validate_ticks, check_data_quality


# ============================================================
# validate_ticks tests
# ============================================================

def test_validate_ticks_valid_data(sample_market_data_batch):
    """Valid data passes through unchanged."""
    result = validate_ticks(batch=sample_market_data_batch)
    assert len(result.prices) == 3
    assert result.data_quality_score == 1.0
    assert "AAPL" in result.prices
    assert "MSFT" in result.prices
    assert "GOOGL" in result.prices


def test_validate_ticks_filters_negative_prices(incomplete_market_data_batch):
    """Negative prices are filtered out."""
    result = validate_ticks(batch=incomplete_market_data_batch)
    assert "MSFT" not in result.prices
    assert "AAPL" in result.prices


def test_validate_ticks_filters_missing_symbols(incomplete_market_data_batch):
    """Symbols with no price data are counted as issues."""
    result = validate_ticks(batch=incomplete_market_data_batch)
    assert "INVALID" not in result.prices
    assert result.data_quality_score < 1.0


def test_validate_ticks_quality_score():
    """Quality score reflects ratio of valid/total symbols."""
    batch = MarketDataBatch(
        symbols=["A", "B", "C", "D"],
        date="2026-01-15",
        prices={"A": 100.0, "B": 200.0},  # C and D missing
        volumes={"A": 1000, "B": 2000},
        spreads={"A": 5.0, "B": 10.0},
        market_caps={},
        data_quality_score=0.5,
    )
    result = validate_ticks(batch=batch)
    assert result.data_quality_score == 0.5  # 2 out of 4


def test_validate_ticks_unreasonable_spread():
    """Spreads >1000 bps are filtered out."""
    batch = MarketDataBatch(
        symbols=["GOOD", "BAD"],
        date="2026-01-15",
        prices={"GOOD": 100.0, "BAD": 50.0},
        volumes={"GOOD": 1000, "BAD": 500},
        spreads={"GOOD": 5.0, "BAD": 1500.0},
        market_caps={},
        data_quality_score=1.0,
    )
    result = validate_ticks(batch=batch)
    assert "GOOD" in result.prices
    assert "BAD" not in result.prices
    assert result.data_quality_score == 0.5


def test_validate_ticks_zero_price():
    """Zero prices are filtered out."""
    batch = MarketDataBatch(
        symbols=["ZERO"],
        date="2026-01-15",
        prices={"ZERO": 0.0},
        volumes={"ZERO": 1000},
        spreads={"ZERO": 5.0},
        market_caps={},
        data_quality_score=1.0,
    )
    result = validate_ticks(batch=batch)
    assert "ZERO" not in result.prices
    assert result.data_quality_score == 0.0


def test_validate_ticks_empty_batch():
    """Empty batch returns 0.0 quality score."""
    batch = MarketDataBatch(
        symbols=[], date="2026-01-15",
        prices={}, volumes={}, spreads={}, market_caps={},
        data_quality_score=0.0,
    )
    result = validate_ticks(batch=batch)
    assert len(result.prices) == 0
    assert result.data_quality_score == 0.0


# ============================================================
# check_data_quality tests
# ============================================================

def test_check_data_quality_perfect(sample_market_data_batch):
    """Perfect data generates clean quality report."""
    report = check_data_quality(batch=sample_market_data_batch)
    assert "Quality score:     100.00%" in report
    assert "Missing symbols" not in report
    assert "Data Quality Report" in report


def test_check_data_quality_with_missing():
    """Missing symbols are listed in the quality report."""
    batch = MarketDataBatch(
        symbols=["AAPL", "MISSING1", "MISSING2"],
        date="2026-01-15",
        prices={"AAPL": 195.50},
        volumes={"AAPL": 50_000_000},
        spreads={"AAPL": 5.2},
        market_caps={"AAPL": 3.0e12},
        data_quality_score=0.3333,
    )
    report = check_data_quality(batch=batch)
    assert "Missing symbols:   MISSING1, MISSING2" in report
    assert "Symbols requested: 3" in report
    assert "Symbols fetched:   1" in report


def test_check_data_quality_zero_volume():
    """Zero-volume symbols are flagged in the report."""
    batch = MarketDataBatch(
        symbols=["AAPL"],
        date="2026-01-15",
        prices={"AAPL": 195.50},
        volumes={"AAPL": 0},
        spreads={"AAPL": 5.2},
        market_caps={"AAPL": 3.0e12},
        data_quality_score=1.0,
    )
    report = check_data_quality(batch=batch)
    assert "Zero volume:       AAPL" in report
