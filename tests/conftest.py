"""Shared test fixtures for Quant Trading Workflows.

Provides reusable MarketDataBatch instances for testing
validation logic and quality checks without network access.
"""

import pytest

from src.shared.models import MarketDataBatch


@pytest.fixture
def sample_market_data_batch():
    """A valid MarketDataBatch with 3 symbols - all data present."""
    return MarketDataBatch(
        symbols=["AAPL", "MSFT", "GOOGL"],
        date="2026-01-15",
        prices={"AAPL": 195.50, "MSFT": 415.20, "GOOGL": 175.80},
        volumes={"AAPL": 50_000_000, "MSFT": 25_000_000, "GOOGL": 30_000_000},
        spreads={"AAPL": 5.2, "MSFT": 3.8, "GOOGL": 4.5},
        market_caps={"AAPL": 3.0e12, "MSFT": 3.1e12, "GOOGL": 2.2e12},
        data_quality_score=1.0,
    )


@pytest.fixture
def incomplete_market_data_batch():
    """A MarketDataBatch with validation issues for testing.

    Issues:
    - MSFT has negative price (should be filtered)
    - MSFT has unreasonable spread >1000 bps (should be filtered)
    - INVALID symbol has no price data (missing)
    """
    return MarketDataBatch(
        symbols=["AAPL", "MSFT", "INVALID"],
        date="2026-01-15",
        prices={"AAPL": 195.50, "MSFT": -10.0},
        volumes={"AAPL": 50_000_000, "MSFT": 0},
        spreads={"AAPL": 5.2, "MSFT": 1500.0},
        market_caps={"AAPL": 3.0e12},
        data_quality_score=0.67,
    )
