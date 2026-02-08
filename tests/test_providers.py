"""Tests for data providers.

Tests validate the interface contract and stub behavior.
No network access required (yfinance fetch_eod is not tested here).
"""

from src.shared.providers.base import DataProvider
from src.shared.providers.yfinance_provider import YFinanceProvider


def test_yfinance_provider_is_data_provider():
    """YFinanceProvider must implement DataProvider ABC."""
    provider = YFinanceProvider()
    assert isinstance(provider, DataProvider)


def test_yfinance_provider_fetch_fundamentals_stub():
    """fetch_fundamentals returns not_implemented stub in Phase 1."""
    provider = YFinanceProvider()
    result = provider.fetch_fundamentals("AAPL")
    assert result["status"] == "not_implemented"
    assert result["phase"] == 2


def test_yfinance_provider_fetch_dividends_stub():
    """fetch_dividends returns empty list stub in Phase 1."""
    provider = YFinanceProvider()
    result = provider.fetch_dividends("AAPL")
    assert result == []
