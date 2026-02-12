"""Tests for data providers.

Tests validate the interface contract and behavior.
No network access required (yfinance fetch_eod/fundamentals are not tested
with real API calls here — only return format / error handling).
"""

from unittest.mock import patch, MagicMock

from src.shared.providers.base import DataProvider
from src.shared.providers.yfinance_provider import YFinanceProvider


def test_yfinance_provider_is_data_provider():
    """YFinanceProvider must implement DataProvider ABC."""
    provider = YFinanceProvider()
    assert isinstance(provider, DataProvider)


def test_yfinance_provider_fetch_fundamentals_returns_dict():
    """fetch_fundamentals returns a dict with expected keys."""
    provider = YFinanceProvider()
    # Mock yfinance (lazy import inside method) to avoid network calls
    mock_ticker = MagicMock()
    mock_ticker.info = {
        "trailingPE": 28.5,
        "forwardPE": 25.0,
        "priceToBook": 45.0,
        "dividendYield": 0.005,
        "returnOnEquity": 1.60,
        "debtToEquity": 150.0,  # yfinance returns as percentage
        "currentRatio": 1.07,
        "trailingEps": 6.50,
        "sector": "Technology",
        "industry": "Consumer Electronics",
    }

    mock_yf = MagicMock()
    mock_yf.Ticker.return_value = mock_ticker

    with patch.dict("sys.modules", {"yfinance": mock_yf}):
        result = provider.fetch_fundamentals("AAPL")

    assert result["symbol"] == "AAPL"
    assert result["pe_ratio"] == 28.5
    assert result["debt_to_equity"] == 1.5  # 150 / 100
    assert result["sector"] == "Technology"


def test_yfinance_provider_fetch_fundamentals_error_returns_defaults():
    """fetch_fundamentals returns defaults on exception."""
    provider = YFinanceProvider()

    mock_yf = MagicMock()
    mock_yf.Ticker.side_effect = Exception("Network error")

    with patch.dict("sys.modules", {"yfinance": mock_yf}):
        result = provider.fetch_fundamentals("INVALID")

    assert result["symbol"] == "INVALID"
    assert result["pe_ratio"] == -1.0
    assert result["dividend_yield"] == 0.0


def test_yfinance_provider_fetch_dividends_stub():
    """fetch_dividends returns empty list stub in Phase 1."""
    provider = YFinanceProvider()
    result = provider.fetch_dividends("AAPL")
    assert result == []
