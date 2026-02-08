"""Shared test fixtures for Quant Trading Workflows.

Provides reusable test data for WF1 (MarketDataBatch) and
WF2 (ScreeningConfig, StockMetrics, price_data) without
network or database access.
"""

import pytest

from src.shared.models import MarketDataBatch, ScreeningConfig, StockMetrics


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


# ============================================================
# WF2: Universe & Screening Fixtures
# ============================================================

@pytest.fixture
def sample_screening_config():
    """Default ScreeningConfig for testing with short windows."""
    return ScreeningConfig(
        symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
        lookback_days=60,
        forecast_horizon=5,
        momentum_windows=[5, 10, 21],
        rsi_window=14,
        rsi_oversold=30,
        rsi_overbought=70,
        kmeans_max_k=5,
        factor_weights={
            "momentum": 0.30,
            "low_volatility": 0.25,
            "rsi_signal": 0.20,
            "sharpe": 0.25,
        },
    )


@pytest.fixture
def sample_price_data():
    """Synthetic price data dict for 5 symbols, 40 trading days.

    Generates realistic-looking prices starting from different bases
    with small daily variations. Values are JSON-serialized lists of
    [date, price] pairs — matching the Flytekit-compatible Dict[str, str]
    format used by load_historical_prices.
    """
    import json
    import numpy as np

    np.random.seed(42)
    base_prices = {"AAPL": 190.0, "MSFT": 410.0, "GOOGL": 170.0, "AMZN": 180.0, "NVDA": 800.0}
    n_days = 40

    from datetime import datetime, timedelta

    start_date = datetime(2026, 1, 5)
    dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n_days)]

    result = {}
    for symbol, base in base_prices.items():
        # Random walk with slight upward drift
        returns = np.random.normal(0.001, 0.015, n_days)
        prices = [base]
        for r in returns[1:]:
            prices.append(prices[-1] * (1 + r))
        pairs = [[dates[i], round(prices[i], 2)] for i in range(n_days)]
        result[symbol] = json.dumps(pairs)

    return result


@pytest.fixture
def sample_stock_metrics():
    """List of 5 StockMetrics with realistic values for testing scoring/clustering."""
    return [
        StockMetrics(
            symbol="AAPL", forward_return=0.05, momentum_returns={"5d": 0.02, "10d": 0.04, "21d": 0.08},
            rsi=55.0, rsi_signal="neutral", volatility_252d=0.22,
            cagr=0.15, sharpe=1.2, sortino=1.8, calmar=1.5, max_drawdown=-0.10,
            z_scores={}, composite_score=0.0, quintile=3,
        ),
        StockMetrics(
            symbol="MSFT", forward_return=0.03, momentum_returns={"5d": 0.01, "10d": 0.03, "21d": 0.06},
            rsi=62.0, rsi_signal="neutral", volatility_252d=0.20,
            cagr=0.12, sharpe=1.0, sortino=1.5, calmar=1.2, max_drawdown=-0.10,
            z_scores={}, composite_score=0.0, quintile=3,
        ),
        StockMetrics(
            symbol="GOOGL", forward_return=-0.02, momentum_returns={"5d": -0.01, "10d": -0.03, "21d": -0.05},
            rsi=28.0, rsi_signal="oversold", volatility_252d=0.25,
            cagr=-0.05, sharpe=-0.3, sortino=-0.2, calmar=0.2, max_drawdown=-0.25,
            z_scores={}, composite_score=0.0, quintile=3,
        ),
        StockMetrics(
            symbol="AMZN", forward_return=0.08, momentum_returns={"5d": 0.03, "10d": 0.06, "21d": 0.12},
            rsi=72.0, rsi_signal="overbought", volatility_252d=0.30,
            cagr=0.25, sharpe=1.5, sortino=2.0, calmar=2.0, max_drawdown=-0.12,
            z_scores={}, composite_score=0.0, quintile=3,
        ),
        StockMetrics(
            symbol="NVDA", forward_return=0.10, momentum_returns={"5d": 0.04, "10d": 0.08, "21d": 0.15},
            rsi=65.0, rsi_signal="neutral", volatility_252d=0.35,
            cagr=0.30, sharpe=1.8, sortino=2.5, calmar=2.5, max_drawdown=-0.12,
            z_scores={}, composite_score=0.0, quintile=3,
        ),
    ]
