"""Shared test fixtures for Quant Trading Workflows.

Provides reusable test data for WF1 (MarketDataBatch),
WF2 (ScreeningConfig, StockMetrics, price_data),
WF3 (TechnicalSignals, FundamentalSignals, screening_context),
WF4 (signal_context, portfolio_state, price_data for rebalancing), and
WF5 (portfolio_snapshots, positions, pnl_data, risk_data, alert_data)
without network or database access.
"""

import pytest

from src.shared.models import (
    MarketDataBatch, ScreeningConfig, StockMetrics,
    TechnicalSignals, FundamentalSignals, SignalResult,
)


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


# ============================================================
# WF3: Signal & Analysis Fixtures
# ============================================================

@pytest.fixture
def sample_screening_context():
    """Dict[str, str] mimicking load_screening_context output.

    5 symbols with WF2 composite scores and quintiles.
    """
    import json
    return {
        "AAPL": json.dumps({"composite_score": 1.85, "quintile": 1}),
        "MSFT": json.dumps({"composite_score": 1.42, "quintile": 1}),
        "NVDA": json.dumps({"composite_score": 1.10, "quintile": 1}),
        "AMZN": json.dumps({"composite_score": 0.75, "quintile": 2}),
        "JPM": json.dumps({"composite_score": 0.55, "quintile": 2}),
    }


@pytest.fixture
def sample_technical_signals():
    """List of 5 TechnicalSignals with realistic values."""
    return [
        TechnicalSignals(
            symbol="AAPL", sma_50=192.50, sma_200=185.30,
            sma_crossover_signal="bullish",
            macd_line=1.25, macd_signal_line=0.80, macd_histogram=0.45,
            macd_signal="bullish",
            bb_upper=198.50, bb_middle=192.50, bb_lower=186.50,
            bb_signal="neutral",
            technical_score=68.0,
        ),
        TechnicalSignals(
            symbol="MSFT", sma_50=410.20, sma_200=395.10,
            sma_crossover_signal="bullish",
            macd_line=2.10, macd_signal_line=1.50, macd_histogram=0.60,
            macd_signal="bullish",
            bb_upper=420.00, bb_middle=410.20, bb_lower=400.40,
            bb_signal="neutral",
            technical_score=68.0,
        ),
        TechnicalSignals(
            symbol="NVDA", sma_50=820.00, sma_200=750.00,
            sma_crossover_signal="bullish",
            macd_line=5.50, macd_signal_line=4.20, macd_histogram=1.30,
            macd_signal="bullish",
            bb_upper=860.00, bb_middle=820.00, bb_lower=780.00,
            bb_signal="neutral",
            technical_score=68.0,
        ),
        TechnicalSignals(
            symbol="AMZN", sma_50=178.00, sma_200=180.50,
            sma_crossover_signal="bearish",
            macd_line=-0.50, macd_signal_line=0.10, macd_histogram=-0.60,
            macd_signal="bearish",
            bb_upper=185.00, bb_middle=178.00, bb_lower=171.00,
            bb_signal="neutral",
            technical_score=32.0,
        ),
        TechnicalSignals(
            symbol="JPM", sma_50=195.00, sma_200=190.00,
            sma_crossover_signal="bullish",
            macd_line=0.30, macd_signal_line=0.25, macd_histogram=0.05,
            macd_signal="neutral",
            bb_upper=200.00, bb_middle=195.00, bb_lower=190.00,
            bb_signal="neutral",
            technical_score=56.5,
        ),
    ]


@pytest.fixture
def sample_fundamental_signals():
    """List of 5 FundamentalSignals with realistic values."""
    return [
        FundamentalSignals(
            symbol="AAPL", pe_ratio=28.5, pe_zscore=-0.425,
            forward_pe=25.0, price_to_book=45.0,
            dividend_yield=0.005, return_on_equity=1.60,
            debt_to_equity=1.50, current_ratio=1.07,
            has_pe=True, has_roe=True, has_debt=True,
            fundamental_score=55.0, fundamental_signal="balanced",
        ),
        FundamentalSignals(
            symbol="MSFT", pe_ratio=32.0, pe_zscore=-0.60,
            forward_pe=28.0, price_to_book=12.0,
            dividend_yield=0.008, return_on_equity=0.38,
            debt_to_equity=0.35, current_ratio=1.77,
            has_pe=True, has_roe=True, has_debt=True,
            fundamental_score=62.0, fundamental_signal="balanced",
        ),
        FundamentalSignals(
            symbol="NVDA", pe_ratio=60.0, pe_zscore=-2.0,
            forward_pe=35.0, price_to_book=50.0,
            dividend_yield=0.0003, return_on_equity=1.15,
            debt_to_equity=0.41, current_ratio=4.17,
            has_pe=True, has_roe=True, has_debt=True,
            fundamental_score=45.0, fundamental_signal="growth",
        ),
        FundamentalSignals(
            symbol="AMZN", pe_ratio=55.0, pe_zscore=-1.75,
            forward_pe=40.0, price_to_book=8.0,
            dividend_yield=0.0, return_on_equity=0.22,
            debt_to_equity=0.60, current_ratio=1.05,
            has_pe=True, has_roe=True, has_debt=True,
            fundamental_score=40.0, fundamental_signal="growth",
        ),
        FundamentalSignals(
            symbol="JPM", pe_ratio=12.0, pe_zscore=0.40,
            forward_pe=11.5, price_to_book=1.8,
            dividend_yield=0.025, return_on_equity=0.15,
            debt_to_equity=1.20, current_ratio=-1.0,
            has_pe=True, has_roe=True, has_debt=True,
            fundamental_score=58.0, fundamental_signal="value",
        ),
    ]


# ============================================================
# WF4: Portfolio & Rebalancing Fixtures
# ============================================================

@pytest.fixture
def sample_wf4_signal_context():
    """Dict[str, str] mimicking WF4 load_signal_context output.

    5 symbols with mixed signal strengths for portfolio construction testing.
    """
    import json
    return {
        "AAPL": json.dumps({
            "combined_signal_score": 72.5, "signal_strength": "buy",
            "wf2_quintile": 1, "technical_score": 68.0,
            "fundamental_score": 77.0, "data_quality": "complete",
            "sentiment_score": 65.0, "sentiment_signal": "positive",
        }),
        "MSFT": json.dumps({
            "combined_signal_score": 76.0, "signal_strength": "strong_buy",
            "wf2_quintile": 1, "technical_score": 72.0,
            "fundamental_score": 80.0, "data_quality": "complete",
            "sentiment_score": 70.0, "sentiment_signal": "positive",
        }),
        "NVDA": json.dumps({
            "combined_signal_score": 65.5, "signal_strength": "buy",
            "wf2_quintile": 1, "technical_score": 70.0,
            "fundamental_score": 61.0, "data_quality": "complete",
            "sentiment_score": 55.0, "sentiment_signal": "neutral",
        }),
        "AMZN": json.dumps({
            "combined_signal_score": 45.0, "signal_strength": "hold",
            "wf2_quintile": 2, "technical_score": 40.0,
            "fundamental_score": 50.0, "data_quality": "partial",
            "sentiment_score": 40.0, "sentiment_signal": "negative",
        }),
        "JPM": json.dumps({
            "combined_signal_score": 62.0, "signal_strength": "buy",
            "wf2_quintile": 2, "technical_score": 58.0,
            "fundamental_score": 66.0, "data_quality": "complete",
            "sentiment_score": 60.0, "sentiment_signal": "positive",
        }),
    }


@pytest.fixture
def sample_portfolio_state_empty():
    """Empty portfolio (initial state with EUR 25,000 cash)."""
    import json
    return {
        "cash": "25000.0",
        "positions_json": json.dumps([]),
        "total_value": "25000.0",
    }


@pytest.fixture
def sample_portfolio_state_with_positions():
    """Portfolio with existing positions."""
    import json
    return {
        "cash": "5000.0",
        "positions_json": json.dumps([
            {"symbol": "AAPL", "shares": 5, "avg_cost": 180.0, "current_price": 195.0, "sector": "Technology"},
            {"symbol": "MSFT", "shares": 3, "avg_cost": 400.0, "current_price": 415.0, "sector": "Technology"},
        ]),
        "total_value": "7220.0",
    }


@pytest.fixture
def sample_wf4_price_data():
    """Price data for WF4 trade generation."""
    import json
    return {
        "AAPL": json.dumps({"close": 195.50, "spread_bps": 5.2}),
        "MSFT": json.dumps({"close": 415.20, "spread_bps": 3.8}),
        "NVDA": json.dumps({"close": 850.00, "spread_bps": 4.5}),
        "JPM": json.dumps({"close": 195.00, "spread_bps": 6.0}),
        "AMZN": json.dumps({"close": 185.00, "spread_bps": 4.0}),
    }


# ============================================================
# WF4 Phase 4: Paper Trading Fixtures
# ============================================================

@pytest.fixture
def sample_assembled_result_for_paper_trading():
    """Assembled result with trade orders for paper trading tests."""
    import json
    return {
        "run_date": "2026-02-16",
        "total_value": "25000.0",
        "cash_value": "25000.0",
        "invested_value": "23750.0",
        "num_signals_input": "5",
        "num_target_positions": "4",
        "num_buy_orders": "3",
        "num_sell_orders": "0",
        "total_estimated_cost_bps": "15.0",
        "target_weights_json": json.dumps({
            "MSFT": 0.05, "AAPL": 0.047, "NVDA": 0.035,
        }),
        "exit_symbols_json": "[]",
        "trade_orders_json": json.dumps([
            {
                "symbol": "MSFT", "side": "BUY", "quantity": 2,
                "estimated_price": 415.0, "estimated_cost_bps": 5.5,
                "reason": "NewEntry",
            },
            {
                "symbol": "AAPL", "side": "BUY", "quantity": 6,
                "estimated_price": 195.0, "estimated_cost_bps": 5.0,
                "reason": "NewEntry",
            },
            {
                "symbol": "NVDA", "side": "BUY", "quantity": 1,
                "estimated_price": 850.0, "estimated_cost_bps": 4.5,
                "reason": "NewEntry",
            },
        ]),
        "signal_context_json": "{}",
    }


@pytest.fixture
def sample_paper_trade_result_executed():
    """Paper trade result after successful execution."""
    import json
    return {
        "status": "executed",
        "num_trades_executed": "3",
        "cash_after": "22069.50",
        "positions_after_json": json.dumps([
            {
                "symbol": "AAPL", "shares": 6, "avg_cost": 195.5,
                "current_price": 195.5, "sector": "Technology",
            },
            {
                "symbol": "MSFT", "shares": 2, "avg_cost": 415.2,
                "current_price": 415.2, "sector": "Technology",
            },
            {
                "symbol": "NVDA", "shares": 1, "avg_cost": 850.0,
                "current_price": 850.0, "sector": "Technology",
            },
        ]),
        "total_value_after": "24969.50",
        "run_date": "2026-02-16",
    }


# ============================================================
# WF5: Monitoring & Reporting Fixtures
# ============================================================

@pytest.fixture
def sample_portfolio_snapshots():
    """10 weekly portfolio snapshots for WF5 testing.

    Simulates a portfolio growing from 25000 to ~25400 over 10 weeks
    with realistic small daily P&L variations.
    """
    from datetime import date, timedelta
    base = 25000.0
    start = date(2026, 1, 5)
    snapshots = []
    for i in range(10):
        d = start + timedelta(weeks=i)
        value = base + i * 50 - (10 if i % 3 == 0 else 0)
        cash = 5000.0
        invested = value - cash
        daily_pnl = 50.0 if i > 0 else 0.0
        if i % 3 == 0 and i > 0:
            daily_pnl = -60.0
        snapshots.append((
            d, value, cash, invested, daily_pnl, 0.0, 5
        ))
    return snapshots


@pytest.fixture
def sample_positions_with_market_data():
    """5 positions with latest market data for WF5 testing."""
    return [
        ("AAPL", 6.0, 190.0, 195.0, "Technology", 196.50),
        ("MSFT", 3.0, 400.0, 415.0, "Technology", 418.00),
        ("NVDA", 1.0, 800.0, 850.0, "Technology", 855.00),
        ("JPM", 5.0, 190.0, 195.0, "Financials", 197.00),
        ("PG", 8.0, 150.0, 148.0, "Consumer Staples", 147.00),
    ]


@pytest.fixture
def sample_wf5_pnl_data():
    """Dict[str, str] output from calculate_pnl for downstream task tests."""
    import json
    return {
        "run_date": "2026-02-16",
        "portfolio_value": "25400.0",
        "cash": "5000.0",
        "invested": "20400.0",
        "daily_pnl": "50.0",
        "daily_pnl_pct": "0.1974",
        "mtd_pnl": "200.0",
        "ytd_pnl": "400.0",
        "num_positions": "5",
        "top_winners": "NVDA (+55.00)|MSFT (+54.00)|AAPL (+39.00)",
        "top_losers": "PG (-24.00)",
        "positions_pnl_json": json.dumps({
            "AAPL": {"shares": 6, "avg_cost": 190.0, "current_price": 196.5,
                     "unrealized_pnl": 39.0, "position_value": 1179.0, "sector": "Technology"},
            "MSFT": {"shares": 3, "avg_cost": 400.0, "current_price": 418.0,
                     "unrealized_pnl": 54.0, "position_value": 1254.0, "sector": "Technology"},
            "NVDA": {"shares": 1, "avg_cost": 800.0, "current_price": 855.0,
                     "unrealized_pnl": 55.0, "position_value": 855.0, "sector": "Technology"},
            "JPM": {"shares": 5, "avg_cost": 190.0, "current_price": 197.0,
                    "unrealized_pnl": 35.0, "position_value": 985.0, "sector": "Financials"},
            "PG": {"shares": 8, "avg_cost": 150.0, "current_price": 147.0,
                   "unrealized_pnl": -24.0, "position_value": 1176.0, "sector": "Consumer Staples"},
        }),
        "snapshots_json": json.dumps([
            {"date": "2026-01-05", "total_value": 25000.0},
            {"date": "2026-01-12", "total_value": 25050.0},
            {"date": "2026-01-19", "total_value": 25100.0},
            {"date": "2026-01-26", "total_value": 25090.0},
            {"date": "2026-02-02", "total_value": 25200.0},
            {"date": "2026-02-09", "total_value": 25250.0},
            {"date": "2026-02-16", "total_value": 25400.0},
        ]),
        "no_data": "false",
    }


@pytest.fixture
def sample_wf5_risk_data():
    """Dict[str, str] output from compute_risk_metrics for downstream tests."""
    import json
    return {
        "sharpe_30d": "1.5",
        "sortino_30d": "2.1",
        "max_drawdown_30d": "-0.004",
        "var_95": "-45.0",
        "sector_concentration_json": json.dumps({
            "Technology": 0.6015,
            "Financials": 0.1803,
            "Consumer Staples": 0.2153,
        }),
        "largest_sector": "Technology",
        "largest_sector_pct": "0.6015",
        "data_points": "6",
    }


@pytest.fixture
def sample_wf5_alert_data_no_alerts():
    """Dict[str, str] with zero alerts."""
    return {
        "num_alerts": "0",
        "alerts_csv": "",
        "has_critical": "false",
    }


@pytest.fixture
def sample_wf5_alert_data_with_alerts():
    """Dict[str, str] with multiple alerts triggered."""
    return {
        "num_alerts": "2",
        "alerts_csv": (
            "DRAWDOWN: 30d max drawdown is 6.0% (threshold: 5%)"
            "|LOSS: NKE has -12.0% unrealized loss (threshold: -10%)"
        ),
        "has_critical": "true",
    }


@pytest.fixture
def sample_wf5_pnl_data_no_data():
    """Dict[str, str] when no portfolio snapshots exist."""
    return {
        "run_date": "2026-02-16",
        "portfolio_value": "0.0",
        "cash": "0.0",
        "invested": "0.0",
        "daily_pnl": "0.0",
        "daily_pnl_pct": "0.0",
        "mtd_pnl": "0.0",
        "ytd_pnl": "0.0",
        "num_positions": "0",
        "top_winners": "",
        "top_losers": "",
        "positions_pnl_json": "{}",
        "snapshots_json": "[]",
        "no_data": "true",
    }
