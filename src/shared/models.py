"""Shared data models for Quant Trading Workflows.

All models are plain Python dataclasses. Flytekit serializes them natively
as task inputs/outputs. No special Flyte type registration needed.
"""

from dataclasses import dataclass, field
from typing import Dict, List


# ============================================================
# WF1: Data Ingestion
# ============================================================

@dataclass
class MarketDataBatch:
    """Market data for a batch of symbols on a given date.

    Used by WF1: Data Ingestion Pipeline.
    Passed through: fetch -> validate -> store -> quality_check.
    """
    symbols: List[str]                 # Requested symbols
    date: str                          # Target date (YYYY-MM-DD)
    prices: Dict[str, float]           # Symbol -> Close Price
    volumes: Dict[str, int]            # Symbol -> Volume
    spreads: Dict[str, float]          # Symbol -> Bid-Ask Spread (Bps)
    market_caps: Dict[str, float]      # Symbol -> Market Cap (USD)
    data_quality_score: float          # 0.0 - 1.0


# ============================================================
# WF2: Universe & Screening
# ============================================================

@dataclass
class ScreeningConfig:
    """Configuration for a screening run.

    Passed as input to universe_screening_workflow.
    All parameters have defaults from config.py / environment variables.
    """
    symbols: List[str] = field(default_factory=list)
    lookback_days: int = 252
    forecast_horizon: int = 21                  # Forward return horizon (trading days)
    momentum_windows: List[int] = field(default_factory=lambda: [10, 21, 63, 126, 252])
    rsi_window: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    kmeans_max_k: int = 10
    factor_weights: Dict[str, float] = field(default_factory=lambda: {
        "momentum": 0.30,
        "low_volatility": 0.25,
        "rsi_signal": 0.20,
        "sharpe": 0.25,
    })


@dataclass
class StockMetrics:
    """Computed metrics for a single stock from screening.

    One instance per symbol per screening run.
    """
    symbol: str
    # Returns
    forward_return: float                       # Forward N-day return
    momentum_returns: Dict[str, float]          # e.g. {"10d": 0.05, "21d": 0.08}
    # Technical
    rsi: float                                  # RSI value (0-100)
    rsi_signal: str                             # "oversold", "neutral", "overbought"
    volatility_252d: float                      # Annualized volatility
    # Performance
    cagr: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    # Factor scoring (Brenndoerfer)
    z_scores: Dict[str, float]                  # Factor name -> Z-score
    composite_score: float                      # Weighted sum of Z-scores
    quintile: int                               # 1 (best) to 5 (worst)
    # Clustering
    cluster_id: int = -1                        # K-Means cluster assignment
    cluster_label: str = ""                     # e.g. "HiMom-LoVol"


@dataclass
class ScreeningResult:
    """Complete output of a screening run.

    Contains all stock metrics, benchmark performance, and metadata.
    """
    run_date: str                               # YYYY-MM-DD
    config: ScreeningConfig
    stock_metrics: List[StockMetrics]           # Ordered by composite_score desc
    # Benchmark (equal-weight universe)
    benchmark_cagr: float
    benchmark_sharpe: float
    benchmark_cumulative_return: float
    # Metadata
    num_symbols_input: int
    num_symbols_with_data: int
    optimal_k_clusters: int
    data_quality_notes: List[str] = field(default_factory=list)


# ============================================================
# WF4: Portfolio & Rebalancing
# ============================================================

@dataclass
class Position:
    """A single portfolio position.

    Used by WF4: Portfolio & Rebalancing.
    """
    symbol: str
    shares: float
    avg_cost: float
    current_price: float
    region: str = ""
    sector: str = ""


@dataclass
class TradeOrder:
    """A proposed trade order (not executed automatically).

    Used by WF4: Portfolio & Rebalancing.
    The system generates order reports, not live trades.
    """
    symbol: str
    side: str                          # "BUY" or "SELL"
    quantity: int
    estimated_price: float
    estimated_cost_bps: float          # Spread + Commission + Impact
    reason: str                        # "Rebalance", "NewEntry", "Exit"


# ============================================================
# WF5: Monitoring & Reporting
# ============================================================

@dataclass
class DailyReport:
    """Daily monitoring and P&L report.

    Used by WF5: Monitoring & Reporting.
    """
    date: str
    portfolio_value: float
    daily_pnl: float
    daily_pnl_pct: float
    mtd_pnl: float                     # Month-to-Date
    ytd_pnl: float                     # Year-to-Date
    sharpe_ratio_30d: float
    max_drawdown_30d: float
    var_95: float                      # Value at Risk (95%)
    top_winners: List[str] = field(default_factory=list)
    top_losers: List[str] = field(default_factory=list)
    upcoming_dividends: List[str] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)
