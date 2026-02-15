"""Shared data models for Quant Trading Workflows.

All models are plain Python dataclasses. Flytekit serializes them natively
as task inputs/outputs. No special Flyte type registration needed.

IMPORTANT (Flytekit constraint):
    Never pass dataclasses with Dict/List fields between Flyte tasks —
    causes Promise binding errors. Use Dict[str, str] with JSON serialization
    for complex inter-task data. Dataclasses with only primitive fields
    (str, int, float, bool) are safe in List[Dataclass].
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


# ============================================================
# WF3: Signal & Analysis
# ============================================================

@dataclass
class SentimentSignals:
    """Sentiment analysis signals for a single stock.

    Used by WF3: Signal & Analysis Pipeline (Phase 6).
    Only primitive fields (str, float, int, bool) — safe in
    List[SentimentSignals] between Flyte tasks.

    Missing data (zero articles) is signaled by num_articles=0
    and sentiment_score=50.0 (neutral default).
    """
    symbol: str
    # Article counts
    num_articles: int              # Total articles analyzed
    num_positive: int              # Articles classified positive
    num_neutral: int               # Articles classified neutral
    num_negative: int              # Articles classified negative
    # Provider
    news_provider: str             # "finnhub", "marketaux", or "none"
    # Composite
    sentiment_score: float         # 0-100, time-decay-weighted
    sentiment_signal: str          # "very_positive"/"positive"/"neutral"/"negative"/"very_negative"
    # Data quality
    has_sentiment: bool = True     # False if zero articles found


@dataclass
class TechnicalSignals:
    """Technical indicator signals for a single stock.

    Used by WF3: Signal & Analysis Pipeline.
    Only primitive fields (str, float) — safe in List[TechnicalSignals]
    between Flyte tasks.

    All indicators are close-based (no OHLC needed).
    ATR deferred until WF1 stores OHLC data.
    """
    symbol: str
    # SMA
    sma_50: float                     # 50-day SMA value
    sma_200: float                    # 200-day SMA value
    sma_crossover_signal: str         # "bullish", "bearish", "neutral"
    # MACD
    macd_line: float                  # MACD line value
    macd_signal_line: float           # Signal line value
    macd_histogram: float             # Histogram value
    macd_signal: str                  # "bullish", "bearish", "neutral"
    # Bollinger Bands
    bb_upper: float                   # Upper band
    bb_middle: float                  # Middle band (SMA 20)
    bb_lower: float                   # Lower band
    bb_signal: str                    # "oversold", "overbought", "neutral"
    # Composite
    technical_score: float            # 0-100 weighted tech score


@dataclass
class FundamentalSignals:
    """Fundamental analysis signals for a single stock.

    Used by WF3: Signal & Analysis Pipeline.
    Only primitive fields — safe in List[FundamentalSignals].

    Missing data is signaled by -1.0 (ratios) or 0.0 (yields).
    has_* flags indicate whether the data was available from yfinance.
    """
    symbol: str
    # Valuation
    pe_ratio: float                   # Trailing P/E (-1.0 = missing)
    pe_zscore: float                  # Normalized P/E z-score
    forward_pe: float                 # Forward P/E (-1.0 = missing)
    price_to_book: float              # P/B ratio (-1.0 = missing)
    # Income
    dividend_yield: float             # Dividend yield (0.0 = none)
    # Efficiency
    return_on_equity: float           # ROE as decimal (-1.0 = missing)
    # Leverage
    debt_to_equity: float             # D/E ratio (-1.0 = missing)
    current_ratio: float              # Current ratio (-1.0 = missing)
    # Data completeness flags
    has_pe: bool = True
    has_roe: bool = True
    has_debt: bool = True
    # Composite
    fundamental_score: float = 50.0   # 0-100 weighted fundamental score
    fundamental_signal: str = "balanced"  # "value", "growth", "balanced"


@dataclass
class SignalResult:
    """Combined signal result for a single stock.

    Used by WF3: Signal & Analysis Pipeline.
    Only primitive fields — safe in List[SignalResult].
    """
    symbol: str
    run_date: str
    # WF2 context
    wf2_composite_score: float        # From screening_results
    wf2_quintile: int                 # 1 (best) to 5 (worst)
    # Technical
    technical_score: float            # 0-100
    technical_signal: str             # "bullish", "bearish", "neutral"
    # Fundamental
    fundamental_score: float          # 0-100
    fundamental_signal: str           # "value", "growth", "balanced"
    # Sentiment (Phase 6 — defaults for backward compatibility)
    sentiment_score: float = 50.0     # 0-100 (50 = neutral default)
    sentiment_signal: str = "neutral" # "very_positive"/.../""very_negative"
    num_articles: int = 0             # Total articles analyzed
    news_provider: str = "none"       # "finnhub", "marketaux", "none"
    # Combined
    combined_signal_score: float = 50.0  # 0-100 (weighted tech + fund + sent)
    signal_strength: str = "hold"     # "strong_buy"/"buy"/"hold"/"sell"/"strong_sell"
    # Quality
    data_quality: str = "complete"    # "complete", "partial", "minimal"


@dataclass
class SignalAnalysisResult:
    """Complete output of a WF3 signal analysis run.

    Contains all per-stock signal results and run metadata.
    Note: List[SignalResult] field means this dataclass should NOT be
    passed between Flyte tasks. Only used within assemble_signal_result
    and for final reporting/storage.
    """
    run_date: str
    signal_results: List[SignalResult]
    # Metadata
    num_symbols_analyzed: int
    num_with_complete_data: int
    num_with_partial_data: int
    # Top signals (comma-separated strings for Flytekit safety)
    top_buy_signals: str = ""         # e.g., "AAPL,MSFT,NVDA"
    top_sell_signals: str = ""        # e.g., "INTC,NKE"
