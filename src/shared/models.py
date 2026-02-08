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
