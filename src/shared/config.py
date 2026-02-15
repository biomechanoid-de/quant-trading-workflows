"""Configuration for Quant Trading Workflows.

All settings are read from environment variables with sensible defaults
for the Pi Cluster setup. Override via env vars for different environments.
"""

import os

# ============================================================
# Stock Symbols
# ============================================================

# Phase 1: 10 US Large Caps for initial testing
PHASE1_SYMBOLS = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "GOOGL",  # Alphabet
    "AMZN",   # Amazon
    "NVDA",   # NVIDIA
    "META",   # Meta Platforms
    "TSLA",   # Tesla
    "JPM",    # JPMorgan Chase
    "V",      # Visa
    "JNJ",    # Johnson & Johnson
]

# Phase 2: ~50 US Large + Mid Caps for universe screening
# Spans all 11 GICS sectors for diversification analysis
PHASE2_SYMBOLS = [
    # Technology (10)
    "AAPL", "MSFT", "GOOGL", "NVDA", "META",
    "AVGO", "ADBE", "CRM", "CSCO", "INTC",
    # Consumer Discretionary (5)
    "AMZN", "TSLA", "HD", "MCD", "NKE",
    # Financials (6)
    "JPM", "V", "BAC", "GS", "MS", "BLK",
    # Healthcare (5)
    "JNJ", "UNH", "PFE", "ABT", "TMO",
    # Industrials (4)
    "CAT", "HON", "UPS", "RTX",
    # Consumer Staples (4)
    "PG", "KO", "PEP", "WMT",
    # Energy (3)
    "XOM", "CVX", "COP",
    # Communication Services (3)
    "NFLX", "DIS", "CMCSA",
    # Utilities (3)
    "NEE", "DUK", "SO",
    # Real Estate (3)
    "AMT", "PLD", "CCI",
    # Materials (3)
    "LIN", "APD", "SHW",
]

# Development: Small subset for fast testing
DEV_SYMBOLS = ["AAPL", "MSFT", "GOOGL"]

# ============================================================
# Database Configuration (PostgreSQL on pi5-1tb)
# ============================================================
# Default: In-cluster DNS name (for Flyte tasks running in K8s pods)
# Override with DB_HOST=192.168.178.45 for local development

DB_HOST = os.environ.get("DB_HOST", "postgres.flyte.svc.cluster.local")
DB_PORT = int(os.environ.get("DB_PORT", "5432"))
DB_NAME = os.environ.get("DB_NAME", "quant_trading")
DB_USER = os.environ.get("DB_USER", "flyte")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "flyte")

# ============================================================
# MinIO Configuration (S3-compatible on pi5-1tb)
# ============================================================
# In-cluster: minio.flyte.svc.cluster.local:9000
# External:   192.168.178.45:30900

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://minio.flyte.svc.cluster.local:9000")
MINIO_EXTERNAL_ENDPOINT = os.environ.get("MINIO_EXTERNAL_ENDPOINT", "http://192.168.178.45:30900")
MINIO_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "minio")
MINIO_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "miniostorage")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "quant-trading")

# S3 Parquet Storage (Data Lake on MinIO)
S3_DATA_BUCKET = os.environ.get("S3_DATA_BUCKET", "quant-data")

# ============================================================
# Data Provider
# ============================================================

DATA_PROVIDER = os.environ.get("DATA_PROVIDER", "yfinance")

# ============================================================
# WF2: Universe Screening Defaults
# ============================================================

WF2_LOOKBACK_DAYS = int(os.environ.get("WF2_LOOKBACK_DAYS", "252"))
WF2_RSI_WINDOW = int(os.environ.get("WF2_RSI_WINDOW", "14"))
WF2_RSI_OVERSOLD = int(os.environ.get("WF2_RSI_OVERSOLD", "30"))
WF2_RSI_OVERBOUGHT = int(os.environ.get("WF2_RSI_OVERBOUGHT", "70"))
WF2_MOMENTUM_WINDOWS = [10, 21, 63, 126, 252]  # 2w, 1m, 3m, 6m, 1y in trading days
WF2_FORECAST_HORIZON = int(os.environ.get("WF2_FORECAST_HORIZON", "21"))  # 1 month forward
WF2_KMEANS_MAX_K = int(os.environ.get("WF2_KMEANS_MAX_K", "10"))

# ============================================================
# WF3: Signal & Analysis Defaults
# ============================================================

WF3_TECH_WEIGHT = float(os.environ.get("WF3_TECH_WEIGHT", "0.50"))       # Phase 2: 50% tech
WF3_FUND_WEIGHT = float(os.environ.get("WF3_FUND_WEIGHT", "0.50"))       # Phase 2: 50% fund
WF3_MAX_QUINTILE = int(os.environ.get("WF3_MAX_QUINTILE", "2"))          # Analyze top 40%
WF3_LOOKBACK_DAYS = int(os.environ.get("WF3_LOOKBACK_DAYS", "252"))      # 1 year of trading days
WF3_SMA_SHORT = int(os.environ.get("WF3_SMA_SHORT", "50"))              # Short SMA window
WF3_SMA_LONG = int(os.environ.get("WF3_SMA_LONG", "200"))               # Long SMA window

# ============================================================
# WF4: Portfolio & Rebalancing Defaults
# ============================================================

WF4_INITIAL_CAPITAL = float(os.environ.get("WF4_INITIAL_CAPITAL", "25000.0"))  # EUR 25,000
WF4_MAX_POSITION_PCT = float(os.environ.get("WF4_MAX_POSITION_PCT", "0.05"))   # 5% max per stock
WF4_MAX_SECTOR_PCT = float(os.environ.get("WF4_MAX_SECTOR_PCT", "0.25"))       # 25% max per sector
WF4_CASH_RESERVE_PCT = float(os.environ.get("WF4_CASH_RESERVE_PCT", "0.05"))   # 5% cash reserve
WF4_MIN_TRADE_VALUE = float(os.environ.get("WF4_MIN_TRADE_VALUE", "100.0"))    # Skip trades < EUR 100
WF4_COMMISSION_PER_SHARE = float(os.environ.get("WF4_COMMISSION_PER_SHARE", "0.005"))  # $0.005/share
WF4_EXCHANGE_FEE_BPS = float(os.environ.get("WF4_EXCHANGE_FEE_BPS", "3.0"))    # 3 bps exchange fee
WF4_IMPACT_BPS_PER_1K = float(os.environ.get("WF4_IMPACT_BPS_PER_1K", "0.1")) # 0.1 bps market impact per $1000

# Phase 4: Paper Trading Mode
# When True, WF4 simulates trade execution, updates positions table,
# and takes portfolio snapshots for performance tracking.
# When False, WF4 generates order reports only (Phase 3 behavior).
WF4_PAPER_TRADING_ENABLED = os.environ.get("WF4_PAPER_TRADING_ENABLED", "true").lower() == "true"

# ============================================================
# WF5: Monitoring & Reporting Defaults
# ============================================================

WF5_DRAWDOWN_ALERT_PCT = float(os.environ.get("WF5_DRAWDOWN_ALERT_PCT", "0.05"))   # Alert if 30d drawdown > 5%
WF5_POSITION_ALERT_PCT = float(os.environ.get("WF5_POSITION_ALERT_PCT", "0.07"))   # Alert if single position > 7%
WF5_VAR_ALERT_PCT = float(os.environ.get("WF5_VAR_ALERT_PCT", "0.03"))             # Alert if VaR(95%) > 3% of portfolio
WF5_LOSS_ALERT_PCT = float(os.environ.get("WF5_LOSS_ALERT_PCT", "0.10"))           # Alert if unrealized loss > 10%
WF5_RISK_FREE_RATE = float(os.environ.get("WF5_RISK_FREE_RATE", "0.05"))           # Annualized risk-free rate
WF5_LOOKBACK_DAYS = int(os.environ.get("WF5_LOOKBACK_DAYS", "30"))                 # Risk metric lookback (trading days)

# ============================================================
# GICS Sector Mapping for PHASE2_SYMBOLS
# ============================================================
# Hardcoded mapping — more reliable than yfinance lookups and avoids rate limiting.
# Must be updated when adding new symbols to PHASE2_SYMBOLS.

SYMBOL_SECTORS = {
    # Technology (10)
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "NVDA": "Technology", "META": "Technology", "AVGO": "Technology",
    "ADBE": "Technology", "CRM": "Technology", "CSCO": "Technology", "INTC": "Technology",
    # Consumer Discretionary (5)
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary", "NKE": "Consumer Discretionary",
    # Financials (6)
    "JPM": "Financials", "V": "Financials", "BAC": "Financials",
    "GS": "Financials", "MS": "Financials", "BLK": "Financials",
    # Healthcare (5)
    "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare",
    "ABT": "Healthcare", "TMO": "Healthcare",
    # Industrials (4)
    "CAT": "Industrials", "HON": "Industrials", "UPS": "Industrials", "RTX": "Industrials",
    # Consumer Staples (4)
    "PG": "Consumer Staples", "KO": "Consumer Staples",
    "PEP": "Consumer Staples", "WMT": "Consumer Staples",
    # Energy (3)
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    # Communication Services (3)
    "NFLX": "Communication Services", "DIS": "Communication Services", "CMCSA": "Communication Services",
    # Utilities (3)
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    # Real Estate (3)
    "AMT": "Real Estate", "PLD": "Real Estate", "CCI": "Real Estate",
    # Materials (3)
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials",
}
