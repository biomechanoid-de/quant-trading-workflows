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

# Development: Small subset for fast testing
DEV_SYMBOLS = ["AAPL", "MSFT", "GOOGL"]

# ============================================================
# Database Configuration (PostgreSQL on pi5-1tb)
# ============================================================

DB_HOST = os.environ.get("DB_HOST", "192.168.178.45")
DB_PORT = int(os.environ.get("DB_PORT", "5432"))
DB_NAME = os.environ.get("DB_NAME", "quant_trading")
DB_USER = os.environ.get("DB_USER", "flyte")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")

# ============================================================
# MinIO Configuration (S3-compatible on pi5-1tb)
# ============================================================

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://192.168.178.45:30900")
MINIO_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "minio")
MINIO_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "miniostorage")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "quant-trading")

# ============================================================
# Data Provider
# ============================================================

DATA_PROVIDER = os.environ.get("DATA_PROVIDER", "yfinance")
