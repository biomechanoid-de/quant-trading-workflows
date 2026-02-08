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
