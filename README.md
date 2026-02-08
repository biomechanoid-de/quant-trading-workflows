# Quant Trading Workflows

Flyte-orchestrated quantitative trading system running on a Raspberry Pi K3s cluster. Inspired by the **Norwegian Government Pension Fund (NBIM/GPFG)**: broadly diversified, rule-based, transparent, and focused on long-term monthly income.

**Not the goal:** High-frequency trading or day-trading.
**The goal:** A systematic, data-driven investment workflow that runs monthly/weekly, generates signals, considers transaction costs, and manages a model portfolio -- all orchestrated by Flyte on a Pi cluster.

---

## Architecture: 5 Flyte Workflows

```
                        FLYTE WORKFLOW ORCHESTRATION
 ┌──────────────────────────────────────────────────────────────────────┐
 │                                                                      │
 │  WF1: Data Ingestion           (daily, 06:00 UTC)        [Phase 1]  │
 │   └─> Fetch prices -> Validate -> Store to DB -> Quality check      │
 │                                                                      │
 │  WF2: Universe & Screening     (weekly, Sunday)           [Phase 2]  │
 │   └─> Define universe -> Apply filters -> Rank stocks               │
 │                                                                      │
 │  WF3: Signal & Analysis        (weekly, after WF2)        [Phase 2]  │
 │   └─> Technical indicators -> Fundamentals -> Sentiment (Hailo)     │
 │   └─> Combine signals -> Composite scoring                          │
 │                                                                      │
 │  WF4: Portfolio & Rebalancing  (monthly, 1st Monday)      [Phase 3]  │
 │   └─> Target weights -> Transaction costs -> Order report           │
 │                                                                      │
 │  WF5: Monitoring & Reporting   (daily, 18:00 UTC)         [Phase 5]  │
 │   └─> P&L -> Risk metrics -> Grafana dashboard -> Alerts           │
 │                                                                      │
 └──────────────────────────────────────────────────────────────────────┘
```

---

## Cluster Hardware

| Node | Hostname | IP | Role | RAM | Storage |
|------|----------|-------|------|-----|---------|
| Raspberry Pi 4 | pi4-master | 192.168.178.24 | K3s Server (Control Plane) | 8 GB | 64 GB SD |
| Raspberry Pi 4 | pi4-worker1 | 192.168.178.37 | K3s Agent (Workflows) | 8 GB | 64 GB SD |
| Raspberry Pi 4 | pi4-worker2 | 192.168.178.40 | K3s Agent (Workflows) | 8 GB | 64 GB SD |
| Raspberry Pi 4 | pi4-worker3 | 192.168.178.42 | K3s Agent (Workflows) | 8 GB | 64 GB SD |
| Raspberry Pi 5 | pi5-ai | 192.168.178.61 | K3s Agent (Hailo-10H NPU) | 16 GB | SD |
| Raspberry Pi 5 | pi5-1tb | 192.168.178.45 | K3s Agent (PostgreSQL + MinIO) | 16 GB | 1 TB NVMe SSD |

**Services:** Flyte v1.16.3 | PostgreSQL | MinIO | Prometheus + Grafana | GitHub Actions CI/CD

---

## Project Structure

```
quant-trading-workflows/
├── .github/workflows/deploy.yml       # CI/CD: test -> build -> register
├── Dockerfile                         # ARM64 Python 3.11 container
├── Makefile                           # Build, test, deploy commands
├── pyproject.toml                     # Dependencies & project config
├── sql/
│   └── schema.sql                     # PostgreSQL schema (5 tables)
├── src/
│   ├── shared/                        # Shared across all workflows
│   │   ├── models.py                  # Data models (MarketDataBatch, Position, ...)
│   │   ├── config.py                  # Symbols, DB config, MinIO config
│   │   ├── db.py                      # PostgreSQL connection & helpers
│   │   └── providers/
│   │       ├── base.py                # DataProvider ABC (pluggable)
│   │       └── yfinance_provider.py   # Phase 1: free data via yfinance
│   ├── wf1_data_ingestion/            # Phase 1: IMPLEMENTED
│   │   ├── tasks.py                   # fetch, validate, store, quality_check
│   │   └── workflow.py                # data_ingestion_workflow
│   ├── wf2_universe_screening/        # Phase 2: stub
│   ├── wf3_signal_analysis/           # Phase 2: stub
│   ├── wf4_portfolio_rebalancing/     # Phase 3: stub
│   └── wf5_monitoring/                # Phase 5: stub
├── launch_plans/
│   ├── development.py                 # No schedules (all dev runs manual)
│   └── production.py                  # WF1 daily 06:00 UTC, 10 symbols
├── tests/                             # Unit tests (no network/DB required)
└── scripts/
    └── run_local.sh                   # Local WF1 testing
```

---

## Phase 1: Data Ingestion (WF1)

The first workflow is fully implemented and handles the daily market data pipeline.

### Pipeline (Dual-Write)

```
fetch_market_data ──> validate_ticks ──> ┬─ store_to_database (PostgreSQL) ─┬──> check_data_quality
   (500m/512Mi)       (200m/256Mi)       └─ store_to_parquet  (MinIO/S3)   ┘      (100m/128Mi)
```

**PostgreSQL (Hot Store):** Real-time SQL queries for WF2-WF5 (last 90 days)
**MinIO Parquet (Cold Store):** Hive-partitioned for backtesting, ML, archival:
```
s3://quant-data/market_data/source=yfinance/year=2026/month=02/day=08/market_data.parquet
```

### Symbols (Phase 1)

| Symbol | Company | Sector |
|--------|---------|--------|
| AAPL | Apple | Technology |
| MSFT | Microsoft | Technology |
| GOOGL | Alphabet | Technology |
| AMZN | Amazon | Consumer Discretionary |
| NVDA | NVIDIA | Technology |
| META | Meta Platforms | Technology |
| TSLA | Tesla | Consumer Discretionary |
| JPM | JPMorgan Chase | Financials |
| V | Visa | Financials |
| JNJ | Johnson & Johnson | Healthcare |

### Data Validation (Brenndoerfer Patterns)

- Zero or negative prices -> filtered
- Unreasonable spreads (>1000 bps) -> filtered
- Missing data -> reflected in quality score
- Stale data detection (planned)

### Provider Abstraction

```python
class DataProvider(ABC):
    def fetch_eod(symbols, date) -> MarketDataBatch: ...
    def fetch_fundamentals(symbol) -> dict: ...
    def fetch_dividends(symbol) -> list: ...

# Phase 1: YFinanceProvider (free)
# Later:   EODHDProvider, MassiveProvider, AlphaVantageProvider
```

Switching data providers is a **1-line change** in `wf1_data_ingestion/tasks.py`.

---

## Database Schema

PostgreSQL on pi5-1tb (NVMe SSD). Five tables:

| Table | Workflow | Purpose |
|-------|----------|---------|
| `market_data` | WF1 | OHLCV price data with UNIQUE(symbol, date) |
| `positions` | WF4 | Current portfolio positions |
| `trades` | WF4 | Executed trades with cost breakdown |
| `dividends` | WF4 | Dividend tracking and reinvestment |
| `portfolio_snapshots` | WF5 | Daily portfolio value snapshots |

Initialize: `make init-db` or `psql -f sql/schema.sql`

---

## Quick Start

### Prerequisites

- Python >= 3.11
- pip (with dev dependencies: `pip install -e ".[dev]"`)
- For cluster deployment: K3s cluster with Flyte, PostgreSQL, MinIO

### Run Tests

```bash
make test
# or: pytest tests/ -v --cov --cov-report=term-missing
```

### Run WF1 Locally

```bash
# Using pyflyte directly (needs yfinance, no DB required for fetch+validate)
pyflyte run src/wf1_data_ingestion/workflow.py data_ingestion_workflow \
    --symbols '["AAPL", "MSFT", "GOOGL"]' --date 2026-02-07

# Using the convenience script
./scripts/run_local.sh 2026-02-07
```

### Build Docker Image (ARM64)

```bash
make build
```

### Register to Flyte

```bash
make register-dev   # -> quant-trading / development
make register-prod  # -> quant-trading / production
```

---

## CI/CD Pipeline

Automated via GitHub Actions on the self-hosted runner (Pi cluster):

```
Push to development  ──>  test  ──>  build (ARM64)  ──>  register-dev
Push to main         ──>  test  ──>  build (ARM64)  ──>  register-prod  ──>  activate LPs
Pull Request         ──>  test only
```

Docker images: `ghcr.io/biomechanoid-de/quant-trading-workflows:{dev|latest|sha}`

### Launch Plan Management

Schedules are controlled exclusively via two files:

| File | Domain | Content |
|------|--------|---------|
| `launch_plans/production.py` | production | `wf1_data_ingestion_prod_daily` — Cron `0 6 * * *` |
| `launch_plans/development.py` | development | Empty — all dev runs are triggered manually |

CI/CD explicitly activates only named cron launch plans (not `--activate-launchplans` which would activate all). To add a new schedule: define it in the appropriate launch plan file and add an activation step in `deploy.yml`.

---

## Design Decisions

### 1. No Live Trading
The system generates **order reports**, not automatic trades. You decide whether and how to execute. Like the Norwegian Pension Fund's investment committee.

### 2. Dividends + Long-Term Growth
Pension fund model: broad diversification, low costs, regular dividend income as the primary return source.

### 3. Transaction Costs as First-Class Citizen
Every potential trade is evaluated against its costs (Brenndoerfer model: commission + spread + market impact). Only trades with positive expected net alpha are proposed.

### 4. Hailo NPU for Edge Intelligence (Phase 4)
Sentiment analysis runs locally on Pi 5 with Hailo-10H (40 TOPS). No cloud APIs, no ongoing costs, full data control.

### 5. Everything in Flyte
Every step is reproducible, versioned, and traceable via Flyte Console. No "it ran on my laptop" problem.

### 6. Provider Abstraction: Build First, Decide Later
All data access goes through `DataProvider` ABC. Phase 1 uses yfinance (free). The paid provider decision comes after Phase 5, based on real operational experience.

---

## Roadmap

| Phase | Weeks | Goal | Workflows |
|-------|-------|------|-----------|
| **1. Foundation** | 1-2 | Data flows, DB schema, WF1 runs on cluster | WF1 |
| **2. Analysis** | 3-4 | Stocks screened and scored | WF2, WF3 (tech+fund) |
| **3. Portfolio** | 5-6 | System proposes trades, tracks model portfolio | WF4 |
| **4. Intelligence** | 7-8 | Hailo NPU sentiment, backtesting | WF3 (sentiment) |
| **5. Production** | 9-10 | Full system autonomous, monitoring complete | WF5 |
| **6. Data Provider** | 10+ | Choose paid provider based on experience | All |

---

## Current Status (Phase 1 Complete)

| Component | Status |
|-----------|--------|
| WF1: Data Ingestion (5 tasks, dual-write) | ✅ Running daily at 06:00 UTC |
| WF2-WF5 Stubs | ✅ Registered, ready for Phase 2-5 |
| Historical Backfill | ✅ 26 trading days (Jan 1 - Feb 8, 2026) |
| CI/CD Pipeline | ✅ 18+ successful runs |
| Launch Plan Management | ✅ Explicit activation via `pyflyte launchplan` |
| Flyte Domains | ✅ Only development + production (staging removed) |
| Unit Tests | ✅ 33 tests, 90% coverage |
| Helm Revision | 16 |

## Related Repositories

| Repository | Purpose |
|------------|---------|
| [cluster-infra](https://github.com/biomechanoid-de/cluster-infra) | Terraform: K8s namespaces, secrets, quotas |
| [flyte-workflow-template](https://github.com/biomechanoid-de/flyte-workflow-template) | Base template for Flyte workflows on Pi cluster |

---

## Resource Allocation

| Workflow | Node | CPU Request | Memory Request | Why |
|----------|------|-------------|----------------|-----|
| WF1: Data Ingestion | Pi 4 Workers | 100-500m | 128-512Mi | I/O-bound (API calls) |
| WF2: Universe Screening | Pi 4 Workers | TBD | TBD | CPU-bound, moderate |
| WF3: Signal (Tech+Fund) | Pi 4 Workers | TBD | TBD | CPU-bound, parallelizable |
| WF3: Sentiment | **Pi 5 AI (Hailo)** | TBD | TBD | NPU for ML inference |
| WF4: Portfolio | Pi 4 Workers | TBD | TBD | CPU-bound, moderate |
| WF5: Monitoring | Pi 4 Workers | TBD | TBD | Lightweight |

Flyte domains (staging permanently removed):
- `quant-trading-development`: 2 CPU, 2 Gi, 10 pods max
- `quant-trading-production`: 4 CPU, 6 Gi, 20 pods max
