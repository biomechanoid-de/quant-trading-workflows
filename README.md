# Quant Trading Workflows

Flyte-orchestrated quantitative trading system running on a Raspberry Pi K3s cluster. Inspired by the **Norwegian Government Pension Fund (NBIM/GPFG)**: broadly diversified, rule-based, transparent, and focused on long-term monthly income.

**Not the goal:** High-frequency trading or day-trading.
**The goal:** A systematic, data-driven investment workflow that runs monthly/weekly, generates signals, considers transaction costs, and manages a model portfolio -- all orchestrated by Flyte on a Pi cluster.

---

## Architecture: 5 Flyte Workflows

```
                        FLYTE WORKFLOW ORCHESTRATION
 +----------------------------------------------------------------------+
 |                                                                      |
 |  WF1: Data Ingestion           (daily, 06:00 UTC)        [Phase 1]  |
 |   +-> Fetch prices -> Validate -> Store to DB -> Quality check      |
 |                                                                      |
 |  WF2: Universe & Screening     (weekly, Sun 08:00 UTC)   [Phase 2]  |
 |   +-> Load prices -> Metrics -> Cluster + Score -> Report           |
 |                                                                      |
 |  WF3: Signal & Analysis        (weekly, Sun 12:00 UTC)   [Phase 2]  |
 |   +-> Load WF2 context -> Technical + Fundamentals (parallel)       |
 |   +-> Combine signals -> Store + Report                             |
 |                                                                      |
 |  WF4: Portfolio & Rebalancing  (monthly, 1st Monday)      [Phase 3]  |
 |   +-> Target weights -> Transaction costs -> Order report           |
 |                                                                      |
 |  WF5: Monitoring & Reporting   (daily, 18:00 UTC)         [Phase 5]  |
 |   +-> P&L -> Risk metrics -> Grafana dashboard -> Alerts           |
 |                                                                      |
 +----------------------------------------------------------------------+
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
+-- .github/workflows/deploy.yml       # CI/CD: test -> build -> register
+-- Dockerfile                         # ARM64 Python 3.11 container
+-- Makefile                           # Build, test, deploy commands
+-- pyproject.toml                     # Dependencies & project config
+-- sql/
|   +-- schema.sql                     # PostgreSQL schema (9 tables)
+-- src/
|   +-- shared/                        # Shared across all workflows
|   |   +-- models.py                  # Data models (11 dataclasses)
|   |   +-- config.py                  # Symbols, DB config, MinIO config, WF2/WF3 params
|   |   +-- db.py                      # PostgreSQL helpers (store, query, screening, signals)
|   |   +-- analytics.py              # 20 pure functions (RSI, SMA, MACD, Bollinger, ...)
|   |   +-- storage.py                # MinIO/S3 Parquet storage
|   |   +-- providers/
|   |       +-- base.py                # DataProvider ABC (pluggable)
|   |       +-- yfinance_provider.py   # YFinance: EOD + fundamentals
|   +-- wf1_data_ingestion/            # Phase 1: IMPLEMENTED
|   |   +-- tasks.py                   # fetch, validate, store, quality_check
|   |   +-- workflow.py                # data_ingestion_workflow
|   +-- wf2_universe_screening/        # Phase 2: IMPLEMENTED
|   |   +-- tasks.py                   # 9 tasks: load, compute, cluster, score, store, report
|   |   +-- workflow.py                # universe_screening_workflow
|   +-- wf3_signal_analysis/           # Phase 2: IMPLEMENTED
|   |   +-- tasks.py                   # 8 tasks: load, tech, fund, combine, assemble, store, report
|   |   +-- workflow.py                # signal_analysis_workflow
|   +-- wf4_portfolio_rebalancing/     # Phase 3: stub
|   +-- wf5_monitoring/                # Phase 5: stub
+-- launch_plans/
|   +-- development.py                 # No schedules (all dev runs manual)
|   +-- production.py                  # WF1 daily + WF2 weekly + WF3 weekly
+-- tests/                             # 165 unit tests (no network/DB required)
+-- scripts/
    +-- run_local.sh                   # Local WF1 testing
```

---

## Phase 1: Data Ingestion (WF1)

The first workflow handles the daily market data pipeline.

### Pipeline (Dual-Write)

```
fetch_market_data --> validate_ticks --> +- store_to_database (PostgreSQL) -+--> check_data_quality
   (500m/512Mi)       (200m/256Mi)      +- store_to_parquet  (MinIO/S3)   +      (100m/128Mi)
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

## Phase 2: Universe & Screening (WF2)

Multi-factor stock screening workflow across 49 stocks spanning all 11 GICS sectors. Runs weekly on Sundays to rank the investable universe.

### Pipeline (9 Tasks, Parallel DAG)

```
load_historical_prices --> compute_returns_and_metrics --> +- cluster_stocks --------+--> merge_cluster_assignments
      (500m/512Mi)               (500m/512Mi)             +- score_and_rank_factors -+          (200m/256Mi)
                                                                                                     |
                                                                                                     v
                                                                                        assemble_screening_result
                                                                                              (500m/512Mi)
                                                                                                     |
                                                                               +---------------------+--------------------+
                                                                               v                     v                    v
                                                                       store_screening_to_db  store_to_parquet  generate_screening_report
                                                                         (200m/256Mi)         (200m/256Mi)          (100m/128Mi)
```

### Multi-Factor Model (Brenndoerfer)

Four factors with Z-score normalization and quintile ranking (Q1=best, Q5=worst):

| Factor | Weight | Description |
|--------|--------|-------------|
| Momentum | 30% | Multi-window returns (10d, 21d, 63d, 126d, 252d) |
| Low Volatility | 25% | Inverse 252-day annualized volatility |
| RSI Signal | 20% | Contrarian signal (oversold=bullish, overbought=bearish) |
| Sharpe Ratio | 25% | Risk-adjusted return quality |

### K-Means Clustering

- Stocks clustered by risk-return characteristics (volatility, returns, Sharpe, drawdown)
- Elbow method with distance-from-line heuristic to find optimal K (max K=10)
- Descriptive cluster labels: HiMom-LoVol, LoMom-HiVol, etc.
- StandardScaler normalization before clustering

### Analytics Functions (`src/shared/analytics.py`)

11 pure computation functions for WF2, fully unit-tested, no side effects (see WF3 section for 9 additional functions):

| Function | Description |
|----------|-------------|
| `calculate_rsi` | Relative Strength Index (Wilder's smoothing) |
| `classify_rsi_signal` | Oversold/Neutral/Overbought classification |
| `compute_cagr` | Compound Annual Growth Rate |
| `compute_sharpe` | Sharpe ratio (annualized, risk-free rate adjusted) |
| `compute_sortino` | Sortino ratio (downside deviation only) |
| `compute_max_drawdown` | Maximum peak-to-trough drawdown |
| `compute_calmar` | Calmar ratio (CAGR / max drawdown) |
| `compute_benchmark_performance` | Equal-weight benchmark metrics |
| `zscore_normalize` | Cross-sectional Z-score normalization |
| `assign_quintiles` | Quintile ranking (Q1=best) |

### Symbols (Phase 2 -- 49 stocks, 11 GICS sectors)

| Sector | Stocks |
|--------|--------|
| Technology (10) | AAPL, MSFT, GOOGL, NVDA, META, AVGO, ADBE, CRM, CSCO, INTC |
| Consumer Discretionary (5) | AMZN, TSLA, HD, MCD, NKE |
| Financials (6) | JPM, V, BAC, GS, MS, BLK |
| Healthcare (5) | JNJ, UNH, PFE, ABT, TMO |
| Industrials (4) | CAT, HON, UPS, RTX |
| Consumer Staples (4) | PG, KO, PEP, WMT |
| Energy (3) | XOM, CVX, COP |
| Communication Services (3) | NFLX, DIS, CMCSA |
| Utilities (3) | NEE, DUK, SO |
| Real Estate (3) | AMT, PLD, CCI |
| Materials (3) | LIN, APD, SHW |

### Dual-Write Output

- **PostgreSQL:** `screening_runs` (run metadata) + `screening_results` (per-symbol metrics)
- **MinIO Parquet:** `s3://quant-data/screening/year=YYYY/month=MM/day=DD/screening.parquet`
- **Text Report:** Full screening summary returned as workflow output

---

## Phase 2: Signal & Analysis (WF3)

Deeper technical and fundamental analysis for WF2's top-ranked stocks (quintiles 1-2). Produces composite buy/hold/sell signals with configurable weights.

### Pipeline (8 Tasks, Parallel DAG)

```
load_screening_context --> +- compute_technical_signals  -+--> combine_signals --> assemble_signal_result
      (200m/256Mi)         +- fetch_fundamental_data     -+     (300m/256Mi)          (200m/256Mi)
                                                                                           |
                                                                          +----------------+----------------+
                                                                          v                v                v
                                                                  store_signals_to_db  store_to_parquet  generate_report
                                                                    (200m/256Mi)       (200m/256Mi)      (100m/128Mi)
```

### Technical Indicators (Close-Based)

All indicators work with close prices only (WF1 currently stores close, not OHLC). ATR deferred until WF1 is upgraded.

| Indicator | Parameters | Signal Output |
|-----------|------------|---------------|
| **SMA Crossover** | Short=50, Long=200 | bullish (Golden Cross) / bearish (Death Cross) / neutral |
| **MACD** | Fast=12, Slow=26, Signal=9 | bullish / bearish / neutral |
| **Bollinger Bands** | Window=20, 2 std dev | oversold / overbought / neutral |

Technical Score = SMA (40%) + MACD (35%) + Bollinger (25%)

### Fundamental Analysis (via yfinance)

| Metric | Weight | Interpretation |
|--------|--------|----------------|
| **P/E Ratio** | 30% | Z-score vs sector median (lower = more undervalued) |
| **Dividend Yield** | 15% | Higher yield = better for income |
| **ROE** | 25% | Higher efficiency = better |
| **Debt/Equity** | 15% | Lower leverage = better |
| **Current Ratio** | 15% | Closer to 1.5 = healthier |

### Signal Strength Classification

| Combined Score | Signal |
|---------------|--------|
| >= 75 | **Strong Buy** |
| 60-74 | **Buy** |
| 40-59 | **Hold** |
| 25-39 | **Sell** |
| < 25 | **Strong Sell** |

### Analytics Functions (added for WF3)

9 new pure computation functions in `src/shared/analytics.py`:

| Function | Description |
|----------|-------------|
| `calculate_sma` | Simple Moving Average |
| `calculate_sma_crossover_signal` | Golden/Death Cross detection |
| `calculate_macd` | MACD line, signal line, histogram |
| `classify_macd_signal` | Bullish/bearish/neutral from MACD |
| `calculate_bollinger_bands` | Upper, middle, lower bands |
| `classify_bollinger_signal` | Oversold/overbought/neutral |
| `normalize_pe_ratio` | P/E z-score vs sector median |
| `compute_fundamental_score` | Weighted 0-100 fundamental score |
| `classify_value_signal` | Value/growth/balanced classification |

### Dual-Write Output

- **PostgreSQL:** `signal_runs` (metadata) + `signal_results` (per-symbol technical + fundamental + combined)
- **MinIO Parquet:** `s3://quant-data/signals/year=YYYY/month=MM/day=DD/signals.parquet`
- **Text Report:** Signal distribution, top buys/sells, per-stock details

### Key Design Decisions

- **Top 40% filter:** Only analyzes WF2 quintiles 1-2 (~20 stocks) to respect Pi cluster + yfinance rate limits
- **50/50 weights:** Phase 2 has no sentiment. Phase 4 shifts to 30% tech / 40% fund / 30% sentiment
- **Graceful fundamentals:** Missing P/E, ROE, D/E default to neutral scores (not exclusion). `data_quality` field tracks completeness
- **Rate limiting:** 0.2s sleep between yfinance fundamental fetches (5 req/s)

---

## Database Schema

PostgreSQL on pi5-1tb (NVMe SSD). Nine tables:

| Table | Workflow | Purpose |
|-------|----------|---------|
| `market_data` | WF1 | OHLCV price data with UNIQUE(symbol, date) |
| `screening_runs` | WF2 | Screening run metadata (benchmark, optimal K) |
| `screening_results` | WF2 | Per-symbol metrics, scores, quintiles, clusters |
| `signal_runs` | WF3 | Signal analysis run metadata |
| `signal_results` | WF3 | Per-symbol technical + fundamental + combined signals |
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
Push to development  -->  test  -->  build (ARM64)  -->  register-dev
Push to main         -->  test  -->  build (ARM64)  -->  register-prod  -->  activate LPs
Pull Request         -->  test only
```

Docker images: `ghcr.io/biomechanoid-de/quant-trading-workflows:{dev|latest|sha}`

### Launch Plan Management

Schedules are controlled exclusively via two files:

| File | Domain | Content |
|------|--------|---------|
| `launch_plans/production.py` | production | `wf1_data_ingestion_prod_daily` -- Cron `0 6 * * *` (daily 06:00 UTC) |
| `launch_plans/production.py` | production | `wf2_universe_screening_prod_weekly` -- Cron `0 8 * * 0` (Sunday 08:00 UTC) |
| `launch_plans/production.py` | production | `wf3_signal_analysis_prod_weekly` -- Cron `0 12 * * 0` (Sunday 12:00 UTC) |
| `launch_plans/development.py` | development | Empty -- all dev runs are triggered manually |

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

### 7. Flytekit Type Safety
Complex types (`List[List]`, `Dict[str, List]`, dataclasses with `List`/`Dict` fields) cannot be passed between Flyte tasks -- Flytekit's type engine raises Promise binding errors. Solution: use JSON-serialized `Dict[str, str]` for complex inter-task data, pass individual primitives for configuration, and construct complex objects only inside tasks.

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

## Current Status (Phase 2 Complete)

| Component | Status |
|-----------|--------|
| WF1: Data Ingestion (5 tasks, dual-write) | Running daily at 06:00 UTC |
| WF2: Universe & Screening (9 tasks, parallel DAG) | Running weekly Sun 08:00 UTC |
| WF3: Signal & Analysis (8 tasks, parallel DAG) | Running weekly Sun 12:00 UTC |
| WF4-WF5 Stubs | Registered, ready for Phase 3-5 |
| Historical Backfill | Full 2025 (250 trading days, 2,475 rows) + 2026 YTD |
| CI/CD Pipeline | 20+ successful runs |
| Launch Plan Management | 3 active launch plans (WF1 daily + WF2 weekly + WF3 weekly) |
| Flyte Domains | Only development + production (staging removed) |
| Unit Tests | 165 tests, 90% coverage |

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
| WF2: Universe Screening | Pi 4 Workers | 100-500m | 128-512Mi | CPU-bound (analytics, K-Means) |
| WF3: Signal (Tech+Fund) | Pi 4 Workers | 100-500m | 128Mi-1Gi | CPU-bound (indicators) + I/O (yfinance) |
| WF3: Sentiment (Phase 4) | **Pi 5 AI (Hailo)** | TBD | TBD | NPU for ML inference |
| WF4: Portfolio | Pi 4 Workers | TBD | TBD | CPU-bound, moderate |
| WF5: Monitoring | Pi 4 Workers | TBD | TBD | Lightweight |

Flyte domains (staging permanently removed):
- `quant-trading-development`: 2 CPU, 2 Gi, 10 pods max
- `quant-trading-production`: 4 CPU, 6 Gi, 20 pods max
