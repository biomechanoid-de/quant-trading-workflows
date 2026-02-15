# Quant Trading Workflows

Flyte-orchestrated quantitative trading system running on a Raspberry Pi K3s cluster. Inspired by the **Norwegian Government Pension Fund (NBIM/GPFG)**: broadly diversified, rule-based, transparent, and focused on long-term monthly income.

**Not the goal:** High-frequency trading or day-trading.
**The goal:** A systematic, data-driven investment workflow that runs monthly/weekly, generates signals, considers transaction costs, and manages a model portfolio -- all orchestrated by Flyte on a Pi cluster.

---

## Architecture: 6 Flyte Workflows

```
                        FLYTE WORKFLOW ORCHESTRATION
 +----------------------------------------------------------------------+
 |                                                                      |
 |  WF1: Data Ingestion           (daily, 06:00 UTC)        [Phase 1]  |
 |   +-> Fetch prices -> Validate -> Store to DB -> Quality check      |
 |                                                                      |
 |  WF2: Universe & Screening     (daily, 07:00 UTC)        [Phase 2]  |
 |   +-> Load prices -> Metrics -> Cluster + Score -> Report           |
 |                                                                      |
 |  WF3: Signal & Analysis        (daily, 08:00 UTC)        [Phase 6]  |
 |   +-> Load WF2 context -> Tech + Fund + Sentiment (3 parallel)     |
 |   +-> Combine signals (30/40/30) -> Store + Report                  |
 |                                                                      |
 |  WF4: Portfolio & Rebalancing  (daily, 09:00 UTC)        [Phase 4]  |
 |   +-> Target weights -> Transaction costs -> Order report           |
 |   +-> Paper trading -> Portfolio snapshot                           |
 |                                                                      |
 |  WF5: Monitoring & Reporting   (daily, 10:00 UTC)        [Phase 5]  |
 |   +-> P&L -> Risk metrics -> Alerts -> Markdown report             |
 |                                                                      |
 |  WF6: Backtesting              (daily, 11:00 UTC)        [Phase 6]  |
 |   +-> Load signals -> Simulate portfolio vs benchmark -> Report     |
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
| Raspberry Pi 5 | pi5-ai | 192.168.178.33 | K3s Agent (Hailo-10H NPU) | 16 GB | SD |
| Raspberry Pi 5 | pi5-1tb | 192.168.178.45 | K3s Agent (PostgreSQL + MinIO) | 16 GB | 64 GB SD + 1 TB NVMe SSD |

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
|   +-- schema.sql                     # PostgreSQL schema (12 tables)
+-- src/
|   +-- shared/                        # Shared across all workflows
|   |   +-- models.py                  # Data models (13 dataclasses)
|   |   +-- config.py                  # Symbols, DB config, MinIO config, WF2-WF6 params
|   |   +-- db.py                      # PostgreSQL helpers (store, query, screening, signals, monitoring)
|   |   +-- analytics.py              # 24 pure functions (RSI, SMA, MACD, Bollinger, sentiment, ...)
|   |   +-- storage.py                # MinIO/S3 Parquet storage
|   |   +-- providers/
|   |   |   +-- base.py                # DataProvider ABC (pluggable)
|   |   |   +-- yfinance_provider.py   # YFinance: EOD + fundamentals
|   |   |   +-- sentiment_base.py      # SentimentProvider ABC
|   |   |   +-- finnhub_provider.py    # Finnhub: financial news (primary)
|   |   |   +-- marketaux_provider.py  # Marketaux: financial news (fallback)
|   |   +-- inference/
|   |       +-- base.py                # SentimentClassifier ABC
|   |       +-- onnx_cpu.py            # ONNX Runtime CPU inference (DistilRoBERTa)
|   +-- wf1_data_ingestion/            # Phase 1: IMPLEMENTED
|   |   +-- tasks.py                   # fetch, validate, store, quality_check
|   |   +-- workflow.py                # data_ingestion_workflow
|   +-- wf2_universe_screening/        # Phase 2: IMPLEMENTED
|   |   +-- tasks.py                   # 9 tasks: load, compute, cluster, score, store, report
|   |   +-- workflow.py                # universe_screening_workflow
|   +-- wf3_signal_analysis/           # Phase 6: IMPLEMENTED (sentiment + tech + fund)
|   |   +-- tasks.py                   # 9 tasks: load, tech, fund, sentiment, combine, assemble, store, report
|   |   +-- workflow.py                # signal_analysis_workflow (3 parallel branches)
|   +-- wf4_portfolio_rebalancing/     # Phase 3+4: IMPLEMENTED (10 tasks + 2 paper trading)
|   |   +-- tasks.py                   # resolve, load, weights, prices, orders, paper trade, snapshot
|   |   +-- workflow.py                # portfolio_rebalancing_workflow
|   +-- wf5_monitoring/                # Phase 5: IMPLEMENTED
|   |   +-- tasks.py                   # 4 tasks: pnl, risk_metrics, alerts, report
|   |   +-- workflow.py                # monitoring_workflow
|   +-- wf6_backtesting/              # Phase 6: IMPLEMENTED
|   |   +-- tasks.py                   # 6 tasks: resolve, load, simulate, benchmark, compare, report
|   |   +-- workflow.py                # backtesting_workflow
+-- launch_plans/
|   +-- development.py                 # No schedules (all dev runs manual)
|   +-- production.py                  # WF1-WF6 daily schedules
+-- scripts/
|   +-- run_local.sh                   # Local WF1 testing
|   +-- export_sentiment_model.py      # One-time: export DistilRoBERTa to ONNX INT8
+-- tests/                             # 347 unit tests (no network/DB required)
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

Multi-factor stock screening workflow across 49 stocks spanning all 11 GICS sectors. Runs daily to rank the investable universe.

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

## Phase 6: Signal & Analysis (WF3)

Deeper technical, fundamental, and sentiment analysis for WF2's top-ranked stocks (quintiles 1-2). Produces composite buy/hold/sell signals with three-way weighted scoring: 30% technical + 40% fundamental + 30% sentiment.

### Pipeline (9 Tasks, 3-Branch Parallel DAG)

```
load_screening_context --> +- compute_technical_signals  -+
      (200m/256Mi)         +- fetch_fundamental_data     -+--> combine_signals --> assemble_signal_result
                           +- fetch_sentiment_data       -+     (300m/256Mi)          (200m/256Mi)
                                (500m/1536Mi)                                              |
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

### Sentiment Analysis (Phase 6)

Financial news sentiment via DistilRoBERTa ONNX model with dual news providers:

| Component | Description |
|-----------|-------------|
| **News Provider (Primary)** | Finnhub — 60 req/min free tier, company news endpoint |
| **News Provider (Fallback)** | Marketaux — 100 req/day free tier, used when Finnhub returns empty |
| **Classifier** | DistilRoBERTa-Finance (INT8 ONNX, ~70MB) — 3-class: positive/neutral/negative |
| **Aggregation** | Time-decay weighted (half-life 3 days) over last 7 days of headlines |
| **Score** | 0-100 sentiment score mapped to very_positive/positive/neutral/negative/very_negative |

**Provider abstraction:** `SentimentProvider` ABC allows swapping news sources. `SentimentClassifier` ABC allows migrating from ONNX CPU to Hailo NPU when model support is available.

### Signal Strength Classification

| Combined Score | Signal |
|---------------|--------|
| >= 75 | **Strong Buy** |
| 60-74 | **Buy** |
| 40-59 | **Hold** |
| 25-39 | **Sell** |
| < 25 | **Strong Sell** |

### Analytics Functions (added for WF3)

12 pure computation functions in `src/shared/analytics.py`:

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
| `compute_sentiment_score` | Time-decay-weighted news sentiment aggregation |
| `classify_sentiment_signal` | Sentiment strength classification |
| `aggregate_article_sentiments` | Merge provider articles with classifier output |

### Dual-Write Output

- **PostgreSQL:** `signal_runs` (metadata + sent_weight) + `signal_results` (per-symbol tech + fund + sentiment + combined)
- **MinIO Parquet:** `s3://quant-data/signals/year=YYYY/month=MM/day=DD/signals.parquet`
- **Text Report:** Signal distribution, top buys/sells, per-stock details with sentiment column

### Key Design Decisions

- **Top 40% filter:** Only analyzes WF2 quintiles 1-2 (~20 stocks) to respect Pi cluster + yfinance rate limits
- **30/40/30 weights:** 30% technical + 40% fundamental + 30% sentiment (Phase 6). Rollback: set `WF3_SENT_WEIGHT=0.0`, `WF3_TECH_WEIGHT=0.5`, `WF3_FUND_WEIGHT=0.5`
- **Graceful degradation:** Missing fundamentals default to neutral. Zero news articles → neutral sentiment (50.0) with `has_sentiment=False`
- **Rate limiting:** 0.2s sleep between yfinance fetches, 0.5s between news API calls
- **CPU inference:** ONNX Runtime on ARM64 CPU (future: Hailo NPU when DistilBERT support is added)

---

## Phase 3+4: Portfolio & Rebalancing (WF4)

Computes target portfolio weights from WF3 signals, generates trade orders with full transaction cost modeling, and executes paper trades to track simulated portfolio performance.

### Pipeline (12 Tasks)

```
resolve_run_date --> load_signal_context --> load_current_portfolio --> calculate_target_weights
                                                                            |
                                                                            v
                                                                    fetch_current_prices
                                                                            |
                                                                            v
                                                                    generate_trade_orders
                                                                            |
                                                                            v
                                                                    assemble_rebalancing_result
                                                                            |
                                                  +-------------------------+-------------------------+
                                                  v                         v                         v
                                          store_to_db              store_to_parquet          generate_order_report
                                                                                                      |
                                                                                                      v
                                                                                            execute_paper_trades
                                                                                                      |
                                                                                                      v
                                                                                            snapshot_portfolio
```

### Pension Fund Allocation Model

Signal-weighted allocation inspired by the Norwegian Government Pension Fund:

| Signal | Weight Multiplier |
|--------|-------------------|
| Strong Buy | 3x base weight |
| Buy | 2x base weight |
| Hold | 0 (no allocation) |
| Sell | 0 (no allocation) |
| Strong Sell | 0 (no allocation) |

**Constraints:** Max 5% per stock, 25% per sector, 5% cash reserve.

### Transaction Cost Model (Brenndoerfer)

Every trade is evaluated against its full cost before execution:

| Component | Default |
|-----------|---------|
| Commission | $0.005/share |
| Spread | 5 bps |
| Exchange Fee | 3 bps |
| Market Impact | 0.1 bps per $1,000 |

Minimum trade value filter: EUR 100 (skips uneconomical small trades).

### Paper Trading

Paper trading is **enabled by default** (`WF4_PAPER_TRADING_ENABLED=true`). When enabled:

- `execute_paper_trades` simulates order execution, updates the `positions` table with weighted average cost tracking
- `snapshot_portfolio` takes a daily portfolio snapshot for performance tracking
- Cash tracking via `portfolio_snapshots` table (latest snapshot's cash = source of truth)
- Initial capital: EUR 25,000 (configurable via `WF4_INITIAL_CAPITAL`)

### Dual-Write Output

- **PostgreSQL:** `rebalancing_runs` (metadata), `positions` (current holdings), `trades` (executed orders), `portfolio_snapshots` (daily NAV)
- **MinIO Parquet:** `s3://quant-data/rebalancing/year=YYYY/month=MM/day=DD/{target_weights,trade_orders}.parquet`
- **Order Report:** `s3://quant-data/reports/wf4/year=YYYY/month=MM/day=DD/`

---

## Phase 5: Monitoring & Reporting (WF5)

Core monitoring workflow that computes P&L, risk metrics, checks alert thresholds, and generates a comprehensive markdown report.

### Pipeline (4 Tasks)

```
calculate_pnl --> compute_risk_metrics --> check_alerts --> generate_monitoring_report
 (300m/256Mi)      (300m/256Mi)           (200m/128Mi)       (200m/256Mi)
```

### P&L Calculation

| Metric | Source |
|--------|--------|
| Daily P&L | Latest vs. previous portfolio snapshot |
| MTD P&L | Current vs. first-of-month snapshot |
| YTD P&L | Current vs. first-of-year snapshot |
| Unrealized P&L | Per-position: (current_price - avg_cost) * shares |

### Risk Metrics (30-Day Window)

| Metric | Method |
|--------|--------|
| Sharpe Ratio | Annualized excess return / volatility (rf=5%) |
| Sortino Ratio | Annualized excess return / downside deviation |
| Max Drawdown | Maximum peak-to-trough decline |
| VaR (95%) | Historical 5th percentile of daily P&L |

Reuses existing analytics functions: `compute_sharpe()`, `compute_sortino()`, `compute_max_drawdown()` from `shared/analytics.py`.

### Alert System

| Alert Type | Default Threshold |
|------------|-------------------|
| Drawdown | > 5% (30-day) |
| Position Concentration | > 7% (single stock) |
| VaR Breach | > 3% of portfolio |
| Unrealized Loss | > 10% (single position) |

All thresholds are configurable via environment variables (`WF5_DRAWDOWN_ALERT_PCT`, etc.).

### Output

- **PostgreSQL:** `monitoring_runs` table with UPSERT on run_date (idempotent)
- **MinIO:** `s3://quant-data/reports/wf5/year=YYYY/month=MM/day=DD/monitoring_report.md`
- **Report:** Markdown with sections: Summary, P&L, Winners/Losers, Risk, Sectors, Alerts
- DB and MinIO failures are non-fatal (report is still returned as workflow output)

---

## Phase 6: Backtesting (WF6)

Historical backtesting workflow that replays the pension fund portfolio construction on past signal results and compares against an equal-weight buy-and-hold benchmark.

### Pipeline (6 Tasks, Parallel DAG)

```
resolve_backtest_params --> load_historical_signals --> +- simulate_signal_portfolio -+--> compare_strategies
      (100m/128Mi)              (300m/256Mi)            +- compute_benchmark_returns -+      (200m/256Mi)
                                                                                                  |
                                                                                                  v
                                                                                      generate_backtest_report
                                                                                           (200m/256Mi)
```

### Strategy Simulation

- **Signal Portfolio:** Replays pension fund allocation model (strong_buy=3x, buy=2x, hold/sell=0) with position caps (5%), sector caps (25%), and full transaction cost modeling
- **Benchmark:** Equal-weight buy-and-hold across all buy/strong_buy stocks from the first signal date

### Comparison Metrics

| Metric | Description |
|--------|-------------|
| CAGR | Compound Annual Growth Rate |
| Sharpe | Risk-adjusted return (annualized) |
| Sortino | Downside risk-adjusted return |
| Max Drawdown | Maximum peak-to-trough decline |
| Calmar | CAGR / max drawdown |

All metrics computed for both signal portfolio and benchmark, plus excess (signal - benchmark).

### Output

- **PostgreSQL:** `backtest_runs` table with UPSERT on (start_date, end_date)
- **MinIO:** `s3://quant-data/reports/wf6/year=YYYY/month=MM/day=DD/backtest_report.md`
- **Report:** Markdown with strategy vs benchmark comparison, per-period returns, excess metrics
- DB and MinIO failures are non-fatal (report is still returned as workflow output)

---

## Database Schema

PostgreSQL on pi5-1tb (64 GB SD card; MinIO on 1 TB NVMe SSD). Twelve tables:

| Table | Workflow | Purpose |
|-------|----------|---------|
| `market_data` | WF1 | OHLCV price data with UNIQUE(symbol, date) |
| `screening_runs` | WF2 | Screening run metadata (benchmark, optimal K) |
| `screening_results` | WF2 | Per-symbol metrics, scores, quintiles, clusters |
| `signal_runs` | WF3 | Signal analysis run metadata (tech/fund/sent weights) |
| `signal_results` | WF3 | Per-symbol tech + fund + sentiment + combined signals |
| `rebalancing_runs` | WF4 | Rebalancing run metadata and order summaries |
| `positions` | WF4 | Current portfolio positions (paper trading) |
| `trades` | WF4 | Executed trades with cost breakdown |
| `dividends` | WF4 | Dividend tracking and reinvestment |
| `portfolio_snapshots` | WF4/WF5 | Portfolio value snapshots for performance tracking |
| `monitoring_runs` | WF5 | Monitoring run metadata, P&L, risk metrics, alerts |
| `backtest_runs` | WF6 | Backtest results: strategy vs benchmark metrics |

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
| `launch_plans/production.py` | production | `wf1_data_ingestion_prod_daily` — Cron `0 6 * * *` (daily 06:00 UTC) |
| `launch_plans/production.py` | production | `wf2_universe_screening_prod_daily` — Cron `0 7 * * *` (daily 07:00 UTC) |
| `launch_plans/production.py` | production | `wf3_signal_analysis_prod_daily` — Cron `0 8 * * *` (daily 08:00 UTC) |
| `launch_plans/production.py` | production | `wf4_portfolio_rebalancing_prod_daily` — Cron `0 9 * * *` (daily 09:00 UTC) |
| `launch_plans/production.py` | production | `wf5_monitoring_prod_daily` — Cron `0 10 * * *` (daily 10:00 UTC) |
| `launch_plans/production.py` | production | `wf6_backtesting_prod_daily` -- Cron `0 11 * * *` (daily 11:00 UTC) |
| `launch_plans/development.py` | development | Empty -- all dev runs are triggered manually |

CI/CD explicitly activates only named cron launch plans (not `--activate-launchplans` which would activate all). To add a new schedule: define it in the appropriate launch plan file and add an activation step in `deploy.yml`.

---

## Design Decisions

### 1. Paper Trading Only (No Live Execution)
The system runs **simulated trades** via paper trading, tracking a model portfolio with full transaction cost modeling. No real broker connections. You decide whether and how to execute real trades. Like the Norwegian Pension Fund's investment committee.

### 2. Dividends + Long-Term Growth
Pension fund model: broad diversification, low costs, regular dividend income as the primary return source.

### 3. Transaction Costs as First-Class Citizen
Every potential trade is evaluated against its costs (Brenndoerfer model: commission + spread + market impact). Only trades with positive expected net alpha are proposed.

### 4. Edge Intelligence with Abstraction Layer (Phase 6)
Sentiment analysis uses ONNX Runtime on CPU now, with a `SentimentClassifier` ABC designed for future Hailo-10H NPU migration when DistilBERT support is added. News from Finnhub (primary) + Marketaux (fallback). No cloud inference APIs, full data control.

### 5. Everything in Flyte
Every step is reproducible, versioned, and traceable via Flyte Console. No "it ran on my laptop" problem.

### 6. Provider Abstraction: Build First, Decide Later
All data access goes through `DataProvider` ABC. Phase 1 uses yfinance (free). The paid provider decision comes after Phase 5, based on real operational experience.

### 7. Flytekit Type Safety
Complex types (`List[List]`, `Dict[str, List]`, dataclasses with `List`/`Dict` fields) cannot be passed between Flyte tasks -- Flytekit's type engine raises Promise binding errors. Solution: use JSON-serialized `Dict[str, str]` for complex inter-task data, pass individual primitives for configuration, and construct complex objects only inside tasks.

---

## Roadmap

| Phase | Weeks | Goal | Workflows | Status |
|-------|-------|------|-----------|--------|
| **1. Foundation** | 1-2 | Data flows, DB schema, WF1 runs on cluster | WF1 | Complete |
| **2. Analysis** | 3-4 | Stocks screened and scored | WF2, WF3 (tech+fund) | Complete |
| **3. Portfolio** | 5-6 | System proposes trades, tracks model portfolio | WF4 | Complete |
| **4. Paper Trading** | 7-8 | Simulated trade execution, portfolio tracking | WF4 (paper trading) | Complete |
| **5. Monitoring** | 9-10 | Full system autonomous, monitoring complete | WF5 | Complete |
| **6. Intelligence** | 11-12 | Sentiment analysis (ONNX CPU), backtesting | WF3 (sentiment), WF6 | Complete |
| **7. Data Provider** | TBD | Choose paid provider based on experience | All | Planned |

---

## Current Status (Phase 6 Complete)

| Component | Status |
|-----------|--------|
| WF1: Data Ingestion (5 tasks, dual-write) | LIVE -- daily at 06:00 UTC |
| WF2: Universe & Screening (9 tasks, parallel DAG) | LIVE -- daily at 07:00 UTC |
| WF3: Signal & Analysis (9 tasks, 3-branch parallel DAG) | LIVE -- daily at 08:00 UTC (30/40/30 tech+fund+sent) |
| WF4: Portfolio & Rebalancing (12 tasks, paper trading) | LIVE -- daily at 09:00 UTC |
| WF5: Monitoring & Reporting (4 tasks, P&L + risk + alerts) | LIVE -- daily at 10:00 UTC |
| WF6: Backtesting (6 tasks, strategy vs benchmark) | LIVE -- daily at 11:00 UTC |
| Historical Backfill | Full 2025 (250 trading days, 2,475 rows) + 2026 YTD |
| CI/CD Pipeline | 30+ successful runs |
| Launch Plan Management | 6 active launch plans (all daily) |
| Flyte Domains | Only development + production (staging removed) |
| Unit Tests | 347 tests, 90% coverage |
| Paper Trading | Enabled (WF4_PAPER_TRADING_ENABLED=true since 15.02.2026) |
| Storage Architecture | PostgreSQL on SD card, MinIO on NVMe SSD (pi5-1tb split) |
| Sentiment Analysis | ONNX CPU (DistilRoBERTa), Finnhub + Marketaux news providers |

## Related Repositories

| Repository | Purpose |
|------------|---------|
| [cluster-infra](https://github.com/biomechanoid-de/cluster-infra) | Terraform: K8s namespaces, secrets, quotas |

---

## Resource Allocation

| Workflow | Node | CPU Request | Memory Request | Why |
|----------|------|-------------|----------------|-----|
| WF1: Data Ingestion | Pi 4 Workers | 100-500m | 128-512Mi | I/O-bound (API calls) |
| WF2: Universe Screening | Pi 4 Workers | 100-500m | 128-512Mi | CPU-bound (analytics, K-Means) |
| WF3: Signal (Tech+Fund) | Pi 4 Workers | 100-500m | 128Mi-1Gi | CPU-bound (indicators) + I/O (yfinance) |
| WF3: Sentiment | Pi 4 Workers | 500m | 768Mi-1536Mi | ONNX Runtime CPU inference (~70MB model) |
| WF4: Portfolio | Pi 4 Workers | 100-500m | 128Mi-512Mi | CPU-bound (optimization) + I/O (prices) |
| WF5: Monitoring | Pi 4 Workers | 100-300m | 128Mi-256Mi | Lightweight (DB reads + report gen) |
| WF6: Backtesting | Pi 4 Workers | 200-300m | 128Mi-256Mi | DB reads + portfolio simulation |

Flyte domains (staging permanently removed):
- `quant-trading-development`: 2 CPU, 2 Gi, 10 pods max
- `quant-trading-production`: 4 CPU, 6 Gi, 20 pods max
