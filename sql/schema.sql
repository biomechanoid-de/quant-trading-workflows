-- Quant Trading Workflows: PostgreSQL Schema
-- Target: pi5-1tb (192.168.178.45) with NVMe SSD
-- Database: quant_trading
--
-- Usage: psql -h 192.168.178.45 -U flyte -d quant_trading -f sql/schema.sql

-- ============================================================
-- Market Data (WF1: Data Ingestion)
-- ============================================================
CREATE TABLE IF NOT EXISTS market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(12,4),
    high DECIMAL(12,4),
    low DECIMAL(12,4),
    close DECIMAL(12,4),
    adj_close DECIMAL(12,4),
    volume BIGINT,
    spread_bps DECIMAL(8,2),
    data_source VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(symbol, date)
);

CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol);
CREATE INDEX IF NOT EXISTS idx_market_data_date ON market_data(date);
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_date ON market_data(symbol, date);

-- ============================================================
-- Portfolio Positions (WF4: Portfolio & Rebalancing)
-- ============================================================
CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    shares DECIMAL(12,4) NOT NULL,
    avg_cost DECIMAL(12,4) NOT NULL,
    current_price DECIMAL(12,4),
    region VARCHAR(10),
    sector VARCHAR(50),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);

-- ============================================================
-- Trades (WF4: Portfolio & Rebalancing)
-- ============================================================
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(4) NOT NULL,          -- BUY/SELL
    quantity DECIMAL(12,4) NOT NULL,
    price DECIMAL(12,4) NOT NULL,
    commission DECIMAL(8,4),
    spread_cost DECIMAL(8,4),
    impact_cost DECIMAL(8,4),
    reason VARCHAR(100),
    workflow_execution_id VARCHAR(100)
);

CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(date);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);

-- ============================================================
-- Dividends (WF4: Portfolio & Rebalancing)
-- ============================================================
CREATE TABLE IF NOT EXISTS dividends (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    ex_date DATE NOT NULL,
    pay_date DATE,
    amount_per_share DECIMAL(8,4),
    shares_held DECIMAL(12,4),
    total_amount DECIMAL(12,4),
    reinvested BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_dividends_symbol ON dividends(symbol);
CREATE INDEX IF NOT EXISTS idx_dividends_ex_date ON dividends(ex_date);

-- ============================================================
-- Daily Portfolio Snapshots (WF5: Monitoring & Reporting)
-- ============================================================
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    total_value DECIMAL(14,2),
    cash DECIMAL(14,2),
    invested DECIMAL(14,2),
    daily_pnl DECIMAL(12,2),
    cumulative_dividends DECIMAL(12,2),
    num_positions INT
);

CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_date ON portfolio_snapshots(date);

-- ============================================================
-- Screening Runs (WF2: Universe & Screening)
-- ============================================================
CREATE TABLE IF NOT EXISTS screening_runs (
    id SERIAL PRIMARY KEY,
    run_date DATE NOT NULL UNIQUE,
    num_symbols INT,
    optimal_k INT,
    benchmark_cagr DECIMAL(10,6),
    benchmark_sharpe DECIMAL(10,6),
    benchmark_cumulative_return DECIMAL(10,6),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_screening_runs_date ON screening_runs(run_date);

-- ============================================================
-- Screening Results (WF2: Universe & Screening)
-- ============================================================
CREATE TABLE IF NOT EXISTS screening_results (
    id SERIAL PRIMARY KEY,
    run_date DATE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    forward_return DECIMAL(10,6),
    rsi DECIMAL(8,4),
    rsi_signal VARCHAR(20),
    volatility_252d DECIMAL(10,6),
    cagr DECIMAL(10,6),
    sharpe DECIMAL(10,6),
    sortino DECIMAL(10,6),
    calmar DECIMAL(10,6),
    max_drawdown DECIMAL(10,6),
    composite_score DECIMAL(10,6),
    quintile INT,
    cluster_id INT,
    cluster_label VARCHAR(50),
    momentum_returns_json TEXT,
    z_scores_json TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(run_date, symbol)
);

CREATE INDEX IF NOT EXISTS idx_screening_results_run_date ON screening_results(run_date);
CREATE INDEX IF NOT EXISTS idx_screening_results_symbol ON screening_results(symbol);
CREATE INDEX IF NOT EXISTS idx_screening_results_quintile ON screening_results(run_date, quintile);
