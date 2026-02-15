"""Launch plans for production domain.

Only define launch plans here that should run on a schedule in production.
CI/CD registers this file to the production domain on push to main.
If no launch plans are defined here, only the default (manual) launch plans exist.

Current schedules:
- WF1 Data Ingestion: Daily at 06:00 UTC (all 49 Phase 2 symbols, before EU market open)
- WF2 Universe Screening: Daily at 07:00 UTC (49 Phase 2 symbols, full screening)
- WF3 Signal Analysis: Daily at 08:00 UTC (after WF2, 30/40/30 tech+fund+sent)
- WF4 Portfolio Rebalancing: Daily at 09:00 UTC (after WF3, paper trading enabled)
- WF5 Monitoring & Reporting: Daily at 10:00 UTC (after WF4, P&L + risk + alerts)
- WF6 Backtesting: Weekly Sunday 11:00 UTC (signal strategy vs buy-and-hold)
"""

from flytekit import CronSchedule, LaunchPlan

from src.shared.config import (
    PHASE2_SYMBOLS,
    WF3_MAX_QUINTILE, WF3_LOOKBACK_DAYS,
    WF3_TECH_WEIGHT, WF3_FUND_WEIGHT, WF3_SENT_WEIGHT,
    WF3_SMA_SHORT, WF3_SMA_LONG,
    SENTIMENT_NEWS_DAYS, SENTIMENT_DECAY_HALF_LIFE,
    WF4_INITIAL_CAPITAL, WF4_MAX_POSITION_PCT, WF4_MAX_SECTOR_PCT,
    WF4_CASH_RESERVE_PCT, WF4_COMMISSION_PER_SHARE, WF4_EXCHANGE_FEE_BPS,
    WF4_IMPACT_BPS_PER_1K, WF4_MIN_TRADE_VALUE, WF4_PAPER_TRADING_ENABLED,
    WF4_DIVIDEND_REINVEST,
    WF5_LOOKBACK_DAYS, WF5_RISK_FREE_RATE,
    WF5_DRAWDOWN_ALERT_PCT, WF5_POSITION_ALERT_PCT,
    WF5_VAR_ALERT_PCT, WF5_LOSS_ALERT_PCT,
)
from src.wf1_data_ingestion.workflow import data_ingestion_workflow
from src.wf2_universe_screening.workflow import universe_screening_workflow
from src.wf3_signal_analysis.workflow import signal_analysis_workflow
from src.wf4_portfolio_rebalancing.workflow import portfolio_rebalancing_workflow
from src.wf5_monitoring.workflow import monitoring_workflow
from src.wf6_backtesting.workflow import backtesting_workflow

# WF1 Data Ingestion - daily at 06:00 UTC (all 49 Phase 2 symbols)
# Expanded from PHASE1 (10) to PHASE2 (49) so WF2 has full universe data
wf1_prod_daily = LaunchPlan.get_or_create(
    name="wf1_data_ingestion_prod_daily",
    workflow=data_ingestion_workflow,
    default_inputs={"symbols": PHASE2_SYMBOLS, "date": ""},
    schedule=CronSchedule(schedule="0 6 * * *"),
)

# WF2 Universe Screening - daily at 07:00 UTC (49 Phase 2 symbols)
# Runs after WF1 has ingested today's data
wf2_prod_daily = LaunchPlan.get_or_create(
    name="wf2_universe_screening_prod_daily",
    workflow=universe_screening_workflow,
    default_inputs={
        "symbols": PHASE2_SYMBOLS,
        "lookback_days": 252,
        "run_date": "",
    },
    schedule=CronSchedule(schedule="0 7 * * *"),
)

# WF3 Signal Analysis - daily at 08:00 UTC (1 hour after WF2)
# Phase 6: 30% tech + 40% fund + 30% sentiment (was 50/50 tech+fund)
# Rollback: set WF3_SENT_WEIGHT=0.0, WF3_TECH_WEIGHT=0.5, WF3_FUND_WEIGHT=0.5
wf3_prod_daily = LaunchPlan.get_or_create(
    name="wf3_signal_analysis_prod_daily",
    workflow=signal_analysis_workflow,
    default_inputs={
        "run_date": "",
        "max_quintile": WF3_MAX_QUINTILE,
        "lookback_days": WF3_LOOKBACK_DAYS,
        "tech_weight": WF3_TECH_WEIGHT,
        "fund_weight": WF3_FUND_WEIGHT,
        "sent_weight": WF3_SENT_WEIGHT,
        "sma_short": WF3_SMA_SHORT,
        "sma_long": WF3_SMA_LONG,
        "news_days": SENTIMENT_NEWS_DAYS,
        "decay_half_life": SENTIMENT_DECAY_HALF_LIFE,
    },
    schedule=CronSchedule(schedule="0 8 * * *"),
)

# WF4 Portfolio Rebalancing - daily at 09:00 UTC (1 hour after WF3)
# Reads WF3 signal results, computes target portfolio, generates order report
# Paper trading enabled — executes simulated trades and snapshots portfolio
wf4_prod_daily = LaunchPlan.get_or_create(
    name="wf4_portfolio_rebalancing_prod_daily",
    workflow=portfolio_rebalancing_workflow,
    default_inputs={
        "run_date": "",
        "initial_capital": WF4_INITIAL_CAPITAL,
        "max_position_pct": WF4_MAX_POSITION_PCT,
        "max_sector_pct": WF4_MAX_SECTOR_PCT,
        "cash_reserve_pct": WF4_CASH_RESERVE_PCT,
        "commission_per_share": WF4_COMMISSION_PER_SHARE,
        "exchange_fee_bps": WF4_EXCHANGE_FEE_BPS,
        "impact_bps_per_1k": WF4_IMPACT_BPS_PER_1K,
        "min_trade_value": WF4_MIN_TRADE_VALUE,
        "paper_trading": WF4_PAPER_TRADING_ENABLED,
        "dividend_reinvest": WF4_DIVIDEND_REINVEST,
    },
    schedule=CronSchedule(schedule="0 9 * * *"),
)

# WF5 Monitoring & Reporting - daily at 10:00 UTC (1 hour after WF4)
# Reads WF4 portfolio snapshots, computes risk metrics, checks alerts, generates report
wf5_prod_daily = LaunchPlan.get_or_create(
    name="wf5_monitoring_prod_daily",
    workflow=monitoring_workflow,
    default_inputs={
        "run_date": "",
        "lookback_days": WF5_LOOKBACK_DAYS,
        "risk_free_rate": WF5_RISK_FREE_RATE,
        "drawdown_threshold": WF5_DRAWDOWN_ALERT_PCT,
        "position_threshold": WF5_POSITION_ALERT_PCT,
        "var_threshold": WF5_VAR_ALERT_PCT,
        "loss_threshold": WF5_LOSS_ALERT_PCT,
    },
    schedule=CronSchedule(schedule="0 10 * * *"),
)

# WF6 Backtesting - weekly Sunday at 11:00 UTC
# Replays historical signal results through pension fund model,
# compares vs equal-weight buy-and-hold benchmark
wf6_prod_weekly = LaunchPlan.get_or_create(
    name="wf6_backtesting_prod_weekly",
    workflow=backtesting_workflow,
    default_inputs={
        "start_date": "2026-01-01",
        "end_date": "",
        "initial_capital": WF4_INITIAL_CAPITAL,
        "max_position_pct": WF4_MAX_POSITION_PCT,
        "max_sector_pct": WF4_MAX_SECTOR_PCT,
        "cash_reserve_pct": WF4_CASH_RESERVE_PCT,
    },
    schedule=CronSchedule(schedule="0 11 * * 0"),
)
