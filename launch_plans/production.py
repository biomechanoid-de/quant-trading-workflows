"""Launch plans for production domain.

Only define launch plans here that should run on a schedule in production.
CI/CD registers this file to the production domain on push to main.
If no launch plans are defined here, only the default (manual) launch plans exist.

Current schedules:
- WF1 Data Ingestion: Daily at 06:00 UTC (all 49 Phase 2 symbols, before EU market open)
- WF2 Universe Screening: Weekly Sunday at 08:00 UTC (49 Phase 2 symbols, full screening)
- WF3 Signal Analysis: Weekly Sunday at 12:00 UTC (after WF2, top quintiles, 50/50 tech+fund)
- WF4 Portfolio Rebalancing: Weekly Sunday at 16:00 UTC (after WF3, order reports only)
"""

from flytekit import CronSchedule, LaunchPlan

from src.shared.config import (
    PHASE2_SYMBOLS,
    WF3_MAX_QUINTILE, WF3_LOOKBACK_DAYS,
    WF3_TECH_WEIGHT, WF3_FUND_WEIGHT,
    WF3_SMA_SHORT, WF3_SMA_LONG,
    WF4_INITIAL_CAPITAL, WF4_MAX_POSITION_PCT, WF4_MAX_SECTOR_PCT,
    WF4_CASH_RESERVE_PCT, WF4_COMMISSION_PER_SHARE, WF4_EXCHANGE_FEE_BPS,
    WF4_IMPACT_BPS_PER_1K, WF4_MIN_TRADE_VALUE, WF4_PAPER_TRADING_ENABLED,
)
from src.wf1_data_ingestion.workflow import data_ingestion_workflow
from src.wf2_universe_screening.workflow import universe_screening_workflow
from src.wf3_signal_analysis.workflow import signal_analysis_workflow
from src.wf4_portfolio_rebalancing.workflow import portfolio_rebalancing_workflow

# WF1 Data Ingestion - daily at 06:00 UTC (all 49 Phase 2 symbols)
# Expanded from PHASE1 (10) to PHASE2 (49) so WF2 has full universe data
wf1_prod_daily = LaunchPlan.get_or_create(
    name="wf1_data_ingestion_prod_daily",
    workflow=data_ingestion_workflow,
    default_inputs={"symbols": PHASE2_SYMBOLS, "date": ""},
    schedule=CronSchedule(schedule="0 6 * * *"),
)

# WF2 Universe Screening - weekly Sunday at 08:00 UTC (49 Phase 2 symbols)
# Runs after WF1 has accumulated a week of data
wf2_prod_weekly = LaunchPlan.get_or_create(
    name="wf2_universe_screening_prod_weekly",
    workflow=universe_screening_workflow,
    default_inputs={
        "symbols": PHASE2_SYMBOLS,
        "lookback_days": 252,
        "run_date": "",
    },
    schedule=CronSchedule(schedule="0 8 * * 0"),
)

# WF3 Signal Analysis - weekly Sunday at 12:00 UTC (4 hours after WF2)
# Analyzes top quintile stocks from WF2 with 50% tech + 50% fundamental signals
wf3_prod_weekly = LaunchPlan.get_or_create(
    name="wf3_signal_analysis_prod_weekly",
    workflow=signal_analysis_workflow,
    default_inputs={
        "run_date": "",
        "max_quintile": WF3_MAX_QUINTILE,
        "lookback_days": WF3_LOOKBACK_DAYS,
        "tech_weight": WF3_TECH_WEIGHT,
        "fund_weight": WF3_FUND_WEIGHT,
        "sma_short": WF3_SMA_SHORT,
        "sma_long": WF3_SMA_LONG,
    },
    schedule=CronSchedule(schedule="0 12 * * 0"),
)

# WF4 Portfolio Rebalancing - weekly Sunday at 16:00 UTC (4 hours after WF3)
# Reads WF3 signal results, computes target portfolio, generates order report
# ORDER REPORTS ONLY — does not execute trades automatically
wf4_prod_weekly = LaunchPlan.get_or_create(
    name="wf4_portfolio_rebalancing_prod_weekly",
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
    },
    schedule=CronSchedule(schedule="0 16 * * 0"),
)
