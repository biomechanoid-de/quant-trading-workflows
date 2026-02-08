"""Launch plans for production domain.

Only define launch plans here that should run on a schedule in production.
CI/CD registers this file to the production domain on push to main.
If no launch plans are defined here, only the default (manual) launch plans exist.

Current schedules:
- WF1 Data Ingestion: Daily at 06:00 UTC (all 10 Phase 1 symbols, before EU market open)
- WF2 Universe Screening: Weekly Sunday at 08:00 UTC (49 Phase 2 symbols, full screening)
"""

from flytekit import CronSchedule, LaunchPlan

from src.shared.config import PHASE1_SYMBOLS, PHASE2_SYMBOLS
from src.wf1_data_ingestion.workflow import data_ingestion_workflow
from src.wf2_universe_screening.workflow import universe_screening_workflow

# WF1 Data Ingestion - daily at 06:00 UTC (all 10 Phase 1 symbols)
wf1_prod_daily = LaunchPlan.get_or_create(
    name="wf1_data_ingestion_prod_daily",
    workflow=data_ingestion_workflow,
    default_inputs={"symbols": PHASE1_SYMBOLS, "date": ""},
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
