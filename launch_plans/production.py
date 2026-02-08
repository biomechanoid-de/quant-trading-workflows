"""Launch plans for production domain.

Production schedules run at specific times with full data sets.
WF1: Daily at 06:00 UTC (before European market open).
"""

from flytekit import CronSchedule, LaunchPlan

from src.shared.config import PHASE1_SYMBOLS
from src.wf1_data_ingestion.workflow import data_ingestion_workflow

# WF1 Data Ingestion - daily at 06:00 UTC in PROD (all 10 symbols)
wf1_prod_daily = LaunchPlan.get_or_create(
    name="wf1_data_ingestion_prod_daily",
    workflow=data_ingestion_workflow,
    default_inputs={"symbols": PHASE1_SYMBOLS, "date": ""},
    schedule=CronSchedule(schedule="0 6 * * *"),
)
