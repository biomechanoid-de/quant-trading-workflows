"""Launch plans for development domain.

Development schedules run more frequently with smaller data sets
for fast iteration and testing.
"""

from flytekit import CronSchedule, LaunchPlan

from src.shared.config import DEV_SYMBOLS
from src.wf1_data_ingestion.workflow import data_ingestion_workflow

# WF1 Data Ingestion - every 6 hours in DEV (3 test symbols)
wf1_dev_schedule = LaunchPlan.get_or_create(
    name="wf1_data_ingestion_dev_6h",
    workflow=data_ingestion_workflow,
    default_inputs={"symbols": DEV_SYMBOLS, "date": ""},
    schedule=CronSchedule(schedule="0 */6 * * *"),
)
