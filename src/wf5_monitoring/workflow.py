"""WF5: Monitoring & Reporting - Workflow (Phase 5).

Planned pipeline:
    calculate_pnl -> risk_metrics -> update_grafana -> check_alerts -> daily_summary

Outputs:
- DailyReport with P&L, risk metrics, winners/losers, alerts
- Prometheus metrics for Grafana Trading Dashboard
- Sense HAT LED matrix update (portfolio status)
"""

from flytekit import workflow

from src.wf5_monitoring.tasks import monitoring_placeholder


@workflow
def monitoring_workflow() -> str:
    """WF5: Daily monitoring and reporting workflow (stub).

    Phase 5: P&L, risk metrics, Grafana, alerts, Sense HAT.
    """
    return monitoring_placeholder()
