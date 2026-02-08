"""WF5: Monitoring & Reporting - Tasks (Phase 5).

Daily pipeline that calculates P&L, computes risk metrics,
updates Grafana dashboards, and checks alert conditions.

Schedule: Daily 18:00 UTC (after US market close)
Node: Any Pi 4 Worker

Planned tasks (Phase 5):
- calculate_pnl: Daily, MTD, YTD P&L
- risk_metrics: VaR (95%), Sharpe ratio, max drawdown, sector concentration
- update_grafana: Push metrics to Prometheus for Grafana dashboards
- check_alerts: Drawdown >5%, position >7%, unusual volatility
- daily_summary: Generate and store daily report
"""

from flytekit import task, Resources


@task(
    requests=Resources(cpu="100m", mem="128Mi"),
    limits=Resources(cpu="200m", mem="256Mi"),
)
def monitoring_placeholder() -> str:
    """Placeholder for WF5 monitoring and reporting tasks.

    Will be replaced with actual implementation in Phase 5:
    - P&L calculation (daily, MTD, YTD)
    - Risk metrics (VaR, Sharpe, drawdown, correlation)
    - Grafana dashboard updates via Prometheus
    - Alert system (drawdown, concentration, volatility)
    - Sense HAT LED matrix status (green/red)
    """
    return "WF5: Monitoring & Reporting - Not implemented yet (Phase 5)"
