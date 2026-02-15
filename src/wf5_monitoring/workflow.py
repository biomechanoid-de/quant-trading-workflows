"""WF5: Monitoring & Reporting - Workflow (Phase 5).

Pipeline:
    calculate_pnl -> compute_risk_metrics -> check_alerts -> generate_monitoring_report

Reads WF4 portfolio snapshots and positions, computes P&L and risk metrics,
checks alert thresholds, and generates a weekly markdown monitoring report
stored to PostgreSQL and MinIO.
"""

from flytekit import workflow

from src.shared.config import (
    WF5_LOOKBACK_DAYS,
    WF5_RISK_FREE_RATE,
    WF5_DRAWDOWN_ALERT_PCT,
    WF5_POSITION_ALERT_PCT,
    WF5_VAR_ALERT_PCT,
    WF5_LOSS_ALERT_PCT,
)
from src.wf5_monitoring.tasks import (
    calculate_pnl,
    compute_risk_metrics,
    check_alerts,
    generate_monitoring_report,
)


@workflow
def monitoring_workflow(
    run_date: str = "",
    lookback_days: int = WF5_LOOKBACK_DAYS,
    risk_free_rate: float = WF5_RISK_FREE_RATE,
    drawdown_threshold: float = WF5_DRAWDOWN_ALERT_PCT,
    position_threshold: float = WF5_POSITION_ALERT_PCT,
    var_threshold: float = WF5_VAR_ALERT_PCT,
    loss_threshold: float = WF5_LOSS_ALERT_PCT,
) -> str:
    """WF5: Weekly monitoring and reporting workflow.

    Reads portfolio snapshots and positions from WF4, computes P&L
    (daily/MTD/YTD), risk metrics (Sharpe, Sortino, drawdown, VaR),
    checks alert thresholds, and generates a markdown report.

    Args:
        run_date: Target date (YYYY-MM-DD). Empty = today.
        lookback_days: Risk metric lookback window (default: 30).
        risk_free_rate: Annualized risk-free rate (default: 5%).
        drawdown_threshold: Max 30d drawdown before alert (default: 5%).
        position_threshold: Max single position weight (default: 7%).
        var_threshold: Max VaR as % of portfolio (default: 3%).
        loss_threshold: Max unrealized loss per position (default: 10%).

    Returns:
        Monitoring report as markdown string.
    """
    # Step 1: Calculate P&L from portfolio snapshots + positions
    pnl_data = calculate_pnl(run_date=run_date, lookback_days=lookback_days)

    # Step 2: Compute risk metrics from snapshot time series
    risk_data = compute_risk_metrics(
        pnl_data=pnl_data,
        risk_free_rate=risk_free_rate,
        lookback_days=lookback_days,
    )

    # Step 3: Check alert conditions
    alert_data = check_alerts(
        pnl_data=pnl_data,
        risk_data=risk_data,
        drawdown_threshold=drawdown_threshold,
        position_threshold=position_threshold,
        var_threshold=var_threshold,
        loss_threshold=loss_threshold,
    )

    # Step 4: Generate report, store to DB + MinIO
    return generate_monitoring_report(
        pnl_data=pnl_data,
        risk_data=risk_data,
        alert_data=alert_data,
    )
