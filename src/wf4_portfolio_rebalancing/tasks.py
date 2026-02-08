"""WF4: Portfolio & Rebalancing - Tasks (Phase 3).

Monthly pipeline that calculates target weights, estimates transaction costs,
determines trades, and generates order reports.

Schedule: Monthly, 1st Monday 10:00 UTC
Node: Any Pi 4 Worker

Planned tasks (Phase 3):
- load_current_portfolio: Load positions from PostgreSQL
- calculate_target_weights: Pension fund principles (max 5%/stock, regional diversification)
- estimate_transaction_costs: Brenndoerfer model (commission + spread + market impact)
- determine_trades: Only trades with positive expected net alpha
- generate_order_report: Markdown/PDF report to MinIO
- send_notification: Alert on completion
"""

from flytekit import task, Resources


@task(
    requests=Resources(cpu="100m", mem="128Mi"),
    limits=Resources(cpu="200m", mem="256Mi"),
)
def portfolio_rebalancing_placeholder() -> str:
    """Placeholder for WF4 portfolio rebalancing tasks.

    Will be replaced with actual implementation in Phase 3:
    - Target weights: Max 5%/stock, US 40%, EU 30%, APAC 20%, EM 10%
    - Sector caps: Max 25%/sector
    - Transaction costs: Commission ($0.005/share) + Exchange (3 bps) + Half-spread + Impact
    - Order reports in MinIO
    """
    return "WF4: Portfolio & Rebalancing - Not implemented yet (Phase 3)"
