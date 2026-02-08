"""WF4: Portfolio & Rebalancing - Workflow (Phase 3).

Planned pipeline:
    load_current_portfolio -> calculate_target_weights -> estimate_transaction_costs
    -> determine_trades -> generate_order_report -> send_notification

Key design: System generates ORDER REPORTS, not automatic trades.
You decide whether and how to execute. Like the Norwegian Pension Fund's
investment committee approach.
"""

from flytekit import workflow

from src.wf4_portfolio_rebalancing.tasks import portfolio_rebalancing_placeholder


@workflow
def portfolio_rebalancing_workflow() -> str:
    """WF4: Monthly portfolio rebalancing workflow (stub).

    Phase 3: Target weights, transaction costs, order reports.
    """
    return portfolio_rebalancing_placeholder()
