"""WF4: Portfolio & Rebalancing - Workflow (Phase 4).

Pipeline:
    resolve_run_date -> [load_signal_context || load_current_portfolio]
    -> calculate_target_weights -> fetch_current_prices
    -> generate_trade_orders -> assemble_rebalancing_result
    -> [store_rebalancing_to_db || store_rebalancing_to_parquet || generate_order_report]
    -> execute_paper_trades -> snapshot_portfolio

Key design: System generates ORDER REPORTS. When paper_trading=True,
also simulates trade execution, updates positions, and tracks performance.
Like the Norwegian Pension Fund's investment committee approach.
"""

from flytekit import workflow

from src.shared.config import (
    WF4_INITIAL_CAPITAL,
    WF4_MAX_POSITION_PCT,
    WF4_MAX_SECTOR_PCT,
    WF4_CASH_RESERVE_PCT,
    WF4_COMMISSION_PER_SHARE,
    WF4_EXCHANGE_FEE_BPS,
    WF4_IMPACT_BPS_PER_1K,
    WF4_MIN_TRADE_VALUE,
    WF4_PAPER_TRADING_ENABLED,
)
from src.wf4_portfolio_rebalancing.tasks import (
    resolve_run_date,
    load_signal_context,
    load_current_portfolio,
    calculate_target_weights,
    fetch_current_prices,
    generate_trade_orders,
    assemble_rebalancing_result,
    store_rebalancing_to_db,
    store_rebalancing_to_parquet,
    generate_order_report,
    execute_paper_trades,
    snapshot_portfolio,
)


@workflow
def portfolio_rebalancing_workflow(
    run_date: str = "",
    initial_capital: float = WF4_INITIAL_CAPITAL,
    max_position_pct: float = WF4_MAX_POSITION_PCT,
    max_sector_pct: float = WF4_MAX_SECTOR_PCT,
    cash_reserve_pct: float = WF4_CASH_RESERVE_PCT,
    commission_per_share: float = WF4_COMMISSION_PER_SHARE,
    exchange_fee_bps: float = WF4_EXCHANGE_FEE_BPS,
    impact_bps_per_1k: float = WF4_IMPACT_BPS_PER_1K,
    min_trade_value: float = WF4_MIN_TRADE_VALUE,
    paper_trading: bool = WF4_PAPER_TRADING_ENABLED,
) -> str:
    """WF4: Weekly portfolio rebalancing workflow.

    Reads WF3 signal results, computes target portfolio weights
    with pension fund principles (max 5% per stock, max 25% per sector),
    estimates transaction costs (Brenndoerfer model), generates trade
    orders, and produces a markdown order report stored to MinIO.

    Phase 4: When paper_trading=True, simulates trade execution,
    updates the positions table, and takes portfolio snapshots for
    performance tracking. When False (default), identical to Phase 3.

    Args:
        run_date: Target date (YYYY-MM-DD). Empty = latest signal run.
        initial_capital: Starting capital (default: EUR 25,000).
        max_position_pct: Max weight per stock (default: 5%).
        max_sector_pct: Max weight per sector (default: 25%).
        cash_reserve_pct: Target cash reserve (default: 5%).
        commission_per_share: Commission per share USD (default: 0.005).
        exchange_fee_bps: Exchange fee bps (default: 3.0).
        impact_bps_per_1k: Market impact per $1000 (default: 0.1).
        min_trade_value: Minimum trade value EUR (default: 100).
        paper_trading: Enable paper trade simulation (default: False).

    Returns:
        Order report as markdown string.
    """
    # Step 0: Resolve date
    resolved_date = resolve_run_date(run_date=run_date)

    # Step 1: PARALLEL — Load signal context + current portfolio
    signal_ctx = load_signal_context(run_date=resolved_date)
    portfolio_state = load_current_portfolio(initial_capital=initial_capital)

    # Step 2: Calculate target weights (depends on both parallel branches)
    target_weights = calculate_target_weights(
        signal_context=signal_ctx,
        portfolio_state=portfolio_state,
        max_position_pct=max_position_pct,
        max_sector_pct=max_sector_pct,
        cash_reserve_pct=cash_reserve_pct,
    )

    # Step 3: Fetch current prices
    price_data = fetch_current_prices(
        target_weights=target_weights,
        portfolio_state=portfolio_state,
    )

    # Step 4: Generate trade orders
    trade_orders = generate_trade_orders(
        target_weights=target_weights,
        portfolio_state=portfolio_state,
        price_data=price_data,
        commission_per_share=commission_per_share,
        exchange_fee_bps=exchange_fee_bps,
        impact_bps_per_1k=impact_bps_per_1k,
        min_trade_value=min_trade_value,
    )

    # Step 5: Assemble result
    assembled = assemble_rebalancing_result(
        run_date=resolved_date,
        target_weights=target_weights,
        portfolio_state=portfolio_state,
        trade_orders=trade_orders,
        signal_context=signal_ctx,
    )

    # Step 6: PARALLEL — Store to DB + Store to Parquet + Generate report
    store_result = store_rebalancing_to_db(assembled_result=assembled)
    store_parquet_result = store_rebalancing_to_parquet(assembled_result=assembled)
    report = generate_order_report(assembled_result=assembled)

    # Step 7: Paper trading execution (Phase 4)
    # When paper_trading=False, tasks short-circuit immediately.
    paper_result = execute_paper_trades(
        assembled_result=assembled,
        portfolio_state=portfolio_state,
        price_data=price_data,
        paper_trading=paper_trading,
        initial_capital=initial_capital,
        commission_per_share=commission_per_share,
        exchange_fee_bps=exchange_fee_bps,
        impact_bps_per_1k=impact_bps_per_1k,
    )

    # Step 8: Portfolio snapshot (Phase 4)
    snapshot_result = snapshot_portfolio(
        paper_trade_result=paper_result,
        paper_trading=paper_trading,
        initial_capital=initial_capital,
    )

    return report
