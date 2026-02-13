"""WF4: Portfolio & Rebalancing - Tasks (Phase 3).

Weekly pipeline that reads WF3 signal results, calculates target portfolio
weights with pension fund principles, estimates transaction costs, generates
proposed trade orders, and produces a markdown order report to MinIO.

Schedule: Weekly Sunday 16:00 UTC (after WF3 at 12:00 UTC)
Node: Any Pi 4 Worker

ORDER REPORTS ONLY — does not execute trades automatically.
# TODO (Phase 4): Add paper trading mode that updates the positions table
# after each rebalancing run, enabling position tracking and differential
# order generation across runs. See Position dataclass in shared/models.py.

Tasks:
1. resolve_run_date: Resolve empty date to latest WF3 signal run
2. load_signal_context: Load WF3 signal results from PostgreSQL
3. load_current_portfolio: Load current positions (empty on first run)
4. calculate_target_weights: Compute target weights with constraints
5. fetch_current_prices: Get latest close prices + spreads
6. generate_trade_orders: Diff current vs target, create TradeOrder list
7. assemble_rebalancing_result: Package all results for storage/reporting
8. store_rebalancing_to_db: Write to rebalancing_runs + trades tables
9. generate_order_report: Create markdown report and upload to MinIO
"""

from typing import Dict, List

from flytekit import task, Resources

from src.shared.models import TradeOrder


# ============================================================
# Task 1: Resolve Run Date
# ============================================================

@task(
    requests=Resources(cpu="100m", mem="128Mi"),
    limits=Resources(cpu="200m", mem="256Mi"),
)
def resolve_run_date(run_date: str) -> str:
    """Resolve empty run_date to the latest WF3 signal run date.

    Queries signal_runs table for MAX(run_date). Falls back to today
    if no signal runs exist.

    Args:
        run_date: Target date (YYYY-MM-DD) or empty string.

    Returns:
        Resolved date string (YYYY-MM-DD).
    """
    if run_date:
        return run_date

    from src.shared.db import get_connection
    from datetime import date

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT MAX(run_date) FROM signal_runs")
        result = cursor.fetchone()
        if result and result[0]:
            return str(result[0])
        return str(date.today())
    finally:
        cursor.close()
        conn.close()


# ============================================================
# Task 2: Load Signal Context (PARALLEL with Task 3)
# ============================================================

@task(
    requests=Resources(cpu="200m", mem="256Mi"),
    limits=Resources(cpu="500m", mem="512Mi"),
)
def load_signal_context(run_date: str) -> Dict[str, str]:
    """Load WF3 signal results for portfolio construction.

    Reads signal_results for the given run_date. Returns Dict[str, str]
    where each value is a JSON string with signal data for one symbol.

    Args:
        run_date: Signal run date (YYYY-MM-DD).

    Returns:
        Dict mapping symbol -> JSON string with keys:
        combined_signal_score, signal_strength, wf2_quintile,
        technical_score, fundamental_score, data_quality.
    """
    import json
    from src.shared.db import get_latest_signal_results

    rows = get_latest_signal_results(run_date)
    result = {}
    for row in rows:
        symbol, combined_score, strength, quintile, tech, fund, quality = row
        result[symbol] = json.dumps({
            "combined_signal_score": float(combined_score) if combined_score else 50.0,
            "signal_strength": strength or "hold",
            "wf2_quintile": int(quintile) if quintile else 3,
            "technical_score": float(tech) if tech else 50.0,
            "fundamental_score": float(fund) if fund else 50.0,
            "data_quality": quality or "minimal",
        })

    return result


# ============================================================
# Task 3: Load Current Portfolio (PARALLEL with Task 2)
# ============================================================

@task(
    requests=Resources(cpu="100m", mem="128Mi"),
    limits=Resources(cpu="200m", mem="256Mi"),
)
def load_current_portfolio(initial_capital: float) -> Dict[str, str]:
    """Load current portfolio positions from PostgreSQL.

    On first run (empty positions table), returns initial_capital as cash
    with no positions.

    # TODO (Phase 4): When paper trading is enabled, positions table will
    # be populated by previous WF4 runs, and this task will return actual
    # portfolio state with positions and remaining cash.

    Args:
        initial_capital: Starting capital in EUR (default: 25000).

    Returns:
        Dict with keys:
        - "cash": str(remaining cash)
        - "positions_json": JSON array of position dicts
        - "total_value": str(total portfolio value)
    """
    import json
    from src.shared.db import get_current_positions

    positions = get_current_positions()

    if not positions:
        # First run or no positions: all cash
        return {
            "cash": str(initial_capital),
            "positions_json": json.dumps([]),
            "total_value": str(initial_capital),
        }

    # Calculate total value from positions + infer remaining cash
    position_list = []
    invested_value = 0.0
    for symbol, shares, avg_cost, current_price, sector in positions:
        shares_f = float(shares) if shares else 0.0
        price_f = float(current_price) if current_price else float(avg_cost)
        position_list.append({
            "symbol": symbol,
            "shares": shares_f,
            "avg_cost": float(avg_cost) if avg_cost else 0.0,
            "current_price": price_f,
            "sector": sector or "",
        })
        invested_value += shares_f * price_f

    # Cash = initial_capital - invested (simplified for order-report-only mode)
    cash = max(0.0, initial_capital - invested_value)
    total_value = cash + invested_value

    return {
        "cash": str(round(cash, 2)),
        "positions_json": json.dumps(position_list),
        "total_value": str(round(total_value, 2)),
    }


# ============================================================
# Task 4: Calculate Target Weights
# ============================================================

@task(
    requests=Resources(cpu="300m", mem="256Mi"),
    limits=Resources(cpu="500m", mem="512Mi"),
)
def calculate_target_weights(
    signal_context: Dict[str, str],
    portfolio_state: Dict[str, str],
    max_position_pct: float,
    max_sector_pct: float,
    cash_reserve_pct: float,
) -> Dict[str, str]:
    """Calculate target portfolio weights from signal results.

    Uses pension fund principles:
    - strong_buy: 3x base allocation
    - buy: 2x base allocation
    - hold/sell/strong_sell: 0 (exit if held)
    - Max 5% per stock, max 25% per sector
    - 5% cash reserve

    Args:
        signal_context: Dict[str, str] from load_signal_context.
        portfolio_state: Dict[str, str] from load_current_portfolio.
        max_position_pct: Maximum weight per stock.
        max_sector_pct: Maximum weight per sector.
        cash_reserve_pct: Target cash reserve fraction.

    Returns:
        Dict with keys:
        - "target_weights_json": JSON dict {symbol: weight}
        - "exit_symbols_json": JSON list of symbols to sell
        - "num_target_positions": str
        - "investable_value": str
    """
    import json
    from src.shared.analytics import calculate_signal_weights
    from src.shared.config import SYMBOL_SECTORS

    # Parse signal context
    signals = []
    for symbol, signal_json in signal_context.items():
        data = json.loads(signal_json)
        signals.append({
            "symbol": symbol,
            "signal_strength": data["signal_strength"],
            "combined_signal_score": data["combined_signal_score"],
        })

    # Calculate target weights
    weights = calculate_signal_weights(
        signals=signals,
        max_position_pct=max_position_pct,
        max_sector_pct=max_sector_pct,
        cash_reserve_pct=cash_reserve_pct,
        sector_map=SYMBOL_SECTORS,
    )

    # Determine exit symbols: currently held but not in target weights
    current_positions = json.loads(portfolio_state.get("positions_json", "[]"))
    current_symbols = {p["symbol"] for p in current_positions if p.get("shares", 0) > 0}
    target_symbols = set(weights.keys())
    exit_symbols = sorted(current_symbols - target_symbols)

    # Investable value
    total_value = float(portfolio_state.get("total_value", "0"))
    investable = total_value * (1.0 - cash_reserve_pct)

    return {
        "target_weights_json": json.dumps(weights),
        "exit_symbols_json": json.dumps(exit_symbols),
        "num_target_positions": str(len(weights)),
        "investable_value": str(round(investable, 2)),
    }


# ============================================================
# Task 5: Fetch Current Prices
# ============================================================

@task(
    requests=Resources(cpu="200m", mem="256Mi"),
    limits=Resources(cpu="500m", mem="512Mi"),
)
def fetch_current_prices(
    target_weights: Dict[str, str],
    portfolio_state: Dict[str, str],
) -> Dict[str, str]:
    """Fetch latest close prices for all target + current position symbols.

    Queries market_data table for the most recent close price and spread
    per symbol.

    Args:
        target_weights: Dict from calculate_target_weights.
        portfolio_state: Dict from load_current_portfolio.

    Returns:
        Dict mapping symbol -> JSON{"close": float, "spread_bps": float}.
    """
    import json
    from src.shared.db import get_latest_market_data

    # Collect all symbols that need prices
    weights = json.loads(target_weights.get("target_weights_json", "{}"))
    exits = json.loads(target_weights.get("exit_symbols_json", "[]"))
    positions = json.loads(portfolio_state.get("positions_json", "[]"))

    all_symbols = set(weights.keys()) | set(exits)
    for p in positions:
        if p.get("shares", 0) > 0:
            all_symbols.add(p["symbol"])

    # Fetch latest price for each symbol
    result = {}
    for symbol in sorted(all_symbols):
        rows = get_latest_market_data(symbol, days=5)
        if rows:
            # rows: [(symbol, date, close, volume, spread_bps), ...]
            _, _, close, _, spread_bps = rows[0]
            result[symbol] = json.dumps({
                "close": float(close) if close else 0.0,
                "spread_bps": float(spread_bps) if spread_bps else 5.0,
            })
        else:
            result[symbol] = json.dumps({"close": 0.0, "spread_bps": 5.0})

    return result


# ============================================================
# Task 6: Generate Trade Orders
# ============================================================

@task(
    requests=Resources(cpu="300m", mem="256Mi"),
    limits=Resources(cpu="500m", mem="512Mi"),
)
def generate_trade_orders(
    target_weights: Dict[str, str],
    portfolio_state: Dict[str, str],
    price_data: Dict[str, str],
    commission_per_share: float,
    exchange_fee_bps: float,
    impact_bps_per_1k: float,
    min_trade_value: float,
) -> List[TradeOrder]:
    """Generate proposed trade orders by diffing current vs. target.

    For each symbol:
    - Calculate target_shares = target_weight * total_value / price
    - Calculate diff = target_shares - current_shares
    - BUY if diff > 0, SELL if diff < 0
    - Skip if |diff * price| < min_trade_value

    Args:
        target_weights: Dict from calculate_target_weights.
        portfolio_state: Dict from load_current_portfolio.
        price_data: Dict from fetch_current_prices.
        commission_per_share: Per-share commission USD.
        exchange_fee_bps: Exchange fee bps.
        impact_bps_per_1k: Market impact per $1000.
        min_trade_value: Skip trades below this EUR value.

    Returns:
        List[TradeOrder] (primitive-only fields, Flytekit-safe).
    """
    import json
    import math
    from src.shared.analytics import estimate_transaction_cost

    weights = json.loads(target_weights.get("target_weights_json", "{}"))
    exits = json.loads(target_weights.get("exit_symbols_json", "[]"))
    positions = json.loads(portfolio_state.get("positions_json", "[]"))
    total_value = float(portfolio_state.get("total_value", "0"))

    # Build current holdings map
    current_holdings = {}
    for p in positions:
        if p.get("shares", 0) > 0:
            current_holdings[p["symbol"]] = float(p["shares"])

    orders = []

    # BUY orders for target positions
    for symbol, weight in weights.items():
        price_info = json.loads(price_data.get(symbol, '{"close": 0, "spread_bps": 5}'))
        price = price_info["close"]
        spread_bps = price_info["spread_bps"]

        if price <= 0:
            continue

        target_value = total_value * weight
        target_shares = math.floor(target_value / price)
        current_shares = current_holdings.get(symbol, 0)
        diff = target_shares - current_shares

        if diff <= 0:
            continue

        trade_value = diff * price
        if trade_value < min_trade_value:
            continue

        cost_bps = estimate_transaction_cost(
            quantity=diff,
            price=price,
            spread_bps=spread_bps,
            commission_per_share=commission_per_share,
            exchange_fee_bps=exchange_fee_bps,
            impact_bps_per_1k=impact_bps_per_1k,
        )

        reason = "NewEntry" if symbol not in current_holdings else "Rebalance"

        orders.append(TradeOrder(
            symbol=symbol,
            side="BUY",
            quantity=diff,
            estimated_price=round(price, 2),
            estimated_cost_bps=round(cost_bps, 2),
            reason=reason,
        ))

    # SELL orders for exit positions
    for symbol in exits:
        current_shares = current_holdings.get(symbol, 0)
        if current_shares <= 0:
            continue

        price_info = json.loads(price_data.get(symbol, '{"close": 0, "spread_bps": 5}'))
        price = price_info["close"]
        spread_bps = price_info["spread_bps"]

        if price <= 0:
            continue

        cost_bps = estimate_transaction_cost(
            quantity=int(current_shares),
            price=price,
            spread_bps=spread_bps,
            commission_per_share=commission_per_share,
            exchange_fee_bps=exchange_fee_bps,
            impact_bps_per_1k=impact_bps_per_1k,
        )

        orders.append(TradeOrder(
            symbol=symbol,
            side="SELL",
            quantity=int(current_shares),
            estimated_price=round(price, 2),
            estimated_cost_bps=round(cost_bps, 2),
            reason="Exit",
        ))

    # SELL orders for current positions that need reduction
    for symbol, current_shares in current_holdings.items():
        if symbol in exits:
            continue  # Already handled above
        if symbol not in weights:
            continue

        price_info = json.loads(price_data.get(symbol, '{"close": 0, "spread_bps": 5}'))
        price = price_info["close"]
        spread_bps = price_info["spread_bps"]

        if price <= 0:
            continue

        target_value = total_value * weights[symbol]
        target_shares = math.floor(target_value / price)
        diff = current_shares - target_shares

        if diff <= 0:
            continue

        trade_value = diff * price
        if trade_value < min_trade_value:
            continue

        cost_bps = estimate_transaction_cost(
            quantity=int(diff),
            price=price,
            spread_bps=spread_bps,
            commission_per_share=commission_per_share,
            exchange_fee_bps=exchange_fee_bps,
            impact_bps_per_1k=impact_bps_per_1k,
        )

        orders.append(TradeOrder(
            symbol=symbol,
            side="SELL",
            quantity=int(diff),
            estimated_price=round(price, 2),
            estimated_cost_bps=round(cost_bps, 2),
            reason="Rebalance",
        ))

    # Sort: BUY orders first (alphabetical), then SELL orders
    orders.sort(key=lambda o: (0 if o.side == "BUY" else 1, o.symbol))
    return orders


# ============================================================
# Task 7: Assemble Rebalancing Result
# ============================================================

@task(
    requests=Resources(cpu="200m", mem="256Mi"),
    limits=Resources(cpu="500m", mem="512Mi"),
)
def assemble_rebalancing_result(
    run_date: str,
    target_weights: Dict[str, str],
    portfolio_state: Dict[str, str],
    trade_orders: List[TradeOrder],
    signal_context: Dict[str, str],
) -> Dict[str, str]:
    """Assemble all rebalancing results into a serialized dict.

    Packages everything needed for DB storage and report generation
    into a Dict[str, str] for safe passing to downstream tasks.

    Args:
        run_date: Resolved run date.
        target_weights: Dict from calculate_target_weights.
        portfolio_state: Dict from load_current_portfolio.
        trade_orders: List[TradeOrder] from generate_trade_orders.
        signal_context: Dict from load_signal_context.

    Returns:
        Dict[str, str] with all serialized rebalancing data.
    """
    import json

    weights = json.loads(target_weights.get("target_weights_json", "{}"))

    buy_orders = [o for o in trade_orders if o.side == "BUY"]
    sell_orders = [o for o in trade_orders if o.side == "SELL"]

    total_cost_bps = sum(o.estimated_cost_bps for o in trade_orders)

    # Serialize trade orders
    orders_json = json.dumps([
        {
            "symbol": o.symbol,
            "side": o.side,
            "quantity": o.quantity,
            "estimated_price": o.estimated_price,
            "estimated_cost_bps": o.estimated_cost_bps,
            "reason": o.reason,
        }
        for o in trade_orders
    ])

    return {
        "run_date": run_date,
        "total_value": portfolio_state.get("total_value", "0"),
        "cash_value": portfolio_state.get("cash_value", portfolio_state.get("cash", "0")),
        "invested_value": target_weights.get("investable_value", "0"),
        "num_signals_input": str(len(signal_context)),
        "num_target_positions": target_weights.get("num_target_positions", "0"),
        "num_buy_orders": str(len(buy_orders)),
        "num_sell_orders": str(len(sell_orders)),
        "total_estimated_cost_bps": str(round(total_cost_bps, 4)),
        "target_weights_json": target_weights.get("target_weights_json", "{}"),
        "exit_symbols_json": target_weights.get("exit_symbols_json", "[]"),
        "trade_orders_json": orders_json,
        "signal_context_json": json.dumps({
            sym: json.loads(data) for sym, data in signal_context.items()
        }),
    }


# ============================================================
# Task 8: Store to DB (PARALLEL with Task 9)
# ============================================================

@task(
    requests=Resources(cpu="200m", mem="256Mi"),
    limits=Resources(cpu="500m", mem="512Mi"),
)
def store_rebalancing_to_db(assembled_result: Dict[str, str]) -> str:
    """Store rebalancing results to PostgreSQL.

    Writes to rebalancing_runs + trades tables.
    Uses UPSERT for idempotency on Flyte retries.

    Args:
        assembled_result: Dict from assemble_rebalancing_result.

    Returns:
        Summary string.
    """
    import json
    from src.shared.db import store_rebalancing_results
    from src.shared.models import TradeOrder

    run_date = assembled_result["run_date"]
    orders_data = json.loads(assembled_result.get("trade_orders_json", "[]"))

    # Reconstruct TradeOrder objects
    trade_orders = [
        TradeOrder(
            symbol=o["symbol"],
            side=o["side"],
            quantity=o["quantity"],
            estimated_price=o["estimated_price"],
            estimated_cost_bps=o["estimated_cost_bps"],
            reason=o["reason"],
        )
        for o in orders_data
    ]

    run_metadata = {
        "total_portfolio_value": float(assembled_result.get("total_value", 0)),
        "cash_value": float(assembled_result.get("cash_value", 0)),
        "invested_value": float(assembled_result.get("invested_value", 0)),
        "num_signals_input": int(assembled_result.get("num_signals_input", 0)),
        "num_target_positions": int(assembled_result.get("num_target_positions", 0)),
        "num_buy_orders": int(assembled_result.get("num_buy_orders", 0)),
        "num_sell_orders": int(assembled_result.get("num_sell_orders", 0)),
        "total_estimated_cost": float(assembled_result.get("total_estimated_cost_bps", 0)),
        "report_s3_path": "",
    }

    rows = store_rebalancing_results(run_date, run_metadata, trade_orders)
    return (
        f"WF4 stored: {rows} trade orders for {run_date} "
        f"(value: EUR {run_metadata['total_portfolio_value']:,.2f})"
    )


# ============================================================
# Task 9: Generate Order Report (PARALLEL with Task 8)
# ============================================================

@task(
    requests=Resources(cpu="200m", mem="256Mi"),
    limits=Resources(cpu="500m", mem="512Mi"),
)
def generate_order_report(assembled_result: Dict[str, str]) -> str:
    """Generate markdown order report and upload to MinIO.

    Creates a comprehensive rebalancing report with:
    - Portfolio summary (value, cash, invested)
    - Signal summary (count by strength)
    - Target weights table
    - Proposed trades table
    - Cost analysis
    - Sector allocation
    - Risk notes

    Uploads to: s3://quant-data/reports/wf4/year=YYYY/month=MM/day=DD/order_report.md

    Args:
        assembled_result: Dict from assemble_rebalancing_result.

    Returns:
        The markdown report as a string.
    """
    import json
    from src.shared.config import SYMBOL_SECTORS

    run_date = assembled_result["run_date"]
    total_value = float(assembled_result.get("total_value", 0))
    cash_value = float(assembled_result.get("cash_value", 0))
    invested_value = float(assembled_result.get("invested_value", 0))
    num_signals = int(assembled_result.get("num_signals_input", 0))
    num_targets = int(assembled_result.get("num_target_positions", 0))
    num_buys = int(assembled_result.get("num_buy_orders", 0))
    num_sells = int(assembled_result.get("num_sell_orders", 0))
    total_cost_bps = float(assembled_result.get("total_estimated_cost_bps", 0))

    weights = json.loads(assembled_result.get("target_weights_json", "{}"))
    orders = json.loads(assembled_result.get("trade_orders_json", "[]"))
    signal_ctx = json.loads(assembled_result.get("signal_context_json", "{}"))

    # Build report
    lines = []
    lines.append(f"# WF4 Portfolio Rebalancing Report - {run_date}")
    lines.append("")

    # Portfolio Summary
    cash_pct = (cash_value / total_value * 100) if total_value > 0 else 0
    lines.append("## Portfolio Summary")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Portfolio Value | EUR {total_value:,.2f} |")
    lines.append(f"| Cash | EUR {cash_value:,.2f} ({cash_pct:.1f}%) |")
    lines.append(f"| Investable | EUR {invested_value:,.2f} |")
    lines.append(f"| Signal Results (WF3) | {num_signals} |")
    lines.append(f"| Target Positions | {num_targets} |")
    lines.append("")

    # Signal Summary
    strength_counts = {}
    for sym, data in signal_ctx.items():
        strength = data.get("signal_strength", "hold")
        strength_counts[strength] = strength_counts.get(strength, 0) + 1

    lines.append("## Signal Summary")
    lines.append("| Strength | Count |")
    lines.append("|----------|-------|")
    for strength in ["strong_buy", "buy", "hold", "sell", "strong_sell"]:
        count = strength_counts.get(strength, 0)
        if count > 0:
            lines.append(f"| {strength} | {count} |")
    lines.append("")

    # Target Weights
    if weights:
        lines.append("## Target Weights")
        lines.append("| Symbol | Sector | Signal | Score | Weight | Value (EUR) |")
        lines.append("|--------|--------|--------|-------|--------|-------------|")
        for sym in sorted(weights, key=lambda s: weights[s], reverse=True):
            w = weights[sym]
            sector = SYMBOL_SECTORS.get(sym, "Unknown")
            sig_data = signal_ctx.get(sym, {})
            strength = sig_data.get("signal_strength", "?")
            score = sig_data.get("combined_signal_score", 0)
            value = total_value * w
            lines.append(
                f"| {sym} | {sector} | {strength} | {score:.1f} | "
                f"{w*100:.2f}% | {value:,.2f} |"
            )
        lines.append("")

    # Proposed Trades
    lines.append("## Proposed Trades")
    if orders:
        lines.append("| Symbol | Side | Qty | Price (USD) | Est. Cost (bps) | Reason |")
        lines.append("|--------|------|-----|-------------|-----------------|--------|")
        for o in orders:
            lines.append(
                f"| {o['symbol']} | {o['side']} | {o['quantity']} | "
                f"${o['estimated_price']:,.2f} | {o['estimated_cost_bps']:.1f} | "
                f"{o['reason']} |"
            )
        lines.append("")
        lines.append(f"**Total orders:** {num_buys} BUY, {num_sells} SELL")
        total_cost_eur = total_value * total_cost_bps / 10000
        lines.append(
            f"**Estimated total transaction cost:** {total_cost_bps:.2f} bps "
            f"(EUR {total_cost_eur:,.2f})"
        )
    else:
        lines.append("*No trades needed — portfolio is on target.*")
    lines.append("")

    # Sector Allocation
    if weights:
        sector_alloc = {}
        sector_count = {}
        for sym, w in weights.items():
            sector = SYMBOL_SECTORS.get(sym, "Unknown")
            sector_alloc[sector] = sector_alloc.get(sector, 0) + w
            sector_count[sector] = sector_count.get(sector, 0) + 1

        lines.append("## Sector Allocation")
        lines.append("| Sector | Weight | # Stocks |")
        lines.append("|--------|--------|----------|")
        for sector in sorted(sector_alloc, key=lambda s: sector_alloc[s], reverse=True):
            lines.append(
                f"| {sector} | {sector_alloc[sector]*100:.1f}% | "
                f"{sector_count[sector]} |"
            )
        lines.append("")

    # Risk Notes
    lines.append("## Risk Notes")
    lines.append("- All positions are US equities (no regional diversification yet)")
    if weights:
        max_weight = max(weights.values())
        max_sym = max(weights, key=weights.get)
        lines.append(f"- Largest position: {max_sym} at {max_weight*100:.1f}%")
        if sector_alloc:
            max_sector = max(sector_alloc, key=sector_alloc.get)
            lines.append(
                f"- Largest sector: {max_sector} at "
                f"{sector_alloc[max_sector]*100:.1f}%"
            )
    lines.append("")
    lines.append("---")
    lines.append(
        "*Generated by WF4 Portfolio Rebalancing | "
        "Order report mode | Not investment advice*"
    )

    report = "\n".join(lines)

    # Upload to MinIO
    try:
        _upload_report_to_minio(report, run_date)
    except Exception as e:
        # Don't fail the task if MinIO upload fails
        report += f"\n\n*Warning: MinIO upload failed: {e}*"

    return report


def _upload_report_to_minio(report: str, run_date: str) -> str:
    """Upload markdown report to MinIO (helper function).

    Args:
        report: Markdown content.
        run_date: Run date for path construction.

    Returns:
        S3 path where the report was uploaded.
    """
    from src.shared.storage import get_s3_client
    from src.shared.config import S3_DATA_BUCKET

    year, month, day = run_date.split("-")
    s3_key = f"reports/wf4/year={year}/month={month}/day={day}/order_report.md"

    client = get_s3_client()
    client.put_object(
        Bucket=S3_DATA_BUCKET,
        Key=s3_key,
        Body=report.encode("utf-8"),
        ContentType="text/markdown",
    )
    return f"s3://{S3_DATA_BUCKET}/{s3_key}"
