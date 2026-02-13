"""PostgreSQL database helpers for Quant Trading Workflows.

Target: PostgreSQL on pi5-1tb (192.168.178.45) with NVMe SSD.
Uses psycopg2 with lazy imports to avoid import errors in test environments.
"""

from src.shared.config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
from src.shared.models import MarketDataBatch


def get_connection():
    """Get a PostgreSQL database connection.

    Connection parameters are read from environment variables via config.py.
    """
    import psycopg2

    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def store_market_data(batch: MarketDataBatch) -> int:
    """Store a MarketDataBatch into the market_data table.

    Uses UPSERT (ON CONFLICT UPDATE) for idempotency.
    Flyte may retry failed tasks, so this ensures no duplicates.

    Returns:
        Number of rows inserted/updated.
    """
    conn = get_connection()
    cursor = conn.cursor()
    rows_inserted = 0

    try:
        for symbol in batch.symbols:
            if symbol in batch.prices:
                cursor.execute(
                    """INSERT INTO market_data
                           (symbol, date, close, volume, spread_bps, data_source)
                       VALUES (%s, %s, %s, %s, %s, %s)
                       ON CONFLICT (symbol, date) DO UPDATE SET
                           close = EXCLUDED.close,
                           volume = EXCLUDED.volume,
                           spread_bps = EXCLUDED.spread_bps,
                           data_source = EXCLUDED.data_source,
                           created_at = NOW()""",
                    (
                        symbol,
                        batch.date,
                        batch.prices.get(symbol),
                        batch.volumes.get(symbol),
                        batch.spreads.get(symbol),
                        "yfinance",
                    ),
                )
                rows_inserted += 1
        conn.commit()
    finally:
        cursor.close()
        conn.close()

    return rows_inserted


def get_historical_prices(symbols: list, lookback_days: int = 252) -> list:
    """Get historical close prices for multiple symbols.

    Returns data for the last N calendar days ordered chronologically.
    Used by WF2 for returns calculation and screening.

    Args:
        symbols: List of stock ticker symbols.
        lookback_days: Number of calendar days to look back.

    Returns:
        List of tuples (symbol, date, close) ordered by date ASC.
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        placeholders = ",".join(["%s"] * len(symbols))
        cursor.execute(
            f"""SELECT symbol, date, close
                FROM market_data
                WHERE symbol IN ({placeholders})
                  AND close IS NOT NULL
                  AND date >= (
                      SELECT MAX(date) - INTERVAL '{int(lookback_days)} days'
                      FROM market_data
                  )
                ORDER BY date ASC, symbol ASC""",
            symbols,
        )
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()


def store_screening_results(run_date: str, stock_metrics: list, run_metadata: dict) -> int:
    """Store screening results to PostgreSQL.

    Writes to both screening_runs (one row) and screening_results (one per symbol).
    Uses UPSERT for idempotency on Flyte retries.

    Args:
        run_date: Screening run date (YYYY-MM-DD).
        stock_metrics: List of StockMetrics dataclass instances.
        run_metadata: Dict with benchmark_cagr, benchmark_sharpe, etc.

    Returns:
        Number of stock result rows inserted/updated.
    """
    import json

    conn = get_connection()
    cursor = conn.cursor()
    rows_inserted = 0

    try:
        # Upsert screening run metadata
        cursor.execute(
            """INSERT INTO screening_runs
                   (run_date, num_symbols, optimal_k, benchmark_cagr,
                    benchmark_sharpe, benchmark_cumulative_return)
               VALUES (%s, %s, %s, %s, %s, %s)
               ON CONFLICT (run_date) DO UPDATE SET
                   num_symbols = EXCLUDED.num_symbols,
                   optimal_k = EXCLUDED.optimal_k,
                   benchmark_cagr = EXCLUDED.benchmark_cagr,
                   benchmark_sharpe = EXCLUDED.benchmark_sharpe,
                   benchmark_cumulative_return = EXCLUDED.benchmark_cumulative_return,
                   created_at = NOW()""",
            (
                run_date,
                run_metadata.get("num_symbols", 0),
                run_metadata.get("optimal_k", 0),
                run_metadata.get("benchmark_cagr", 0.0),
                run_metadata.get("benchmark_sharpe", 0.0),
                run_metadata.get("benchmark_cumulative_return", 0.0),
            ),
        )

        # Upsert per-symbol screening results
        for m in stock_metrics:
            cursor.execute(
                """INSERT INTO screening_results
                       (run_date, symbol, forward_return, rsi, rsi_signal,
                        volatility_252d, cagr, sharpe, sortino, calmar,
                        max_drawdown, composite_score, quintile, cluster_id,
                        cluster_label, momentum_returns_json, z_scores_json)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (run_date, symbol) DO UPDATE SET
                       forward_return = EXCLUDED.forward_return,
                       rsi = EXCLUDED.rsi,
                       rsi_signal = EXCLUDED.rsi_signal,
                       volatility_252d = EXCLUDED.volatility_252d,
                       cagr = EXCLUDED.cagr,
                       sharpe = EXCLUDED.sharpe,
                       sortino = EXCLUDED.sortino,
                       calmar = EXCLUDED.calmar,
                       max_drawdown = EXCLUDED.max_drawdown,
                       composite_score = EXCLUDED.composite_score,
                       quintile = EXCLUDED.quintile,
                       cluster_id = EXCLUDED.cluster_id,
                       cluster_label = EXCLUDED.cluster_label,
                       momentum_returns_json = EXCLUDED.momentum_returns_json,
                       z_scores_json = EXCLUDED.z_scores_json,
                       created_at = NOW()""",
                (
                    run_date, m.symbol, m.forward_return, m.rsi, m.rsi_signal,
                    m.volatility_252d, m.cagr, m.sharpe, m.sortino, m.calmar,
                    m.max_drawdown, m.composite_score, m.quintile, m.cluster_id,
                    m.cluster_label, json.dumps(m.momentum_returns),
                    json.dumps(m.z_scores),
                ),
            )
            rows_inserted += 1

        conn.commit()
    finally:
        cursor.close()
        conn.close()

    return rows_inserted


def get_screening_top_quintiles(run_date: str, max_quintile: int = 2) -> list:
    """Get top-ranked stocks from WF2 screening results.

    Retrieves symbols with quintile <= max_quintile (default: top 40%).
    Used by WF3 to determine which stocks to analyze.

    Args:
        run_date: Screening run date (YYYY-MM-DD). Empty string = latest.
        max_quintile: Maximum quintile to include (1-5, default: 2).

    Returns:
        List of tuples (symbol, composite_score, quintile).
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        if not run_date:
            # Get the latest screening run date
            cursor.execute(
                "SELECT MAX(run_date) FROM screening_runs"
            )
            result = cursor.fetchone()
            if not result or not result[0]:
                return []
            run_date = str(result[0])

        cursor.execute(
            """SELECT symbol, composite_score, quintile
               FROM screening_results
               WHERE run_date = %s
                 AND quintile <= %s
               ORDER BY composite_score DESC""",
            (run_date, max_quintile),
        )
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()


def get_price_history_for_technicals(symbols: list, days: int = 252) -> dict:
    """Get historical close prices for technical indicator computation.

    Returns a dict of symbol → list of (date, close) pairs, ordered
    chronologically (oldest first). Used by WF3 for SMA, MACD,
    Bollinger Bands calculations.

    Args:
        symbols: List of stock ticker symbols.
        days: Number of calendar days to look back (default: 252).

    Returns:
        Dict mapping symbol → list of (date_str, close_float) tuples.
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        placeholders = ",".join(["%s"] * len(symbols))
        cursor.execute(
            f"""SELECT symbol, date, close
                FROM market_data
                WHERE symbol IN ({placeholders})
                  AND close IS NOT NULL
                  AND date >= (
                      SELECT MAX(date) - INTERVAL '{int(days)} days'
                      FROM market_data
                  )
                ORDER BY symbol ASC, date ASC""",
            symbols,
        )
        rows = cursor.fetchall()

        # Group by symbol
        result = {}
        for symbol, date, close in rows:
            if symbol not in result:
                result[symbol] = []
            result[symbol].append((str(date), float(close)))
        return result
    finally:
        cursor.close()
        conn.close()


def store_signal_results(run_date: str, signal_results: list, run_metadata: dict) -> int:
    """Store WF3 signal analysis results to PostgreSQL.

    Writes to both signal_runs (one row) and signal_results (one per symbol).
    Uses UPSERT for idempotency on Flyte retries.

    Args:
        run_date: Signal analysis run date (YYYY-MM-DD).
        signal_results: List of SignalResult dataclass instances.
        run_metadata: Dict with num_symbols_analyzed, num_with_complete_data, etc.

    Returns:
        Number of signal result rows inserted/updated.
    """
    conn = get_connection()
    cursor = conn.cursor()
    rows_inserted = 0

    try:
        # Upsert signal run metadata
        cursor.execute(
            """INSERT INTO signal_runs
                   (run_date, num_symbols_analyzed, num_with_complete_data,
                    num_with_partial_data, tech_weight, fund_weight)
               VALUES (%s, %s, %s, %s, %s, %s)
               ON CONFLICT (run_date) DO UPDATE SET
                   num_symbols_analyzed = EXCLUDED.num_symbols_analyzed,
                   num_with_complete_data = EXCLUDED.num_with_complete_data,
                   num_with_partial_data = EXCLUDED.num_with_partial_data,
                   tech_weight = EXCLUDED.tech_weight,
                   fund_weight = EXCLUDED.fund_weight,
                   created_at = NOW()""",
            (
                run_date,
                run_metadata.get("num_symbols_analyzed", 0),
                run_metadata.get("num_with_complete_data", 0),
                run_metadata.get("num_with_partial_data", 0),
                run_metadata.get("tech_weight", 0.5),
                run_metadata.get("fund_weight", 0.5),
            ),
        )

        # Upsert per-symbol signal results
        for sr in signal_results:
            cursor.execute(
                """INSERT INTO signal_results
                       (run_date, symbol, wf2_composite_score, wf2_quintile,
                        technical_score, technical_signal,
                        sma_crossover_signal, macd_signal, bb_signal,
                        fundamental_score, fundamental_signal,
                        pe_ratio, pe_zscore, dividend_yield,
                        return_on_equity, debt_to_equity,
                        combined_signal_score, signal_strength, data_quality)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (run_date, symbol) DO UPDATE SET
                       wf2_composite_score = EXCLUDED.wf2_composite_score,
                       wf2_quintile = EXCLUDED.wf2_quintile,
                       technical_score = EXCLUDED.technical_score,
                       technical_signal = EXCLUDED.technical_signal,
                       sma_crossover_signal = EXCLUDED.sma_crossover_signal,
                       macd_signal = EXCLUDED.macd_signal,
                       bb_signal = EXCLUDED.bb_signal,
                       fundamental_score = EXCLUDED.fundamental_score,
                       fundamental_signal = EXCLUDED.fundamental_signal,
                       pe_ratio = EXCLUDED.pe_ratio,
                       pe_zscore = EXCLUDED.pe_zscore,
                       dividend_yield = EXCLUDED.dividend_yield,
                       return_on_equity = EXCLUDED.return_on_equity,
                       debt_to_equity = EXCLUDED.debt_to_equity,
                       combined_signal_score = EXCLUDED.combined_signal_score,
                       signal_strength = EXCLUDED.signal_strength,
                       data_quality = EXCLUDED.data_quality,
                       created_at = NOW()""",
                (
                    run_date, sr.symbol, sr.wf2_composite_score, sr.wf2_quintile,
                    sr.technical_score, sr.technical_signal,
                    getattr(sr, "_sma_signal", "neutral"),
                    getattr(sr, "_macd_signal", "neutral"),
                    getattr(sr, "_bb_signal", "neutral"),
                    sr.fundamental_score, sr.fundamental_signal,
                    getattr(sr, "_pe_ratio", -1.0),
                    getattr(sr, "_pe_zscore", 0.0),
                    getattr(sr, "_dividend_yield", 0.0),
                    getattr(sr, "_return_on_equity", -1.0),
                    getattr(sr, "_debt_to_equity", -1.0),
                    sr.combined_signal_score, sr.signal_strength, sr.data_quality,
                ),
            )
            rows_inserted += 1

        conn.commit()
    finally:
        cursor.close()
        conn.close()

    return rows_inserted


def get_latest_signal_results(run_date: str = "") -> list:
    """Get the latest WF3 signal results for WF4 portfolio construction.

    Args:
        run_date: Target signal run date (YYYY-MM-DD). Empty = latest available.

    Returns:
        List of tuples (symbol, combined_signal_score, signal_strength,
        wf2_quintile, technical_score, fundamental_score, data_quality).
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        if not run_date:
            cursor.execute("SELECT MAX(run_date) FROM signal_runs")
            result = cursor.fetchone()
            if not result or not result[0]:
                return []
            run_date = str(result[0])

        cursor.execute(
            """SELECT symbol, combined_signal_score, signal_strength,
                      wf2_quintile, technical_score, fundamental_score,
                      data_quality
               FROM signal_results
               WHERE run_date = %s
               ORDER BY combined_signal_score DESC""",
            (run_date,),
        )
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()


def get_current_positions() -> list:
    """Get all current portfolio positions.

    Returns:
        List of tuples (symbol, shares, avg_cost, current_price, sector).
        Empty list if no positions (initial state).
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """SELECT symbol, shares, avg_cost, current_price, sector
               FROM positions
               ORDER BY symbol ASC"""
        )
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()


def store_rebalancing_results(
    run_date: str,
    run_metadata: dict,
    trade_orders: list,
) -> int:
    """Store WF4 rebalancing run metadata and proposed trade orders.

    Writes to rebalancing_runs (one row) and trades (one per order).
    Uses UPSERT for idempotency on Flyte retries.

    Args:
        run_date: Rebalancing run date (YYYY-MM-DD).
        run_metadata: Dict with total_portfolio_value, cash_value, etc.
        trade_orders: List of TradeOrder dataclass instances.

    Returns:
        Number of trade order rows inserted.
    """
    conn = get_connection()
    cursor = conn.cursor()
    rows_inserted = 0

    try:
        # Upsert rebalancing run metadata
        cursor.execute(
            """INSERT INTO rebalancing_runs
                   (run_date, total_portfolio_value, cash_value, invested_value,
                    num_signals_input, num_target_positions, num_buy_orders,
                    num_sell_orders, total_estimated_cost, report_s3_path)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (run_date) DO UPDATE SET
                   total_portfolio_value = EXCLUDED.total_portfolio_value,
                   cash_value = EXCLUDED.cash_value,
                   invested_value = EXCLUDED.invested_value,
                   num_signals_input = EXCLUDED.num_signals_input,
                   num_target_positions = EXCLUDED.num_target_positions,
                   num_buy_orders = EXCLUDED.num_buy_orders,
                   num_sell_orders = EXCLUDED.num_sell_orders,
                   total_estimated_cost = EXCLUDED.total_estimated_cost,
                   report_s3_path = EXCLUDED.report_s3_path,
                   created_at = NOW()""",
            (
                run_date,
                run_metadata.get("total_portfolio_value", 0.0),
                run_metadata.get("cash_value", 0.0),
                run_metadata.get("invested_value", 0.0),
                run_metadata.get("num_signals_input", 0),
                run_metadata.get("num_target_positions", 0),
                run_metadata.get("num_buy_orders", 0),
                run_metadata.get("num_sell_orders", 0),
                run_metadata.get("total_estimated_cost", 0.0),
                run_metadata.get("report_s3_path", ""),
            ),
        )

        # Insert proposed trade orders
        for order in trade_orders:
            cursor.execute(
                """INSERT INTO trades
                       (date, symbol, side, quantity, price, commission,
                        spread_cost, impact_cost, reason)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    run_date,
                    order.symbol,
                    order.side,
                    order.quantity,
                    order.estimated_price,
                    0.0,  # Detailed cost breakdown not stored separately yet
                    0.0,
                    0.0,
                    order.reason,
                ),
            )
            rows_inserted += 1

        conn.commit()
    finally:
        cursor.close()
        conn.close()

    return rows_inserted


# ============================================================
# Phase 4: Paper Trading DB Functions
# ============================================================


def upsert_positions(positions_data: list) -> int:
    """Upsert portfolio positions after paper trade execution.

    Inserts new positions, updates existing ones, deletes fully exited
    positions (shares=0). Uses ON CONFLICT on symbol UNIQUE constraint.

    Args:
        positions_data: List of dicts with keys:
            symbol, shares, avg_cost, current_price, sector

    Returns:
        Number of rows affected.
    """
    conn = get_connection()
    cursor = conn.cursor()
    rows_affected = 0

    try:
        for pos in positions_data:
            shares = float(pos["shares"])
            if shares <= 0:
                # Full exit: remove position
                cursor.execute(
                    "DELETE FROM positions WHERE symbol = %s",
                    (pos["symbol"],),
                )
            else:
                cursor.execute(
                    """INSERT INTO positions
                           (symbol, shares, avg_cost, current_price, sector,
                            updated_at)
                       VALUES (%s, %s, %s, %s, %s, NOW())
                       ON CONFLICT (symbol) DO UPDATE SET
                           shares = EXCLUDED.shares,
                           avg_cost = EXCLUDED.avg_cost,
                           current_price = EXCLUDED.current_price,
                           sector = EXCLUDED.sector,
                           updated_at = NOW()""",
                    (
                        pos["symbol"],
                        shares,
                        float(pos["avg_cost"]),
                        float(pos.get("current_price", pos["avg_cost"])),
                        pos.get("sector", ""),
                    ),
                )
            rows_affected += 1

        conn.commit()
    finally:
        cursor.close()
        conn.close()

    return rows_affected


def store_executed_trades(run_date: str, executed_trades: list) -> int:
    """Store paper-executed trades with actual cost breakdown.

    Unlike store_rebalancing_results() which stores PROPOSED trades with
    zero costs, this stores executed paper trades with real commission,
    spread, and impact costs in USD.

    Args:
        run_date: Trade execution date (YYYY-MM-DD).
        executed_trades: List of dicts with keys:
            symbol, side, quantity, price, commission, spread_cost,
            impact_cost, reason

    Returns:
        Number of trade rows inserted.
    """
    conn = get_connection()
    cursor = conn.cursor()
    rows_inserted = 0

    try:
        for trade in executed_trades:
            cursor.execute(
                """INSERT INTO trades
                       (date, symbol, side, quantity, price, commission,
                        spread_cost, impact_cost, reason)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    run_date,
                    trade["symbol"],
                    trade["side"],
                    trade["quantity"],
                    trade["price"],
                    trade["commission"],
                    trade["spread_cost"],
                    trade["impact_cost"],
                    trade["reason"],
                ),
            )
            rows_inserted += 1

        conn.commit()
    finally:
        cursor.close()
        conn.close()

    return rows_inserted


def store_portfolio_snapshot(
    snapshot_date: str,
    total_value: float,
    cash: float,
    invested: float,
    daily_pnl: float,
    cumulative_dividends: float,
    num_positions: int,
) -> int:
    """Store a portfolio snapshot for performance tracking.

    Uses UPSERT on date for idempotency on Flyte retries.

    Args:
        snapshot_date: Date string (YYYY-MM-DD).
        total_value: Total portfolio value (cash + invested).
        cash: Remaining cash balance.
        invested: Value of all positions at current prices.
        daily_pnl: Change in total_value vs. previous snapshot.
        cumulative_dividends: Total dividends received to date.
        num_positions: Number of active positions.

    Returns:
        1 on success.
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """INSERT INTO portfolio_snapshots
                   (date, total_value, cash, invested, daily_pnl,
                    cumulative_dividends, num_positions)
               VALUES (%s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (date) DO UPDATE SET
                   total_value = EXCLUDED.total_value,
                   cash = EXCLUDED.cash,
                   invested = EXCLUDED.invested,
                   daily_pnl = EXCLUDED.daily_pnl,
                   cumulative_dividends = EXCLUDED.cumulative_dividends,
                   num_positions = EXCLUDED.num_positions""",
            (snapshot_date, total_value, cash, invested,
             daily_pnl, cumulative_dividends, num_positions),
        )
        conn.commit()
        return 1
    finally:
        cursor.close()
        conn.close()


def get_latest_portfolio_snapshot():
    """Get the most recent portfolio snapshot.

    Used by load_current_portfolio to read accurate cash balance
    after paper trading runs.

    Returns:
        Tuple (date, total_value, cash, invested, daily_pnl,
        cumulative_dividends, num_positions) or None if no snapshots.
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """SELECT date, total_value, cash, invested, daily_pnl,
                      cumulative_dividends, num_positions
               FROM portfolio_snapshots
               ORDER BY date DESC
               LIMIT 1"""
        )
        return cursor.fetchone()
    finally:
        cursor.close()
        conn.close()


def get_latest_market_data(symbol: str, days: int = 30) -> list:
    """Get the latest N days of market data for a symbol.

    Args:
        symbol: Stock ticker symbol.
        days: Number of days to retrieve.

    Returns:
        List of tuples (symbol, date, close, volume, spread_bps).
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """SELECT symbol, date, close, volume, spread_bps
               FROM market_data
               WHERE symbol = %s
               ORDER BY date DESC
               LIMIT %s""",
            (symbol, days),
        )
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()
