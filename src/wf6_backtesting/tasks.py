"""WF6: Backtesting — Tasks.

Reads historical signal_results from PostgreSQL, replays the pension fund
portfolio construction logic week by week, computes a buy-and-hold benchmark,
and compares the two strategies via CAGR, Sharpe, Sortino, max drawdown, Calmar.

All inter-task data uses Dict[str, str] with JSON serialization for Flytekit safety.
"""

import json
from typing import Dict

from flytekit import task, Resources


# ============================================================
# Task 1: Resolve Backtest Parameters
# ============================================================

@task(
    requests=Resources(cpu="100m", mem="128Mi"),
    limits=Resources(cpu="200m", mem="256Mi"),
)
def resolve_backtest_params(
    start_date: str,
    end_date: str,
    initial_capital: float,
    max_position_pct: float,
    max_sector_pct: float,
    cash_reserve_pct: float,
) -> Dict[str, str]:
    """Resolve and validate backtest parameters.

    Args:
        start_date: Backtest start date (YYYY-MM-DD).
        end_date: Backtest end date (YYYY-MM-DD). Empty = today.
        initial_capital: Starting capital in EUR.
        max_position_pct: Maximum weight per stock.
        max_sector_pct: Maximum weight per sector.
        cash_reserve_pct: Target cash reserve fraction.

    Returns:
        Dict with validated backtest parameters.
    """
    from datetime import date

    if not end_date:
        end_date = date.today().isoformat()

    return {
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": str(initial_capital),
        "max_position_pct": str(max_position_pct),
        "max_sector_pct": str(max_sector_pct),
        "cash_reserve_pct": str(cash_reserve_pct),
    }


# ============================================================
# Task 2: Load Historical Signals
# ============================================================

@task(
    requests=Resources(cpu="200m", mem="256Mi"),
    limits=Resources(cpu="500m", mem="512Mi"),
)
def load_historical_signals(
    params: Dict[str, str],
) -> Dict[str, str]:
    """Load all signal_results from DB for the backtest period.

    Reads each signal_runs date within [start_date, end_date] and loads
    the corresponding signal_results rows.

    Args:
        params: Backtest parameters from resolve_backtest_params.

    Returns:
        Dict with keys:
        - "run_dates_json": JSON list of signal run dates
        - "signals_by_date_json": JSON dict {date -> list of signal dicts}
        - "num_run_dates": count of signal run dates found
    """
    from src.shared.db import get_connection

    start_date = params["start_date"]
    end_date = params["end_date"]

    conn = get_connection()
    cursor = conn.cursor()

    try:
        # Get all signal run dates in range
        cursor.execute(
            """SELECT run_date FROM signal_runs
               WHERE run_date >= %s AND run_date <= %s
               ORDER BY run_date ASC""",
            (start_date, end_date),
        )
        run_dates = [str(row[0]) for row in cursor.fetchall()]

        # Load signals for each date
        signals_by_date = {}
        for rd in run_dates:
            cursor.execute(
                """SELECT symbol, combined_signal_score, signal_strength,
                          wf2_quintile, technical_score, fundamental_score,
                          data_quality, sentiment_score, sentiment_signal
                   FROM signal_results
                   WHERE run_date = %s
                   ORDER BY combined_signal_score DESC""",
                (rd,),
            )
            rows = cursor.fetchall()
            signals_by_date[rd] = [
                {
                    "symbol": row[0],
                    "combined_signal_score": float(row[1]) if row[1] else 50.0,
                    "signal_strength": row[2] or "hold",
                    "wf2_quintile": int(row[3]) if row[3] else 3,
                    "technical_score": float(row[4]) if row[4] else 50.0,
                    "fundamental_score": float(row[5]) if row[5] else 50.0,
                    "data_quality": row[6] or "minimal",
                    "sentiment_score": float(row[7]) if row[7] else 50.0,
                    "sentiment_signal": row[8] or "neutral",
                }
                for row in rows
            ]

        return {
            "run_dates_json": json.dumps(run_dates),
            "signals_by_date_json": json.dumps(signals_by_date),
            "num_run_dates": str(len(run_dates)),
        }
    finally:
        cursor.close()
        conn.close()


# ============================================================
# Task 3: Simulate Signal-Based Portfolio
# ============================================================

@task(
    requests=Resources(cpu="500m", mem="512Mi"),
    limits=Resources(cpu="1000m", mem="1024Mi"),
)
def simulate_signal_portfolio(
    params: Dict[str, str],
    signals: Dict[str, str],
) -> Dict[str, str]:
    """Replay the pension fund portfolio construction on historical signals.

    For each signal run date:
    1. Calculate target weights (same logic as WF4)
    2. Apply weights to get positions
    3. Price positions at next rebalance date using market_data
    4. Track portfolio value over time

    Args:
        params: Backtest parameters.
        signals: Historical signal data from load_historical_signals.

    Returns:
        Dict with keys:
        - "dates_json": JSON list of valuation dates
        - "values_json": JSON list of portfolio values
        - "daily_returns_json": JSON list of daily returns
        - "final_value": str
        - "num_rebalances": str
    """
    from src.shared.analytics import calculate_signal_weights
    from src.shared.config import SYMBOL_SECTORS
    from src.shared.db import get_connection

    initial_capital = float(params["initial_capital"])
    max_pos = float(params["max_position_pct"])
    max_sec = float(params["max_sector_pct"])
    cash_res = float(params["cash_reserve_pct"])

    run_dates = json.loads(signals["run_dates_json"])
    signals_by_date = json.loads(signals["signals_by_date_json"])

    if not run_dates:
        return {
            "dates_json": "[]",
            "values_json": "[]",
            "daily_returns_json": "[]",
            "final_value": str(initial_capital),
            "num_rebalances": "0",
        }

    conn = get_connection()
    cursor = conn.cursor()

    try:
        # Build a mapping of all trading days and prices within the backtest window
        start = run_dates[0]
        end = params["end_date"]

        # Get all unique symbols across all signal dates
        all_symbols = set()
        for rd, sigs in signals_by_date.items():
            for s in sigs:
                all_symbols.add(s["symbol"])

        if not all_symbols:
            return {
                "dates_json": "[]",
                "values_json": "[]",
                "daily_returns_json": "[]",
                "final_value": str(initial_capital),
                "num_rebalances": "0",
            }

        # Load all market data for these symbols in range
        placeholders = ",".join(["%s"] * len(all_symbols))
        cursor.execute(
            f"""SELECT date, symbol, adj_close
                FROM market_data
                WHERE symbol IN ({placeholders})
                  AND date >= %s AND date <= %s
                ORDER BY date ASC""",
            (*sorted(all_symbols), start, end),
        )
        rows = cursor.fetchall()

        # Build price lookup: {date_str: {symbol: price}}
        price_data = {}
        trading_dates = []
        seen_dates = set()
        for row in rows:
            d = str(row[0])
            sym = row[1]
            price = float(row[2]) if row[2] else None
            if d not in seen_dates:
                seen_dates.add(d)
                trading_dates.append(d)
            price_data.setdefault(d, {})[sym] = price

        if not trading_dates:
            return {
                "dates_json": "[]",
                "values_json": "[]",
                "daily_returns_json": "[]",
                "final_value": str(initial_capital),
                "num_rebalances": "0",
            }

        # Simulate portfolio
        cash = initial_capital
        positions = {}  # {symbol: {"shares": float, "price": float}}
        portfolio_values = []
        portfolio_dates = []
        num_rebalances = 0

        # Convert run_dates to a set for fast lookup
        run_date_set = set(run_dates)

        for td in trading_dates:
            prices_today = price_data.get(td, {})

            # Check if this is a rebalance date (signal run date)
            if td in run_date_set and td in signals_by_date:
                sigs = signals_by_date[td]

                # Calculate target weights
                weights = calculate_signal_weights(
                    signals=sigs,
                    max_position_pct=max_pos,
                    max_sector_pct=max_sec,
                    cash_reserve_pct=cash_res,
                    sector_map=SYMBOL_SECTORS,
                )

                # Calculate current portfolio value
                port_value = cash
                for sym, pos in positions.items():
                    p = prices_today.get(sym, pos["price"])
                    port_value += pos["shares"] * p

                # Liquidate all positions (sell everything)
                for sym, pos in positions.items():
                    p = prices_today.get(sym, pos["price"])
                    cash += pos["shares"] * p
                positions = {}

                # Buy new positions according to target weights
                for sym, weight in weights.items():
                    if sym in prices_today and prices_today[sym] and prices_today[sym] > 0:
                        alloc = port_value * weight
                        shares = alloc / prices_today[sym]
                        if shares > 0:
                            positions[sym] = {
                                "shares": shares,
                                "price": prices_today[sym],
                            }
                            cash -= alloc

                num_rebalances += 1

            # Mark-to-market
            total = cash
            for sym, pos in positions.items():
                p = prices_today.get(sym, pos["price"])
                if p:
                    pos["price"] = p  # Update last known price
                total += pos["shares"] * pos["price"]

            portfolio_dates.append(td)
            portfolio_values.append(round(total, 2))

        # Calculate daily returns
        daily_returns = [0.0]
        for i in range(1, len(portfolio_values)):
            if portfolio_values[i - 1] > 0:
                daily_returns.append(
                    (portfolio_values[i] - portfolio_values[i - 1]) / portfolio_values[i - 1]
                )
            else:
                daily_returns.append(0.0)

        return {
            "dates_json": json.dumps(portfolio_dates),
            "values_json": json.dumps(portfolio_values),
            "daily_returns_json": json.dumps(daily_returns),
            "final_value": str(portfolio_values[-1] if portfolio_values else initial_capital),
            "num_rebalances": str(num_rebalances),
        }
    finally:
        cursor.close()
        conn.close()


# ============================================================
# Task 4: Compute Benchmark Returns
# ============================================================

@task(
    requests=Resources(cpu="500m", mem="512Mi"),
    limits=Resources(cpu="1000m", mem="1024Mi"),
)
def compute_benchmark_returns(
    params: Dict[str, str],
    signals: Dict[str, str],
) -> Dict[str, str]:
    """Compute equal-weight buy-and-hold benchmark.

    On the first signal run date, invest equally in all signaled stocks.
    Hold without rebalancing until end of backtest.

    Args:
        params: Backtest parameters.
        signals: Historical signal data from load_historical_signals.

    Returns:
        Dict with keys:
        - "dates_json": JSON list of valuation dates
        - "values_json": JSON list of benchmark portfolio values
        - "daily_returns_json": JSON list of daily returns
        - "final_value": str
    """
    from src.shared.db import get_connection

    initial_capital = float(params["initial_capital"])
    run_dates = json.loads(signals["run_dates_json"])
    signals_by_date = json.loads(signals["signals_by_date_json"])

    if not run_dates:
        return {
            "dates_json": "[]",
            "values_json": "[]",
            "daily_returns_json": "[]",
            "final_value": str(initial_capital),
        }

    # Get all symbols from the first signal run date
    first_date = run_dates[0]
    first_signals = signals_by_date.get(first_date, [])
    symbols = [s["symbol"] for s in first_signals
               if s.get("signal_strength") in ("strong_buy", "buy")]

    if not symbols:
        # Fall back to all symbols
        symbols = [s["symbol"] for s in first_signals]

    if not symbols:
        return {
            "dates_json": "[]",
            "values_json": "[]",
            "daily_returns_json": "[]",
            "final_value": str(initial_capital),
        }

    conn = get_connection()
    cursor = conn.cursor()

    try:
        start = first_date
        end = params["end_date"]

        placeholders = ",".join(["%s"] * len(symbols))
        cursor.execute(
            f"""SELECT date, symbol, adj_close
                FROM market_data
                WHERE symbol IN ({placeholders})
                  AND date >= %s AND date <= %s
                ORDER BY date ASC""",
            (*sorted(symbols), start, end),
        )
        rows = cursor.fetchall()

        # Build price lookup
        price_data = {}
        trading_dates = []
        seen_dates = set()
        for row in rows:
            d = str(row[0])
            sym = row[1]
            price = float(row[2]) if row[2] else None
            if d not in seen_dates:
                seen_dates.add(d)
                trading_dates.append(d)
            price_data.setdefault(d, {})[sym] = price

        if not trading_dates:
            return {
                "dates_json": "[]",
                "values_json": "[]",
                "daily_returns_json": "[]",
                "final_value": str(initial_capital),
            }

        # Buy equal-weight on first trading day
        first_prices = price_data.get(trading_dates[0], {})
        per_stock = initial_capital / len(symbols)
        positions = {}
        cash = initial_capital

        for sym in symbols:
            if sym in first_prices and first_prices[sym] and first_prices[sym] > 0:
                shares = per_stock / first_prices[sym]
                positions[sym] = {"shares": shares, "price": first_prices[sym]}
                cash -= per_stock

        # Mark-to-market each day
        benchmark_dates = []
        benchmark_values = []

        for td in trading_dates:
            prices_today = price_data.get(td, {})
            total = cash
            for sym, pos in positions.items():
                p = prices_today.get(sym, pos["price"])
                if p:
                    pos["price"] = p
                total += pos["shares"] * pos["price"]

            benchmark_dates.append(td)
            benchmark_values.append(round(total, 2))

        # Daily returns
        daily_returns = [0.0]
        for i in range(1, len(benchmark_values)):
            if benchmark_values[i - 1] > 0:
                daily_returns.append(
                    (benchmark_values[i] - benchmark_values[i - 1]) / benchmark_values[i - 1]
                )
            else:
                daily_returns.append(0.0)

        return {
            "dates_json": json.dumps(benchmark_dates),
            "values_json": json.dumps(benchmark_values),
            "daily_returns_json": json.dumps(daily_returns),
            "final_value": str(benchmark_values[-1] if benchmark_values else initial_capital),
        }
    finally:
        cursor.close()
        conn.close()


# ============================================================
# Task 5: Compare Strategies
# ============================================================

@task(
    requests=Resources(cpu="200m", mem="256Mi"),
    limits=Resources(cpu="500m", mem="512Mi"),
)
def compare_strategies(
    params: Dict[str, str],
    portfolio: Dict[str, str],
    benchmark: Dict[str, str],
) -> Dict[str, str]:
    """Compare signal-based portfolio vs buy-and-hold benchmark.

    Computes CAGR, Sharpe, Sortino, max drawdown, Calmar for both strategies.

    Args:
        params: Backtest parameters.
        portfolio: Signal-based portfolio results.
        benchmark: Buy-and-hold benchmark results.

    Returns:
        Dict with comparison metrics as JSON.
    """
    import pandas as pd
    from src.shared.analytics import (
        compute_cagr,
        compute_sharpe,
        compute_sortino,
        compute_max_drawdown,
        compute_calmar,
    )

    initial_capital = float(params["initial_capital"])

    def compute_metrics(values_json: str, returns_json: str) -> dict:
        values = json.loads(values_json)
        returns = json.loads(returns_json)

        if not values or len(values) < 2:
            return {
                "cagr": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "max_drawdown": 0.0,
                "calmar": 0.0,
                "total_return_pct": 0.0,
                "final_value": initial_capital,
            }

        # Cumulative returns (normalized to 1.0)
        cum_returns = pd.Series([v / values[0] for v in values])
        daily_returns = pd.Series(returns)

        cagr = compute_cagr(cum_returns)
        sharpe = compute_sharpe(daily_returns)
        sortino = compute_sortino(daily_returns)
        max_dd = compute_max_drawdown(cum_returns)
        calmar = compute_calmar(cagr, max_dd)
        total_ret = (values[-1] - values[0]) / values[0] * 100

        return {
            "cagr": round(cagr, 6),
            "sharpe": round(sharpe, 4),
            "sortino": round(sortino, 4),
            "max_drawdown": round(max_dd, 6),
            "calmar": round(calmar, 4),
            "total_return_pct": round(total_ret, 2),
            "final_value": round(values[-1], 2),
        }

    portfolio_metrics = compute_metrics(
        portfolio.get("values_json", "[]"),
        portfolio.get("daily_returns_json", "[]"),
    )
    benchmark_metrics = compute_metrics(
        benchmark.get("values_json", "[]"),
        benchmark.get("daily_returns_json", "[]"),
    )

    # Excess metrics
    excess = {
        "cagr_excess": round(portfolio_metrics["cagr"] - benchmark_metrics["cagr"], 6),
        "sharpe_excess": round(portfolio_metrics["sharpe"] - benchmark_metrics["sharpe"], 4),
        "total_return_excess_pct": round(
            portfolio_metrics["total_return_pct"] - benchmark_metrics["total_return_pct"], 2
        ),
    }

    return {
        "portfolio_metrics_json": json.dumps(portfolio_metrics),
        "benchmark_metrics_json": json.dumps(benchmark_metrics),
        "excess_metrics_json": json.dumps(excess),
        "num_rebalances": portfolio.get("num_rebalances", "0"),
        "start_date": params["start_date"],
        "end_date": params["end_date"],
    }


# ============================================================
# Task 6: Generate Backtest Report
# ============================================================

@task(
    requests=Resources(cpu="200m", mem="256Mi"),
    limits=Resources(cpu="500m", mem="512Mi"),
)
def generate_backtest_report(
    params: Dict[str, str],
    comparison: Dict[str, str],
    portfolio: Dict[str, str],
    benchmark: Dict[str, str],
) -> str:
    """Generate a markdown backtest report and store to DB + MinIO.

    Args:
        params: Backtest parameters.
        comparison: Strategy comparison metrics.
        portfolio: Signal-based portfolio results.
        benchmark: Benchmark results.

    Returns:
        Formatted markdown report.
    """
    port_metrics = json.loads(comparison["portfolio_metrics_json"])
    bench_metrics = json.loads(comparison["benchmark_metrics_json"])
    excess = json.loads(comparison["excess_metrics_json"])

    start = comparison["start_date"]
    end = comparison["end_date"]
    initial_capital = float(params["initial_capital"])
    num_rebalances = comparison["num_rebalances"]

    lines = [
        f"{'=' * 65}",
        f"  WF6 Backtest Report",
        f"  {start} to {end}",
        f"{'=' * 65}",
        "",
        f"Initial Capital:   EUR {initial_capital:,.2f}",
        f"Rebalances:        {num_rebalances}",
        "",
        "## Strategy Comparison",
        "",
        f"  {'Metric':<22s} {'Signal':>12s} {'Buy&Hold':>12s} {'Excess':>12s}",
        f"  {'-'*22} {'-'*12} {'-'*12} {'-'*12}",
        f"  {'Total Return %':<22s} {port_metrics['total_return_pct']:>11.2f}% {bench_metrics['total_return_pct']:>11.2f}% {excess['total_return_excess_pct']:>+11.2f}%",
        f"  {'CAGR':<22s} {port_metrics['cagr']:>12.4f} {bench_metrics['cagr']:>12.4f} {excess['cagr_excess']:>+12.4f}",
        f"  {'Sharpe Ratio':<22s} {port_metrics['sharpe']:>12.4f} {bench_metrics['sharpe']:>12.4f} {excess['sharpe_excess']:>+12.4f}",
        f"  {'Sortino Ratio':<22s} {port_metrics['sortino']:>12.4f} {bench_metrics['sortino']:>12.4f}",
        f"  {'Max Drawdown':<22s} {port_metrics['max_drawdown']:>12.4f} {bench_metrics['max_drawdown']:>12.4f}",
        f"  {'Calmar Ratio':<22s} {port_metrics['calmar']:>12.4f} {bench_metrics['calmar']:>12.4f}",
        "",
        f"  {'Final Value':<22s} EUR {port_metrics['final_value']:>10,.2f} EUR {bench_metrics['final_value']:>10,.2f}",
        "",
    ]

    # Verdict
    if excess["cagr_excess"] > 0 and excess["sharpe_excess"] > 0:
        verdict = "Signal strategy OUTPERFORMS buy-and-hold on both return and risk-adjusted basis."
    elif excess["cagr_excess"] > 0:
        verdict = "Signal strategy has HIGHER returns but LOWER risk-adjusted performance."
    elif excess["sharpe_excess"] > 0:
        verdict = "Signal strategy has BETTER risk-adjusted returns but LOWER absolute returns."
    else:
        verdict = "Signal strategy UNDERPERFORMS buy-and-hold. Review signal weights and model."

    lines.append(f"## Verdict")
    lines.append(f"  {verdict}")
    lines.append("")
    lines.append(f"{'=' * 65}")

    report = "\n".join(lines)

    # Store to DB (non-fatal)
    try:
        _store_backtest_to_db(start, end, port_metrics, bench_metrics, excess, num_rebalances, report)
    except Exception:
        pass  # DB failure is non-fatal

    # Upload to MinIO (non-fatal)
    try:
        _upload_backtest_report(start, end, report)
    except Exception:
        pass  # MinIO failure is non-fatal

    return report


def _store_backtest_to_db(
    start_date: str,
    end_date: str,
    port_metrics: dict,
    bench_metrics: dict,
    excess: dict,
    num_rebalances: str,
    report: str,
) -> None:
    """Store backtest run to PostgreSQL (non-fatal)."""
    from src.shared.db import get_connection

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """INSERT INTO backtest_runs
                   (start_date, end_date, num_rebalances,
                    signal_cagr, signal_sharpe, signal_sortino,
                    signal_max_drawdown, signal_calmar, signal_final_value,
                    benchmark_cagr, benchmark_sharpe, benchmark_sortino,
                    benchmark_max_drawdown, benchmark_calmar, benchmark_final_value,
                    excess_cagr, excess_sharpe, report_text)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (start_date, end_date) DO UPDATE SET
                   num_rebalances = EXCLUDED.num_rebalances,
                   signal_cagr = EXCLUDED.signal_cagr,
                   signal_sharpe = EXCLUDED.signal_sharpe,
                   signal_sortino = EXCLUDED.signal_sortino,
                   signal_max_drawdown = EXCLUDED.signal_max_drawdown,
                   signal_calmar = EXCLUDED.signal_calmar,
                   signal_final_value = EXCLUDED.signal_final_value,
                   benchmark_cagr = EXCLUDED.benchmark_cagr,
                   benchmark_sharpe = EXCLUDED.benchmark_sharpe,
                   benchmark_sortino = EXCLUDED.benchmark_sortino,
                   benchmark_max_drawdown = EXCLUDED.benchmark_max_drawdown,
                   benchmark_calmar = EXCLUDED.benchmark_calmar,
                   benchmark_final_value = EXCLUDED.benchmark_final_value,
                   excess_cagr = EXCLUDED.excess_cagr,
                   excess_sharpe = EXCLUDED.excess_sharpe,
                   report_text = EXCLUDED.report_text,
                   created_at = NOW()""",
            (
                start_date, end_date, int(num_rebalances),
                port_metrics["cagr"], port_metrics["sharpe"], port_metrics["sortino"],
                port_metrics["max_drawdown"], port_metrics["calmar"], port_metrics["final_value"],
                bench_metrics["cagr"], bench_metrics["sharpe"], bench_metrics["sortino"],
                bench_metrics["max_drawdown"], bench_metrics["calmar"], bench_metrics["final_value"],
                excess["cagr_excess"], excess["sharpe_excess"], report,
            ),
        )
        conn.commit()
    finally:
        cursor.close()
        conn.close()


def _upload_backtest_report(start_date: str, end_date: str, report: str) -> None:
    """Upload backtest report to MinIO (non-fatal)."""
    from src.shared.config import S3_DATA_BUCKET
    from src.shared.storage import get_s3_client

    parts = end_date.split("-")
    s3_key = f"reports/wf6/year={parts[0]}/month={parts[1]}/day={parts[2]}/backtest_{start_date}_to_{end_date}.md"

    client = get_s3_client()
    client.put_object(
        Bucket=S3_DATA_BUCKET,
        Key=s3_key,
        Body=report.encode("utf-8"),
        ContentType="text/markdown",
    )
