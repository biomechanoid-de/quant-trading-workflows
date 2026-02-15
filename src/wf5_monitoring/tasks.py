"""WF5: Monitoring & Reporting - Tasks (Phase 5).

Weekly pipeline that calculates P&L, computes risk metrics,
checks alert conditions, and generates a monitoring report.

Schedule: Weekly Monday 10:00 UTC (after WF4 at 09:00)
Node: Any Pi 4 Worker

Pipeline:
    calculate_pnl -> compute_risk_metrics -> check_alerts -> generate_monitoring_report
"""

from typing import Dict

from flytekit import task, Resources


@task(
    requests=Resources(cpu="200m", mem="256Mi"),
    limits=Resources(cpu="500m", mem="512Mi"),
)
def calculate_pnl(run_date: str, lookback_days: int = 30) -> Dict[str, str]:
    """Calculate P&L metrics from portfolio snapshots and positions.

    Reads portfolio_snapshots, positions, and market_data from PostgreSQL.
    Computes daily, MTD, and YTD P&L plus per-position unrealized P&L.

    Args:
        run_date: Target date (YYYY-MM-DD). Empty = today.
        lookback_days: Days of snapshot history to retrieve.

    Returns:
        Dict[str, str] with P&L data for downstream tasks.
    """
    import json
    from datetime import date

    from src.shared.db import (
        get_portfolio_snapshots,
        get_positions_with_market_data,
    )

    # Resolve run date
    if not run_date:
        run_date = date.today().isoformat()

    # Load snapshots (up to 90 days for MTD/YTD calculation)
    snapshots = get_portfolio_snapshots(lookback_days=max(lookback_days, 90))

    if not snapshots:
        return {
            "run_date": run_date,
            "portfolio_value": "0.0",
            "cash": "0.0",
            "invested": "0.0",
            "daily_pnl": "0.0",
            "daily_pnl_pct": "0.0",
            "mtd_pnl": "0.0",
            "ytd_pnl": "0.0",
            "num_positions": "0",
            "top_winners": "",
            "top_losers": "",
            "positions_pnl_json": "{}",
            "snapshots_json": "[]",
            "no_data": "true",
        }

    # Latest snapshot values
    # Tuple: (date, total_value, cash, invested, daily_pnl, cum_div, num_pos)
    latest = snapshots[-1]
    portfolio_value = float(latest[1]) if latest[1] else 0.0
    cash = float(latest[2]) if latest[2] else 0.0
    invested = float(latest[3]) if latest[3] else 0.0
    daily_pnl = float(latest[4]) if latest[4] else 0.0
    num_positions = int(latest[6]) if latest[6] else 0

    # Daily P&L percentage
    if len(snapshots) >= 2:
        prev_value = float(snapshots[-2][1]) if snapshots[-2][1] else 0.0
        daily_pnl_pct = (daily_pnl / prev_value * 100) if prev_value > 0 else 0.0
    else:
        daily_pnl_pct = 0.0

    # MTD P&L: current value - first value of current month
    run_month = run_date[:7]  # YYYY-MM
    mtd_start_value = None
    for snap in snapshots:
        snap_date_str = str(snap[0])
        if snap_date_str[:7] == run_month:
            mtd_start_value = float(snap[1]) if snap[1] else 0.0
            break
    mtd_pnl = (portfolio_value - mtd_start_value) if mtd_start_value is not None else 0.0

    # YTD P&L: current value - first value of current year
    run_year = run_date[:4]
    ytd_start_value = None
    for snap in snapshots:
        snap_date_str = str(snap[0])
        if snap_date_str[:4] == run_year:
            ytd_start_value = float(snap[1]) if snap[1] else 0.0
            break
    ytd_pnl = (portfolio_value - ytd_start_value) if ytd_start_value is not None else 0.0

    # Position-level unrealized P&L
    positions = get_positions_with_market_data()
    positions_pnl = {}
    for pos in positions:
        # (symbol, shares, avg_cost, current_price, sector, latest_close)
        symbol = pos[0]
        shares = float(pos[1])
        avg_cost = float(pos[2])
        latest_close = float(pos[5]) if pos[5] else float(pos[3]) if pos[3] else 0.0
        sector = pos[4] or ""
        unrealized_pnl = (latest_close - avg_cost) * shares
        position_value = latest_close * shares
        positions_pnl[symbol] = {
            "shares": shares,
            "avg_cost": avg_cost,
            "current_price": latest_close,
            "unrealized_pnl": round(unrealized_pnl, 2),
            "position_value": round(position_value, 2),
            "sector": sector,
        }

    # Top 3 winners and losers by unrealized P&L
    sorted_by_pnl = sorted(
        positions_pnl.items(),
        key=lambda x: x[1]["unrealized_pnl"],
        reverse=True,
    )
    top_winners = "|".join(
        f"{sym} ({d['unrealized_pnl']:+.2f})"
        for sym, d in sorted_by_pnl[:3]
        if d["unrealized_pnl"] > 0
    )
    top_losers = "|".join(
        f"{sym} ({d['unrealized_pnl']:+.2f})"
        for sym, d in sorted_by_pnl[-3:]
        if d["unrealized_pnl"] < 0
    )

    # Serialize snapshots for risk metric computation
    snapshots_list = [
        {"date": str(s[0]), "total_value": float(s[1]) if s[1] else 0.0}
        for s in snapshots
    ]

    return {
        "run_date": run_date,
        "portfolio_value": str(round(portfolio_value, 2)),
        "cash": str(round(cash, 2)),
        "invested": str(round(invested, 2)),
        "daily_pnl": str(round(daily_pnl, 2)),
        "daily_pnl_pct": str(round(daily_pnl_pct, 4)),
        "mtd_pnl": str(round(mtd_pnl, 2)),
        "ytd_pnl": str(round(ytd_pnl, 2)),
        "num_positions": str(num_positions),
        "top_winners": top_winners,
        "top_losers": top_losers,
        "positions_pnl_json": json.dumps(positions_pnl),
        "snapshots_json": json.dumps(snapshots_list),
        "no_data": "false",
    }


@task(
    requests=Resources(cpu="300m", mem="256Mi"),
    limits=Resources(cpu="500m", mem="512Mi"),
)
def compute_risk_metrics(
    pnl_data: Dict[str, str],
    risk_free_rate: float = 0.05,
    lookback_days: int = 30,
) -> Dict[str, str]:
    """Compute risk metrics from portfolio snapshot time series.

    Computes Sharpe, Sortino, max drawdown (30d), historical VaR(95%),
    and sector concentration using existing analytics functions.

    Args:
        pnl_data: Output from calculate_pnl.
        risk_free_rate: Annualized risk-free rate for Sharpe/Sortino.
        lookback_days: Lookback window for risk metrics.

    Returns:
        Dict[str, str] with risk metric data.
    """
    import json

    import numpy as np
    import pandas as pd

    from src.shared.analytics import (
        compute_sharpe,
        compute_sortino,
        compute_max_drawdown,
    )

    # Short-circuit if no data
    if pnl_data.get("no_data") == "true":
        return {
            "sharpe_30d": "0.0",
            "sortino_30d": "0.0",
            "max_drawdown_30d": "0.0",
            "var_95": "0.0",
            "sector_concentration_json": "{}",
            "largest_sector": "",
            "largest_sector_pct": "0.0",
            "data_points": "0",
        }

    # Parse snapshots time series
    snapshots = json.loads(pnl_data.get("snapshots_json", "[]"))
    if len(snapshots) < 2:
        return {
            "sharpe_30d": "0.0",
            "sortino_30d": "0.0",
            "max_drawdown_30d": "0.0",
            "var_95": "0.0",
            "sector_concentration_json": "{}",
            "largest_sector": "",
            "largest_sector_pct": "0.0",
            "data_points": str(len(snapshots)),
        }

    # Build total_value series
    values = pd.Series(
        [s["total_value"] for s in snapshots],
        index=pd.to_datetime([s["date"] for s in snapshots]),
    )

    # Use the last N data points for risk metrics
    values_window = values.iloc[-lookback_days:] if len(values) > lookback_days else values

    # Daily returns
    daily_returns = values_window.pct_change().dropna()

    # Cumulative returns (starting from 1.0)
    cumulative = (1 + daily_returns).cumprod()

    # Risk metrics using existing analytics functions
    sharpe = compute_sharpe(daily_returns, risk_free_rate)
    sortino = compute_sortino(daily_returns, risk_free_rate)
    max_dd = compute_max_drawdown(cumulative)

    # Historical VaR(95%): 5th percentile of daily P&L in EUR
    daily_pnl_series = values_window.diff().dropna()
    if len(daily_pnl_series) >= 2:
        var_95 = float(np.percentile(daily_pnl_series, 5))
    else:
        var_95 = 0.0

    # Sector concentration from positions
    positions_pnl = json.loads(pnl_data.get("positions_pnl_json", "{}"))
    sector_totals = {}
    total_invested = 0.0
    for _sym, data in positions_pnl.items():
        sector = data.get("sector", "Unknown") or "Unknown"
        pos_value = data.get("position_value", 0.0)
        sector_totals[sector] = sector_totals.get(sector, 0.0) + pos_value
        total_invested += pos_value

    sector_pcts = {}
    if total_invested > 0:
        sector_pcts = {
            s: round(v / total_invested, 4)
            for s, v in sorted(sector_totals.items(), key=lambda x: -x[1])
        }

    largest_sector = ""
    largest_sector_pct = 0.0
    if sector_pcts:
        largest_sector = next(iter(sector_pcts))
        largest_sector_pct = sector_pcts[largest_sector]

    return {
        "sharpe_30d": str(round(sharpe, 6)),
        "sortino_30d": str(round(sortino, 6)),
        "max_drawdown_30d": str(round(max_dd, 6)),
        "var_95": str(round(var_95, 2)),
        "sector_concentration_json": json.dumps(sector_pcts),
        "largest_sector": largest_sector,
        "largest_sector_pct": str(round(largest_sector_pct, 4)),
        "data_points": str(len(daily_returns)),
    }


@task(
    requests=Resources(cpu="100m", mem="128Mi"),
    limits=Resources(cpu="200m", mem="256Mi"),
)
def check_alerts(
    pnl_data: Dict[str, str],
    risk_data: Dict[str, str],
    drawdown_threshold: float = 0.05,
    position_threshold: float = 0.07,
    var_threshold: float = 0.03,
    loss_threshold: float = 0.10,
) -> Dict[str, str]:
    """Check alert conditions against risk thresholds.

    Checks four conditions: drawdown breach, position concentration,
    VaR breach, and unrealized loss per position.

    Args:
        pnl_data: Output from calculate_pnl.
        risk_data: Output from compute_risk_metrics.
        drawdown_threshold: Max acceptable 30d drawdown (default: 5%).
        position_threshold: Max single position weight (default: 7%).
        var_threshold: Max VaR as % of portfolio (default: 3%).
        loss_threshold: Max unrealized loss per position (default: 10%).

    Returns:
        Dict[str, str] with alert information.
    """
    import json

    alerts = []

    # Short-circuit if no data
    if pnl_data.get("no_data") == "true":
        return {
            "num_alerts": "0",
            "alerts_csv": "",
            "has_critical": "false",
        }

    portfolio_value = float(pnl_data.get("portfolio_value", "0"))
    has_critical = False

    # Check 1: Drawdown alert
    max_dd = abs(float(risk_data.get("max_drawdown_30d", "0")))
    if max_dd > drawdown_threshold:
        alerts.append(
            f"DRAWDOWN: 30d max drawdown is {max_dd:.1%} "
            f"(threshold: {drawdown_threshold:.0%})"
        )
        has_critical = True

    # Check 2: Position concentration alert
    if portfolio_value > 0:
        positions_pnl = json.loads(pnl_data.get("positions_pnl_json", "{}"))
        for sym, data in positions_pnl.items():
            pos_value = data.get("position_value", 0.0)
            weight = pos_value / portfolio_value
            if weight > position_threshold:
                alerts.append(
                    f"CONCENTRATION: {sym} is {weight:.1%} of portfolio "
                    f"(threshold: {position_threshold:.0%})"
                )

    # Check 3: VaR breach alert
    var_95 = abs(float(risk_data.get("var_95", "0")))
    if portfolio_value > 0 and var_95 > portfolio_value * var_threshold:
        var_pct = var_95 / portfolio_value
        alerts.append(
            f"VAR: Daily VaR(95%) is EUR {var_95:,.0f} "
            f"({var_pct:.1%} of portfolio, threshold: {var_threshold:.0%})"
        )
        has_critical = True

    # Check 4: Unrealized loss per position
    positions_pnl = json.loads(pnl_data.get("positions_pnl_json", "{}"))
    for sym, data in positions_pnl.items():
        avg_cost = data.get("avg_cost", 0.0)
        shares = data.get("shares", 0.0)
        cost_basis = avg_cost * shares
        if cost_basis > 0:
            loss_pct = data.get("unrealized_pnl", 0.0) / cost_basis
            if loss_pct < -loss_threshold:
                alerts.append(
                    f"LOSS: {sym} has {loss_pct:.1%} unrealized loss "
                    f"(threshold: {-loss_threshold:.0%})"
                )

    return {
        "num_alerts": str(len(alerts)),
        "alerts_csv": "|".join(alerts),
        "has_critical": "true" if has_critical else "false",
    }


@task(
    requests=Resources(cpu="200m", mem="256Mi"),
    limits=Resources(cpu="500m", mem="512Mi"),
)
def generate_monitoring_report(
    pnl_data: Dict[str, str],
    risk_data: Dict[str, str],
    alert_data: Dict[str, str],
) -> str:
    """Generate markdown monitoring report and store to DB + MinIO.

    Assembles all monitoring data into a readable report, stores
    the run metadata in monitoring_runs table, and uploads the
    report to MinIO.

    Args:
        pnl_data: Output from calculate_pnl.
        risk_data: Output from compute_risk_metrics.
        alert_data: Output from check_alerts.

    Returns:
        Markdown report string.
    """
    import json

    run_date = pnl_data.get("run_date", "unknown")
    no_data = pnl_data.get("no_data") == "true"

    # Build markdown report
    lines = []
    lines.append(f"# WF5 Monitoring Report — {run_date}")
    lines.append("")

    if no_data:
        lines.append("> **No portfolio data available.** Enable WF4 paper trading")
        lines.append("> (`WF4_PAPER_TRADING_ENABLED=true`) to start tracking performance.")
        lines.append("")
        lines.append("*Generated by WF5 Monitoring & Reporting | Not investment advice*")
        report = "\n".join(lines)
        _store_and_upload(run_date, pnl_data, risk_data, alert_data, report)
        return report

    # Portfolio Summary
    lines.append("## Portfolio Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Value | EUR {float(pnl_data['portfolio_value']):,.2f} |")
    lines.append(f"| Cash | EUR {float(pnl_data['cash']):,.2f} |")
    lines.append(f"| Invested | EUR {float(pnl_data['invested']):,.2f} |")
    lines.append(f"| Positions | {pnl_data['num_positions']} |")
    lines.append("")

    # P&L Summary
    lines.append("## P&L Summary")
    lines.append("")
    lines.append("| Period | P&L |")
    lines.append("|--------|-----|")
    daily_pnl = float(pnl_data["daily_pnl"])
    daily_pct = float(pnl_data["daily_pnl_pct"])
    lines.append(f"| Daily | EUR {daily_pnl:+,.2f} ({daily_pct:+.2f}%) |")
    lines.append(f"| MTD | EUR {float(pnl_data['mtd_pnl']):+,.2f} |")
    lines.append(f"| YTD | EUR {float(pnl_data['ytd_pnl']):+,.2f} |")
    lines.append("")

    # Dividends
    try:
        from src.shared.db import get_dividend_summary
        from src.shared.config import WF4_DIVIDEND_REINVEST

        div_summary = get_dividend_summary(run_date)
        if div_summary["cumulative"] > 0:
            lines.append("## Dividends")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Cumulative | EUR {div_summary['cumulative']:,.2f} |")
            lines.append(f"| MTD | EUR {div_summary['mtd']:,.2f} |")
            lines.append(f"| YTD | EUR {div_summary['ytd']:,.2f} |")
            mode = "DRIP" if WF4_DIVIDEND_REINVEST else "Cash"
            lines.append(f"| Mode | {mode} |")
            lines.append("")
    except Exception:
        pass  # Dividend section is optional; skip on DB errors

    # Top Winners / Losers
    lines.append("## Top Winners & Losers")
    lines.append("")
    winners = pnl_data.get("top_winners", "")
    losers = pnl_data.get("top_losers", "")
    if winners:
        lines.append(f"**Winners:** {winners.replace('|', ', ')}")
    else:
        lines.append("**Winners:** None")
    if losers:
        lines.append(f"**Losers:** {losers.replace('|', ', ')}")
    else:
        lines.append("**Losers:** None")
    lines.append("")

    # Risk Metrics
    data_points = int(risk_data.get("data_points", "0"))
    lines.append("## Risk Metrics (30d)")
    lines.append("")
    if data_points < 5:
        lines.append(f"> *Computed from {data_points} data points. Recommend >20 for reliability.*")
        lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Sharpe Ratio | {float(risk_data['sharpe_30d']):.2f} |")
    lines.append(f"| Sortino Ratio | {float(risk_data['sortino_30d']):.2f} |")
    lines.append(f"| Max Drawdown | {float(risk_data['max_drawdown_30d']):.2%} |")
    lines.append(f"| VaR (95%) | EUR {float(risk_data['var_95']):,.2f} |")
    lines.append(f"| Data Points | {data_points} |")
    lines.append("")

    # Sector Allocation
    sector_conc = json.loads(risk_data.get("sector_concentration_json", "{}"))
    if sector_conc:
        lines.append("## Sector Allocation")
        lines.append("")
        lines.append("| Sector | Weight |")
        lines.append("|--------|--------|")
        for sector, weight in sector_conc.items():
            lines.append(f"| {sector} | {weight:.1%} |")
        lines.append("")

    # Alerts
    num_alerts = int(alert_data.get("num_alerts", "0"))
    alerts_csv = alert_data.get("alerts_csv", "")
    lines.append("## Alerts")
    lines.append("")
    if num_alerts == 0:
        lines.append("No alerts triggered.")
    else:
        has_critical = alert_data.get("has_critical") == "true"
        if has_critical:
            lines.append(f"> **{num_alerts} alert(s) — CRITICAL alerts present**")
        else:
            lines.append(f"> {num_alerts} alert(s)")
        lines.append("")
        for alert in alerts_csv.split("|"):
            alert = alert.strip()
            if alert:
                lines.append(f"- {alert}")
    lines.append("")

    # Footer
    lines.append("---")
    lines.append("*Generated by WF5 Monitoring & Reporting | Not investment advice*")

    report = "\n".join(lines)

    # Store to DB + upload to MinIO
    _store_and_upload(run_date, pnl_data, risk_data, alert_data, report)

    return report


def _store_and_upload(
    run_date: str,
    pnl_data: Dict[str, str],
    risk_data: Dict[str, str],
    alert_data: Dict[str, str],
    report: str,
) -> None:
    """Store monitoring run to DB and upload report to MinIO (helper)."""
    import json

    from src.shared.db import store_monitoring_run

    # Upload report to MinIO
    s3_path = ""
    try:
        s3_path = _upload_report_to_minio(report, run_date)
    except Exception:
        pass  # Non-fatal: report is returned as string regardless

    # Store run metadata to DB
    alerts_list = [
        a.strip()
        for a in alert_data.get("alerts_csv", "").split("|")
        if a.strip()
    ]
    monitoring_data = {
        "portfolio_value": float(pnl_data.get("portfolio_value", "0")),
        "daily_pnl": float(pnl_data.get("daily_pnl", "0")),
        "daily_pnl_pct": float(pnl_data.get("daily_pnl_pct", "0")),
        "mtd_pnl": float(pnl_data.get("mtd_pnl", "0")),
        "ytd_pnl": float(pnl_data.get("ytd_pnl", "0")),
        "sharpe_30d": float(risk_data.get("sharpe_30d", "0")),
        "sortino_30d": float(risk_data.get("sortino_30d", "0")),
        "max_drawdown_30d": float(risk_data.get("max_drawdown_30d", "0")),
        "var_95": float(risk_data.get("var_95", "0")),
        "num_positions": int(pnl_data.get("num_positions", "0")),
        "num_alerts": int(alert_data.get("num_alerts", "0")),
        "alerts_json": json.dumps(alerts_list),
        "report_s3_path": s3_path,
    }
    try:
        store_monitoring_run(run_date, monitoring_data)
    except Exception:
        pass  # Non-fatal: report is returned as string regardless


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
    s3_key = f"reports/wf5/year={year}/month={month}/day={day}/monitoring_report.md"

    client = get_s3_client()
    client.put_object(
        Bucket=S3_DATA_BUCKET,
        Key=s3_key,
        Body=report.encode("utf-8"),
        ContentType="text/markdown",
    )
    return f"s3://{S3_DATA_BUCKET}/{s3_key}"
