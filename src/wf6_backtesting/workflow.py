"""WF6: Backtesting — Workflow.

Weekly workflow that replays historical signal results through the pension fund
portfolio construction model and compares performance against a buy-and-hold
benchmark.

Schedule: Weekly Sunday 11:00 UTC

Pipeline:
    resolve_backtest_params
            │
      ┌─────┴──────┐
      │            │
  load_signals  (params)
      │            │
      ├────────────┤
      │            │
  simulate_    compute_
  portfolio    benchmark         ← PARALLEL
      │            │
      └─────┬──────┘
            │
      compare_strategies
            │
      generate_report

Example local run:
    pyflyte run src/wf6_backtesting/workflow.py backtesting_workflow \\
        --start_date 2026-01-01 --end_date 2026-02-15
"""

from flytekit import workflow

from src.shared.config import WF4_INITIAL_CAPITAL
from src.wf6_backtesting.tasks import (
    resolve_backtest_params,
    load_historical_signals,
    simulate_signal_portfolio,
    compute_benchmark_returns,
    compare_strategies,
    generate_backtest_report,
)


@workflow
def backtesting_workflow(
    start_date: str = "2026-01-01",
    end_date: str = "",
    initial_capital: float = WF4_INITIAL_CAPITAL,
    max_position_pct: float = 0.05,
    max_sector_pct: float = 0.25,
    cash_reserve_pct: float = 0.05,
) -> str:
    """WF6: Weekly backtesting workflow.

    Reads historical signal_results from PostgreSQL, replays portfolio
    construction using the pension fund model, and compares against
    equal-weight buy-and-hold.

    Args:
        start_date: Backtest start date (YYYY-MM-DD, default: 2026-01-01).
        end_date: Backtest end date (YYYY-MM-DD, empty = today).
        initial_capital: Starting capital in EUR (default: from WF4_INITIAL_CAPITAL).
        max_position_pct: Maximum weight per stock (default: 5%).
        max_sector_pct: Maximum weight per sector (default: 25%).
        cash_reserve_pct: Target cash reserve fraction (default: 5%).

    Returns:
        Backtest comparison report as formatted text.
    """
    # Step 1: Resolve and validate parameters
    params = resolve_backtest_params(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        max_position_pct=max_position_pct,
        max_sector_pct=max_sector_pct,
        cash_reserve_pct=cash_reserve_pct,
    )

    # Step 2: Load historical signals from DB
    signals = load_historical_signals(params=params)

    # Step 3: PARALLEL — Simulate portfolio + compute benchmark
    portfolio = simulate_signal_portfolio(params=params, signals=signals)
    benchmark = compute_benchmark_returns(params=params, signals=signals)

    # Step 4: Compare strategies
    comparison = compare_strategies(
        params=params,
        portfolio=portfolio,
        benchmark=benchmark,
    )

    # Step 5: Generate report + store to DB + MinIO
    report = generate_backtest_report(
        params=params,
        comparison=comparison,
        portfolio=portfolio,
        benchmark=benchmark,
    )

    return report
