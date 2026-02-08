"""WF2: Universe & Screening - Workflow.

Weekly pipeline that screens stocks via multi-factor model:
1. Load historical prices from PostgreSQL (WF1 data)
2. Compute returns, RSI, and performance metrics per stock
3. In parallel: K-Means clustering + Multi-factor scoring
4. Merge cluster assignments into ranked results
5. Assemble final ScreeningResult with benchmark performance
6. In parallel: Store to DB + Store to Parquet + Generate report

Schedule: Weekly Sunday 08:00 UTC (configured in launch_plans/)

Task DAG:
    load_historical_prices -> compute_returns_and_metrics -> +- cluster_stocks        -+
                                                             +- score_and_rank_factors -+
                                                                                        v
                                                              merge_cluster_assignments
                                                                         |
                                                         assemble_screening_result
                                                                         |
                                              +---------------------------+---------------+
                                              v                           v               v
                                        store_to_db             store_to_parquet   generate_report

Example local run:
    pyflyte run src/wf2_universe_screening/workflow.py universe_screening_workflow \\
        --symbols '["AAPL", "MSFT", "GOOGL"]' --lookback_days 30 --run_date 2026-02-08
"""

from typing import Dict, List

from flytekit import workflow

from src.shared.config import (
    PHASE2_SYMBOLS,
    WF2_LOOKBACK_DAYS,
    WF2_MOMENTUM_WINDOWS,
    WF2_RSI_WINDOW,
    WF2_RSI_OVERSOLD,
    WF2_RSI_OVERBOUGHT,
    WF2_FORECAST_HORIZON,
    WF2_KMEANS_MAX_K,
)
from src.wf2_universe_screening.tasks import (
    load_historical_prices,
    compute_returns_and_metrics,
    cluster_stocks,
    score_and_rank_factors,
    merge_cluster_assignments,
    assemble_screening_result,
    store_screening_to_db,
    store_screening_to_parquet,
    generate_screening_report,
)

# Default factor weights (constant, not a workflow parameter to keep interface simple)
_FACTOR_WEIGHTS: Dict[str, float] = {
    "momentum": 0.30,
    "low_volatility": 0.25,
    "rsi_signal": 0.20,
    "sharpe": 0.25,
}


@workflow
def universe_screening_workflow(
    symbols: List[str] = PHASE2_SYMBOLS,
    lookback_days: int = WF2_LOOKBACK_DAYS,
    run_date: str = "",
) -> str:
    """WF2: Weekly universe screening workflow.

    Multi-factor screening pipeline inspired by NBIM/Brenndoerfer:
    - Compute returns, RSI, and performance metrics
    - K-Means clustering by momentum/volatility
    - Z-score normalized factor scoring with quintile ranking
    - Store results to PostgreSQL + Parquet/MinIO

    ScreeningConfig is constructed INSIDE tasks (never passed between them)
    to avoid Flytekit serialization issues with dataclasses containing
    List/Dict fields ("Promise objects are not iterable").

    Args:
        symbols: Stock symbols to screen. Default: 49 US Large/Mid Caps.
        lookback_days: Historical data lookback (calendar days). Default: 252.
        run_date: Screening date (YYYY-MM-DD). Empty string = today.

    Returns:
        Screening report as formatted string.
    """
    # Step 1: Load historical prices from PostgreSQL
    price_data = load_historical_prices(
        symbols=symbols,
        lookback_days=lookback_days,
    )

    # Step 2: Compute returns and metrics for each stock
    # Pass individual params instead of ScreeningConfig (Flytekit compatibility)
    stock_metrics = compute_returns_and_metrics(
        price_data=price_data,
        forecast_horizon=WF2_FORECAST_HORIZON,
        momentum_windows=WF2_MOMENTUM_WINDOWS,
        rsi_window=WF2_RSI_WINDOW,
        rsi_oversold=WF2_RSI_OVERSOLD,
        rsi_overbought=WF2_RSI_OVERBOUGHT,
    )

    # Step 3a + 3b: PARALLEL - Cluster stocks and score/rank factors
    # Flyte executes these in parallel (no dependency between them)
    cluster_assignments = cluster_stocks(
        stock_metrics=stock_metrics,
        max_k=WF2_KMEANS_MAX_K,
    )
    ranked_metrics = score_and_rank_factors(
        stock_metrics=stock_metrics,
        factor_weights=_FACTOR_WEIGHTS,
    )

    # Step 4: Merge cluster assignments into ranked metrics
    final_metrics = merge_cluster_assignments(
        ranked_metrics=ranked_metrics,
        cluster_assignments=cluster_assignments,
    )

    # Step 5: Assemble ScreeningResult with benchmark performance
    # Pass individual config params (ScreeningConfig constructed inside task)
    result = assemble_screening_result(
        run_date=run_date,
        final_metrics=final_metrics,
        price_data=price_data,
        symbols=symbols,
        lookback_days=lookback_days,
        forecast_horizon=WF2_FORECAST_HORIZON,
        rsi_window=WF2_RSI_WINDOW,
        rsi_oversold=WF2_RSI_OVERSOLD,
        rsi_overbought=WF2_RSI_OVERBOUGHT,
        kmeans_max_k=WF2_KMEANS_MAX_K,
    )

    # Step 6a + 6b + 6c: PARALLEL - Store and report
    # Flyte runs these in parallel (all depend only on 'result')
    store_screening_to_db(result=result)
    store_screening_to_parquet(result=result)
    report = generate_screening_report(result=result)

    return report
