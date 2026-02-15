"""WF3: Signal & Analysis - Workflow.

Daily workflow that computes technical indicators, fundamental analysis,
and sentiment analysis for WF2's top-ranked stocks, generating composite
buy/hold/sell signals.

Schedule: Daily 08:00 UTC (1 hour after WF2)

Pipeline:
    load_screening_context
            │
      ┌─────┼──────────┐
      │     │          │
  compute_  fetch_     fetch_
  technical fundamental sentiment    ← PARALLEL (Flyte DAG, 3 branches)
  signals   data       data
      │     │          │
      └─────┼──────────┘
            │
      combine_signals
            │
      assemble_signal_result
            │
      ┌─────┼──────────┐
      │     │          │
  store_db  store_parq  report       ← PARALLEL (Flyte DAG)

Example local run:
    pyflyte run src/wf3_signal_analysis/workflow.py signal_analysis_workflow \\
        --run_date 2026-02-09 --max_quintile 2 --lookback_days 252
"""

from flytekit import workflow

from src.shared.config import (
    WF3_MAX_QUINTILE,
    WF3_LOOKBACK_DAYS,
    WF3_TECH_WEIGHT,
    WF3_FUND_WEIGHT,
    WF3_SENT_WEIGHT,
    WF3_SMA_SHORT,
    WF3_SMA_LONG,
    SENTIMENT_NEWS_DAYS,
    SENTIMENT_DECAY_HALF_LIFE,
)
from src.wf3_signal_analysis.tasks import (
    resolve_run_date,
    load_screening_context,
    compute_technical_signals,
    fetch_fundamental_data,
    fetch_sentiment_data,
    combine_signals,
    assemble_signal_result,
    store_signals_to_db,
    store_signals_to_parquet,
    generate_signal_report,
)


@workflow
def signal_analysis_workflow(
    run_date: str = "",
    max_quintile: int = WF3_MAX_QUINTILE,
    lookback_days: int = WF3_LOOKBACK_DAYS,
    tech_weight: float = WF3_TECH_WEIGHT,
    fund_weight: float = WF3_FUND_WEIGHT,
    sent_weight: float = WF3_SENT_WEIGHT,
    sma_short: int = WF3_SMA_SHORT,
    sma_long: int = WF3_SMA_LONG,
    news_days: int = SENTIMENT_NEWS_DAYS,
    decay_half_life: float = SENTIMENT_DECAY_HALF_LIFE,
) -> str:
    """WF3: Daily signal analysis workflow.

    Reads WF2 screening results for top-quintile stocks, computes
    technical indicators (SMA, MACD, Bollinger), fundamental metrics
    (P/E, ROE, D/E), and news sentiment (DistilBERT-Finance via ONNX),
    combines them with configurable weights, and stores results to
    PostgreSQL + MinIO.

    Phase 6: 30% technical + 40% fundamental + 30% sentiment.
    Rollback: set sent_weight=0.0, tech_weight=0.5, fund_weight=0.5.

    Args:
        run_date: Target date (YYYY-MM-DD). Empty = latest WF2 run.
        max_quintile: Max quintile from WF2 to include (default: 2 = top 40%).
        lookback_days: Price history days for technical indicators (default: 252).
        tech_weight: Technical signal weight (default: 0.30).
        fund_weight: Fundamental signal weight (default: 0.40).
        sent_weight: Sentiment signal weight (default: 0.30).
        sma_short: Short SMA window (default: 50).
        sma_long: Long SMA window (default: 200).
        news_days: Days of news history for sentiment (default: 7).
        decay_half_life: Time-decay half-life in days for sentiment (default: 3.0).

    Returns:
        Signal analysis report as formatted text.
    """
    # Step 0: Resolve empty run_date to latest WF2 screening date
    # (PostgreSQL DATE columns cannot accept empty strings)
    resolved_date = resolve_run_date(run_date=run_date)

    # Step 1: Load WF2 screening context (top quintiles)
    screening_ctx = load_screening_context(
        run_date=resolved_date,
        max_quintile=max_quintile,
    )

    # Step 2: PARALLEL — Technical + Fundamental + Sentiment (3 branches)
    tech_signals = compute_technical_signals(
        screening_context=screening_ctx,
        lookback_days=lookback_days,
        sma_short=sma_short,
        sma_long=sma_long,
    )
    fund_signals = fetch_fundamental_data(
        screening_context=screening_ctx,
    )
    sent_signals = fetch_sentiment_data(
        screening_context=screening_ctx,
        run_date=resolved_date,
        news_days=news_days,
        decay_half_life=decay_half_life,
    )

    # Step 3: Combine all three signal types
    signal_results = combine_signals(
        screening_context=screening_ctx,
        tech_signals=tech_signals,
        fund_signals=fund_signals,
        sent_signals=sent_signals,
        tech_weight=tech_weight,
        fund_weight=fund_weight,
        sent_weight=sent_weight,
        run_date=resolved_date,
    )

    # Step 4: Assemble result (serialize for downstream tasks)
    assembled = assemble_signal_result(
        run_date=resolved_date,
        signal_results=signal_results,
        tech_weight=tech_weight,
        fund_weight=fund_weight,
        sent_weight=sent_weight,
    )

    # Step 5: PARALLEL — Store to DB, store to Parquet, generate report
    store_db_result = store_signals_to_db(assembled_result=assembled)
    store_s3_result = store_signals_to_parquet(assembled_result=assembled)
    report = generate_signal_report(assembled_result=assembled)

    return report
