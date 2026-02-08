"""WF3: Signal & Analysis - Tasks (Phase 2/4).

Weekly pipeline that computes technical indicators, fundamental analysis,
and sentiment analysis (Hailo NPU) to generate composite scores.

Schedule: Weekly after WF2 (Sunday 12:00 UTC)
Node: Pi 4 Workers + Pi 5 AI (Hailo) for sentiment

Planned tasks (Phase 2):
- technical_analysis: SMA, RSI, MACD, Bollinger, ATR (252-day lookback)
- fundamental_analysis: P/E z-score, dividend yield, ROE, D/E ratio

Planned tasks (Phase 4):
- sentiment_analysis: NLP on Hailo-10H NPU (DistilBERT-Finance)
- combine_signals: Weighted composite score (Tech 30% + Fund 40% + Sent 30%)
"""

from flytekit import task, Resources


@task(
    requests=Resources(cpu="100m", mem="128Mi"),
    limits=Resources(cpu="200m", mem="256Mi"),
)
def signal_analysis_placeholder() -> str:
    """Placeholder for WF3 signal analysis tasks.

    Will be replaced with actual implementation:
    Phase 2: Technical (SMA, RSI, MACD, Bollinger, ATR) + Fundamental analysis
    Phase 4: Sentiment analysis on Hailo NPU + composite scoring
    """
    return "WF3: Signal & Analysis - Not implemented yet (Phase 2/4)"
