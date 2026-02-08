"""WF3: Signal & Analysis - Workflow (Phase 2/4).

Planned pipeline (parallel branches):
    watchlist_input -> [technical_analysis, fundamental_analysis, sentiment_analysis]
                    -> combine_signals -> generate_scores

Parallel execution: Technical + Fundamental + Sentiment run in parallel,
then results are combined with configurable weights.
"""

from flytekit import workflow

from src.wf3_signal_analysis.tasks import signal_analysis_placeholder


@workflow
def signal_analysis_workflow() -> str:
    """WF3: Weekly signal analysis workflow (stub).

    Phase 2: Technical + Fundamental analysis (weights: 50/50)
    Phase 4: + Sentiment via Hailo NPU (weights: 30/40/30)
    """
    return signal_analysis_placeholder()
