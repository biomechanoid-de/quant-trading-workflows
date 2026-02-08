"""WF2: Universe & Screening - Tasks (Phase 2).

Weekly pipeline that defines the stock universe, applies filters,
and ranks candidates using a multi-factor model.

Schedule: Weekly Sunday 08:00 UTC
Node: Any Pi 4 Worker

Planned tasks (Phase 2):
- define_universe: Build global stock universe by region (FTSE Global All Cap)
- apply_filters: Liquidity, market cap, ethical exclusions
- rank_stocks: Multi-factor ranking (Value, Quality, Momentum, Low Volatility)
- output_watchlist: Generate ranked watchlist for WF3
"""

from flytekit import task, Resources


@task(
    requests=Resources(cpu="100m", mem="128Mi"),
    limits=Resources(cpu="200m", mem="256Mi"),
)
def universe_screening_placeholder() -> str:
    """Placeholder for WF2 universe screening tasks.

    Will be replaced with actual implementation in Phase 2:
    - UniverseConfig with regions, market cap filters, ethical exclusions
    - ScreeningResult with ranked StockCandidates
    - Multi-factor model (P/E, P/B, Dividend Yield, ROE, Momentum)
    """
    return "WF2: Universe & Screening - Not implemented yet (Phase 2)"
