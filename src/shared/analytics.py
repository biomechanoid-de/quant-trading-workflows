"""Shared analytics functions for Quant Trading Workflows.

Pure computational functions used by WF2 (screening), WF3 (signals),
and WF5 (monitoring). No database or storage access — takes and returns
pandas objects or scalars.

Sources:
- Udacity AI Trading Strategies Course 2 (func_lib.py: computing_returns,
  calculate_rsi, compute_BM_Perf, compute_strat_perf)
- Brenndoerfer (factor-investing, portfolio-performance-measurement,
  machine-learning-techniques)

Lazy imports for pandas/numpy inside functions (test isolation pattern).
"""


def calculate_rsi(returns_series: "pd.Series", window: int = 14) -> "pd.Series":
    """Calculate RSI (Relative Strength Index) from a returns series.

    Following Udacity Course 2 methodology:
    - Separate positive gains and negative losses (abs)
    - Rolling mean of gains and losses
    - RS = avg_gain / avg_loss (with floor of 0.001 on losses)
    - RSI = 100 - (100 / (1 + RS))

    Args:
        returns_series: Daily returns (pct_change).
        window: RSI lookback window (default: 14).

    Returns:
        Series of RSI values (0-100).
    """
    gains = returns_series.clip(lower=0)
    losses = returns_series.clip(upper=0).abs()

    avg_gain = gains.rolling(window=window, min_periods=window).mean()
    avg_loss = losses.rolling(window=window, min_periods=window).mean()

    # Floor to prevent division by zero
    avg_loss = avg_loss.clip(lower=0.001)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def classify_rsi_signal(
    rsi_value: float,
    oversold: int = 30,
    overbought: int = 70,
) -> str:
    """Classify RSI value into a trading signal.

    Args:
        rsi_value: RSI value (0-100).
        oversold: Threshold below which stock is considered oversold.
        overbought: Threshold above which stock is considered overbought.

    Returns:
        One of: "oversold", "neutral", "overbought"
    """
    if rsi_value <= oversold:
        return "oversold"
    elif rsi_value >= overbought:
        return "overbought"
    return "neutral"


def compute_cagr(cumulative_returns: "pd.Series") -> float:
    """Compute Compound Annual Growth Rate.

    CAGR = (ending / beginning) ^ (1 / years) - 1
    where years = trading_days / 252

    Following Udacity compute_BM_Perf pattern.

    Args:
        cumulative_returns: Cumulative return series (starting from 1.0).

    Returns:
        CAGR as a decimal (e.g., 0.12 = 12%).
    """
    if len(cumulative_returns) < 2:
        return 0.0
    ending = cumulative_returns.iloc[-1]
    beginning = cumulative_returns.iloc[0]
    if beginning <= 0:
        return 0.0
    years = len(cumulative_returns) / 252
    if years <= 0:
        return 0.0
    return (ending / beginning) ** (1 / years) - 1


def compute_sharpe(
    daily_returns: "pd.Series",
    risk_free_rate: float = 0.0,
) -> float:
    """Compute annualized Sharpe ratio.

    Sharpe = (mean_daily - rf_daily) / std_daily * sqrt(252)

    Following Brenndoerfer + Udacity formula.

    Args:
        daily_returns: Series of daily returns.
        risk_free_rate: Annualized risk-free rate (default: 0.0).

    Returns:
        Annualized Sharpe ratio.
    """
    import numpy as np

    clean = daily_returns.dropna()
    if len(clean) < 2 or clean.std() == 0:
        return 0.0
    rf_daily = risk_free_rate / 252
    excess = clean.mean() - rf_daily
    return float((excess / clean.std()) * np.sqrt(252))


def compute_sortino(
    daily_returns: "pd.Series",
    risk_free_rate: float = 0.0,
) -> float:
    """Compute annualized Sortino ratio.

    Sortino = (mean_daily - rf_daily) / downside_std * sqrt(252)
    Downside std uses only negative returns.

    Following Brenndoerfer portfolio-performance-measurement pattern.

    Args:
        daily_returns: Series of daily returns.
        risk_free_rate: Annualized risk-free rate (default: 0.0).

    Returns:
        Annualized Sortino ratio.
    """
    import numpy as np

    clean = daily_returns.dropna()
    if len(clean) < 2:
        return 0.0
    rf_daily = risk_free_rate / 252
    downside = clean[clean < 0]
    if len(downside) < 2 or downside.std() == 0:
        return 0.0
    excess = clean.mean() - rf_daily
    return float((excess / downside.std()) * np.sqrt(252))


def compute_max_drawdown(cumulative_returns: "pd.Series") -> float:
    """Compute maximum drawdown.

    Max Drawdown = min((current - peak) / peak)
    Returns a negative number (e.g., -0.15 = 15% drawdown).

    Following Brenndoerfer calculate_drawdown pattern.

    Args:
        cumulative_returns: Cumulative return series (starting from 1.0).

    Returns:
        Maximum drawdown as a negative decimal.
    """
    if len(cumulative_returns) < 2:
        return 0.0
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return float(drawdown.min())


def compute_calmar(cagr: float, max_drawdown: float) -> float:
    """Compute Calmar ratio.

    Calmar = CAGR / |Max Drawdown|

    Following Brenndoerfer benchmark:
    - Calmar < 0.5: Poor
    - 0.5-1.0: Acceptable
    - 1.0-2.0: Good
    - > 2.0: Excellent

    Args:
        cagr: Compound annual growth rate.
        max_drawdown: Maximum drawdown (negative number).

    Returns:
        Calmar ratio. Returns 0.0 if max_drawdown is 0.
    """
    if max_drawdown == 0:
        return 0.0
    return cagr / abs(max_drawdown)


def compute_benchmark_performance(
    prices_df: "pd.DataFrame",
) -> tuple:
    """Compute equal-weight benchmark performance for the universe.

    Following Udacity compute_BM_Perf pattern:
    - Equal-weight daily returns across all symbols
    - Cumulative returns = (1 + daily_mean).cumprod()
    - CAGR and Sharpe from the benchmark series

    Args:
        prices_df: DataFrame with columns = symbols, index = dates, values = close.

    Returns:
        Tuple of (cagr, sharpe, cumulative_return_final).
    """
    daily_returns = prices_df.pct_change().dropna()
    if daily_returns.empty:
        return (0.0, 0.0, 0.0)

    # Equal-weight: mean across all symbols each day
    bm_daily = daily_returns.mean(axis=1)
    bm_cumulative = (1 + bm_daily).cumprod()

    cagr = compute_cagr(bm_cumulative)
    sharpe = compute_sharpe(bm_daily)
    cumulative_final = float(bm_cumulative.iloc[-1]) - 1  # Total return

    return (cagr, sharpe, cumulative_final)


def zscore_normalize(series: "pd.Series") -> "pd.Series":
    """Z-score normalize a series: (x - mean) / std.

    Brenndoerfer factor investing pattern.

    Args:
        series: Raw factor values.

    Returns:
        Z-score normalized series.
    """
    if series.std() == 0:
        return series * 0  # All zeros if no variance
    return (series - series.mean()) / series.std()


def assign_quintiles(scores: "pd.Series") -> "pd.Series":
    """Assign quintile rankings (1 = best, 5 = worst).

    Brenndoerfer pattern: pd.qcut with 5 bins.
    Higher composite score = better = Quintile 1.

    Args:
        scores: Composite scores (higher is better).

    Returns:
        Series of quintile labels (1-5 as integers).
    """
    import pandas as pd

    if len(scores) < 5:
        # Not enough data for 5 bins — rank relatively
        ranks = scores.rank(ascending=False, method="min")
        # Scale 1..N to 1..5
        n = len(scores)
        quintiles = ((ranks - 1) / max(n - 1, 1) * 4 + 1).round().astype(int).clip(1, 5)
        return quintiles

    try:
        quintiles = pd.qcut(
            scores.rank(method="first"),
            q=5,
            labels=[5, 4, 3, 2, 1],  # Lowest rank scores = Q5, highest = Q1
        )
        return quintiles.astype(int)
    except ValueError:
        # Not enough unique values for 5 bins
        import pandas as pd
        return pd.Series(3, index=scores.index)
