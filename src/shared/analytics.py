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


# ============================================================
# WF3: Technical Indicators (close-only — no OHLC needed)
# ============================================================

def calculate_sma(prices: list, window: int) -> list:
    """Calculate Simple Moving Average from a list of close prices.

    Uses a rolling window average. Returns NaN for positions where
    there are fewer than `window` data points.

    Args:
        prices: List of close prices (chronological order, oldest first).
        window: SMA lookback window (e.g., 50, 200).

    Returns:
        List of SMA values (same length as prices, leading NaNs).
    """
    if not prices or window <= 0:
        return []
    result = []
    for i in range(len(prices)):
        if i < window - 1:
            result.append(float("nan"))
        else:
            window_slice = prices[i - window + 1:i + 1]
            result.append(sum(window_slice) / window)
    return result


def calculate_sma_crossover_signal(
    prices: list,
    short_window: int = 50,
    long_window: int = 200,
) -> str:
    """Determine SMA crossover signal (Golden Cross / Death Cross).

    - Golden Cross: Short SMA crosses above Long SMA → "bullish"
    - Death Cross: Short SMA crosses below Long SMA → "bearish"
    - No clear signal → "neutral"

    Requires at least `long_window` data points.

    Args:
        prices: List of close prices (chronological order).
        short_window: Short-term SMA window (default: 50).
        long_window: Long-term SMA window (default: 200).

    Returns:
        One of: "bullish", "bearish", "neutral"
    """
    if len(prices) < long_window:
        return "neutral"

    sma_short = calculate_sma(prices, short_window)
    sma_long = calculate_sma(prices, long_window)

    # Check the last two valid points for crossover detection
    import math
    if math.isnan(sma_short[-1]) or math.isnan(sma_long[-1]):
        return "neutral"

    # Current relationship
    if sma_short[-1] > sma_long[-1]:
        # Check if this is a recent cross (within last 5 days)
        for i in range(max(len(prices) - 5, long_window - 1), len(prices) - 1):
            if not math.isnan(sma_short[i]) and not math.isnan(sma_long[i]):
                if sma_short[i] <= sma_long[i]:
                    return "bullish"  # Golden Cross detected
        # Short above long but no recent cross — still bullish bias
        return "bullish"
    elif sma_short[-1] < sma_long[-1]:
        # Check for recent death cross
        for i in range(max(len(prices) - 5, long_window - 1), len(prices) - 1):
            if not math.isnan(sma_short[i]) and not math.isnan(sma_long[i]):
                if sma_short[i] >= sma_long[i]:
                    return "bearish"  # Death Cross detected
        return "bearish"
    return "neutral"


def _ema(prices: list, span: int) -> list:
    """Calculate Exponential Moving Average (helper for MACD).

    Uses the standard EMA formula: EMA_t = α * price_t + (1-α) * EMA_{t-1}
    where α = 2 / (span + 1).

    Args:
        prices: List of close prices.
        span: EMA span (e.g., 12, 26).

    Returns:
        List of EMA values (same length as prices).
    """
    if not prices:
        return []
    alpha = 2.0 / (span + 1)
    ema = [prices[0]]
    for i in range(1, len(prices)):
        ema.append(alpha * prices[i] + (1 - alpha) * ema[-1])
    return ema


def calculate_macd(
    prices: list,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple:
    """Calculate MACD (Moving Average Convergence Divergence).

    MACD Line = EMA(fast) - EMA(slow)
    Signal Line = EMA(MACD Line, signal period)
    Histogram = MACD Line - Signal Line

    Args:
        prices: List of close prices (chronological order).
        fast: Fast EMA period (default: 12).
        slow: Slow EMA period (default: 26).
        signal: Signal line EMA period (default: 9).

    Returns:
        Tuple of (macd_line, signal_line, histogram) — each a float
        representing the latest value. Returns (0.0, 0.0, 0.0) if
        insufficient data.
    """
    if len(prices) < slow + signal:
        return (0.0, 0.0, 0.0)

    ema_fast = _ema(prices, fast)
    ema_slow = _ema(prices, slow)

    # MACD line = fast EMA - slow EMA
    macd_line = [ema_fast[i] - ema_slow[i] for i in range(len(prices))]

    # Signal line = EMA of MACD line
    signal_line = _ema(macd_line, signal)

    # Histogram = MACD - Signal
    histogram = [macd_line[i] - signal_line[i] for i in range(len(prices))]

    return (
        round(macd_line[-1], 6),
        round(signal_line[-1], 6),
        round(histogram[-1], 6),
    )


def classify_macd_signal(macd: float, signal_line: float, histogram: float) -> str:
    """Classify MACD into a trading signal.

    - Bullish: MACD > Signal Line AND histogram positive (upward momentum)
    - Bearish: MACD < Signal Line AND histogram negative (downward momentum)
    - Neutral: otherwise

    Args:
        macd: MACD line value.
        signal_line: Signal line value.
        histogram: MACD histogram value.

    Returns:
        One of: "bullish", "bearish", "neutral"
    """
    if macd > signal_line and histogram > 0:
        return "bullish"
    elif macd < signal_line and histogram < 0:
        return "bearish"
    return "neutral"


def calculate_bollinger_bands(
    prices: list,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple:
    """Calculate Bollinger Bands from close prices.

    - Middle Band = SMA(window)
    - Upper Band = Middle + num_std * std(window)
    - Lower Band = Middle - num_std * std(window)

    Args:
        prices: List of close prices (chronological order).
        window: Rolling window (default: 20).
        num_std: Number of standard deviations (default: 2.0).

    Returns:
        Tuple of (upper, middle, lower) — each a float representing
        the latest value. Returns (0.0, 0.0, 0.0) if insufficient data.
    """
    if len(prices) < window:
        return (0.0, 0.0, 0.0)

    # Use the last `window` prices
    window_prices = prices[-window:]
    middle = sum(window_prices) / window

    # Standard deviation
    variance = sum((p - middle) ** 2 for p in window_prices) / window
    std = variance ** 0.5

    upper = middle + num_std * std
    lower = middle - num_std * std

    return (round(upper, 4), round(middle, 4), round(lower, 4))


def classify_bollinger_signal(
    price: float,
    upper: float,
    lower: float,
    middle: float,
) -> str:
    """Classify current price position relative to Bollinger Bands.

    - "oversold": Price at or below lower band (potential buy)
    - "overbought": Price at or above upper band (potential sell)
    - "neutral": Price between bands

    Args:
        price: Current close price.
        upper: Upper Bollinger Band.
        lower: Lower Bollinger Band.
        middle: Middle Bollinger Band (SMA).

    Returns:
        One of: "oversold", "overbought", "neutral"
    """
    if upper == 0 and lower == 0:
        return "neutral"
    if price <= lower:
        return "oversold"
    elif price >= upper:
        return "overbought"
    return "neutral"


# ============================================================
# WF3: Fundamental Analysis
# ============================================================

def normalize_pe_ratio(pe: float, sector_median: float = 20.0) -> float:
    """Normalize P/E ratio to a z-score relative to sector median.

    Lower P/E relative to sector = more undervalued = positive z-score
    (value-investing perspective).

    Args:
        pe: Trailing P/E ratio. -1.0 means missing data.
        sector_median: Sector median P/E (default: 20.0).

    Returns:
        Z-score float. Returns 0.0 for missing data (pe <= 0 or pe == -1).
    """
    if pe <= 0 or pe == -1.0:
        return 0.0
    # Invert: lower PE = higher score (more undervalued)
    return (sector_median - pe) / sector_median


def compute_fundamental_score(
    pe_zscore: float,
    div_yield: float,
    roe: float,
    de_ratio: float,
    current_ratio: float = 1.5,
) -> float:
    """Compute a composite fundamental score (0-100).

    Weighted factors (value-oriented):
    - P/E z-score: 30% (lower PE = higher score)
    - Dividend yield: 15% (higher = better)
    - ROE: 25% (higher = better, capped at 50%)
    - D/E ratio: 15% (lower = better, capped score)
    - Current ratio: 15% (closer to 1.5 = better)

    Missing data (signaled by -1.0 or 0.0) gets a neutral 50 for that factor.

    Args:
        pe_zscore: Normalized P/E z-score from normalize_pe_ratio.
        div_yield: Dividend yield as decimal (e.g., 0.02 = 2%).
        roe: Return on equity as decimal (e.g., 0.25 = 25%).
        de_ratio: Debt-to-equity ratio (e.g., 0.5). -1.0 = missing.
        current_ratio: Current ratio (default: 1.5). -1.0 = missing.

    Returns:
        Score from 0 to 100.
    """
    # P/E component: z-score mapped to 0-100 (z=+1 → 75, z=-1 → 25)
    pe_score = max(0, min(100, 50 + pe_zscore * 25))

    # Dividend yield: 0% → 30, 2% → 60, 4%+ → 90
    if div_yield < 0:
        dy_score = 50.0  # missing
    else:
        dy_score = max(0, min(100, 30 + div_yield * 100 * 15))

    # ROE: 0% → 20, 15% → 60, 30%+ → 90
    if roe <= -1.0:
        roe_score = 50.0  # missing
    else:
        roe_score = max(0, min(100, 20 + roe * 100 * 2.33))

    # D/E ratio: 0 → 90 (no debt), 1.0 → 50, 2.0+ → 10
    if de_ratio < 0:
        de_score = 50.0  # missing
    else:
        de_score = max(0, min(100, 90 - de_ratio * 40))

    # Current ratio: 1.5 is ideal, below 1.0 or above 3.0 is worse
    if current_ratio < 0:
        cr_score = 50.0  # missing
    else:
        deviation = abs(current_ratio - 1.5)
        cr_score = max(0, min(100, 80 - deviation * 20))

    # Weighted combination
    score = (
        pe_score * 0.30
        + dy_score * 0.15
        + roe_score * 0.25
        + de_score * 0.15
        + cr_score * 0.15
    )
    return round(max(0, min(100, score)), 2)


def classify_value_signal(
    pe_zscore: float,
    div_yield: float,
    roe: float,
    de_ratio: float,
) -> str:
    """Classify a stock's fundamental profile.

    - "value": Low P/E + decent dividend yield (undervalued income stock)
    - "growth": High P/E + high ROE + low dividend (growth reinvestor)
    - "balanced": Mix of characteristics

    Args:
        pe_zscore: Normalized P/E z-score.
        div_yield: Dividend yield as decimal.
        roe: Return on equity as decimal. -1.0 = missing.
        de_ratio: Debt-to-equity ratio. -1.0 = missing.

    Returns:
        One of: "value", "growth", "balanced"
    """
    # Value: positive PE z-score (low PE) AND decent dividend
    is_low_pe = pe_zscore > 0.2
    has_dividend = div_yield > 0.01  # >1% yield
    is_high_roe = roe > 0.15 if roe > -1.0 else False
    is_high_pe = pe_zscore < -0.2

    if is_low_pe and has_dividend:
        return "value"
    elif is_high_pe and is_high_roe and div_yield < 0.01:
        return "growth"
    return "balanced"


# ============================================================
# WF4: Portfolio Construction (Pension Fund Principles)
# ============================================================

def calculate_signal_weights(
    signals: list,
    max_position_pct: float = 0.05,
    max_sector_pct: float = 0.25,
    cash_reserve_pct: float = 0.05,
    sector_map: dict = None,
) -> dict:
    """Calculate target portfolio weights from signal results.

    Three-pass allocation algorithm (pension fund principles):
    1. Raw score-based allocation: strong_buy (3x), buy (2x), others excluded
    2. Position cap enforcement: max 5% per stock, excess redistributed
    3. Sector cap enforcement: max 25% per sector, excess redistributed

    Args:
        signals: List of dicts with keys:
            'symbol', 'signal_strength', 'combined_signal_score'
        max_position_pct: Maximum weight per stock (default: 5%).
        max_sector_pct: Maximum weight per sector (default: 25%).
        cash_reserve_pct: Target cash reserve (default: 5%).
        sector_map: Dict mapping symbol -> GICS sector. If None, no sector cap.

    Returns:
        Dict mapping symbol -> target_weight (0.0 to max_position_pct).
        Weights sum to <= (1 - cash_reserve_pct).
    """
    if not signals:
        return {}

    # Multipliers: only strong_buy and buy get allocation
    multipliers = {"strong_buy": 3.0, "buy": 2.0}

    # Pass 1: Raw score-based allocation
    raw_weights = {}
    for s in signals:
        strength = s.get("signal_strength", "hold")
        mult = multipliers.get(strength, 0.0)
        if mult > 0:
            score = max(s.get("combined_signal_score", 50.0), 1.0)
            raw_weights[s["symbol"]] = mult * score

    if not raw_weights:
        return {}

    # Normalize to sum to investable fraction
    investable = 1.0 - cash_reserve_pct
    total_raw = sum(raw_weights.values())
    weights = {sym: (w / total_raw) * investable for sym, w in raw_weights.items()}

    # Pass 2: Position cap enforcement (iterate until converged)
    for _ in range(10):  # Max 10 iterations, typically converges in 2-3
        excess = 0.0
        uncapped = {}
        for sym, w in weights.items():
            if w > max_position_pct:
                excess += w - max_position_pct
                weights[sym] = max_position_pct
            else:
                uncapped[sym] = w

        if excess <= 1e-8:
            break

        # Redistribute excess proportionally among uncapped positions
        uncapped_total = sum(uncapped.values())
        if uncapped_total <= 0:
            break
        for sym in uncapped:
            weights[sym] += excess * (uncapped[sym] / uncapped_total)

    # Pass 3: Sector cap enforcement
    if sector_map:
        for _ in range(10):
            # Calculate sector totals
            sector_weights = {}
            for sym, w in weights.items():
                sector = sector_map.get(sym, "Unknown")
                sector_weights.setdefault(sector, 0.0)
                sector_weights[sector] += w

            excess_sectors = {
                s: w - max_sector_pct
                for s, w in sector_weights.items()
                if w > max_sector_pct
            }
            if not excess_sectors:
                break

            total_excess = 0.0
            for sector, excess in excess_sectors.items():
                # Proportionally reduce positions in this sector
                sector_syms = [
                    sym for sym, w in weights.items()
                    if sector_map.get(sym, "Unknown") == sector and w > 0
                ]
                if not sector_syms:
                    continue
                sector_total = sum(weights[sym] for sym in sector_syms)
                if sector_total <= 0:
                    continue
                for sym in sector_syms:
                    reduction = excess * (weights[sym] / sector_total)
                    weights[sym] -= reduction
                    total_excess += reduction

            # Redistribute to non-overweight sectors
            other_syms = [
                sym for sym, w in weights.items()
                if sector_map.get(sym, "Unknown") not in excess_sectors
                and w > 0 and w < max_position_pct
            ]
            other_total = sum(weights[sym] for sym in other_syms)
            if other_total > 0 and total_excess > 0:
                for sym in other_syms:
                    addition = total_excess * (weights[sym] / other_total)
                    weights[sym] = min(weights[sym] + addition, max_position_pct)

    # Clean up: remove zero/tiny weights
    weights = {sym: round(w, 6) for sym, w in weights.items() if w > 1e-6}
    return weights


def estimate_transaction_cost(
    quantity: int,
    price: float,
    spread_bps: float = 5.0,
    commission_per_share: float = 0.005,
    exchange_fee_bps: float = 3.0,
    impact_bps_per_1k: float = 0.1,
) -> float:
    """Estimate total transaction cost in basis points (Brenndoerfer model).

    Components:
    - Commission: commission_per_share / price * 10000 bps
    - Exchange fee: flat bps
    - Half-spread: spread_bps / 2
    - Market impact: impact_bps_per_1k * (order_value / 1000)

    Args:
        quantity: Number of shares (absolute, positive).
        price: Estimated execution price per share.
        spread_bps: Bid-ask spread in basis points.
        commission_per_share: Per-share commission in USD (default: $0.005).
        exchange_fee_bps: Exchange fee in basis points (default: 3.0).
        impact_bps_per_1k: Market impact per $1000 order value (default: 0.1).

    Returns:
        Estimated total cost in basis points. Returns 0.0 for invalid inputs.
    """
    if quantity <= 0 or price <= 0:
        return 0.0

    # Commission component (bps)
    commission_bps = (commission_per_share / price) * 10000

    # Half-spread component (bps)
    half_spread_bps = spread_bps / 2.0

    # Market impact component (bps)
    order_value = quantity * price
    impact_bps = impact_bps_per_1k * (order_value / 1000.0)

    total_bps = commission_bps + exchange_fee_bps + half_spread_bps + impact_bps
    return round(total_bps, 4)


def calculate_cost_breakdown(
    quantity: int,
    price: float,
    spread_bps: float = 5.0,
    commission_per_share: float = 0.005,
    exchange_fee_bps: float = 3.0,
    impact_bps_per_1k: float = 0.1,
) -> dict:
    """Calculate individual transaction cost components in USD.

    Same Brenndoerfer model as estimate_transaction_cost() but returns
    absolute USD values broken into components for trade accounting.

    Args:
        quantity: Number of shares (positive).
        price: Execution price per share.
        spread_bps: Bid-ask spread in basis points.
        commission_per_share: Per-share commission USD.
        exchange_fee_bps: Exchange fee basis points.
        impact_bps_per_1k: Market impact per $1000 order value.

    Returns:
        Dict with keys: commission, spread_cost, impact_cost, total_cost
        (all in USD). Returns all zeros for invalid inputs.
    """
    if quantity <= 0 or price <= 0:
        return {
            "commission": 0.0,
            "spread_cost": 0.0,
            "impact_cost": 0.0,
            "total_cost": 0.0,
        }

    order_value = quantity * price

    # Commission: per-share * quantity
    commission = commission_per_share * quantity

    # Half-spread applied to order value
    spread_cost = order_value * (spread_bps / 2.0) / 10000.0

    # Exchange fee applied to order value
    exchange_cost = order_value * exchange_fee_bps / 10000.0

    # Market impact scales with order size
    impact_cost = order_value * (impact_bps_per_1k * (order_value / 1000.0)) / 10000.0

    total_cost = commission + spread_cost + exchange_cost + impact_cost

    return {
        "commission": round(commission, 4),
        "spread_cost": round(spread_cost, 4),
        "impact_cost": round(exchange_cost + impact_cost, 4),
        "total_cost": round(total_cost, 4),
    }


# ============================================================
# WF3: Sentiment Analysis (Phase 6)
# ============================================================

def aggregate_article_sentiments(
    articles: list,
    classifier_results: list,
) -> list:
    """Merge raw articles with classifier results into enriched article sentiments.

    Args:
        articles: List of article dicts from SentimentProvider.fetch_news().
            Required keys: headline, published_at.
        classifier_results: List of sentiment dicts from SentimentClassifier.classify().
            Required keys: positive, neutral, negative.

    Returns:
        List of dicts with merged fields:
            - headline, published_at (from articles)
            - positive, neutral, negative (from classifier)
        Length = min(len(articles), len(classifier_results)).
    """
    count = min(len(articles), len(classifier_results))
    merged = []
    for i in range(count):
        merged.append({
            "headline": articles[i].get("headline", ""),
            "published_at": articles[i].get("published_at", ""),
            "positive": classifier_results[i].get("positive", 0.0),
            "neutral": classifier_results[i].get("neutral", 0.0),
            "negative": classifier_results[i].get("negative", 0.0),
        })
    return merged


def compute_sentiment_score(
    article_sentiments: list,
    decay_half_life_days: float = 3.0,
    run_date: str = "",
) -> float:
    """Compute time-decay-weighted sentiment score (0-100) from article sentiments.

    Recent articles are weighted more heavily using exponential decay:
        weight = 2 ^ (-age_days / half_life)

    Args:
        article_sentiments: List of dicts with keys:
            - positive: float (0-1)
            - negative: float (0-1)
            - published_at: str (ISO datetime or YYYY-MM-DD)
        decay_half_life_days: Half-life for time decay (default: 3 days).
        run_date: Reference date for age calculation (YYYY-MM-DD). Empty = today.

    Returns:
        Sentiment score from 0 (most negative) to 100 (most positive).
        Returns 50.0 (neutral) if no articles.
    """
    from datetime import datetime

    if not article_sentiments:
        return 50.0

    # Parse reference date
    if run_date:
        try:
            ref_date = datetime.strptime(run_date, "%Y-%m-%d")
        except ValueError:
            ref_date = datetime.utcnow()
    else:
        ref_date = datetime.utcnow()

    weighted_scores = []
    total_weight = 0.0

    for article in article_sentiments:
        positive = article.get("positive", 0.0)
        negative = article.get("negative", 0.0)

        # Per-article sentiment: positive maps to 100, negative to 0
        # Formula: positive_prob * 100 gives 0-100 range directly
        # (neutral articles get ~50 since positive + negative ≈ 0)
        article_score = positive * 100.0

        # Calculate age for time decay
        published_at = article.get("published_at", "")
        age_days = 0.0
        if published_at:
            try:
                # Try ISO format first
                pub_date = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                pub_date = pub_date.replace(tzinfo=None)
                age_days = max((ref_date - pub_date).total_seconds() / 86400.0, 0.0)
            except (ValueError, TypeError):
                try:
                    # Try YYYY-MM-DD
                    pub_date = datetime.strptime(published_at[:10], "%Y-%m-%d")
                    age_days = max((ref_date - pub_date).days, 0.0)
                except (ValueError, TypeError):
                    age_days = 0.0

        # Exponential decay weight
        if decay_half_life_days > 0:
            weight = 2.0 ** (-age_days / decay_half_life_days)
        else:
            weight = 1.0

        weighted_scores.append(article_score * weight)
        total_weight += weight

    if total_weight == 0.0:
        return 50.0

    return round(sum(weighted_scores) / total_weight, 2)


def classify_sentiment_signal(sentiment_score: float) -> str:
    """Classify sentiment score into a trading signal.

    Args:
        sentiment_score: Score from 0-100.

    Returns:
        One of: "very_positive", "positive", "neutral",
                "negative", "very_negative"
    """
    if sentiment_score >= 70:
        return "very_positive"
    elif sentiment_score >= 55:
        return "positive"
    elif sentiment_score >= 45:
        return "neutral"
    elif sentiment_score >= 30:
        return "negative"
    return "very_negative"
