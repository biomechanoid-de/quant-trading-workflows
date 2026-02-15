"""WF3: Signal & Analysis - Tasks.

Weekly pipeline that computes technical indicators (SMA, MACD, Bollinger Bands)
and fundamental analysis (P/E, ROE, dividend yield, D/E) to generate composite
buy/hold/sell signals for stocks ranked in WF2's top quintiles.

Schedule: Weekly after WF2 (Sunday 12:00 UTC)
Node: Any Pi 4 Worker

Task chain:
    load_screening_context
            │
      ┌─────┴──────┐
      │             │
  compute_          fetch_
  technical_        fundamental_     ← PARALLEL
  signals           data
      │             │
      └─────┬───────┘
            │
      combine_signals
            │
      assemble_signal_result
            │
      ┌─────┼──────────┐
      │     │          │
  store_db  store_parq  report       ← PARALLEL

Important Flytekit constraints:
- Dict[str, str] for complex inter-task data (JSON serialization)
- List[Dataclass] is OK if the dataclass has only primitive fields
- Lazy imports inside task functions
- Max limits=Resources(cpu="1000m") per platform constraint
"""

import json
from typing import Dict, List

from flytekit import task, Resources

from src.shared.models import TechnicalSignals, FundamentalSignals, SentimentSignals, SignalResult


# ============================================================
# Helper functions (not Flyte tasks — called inside tasks)
# ============================================================

def _compute_technical_score(
    sma_signal: str,
    macd_signal: str,
    bb_signal: str,
) -> float:
    """Compute weighted technical score (0-100).

    Weights: SMA 40%, MACD 35%, Bollinger 25%
    Each signal maps to a sub-score: bullish/oversold → 80, neutral → 50, bearish/overbought → 20.

    Args:
        sma_signal: SMA crossover signal ("bullish", "bearish", "neutral").
        macd_signal: MACD signal ("bullish", "bearish", "neutral").
        bb_signal: Bollinger Band signal ("oversold", "overbought", "neutral").

    Returns:
        Technical score from 0 to 100.
    """
    signal_scores = {
        "bullish": 80.0,
        "oversold": 80.0,   # Oversold = potential buy (like bullish)
        "neutral": 50.0,
        "bearish": 20.0,
        "overbought": 20.0,  # Overbought = potential sell (like bearish)
    }
    sma_score = signal_scores.get(sma_signal, 50.0)
    macd_score = signal_scores.get(macd_signal, 50.0)
    bb_score = signal_scores.get(bb_signal, 50.0)

    return round(sma_score * 0.40 + macd_score * 0.35 + bb_score * 0.25, 2)


def _classify_signal_strength(score: float) -> str:
    """Classify combined signal score into signal strength.

    Args:
        score: Combined signal score (0-100).

    Returns:
        One of: "strong_buy", "buy", "hold", "sell", "strong_sell"
    """
    if score >= 75:
        return "strong_buy"
    elif score >= 60:
        return "buy"
    elif score >= 40:
        return "hold"
    elif score >= 25:
        return "sell"
    return "strong_sell"


# ============================================================
# Flyte Tasks
# ============================================================

@task(
    requests=Resources(cpu="100m", mem="128Mi"),
    limits=Resources(cpu="200m", mem="256Mi"),
)
def resolve_run_date(
    run_date: str,
) -> str:
    """Resolve empty run_date to the latest WF2 screening run date.

    PostgreSQL DATE columns cannot accept empty strings, so this task
    ensures all downstream tasks receive a valid YYYY-MM-DD string.

    Args:
        run_date: Target date (YYYY-MM-DD). Empty = resolve to latest.

    Returns:
        Resolved date string in YYYY-MM-DD format.
    """
    if run_date:
        return run_date

    from src.shared.db import get_connection

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT MAX(run_date) FROM screening_runs")
        result = cursor.fetchone()
        if result and result[0]:
            return str(result[0])
        # Fallback: use today's date
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d")
    finally:
        cursor.close()
        conn.close()


@task(
    requests=Resources(cpu="200m", mem="256Mi"),
    limits=Resources(cpu="500m", mem="512Mi"),
)
def load_screening_context(
    run_date: str,
    max_quintile: int,
) -> Dict[str, str]:
    """Load top-ranked stocks from WF2 screening results.

    Reads the latest WF2 screening_results for stocks in quintiles 1-2
    (top 40%) and returns them as a Dict[str, str] for Flytekit safety.

    Each value is a JSON string: '{"composite_score": 1.23, "quintile": 1}'

    Args:
        run_date: Target date (YYYY-MM-DD). Must be resolved (not empty).
        max_quintile: Maximum quintile to include (default: 2).

    Returns:
        Dict mapping symbol → JSON string with WF2 context.
    """
    from src.shared.db import get_screening_top_quintiles

    rows = get_screening_top_quintiles(run_date=run_date, max_quintile=max_quintile)

    result = {}
    for symbol, composite_score, quintile in rows:
        result[symbol] = json.dumps({
            "composite_score": float(composite_score),
            "quintile": int(quintile),
        })

    return result


@task(
    requests=Resources(cpu="500m", mem="512Mi"),
    limits=Resources(cpu="1000m", mem="1024Mi"),
)
def compute_technical_signals(
    screening_context: Dict[str, str],
    lookback_days: int,
    sma_short: int,
    sma_long: int,
) -> List[TechnicalSignals]:
    """Compute technical indicators for all screened stocks.

    Fetches historical close prices from PostgreSQL, then computes:
    - SMA crossover (50/200 day)
    - MACD (12/26/9)
    - Bollinger Bands (20 day, 2 std)

    All indicators are close-based (no OHLC needed).
    ATR deferred until WF1 stores OHLC data.

    Args:
        screening_context: Dict[symbol → JSON] from load_screening_context.
        lookback_days: Number of days of price history to fetch.
        sma_short: Short SMA window (default: 50).
        sma_long: Long SMA window (default: 200).

    Returns:
        List of TechnicalSignals (one per symbol with sufficient data).
    """
    from src.shared.db import get_price_history_for_technicals
    from src.shared.analytics import (
        calculate_sma,
        calculate_sma_crossover_signal,
        calculate_macd,
        classify_macd_signal,
        calculate_bollinger_bands,
        classify_bollinger_signal,
    )

    symbols = list(screening_context.keys())
    if not symbols:
        return []

    # Fetch price history from DB
    price_history = get_price_history_for_technicals(symbols, days=lookback_days)

    results = []
    for symbol in symbols:
        history = price_history.get(symbol, [])
        if len(history) < 30:
            # Not enough data for any meaningful indicator
            results.append(TechnicalSignals(
                symbol=symbol,
                sma_50=0.0, sma_200=0.0, sma_crossover_signal="neutral",
                macd_line=0.0, macd_signal_line=0.0, macd_histogram=0.0, macd_signal="neutral",
                bb_upper=0.0, bb_middle=0.0, bb_lower=0.0, bb_signal="neutral",
                technical_score=50.0,
            ))
            continue

        # Extract close prices (chronological order)
        close_prices = [close for _, close in history]

        # SMA crossover
        sma_short_vals = calculate_sma(close_prices, sma_short)
        sma_long_vals = calculate_sma(close_prices, sma_long)
        sma_50_val = sma_short_vals[-1] if sma_short_vals and not _is_nan(sma_short_vals[-1]) else 0.0
        sma_200_val = sma_long_vals[-1] if sma_long_vals and not _is_nan(sma_long_vals[-1]) else 0.0
        sma_signal = calculate_sma_crossover_signal(close_prices, sma_short, sma_long)

        # MACD
        macd_line, macd_sig_line, macd_hist = calculate_macd(close_prices)
        macd_sig = classify_macd_signal(macd_line, macd_sig_line, macd_hist)

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close_prices)
        bb_sig = classify_bollinger_signal(close_prices[-1], bb_upper, bb_lower, bb_middle)

        # Composite technical score
        tech_score = _compute_technical_score(sma_signal, macd_sig, bb_sig)

        results.append(TechnicalSignals(
            symbol=symbol,
            sma_50=round(sma_50_val, 2),
            sma_200=round(sma_200_val, 2),
            sma_crossover_signal=sma_signal,
            macd_line=macd_line,
            macd_signal_line=macd_sig_line,
            macd_histogram=macd_hist,
            macd_signal=macd_sig,
            bb_upper=bb_upper,
            bb_middle=bb_middle,
            bb_lower=bb_lower,
            bb_signal=bb_sig,
            technical_score=tech_score,
        ))

    return results


def _is_nan(value: float) -> bool:
    """Check if a float is NaN (helper to avoid importing math in tasks)."""
    return value != value


@task(
    requests=Resources(cpu="500m", mem="512Mi"),
    limits=Resources(cpu="1000m", mem="1024Mi"),
)
def fetch_fundamental_data(
    screening_context: Dict[str, str],
) -> List[FundamentalSignals]:
    """Fetch fundamental data for all screened stocks via yfinance.

    Calls yfinance ticker.info for each symbol to get P/E ratio,
    dividend yield, ROE, D/E ratio, and current ratio.

    Rate limited: 0.2s sleep between requests (5 req/s) to avoid
    Yahoo Finance throttling on the Pi cluster.

    Args:
        screening_context: Dict[symbol → JSON] from load_screening_context.

    Returns:
        List of FundamentalSignals (one per symbol).
    """
    import time
    from src.shared.providers.yfinance_provider import YFinanceProvider
    from src.shared.analytics import (
        normalize_pe_ratio,
        compute_fundamental_score,
        classify_value_signal,
    )

    symbols = list(screening_context.keys())
    if not symbols:
        return []

    provider = YFinanceProvider()
    results = []

    for i, symbol in enumerate(symbols):
        # Rate limiting: 0.2s between requests
        if i > 0:
            time.sleep(0.2)

        fund_data = provider.fetch_fundamentals(symbol)

        pe = fund_data.get("pe_ratio", -1.0)
        forward_pe = fund_data.get("forward_pe", -1.0)
        ptb = fund_data.get("price_to_book", -1.0)
        div_yield = fund_data.get("dividend_yield", 0.0)
        roe = fund_data.get("return_on_equity", -1.0)
        de = fund_data.get("debt_to_equity", -1.0)
        cr = fund_data.get("current_ratio", -1.0)

        # Data completeness flags
        has_pe = pe > 0
        has_roe = roe > -1.0
        has_debt = de > -1.0

        # Normalize P/E and compute scores
        pe_z = normalize_pe_ratio(pe)
        fund_score = compute_fundamental_score(pe_z, div_yield, roe, de, cr)
        fund_signal = classify_value_signal(pe_z, div_yield, roe, de)

        results.append(FundamentalSignals(
            symbol=symbol,
            pe_ratio=round(pe, 4),
            pe_zscore=round(pe_z, 4),
            forward_pe=round(forward_pe, 4),
            price_to_book=round(ptb, 4),
            dividend_yield=round(div_yield, 6),
            return_on_equity=round(roe, 6),
            debt_to_equity=round(de, 4),
            current_ratio=round(cr, 4),
            has_pe=has_pe,
            has_roe=has_roe,
            has_debt=has_debt,
            fundamental_score=fund_score,
            fundamental_signal=fund_signal,
        ))

    return results


@task(
    requests=Resources(cpu="500m", mem="768Mi"),
    limits=Resources(cpu="1000m", mem="1024Mi"),
)
def fetch_sentiment_data(
    screening_context: Dict[str, str],
    run_date: str,
    news_days: int = 7,
    decay_half_life: float = 3.0,
) -> List[SentimentSignals]:
    """Fetch news articles and classify sentiment for all screened stocks.

    For each symbol:
    1. Fetch news from Finnhub (primary); fall back to Marketaux if empty/error.
    2. Classify headlines using ONNX DistilBERT-Finance on CPU.
    3. Aggregate with time-decay weighting into a single sentiment score (0-100).

    Rate limited: 0.5s between API calls (within Finnhub 60/min free tier).
    Graceful: zero articles -> neutral score (50.0) + has_sentiment=False.

    Args:
        screening_context: Dict[symbol -> JSON] from load_screening_context.
        run_date: Signal analysis run date (YYYY-MM-DD).
        news_days: Days of news history to fetch per symbol (default: 7).
        decay_half_life: Time-decay half-life in days (default: 3.0).

    Returns:
        List of SentimentSignals (one per symbol, primitive fields only).
    """
    import time
    from datetime import datetime, timedelta

    from src.shared.config import (
        FINNHUB_API_KEY, MARKETAUX_API_KEY, SENTIMENT_MODEL_DIR,
    )
    from src.shared.providers.finnhub_provider import FinnhubSentimentProvider
    from src.shared.providers.marketaux_provider import MarketauxSentimentProvider
    from src.shared.inference.onnx_cpu import OnnxCpuClassifier
    from src.shared.analytics import (
        compute_sentiment_score,
        classify_sentiment_signal,
        aggregate_article_sentiments,
    )

    symbols = list(screening_context.keys())
    if not symbols:
        return []

    # Initialize providers (only if API keys are set)
    finnhub = FinnhubSentimentProvider(api_key=FINNHUB_API_KEY) if FINNHUB_API_KEY else None
    marketaux = MarketauxSentimentProvider(api_key=MARKETAUX_API_KEY) if MARKETAUX_API_KEY else None

    # Initialize classifier
    classifier = OnnxCpuClassifier(model_dir=SENTIMENT_MODEL_DIR)

    # Date range for news lookup
    to_date = run_date
    from_date = (datetime.strptime(run_date, "%Y-%m-%d") - timedelta(days=news_days)).strftime("%Y-%m-%d")

    results = []
    for i, symbol in enumerate(symbols):
        # Rate limiting: 0.5s between API calls
        if i > 0:
            time.sleep(0.5)

        articles = []
        provider_used = "none"

        # Try Finnhub first
        if finnhub:
            try:
                articles = finnhub.fetch_news(symbol, from_date, to_date)
                if articles:
                    provider_used = "finnhub"
            except Exception:
                articles = []

        # Fallback to Marketaux if Finnhub returned nothing
        if not articles and marketaux:
            try:
                articles = marketaux.fetch_news(symbol, from_date, to_date)
                if articles:
                    provider_used = "marketaux"
            except Exception:
                articles = []

        # Classify headlines
        if articles:
            headlines = [a.get("headline", "") or a.get("summary", "") for a in articles]
            headlines = [h for h in headlines if h.strip()]

            if headlines:
                sentiments = classifier.classify(headlines)
                enriched = aggregate_article_sentiments(articles[:len(sentiments)], sentiments)
                score = compute_sentiment_score(enriched, decay_half_life, run_date)
                signal = classify_sentiment_signal(score)
                n_pos = sum(1 for s in sentiments if s.get("positive", 0) > 0.5)
                n_neg = sum(1 for s in sentiments if s.get("negative", 0) > 0.5)
                n_neu = len(sentiments) - n_pos - n_neg

                results.append(SentimentSignals(
                    symbol=symbol,
                    num_articles=len(sentiments),
                    num_positive=n_pos,
                    num_neutral=n_neu,
                    num_negative=n_neg,
                    news_provider=provider_used,
                    sentiment_score=round(score, 2),
                    sentiment_signal=signal,
                    has_sentiment=True,
                ))
                continue

        # No articles -> neutral default
        results.append(SentimentSignals(
            symbol=symbol,
            num_articles=0,
            num_positive=0,
            num_neutral=0,
            num_negative=0,
            news_provider=provider_used,
            sentiment_score=50.0,
            sentiment_signal="neutral",
            has_sentiment=False,
        ))

    return results


@task(
    requests=Resources(cpu="300m", mem="256Mi"),
    limits=Resources(cpu="500m", mem="512Mi"),
)
def combine_signals(
    screening_context: Dict[str, str],
    tech_signals: List[TechnicalSignals],
    fund_signals: List[FundamentalSignals],
    sent_signals: List[SentimentSignals],
    tech_weight: float,
    fund_weight: float,
    sent_weight: float,
    run_date: str,
) -> List[SignalResult]:
    """Combine technical, fundamental, and sentiment signals into composite scores.

    Phase 6 weights: tech=0.30, fund=0.40, sent=0.30
    When sent_weight=0.0, behaves identically to Phase 2 (backward compatible).

    Args:
        screening_context: Dict[symbol -> JSON] with WF2 context.
        tech_signals: List of TechnicalSignals from compute_technical_signals.
        fund_signals: List of FundamentalSignals from fetch_fundamental_data.
        sent_signals: List of SentimentSignals from fetch_sentiment_data.
        tech_weight: Weight for technical score (0-1, default: 0.30).
        fund_weight: Weight for fundamental score (0-1, default: 0.40).
        sent_weight: Weight for sentiment score (0-1, default: 0.30).
        run_date: Signal analysis run date.

    Returns:
        List of SignalResult sorted by combined_signal_score descending.
    """
    # Build lookup dicts
    tech_by_symbol = {ts.symbol: ts for ts in tech_signals}
    fund_by_symbol = {fs.symbol: fs for fs in fund_signals}
    sent_by_symbol = {ss.symbol: ss for ss in sent_signals}

    results = []
    for symbol, context_json in screening_context.items():
        ctx = json.loads(context_json)
        wf2_score = ctx.get("composite_score", 0.0)
        wf2_quintile = ctx.get("quintile", 3)

        tech = tech_by_symbol.get(symbol)
        fund = fund_by_symbol.get(symbol)
        sent = sent_by_symbol.get(symbol)

        # Technical score (default to 50 = neutral if missing)
        t_score = tech.technical_score if tech else 50.0
        t_signal = tech.sma_crossover_signal if tech else "neutral"

        # Fundamental score (default to 50 = neutral if missing)
        f_score = fund.fundamental_score if fund else 50.0
        f_signal = fund.fundamental_signal if fund else "balanced"

        # Sentiment score (default to 50 = neutral if missing)
        s_score = sent.sentiment_score if sent else 50.0
        s_signal = sent.sentiment_signal if sent else "neutral"

        # Combined score (3-way weighted average)
        combined = round(
            t_score * tech_weight + f_score * fund_weight + s_score * sent_weight,
            2,
        )
        strength = _classify_signal_strength(combined)

        # Data quality assessment
        has_tech = tech is not None and tech.technical_score != 50.0
        has_fund = fund is not None
        has_sent = sent is not None and sent.has_sentiment
        if has_tech and has_fund and (has_sent or sent_weight == 0.0):
            if fund.has_pe and fund.has_roe:
                quality = "complete"
            else:
                quality = "partial"
        elif has_tech or has_fund or has_sent:
            quality = "partial"
        else:
            quality = "minimal"

        results.append(SignalResult(
            symbol=symbol,
            run_date=run_date,
            wf2_composite_score=wf2_score,
            wf2_quintile=wf2_quintile,
            technical_score=t_score,
            technical_signal=t_signal,
            fundamental_score=f_score,
            fundamental_signal=f_signal,
            sentiment_score=s_score,
            sentiment_signal=s_signal,
            num_articles=sent.num_articles if sent else 0,
            news_provider=sent.news_provider if sent else "none",
            combined_signal_score=combined,
            signal_strength=strength,
            data_quality=quality,
        ))

    # Sort by combined score descending
    results.sort(key=lambda sr: sr.combined_signal_score, reverse=True)
    return results


@task(
    requests=Resources(cpu="200m", mem="256Mi"),
    limits=Resources(cpu="500m", mem="512Mi"),
)
def assemble_signal_result(
    run_date: str,
    signal_results: List[SignalResult],
    tech_weight: float = 0.3,
    fund_weight: float = 0.4,
    sent_weight: float = 0.3,
) -> Dict[str, str]:
    """Assemble signal results into a serialized SignalAnalysisResult.

    Packages all per-stock signal results with run metadata.
    Returns Dict[str, str] for Flytekit safety (no complex nested types).

    Keys:
    - "run_date": YYYY-MM-DD
    - "num_symbols_analyzed": count
    - "num_with_complete_data": count
    - "num_with_partial_data": count
    - "top_buy_signals": comma-separated symbols
    - "top_sell_signals": comma-separated symbols
    - "tech_weight", "fund_weight", "sent_weight": signal weights
    - "signal_results_json": JSON array of all SignalResult dicts

    Args:
        run_date: Signal analysis run date.
        signal_results: List of SignalResult from combine_signals.
        tech_weight: Technical signal weight used.
        fund_weight: Fundamental signal weight used.
        sent_weight: Sentiment signal weight used.

    Returns:
        Dict[str, str] with serialized analysis result.
    """
    import dataclasses

    num_complete = sum(1 for sr in signal_results if sr.data_quality == "complete")
    num_partial = sum(1 for sr in signal_results if sr.data_quality == "partial")

    # Top buy signals (strong_buy and buy, sorted by score)
    buys = [sr.symbol for sr in signal_results
            if sr.signal_strength in ("strong_buy", "buy")]
    sells = [sr.symbol for sr in signal_results
             if sr.signal_strength in ("strong_sell", "sell")]

    # Serialize signal results
    sr_dicts = [dataclasses.asdict(sr) for sr in signal_results]

    return {
        "run_date": run_date,
        "num_symbols_analyzed": str(len(signal_results)),
        "num_with_complete_data": str(num_complete),
        "num_with_partial_data": str(num_partial),
        "top_buy_signals": ",".join(buys[:5]),
        "top_sell_signals": ",".join(sells[:5]),
        "tech_weight": str(tech_weight),
        "fund_weight": str(fund_weight),
        "sent_weight": str(sent_weight),
        "signal_results_json": json.dumps(sr_dicts),
    }


@task(
    requests=Resources(cpu="200m", mem="256Mi"),
    limits=Resources(cpu="500m", mem="512Mi"),
)
def store_signals_to_db(
    assembled_result: Dict[str, str],
) -> str:
    """Store signal analysis results to PostgreSQL.

    Writes to signal_runs (metadata) and signal_results (per-symbol).
    Uses UPSERT for idempotency on Flyte retries.

    Args:
        assembled_result: Serialized SignalAnalysisResult dict.

    Returns:
        Summary string.
    """
    from src.shared.db import store_signal_results
    from src.shared.models import SignalResult

    run_date = assembled_result["run_date"]
    sr_dicts = json.loads(assembled_result["signal_results_json"])

    # Reconstruct SignalResult objects
    signal_results = [SignalResult(**d) for d in sr_dicts]

    run_metadata = {
        "num_symbols_analyzed": int(assembled_result["num_symbols_analyzed"]),
        "num_with_complete_data": int(assembled_result["num_with_complete_data"]),
        "num_with_partial_data": int(assembled_result["num_with_partial_data"]),
        "tech_weight": float(assembled_result.get("tech_weight", "0.3")),
        "fund_weight": float(assembled_result.get("fund_weight", "0.4")),
        "sent_weight": float(assembled_result.get("sent_weight", "0.3")),
    }

    rows = store_signal_results(run_date, signal_results, run_metadata)
    return f"Stored {rows} signal results for {run_date}"


@task(
    requests=Resources(cpu="200m", mem="256Mi"),
    limits=Resources(cpu="500m", mem="512Mi"),
)
def store_signals_to_parquet(
    assembled_result: Dict[str, str],
) -> str:
    """Store signal analysis results as Parquet to MinIO/S3.

    Writes to Hive-style partitioned path:
        s3://quant-data/signals/year=YYYY/month=MM/day=DD/signals.parquet

    Args:
        assembled_result: Serialized SignalAnalysisResult dict.

    Returns:
        Summary string with S3 path.
    """
    import io
    from src.shared.config import S3_DATA_BUCKET
    from src.shared.storage import get_s3_client

    run_date = assembled_result["run_date"]
    sr_dicts = json.loads(assembled_result["signal_results_json"])

    if not sr_dicts:
        return f"No signal results to store for {run_date}"

    import pyarrow as pa
    import pyarrow.parquet as pq

    # Build Parquet table
    table = pa.table({
        "symbol": pa.array([d["symbol"] for d in sr_dicts], type=pa.string()),
        "run_date": pa.array([d["run_date"] for d in sr_dicts], type=pa.string()),
        "wf2_composite_score": pa.array([d["wf2_composite_score"] for d in sr_dicts], type=pa.float64()),
        "wf2_quintile": pa.array([d["wf2_quintile"] for d in sr_dicts], type=pa.int32()),
        "technical_score": pa.array([d["technical_score"] for d in sr_dicts], type=pa.float64()),
        "technical_signal": pa.array([d["technical_signal"] for d in sr_dicts], type=pa.string()),
        "fundamental_score": pa.array([d["fundamental_score"] for d in sr_dicts], type=pa.float64()),
        "fundamental_signal": pa.array([d["fundamental_signal"] for d in sr_dicts], type=pa.string()),
        "sentiment_score": pa.array([d.get("sentiment_score", 50.0) for d in sr_dicts], type=pa.float64()),
        "sentiment_signal": pa.array([d.get("sentiment_signal", "neutral") for d in sr_dicts], type=pa.string()),
        "num_articles": pa.array([d.get("num_articles", 0) for d in sr_dicts], type=pa.int32()),
        "news_provider": pa.array([d.get("news_provider", "none") for d in sr_dicts], type=pa.string()),
        "combined_signal_score": pa.array([d["combined_signal_score"] for d in sr_dicts], type=pa.float64()),
        "signal_strength": pa.array([d["signal_strength"] for d in sr_dicts], type=pa.string()),
        "data_quality": pa.array([d["data_quality"] for d in sr_dicts], type=pa.string()),
    })

    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    parquet_bytes = buf.getvalue()

    # Build S3 key
    parts = run_date.split("-")
    s3_key = f"signals/year={parts[0]}/month={parts[1]}/day={parts[2]}/signals.parquet"

    client = get_s3_client()
    client.put_object(
        Bucket=S3_DATA_BUCKET,
        Key=s3_key,
        Body=parquet_bytes,
        ContentType="application/octet-stream",
    )

    s3_path = f"s3://{S3_DATA_BUCKET}/{s3_key}"
    return f"Stored {len(sr_dicts)} signals to {s3_path} ({len(parquet_bytes)} bytes)"


@task(
    requests=Resources(cpu="100m", mem="128Mi"),
    limits=Resources(cpu="200m", mem="256Mi"),
)
def generate_signal_report(
    assembled_result: Dict[str, str],
) -> str:
    """Generate a human-readable signal analysis report.

    Args:
        assembled_result: Serialized SignalAnalysisResult dict.

    Returns:
        Formatted text report.
    """
    run_date = assembled_result["run_date"]
    num_analyzed = assembled_result["num_symbols_analyzed"]
    num_complete = assembled_result["num_with_complete_data"]
    num_partial = assembled_result["num_with_partial_data"]
    top_buys = assembled_result["top_buy_signals"]
    top_sells = assembled_result["top_sell_signals"]
    tech_w = assembled_result.get("tech_weight", "0.3")
    fund_w = assembled_result.get("fund_weight", "0.4")
    sent_w = assembled_result.get("sent_weight", "0.3")
    sr_dicts = json.loads(assembled_result["signal_results_json"])

    lines = [
        f"{'=' * 65}",
        f"  WF3 Signal Analysis Report — {run_date}",
        f"{'=' * 65}",
        f"",
        f"Symbols analyzed:  {num_analyzed}",
        f"Complete data:     {num_complete}",
        f"Partial data:      {num_partial}",
        f"Weights:           tech={tech_w}  fund={fund_w}  sent={sent_w}",
        f"",
    ]

    # Signal distribution
    strength_counts = {}
    for d in sr_dicts:
        s = d.get("signal_strength", "hold")
        strength_counts[s] = strength_counts.get(s, 0) + 1

    lines.append("Signal Distribution:")
    for strength in ["strong_buy", "buy", "hold", "sell", "strong_sell"]:
        count = strength_counts.get(strength, 0)
        bar = "█" * count
        lines.append(f"  {strength:12s} {count:2d} {bar}")
    lines.append("")

    # Top buy signals
    if top_buys:
        lines.append(f"Top Buy Signals:   {top_buys}")
    else:
        lines.append("Top Buy Signals:   (none)")

    if top_sells:
        lines.append(f"Top Sell Signals:  {top_sells}")
    else:
        lines.append("Top Sell Signals:  (none)")
    lines.append("")

    # Per-stock details (top 10)
    lines.append("Top Signals by Combined Score:")
    lines.append(f"  {'Symbol':8s} {'Tech':>5s} {'Fund':>5s} {'Sent':>5s} {'Comb':>5s} {'Strength':>12s} {'Quality':>10s}")
    lines.append(f"  {'-'*8} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*12} {'-'*10}")

    for d in sr_dicts[:10]:
        lines.append(
            f"  {d['symbol']:8s} "
            f"{d['technical_score']:5.1f} "
            f"{d['fundamental_score']:5.1f} "
            f"{d.get('sentiment_score', 50.0):5.1f} "
            f"{d['combined_signal_score']:5.1f} "
            f"{d['signal_strength']:>12s} "
            f"{d['data_quality']:>10s}"
        )

    lines.append("")
    lines.append(f"{'=' * 65}")

    return "\n".join(lines)
