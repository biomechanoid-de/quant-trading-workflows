"""WF2: Universe & Screening - Tasks.

Weekly pipeline that loads historical prices, computes returns and metrics,
applies K-Means clustering, scores stocks via multi-factor model (Brenndoerfer),
and stores ranked results.

Sources:
- Udacity Course 2: computing_returns, calculate_rsi, compute_BM_Perf, K-Means
- Brenndoerfer: factor-investing, Z-score normalization, quintile ranking

Task DAG:
    load_historical_prices -> compute_returns_and_metrics -> +- cluster_stocks        -+
                                                             +- score_and_rank_factors -+
                                                                                        v
                                                                                  merge_results
                                                                                        |
                                                       +----------------+---------------+
                                                       v                v               v
                                              store_to_db     store_to_parquet   generate_report
"""

from typing import Dict, List

from flytekit import task, Resources

from src.shared.models import ScreeningConfig, ScreeningResult, StockMetrics


@task(
    requests=Resources(cpu="500m", mem="512Mi"),
    limits=Resources(cpu="1000m", mem="1024Mi"),
)
def load_historical_prices(
    symbols: List[str],
    lookback_days: int,
) -> Dict[str, str]:
    """Load historical close prices from PostgreSQL for all symbols.

    Queries the market_data table populated by WF1, returns a
    symbol-keyed dict where values are JSON-serialized lists of
    [date_str, close_price] pairs. JSON encoding avoids Flytekit's
    nested List[List] type limitation.

    Args:
        symbols: Stock symbols to load.
        lookback_days: Number of calendar days to look back.

    Returns:
        Dict mapping symbol -> JSON string of [[date_str, close_price], ...],
        ordered by date ascending.
    """
    import json
    from src.shared.db import get_historical_prices

    rows = get_historical_prices(symbols=symbols, lookback_days=lookback_days)

    raw: dict = {}
    for symbol, date, close in rows:
        date_str = date.isoformat() if hasattr(date, "isoformat") else str(date)
        if symbol not in raw:
            raw[symbol] = []
        raw[symbol].append([date_str, float(close)])

    # JSON-serialize each symbol's data for Flytekit compatibility
    return {symbol: json.dumps(pairs) for symbol, pairs in raw.items()}


@task(
    requests=Resources(cpu="1000m", mem="1024Mi"),
    limits=Resources(cpu="2000m", mem="2048Mi"),
)
def compute_returns_and_metrics(
    price_data: Dict[str, str],
    config: ScreeningConfig,
) -> List[StockMetrics]:
    """Compute returns, RSI, and performance metrics for each stock.

    For each symbol with sufficient data:
    1. Build price Series from [date, close] pairs
    2. Compute daily returns, forward return, momentum returns
    3. Calculate RSI with configurable window
    4. Compute performance: CAGR, Sharpe, Sortino, Calmar, Max Drawdown
    5. Compute annualized volatility

    Args:
        price_data: Dict[symbol -> JSON string of [[date, close], ...]]
            from load_historical_prices.
        config: ScreeningConfig with momentum_windows, rsi_window, etc.

    Returns:
        List of StockMetrics (one per symbol with sufficient data).
        Factor scores and cluster fields are left at defaults (populated later).
    """
    import json
    import pandas as pd
    import numpy as np
    from src.shared.analytics import (
        calculate_rsi,
        classify_rsi_signal,
        compute_cagr,
        compute_sharpe,
        compute_sortino,
        compute_max_drawdown,
        compute_calmar,
    )

    # Minimum data points needed
    min_required = max(
        config.rsi_window + 1,
        min(config.momentum_windows) + 1 if config.momentum_windows else 11,
        30,
    )

    metrics_list = []

    for symbol, price_json in price_data.items():
        date_prices = json.loads(price_json) if isinstance(price_json, str) else price_json
        if len(date_prices) < min_required:
            continue

        dates = [dp[0] for dp in date_prices]
        prices = [dp[1] for dp in date_prices]
        price_series = pd.Series(
            prices, index=pd.to_datetime(dates), name=symbol
        )

        # Daily returns
        daily_returns = price_series.pct_change().dropna()
        if daily_returns.empty:
            continue

        # Forward return (latest available)
        forward_ret = price_series.pct_change(config.forecast_horizon).shift(
            -config.forecast_horizon
        )
        latest_forward = (
            float(forward_ret.dropna().iloc[-1])
            if not forward_ret.dropna().empty
            else 0.0
        )

        # Momentum returns for each window
        momentum_rets: Dict[str, float] = {}
        for window in config.momentum_windows:
            if len(price_series) > window:
                mom = price_series.pct_change(window)
                latest_mom = (
                    float(mom.dropna().iloc[-1])
                    if not mom.dropna().empty
                    else 0.0
                )
                momentum_rets[f"{window}d"] = round(latest_mom, 6)

        # RSI
        rsi_series = calculate_rsi(daily_returns, window=config.rsi_window)
        latest_rsi = (
            float(rsi_series.dropna().iloc[-1])
            if not rsi_series.dropna().empty
            else 50.0
        )
        rsi_signal = classify_rsi_signal(
            latest_rsi, config.rsi_oversold, config.rsi_overbought
        )

        # Volatility (annualized)
        volatility = float(daily_returns.std()) * np.sqrt(252)

        # Cumulative returns for performance metrics
        cumulative = (1 + daily_returns).cumprod()

        cagr = compute_cagr(cumulative)
        sharpe = compute_sharpe(daily_returns)
        sortino = compute_sortino(daily_returns)
        max_dd = compute_max_drawdown(cumulative)
        calmar = compute_calmar(cagr, max_dd)

        metrics_list.append(
            StockMetrics(
                symbol=symbol,
                forward_return=round(float(latest_forward), 6),
                momentum_returns=momentum_rets,
                rsi=round(latest_rsi, 4),
                rsi_signal=rsi_signal,
                volatility_252d=round(volatility, 6),
                cagr=round(cagr, 6),
                sharpe=round(sharpe, 6),
                sortino=round(sortino, 6),
                calmar=round(calmar, 6),
                max_drawdown=round(max_dd, 6),
                z_scores={},
                composite_score=0.0,
                quintile=3,
            )
        )

    return metrics_list


@task(
    requests=Resources(cpu="500m", mem="512Mi"),
    limits=Resources(cpu="1000m", mem="1024Mi"),
)
def cluster_stocks(
    stock_metrics: List[StockMetrics],
    max_k: int,
) -> Dict[str, str]:
    """Group stocks using K-Means clustering by performance characteristics.

    Following Udacity Course 2 K-Means pattern:
    1. Extract features: mean momentum return, volatility
    2. StandardScaler normalization
    3. Elbow method to find optimal K (1..max_k)
    4. Fit KMeans with optimal K

    Args:
        stock_metrics: List of StockMetrics (from compute_returns_and_metrics).
        max_k: Maximum number of clusters to evaluate.

    Returns:
        Dict mapping symbol -> JSON string of [cluster_id, cluster_label].
        JSON encoding avoids Flytekit's nested List type limitation.
    """
    import json
    import numpy as np

    if len(stock_metrics) < 3:
        return {m.symbol: json.dumps([0, "single_cluster"]) for m in stock_metrics}

    # Extract features: average momentum return and volatility
    symbols = []
    features = []
    for m in stock_metrics:
        mom_values = list(m.momentum_returns.values())
        avg_momentum = float(np.mean(mom_values)) if mom_values else 0.0
        symbols.append(m.symbol)
        features.append([avg_momentum, m.volatility_252d])

    features_array = np.array(features)

    # StandardScaler normalization
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_array)

    # Elbow method: find optimal K
    from sklearn.cluster import KMeans

    effective_max_k = min(max_k, len(stock_metrics) - 1, 10)
    if effective_max_k < 2:
        return {m.symbol: json.dumps([0, "single_cluster"]) for m in stock_metrics}

    inertias = []
    for k in range(1, effective_max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(scaled_features)
        inertias.append(km.inertia_)

    optimal_k = _find_elbow(inertias)

    # Final clustering
    km_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = km_final.fit_predict(scaled_features)
    centroids = scaler.inverse_transform(km_final.cluster_centers_)

    # Generate descriptive labels from centroids
    median_vol = float(np.median(features_array[:, 1]))
    cluster_labels = {}
    for i, centroid in enumerate(centroids):
        mom_level = "HiMom" if centroid[0] > 0 else "LoMom"
        vol_level = "HiVol" if centroid[1] > median_vol else "LoVol"
        cluster_labels[i] = f"{mom_level}-{vol_level}"

    result: Dict[str, str] = {}
    for i, symbol in enumerate(symbols):
        cid = int(labels[i])
        result[symbol] = json.dumps([cid, cluster_labels[cid]])

    return result


def _find_elbow(inertias: list) -> int:
    """Find the elbow point using the distance-from-line method.

    Following Udacity Course 2 find_elbow_point pattern.
    Draws a line from first to last inertia, finds K with max
    perpendicular distance.

    Args:
        inertias: List of inertia values for K=1..N.

    Returns:
        Optimal K (1-indexed).
    """
    import numpy as np

    if len(inertias) <= 2:
        return max(1, len(inertias))

    n = len(inertias)
    coords = np.array([[i, inertias[i]] for i in range(n)])

    line_start = coords[0]
    line_end = coords[-1]
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)

    if line_len == 0:
        return 2

    distances = []
    for point in coords:
        # Perpendicular distance from point to line (2D cross product scalar)
        diff = line_start - point
        d = abs(line_vec[0] * diff[1] - line_vec[1] * diff[0]) / line_len
        distances.append(d)

    return int(np.argmax(distances)) + 1  # +1 because K is 1-indexed


@task(
    requests=Resources(cpu="500m", mem="512Mi"),
    limits=Resources(cpu="1000m", mem="1024Mi"),
)
def score_and_rank_factors(
    stock_metrics: List[StockMetrics],
    factor_weights: Dict[str, float],
) -> List[StockMetrics]:
    """Apply Brenndoerfer multi-factor scoring with Z-score normalization.

    Factors:
    - momentum: Average of all momentum returns (higher = better)
    - low_volatility: Negative volatility (lower vol = higher score)
    - rsi_signal: Numeric RSI contrarian (oversold=1.0, neutral=0, overbought=-0.5)
    - sharpe: Sharpe ratio (higher = better)

    Process:
    1. Z-score normalize each factor across all stocks
    2. Compute weighted composite score
    3. Assign quintiles (Q1 = best, Q5 = worst)

    Args:
        stock_metrics: List of StockMetrics.
        factor_weights: Dict of factor_name -> weight.

    Returns:
        Updated List of StockMetrics sorted by composite_score descending.
    """
    import pandas as pd
    import numpy as np
    from src.shared.analytics import zscore_normalize, assign_quintiles

    if not stock_metrics:
        return stock_metrics

    symbols = [m.symbol for m in stock_metrics]
    metrics_by_symbol = {m.symbol: m for m in stock_metrics}

    # Build raw factor DataFrame
    raw_factors = pd.DataFrame(index=symbols)

    # Momentum factor: average of all momentum return windows
    raw_factors["momentum"] = [
        float(np.mean(list(m.momentum_returns.values())))
        if m.momentum_returns
        else 0.0
        for m in stock_metrics
    ]

    # Low volatility factor: negate so lower vol = higher score
    raw_factors["low_volatility"] = [-m.volatility_252d for m in stock_metrics]

    # RSI signal factor: contrarian (oversold = buy signal = positive)
    rsi_signal_map = {"oversold": 1.0, "neutral": 0.0, "overbought": -0.5}
    raw_factors["rsi_signal"] = [
        rsi_signal_map.get(m.rsi_signal, 0.0) for m in stock_metrics
    ]

    # Sharpe factor
    raw_factors["sharpe"] = [m.sharpe for m in stock_metrics]

    # Z-score normalize each factor
    z_scores_df = raw_factors.apply(zscore_normalize, axis=0)

    # Compute composite score as weighted sum
    composite = pd.Series(0.0, index=symbols)
    for factor_name, weight in factor_weights.items():
        if factor_name in z_scores_df.columns:
            composite += z_scores_df[factor_name] * weight

    # Assign quintiles
    quintiles = assign_quintiles(composite)

    # Build updated StockMetrics list
    updated_metrics = []
    for symbol in symbols:
        m = metrics_by_symbol[symbol]
        z_dict = {
            col: round(float(z_scores_df.loc[symbol, col]), 6)
            for col in z_scores_df.columns
        }

        updated_metrics.append(
            StockMetrics(
                symbol=m.symbol,
                forward_return=m.forward_return,
                momentum_returns=m.momentum_returns,
                rsi=m.rsi,
                rsi_signal=m.rsi_signal,
                volatility_252d=m.volatility_252d,
                cagr=m.cagr,
                sharpe=m.sharpe,
                sortino=m.sortino,
                calmar=m.calmar,
                max_drawdown=m.max_drawdown,
                z_scores=z_dict,
                composite_score=round(float(composite[symbol]), 6),
                quintile=int(quintiles[symbol]),
                cluster_id=m.cluster_id,
                cluster_label=m.cluster_label,
            )
        )

    # Sort by composite score descending
    updated_metrics.sort(key=lambda x: x.composite_score, reverse=True)
    return updated_metrics


@task(
    requests=Resources(cpu="200m", mem="256Mi"),
    limits=Resources(cpu="500m", mem="512Mi"),
)
def merge_cluster_assignments(
    ranked_metrics: List[StockMetrics],
    cluster_assignments: Dict[str, str],
) -> List[StockMetrics]:
    """Merge cluster IDs and labels into the ranked StockMetrics.

    Runs after both cluster_stocks and score_and_rank_factors complete.

    Args:
        ranked_metrics: StockMetrics with factor scores and quintiles.
        cluster_assignments: Dict[symbol -> JSON string of [cluster_id, cluster_label]].

    Returns:
        Final List of StockMetrics with all fields populated.
    """
    import json

    merged = []
    for m in ranked_metrics:
        raw = cluster_assignments.get(m.symbol, json.dumps([-1, "unknown"]))
        assignment = json.loads(raw) if isinstance(raw, str) else raw
        cluster_id = int(assignment[0])
        cluster_label = str(assignment[1])
        merged.append(
            StockMetrics(
                symbol=m.symbol,
                forward_return=m.forward_return,
                momentum_returns=m.momentum_returns,
                rsi=m.rsi,
                rsi_signal=m.rsi_signal,
                volatility_252d=m.volatility_252d,
                cagr=m.cagr,
                sharpe=m.sharpe,
                sortino=m.sortino,
                calmar=m.calmar,
                max_drawdown=m.max_drawdown,
                z_scores=m.z_scores,
                composite_score=m.composite_score,
                quintile=m.quintile,
                cluster_id=cluster_id,
                cluster_label=cluster_label,
            )
        )
    return merged


@task(
    requests=Resources(cpu="500m", mem="512Mi"),
    limits=Resources(cpu="1000m", mem="1024Mi"),
)
def assemble_screening_result(
    run_date: str,
    config: ScreeningConfig,
    final_metrics: List[StockMetrics],
    price_data: Dict[str, str],
) -> ScreeningResult:
    """Assemble the final ScreeningResult with benchmark performance.

    Computes equal-weight benchmark from price_data and packages
    everything into a single ScreeningResult dataclass.

    Args:
        run_date: Screening date (YYYY-MM-DD). Empty string = today.
        config: ScreeningConfig used for this run.
        final_metrics: Fully populated StockMetrics list.
        price_data: Raw price data dict (JSON-encoded) for benchmark computation.

    Returns:
        Complete ScreeningResult ready for storage and reporting.
    """
    import json
    import pandas as pd
    from datetime import datetime
    from src.shared.analytics import compute_benchmark_performance

    if not run_date:
        run_date = datetime.now().strftime("%Y-%m-%d")

    # Build prices DataFrame for benchmark
    frames = {}
    for symbol, price_json in price_data.items():
        date_prices = json.loads(price_json) if isinstance(price_json, str) else price_json
        dates = [dp[0] for dp in date_prices]
        prices = [dp[1] for dp in date_prices]
        frames[symbol] = pd.Series(prices, index=pd.to_datetime(dates))

    if frames:
        prices_df = pd.DataFrame(frames).sort_index()
        bm_cagr, bm_sharpe, bm_cumulative = compute_benchmark_performance(
            prices_df
        )
    else:
        bm_cagr, bm_sharpe, bm_cumulative = 0.0, 0.0, 0.0

    # Count clusters
    cluster_ids = set(m.cluster_id for m in final_metrics if m.cluster_id >= 0)
    optimal_k = len(cluster_ids) if cluster_ids else 0

    # Data quality notes
    notes: List[str] = []
    symbols_with_data = len(final_metrics)
    symbols_requested = len(config.symbols)
    if symbols_with_data < symbols_requested:
        missing_count = symbols_requested - symbols_with_data
        notes.append(
            f"{missing_count} symbols had insufficient historical data"
        )

    return ScreeningResult(
        run_date=run_date,
        config=config,
        stock_metrics=final_metrics,
        benchmark_cagr=round(bm_cagr, 6),
        benchmark_sharpe=round(bm_sharpe, 6),
        benchmark_cumulative_return=round(bm_cumulative, 6),
        num_symbols_input=symbols_requested,
        num_symbols_with_data=symbols_with_data,
        optimal_k_clusters=optimal_k,
        data_quality_notes=notes,
    )


@task(
    requests=Resources(cpu="200m", mem="256Mi"),
    limits=Resources(cpu="500m", mem="512Mi"),
)
def store_screening_to_db(result: ScreeningResult) -> str:
    """Store screening results to PostgreSQL.

    Writes to screening_runs and screening_results tables.
    Uses UPSERT for idempotency on Flyte retries.

    Args:
        result: Complete ScreeningResult.

    Returns:
        Summary string.
    """
    from src.shared.db import store_screening_results

    run_metadata = {
        "num_symbols": result.num_symbols_with_data,
        "optimal_k": result.optimal_k_clusters,
        "benchmark_cagr": result.benchmark_cagr,
        "benchmark_sharpe": result.benchmark_sharpe,
        "benchmark_cumulative_return": result.benchmark_cumulative_return,
    }

    rows = store_screening_results(
        run_date=result.run_date,
        stock_metrics=result.stock_metrics,
        run_metadata=run_metadata,
    )
    return f"Stored {rows} screening results for {result.run_date}"


@task(
    requests=Resources(cpu="200m", mem="256Mi"),
    limits=Resources(cpu="500m", mem="512Mi"),
)
def store_screening_to_parquet(result: ScreeningResult) -> str:
    """Store screening results as Parquet to MinIO/S3.

    Hive-partitioned path:
        s3://quant-data/screening/year=YYYY/month=MM/day=DD/screening_results.parquet

    Args:
        result: Complete ScreeningResult.

    Returns:
        Summary string with S3 path.
    """
    import io
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    from src.shared.storage import get_s3_client
    from src.shared.config import S3_DATA_BUCKET

    metrics = result.stock_metrics
    if not metrics:
        return "No screening results to store"

    table = pa.table(
        {
            "symbol": pa.array(
                [m.symbol for m in metrics], type=pa.string()
            ),
            "run_date": pa.array(
                [result.run_date] * len(metrics), type=pa.string()
            ),
            "composite_score": pa.array(
                [m.composite_score for m in metrics], type=pa.float64()
            ),
            "quintile": pa.array(
                [m.quintile for m in metrics], type=pa.int32()
            ),
            "cagr": pa.array(
                [m.cagr for m in metrics], type=pa.float64()
            ),
            "sharpe": pa.array(
                [m.sharpe for m in metrics], type=pa.float64()
            ),
            "sortino": pa.array(
                [m.sortino for m in metrics], type=pa.float64()
            ),
            "calmar": pa.array(
                [m.calmar for m in metrics], type=pa.float64()
            ),
            "max_drawdown": pa.array(
                [m.max_drawdown for m in metrics], type=pa.float64()
            ),
            "rsi": pa.array(
                [m.rsi for m in metrics], type=pa.float64()
            ),
            "rsi_signal": pa.array(
                [m.rsi_signal for m in metrics], type=pa.string()
            ),
            "volatility_252d": pa.array(
                [m.volatility_252d for m in metrics], type=pa.float64()
            ),
            "forward_return": pa.array(
                [m.forward_return for m in metrics], type=pa.float64()
            ),
            "cluster_id": pa.array(
                [m.cluster_id for m in metrics], type=pa.int32()
            ),
            "cluster_label": pa.array(
                [m.cluster_label for m in metrics], type=pa.string()
            ),
            "momentum_returns_json": pa.array(
                [json.dumps(m.momentum_returns) for m in metrics],
                type=pa.string(),
            ),
            "z_scores_json": pa.array(
                [json.dumps(m.z_scores) for m in metrics], type=pa.string()
            ),
        }
    )

    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    parquet_bytes = buf.getvalue()

    # Build Hive-style path
    parts = result.run_date.split("-")
    year, month, day = parts
    s3_key = (
        f"screening/year={year}/month={month}/day={day}/"
        f"screening_results.parquet"
    )

    client = get_s3_client()
    bucket = S3_DATA_BUCKET
    client.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=parquet_bytes,
        ContentType="application/octet-stream",
    )

    s3_path = f"s3://{bucket}/{s3_key}"
    return f"Stored {len(metrics)} screening results to {s3_path}"


@task(
    requests=Resources(cpu="100m", mem="128Mi"),
    limits=Resources(cpu="200m", mem="256Mi"),
)
def generate_screening_report(result: ScreeningResult) -> str:
    """Generate a human-readable screening report.

    Reports:
    - Run metadata (date, symbols, clusters)
    - Benchmark performance (CAGR, Sharpe)
    - Top 10 stocks by composite score (Q1)
    - Bottom 10 stocks (Q5)
    - RSI signals (oversold/overbought)

    Args:
        result: Complete ScreeningResult.

    Returns:
        Formatted report string.
    """
    lines = [
        f"{'=' * 60}",
        f"  WF2 Screening Report: {result.run_date}",
        f"{'=' * 60}",
        "",
        f"Universe: {result.num_symbols_input} symbols requested, "
        f"{result.num_symbols_with_data} with sufficient data",
        f"Clusters: {result.optimal_k_clusters} (K-Means)",
        "",
        "--- Benchmark (Equal-Weight Universe) ---",
        f"  CAGR:              {result.benchmark_cagr:.4%}",
        f"  Sharpe:            {result.benchmark_sharpe:.4f}",
        f"  Cumulative Return: {result.benchmark_cumulative_return:.4%}",
        "",
    ]

    # Top 10
    lines.append("--- Top 10 (Quintile 1) ---")
    lines.append(
        f"{'Symbol':<8} {'Score':>8} {'CAGR':>8} {'Sharpe':>8} "
        f"{'RSI':>6} {'Cluster':<15}"
    )
    for m in result.stock_metrics[:10]:
        lines.append(
            f"{m.symbol:<8} {m.composite_score:>8.4f} {m.cagr:>8.4%} "
            f"{m.sharpe:>8.4f} {m.rsi:>6.1f} {m.cluster_label:<15}"
        )

    lines.append("")

    # Bottom 10
    if len(result.stock_metrics) > 10:
        lines.append("--- Bottom 10 (Quintile 5) ---")
        lines.append(
            f"{'Symbol':<8} {'Score':>8} {'CAGR':>8} {'Sharpe':>8} "
            f"{'RSI':>6} {'Cluster':<15}"
        )
        for m in result.stock_metrics[-10:]:
            lines.append(
                f"{m.symbol:<8} {m.composite_score:>8.4f} {m.cagr:>8.4%} "
                f"{m.sharpe:>8.4f} {m.rsi:>6.1f} {m.cluster_label:<15}"
            )
        lines.append("")

    # RSI signal summary
    oversold = [m for m in result.stock_metrics if m.rsi_signal == "oversold"]
    overbought = [
        m for m in result.stock_metrics if m.rsi_signal == "overbought"
    ]
    lines.append("--- RSI Signals ---")
    lines.append(
        f"  Oversold ({len(oversold)}):  "
        f"{', '.join(m.symbol for m in oversold[:10])}"
    )
    lines.append(
        f"  Overbought ({len(overbought)}): "
        f"{', '.join(m.symbol for m in overbought[:10])}"
    )

    # Quality notes
    if result.data_quality_notes:
        lines.append("")
        lines.append("--- Data Quality Notes ---")
        for note in result.data_quality_notes:
            lines.append(f"  - {note}")

    lines.append(f"{'=' * 60}")
    return "\n".join(lines)
