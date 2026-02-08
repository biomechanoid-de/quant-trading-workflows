"""Tests for WF2: Universe & Screening.

Tests validate:
- Analytics functions (RSI, CAGR, Sharpe, Sortino, MaxDD, Calmar, Z-score, quintiles)
- Compute tasks (returns_and_metrics, cluster_stocks, score_and_rank_factors)
- Merge, assemble, and report tasks
- Elbow method for K-Means

All tests are pure computation — no database or network access.
"""

import numpy as np
import pandas as pd
import pytest

from src.shared.analytics import (
    calculate_rsi,
    classify_rsi_signal,
    compute_cagr,
    compute_sharpe,
    compute_sortino,
    compute_max_drawdown,
    compute_calmar,
    compute_benchmark_performance,
    zscore_normalize,
    assign_quintiles,
)
from src.shared.models import ScreeningConfig, StockMetrics, ScreeningResult
from src.wf2_universe_screening.tasks import (
    compute_returns_and_metrics,
    cluster_stocks,
    score_and_rank_factors,
    merge_cluster_assignments,
    assemble_screening_result,
    generate_screening_report,
    _find_elbow,
)


# ============================================================
# Analytics: RSI
# ============================================================

class TestCalculateRSI:
    """Tests for calculate_rsi()."""

    def test_rsi_range(self):
        """RSI values should be between 0 and 100."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        rsi = calculate_rsi(returns, window=14)
        valid = rsi.dropna()
        assert len(valid) > 0
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_rsi_all_gains(self):
        """All positive returns should give RSI close to 100."""
        returns = pd.Series([0.01] * 30)
        rsi = calculate_rsi(returns, window=14)
        latest = rsi.dropna().iloc[-1]
        assert latest > 90  # Should be very high (avg_loss floored at 0.001)

    def test_rsi_all_losses(self):
        """All negative returns should give RSI close to 0."""
        returns = pd.Series([-0.01] * 30)
        rsi = calculate_rsi(returns, window=14)
        latest = rsi.dropna().iloc[-1]
        assert latest < 5  # Should be very low

    def test_rsi_custom_window(self):
        """RSI with custom window should have fewer NaN values for shorter windows."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 50))
        rsi_7 = calculate_rsi(returns, window=7)
        rsi_21 = calculate_rsi(returns, window=21)
        # Shorter window produces more valid values
        assert rsi_7.dropna().count() > rsi_21.dropna().count()


class TestClassifyRSISignal:
    """Tests for classify_rsi_signal()."""

    def test_oversold(self):
        assert classify_rsi_signal(25.0) == "oversold"
        assert classify_rsi_signal(30.0) == "oversold"  # Boundary

    def test_overbought(self):
        assert classify_rsi_signal(75.0) == "overbought"
        assert classify_rsi_signal(70.0) == "overbought"  # Boundary

    def test_neutral(self):
        assert classify_rsi_signal(50.0) == "neutral"
        assert classify_rsi_signal(31.0) == "neutral"
        assert classify_rsi_signal(69.0) == "neutral"

    def test_custom_thresholds(self):
        """Custom oversold/overbought thresholds."""
        assert classify_rsi_signal(20.0, oversold=25, overbought=75) == "oversold"
        assert classify_rsi_signal(80.0, oversold=25, overbought=75) == "overbought"
        assert classify_rsi_signal(50.0, oversold=25, overbought=75) == "neutral"


# ============================================================
# Analytics: Performance Metrics
# ============================================================

class TestComputeCAGR:
    """Tests for compute_cagr()."""

    def test_positive_cagr(self):
        """Increasing cumulative returns -> positive CAGR."""
        cumulative = pd.Series([1.0, 1.05, 1.10, 1.15, 1.20])
        cagr = compute_cagr(cumulative)
        assert cagr > 0

    def test_negative_cagr(self):
        """Decreasing cumulative returns -> negative CAGR."""
        cumulative = pd.Series([1.0, 0.95, 0.90, 0.85, 0.80])
        cagr = compute_cagr(cumulative)
        assert cagr < 0

    def test_flat_cagr(self):
        """Flat cumulative returns -> CAGR close to 0."""
        cumulative = pd.Series([1.0] * 252)
        cagr = compute_cagr(cumulative)
        assert abs(cagr) < 0.001

    def test_insufficient_data(self):
        """Less than 2 data points -> 0.0."""
        assert compute_cagr(pd.Series([1.0])) == 0.0
        assert compute_cagr(pd.Series([])) == 0.0

    def test_one_year_doubling(self):
        """Doubling over 252 trading days should give ~100% CAGR."""
        cumulative = pd.Series(np.linspace(1.0, 2.0, 252))
        cagr = compute_cagr(cumulative)
        assert 0.9 < cagr < 1.1  # Approximately 100%


class TestComputeSharpe:
    """Tests for compute_sharpe()."""

    def test_positive_sharpe(self):
        """Positive mean returns -> positive Sharpe."""
        returns = pd.Series([0.01, 0.005, 0.008, 0.003, 0.012] * 50)
        sharpe = compute_sharpe(returns)
        assert sharpe > 0

    def test_negative_sharpe(self):
        """Negative mean returns -> negative Sharpe."""
        returns = pd.Series([-0.01, -0.005, -0.008, -0.003, -0.012] * 50)
        sharpe = compute_sharpe(returns)
        assert sharpe < 0

    def test_zero_std(self):
        """Zero standard deviation -> 0.0."""
        returns = pd.Series([0.01] * 50)
        sharpe = compute_sharpe(returns)
        assert sharpe == 0.0

    def test_insufficient_data(self):
        """Less than 2 data points -> 0.0."""
        assert compute_sharpe(pd.Series([0.01])) == 0.0


class TestComputeSortino:
    """Tests for compute_sortino()."""

    def test_positive_sortino(self):
        """Positive mean with some downside -> positive Sortino."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.002, 0.01, 100))
        sortino = compute_sortino(returns)
        assert sortino > 0

    def test_all_positive_returns(self):
        """No downside returns -> 0.0 (no downside std)."""
        returns = pd.Series([0.01, 0.02, 0.015, 0.005, 0.01])
        sortino = compute_sortino(returns)
        assert sortino == 0.0

    def test_insufficient_data(self):
        assert compute_sortino(pd.Series([0.01])) == 0.0


class TestComputeMaxDrawdown:
    """Tests for compute_max_drawdown()."""

    def test_max_drawdown_is_negative(self):
        """Max drawdown should be <= 0."""
        cumulative = pd.Series([1.0, 1.1, 1.05, 0.90, 0.95, 1.0])
        mdd = compute_max_drawdown(cumulative)
        assert mdd <= 0

    def test_known_drawdown(self):
        """Known drawdown: peak=1.1, trough=0.90 -> dd = (0.90-1.1)/1.1 ~ -0.1818."""
        cumulative = pd.Series([1.0, 1.1, 0.90, 1.0])
        mdd = compute_max_drawdown(cumulative)
        assert abs(mdd - (-0.1818)) < 0.01

    def test_no_drawdown(self):
        """Monotonically increasing -> 0.0 drawdown."""
        cumulative = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4])
        mdd = compute_max_drawdown(cumulative)
        assert mdd == 0.0

    def test_insufficient_data(self):
        assert compute_max_drawdown(pd.Series([1.0])) == 0.0


class TestComputeCalmar:
    """Tests for compute_calmar()."""

    def test_positive_calmar(self):
        """Positive CAGR with drawdown -> positive Calmar."""
        calmar = compute_calmar(0.15, -0.10)
        assert calmar == pytest.approx(1.5)

    def test_zero_drawdown(self):
        """Zero drawdown -> 0.0 (avoid division by zero)."""
        assert compute_calmar(0.15, 0.0) == 0.0

    def test_negative_cagr(self):
        """Negative CAGR -> negative Calmar."""
        calmar = compute_calmar(-0.05, -0.10)
        assert calmar < 0


# ============================================================
# Analytics: Z-score and Quintiles
# ============================================================

class TestZscoreNormalize:
    """Tests for zscore_normalize()."""

    def test_mean_zero(self):
        """Z-scores should have mean ~0."""
        s = pd.Series([1, 2, 3, 4, 5])
        z = zscore_normalize(s)
        assert abs(z.mean()) < 0.001

    def test_std_one(self):
        """Z-scores should have std ~1."""
        s = pd.Series([1, 2, 3, 4, 5])
        z = zscore_normalize(s)
        assert abs(z.std() - 1.0) < 0.1

    def test_zero_variance(self):
        """All same values -> all zeros."""
        s = pd.Series([5.0, 5.0, 5.0])
        z = zscore_normalize(s)
        assert (z == 0).all()


class TestAssignQuintiles:
    """Tests for assign_quintiles()."""

    def test_quintile_range(self):
        """Quintiles should be 1-5."""
        scores = pd.Series(range(20))
        q = assign_quintiles(scores)
        assert q.min() >= 1
        assert q.max() <= 5

    def test_best_gets_q1(self):
        """Highest score should be quintile 1."""
        scores = pd.Series([10, 20, 30, 40, 50], index=["A", "B", "C", "D", "E"])
        q = assign_quintiles(scores)
        assert q["E"] == 1  # Highest score = best quintile

    def test_worst_gets_q5(self):
        """Lowest score should be quintile 5."""
        scores = pd.Series([10, 20, 30, 40, 50], index=["A", "B", "C", "D", "E"])
        q = assign_quintiles(scores)
        assert q["A"] == 5  # Lowest score = worst quintile

    def test_few_stocks(self):
        """Fewer than 5 stocks -> still assigns 1-5 range."""
        scores = pd.Series([10, 20, 30], index=["A", "B", "C"])
        q = assign_quintiles(scores)
        assert q.min() >= 1
        assert q.max() <= 5


# ============================================================
# Analytics: Benchmark Performance
# ============================================================

class TestComputeBenchmarkPerformance:
    """Tests for compute_benchmark_performance()."""

    def test_returns_tuple(self):
        """Should return (cagr, sharpe, cumulative)."""
        prices = pd.DataFrame(
            {"A": [100, 101, 102, 103, 104], "B": [50, 51, 52, 53, 54]},
            index=pd.date_range("2026-01-01", periods=5),
        )
        result = compute_benchmark_performance(prices)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_positive_returns(self):
        """Increasing prices -> positive CAGR."""
        prices = pd.DataFrame(
            {"A": list(range(100, 150)), "B": list(range(50, 100))},
            index=pd.date_range("2026-01-01", periods=50),
        )
        cagr, sharpe, cumulative = compute_benchmark_performance(prices)
        assert cagr > 0
        assert cumulative > 0

    def test_empty_dataframe(self):
        """Empty DataFrame -> all zeros."""
        prices = pd.DataFrame()
        cagr, sharpe, cumulative = compute_benchmark_performance(prices)
        assert cagr == 0.0
        assert sharpe == 0.0
        assert cumulative == 0.0


# ============================================================
# Task: compute_returns_and_metrics
# ============================================================

class TestComputeReturnsAndMetrics:
    """Tests for the compute_returns_and_metrics task."""

    def test_produces_metrics(self, sample_price_data, sample_screening_config):
        """Should produce StockMetrics for each symbol with sufficient data."""
        metrics = compute_returns_and_metrics(
            price_data=sample_price_data,
            config=sample_screening_config,
        )
        assert len(metrics) > 0
        assert all(isinstance(m, StockMetrics) for m in metrics)

    def test_has_rsi(self, sample_price_data, sample_screening_config):
        """Each stock should have RSI between 0 and 100."""
        metrics = compute_returns_and_metrics(
            price_data=sample_price_data,
            config=sample_screening_config,
        )
        for m in metrics:
            assert 0 <= m.rsi <= 100
            assert m.rsi_signal in ("oversold", "neutral", "overbought")

    def test_has_momentum(self, sample_price_data, sample_screening_config):
        """Each stock should have momentum returns for configured windows."""
        metrics = compute_returns_and_metrics(
            price_data=sample_price_data,
            config=sample_screening_config,
        )
        for m in metrics:
            assert len(m.momentum_returns) > 0
            # At least the 5d window should be present (40 days > 5)
            assert "5d" in m.momentum_returns

    def test_has_performance_metrics(self, sample_price_data, sample_screening_config):
        """Each stock should have CAGR, Sharpe, Sortino, etc."""
        metrics = compute_returns_and_metrics(
            price_data=sample_price_data,
            config=sample_screening_config,
        )
        for m in metrics:
            assert isinstance(m.cagr, float)
            assert isinstance(m.sharpe, float)
            assert isinstance(m.sortino, float)
            assert isinstance(m.calmar, float)
            assert m.max_drawdown <= 0  # Drawdown is negative or zero

    def test_skips_insufficient_data(self, sample_screening_config):
        """Symbols with too few data points are skipped."""
        import json
        sparse_data = {
            "AAPL": json.dumps([["2026-01-01", 190.0], ["2026-01-02", 191.0]]),  # Only 2 days
        }
        metrics = compute_returns_and_metrics(
            price_data=sparse_data,
            config=sample_screening_config,
        )
        assert len(metrics) == 0

    def test_empty_price_data(self, sample_screening_config):
        """Empty price data -> empty metrics list."""
        metrics = compute_returns_and_metrics(
            price_data={},
            config=sample_screening_config,
        )
        assert metrics == []


# ============================================================
# Task: cluster_stocks
# ============================================================

class TestClusterStocks:
    """Tests for the cluster_stocks task."""

    def test_cluster_assignments(self, sample_stock_metrics):
        """Each stock should have a cluster_id and label."""
        import json
        result = cluster_stocks(stock_metrics=sample_stock_metrics, max_k=5)
        assert len(result) == 5
        for symbol in ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]:
            assert symbol in result
            assignment = json.loads(result[symbol])
            assert len(assignment) == 2
            assert isinstance(assignment[0], int)  # cluster_id
            assert isinstance(assignment[1], str)  # cluster_label

    def test_cluster_labels_descriptive(self, sample_stock_metrics):
        """Cluster labels should contain Mom and Vol descriptors."""
        import json
        result = cluster_stocks(stock_metrics=sample_stock_metrics, max_k=5)
        for symbol, assignment_json in result.items():
            assignment = json.loads(assignment_json)
            label = assignment[1]
            assert "Mom" in label
            assert "Vol" in label

    def test_few_stocks_fallback(self):
        """Fewer than 3 stocks -> single cluster."""
        import json
        metrics = [
            StockMetrics(
                symbol="A", forward_return=0.05, momentum_returns={"5d": 0.02},
                rsi=50.0, rsi_signal="neutral", volatility_252d=0.20,
                cagr=0.10, sharpe=1.0, sortino=1.5, calmar=1.0, max_drawdown=-0.10,
                z_scores={}, composite_score=0.0, quintile=3,
            ),
            StockMetrics(
                symbol="B", forward_return=0.03, momentum_returns={"5d": 0.01},
                rsi=55.0, rsi_signal="neutral", volatility_252d=0.18,
                cagr=0.08, sharpe=0.8, sortino=1.2, calmar=0.8, max_drawdown=-0.10,
                z_scores={}, composite_score=0.0, quintile=3,
            ),
        ]
        result = cluster_stocks(stock_metrics=metrics, max_k=5)
        assert len(result) == 2
        # All in cluster 0
        assert json.loads(result["A"])[0] == 0
        assert json.loads(result["B"])[0] == 0


# ============================================================
# Task: score_and_rank_factors
# ============================================================

class TestScoreAndRankFactors:
    """Tests for the score_and_rank_factors task."""

    def test_z_scores_populated(self, sample_stock_metrics):
        """Each stock should have z_scores for all 4 factors."""
        weights = {"momentum": 0.30, "low_volatility": 0.25, "rsi_signal": 0.20, "sharpe": 0.25}
        result = score_and_rank_factors(
            stock_metrics=sample_stock_metrics,
            factor_weights=weights,
        )
        for m in result:
            assert len(m.z_scores) == 4
            assert "momentum" in m.z_scores
            assert "low_volatility" in m.z_scores
            assert "rsi_signal" in m.z_scores
            assert "sharpe" in m.z_scores

    def test_sorted_by_composite_score(self, sample_stock_metrics):
        """Result should be sorted by composite_score descending."""
        weights = {"momentum": 0.30, "low_volatility": 0.25, "rsi_signal": 0.20, "sharpe": 0.25}
        result = score_and_rank_factors(
            stock_metrics=sample_stock_metrics,
            factor_weights=weights,
        )
        scores = [m.composite_score for m in result]
        assert scores == sorted(scores, reverse=True)

    def test_quintiles_assigned(self, sample_stock_metrics):
        """Each stock should have a quintile 1-5."""
        weights = {"momentum": 0.30, "low_volatility": 0.25, "rsi_signal": 0.20, "sharpe": 0.25}
        result = score_and_rank_factors(
            stock_metrics=sample_stock_metrics,
            factor_weights=weights,
        )
        for m in result:
            assert 1 <= m.quintile <= 5

    def test_empty_input(self):
        """Empty metrics list -> empty result."""
        weights = {"momentum": 0.30, "low_volatility": 0.25, "rsi_signal": 0.20, "sharpe": 0.25}
        result = score_and_rank_factors(
            stock_metrics=[],
            factor_weights=weights,
        )
        assert result == []

    def test_best_stock_has_highest_score(self, sample_stock_metrics):
        """NVDA has highest sharpe + high momentum -> should rank near top."""
        weights = {"momentum": 0.30, "low_volatility": 0.25, "rsi_signal": 0.20, "sharpe": 0.25}
        result = score_and_rank_factors(
            stock_metrics=sample_stock_metrics,
            factor_weights=weights,
        )
        # NVDA has best sharpe (1.8) and best momentum returns
        top_symbols = [m.symbol for m in result[:2]]
        assert "NVDA" in top_symbols


# ============================================================
# Task: merge_cluster_assignments
# ============================================================

class TestMergeClusterAssignments:
    """Tests for the merge_cluster_assignments task."""

    def test_merge_populates_clusters(self, sample_stock_metrics):
        """Cluster IDs and labels should be merged into metrics."""
        import json
        assignments = {
            "AAPL": json.dumps([0, "HiMom-LoVol"]),
            "MSFT": json.dumps([0, "HiMom-LoVol"]),
            "GOOGL": json.dumps([1, "LoMom-HiVol"]),
            "AMZN": json.dumps([2, "HiMom-HiVol"]),
            "NVDA": json.dumps([2, "HiMom-HiVol"]),
        }
        result = merge_cluster_assignments(
            ranked_metrics=sample_stock_metrics,
            cluster_assignments=assignments,
        )
        assert len(result) == 5
        for m in result:
            assert m.cluster_id >= 0
            assert m.cluster_label != ""

    def test_missing_cluster_assignment(self, sample_stock_metrics):
        """Stocks missing from assignments get cluster_id=-1."""
        import json
        assignments = {"AAPL": json.dumps([0, "HiMom-LoVol"])}  # Only AAPL
        result = merge_cluster_assignments(
            ranked_metrics=sample_stock_metrics,
            cluster_assignments=assignments,
        )
        aapl = [m for m in result if m.symbol == "AAPL"][0]
        msft = [m for m in result if m.symbol == "MSFT"][0]
        assert aapl.cluster_id == 0
        assert msft.cluster_id == -1
        assert msft.cluster_label == "unknown"

    def test_preserves_scores(self, sample_stock_metrics):
        """Merge should preserve all original metric fields."""
        import json
        # Give them some scores first
        for i, m in enumerate(sample_stock_metrics):
            sample_stock_metrics[i] = StockMetrics(
                symbol=m.symbol, forward_return=m.forward_return,
                momentum_returns=m.momentum_returns, rsi=m.rsi,
                rsi_signal=m.rsi_signal, volatility_252d=m.volatility_252d,
                cagr=m.cagr, sharpe=m.sharpe, sortino=m.sortino,
                calmar=m.calmar, max_drawdown=m.max_drawdown,
                z_scores={"momentum": 0.5}, composite_score=1.23,
                quintile=2,
            )
        assignments = {m.symbol: json.dumps([0, "test"]) for m in sample_stock_metrics}
        result = merge_cluster_assignments(
            ranked_metrics=sample_stock_metrics,
            cluster_assignments=assignments,
        )
        for m in result:
            assert m.composite_score == 1.23
            assert m.quintile == 2
            assert m.z_scores == {"momentum": 0.5}


# ============================================================
# Task: assemble_screening_result
# ============================================================

class TestAssembleScreeningResult:
    """Tests for assemble_screening_result task."""

    def test_assembles_result(self, sample_stock_metrics, sample_price_data, sample_screening_config):
        """Should produce a valid ScreeningResult."""
        result = assemble_screening_result(
            run_date="2026-02-08",
            config=sample_screening_config,
            final_metrics=sample_stock_metrics,
            price_data=sample_price_data,
        )
        assert isinstance(result, ScreeningResult)
        assert result.run_date == "2026-02-08"
        assert result.num_symbols_with_data == 5
        assert result.num_symbols_input == 5  # From config.symbols

    def test_benchmark_computed(self, sample_stock_metrics, sample_price_data, sample_screening_config):
        """Benchmark CAGR and Sharpe should be computed."""
        result = assemble_screening_result(
            run_date="2026-02-08",
            config=sample_screening_config,
            final_metrics=sample_stock_metrics,
            price_data=sample_price_data,
        )
        # With 40 days of data, benchmark should have non-zero values
        assert isinstance(result.benchmark_cagr, float)
        assert isinstance(result.benchmark_sharpe, float)

    def test_auto_date(self, sample_stock_metrics, sample_price_data, sample_screening_config):
        """Empty run_date should default to today."""
        result = assemble_screening_result(
            run_date="",
            config=sample_screening_config,
            final_metrics=sample_stock_metrics,
            price_data=sample_price_data,
        )
        assert result.run_date != ""
        assert len(result.run_date) == 10  # YYYY-MM-DD

    def test_empty_price_data(self, sample_stock_metrics, sample_screening_config):
        """Empty price data -> zero benchmark metrics."""
        result = assemble_screening_result(
            run_date="2026-02-08",
            config=sample_screening_config,
            final_metrics=sample_stock_metrics,
            price_data={},
        )
        assert result.benchmark_cagr == 0.0
        assert result.benchmark_sharpe == 0.0


# ============================================================
# Task: generate_screening_report
# ============================================================

class TestGenerateScreeningReport:
    """Tests for generate_screening_report task."""

    def test_report_contains_header(self, sample_stock_metrics, sample_screening_config, sample_price_data):
        """Report should contain run date and header."""
        result = ScreeningResult(
            run_date="2026-02-08",
            config=sample_screening_config,
            stock_metrics=sample_stock_metrics,
            benchmark_cagr=0.12,
            benchmark_sharpe=1.5,
            benchmark_cumulative_return=0.08,
            num_symbols_input=5,
            num_symbols_with_data=5,
            optimal_k_clusters=2,
        )
        report = generate_screening_report(result=result)
        assert "2026-02-08" in report
        assert "WF2 Screening Report" in report
        assert "Benchmark" in report

    def test_report_contains_top_stocks(self, sample_stock_metrics, sample_screening_config):
        """Report should list top stocks."""
        result = ScreeningResult(
            run_date="2026-02-08",
            config=sample_screening_config,
            stock_metrics=sample_stock_metrics,
            benchmark_cagr=0.12,
            benchmark_sharpe=1.5,
            benchmark_cumulative_return=0.08,
            num_symbols_input=5,
            num_symbols_with_data=5,
            optimal_k_clusters=2,
        )
        report = generate_screening_report(result=result)
        assert "AAPL" in report
        assert "Top 10" in report

    def test_report_rsi_signals(self, sample_stock_metrics, sample_screening_config):
        """Report should show RSI signal summary."""
        result = ScreeningResult(
            run_date="2026-02-08",
            config=sample_screening_config,
            stock_metrics=sample_stock_metrics,
            benchmark_cagr=0.12,
            benchmark_sharpe=1.5,
            benchmark_cumulative_return=0.08,
            num_symbols_input=5,
            num_symbols_with_data=5,
            optimal_k_clusters=2,
        )
        report = generate_screening_report(result=result)
        assert "RSI Signals" in report
        assert "Oversold" in report
        assert "Overbought" in report

    def test_report_data_quality_notes(self, sample_screening_config):
        """Data quality notes should appear in report."""
        result = ScreeningResult(
            run_date="2026-02-08",
            config=sample_screening_config,
            stock_metrics=[],
            benchmark_cagr=0.0,
            benchmark_sharpe=0.0,
            benchmark_cumulative_return=0.0,
            num_symbols_input=5,
            num_symbols_with_data=0,
            optimal_k_clusters=0,
            data_quality_notes=["3 symbols had insufficient historical data"],
        )
        report = generate_screening_report(result=result)
        assert "Data Quality Notes" in report
        assert "insufficient historical data" in report


# ============================================================
# Helper: _find_elbow
# ============================================================

class TestFindElbow:
    """Tests for the _find_elbow helper."""

    def test_clear_elbow(self):
        """Classic elbow pattern should find the bend point."""
        # Steep drop from K=1 to K=3, then flat
        inertias = [1000, 300, 100, 80, 70, 65, 62, 60]
        elbow = _find_elbow(inertias)
        assert 2 <= elbow <= 4  # Elbow around K=2 or K=3

    def test_two_points(self):
        """Two inertias -> return 2."""
        inertias = [100, 50]
        elbow = _find_elbow(inertias)
        assert elbow >= 1

    def test_monotonically_decreasing(self):
        """Linearly decreasing inertias -> still returns a valid K."""
        inertias = [100, 80, 60, 40, 20]
        elbow = _find_elbow(inertias)
        assert 1 <= elbow <= 5
