"""Tests for WF6: Backtesting.

Tests resolve_backtest_params, compare_strategies, generate_backtest_report,
and the backtesting workflow. Database-dependent tasks (load_historical_signals,
simulate_signal_portfolio, compute_benchmark_returns) are tested with mocked DB.

No actual database or network access required — all data from mocks.
"""

import json
from unittest.mock import patch, MagicMock

import pytest

from src.wf6_backtesting.tasks import (
    resolve_backtest_params,
    compare_strategies,
    generate_backtest_report,
    load_historical_signals,
    simulate_signal_portfolio,
    compute_benchmark_returns,
)


# ============================================================
# resolve_backtest_params
# ============================================================

class TestResolveBacktestParams:
    """Tests for resolve_backtest_params task."""

    def test_all_params_passed_through(self):
        result = resolve_backtest_params(
            start_date="2026-01-01",
            end_date="2026-02-15",
            initial_capital=25000.0,
            max_position_pct=0.05,
            max_sector_pct=0.25,
            cash_reserve_pct=0.05,
        )
        assert result["start_date"] == "2026-01-01"
        assert result["end_date"] == "2026-02-15"
        assert result["initial_capital"] == "25000.0"
        assert result["max_position_pct"] == "0.05"

    def test_empty_end_date_defaults_to_today(self):
        result = resolve_backtest_params(
            start_date="2026-01-01",
            end_date="",
            initial_capital=25000.0,
            max_position_pct=0.05,
            max_sector_pct=0.25,
            cash_reserve_pct=0.05,
        )
        assert result["end_date"] != ""
        # Should be a valid date string
        parts = result["end_date"].split("-")
        assert len(parts) == 3


# ============================================================
# compare_strategies
# ============================================================

class TestCompareStrategies:
    """Tests for compare_strategies task."""

    def test_compare_basic(self):
        """Compare two strategies with valid data."""
        params = {
            "start_date": "2026-01-01",
            "end_date": "2026-02-15",
            "initial_capital": "25000.0",
        }
        # Portfolio that gains 10%
        portfolio_values = [25000.0] + [25000.0 + i * 100 for i in range(1, 31)]
        portfolio_returns = [0.0] + [100 / (25000 + (i - 1) * 100) for i in range(1, 31)]

        # Benchmark that gains 5%
        benchmark_values = [25000.0] + [25000.0 + i * 50 for i in range(1, 31)]
        benchmark_returns = [0.0] + [50 / (25000 + (i - 1) * 50) for i in range(1, 31)]

        portfolio = {
            "values_json": json.dumps(portfolio_values),
            "daily_returns_json": json.dumps(portfolio_returns),
            "num_rebalances": "4",
        }
        benchmark = {
            "values_json": json.dumps(benchmark_values),
            "daily_returns_json": json.dumps(benchmark_returns),
        }

        result = compare_strategies(params=params, portfolio=portfolio, benchmark=benchmark)

        port_metrics = json.loads(result["portfolio_metrics_json"])
        bench_metrics = json.loads(result["benchmark_metrics_json"])
        excess = json.loads(result["excess_metrics_json"])

        assert port_metrics["total_return_pct"] > bench_metrics["total_return_pct"]
        assert excess["total_return_excess_pct"] > 0
        assert result["num_rebalances"] == "4"

    def test_compare_empty_data(self):
        """Compare with empty data should return zeros."""
        params = {
            "start_date": "2026-01-01",
            "end_date": "2026-02-15",
            "initial_capital": "25000.0",
        }
        portfolio = {
            "values_json": "[]",
            "daily_returns_json": "[]",
            "num_rebalances": "0",
        }
        benchmark = {
            "values_json": "[]",
            "daily_returns_json": "[]",
        }

        result = compare_strategies(params=params, portfolio=portfolio, benchmark=benchmark)
        port_metrics = json.loads(result["portfolio_metrics_json"])

        assert port_metrics["cagr"] == 0.0
        assert port_metrics["sharpe"] == 0.0
        assert port_metrics["total_return_pct"] == 0.0

    def test_excess_metrics_computed(self):
        """Excess metrics should be portfolio - benchmark."""
        params = {
            "start_date": "2026-01-01",
            "end_date": "2026-02-15",
            "initial_capital": "1000.0",
        }
        # Both identical -> excess should be zero
        values = [1000.0, 1010.0, 1020.0, 1030.0]
        returns = [0.0, 0.01, 0.0099, 0.0098]

        portfolio = {
            "values_json": json.dumps(values),
            "daily_returns_json": json.dumps(returns),
            "num_rebalances": "1",
        }
        benchmark = {
            "values_json": json.dumps(values),
            "daily_returns_json": json.dumps(returns),
        }

        result = compare_strategies(params=params, portfolio=portfolio, benchmark=benchmark)
        excess = json.loads(result["excess_metrics_json"])

        assert excess["cagr_excess"] == 0.0
        assert excess["sharpe_excess"] == 0.0
        assert excess["total_return_excess_pct"] == 0.0


# ============================================================
# generate_backtest_report
# ============================================================

class TestGenerateBacktestReport:
    """Tests for generate_backtest_report task."""

    def _make_comparison(self):
        return {
            "portfolio_metrics_json": json.dumps({
                "cagr": 0.12, "sharpe": 1.5, "sortino": 2.0,
                "max_drawdown": -0.08, "calmar": 1.5,
                "total_return_pct": 12.0, "final_value": 28000.0,
            }),
            "benchmark_metrics_json": json.dumps({
                "cagr": 0.08, "sharpe": 1.0, "sortino": 1.2,
                "max_drawdown": -0.12, "calmar": 0.67,
                "total_return_pct": 8.0, "final_value": 27000.0,
            }),
            "excess_metrics_json": json.dumps({
                "cagr_excess": 0.04, "sharpe_excess": 0.5,
                "total_return_excess_pct": 4.0,
            }),
            "num_rebalances": "6",
            "start_date": "2026-01-01",
            "end_date": "2026-02-15",
        }

    @patch("src.wf6_backtesting.tasks._upload_backtest_report")
    @patch("src.wf6_backtesting.tasks._store_backtest_to_db")
    def test_report_contains_header(self, mock_db, mock_s3):
        params = {"initial_capital": "25000.0"}
        comparison = self._make_comparison()
        portfolio = {"values_json": "[]", "daily_returns_json": "[]"}
        benchmark = {"values_json": "[]", "daily_returns_json": "[]"}

        report = generate_backtest_report(
            params=params, comparison=comparison,
            portfolio=portfolio, benchmark=benchmark,
        )
        assert "WF6 Backtest Report" in report
        assert "2026-01-01" in report
        assert "2026-02-15" in report

    @patch("src.wf6_backtesting.tasks._upload_backtest_report")
    @patch("src.wf6_backtesting.tasks._store_backtest_to_db")
    def test_report_contains_metrics(self, mock_db, mock_s3):
        params = {"initial_capital": "25000.0"}
        comparison = self._make_comparison()
        portfolio = {"values_json": "[]", "daily_returns_json": "[]"}
        benchmark = {"values_json": "[]", "daily_returns_json": "[]"}

        report = generate_backtest_report(
            params=params, comparison=comparison,
            portfolio=portfolio, benchmark=benchmark,
        )
        assert "CAGR" in report
        assert "Sharpe" in report
        assert "Sortino" in report
        assert "Max Drawdown" in report
        assert "Calmar" in report
        assert "Signal" in report
        assert "Buy&Hold" in report

    @patch("src.wf6_backtesting.tasks._upload_backtest_report")
    @patch("src.wf6_backtesting.tasks._store_backtest_to_db")
    def test_report_outperform_verdict(self, mock_db, mock_s3):
        params = {"initial_capital": "25000.0"}
        comparison = self._make_comparison()
        portfolio = {"values_json": "[]", "daily_returns_json": "[]"}
        benchmark = {"values_json": "[]", "daily_returns_json": "[]"}

        report = generate_backtest_report(
            params=params, comparison=comparison,
            portfolio=portfolio, benchmark=benchmark,
        )
        assert "OUTPERFORMS" in report

    @patch("src.wf6_backtesting.tasks._upload_backtest_report")
    @patch("src.wf6_backtesting.tasks._store_backtest_to_db")
    def test_report_underperform_verdict(self, mock_db, mock_s3):
        params = {"initial_capital": "25000.0"}
        comparison = {
            "portfolio_metrics_json": json.dumps({
                "cagr": 0.05, "sharpe": 0.8, "sortino": 1.0,
                "max_drawdown": -0.15, "calmar": 0.33,
                "total_return_pct": 5.0, "final_value": 26250.0,
            }),
            "benchmark_metrics_json": json.dumps({
                "cagr": 0.10, "sharpe": 1.2, "sortino": 1.5,
                "max_drawdown": -0.08, "calmar": 1.25,
                "total_return_pct": 10.0, "final_value": 27500.0,
            }),
            "excess_metrics_json": json.dumps({
                "cagr_excess": -0.05, "sharpe_excess": -0.4,
                "total_return_excess_pct": -5.0,
            }),
            "num_rebalances": "6",
            "start_date": "2026-01-01",
            "end_date": "2026-02-15",
        }
        portfolio = {"values_json": "[]", "daily_returns_json": "[]"}
        benchmark = {"values_json": "[]", "daily_returns_json": "[]"}

        report = generate_backtest_report(
            params=params, comparison=comparison,
            portfolio=portfolio, benchmark=benchmark,
        )
        assert "UNDERPERFORMS" in report

    @patch("src.wf6_backtesting.tasks._upload_backtest_report")
    @patch("src.wf6_backtesting.tasks._store_backtest_to_db")
    def test_db_failure_non_fatal(self, mock_db, mock_s3):
        """DB failure should not prevent report generation."""
        mock_db.side_effect = Exception("DB down")

        params = {"initial_capital": "25000.0"}
        comparison = self._make_comparison()
        portfolio = {"values_json": "[]", "daily_returns_json": "[]"}
        benchmark = {"values_json": "[]", "daily_returns_json": "[]"}

        report = generate_backtest_report(
            params=params, comparison=comparison,
            portfolio=portfolio, benchmark=benchmark,
        )
        assert "WF6 Backtest Report" in report

    @patch("src.wf6_backtesting.tasks._upload_backtest_report")
    @patch("src.wf6_backtesting.tasks._store_backtest_to_db")
    def test_minio_failure_non_fatal(self, mock_db, mock_s3):
        """MinIO failure should not prevent report generation."""
        mock_s3.side_effect = Exception("MinIO down")

        params = {"initial_capital": "25000.0"}
        comparison = self._make_comparison()
        portfolio = {"values_json": "[]", "daily_returns_json": "[]"}
        benchmark = {"values_json": "[]", "daily_returns_json": "[]"}

        report = generate_backtest_report(
            params=params, comparison=comparison,
            portfolio=portfolio, benchmark=benchmark,
        )
        assert "WF6 Backtest Report" in report


# ============================================================
# load_historical_signals (mocked DB)
# ============================================================

class TestLoadHistoricalSignals:
    """Tests for load_historical_signals task with mocked DB."""

    @patch("src.shared.db.get_connection")
    def test_loads_signals_for_date_range(self, mock_conn_fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn_fn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # First query: signal run dates
        # Second query: signal results for date 1
        # Third query: signal results for date 2
        mock_cursor.fetchall.side_effect = [
            # run dates
            [("2026-02-03",), ("2026-02-10",)],
            # signals for 2026-02-03
            [("AAPL", 65.0, "buy", 1, 68.0, 55.0, "complete", 72.0, "very_positive")],
            # signals for 2026-02-10
            [("MSFT", 70.0, "strong_buy", 1, 72.0, 60.0, "complete", 65.0, "positive")],
        ]

        params = {"start_date": "2026-02-01", "end_date": "2026-02-15"}
        result = load_historical_signals(params=params)

        assert result["num_run_dates"] == "2"
        run_dates = json.loads(result["run_dates_json"])
        assert len(run_dates) == 2
        signals = json.loads(result["signals_by_date_json"])
        assert "2026-02-03" in signals
        assert signals["2026-02-03"][0]["symbol"] == "AAPL"
        assert signals["2026-02-03"][0]["sentiment_score"] == 72.0

    @patch("src.shared.db.get_connection")
    def test_empty_date_range(self, mock_conn_fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn_fn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []

        params = {"start_date": "2026-02-01", "end_date": "2026-02-15"}
        result = load_historical_signals(params=params)

        assert result["num_run_dates"] == "0"
        assert json.loads(result["run_dates_json"]) == []


# ============================================================
# simulate_signal_portfolio (mocked DB)
# ============================================================

class TestSimulateSignalPortfolio:
    """Tests for simulate_signal_portfolio task with mocked DB."""

    @patch("src.shared.config.SYMBOL_SECTORS", {"AAPL": "Technology"})
    @patch("src.shared.db.get_connection")
    def test_empty_signals_returns_initial_capital(self, mock_conn_fn):
        params = {
            "initial_capital": "25000.0",
            "max_position_pct": "0.05",
            "max_sector_pct": "0.25",
            "cash_reserve_pct": "0.05",
            "end_date": "2026-02-15",
        }
        signals = {
            "run_dates_json": "[]",
            "signals_by_date_json": "{}",
        }

        result = simulate_signal_portfolio(params=params, signals=signals)
        assert result["final_value"] == "25000.0"
        assert result["num_rebalances"] == "0"

    @patch("src.shared.config.SYMBOL_SECTORS", {"AAPL": "Technology"})
    @patch("src.shared.db.get_connection")
    def test_single_rebalance(self, mock_conn_fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn_fn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Market data: AAPL at 150 on day 1, 155 on day 2
        mock_cursor.fetchall.return_value = [
            ("2026-02-03", "AAPL", 150.0),
            ("2026-02-04", "AAPL", 155.0),
        ]

        params = {
            "initial_capital": "10000.0",
            "max_position_pct": "0.95",
            "max_sector_pct": "0.95",
            "cash_reserve_pct": "0.05",
            "end_date": "2026-02-04",
        }
        signals = {
            "run_dates_json": json.dumps(["2026-02-03"]),
            "signals_by_date_json": json.dumps({
                "2026-02-03": [
                    {"symbol": "AAPL", "signal_strength": "strong_buy",
                     "combined_signal_score": 80.0},
                ],
            }),
        }

        result = simulate_signal_portfolio(params=params, signals=signals)
        assert int(result["num_rebalances"]) == 1
        # Final value should be > initial due to AAPL gaining
        assert float(result["final_value"]) > 10000.0


# ============================================================
# Workflow import test
# ============================================================

class TestBacktestingWorkflow:
    """Tests for the backtesting workflow definition."""

    def test_workflow_can_be_imported(self):
        from src.wf6_backtesting.workflow import backtesting_workflow
        assert backtesting_workflow is not None

    def test_workflow_defaults(self):
        from src.wf6_backtesting.workflow import backtesting_workflow
        import inspect
        sig = inspect.signature(backtesting_workflow)
        assert sig.parameters["start_date"].default == "2026-01-01"
        assert sig.parameters["end_date"].default == ""
        assert sig.parameters["max_position_pct"].default == 0.05
        assert sig.parameters["cash_reserve_pct"].default == 0.05
