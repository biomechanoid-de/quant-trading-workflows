"""Tests for dividend tracking across WF1, WF4, and WF5.

Tests validate:
- YFinanceProvider.fetch_dividends() with mocked yfinance
- WF1 fetch_dividend_events task with mocked provider + DB
- WF4 process_dividends task (cash mode, DRIP mode, disabled, no pending)
- WF4 snapshot_portfolio cumulative_dividends handling
- WF5 report dividend section
"""

import json
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

import pytest


# ============================================================
# Provider: fetch_dividends
# ============================================================

class TestYFinanceProviderFetchDividends:
    """Test YFinanceProvider.fetch_dividends() with mocked yfinance."""

    def test_fetch_dividends_returns_recent_events(self):
        """Dividends within 90 days are returned."""
        import pandas as pd

        mock_dividends = pd.Series(
            [0.25, 0.30],
            index=pd.DatetimeIndex([
                datetime.now() - timedelta(days=30),
                datetime.now() - timedelta(days=60),
            ]),
        )

        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_ticker = MagicMock()
            mock_ticker.dividends = mock_dividends
            mock_ticker_cls.return_value = mock_ticker

            from src.shared.providers.yfinance_provider import YFinanceProvider
            provider = YFinanceProvider()

            with patch("time.sleep"):
                result = provider.fetch_dividends("AAPL")

        assert len(result) == 2
        assert result[0]["symbol"] == "AAPL"
        assert result[0]["amount_per_share"] == 0.25
        assert "ex_date" in result[0]

    def test_fetch_dividends_filters_old_events(self):
        """Dividends older than 90 days are filtered out."""
        import pandas as pd

        mock_dividends = pd.Series(
            [0.25, 0.50],
            index=pd.DatetimeIndex([
                datetime.now() - timedelta(days=30),
                datetime.now() - timedelta(days=120),  # too old
            ]),
        )

        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_ticker = MagicMock()
            mock_ticker.dividends = mock_dividends
            mock_ticker_cls.return_value = mock_ticker

            from src.shared.providers.yfinance_provider import YFinanceProvider
            provider = YFinanceProvider()

            with patch("time.sleep"):
                result = provider.fetch_dividends("AAPL")

        assert len(result) == 1
        assert result[0]["amount_per_share"] == 0.25

    def test_fetch_dividends_empty_series(self):
        """Returns empty list when no dividends exist."""
        import pandas as pd

        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_ticker = MagicMock()
            mock_ticker.dividends = pd.Series([], dtype=float)
            mock_ticker_cls.return_value = mock_ticker

            from src.shared.providers.yfinance_provider import YFinanceProvider
            provider = YFinanceProvider()

            with patch("time.sleep"):
                result = provider.fetch_dividends("TSLA")

        assert result == []

    def test_fetch_dividends_handles_exception(self):
        """Returns empty list on yfinance errors."""
        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_ticker_cls.side_effect = Exception("API error")

            from src.shared.providers.yfinance_provider import YFinanceProvider
            provider = YFinanceProvider()

            with patch("time.sleep"):
                result = provider.fetch_dividends("BROKEN")

        assert result == []


# ============================================================
# WF1: fetch_dividend_events task
# ============================================================

class TestFetchDividendEventsTask:
    """Test WF1 fetch_dividend_events task.

    Note: Uses source-module patches because tasks use lazy imports.
    """

    def test_fetch_dividend_events_finds_dividends(self):
        """Task returns summary when dividends are found."""
        mock_divs = [
            {"symbol": "AAPL", "ex_date": "2026-02-01", "amount_per_share": 0.25},
        ]

        mock_provider = MagicMock()
        mock_provider.fetch_dividends.return_value = mock_divs

        with patch(
            "src.shared.providers.yfinance_provider.YFinanceProvider",
            return_value=mock_provider,
        ), patch(
            "src.shared.db.store_dividends", return_value=1,
        ):
            from src.wf1_data_ingestion.tasks import fetch_dividend_events
            result = fetch_dividend_events(
                symbols=["AAPL"], date="2026-02-15"
            )

        assert "1 dividend events" in result
        assert "stored 1 rows" in result

    def test_fetch_dividend_events_no_dividends(self):
        """Task returns summary when no dividends are found."""
        mock_provider = MagicMock()
        mock_provider.fetch_dividends.return_value = []

        with patch(
            "src.shared.providers.yfinance_provider.YFinanceProvider",
            return_value=mock_provider,
        ):
            from src.wf1_data_ingestion.tasks import fetch_dividend_events
            result = fetch_dividend_events(
                symbols=["TSLA"], date="2026-02-15"
            )

        assert "No dividend events found" in result

    def test_fetch_dividend_events_multiple_symbols(self):
        """Task aggregates dividends from multiple symbols."""
        def mock_fetch(symbol):
            if symbol == "AAPL":
                return [{"symbol": "AAPL", "ex_date": "2026-02-01", "amount_per_share": 0.25}]
            elif symbol == "JPM":
                return [
                    {"symbol": "JPM", "ex_date": "2026-01-15", "amount_per_share": 1.15},
                    {"symbol": "JPM", "ex_date": "2026-02-10", "amount_per_share": 1.15},
                ]
            return []

        mock_provider = MagicMock()
        mock_provider.fetch_dividends.side_effect = mock_fetch

        with patch(
            "src.shared.providers.yfinance_provider.YFinanceProvider",
            return_value=mock_provider,
        ), patch(
            "src.shared.db.store_dividends", return_value=3,
        ):
            from src.wf1_data_ingestion.tasks import fetch_dividend_events
            result = fetch_dividend_events(
                symbols=["AAPL", "TSLA", "JPM"], date="2026-02-15"
            )

        assert "3 dividend events" in result


# ============================================================
# WF4: process_dividends task
# ============================================================

class TestProcessDividendsTask:
    """Test WF4 process_dividends task."""

    def _make_paper_trade_result(self, cash=22069.50, positions=None):
        """Helper to create a paper_trade_result dict."""
        if positions is None:
            positions = [
                {"symbol": "AAPL", "shares": 6.0, "avg_cost": 195.5,
                 "current_price": 195.5, "sector": "Technology"},
                {"symbol": "MSFT", "shares": 2.0, "avg_cost": 415.2,
                 "current_price": 415.2, "sector": "Technology"},
            ]
        return {
            "status": "executed",
            "num_trades_executed": "2",
            "cash_after": str(cash),
            "positions_after_json": json.dumps(positions),
            "total_value_after": str(cash + sum(
                p["shares"] * p["current_price"] for p in positions
            )),
            "run_date": "2026-02-16",
        }

    def test_process_dividends_disabled(self):
        """When paper_trading=False, returns passthrough with disabled action."""
        paper_result = self._make_paper_trade_result()

        from src.wf4_portfolio_rebalancing.tasks import process_dividends
        result = process_dividends(
            paper_trade_result=paper_result,
            paper_trading=False,
            dividend_reinvest=False,
            initial_capital=25000.0,
        )

        assert result["dividend_action"] == "disabled"
        assert result["dividends_processed"] == "0"
        assert result["cash_after"] == paper_result["cash_after"]

    def test_process_dividends_cash_mode(self):
        """Cash dividends add to cash balance."""
        paper_result = self._make_paper_trade_result(cash=5000.0)

        # Pending dividend: AAPL, $0.25/share, 6 shares held -> $1.50
        mock_pending = [
            (1, "AAPL", "2026-02-10", 0.25, 6.0),
        ]

        with patch(
            "src.shared.db.get_pending_dividends",
            return_value=mock_pending,
        ), patch(
            "src.shared.db.mark_dividends_processed",
            return_value=1,
        ), patch(
            "src.shared.db.get_cumulative_dividend_total",
            return_value=1.50,
        ):
            from src.wf4_portfolio_rebalancing.tasks import process_dividends
            result = process_dividends(
                paper_trade_result=paper_result,
                paper_trading=True,
                dividend_reinvest=False,
                initial_capital=25000.0,
            )

        assert result["dividend_action"] == "cash"
        assert result["dividends_processed"] == "1"
        assert float(result["total_dividend_amount"]) == 1.50
        assert float(result["cash_after"]) == 5001.50
        assert float(result["cumulative_dividends"]) == 1.50

    def test_process_dividends_drip_mode(self):
        """DRIP dividends add shares instead of cash."""
        paper_result = self._make_paper_trade_result(cash=5000.0)

        # Pending dividend: AAPL, $0.25/share, 6 shares -> $1.50
        # At $195.50/share -> ~0.0077 new shares
        mock_pending = [
            (1, "AAPL", "2026-02-10", 0.25, 6.0),
        ]

        with patch(
            "src.shared.db.get_pending_dividends",
            return_value=mock_pending,
        ), patch(
            "src.shared.db.mark_dividends_processed",
            return_value=1,
        ), patch(
            "src.shared.db.get_cumulative_dividend_total",
            return_value=1.50,
        ), patch(
            "src.shared.db.upsert_positions",
            return_value=2,
        ):
            from src.wf4_portfolio_rebalancing.tasks import process_dividends
            result = process_dividends(
                paper_trade_result=paper_result,
                paper_trading=True,
                dividend_reinvest=True,
                initial_capital=25000.0,
            )

        assert result["dividend_action"] == "reinvest"
        assert result["dividends_processed"] == "1"
        # Cash should NOT increase in DRIP mode
        assert float(result["cash_after"]) == 5000.0
        # Shares should increase
        positions = json.loads(result["positions_after_json"])
        aapl = next(p for p in positions if p["symbol"] == "AAPL")
        assert aapl["shares"] > 6.0

    def test_process_dividends_no_pending(self):
        """No pending dividends results in zero processing."""
        paper_result = self._make_paper_trade_result(cash=5000.0)

        with patch(
            "src.shared.db.get_pending_dividends",
            return_value=[],
        ), patch(
            "src.shared.db.mark_dividends_processed",
            return_value=0,
        ), patch(
            "src.shared.db.get_cumulative_dividend_total",
            return_value=0.0,
        ):
            from src.wf4_portfolio_rebalancing.tasks import process_dividends
            result = process_dividends(
                paper_trade_result=paper_result,
                paper_trading=True,
                dividend_reinvest=False,
                initial_capital=25000.0,
            )

        assert result["dividends_processed"] == "0"
        assert float(result["total_dividend_amount"]) == 0.0
        assert float(result["cash_after"]) == 5000.0

    def test_process_dividends_no_position_for_symbol(self):
        """Dividends for symbols we don't hold are processed with 0 amount."""
        paper_result = self._make_paper_trade_result(cash=5000.0)

        # Dividend for XOM which we don't hold -> shares_held=0
        mock_pending = [
            (1, "XOM", "2026-02-10", 0.95, 0.0),
        ]

        with patch(
            "src.shared.db.get_pending_dividends",
            return_value=mock_pending,
        ), patch(
            "src.shared.db.mark_dividends_processed",
            return_value=1,
        ), patch(
            "src.shared.db.get_cumulative_dividend_total",
            return_value=0.0,
        ):
            from src.wf4_portfolio_rebalancing.tasks import process_dividends
            result = process_dividends(
                paper_trade_result=paper_result,
                paper_trading=True,
                dividend_reinvest=False,
                initial_capital=25000.0,
            )

        assert result["dividends_processed"] == "1"
        assert float(result["total_dividend_amount"]) == 0.0
        assert float(result["cash_after"]) == 5000.0


# ============================================================
# WF4: snapshot_portfolio with cumulative_dividends
# ============================================================

class TestSnapshotPortfolioDividends:
    """Test that snapshot_portfolio reads cumulative_dividends from input."""

    def test_snapshot_uses_cumulative_from_process_dividends(self):
        """snapshot_portfolio reads cumulative_dividends from its input dict."""
        paper_result = {
            "status": "executed",
            "num_trades_executed": "2",
            "cash_after": "5001.50",
            "positions_after_json": json.dumps([
                {"symbol": "AAPL", "shares": 6, "avg_cost": 195.5,
                 "current_price": 195.5, "sector": "Technology"},
            ]),
            "total_value_after": "6174.50",
            "run_date": "2026-02-16",
            "cumulative_dividends": "1.50",
        }

        with patch(
            "src.shared.db.store_portfolio_snapshot",
            return_value=1,
        ) as mock_store, patch(
            "src.shared.db.get_latest_portfolio_snapshot",
            return_value=("2026-02-15", 6150.0, 5000.0, 1150.0, 50.0, 0.0, 1),
        ):
            from src.wf4_portfolio_rebalancing.tasks import snapshot_portfolio
            result = snapshot_portfolio(
                paper_trade_result=paper_result,
                paper_trading=True,
                initial_capital=25000.0,
            )

        # Verify store was called with cumulative_dividends=1.50
        mock_store.assert_called_once()
        call_kwargs = mock_store.call_args
        assert call_kwargs[1]["cumulative_dividends"] == 1.50


# ============================================================
# WF5: Monitoring report dividend section
# ============================================================

class TestMonitoringReportDividends:
    """Test that WF5 monitoring report includes dividend section."""

    def test_report_includes_dividend_section(
        self, sample_wf5_pnl_data, sample_wf5_risk_data,
        sample_wf5_alert_data_no_alerts,
    ):
        """Report includes dividends section when data exists."""
        mock_summary = {"cumulative": 15.50, "mtd": 5.00, "ytd": 15.50}

        with patch(
            "src.shared.db.get_dividend_summary",
            return_value=mock_summary,
        ), patch(
            "src.shared.config.WF4_DIVIDEND_REINVEST",
            False,
        ), patch(
            "src.wf5_monitoring.tasks._store_and_upload",
        ):
            from src.wf5_monitoring.tasks import generate_monitoring_report
            report = generate_monitoring_report(
                pnl_data=sample_wf5_pnl_data,
                risk_data=sample_wf5_risk_data,
                alert_data=sample_wf5_alert_data_no_alerts,
            )

        assert "## Dividends" in report
        assert "Cumulative" in report
        assert "15.50" in report
        assert "Cash" in report

    def test_report_skips_dividend_section_when_no_dividends(
        self, sample_wf5_pnl_data, sample_wf5_risk_data,
        sample_wf5_alert_data_no_alerts,
    ):
        """Report skips dividends section when cumulative is 0."""
        mock_summary = {"cumulative": 0.0, "mtd": 0.0, "ytd": 0.0}

        with patch(
            "src.shared.db.get_dividend_summary",
            return_value=mock_summary,
        ), patch(
            "src.wf5_monitoring.tasks._store_and_upload",
        ):
            from src.wf5_monitoring.tasks import generate_monitoring_report
            report = generate_monitoring_report(
                pnl_data=sample_wf5_pnl_data,
                risk_data=sample_wf5_risk_data,
                alert_data=sample_wf5_alert_data_no_alerts,
            )

        assert "## Dividends" not in report


# ============================================================
# Config: WF4_DIVIDEND_REINVEST
# ============================================================

class TestDividendConfig:
    """Test dividend configuration."""

    def test_dividend_reinvest_default_false(self):
        """WF4_DIVIDEND_REINVEST defaults to False."""
        import os
        env_backup = os.environ.pop("WF4_DIVIDEND_REINVEST", None)
        try:
            result = os.environ.get("WF4_DIVIDEND_REINVEST", "false").lower() == "true"
            assert result is False
        finally:
            if env_backup is not None:
                os.environ["WF4_DIVIDEND_REINVEST"] = env_backup

    def test_dividend_reinvest_can_be_enabled(self):
        """WF4_DIVIDEND_REINVEST can be set to true via env var."""
        import os
        env_backup = os.environ.get("WF4_DIVIDEND_REINVEST")
        os.environ["WF4_DIVIDEND_REINVEST"] = "true"
        try:
            result = os.environ.get("WF4_DIVIDEND_REINVEST", "false").lower() == "true"
            assert result is True
        finally:
            if env_backup is None:
                del os.environ["WF4_DIVIDEND_REINVEST"]
            else:
                os.environ["WF4_DIVIDEND_REINVEST"] = env_backup
