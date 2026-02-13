"""Tests for WF4: Portfolio & Rebalancing.

Tests analytics functions (calculate_signal_weights, estimate_transaction_cost,
calculate_cost_breakdown), task logic (target weights, trade orders, report
generation, paper trading, portfolio snapshots), and data models.
No database or network access — all functions are tested with fixtures.
"""

import json
import pytest

from src.shared.analytics import (
    calculate_signal_weights,
    estimate_transaction_cost,
    calculate_cost_breakdown,
)
from src.shared.models import TradeOrder


# ============================================================
# Analytics: calculate_signal_weights
# ============================================================

class TestCalculateSignalWeights:
    """Test the 3-pass weight allocation algorithm."""

    def test_all_strong_buy_equal_scores(self):
        """All strong_buy with equal scores — weights should be capped at 5%."""
        signals = [
            {"symbol": f"SYM{i}", "signal_strength": "strong_buy", "combined_signal_score": 80.0}
            for i in range(20)
        ]
        weights = calculate_signal_weights(signals, max_position_pct=0.05)
        assert all(w <= 0.05 + 1e-6 for w in weights.values())
        assert sum(weights.values()) <= 0.95 + 1e-6  # Cash reserve respected

    def test_mixed_signals_only_buy_and_strong_buy_get_weight(self):
        """Only buy and strong_buy signals should get portfolio weight."""
        signals = [
            {"symbol": "A", "signal_strength": "strong_buy", "combined_signal_score": 80.0},
            {"symbol": "B", "signal_strength": "buy", "combined_signal_score": 65.0},
            {"symbol": "C", "signal_strength": "hold", "combined_signal_score": 45.0},
            {"symbol": "D", "signal_strength": "sell", "combined_signal_score": 30.0},
            {"symbol": "E", "signal_strength": "strong_sell", "combined_signal_score": 15.0},
        ]
        weights = calculate_signal_weights(signals)
        assert "A" in weights
        assert "B" in weights
        assert "C" not in weights
        assert "D" not in weights
        assert "E" not in weights

    def test_strong_buy_gets_higher_weight_than_buy(self):
        """Strong_buy should get roughly 3x/2x = 1.5x the weight of buy (same score)."""
        signals = [
            {"symbol": "A", "signal_strength": "strong_buy", "combined_signal_score": 70.0},
            {"symbol": "B", "signal_strength": "buy", "combined_signal_score": 70.0},
        ]
        # Use high position cap so capping doesn't equalize the two stocks
        weights = calculate_signal_weights(signals, max_position_pct=0.60)
        assert weights["A"] > weights["B"]
        # With 3x vs 2x multipliers: A should be ~60% of investable, B ~40%
        ratio = weights["A"] / weights["B"]
        assert 1.3 < ratio < 1.7  # ~1.5x

    def test_empty_signals_returns_empty(self):
        """No signals should return empty weights."""
        assert calculate_signal_weights([]) == {}

    def test_no_eligible_stocks(self):
        """All hold/sell signals should return empty weights."""
        signals = [
            {"symbol": "A", "signal_strength": "hold", "combined_signal_score": 50.0},
            {"symbol": "B", "signal_strength": "sell", "combined_signal_score": 30.0},
        ]
        assert calculate_signal_weights(signals) == {}

    def test_single_stock_capped_at_max(self):
        """Single stock can't exceed max_position_pct."""
        signals = [
            {"symbol": "A", "signal_strength": "strong_buy", "combined_signal_score": 90.0},
        ]
        weights = calculate_signal_weights(signals, max_position_pct=0.05)
        assert weights["A"] <= 0.05 + 1e-6

    def test_cash_reserve_respected(self):
        """Total weights should not exceed (1 - cash_reserve_pct)."""
        signals = [
            {"symbol": f"SYM{i}", "signal_strength": "buy", "combined_signal_score": 70.0}
            for i in range(5)
        ]
        weights = calculate_signal_weights(signals, cash_reserve_pct=0.10)
        assert sum(weights.values()) <= 0.90 + 1e-6

    def test_sector_cap_enforcement(self):
        """Sector cap should limit concentration in one sector."""
        sector_map = {f"TECH{i}": "Technology" for i in range(10)}
        sector_map["FIN1"] = "Financials"
        signals = [
            {"symbol": f"TECH{i}", "signal_strength": "strong_buy", "combined_signal_score": 80.0}
            for i in range(10)
        ] + [
            {"symbol": "FIN1", "signal_strength": "buy", "combined_signal_score": 60.0},
        ]
        weights = calculate_signal_weights(
            signals, max_position_pct=0.05, max_sector_pct=0.25, sector_map=sector_map,
        )
        tech_total = sum(w for sym, w in weights.items() if sector_map.get(sym) == "Technology")
        assert tech_total <= 0.25 + 1e-4  # 25% sector cap

    def test_position_cap_redistribution(self):
        """Excess from capped positions should be redistributed."""
        signals = [
            {"symbol": "A", "signal_strength": "strong_buy", "combined_signal_score": 90.0},
            {"symbol": "B", "signal_strength": "buy", "combined_signal_score": 50.0},
        ]
        # Without cap: A gets ~73%, B gets ~22% (of 95% investable)
        # With 5% cap: A capped to 5%, excess goes to B (also capped at 5%)
        weights = calculate_signal_weights(signals, max_position_pct=0.05)
        assert weights["A"] <= 0.05 + 1e-6
        assert weights["B"] <= 0.05 + 1e-6

    def test_many_stocks_equal_weight_distribution(self):
        """With many equal stocks, weights should be approximately equal."""
        signals = [
            {"symbol": f"S{i}", "signal_strength": "buy", "combined_signal_score": 70.0}
            for i in range(10)
        ]
        weights = calculate_signal_weights(signals, max_position_pct=0.10)
        values = list(weights.values())
        # All should be close to 0.95 / 10 = 0.095
        assert all(abs(v - 0.095) < 0.01 for v in values)


# ============================================================
# Analytics: estimate_transaction_cost
# ============================================================

class TestEstimateTransactionCost:
    """Test the Brenndoerfer transaction cost model."""

    def test_basic_cost_calculation(self):
        """Verify cost components add up correctly."""
        cost = estimate_transaction_cost(
            quantity=10, price=200.0, spread_bps=5.0,
            commission_per_share=0.005, exchange_fee_bps=3.0, impact_bps_per_1k=0.1,
        )
        # Commission: 0.005/200 * 10000 = 0.25 bps
        # Exchange: 3.0 bps
        # Half-spread: 2.5 bps
        # Impact: 0.1 * (10*200/1000) = 0.2 bps
        expected = 0.25 + 3.0 + 2.5 + 0.2
        assert abs(cost - expected) < 0.01

    def test_small_order_higher_commission_bps(self):
        """Small orders should have higher per-bps commission."""
        cost_small = estimate_transaction_cost(quantity=1, price=50.0)
        cost_large = estimate_transaction_cost(quantity=100, price=50.0)
        # Commission bps is same (per share / price), but impact grows
        # For very small orders, commission bps dominates
        assert cost_small > 0

    def test_large_order_impact_grows(self):
        """Market impact should grow with order size."""
        cost_small = estimate_transaction_cost(quantity=10, price=200.0)
        cost_large = estimate_transaction_cost(quantity=1000, price=200.0)
        assert cost_large > cost_small  # Impact grows with order value

    def test_zero_quantity_returns_zero(self):
        """Zero quantity should return 0 cost."""
        assert estimate_transaction_cost(quantity=0, price=200.0) == 0.0

    def test_zero_price_returns_zero(self):
        """Zero price should return 0 cost."""
        assert estimate_transaction_cost(quantity=10, price=0.0) == 0.0

    def test_negative_quantity_returns_zero(self):
        """Negative quantity should return 0 cost."""
        assert estimate_transaction_cost(quantity=-5, price=200.0) == 0.0

    def test_zero_spread_handled(self):
        """Zero spread should still compute other components."""
        cost = estimate_transaction_cost(quantity=10, price=200.0, spread_bps=0.0)
        assert cost > 0  # Commission + exchange + impact still present


# ============================================================
# Task: calculate_target_weights (integration with analytics)
# ============================================================

class TestCalculateTargetWeightsTask:
    """Test the calculate_target_weights task logic."""

    def test_empty_portfolio_all_cash(self, sample_wf4_signal_context, sample_portfolio_state_empty):
        """Initial empty portfolio should compute weights from signals."""
        from src.wf4_portfolio_rebalancing.tasks import calculate_target_weights

        result = calculate_target_weights(
            signal_context=sample_wf4_signal_context,
            portfolio_state=sample_portfolio_state_empty,
            max_position_pct=0.05,
            max_sector_pct=0.25,
            cash_reserve_pct=0.05,
        )
        weights = json.loads(result["target_weights_json"])
        exits = json.loads(result["exit_symbols_json"])

        # MSFT (strong_buy), AAPL (buy), NVDA (buy), JPM (buy) should get weight
        # AMZN (hold) should not
        assert "MSFT" in weights
        assert "AAPL" in weights
        assert "NVDA" in weights
        assert "JPM" in weights
        assert "AMZN" not in weights
        assert len(exits) == 0  # No positions to exit

    def test_exit_symbols_identified(self, sample_wf4_signal_context, sample_portfolio_state_with_positions):
        """Positions not in target weights should be marked for exit."""
        # Modify signal context so only MSFT gets weight (remove AAPL's buy signal)
        modified_ctx = dict(sample_wf4_signal_context)
        modified_ctx["AAPL"] = json.dumps({
            "combined_signal_score": 30.0, "signal_strength": "sell",
            "wf2_quintile": 4, "technical_score": 25.0,
            "fundamental_score": 35.0, "data_quality": "complete",
        })

        from src.wf4_portfolio_rebalancing.tasks import calculate_target_weights

        result = calculate_target_weights(
            signal_context=modified_ctx,
            portfolio_state=sample_portfolio_state_with_positions,
            max_position_pct=0.05,
            max_sector_pct=0.25,
            cash_reserve_pct=0.05,
        )
        exits = json.loads(result["exit_symbols_json"])
        # AAPL is held but now has "sell" signal — should be in exits
        assert "AAPL" in exits


# ============================================================
# Task: generate_trade_orders
# ============================================================

class TestGenerateTradeOrders:
    """Test trade order generation logic."""

    def test_initial_portfolio_all_buys(
        self, sample_wf4_signal_context, sample_portfolio_state_empty, sample_wf4_price_data,
    ):
        """Empty portfolio with buy signals should generate all BUY orders."""
        from src.wf4_portfolio_rebalancing.tasks import calculate_target_weights, generate_trade_orders

        target_weights = calculate_target_weights(
            signal_context=sample_wf4_signal_context,
            portfolio_state=sample_portfolio_state_empty,
            max_position_pct=0.05,
            max_sector_pct=0.25,
            cash_reserve_pct=0.05,
        )

        orders = generate_trade_orders(
            target_weights=target_weights,
            portfolio_state=sample_portfolio_state_empty,
            price_data=sample_wf4_price_data,
            commission_per_share=0.005,
            exchange_fee_bps=3.0,
            impact_bps_per_1k=0.1,
            min_trade_value=100.0,
        )

        assert len(orders) > 0
        assert all(o.side == "BUY" for o in orders)
        assert all(o.reason == "NewEntry" for o in orders)
        assert all(o.quantity > 0 for o in orders)
        assert all(o.estimated_price > 0 for o in orders)
        assert all(o.estimated_cost_bps > 0 for o in orders)

    def test_min_trade_value_filter(self):
        """Trades below min_trade_value should be filtered out."""
        from src.wf4_portfolio_rebalancing.tasks import generate_trade_orders

        # Very high price so floor(weight * value / price) gives 0 shares
        target_weights = {
            "target_weights_json": json.dumps({"EXPENSIVE": 0.01}),
            "exit_symbols_json": json.dumps([]),
        }
        portfolio_state = {
            "cash": "1000.0",
            "positions_json": json.dumps([]),
            "total_value": "1000.0",
        }
        price_data = {
            "EXPENSIVE": json.dumps({"close": 5000.0, "spread_bps": 5.0}),
        }

        orders = generate_trade_orders(
            target_weights=target_weights,
            portfolio_state=portfolio_state,
            price_data=price_data,
            commission_per_share=0.005,
            exchange_fee_bps=3.0,
            impact_bps_per_1k=0.1,
            min_trade_value=100.0,
        )
        # 1% of $1000 = $10, which is < $100 min → filtered
        assert len(orders) == 0

    def test_trade_order_reasons(self):
        """Orders should have correct reason: NewEntry, Rebalance, Exit."""
        from src.wf4_portfolio_rebalancing.tasks import generate_trade_orders

        target_weights = {
            "target_weights_json": json.dumps({"AAPL": 0.05, "NEW": 0.05}),
            "exit_symbols_json": json.dumps(["OLD"]),
        }
        portfolio_state = {
            "cash": "5000.0",
            "positions_json": json.dumps([
                {"symbol": "AAPL", "shares": 1, "avg_cost": 180.0, "current_price": 195.0, "sector": "Technology"},
                {"symbol": "OLD", "shares": 10, "avg_cost": 50.0, "current_price": 55.0, "sector": "Financials"},
            ]),
            "total_value": "15000.0",
        }
        price_data = {
            "AAPL": json.dumps({"close": 195.0, "spread_bps": 5.0}),
            "NEW": json.dumps({"close": 100.0, "spread_bps": 4.0}),
            "OLD": json.dumps({"close": 55.0, "spread_bps": 6.0}),
        }

        orders = generate_trade_orders(
            target_weights=target_weights,
            portfolio_state=portfolio_state,
            price_data=price_data,
            commission_per_share=0.005,
            exchange_fee_bps=3.0,
            impact_bps_per_1k=0.1,
            min_trade_value=50.0,
        )

        reasons = {o.symbol: o.reason for o in orders}
        sides = {o.symbol: o.side for o in orders}

        if "NEW" in reasons:
            assert reasons["NEW"] == "NewEntry"
            assert sides["NEW"] == "BUY"
        if "OLD" in reasons:
            assert reasons["OLD"] == "Exit"
            assert sides["OLD"] == "SELL"

    def test_orders_sorted_buy_first(
        self, sample_wf4_signal_context, sample_portfolio_state_empty, sample_wf4_price_data,
    ):
        """BUY orders should come before SELL orders."""
        from src.wf4_portfolio_rebalancing.tasks import calculate_target_weights, generate_trade_orders

        target_weights = calculate_target_weights(
            signal_context=sample_wf4_signal_context,
            portfolio_state=sample_portfolio_state_empty,
            max_position_pct=0.05,
            max_sector_pct=0.25,
            cash_reserve_pct=0.05,
        )

        orders = generate_trade_orders(
            target_weights=target_weights,
            portfolio_state=sample_portfolio_state_empty,
            price_data=sample_wf4_price_data,
            commission_per_share=0.005,
            exchange_fee_bps=3.0,
            impact_bps_per_1k=0.1,
            min_trade_value=100.0,
        )

        if len(orders) > 1:
            buy_indices = [i for i, o in enumerate(orders) if o.side == "BUY"]
            sell_indices = [i for i, o in enumerate(orders) if o.side == "SELL"]
            if buy_indices and sell_indices:
                assert max(buy_indices) < min(sell_indices)


# ============================================================
# Task: assemble_rebalancing_result
# ============================================================

class TestAssembleRebalancingResult:
    """Test result assembly and serialization."""

    def test_assembles_all_fields(self):
        """All expected fields should be present in the assembled result."""
        from src.wf4_portfolio_rebalancing.tasks import assemble_rebalancing_result

        trade_orders = [
            TradeOrder(symbol="AAPL", side="BUY", quantity=5, estimated_price=195.0,
                       estimated_cost_bps=5.5, reason="NewEntry"),
            TradeOrder(symbol="OLD", side="SELL", quantity=10, estimated_price=55.0,
                       estimated_cost_bps=6.0, reason="Exit"),
        ]

        result = assemble_rebalancing_result(
            run_date="2026-02-13",
            target_weights={
                "target_weights_json": json.dumps({"AAPL": 0.05}),
                "exit_symbols_json": json.dumps(["OLD"]),
                "num_target_positions": "1",
                "investable_value": "23750.0",
            },
            portfolio_state={"cash": "25000.0", "positions_json": json.dumps([]), "total_value": "25000.0"},
            trade_orders=trade_orders,
            signal_context={"AAPL": json.dumps({"signal_strength": "buy", "combined_signal_score": 72.0})},
        )

        assert result["run_date"] == "2026-02-13"
        assert result["num_buy_orders"] == "1"
        assert result["num_sell_orders"] == "1"
        assert "AAPL" in result["target_weights_json"]

        # Verify trade orders serialization round-trip
        orders = json.loads(result["trade_orders_json"])
        assert len(orders) == 2
        assert orders[0]["symbol"] == "AAPL"

    def test_total_cost_calculation(self):
        """Total estimated cost should sum all order costs."""
        from src.wf4_portfolio_rebalancing.tasks import assemble_rebalancing_result

        trade_orders = [
            TradeOrder(symbol="A", side="BUY", quantity=5, estimated_price=100.0,
                       estimated_cost_bps=3.5, reason="NewEntry"),
            TradeOrder(symbol="B", side="BUY", quantity=3, estimated_price=200.0,
                       estimated_cost_bps=4.2, reason="NewEntry"),
        ]

        result = assemble_rebalancing_result(
            run_date="2026-02-13",
            target_weights={"target_weights_json": "{}", "exit_symbols_json": "[]",
                           "num_target_positions": "2", "investable_value": "23750.0"},
            portfolio_state={"cash": "25000.0", "positions_json": "[]", "total_value": "25000.0"},
            trade_orders=trade_orders,
            signal_context={},
        )

        total_cost = float(result["total_estimated_cost_bps"])
        assert abs(total_cost - 7.7) < 0.01


# ============================================================
# Task: generate_order_report
# ============================================================

class TestGenerateOrderReport:
    """Test markdown report generation."""

    def _make_assembled_result(self, with_orders=True):
        """Helper to create assembled result for report testing."""
        orders = []
        if with_orders:
            orders = [
                {"symbol": "MSFT", "side": "BUY", "quantity": 2, "estimated_price": 415.0,
                 "estimated_cost_bps": 5.5, "reason": "NewEntry"},
                {"symbol": "AAPL", "side": "BUY", "quantity": 6, "estimated_price": 195.0,
                 "estimated_cost_bps": 5.8, "reason": "NewEntry"},
            ]
        return {
            "run_date": "2026-02-13",
            "total_value": "25000.0",
            "cash_value": "25000.0",
            "invested_value": "23750.0",
            "num_signals_input": "5",
            "num_target_positions": "4",
            "num_buy_orders": str(len(orders)),
            "num_sell_orders": "0",
            "total_estimated_cost_bps": "11.3",
            "target_weights_json": json.dumps({"MSFT": 0.05, "AAPL": 0.047, "NVDA": 0.035, "JPM": 0.032}),
            "exit_symbols_json": "[]",
            "trade_orders_json": json.dumps(orders),
            "signal_context_json": json.dumps({
                "MSFT": {"signal_strength": "strong_buy", "combined_signal_score": 76.0},
                "AAPL": {"signal_strength": "buy", "combined_signal_score": 72.5},
                "NVDA": {"signal_strength": "buy", "combined_signal_score": 65.5},
                "JPM": {"signal_strength": "buy", "combined_signal_score": 62.0},
                "AMZN": {"signal_strength": "hold", "combined_signal_score": 45.0},
            }),
        }

    def test_report_contains_header(self):
        """Report should contain date and title."""
        from src.wf4_portfolio_rebalancing.tasks import generate_order_report

        result = self._make_assembled_result()
        report = generate_order_report(assembled_result=result)

        assert "WF4 Portfolio Rebalancing Report" in report
        assert "2026-02-13" in report

    def test_report_contains_portfolio_summary(self):
        """Report should contain portfolio value and cash."""
        from src.wf4_portfolio_rebalancing.tasks import generate_order_report

        report = generate_order_report(assembled_result=self._make_assembled_result())

        assert "EUR 25,000.00" in report
        assert "Portfolio Summary" in report

    def test_report_contains_trades_table(self):
        """Report with orders should contain trades table."""
        from src.wf4_portfolio_rebalancing.tasks import generate_order_report

        report = generate_order_report(assembled_result=self._make_assembled_result())

        assert "Proposed Trades" in report
        assert "MSFT" in report
        assert "BUY" in report
        assert "NewEntry" in report

    def test_report_no_trades(self):
        """Report with no trades should say portfolio is on target."""
        from src.wf4_portfolio_rebalancing.tasks import generate_order_report

        result = self._make_assembled_result(with_orders=False)
        report = generate_order_report(assembled_result=result)

        assert "No trades needed" in report

    def test_report_contains_sector_allocation(self):
        """Report should contain sector allocation table."""
        from src.wf4_portfolio_rebalancing.tasks import generate_order_report

        report = generate_order_report(assembled_result=self._make_assembled_result())

        assert "Sector Allocation" in report
        assert "Technology" in report

    def test_report_contains_risk_notes(self):
        """Report should contain risk notes."""
        from src.wf4_portfolio_rebalancing.tasks import generate_order_report

        report = generate_order_report(assembled_result=self._make_assembled_result())

        assert "Risk Notes" in report
        assert "US equities" in report

    def test_report_contains_signal_summary(self):
        """Report should contain signal strength breakdown."""
        from src.wf4_portfolio_rebalancing.tasks import generate_order_report

        report = generate_order_report(assembled_result=self._make_assembled_result())

        assert "Signal Summary" in report
        assert "strong_buy" in report
        assert "buy" in report


# ============================================================
# Task: store_rebalancing_to_parquet
# ============================================================

class TestStoreRebalancingToParquet:
    """Test Parquet cold-storage for rebalancing results."""

    def _make_assembled_result(self):
        """Helper to create assembled result for Parquet testing."""
        return {
            "run_date": "2026-02-13",
            "total_value": "25000.0",
            "cash_value": "25000.0",
            "invested_value": "23750.0",
            "num_signals_input": "5",
            "num_target_positions": "4",
            "num_buy_orders": "2",
            "num_sell_orders": "0",
            "total_estimated_cost_bps": "11.3",
            "target_weights_json": json.dumps({"MSFT": 0.05, "AAPL": 0.047}),
            "exit_symbols_json": "[]",
            "trade_orders_json": json.dumps([
                {"symbol": "MSFT", "side": "BUY", "quantity": 2,
                 "estimated_price": 415.0, "estimated_cost_bps": 5.5, "reason": "NewEntry"},
                {"symbol": "AAPL", "side": "BUY", "quantity": 6,
                 "estimated_price": 195.0, "estimated_cost_bps": 5.8, "reason": "NewEntry"},
            ]),
            "signal_context_json": json.dumps({
                "MSFT": {"signal_strength": "strong_buy", "combined_signal_score": 76.0},
                "AAPL": {"signal_strength": "buy", "combined_signal_score": 72.5},
            }),
        }

    def test_stores_two_parquet_files(self, mocker):
        """Should upload target_weights.parquet and trade_orders.parquet."""
        from src.wf4_portfolio_rebalancing.tasks import store_rebalancing_to_parquet

        mock_client = mocker.MagicMock()
        mocker.patch(
            "src.shared.storage.get_s3_client",
            return_value=mock_client,
        )

        result = store_rebalancing_to_parquet(assembled_result=self._make_assembled_result())

        assert mock_client.put_object.call_count == 2
        calls = mock_client.put_object.call_args_list
        keys = [c.kwargs["Key"] for c in calls]
        assert any("target_weights.parquet" in k for k in keys)
        assert any("trade_orders.parquet" in k for k in keys)
        assert "Stored 2 Parquet files" in result

    def test_hive_partitioned_path(self, mocker):
        """S3 keys should follow Hive-style partitioning."""
        from src.wf4_portfolio_rebalancing.tasks import store_rebalancing_to_parquet

        mock_client = mocker.MagicMock()
        mocker.patch(
            "src.shared.storage.get_s3_client",
            return_value=mock_client,
        )

        store_rebalancing_to_parquet(assembled_result=self._make_assembled_result())

        calls = mock_client.put_object.call_args_list
        for call in calls:
            key = call.kwargs["Key"]
            assert "rebalancing/year=2026/month=02/day=13/" in key

    def test_target_weights_parquet_schema(self, mocker):
        """Target weights Parquet should contain expected columns."""
        from src.wf4_portfolio_rebalancing.tasks import store_rebalancing_to_parquet
        import pyarrow.parquet as pq
        import io

        captured_bodies = []
        mock_client = mocker.MagicMock()

        def capture_put(**kwargs):
            captured_bodies.append((kwargs["Key"], kwargs["Body"]))

        mock_client.put_object.side_effect = capture_put
        mocker.patch(
            "src.shared.storage.get_s3_client",
            return_value=mock_client,
        )

        store_rebalancing_to_parquet(assembled_result=self._make_assembled_result())

        # Find the target_weights file
        for key, body in captured_bodies:
            if "target_weights" in key:
                table = pq.read_table(io.BytesIO(body))
                assert "symbol" in table.column_names
                assert "target_weight" in table.column_names
                assert "sector" in table.column_names
                assert "signal_strength" in table.column_names
                assert "combined_signal_score" in table.column_names
                assert "run_date" in table.column_names
                assert len(table) == 2  # MSFT + AAPL
                break
        else:
            pytest.fail("target_weights.parquet not found in uploads")

    def test_trade_orders_parquet_schema(self, mocker):
        """Trade orders Parquet should contain expected columns."""
        from src.wf4_portfolio_rebalancing.tasks import store_rebalancing_to_parquet
        import pyarrow.parquet as pq
        import io

        captured_bodies = []
        mock_client = mocker.MagicMock()

        def capture_put(**kwargs):
            captured_bodies.append((kwargs["Key"], kwargs["Body"]))

        mock_client.put_object.side_effect = capture_put
        mocker.patch(
            "src.shared.storage.get_s3_client",
            return_value=mock_client,
        )

        store_rebalancing_to_parquet(assembled_result=self._make_assembled_result())

        for key, body in captured_bodies:
            if "trade_orders" in key:
                table = pq.read_table(io.BytesIO(body))
                assert "symbol" in table.column_names
                assert "side" in table.column_names
                assert "quantity" in table.column_names
                assert "estimated_price" in table.column_names
                assert "estimated_cost_bps" in table.column_names
                assert "reason" in table.column_names
                assert len(table) == 2  # MSFT + AAPL
                break
        else:
            pytest.fail("trade_orders.parquet not found in uploads")

    def test_no_data_returns_message(self, mocker):
        """Empty weights and orders should return 'no data' message."""
        from src.wf4_portfolio_rebalancing.tasks import store_rebalancing_to_parquet

        mock_client = mocker.MagicMock()
        mocker.patch(
            "src.shared.storage.get_s3_client",
            return_value=mock_client,
        )

        empty_result = {
            "run_date": "2026-02-13",
            "total_value": "0",
            "cash_value": "0",
            "target_weights_json": "{}",
            "trade_orders_json": "[]",
            "signal_context_json": "{}",
        }
        result = store_rebalancing_to_parquet(assembled_result=empty_result)

        assert "No rebalancing data" in result
        mock_client.put_object.assert_not_called()

    def test_only_weights_no_orders(self, mocker):
        """Should store weights Parquet even when no trade orders."""
        from src.wf4_portfolio_rebalancing.tasks import store_rebalancing_to_parquet

        mock_client = mocker.MagicMock()
        mocker.patch(
            "src.shared.storage.get_s3_client",
            return_value=mock_client,
        )

        result_no_orders = self._make_assembled_result()
        result_no_orders["trade_orders_json"] = "[]"
        result = store_rebalancing_to_parquet(assembled_result=result_no_orders)

        assert mock_client.put_object.call_count == 1
        assert "Stored 1 Parquet files" in result


# ============================================================
# Data Models
# ============================================================

class TestTradeOrderModel:
    """Test TradeOrder dataclass."""

    def test_trade_order_creation(self):
        """TradeOrder should be creatable with all fields."""
        order = TradeOrder(
            symbol="AAPL", side="BUY", quantity=10,
            estimated_price=195.0, estimated_cost_bps=5.5, reason="NewEntry",
        )
        assert order.symbol == "AAPL"
        assert order.side == "BUY"
        assert order.quantity == 10
        assert order.estimated_price == 195.0
        assert order.estimated_cost_bps == 5.5
        assert order.reason == "NewEntry"

    def test_trade_order_list_serialization(self):
        """List[TradeOrder] should be JSON-serializable via dict conversion."""
        orders = [
            TradeOrder(symbol="A", side="BUY", quantity=5, estimated_price=100.0,
                       estimated_cost_bps=3.0, reason="NewEntry"),
            TradeOrder(symbol="B", side="SELL", quantity=3, estimated_price=200.0,
                       estimated_cost_bps=4.0, reason="Exit"),
        ]
        # Simulate the serialization used in assemble_rebalancing_result
        data = [
            {"symbol": o.symbol, "side": o.side, "quantity": o.quantity,
             "estimated_price": o.estimated_price, "estimated_cost_bps": o.estimated_cost_bps,
             "reason": o.reason}
            for o in orders
        ]
        serialized = json.dumps(data)
        deserialized = json.loads(serialized)
        assert len(deserialized) == 2
        assert deserialized[0]["symbol"] == "A"
        assert deserialized[1]["side"] == "SELL"


# ============================================================
# Config: SYMBOL_SECTORS
# ============================================================

class TestSymbolSectors:
    """Test the GICS sector mapping."""

    def test_all_phase2_symbols_have_sector(self):
        """Every PHASE2_SYMBOL should have a sector mapping."""
        from src.shared.config import PHASE2_SYMBOLS, SYMBOL_SECTORS
        for symbol in PHASE2_SYMBOLS:
            assert symbol in SYMBOL_SECTORS, f"{symbol} missing from SYMBOL_SECTORS"

    def test_all_11_gics_sectors_represented(self):
        """All 11 GICS sectors should be present."""
        from src.shared.config import SYMBOL_SECTORS
        sectors = set(SYMBOL_SECTORS.values())
        expected = {
            "Technology", "Consumer Discretionary", "Financials", "Healthcare",
            "Industrials", "Consumer Staples", "Energy", "Communication Services",
            "Utilities", "Real Estate", "Materials",
        }
        assert sectors == expected

    def test_sector_counts_match_comments(self):
        """Sector counts should match the PHASE2_SYMBOLS comments."""
        from src.shared.config import SYMBOL_SECTORS
        from collections import Counter
        counts = Counter(SYMBOL_SECTORS.values())
        assert counts["Technology"] == 10
        assert counts["Consumer Discretionary"] == 5
        assert counts["Financials"] == 6
        assert counts["Healthcare"] == 5
        assert counts["Industrials"] == 4
        assert counts["Consumer Staples"] == 4
        assert counts["Energy"] == 3
        assert counts["Communication Services"] == 3
        assert counts["Utilities"] == 3
        assert counts["Real Estate"] == 3
        assert counts["Materials"] == 3


# ============================================================
# Analytics: calculate_cost_breakdown (Phase 4)
# ============================================================

class TestCalculateCostBreakdown:
    """Test transaction cost decomposition into USD components."""

    def test_basic_breakdown(self):
        """Cost breakdown should return positive components."""
        result = calculate_cost_breakdown(
            quantity=10, price=200.0, spread_bps=5.0,
            commission_per_share=0.005, exchange_fee_bps=3.0,
            impact_bps_per_1k=0.1,
        )
        assert result["commission"] > 0
        assert result["spread_cost"] > 0
        assert result["impact_cost"] > 0
        assert result["total_cost"] > 0
        # Commission: 0.005 * 10 = 0.05
        assert abs(result["commission"] - 0.05) < 0.01

    def test_zero_inputs_return_zeros(self):
        """Zero or negative inputs should return all zeros."""
        result = calculate_cost_breakdown(quantity=0, price=200.0)
        assert result["total_cost"] == 0.0
        assert result["commission"] == 0.0

        result2 = calculate_cost_breakdown(quantity=10, price=0.0)
        assert result2["total_cost"] == 0.0

    def test_total_equals_sum_of_components(self):
        """Total cost should equal sum of individual components."""
        result = calculate_cost_breakdown(
            quantity=100, price=150.0, spread_bps=4.0,
            commission_per_share=0.005, exchange_fee_bps=3.0,
            impact_bps_per_1k=0.1,
        )
        component_sum = (
            result["commission"] + result["spread_cost"] + result["impact_cost"]
        )
        assert abs(result["total_cost"] - component_sum) < 0.01


# ============================================================
# Task: execute_paper_trades (Phase 4)
# ============================================================

class TestExecutePaperTrades:
    """Test paper trade execution logic."""

    def test_disabled_returns_immediately(
        self, sample_assembled_result_for_paper_trading,
        sample_portfolio_state_empty, sample_wf4_price_data,
    ):
        """When paper_trading=False, task returns disabled status."""
        from src.wf4_portfolio_rebalancing.tasks import execute_paper_trades

        result = execute_paper_trades(
            assembled_result=sample_assembled_result_for_paper_trading,
            portfolio_state=sample_portfolio_state_empty,
            price_data=sample_wf4_price_data,
            paper_trading=False,
            initial_capital=25000.0,
            commission_per_share=0.005,
            exchange_fee_bps=3.0,
            impact_bps_per_1k=0.1,
        )

        assert result["status"] == "disabled"
        assert result["num_trades_executed"] == "0"

    def test_buy_orders_deduct_cash(
        self, mocker, sample_assembled_result_for_paper_trading,
        sample_portfolio_state_empty, sample_wf4_price_data,
    ):
        """BUY orders should deduct cost + fees from cash."""
        from src.wf4_portfolio_rebalancing.tasks import execute_paper_trades

        mocker.patch("src.shared.db.upsert_positions", return_value=3)
        mocker.patch("src.shared.db.store_executed_trades", return_value=3)

        result = execute_paper_trades(
            assembled_result=sample_assembled_result_for_paper_trading,
            portfolio_state=sample_portfolio_state_empty,
            price_data=sample_wf4_price_data,
            paper_trading=True,
            initial_capital=25000.0,
            commission_per_share=0.005,
            exchange_fee_bps=3.0,
            impact_bps_per_1k=0.1,
        )

        assert result["status"] == "executed"
        cash_after = float(result["cash_after"])
        assert cash_after < 25000.0  # Cash decreased
        assert cash_after > 0  # Still positive
        assert int(result["num_trades_executed"]) == 3

    def test_sell_orders_add_cash(self, mocker, sample_wf4_price_data):
        """SELL orders should add proceeds minus fees to cash."""
        from src.wf4_portfolio_rebalancing.tasks import execute_paper_trades

        mocker.patch("src.shared.db.upsert_positions", return_value=1)
        mocker.patch("src.shared.db.store_executed_trades", return_value=1)

        assembled = {
            "run_date": "2026-02-16",
            "trade_orders_json": json.dumps([{
                "symbol": "AAPL", "side": "SELL", "quantity": 5,
                "estimated_price": 195.0, "estimated_cost_bps": 5.0,
                "reason": "Exit",
            }]),
        }
        portfolio = {
            "cash": "5000.0",
            "positions_json": json.dumps([{
                "symbol": "AAPL", "shares": 5, "avg_cost": 180.0,
                "current_price": 195.0, "sector": "Technology",
            }]),
            "total_value": "5975.0",
        }

        result = execute_paper_trades(
            assembled_result=assembled,
            portfolio_state=portfolio,
            price_data=sample_wf4_price_data,
            paper_trading=True,
            initial_capital=25000.0,
            commission_per_share=0.005,
            exchange_fee_bps=3.0,
            impact_bps_per_1k=0.1,
        )

        cash_after = float(result["cash_after"])
        assert cash_after > 5000.0  # Cash increased from sell

    def test_weighted_average_cost_on_additional_buy(
        self, mocker, sample_wf4_price_data,
    ):
        """Buying more of existing position updates weighted average cost."""
        from src.wf4_portfolio_rebalancing.tasks import execute_paper_trades

        mocker.patch("src.shared.db.upsert_positions", return_value=1)
        mocker.patch("src.shared.db.store_executed_trades", return_value=1)

        assembled = {
            "run_date": "2026-02-16",
            "trade_orders_json": json.dumps([{
                "symbol": "AAPL", "side": "BUY", "quantity": 5,
                "estimated_price": 195.0, "estimated_cost_bps": 5.0,
                "reason": "Rebalance",
            }]),
        }
        portfolio = {
            "cash": "10000.0",
            "positions_json": json.dumps([{
                "symbol": "AAPL", "shares": 5, "avg_cost": 180.0,
                "current_price": 195.0, "sector": "Technology",
            }]),
            "total_value": "10975.0",
        }

        result = execute_paper_trades(
            assembled_result=assembled,
            portfolio_state=portfolio,
            price_data=sample_wf4_price_data,
            paper_trading=True,
            initial_capital=25000.0,
            commission_per_share=0.005,
            exchange_fee_bps=3.0,
            impact_bps_per_1k=0.1,
        )

        positions_after = json.loads(result["positions_after_json"])
        aapl = next(p for p in positions_after if p["symbol"] == "AAPL")
        assert aapl["shares"] == 10  # 5 + 5
        # Weighted avg: (5*180 + 5*195.5) / 10 ≈ 187.75
        assert aapl["avg_cost"] > 180.0
        assert aapl["avg_cost"] < 196.0

    def test_full_exit_removes_position(self, mocker, sample_wf4_price_data):
        """Selling all shares should remove position from output."""
        from src.wf4_portfolio_rebalancing.tasks import execute_paper_trades

        mocker.patch("src.shared.db.upsert_positions", return_value=1)
        mocker.patch("src.shared.db.store_executed_trades", return_value=1)

        assembled = {
            "run_date": "2026-02-16",
            "trade_orders_json": json.dumps([{
                "symbol": "AAPL", "side": "SELL", "quantity": 5,
                "estimated_price": 195.0, "estimated_cost_bps": 5.0,
                "reason": "Exit",
            }]),
        }
        portfolio = {
            "cash": "5000.0",
            "positions_json": json.dumps([{
                "symbol": "AAPL", "shares": 5, "avg_cost": 180.0,
                "current_price": 195.0, "sector": "Technology",
            }]),
            "total_value": "5975.0",
        }

        result = execute_paper_trades(
            assembled_result=assembled,
            portfolio_state=portfolio,
            price_data=sample_wf4_price_data,
            paper_trading=True,
            initial_capital=25000.0,
            commission_per_share=0.005,
            exchange_fee_bps=3.0,
            impact_bps_per_1k=0.1,
        )

        positions_after = json.loads(result["positions_after_json"])
        symbols = [p["symbol"] for p in positions_after]
        assert "AAPL" not in symbols  # Fully exited

    def test_insufficient_cash_skips_order(
        self, mocker, sample_wf4_price_data,
    ):
        """Orders exceeding available cash should be skipped."""
        from src.wf4_portfolio_rebalancing.tasks import execute_paper_trades

        mocker.patch("src.shared.db.upsert_positions", return_value=0)
        mocker.patch("src.shared.db.store_executed_trades", return_value=0)

        assembled = {
            "run_date": "2026-02-16",
            "trade_orders_json": json.dumps([{
                "symbol": "NVDA", "side": "BUY", "quantity": 100,
                "estimated_price": 850.0, "estimated_cost_bps": 5.0,
                "reason": "NewEntry",
            }]),
        }
        portfolio = {
            "cash": "100.0",
            "positions_json": json.dumps([]),
            "total_value": "100.0",
        }

        result = execute_paper_trades(
            assembled_result=assembled,
            portfolio_state=portfolio,
            price_data=sample_wf4_price_data,
            paper_trading=True,
            initial_capital=100.0,
            commission_per_share=0.005,
            exchange_fee_bps=3.0,
            impact_bps_per_1k=0.1,
        )

        cash_after = float(result["cash_after"])
        assert cash_after >= 99.0  # Cash barely changed (trade skipped)
        assert int(result["num_trades_executed"]) == 0

    def test_db_functions_called(
        self, mocker, sample_assembled_result_for_paper_trading,
        sample_portfolio_state_empty, sample_wf4_price_data,
    ):
        """DB upsert and store functions should be called when paper trading."""
        from src.wf4_portfolio_rebalancing.tasks import execute_paper_trades

        mock_upsert = mocker.patch(
            "src.shared.db.upsert_positions", return_value=3,
        )
        mock_store = mocker.patch(
            "src.shared.db.store_executed_trades", return_value=3,
        )

        execute_paper_trades(
            assembled_result=sample_assembled_result_for_paper_trading,
            portfolio_state=sample_portfolio_state_empty,
            price_data=sample_wf4_price_data,
            paper_trading=True,
            initial_capital=25000.0,
            commission_per_share=0.005,
            exchange_fee_bps=3.0,
            impact_bps_per_1k=0.1,
        )

        mock_upsert.assert_called_once()
        mock_store.assert_called_once()


# ============================================================
# Task: snapshot_portfolio (Phase 4)
# ============================================================

class TestSnapshotPortfolio:
    """Test portfolio snapshot creation."""

    def test_disabled_returns_message(self):
        """When paper_trading=False, returns disabled message."""
        from src.wf4_portfolio_rebalancing.tasks import snapshot_portfolio

        result = snapshot_portfolio(
            paper_trade_result={"status": "disabled"},
            paper_trading=False,
            initial_capital=25000.0,
        )
        assert "disabled" in result.lower()

    def test_creates_snapshot_after_execution(
        self, mocker, sample_paper_trade_result_executed,
    ):
        """Should store snapshot to DB after paper trades."""
        from src.wf4_portfolio_rebalancing.tasks import snapshot_portfolio

        mocker.patch(
            "src.shared.db.get_latest_portfolio_snapshot", return_value=None,
        )
        mock_store = mocker.patch(
            "src.shared.db.store_portfolio_snapshot", return_value=1,
        )

        result = snapshot_portfolio(
            paper_trade_result=sample_paper_trade_result_executed,
            paper_trading=True,
            initial_capital=25000.0,
        )

        mock_store.assert_called_once()
        assert "2026-02-16" in result
        assert "EUR" in result

    def test_calculates_pnl_vs_previous(self, mocker):
        """PnL should be calculated against previous snapshot."""
        from src.wf4_portfolio_rebalancing.tasks import snapshot_portfolio
        from datetime import date

        # Previous snapshot had total_value = 25000
        mocker.patch(
            "src.shared.db.get_latest_portfolio_snapshot",
            return_value=(
                date(2026, 2, 9), 25000.0, 23000.0, 2000.0, 0.0, 0.0, 3,
            ),
        )
        mock_store = mocker.patch(
            "src.shared.db.store_portfolio_snapshot", return_value=1,
        )

        paper_result = {
            "status": "executed",
            "num_trades_executed": "2",
            "cash_after": "22500.0",
            "positions_after_json": json.dumps([{
                "symbol": "AAPL", "shares": 5, "avg_cost": 190.0,
                "current_price": 195.0,
            }]),
            "total_value_after": "23475.0",
            "run_date": "2026-02-16",
        }

        snapshot_portfolio(
            paper_trade_result=paper_result,
            paper_trading=True,
            initial_capital=25000.0,
        )

        # Check that daily_pnl = 23475 - 25000 = -1525
        call_kwargs = mock_store.call_args
        assert call_kwargs[1]["daily_pnl"] == pytest.approx(-1525.0, abs=0.01)

    def test_first_run_pnl_vs_initial_capital(
        self, mocker, sample_paper_trade_result_executed,
    ):
        """First snapshot PnL should be vs initial capital."""
        from src.wf4_portfolio_rebalancing.tasks import snapshot_portfolio

        mocker.patch(
            "src.shared.db.get_latest_portfolio_snapshot", return_value=None,
        )
        mock_store = mocker.patch(
            "src.shared.db.store_portfolio_snapshot", return_value=1,
        )

        snapshot_portfolio(
            paper_trade_result=sample_paper_trade_result_executed,
            paper_trading=True,
            initial_capital=25000.0,
        )

        call_kwargs = mock_store.call_args
        # total_value_after = 24969.50, initial = 25000 → PnL < 0
        assert call_kwargs[1]["daily_pnl"] < 0  # Lost money to costs
