"""Tests for shared data models."""

from src.shared.models import MarketDataBatch, Position, TradeOrder, DailyReport


def test_market_data_batch_creation(sample_market_data_batch):
    """Test MarketDataBatch creation with all fields."""
    batch = sample_market_data_batch
    assert len(batch.symbols) == 3
    assert batch.date == "2026-01-15"
    assert batch.prices["AAPL"] == 195.50
    assert batch.volumes["MSFT"] == 25_000_000
    assert batch.data_quality_score == 1.0


def test_market_data_batch_empty():
    """Test MarketDataBatch with no data."""
    batch = MarketDataBatch(
        symbols=[], date="2026-01-15",
        prices={}, volumes={}, spreads={}, market_caps={},
        data_quality_score=0.0,
    )
    assert len(batch.symbols) == 0
    assert batch.data_quality_score == 0.0


def test_position_creation():
    """Test Position dataclass with defaults."""
    pos = Position(
        symbol="AAPL", shares=100.0,
        avg_cost=150.0, current_price=195.50,
    )
    assert pos.symbol == "AAPL"
    assert pos.shares == 100.0
    assert pos.region == ""   # default
    assert pos.sector == ""   # default


def test_position_with_metadata():
    """Test Position with region and sector."""
    pos = Position(
        symbol="AAPL", shares=100.0,
        avg_cost=150.0, current_price=195.50,
        region="US", sector="Technology",
    )
    assert pos.region == "US"
    assert pos.sector == "Technology"


def test_trade_order_creation():
    """Test TradeOrder dataclass."""
    order = TradeOrder(
        symbol="AAPL", side="BUY", quantity=50,
        estimated_price=195.50, estimated_cost_bps=8.5,
        reason="NewEntry",
    )
    assert order.side == "BUY"
    assert order.quantity == 50
    assert order.reason == "NewEntry"


def test_daily_report_defaults():
    """Test DailyReport with default list fields."""
    report = DailyReport(
        date="2026-01-15", portfolio_value=100_000.0,
        daily_pnl=500.0, daily_pnl_pct=0.5,
        mtd_pnl=1_500.0, ytd_pnl=3_000.0,
        sharpe_ratio_30d=1.5, max_drawdown_30d=-2.0,
        var_95=-1_500.0,
    )
    assert report.top_winners == []
    assert report.top_losers == []
    assert report.upcoming_dividends == []
    assert report.alerts == []


def test_daily_report_with_data():
    """Test DailyReport with populated lists."""
    report = DailyReport(
        date="2026-01-15", portfolio_value=100_000.0,
        daily_pnl=500.0, daily_pnl_pct=0.5,
        mtd_pnl=1_500.0, ytd_pnl=3_000.0,
        sharpe_ratio_30d=1.5, max_drawdown_30d=-2.0,
        var_95=-1_500.0,
        top_winners=["AAPL", "NVDA"],
        alerts=["Drawdown > 5%"],
    )
    assert len(report.top_winners) == 2
    assert "Drawdown > 5%" in report.alerts
