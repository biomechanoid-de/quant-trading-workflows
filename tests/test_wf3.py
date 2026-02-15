"""Tests for WF3: Signal & Analysis.

Tests all analytics functions (SMA, MACD, Bollinger, fundamentals),
task helper functions, and the combine/assemble/report pipeline.
No database or network access required — all data from fixtures.

Coverage targets: ~50 tests for WF3 components.
"""

import json
import math
import pytest

from src.shared.analytics import (
    calculate_sma,
    calculate_sma_crossover_signal,
    _ema,
    calculate_macd,
    classify_macd_signal,
    calculate_bollinger_bands,
    classify_bollinger_signal,
    normalize_pe_ratio,
    compute_fundamental_score,
    classify_value_signal,
)
from src.shared.models import (
    TechnicalSignals,
    FundamentalSignals,
    SignalResult,
    SignalAnalysisResult,
)
from src.wf3_signal_analysis.tasks import (
    _compute_technical_score,
    _classify_signal_strength,
    _is_nan,
    combine_signals,
    assemble_signal_result,
    generate_signal_report,
)


# ============================================================
# SMA Tests
# ============================================================

class TestCalculateSMA:
    """Tests for calculate_sma."""

    def test_basic_sma(self):
        """SMA of [1, 2, 3, 4, 5] with window 3 should be [nan, nan, 2, 3, 4]."""
        result = calculate_sma([1, 2, 3, 4, 5], 3)
        assert len(result) == 5
        assert math.isnan(result[0])
        assert math.isnan(result[1])
        assert result[2] == pytest.approx(2.0)
        assert result[3] == pytest.approx(3.0)
        assert result[4] == pytest.approx(4.0)

    def test_sma_window_equals_length(self):
        """SMA where window = data length should have one valid value."""
        result = calculate_sma([10, 20, 30], 3)
        assert len(result) == 3
        assert math.isnan(result[0])
        assert math.isnan(result[1])
        assert result[2] == pytest.approx(20.0)

    def test_sma_window_1(self):
        """SMA with window 1 should return the prices themselves."""
        prices = [5.0, 10.0, 15.0]
        result = calculate_sma(prices, 1)
        assert result == pytest.approx(prices)

    def test_sma_empty_prices(self):
        """Empty prices should return empty list."""
        assert calculate_sma([], 5) == []

    def test_sma_zero_window(self):
        """Zero window should return empty list."""
        assert calculate_sma([1, 2, 3], 0) == []

    def test_sma_constant_prices(self):
        """SMA of constant prices should equal the constant."""
        result = calculate_sma([100.0] * 10, 5)
        for val in result[4:]:
            assert val == pytest.approx(100.0)


class TestSMACrossoverSignal:
    """Tests for calculate_sma_crossover_signal."""

    def test_insufficient_data(self):
        """Less than long_window data points should return neutral."""
        result = calculate_sma_crossover_signal([100.0] * 50, 50, 200)
        assert result == "neutral"

    def test_bullish_short_above_long(self):
        """Prices trending up → short SMA above long SMA → bullish."""
        # Create upward trending prices (250 points)
        prices = [100.0 + i * 0.5 for i in range(250)]
        result = calculate_sma_crossover_signal(prices, 50, 200)
        assert result == "bullish"

    def test_bearish_short_below_long(self):
        """Prices trending down → short SMA below long SMA → bearish."""
        # Create downward trending prices
        prices = [200.0 - i * 0.5 for i in range(250)]
        result = calculate_sma_crossover_signal(prices, 50, 200)
        assert result == "bearish"

    def test_neutral_flat_prices(self):
        """Constant prices should converge SMAs → neutral or close."""
        prices = [100.0] * 250
        result = calculate_sma_crossover_signal(prices, 50, 200)
        assert result == "neutral"


# ============================================================
# MACD Tests
# ============================================================

class TestCalculateMACD:
    """Tests for calculate_macd."""

    def test_insufficient_data(self):
        """Too few data points should return zeros."""
        result = calculate_macd([100.0] * 10)
        assert result == (0.0, 0.0, 0.0)

    def test_basic_macd(self):
        """MACD with enough data should return non-zero values."""
        # Create 50 data points with a trend
        prices = [100.0 + i * 0.3 for i in range(50)]
        macd_line, signal_line, histogram = calculate_macd(prices)
        # Uptrend: fast EMA > slow EMA → positive MACD
        assert macd_line > 0
        assert isinstance(signal_line, float)
        assert isinstance(histogram, float)

    def test_macd_constant_prices(self):
        """Constant prices should give MACD close to zero."""
        prices = [150.0] * 50
        macd_line, signal_line, histogram = calculate_macd(prices)
        assert abs(macd_line) < 0.01
        assert abs(signal_line) < 0.01
        assert abs(histogram) < 0.01

    def test_macd_downtrend(self):
        """Downtrending prices should give negative MACD."""
        prices = [200.0 - i * 0.5 for i in range(50)]
        macd_line, signal_line, histogram = calculate_macd(prices)
        assert macd_line < 0


class TestClassifyMACDSignal:
    """Tests for classify_macd_signal."""

    def test_bullish(self):
        """MACD > signal and histogram > 0 → bullish."""
        assert classify_macd_signal(1.5, 1.0, 0.5) == "bullish"

    def test_bearish(self):
        """MACD < signal and histogram < 0 → bearish."""
        assert classify_macd_signal(-1.5, -1.0, -0.5) == "bearish"

    def test_neutral_mixed(self):
        """MACD > signal but histogram ≤ 0 → neutral."""
        assert classify_macd_signal(1.0, 0.5, 0.0) == "neutral"

    def test_neutral_zero(self):
        """All zeros → neutral."""
        assert classify_macd_signal(0.0, 0.0, 0.0) == "neutral"


# ============================================================
# EMA Tests
# ============================================================

class TestEMA:
    """Tests for _ema helper."""

    def test_ema_empty(self):
        """Empty input should return empty list."""
        assert _ema([], 12) == []

    def test_ema_single_value(self):
        """Single value should return that value."""
        assert _ema([100.0], 12) == [100.0]

    def test_ema_converges(self):
        """EMA of constant values should converge to that constant."""
        result = _ema([50.0] * 30, 12)
        assert result[-1] == pytest.approx(50.0)


# ============================================================
# Bollinger Bands Tests
# ============================================================

class TestBollingerBands:
    """Tests for calculate_bollinger_bands."""

    def test_insufficient_data(self):
        """Less than window data → zeros."""
        assert calculate_bollinger_bands([100.0] * 5, 20) == (0.0, 0.0, 0.0)

    def test_basic_bollinger(self):
        """Basic bands: upper > middle > lower."""
        prices = [100.0 + i * 0.1 for i in range(30)]
        upper, middle, lower = calculate_bollinger_bands(prices, 20)
        assert upper > middle > lower
        assert middle > 0

    def test_constant_prices_zero_width(self):
        """Constant prices → zero standard deviation → bands equal middle."""
        prices = [100.0] * 25
        upper, middle, lower = calculate_bollinger_bands(prices, 20)
        assert upper == pytest.approx(middle)
        assert lower == pytest.approx(middle)
        assert middle == pytest.approx(100.0)

    def test_band_width_increases_with_volatility(self):
        """More volatile prices → wider bands."""
        import random
        random.seed(42)
        # Low vol
        low_vol = [100.0 + random.gauss(0, 0.5) for _ in range(30)]
        u_low, m_low, l_low = calculate_bollinger_bands(low_vol, 20)
        # High vol
        high_vol = [100.0 + random.gauss(0, 5.0) for _ in range(30)]
        u_high, m_high, l_high = calculate_bollinger_bands(high_vol, 20)

        width_low = u_low - l_low
        width_high = u_high - l_high
        assert width_high > width_low


class TestClassifyBollingerSignal:
    """Tests for classify_bollinger_signal."""

    def test_oversold(self):
        """Price at or below lower band → oversold."""
        assert classify_bollinger_signal(95.0, 110.0, 100.0, 105.0) == "oversold"

    def test_overbought(self):
        """Price at or above upper band → overbought."""
        assert classify_bollinger_signal(115.0, 110.0, 100.0, 105.0) == "overbought"

    def test_neutral(self):
        """Price between bands → neutral."""
        assert classify_bollinger_signal(105.0, 110.0, 100.0, 105.0) == "neutral"

    def test_zero_bands(self):
        """Zero bands → neutral."""
        assert classify_bollinger_signal(100.0, 0.0, 0.0, 0.0) == "neutral"

    def test_at_lower_bound(self):
        """Price exactly at lower band → oversold."""
        assert classify_bollinger_signal(100.0, 110.0, 100.0, 105.0) == "oversold"


# ============================================================
# Fundamental Analysis Tests
# ============================================================

class TestNormalizePERatio:
    """Tests for normalize_pe_ratio."""

    def test_low_pe_positive_zscore(self):
        """P/E below sector median → positive z-score (undervalued)."""
        result = normalize_pe_ratio(10.0, 20.0)
        assert result > 0
        assert result == pytest.approx(0.5)

    def test_high_pe_negative_zscore(self):
        """P/E above sector median → negative z-score (overvalued)."""
        result = normalize_pe_ratio(40.0, 20.0)
        assert result < 0
        assert result == pytest.approx(-1.0)

    def test_equal_pe_zero_zscore(self):
        """P/E equal to sector median → zero z-score."""
        result = normalize_pe_ratio(20.0, 20.0)
        assert result == pytest.approx(0.0)

    def test_missing_pe_returns_zero(self):
        """Missing P/E (-1.0) → zero z-score."""
        assert normalize_pe_ratio(-1.0) == 0.0
        assert normalize_pe_ratio(0.0) == 0.0

    def test_negative_pe_returns_zero(self):
        """Negative P/E (loss-making) → zero z-score."""
        assert normalize_pe_ratio(-5.0) == 0.0


class TestComputeFundamentalScore:
    """Tests for compute_fundamental_score."""

    def test_score_in_range(self):
        """Score should always be between 0 and 100."""
        score = compute_fundamental_score(0.0, 0.02, 0.15, 0.5, 1.5)
        assert 0 <= score <= 100

    def test_all_missing_returns_neutral(self):
        """All missing data → neutral ~50 score."""
        score = compute_fundamental_score(0.0, -1.0, -1.0, -1.0, -1.0)
        assert 40 <= score <= 60  # Should be roughly neutral

    def test_excellent_fundamentals(self):
        """Strong fundamentals → high score."""
        score = compute_fundamental_score(0.5, 0.04, 0.30, 0.3, 1.5)
        assert score > 60

    def test_poor_fundamentals(self):
        """Weak fundamentals → low score."""
        score = compute_fundamental_score(-1.5, 0.0, -0.05, 3.0, 0.5)
        assert score < 40


class TestClassifyValueSignal:
    """Tests for classify_value_signal."""

    def test_value_stock(self):
        """Low P/E + decent dividend → value."""
        result = classify_value_signal(0.5, 0.03, 0.15, 0.5)
        assert result == "value"

    def test_growth_stock(self):
        """High P/E + high ROE + no dividend → growth."""
        result = classify_value_signal(-0.5, 0.001, 0.30, 0.5)
        assert result == "growth"

    def test_balanced_stock(self):
        """Mixed characteristics → balanced."""
        result = classify_value_signal(0.1, 0.005, 0.10, 0.8)
        assert result == "balanced"

    def test_missing_roe(self):
        """Missing ROE with low PE + dividend → still classifiable."""
        result = classify_value_signal(0.5, 0.03, -1.0, -1.0)
        assert result == "value"


# ============================================================
# Task Helper Tests
# ============================================================

class TestComputeTechnicalScore:
    """Tests for _compute_technical_score."""

    def test_all_bullish(self):
        """All bullish signals → high score."""
        score = _compute_technical_score("bullish", "bullish", "oversold")
        assert score == pytest.approx(80.0)

    def test_all_bearish(self):
        """All bearish signals → low score."""
        score = _compute_technical_score("bearish", "bearish", "overbought")
        assert score == pytest.approx(20.0)

    def test_all_neutral(self):
        """All neutral signals → mid score."""
        score = _compute_technical_score("neutral", "neutral", "neutral")
        assert score == pytest.approx(50.0)

    def test_mixed_signals(self):
        """Mixed signals → intermediate score."""
        score = _compute_technical_score("bullish", "bearish", "neutral")
        # 80*0.40 + 20*0.35 + 50*0.25 = 32 + 7 + 12.5 = 51.5
        assert score == pytest.approx(51.5)


class TestClassifySignalStrength:
    """Tests for _classify_signal_strength."""

    def test_strong_buy(self):
        assert _classify_signal_strength(80.0) == "strong_buy"
        assert _classify_signal_strength(75.0) == "strong_buy"

    def test_buy(self):
        assert _classify_signal_strength(65.0) == "buy"
        assert _classify_signal_strength(60.0) == "buy"

    def test_hold(self):
        assert _classify_signal_strength(50.0) == "hold"
        assert _classify_signal_strength(40.0) == "hold"

    def test_sell(self):
        assert _classify_signal_strength(30.0) == "sell"
        assert _classify_signal_strength(25.0) == "sell"

    def test_strong_sell(self):
        assert _classify_signal_strength(20.0) == "strong_sell"
        assert _classify_signal_strength(0.0) == "strong_sell"


class TestIsNan:
    """Tests for _is_nan helper."""

    def test_nan(self):
        assert _is_nan(float("nan")) is True

    def test_not_nan(self):
        assert _is_nan(42.0) is False
        assert _is_nan(0.0) is False


# ============================================================
# Combine Signals Tests
# ============================================================

class TestCombineSignals:
    """Tests for combine_signals task (called directly, not via Flyte)."""

    def test_combine_basic(
        self,
        sample_screening_context,
        sample_technical_signals,
        sample_fundamental_signals,
    ):
        """Combine 5 signals with Phase 2 weights (no sentiment)."""
        results = combine_signals(
            screening_context=sample_screening_context,
            tech_signals=sample_technical_signals,
            fund_signals=sample_fundamental_signals,
            sent_signals=[],
            tech_weight=0.5,
            fund_weight=0.5,
            sent_weight=0.0,
            run_date="2026-02-09",
        )
        assert len(results) == 5
        # Should be sorted by combined score descending
        scores = [r.combined_signal_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_combine_signal_fields(
        self,
        sample_screening_context,
        sample_technical_signals,
        sample_fundamental_signals,
    ):
        """Verify signal result fields are populated correctly."""
        results = combine_signals(
            screening_context=sample_screening_context,
            tech_signals=sample_technical_signals,
            fund_signals=sample_fundamental_signals,
            sent_signals=[],
            tech_weight=0.5,
            fund_weight=0.5,
            sent_weight=0.0,
            run_date="2026-02-09",
        )
        for r in results:
            assert r.run_date == "2026-02-09"
            assert r.signal_strength in ("strong_buy", "buy", "hold", "sell", "strong_sell")
            assert r.data_quality in ("complete", "partial", "minimal")
            assert 0 <= r.combined_signal_score <= 100

    def test_combine_empty_context(self):
        """Empty screening context → empty results."""
        results = combine_signals(
            screening_context={},
            tech_signals=[],
            fund_signals=[],
            sent_signals=[],
            tech_weight=0.5,
            fund_weight=0.5,
            sent_weight=0.0,
            run_date="2026-02-09",
        )
        assert results == []

    def test_combine_missing_tech(
        self,
        sample_screening_context,
        sample_fundamental_signals,
    ):
        """Missing technical signals → uses neutral default (50)."""
        results = combine_signals(
            screening_context=sample_screening_context,
            tech_signals=[],
            fund_signals=sample_fundamental_signals,
            sent_signals=[],
            tech_weight=0.5,
            fund_weight=0.5,
            sent_weight=0.0,
            run_date="2026-02-09",
        )
        assert len(results) == 5
        # Technical score should be neutral (50) for all
        for r in results:
            assert r.technical_score == 50.0

    def test_combine_with_sentiment_three_way(
        self,
        sample_screening_context,
        sample_technical_signals,
        sample_fundamental_signals,
    ):
        """Phase 6: Combine with 30/40/30 weights including sentiment."""
        from src.shared.models import SentimentSignals

        sent_signals = [
            SentimentSignals(
                symbol=symbol, num_articles=5, num_positive=3, num_neutral=1,
                num_negative=1, news_provider="finnhub", sentiment_score=70.0,
                sentiment_signal="very_positive", has_sentiment=True,
            )
            for symbol in sample_screening_context.keys()
        ]

        results = combine_signals(
            screening_context=sample_screening_context,
            tech_signals=sample_technical_signals,
            fund_signals=sample_fundamental_signals,
            sent_signals=sent_signals,
            tech_weight=0.3,
            fund_weight=0.4,
            sent_weight=0.3,
            run_date="2026-02-09",
        )
        assert len(results) == 5
        for r in results:
            assert r.sentiment_score == 70.0
            assert r.sentiment_signal == "very_positive"
            # Combined should include sentiment contribution
            assert r.combined_signal_score > 0


# ============================================================
# Assemble Signal Result Tests
# ============================================================

class TestAssembleSignalResult:
    """Tests for assemble_signal_result task."""

    def test_assemble_basic(self):
        """Assemble 3 signal results into serialized output."""
        signal_results = [
            SignalResult(
                symbol="AAPL", run_date="2026-02-09",
                wf2_composite_score=1.85, wf2_quintile=1,
                technical_score=68.0, technical_signal="bullish",
                fundamental_score=55.0, fundamental_signal="balanced",
                combined_signal_score=61.5, signal_strength="buy",
                data_quality="complete",
            ),
            SignalResult(
                symbol="MSFT", run_date="2026-02-09",
                wf2_composite_score=1.42, wf2_quintile=1,
                technical_score=68.0, technical_signal="bullish",
                fundamental_score=62.0, fundamental_signal="balanced",
                combined_signal_score=65.0, signal_strength="buy",
                data_quality="complete",
            ),
            SignalResult(
                symbol="AMZN", run_date="2026-02-09",
                wf2_composite_score=0.75, wf2_quintile=2,
                technical_score=32.0, technical_signal="bearish",
                fundamental_score=40.0, fundamental_signal="growth",
                combined_signal_score=36.0, signal_strength="sell",
                data_quality="partial",
            ),
        ]

        result = assemble_signal_result(
            run_date="2026-02-09",
            signal_results=signal_results,
        )

        assert result["run_date"] == "2026-02-09"
        assert result["num_symbols_analyzed"] == "3"
        assert result["num_with_complete_data"] == "2"
        assert result["num_with_partial_data"] == "1"
        assert "AAPL" in result["top_buy_signals"]
        assert "AMZN" in result["top_sell_signals"]

        # Verify JSON is parseable
        sr_dicts = json.loads(result["signal_results_json"])
        assert len(sr_dicts) == 3

    def test_assemble_empty(self):
        """Assemble empty signal results."""
        result = assemble_signal_result(
            run_date="2026-02-09",
            signal_results=[],
        )
        assert result["num_symbols_analyzed"] == "0"
        assert result["top_buy_signals"] == ""


# ============================================================
# Generate Signal Report Tests
# ============================================================

class TestGenerateSignalReport:
    """Tests for generate_signal_report task."""

    def test_report_contains_header(self):
        """Report should contain the date and title."""
        assembled = {
            "run_date": "2026-02-09",
            "num_symbols_analyzed": "5",
            "num_with_complete_data": "3",
            "num_with_partial_data": "2",
            "top_buy_signals": "AAPL,MSFT",
            "top_sell_signals": "AMZN",
            "signal_results_json": json.dumps([
                {"symbol": "AAPL", "run_date": "2026-02-09",
                 "wf2_composite_score": 1.85, "wf2_quintile": 1,
                 "technical_score": 68.0, "technical_signal": "bullish",
                 "fundamental_score": 55.0, "fundamental_signal": "balanced",
                 "combined_signal_score": 61.5, "signal_strength": "buy",
                 "data_quality": "complete"},
            ]),
        }

        report = generate_signal_report(assembled_result=assembled)
        assert "2026-02-09" in report
        assert "Signal Analysis Report" in report
        assert "AAPL" in report
        assert "Signal Distribution" in report

    def test_report_empty_signals(self):
        """Report with no signals should still be valid."""
        assembled = {
            "run_date": "2026-02-09",
            "num_symbols_analyzed": "0",
            "num_with_complete_data": "0",
            "num_with_partial_data": "0",
            "top_buy_signals": "",
            "top_sell_signals": "",
            "signal_results_json": "[]",
        }

        report = generate_signal_report(assembled_result=assembled)
        assert "2026-02-09" in report
        assert "(none)" in report

    def test_report_distribution(self):
        """Report should show signal distribution counts."""
        sr_dicts = [
            {"symbol": f"SYM{i}", "run_date": "2026-02-09",
             "wf2_composite_score": 1.0, "wf2_quintile": 1,
             "technical_score": 70.0, "technical_signal": "bullish",
             "fundamental_score": 60.0, "fundamental_signal": "balanced",
             "combined_signal_score": 65.0, "signal_strength": "buy",
             "data_quality": "complete"}
            for i in range(3)
        ]
        assembled = {
            "run_date": "2026-02-09",
            "num_symbols_analyzed": "3",
            "num_with_complete_data": "3",
            "num_with_partial_data": "0",
            "top_buy_signals": "SYM0,SYM1,SYM2",
            "top_sell_signals": "",
            "signal_results_json": json.dumps(sr_dicts),
        }

        report = generate_signal_report(assembled_result=assembled)
        assert "buy" in report
        assert "3" in report


# ============================================================
# Model Tests
# ============================================================

class TestTechnicalSignalsModel:
    """Tests for TechnicalSignals dataclass."""

    def test_create_technical_signals(self):
        """TechnicalSignals can be created with all fields."""
        ts = TechnicalSignals(
            symbol="AAPL", sma_50=192.5, sma_200=185.3,
            sma_crossover_signal="bullish",
            macd_line=1.25, macd_signal_line=0.80, macd_histogram=0.45,
            macd_signal="bullish",
            bb_upper=198.5, bb_middle=192.5, bb_lower=186.5,
            bb_signal="neutral",
            technical_score=68.0,
        )
        assert ts.symbol == "AAPL"
        assert ts.technical_score == 68.0


class TestFundamentalSignalsModel:
    """Tests for FundamentalSignals dataclass."""

    def test_create_fundamental_signals(self):
        """FundamentalSignals can be created with defaults."""
        fs = FundamentalSignals(
            symbol="AAPL", pe_ratio=28.5, pe_zscore=-0.425,
            forward_pe=25.0, price_to_book=45.0,
            dividend_yield=0.005, return_on_equity=1.60,
            debt_to_equity=1.50, current_ratio=1.07,
        )
        assert fs.symbol == "AAPL"
        assert fs.has_pe is True  # default
        assert fs.fundamental_score == 50.0  # default
        assert fs.fundamental_signal == "balanced"  # default


class TestSignalResultModel:
    """Tests for SignalResult dataclass."""

    def test_create_signal_result(self):
        """SignalResult with all required fields."""
        sr = SignalResult(
            symbol="AAPL", run_date="2026-02-09",
            wf2_composite_score=1.85, wf2_quintile=1,
            technical_score=68.0, technical_signal="bullish",
            fundamental_score=55.0, fundamental_signal="balanced",
            combined_signal_score=61.5, signal_strength="buy",
        )
        assert sr.data_quality == "complete"  # default

    def test_signal_result_sentiment_defaults(self):
        """SignalResult should have neutral sentiment defaults for backward compat."""
        sr = SignalResult(
            symbol="AAPL", run_date="2026-02-09",
            wf2_composite_score=1.85, wf2_quintile=1,
            technical_score=68.0, technical_signal="bullish",
            fundamental_score=55.0, fundamental_signal="balanced",
        )
        assert sr.sentiment_score == 50.0
        assert sr.sentiment_signal == "neutral"
        assert sr.num_articles == 0
        assert sr.news_provider == "none"

    def test_signal_result_with_sentiment(self):
        """SignalResult can store sentiment fields."""
        sr = SignalResult(
            symbol="AAPL", run_date="2026-02-09",
            wf2_composite_score=1.85, wf2_quintile=1,
            technical_score=68.0, technical_signal="bullish",
            fundamental_score=55.0, fundamental_signal="balanced",
            sentiment_score=72.5, sentiment_signal="very_positive",
            num_articles=8, news_provider="finnhub",
        )
        assert sr.sentiment_score == 72.5
        assert sr.sentiment_signal == "very_positive"
        assert sr.num_articles == 8
        assert sr.news_provider == "finnhub"


# ============================================================
# M6: Sentiment Fields in Assembled Result + Report
# ============================================================

class TestAssembleSentimentFields:
    """Tests for sentiment data flowing through assemble_signal_result."""

    def test_assembled_contains_weights(self):
        """Assembled result should include signal weights."""
        signal_results = [
            SignalResult(
                symbol="AAPL", run_date="2026-02-09",
                wf2_composite_score=1.85, wf2_quintile=1,
                technical_score=68.0, technical_signal="bullish",
                fundamental_score=55.0, fundamental_signal="balanced",
                sentiment_score=75.0, sentiment_signal="very_positive",
                num_articles=5, news_provider="finnhub",
                combined_signal_score=65.0, signal_strength="buy",
            ),
        ]
        result = assemble_signal_result(
            run_date="2026-02-09",
            signal_results=signal_results,
            tech_weight=0.3,
            fund_weight=0.4,
            sent_weight=0.3,
        )
        assert result["tech_weight"] == "0.3"
        assert result["fund_weight"] == "0.4"
        assert result["sent_weight"] == "0.3"

    def test_assembled_json_contains_sentiment(self):
        """signal_results_json should include sentiment fields."""
        signal_results = [
            SignalResult(
                symbol="AAPL", run_date="2026-02-09",
                wf2_composite_score=1.85, wf2_quintile=1,
                technical_score=68.0, technical_signal="bullish",
                fundamental_score=55.0, fundamental_signal="balanced",
                sentiment_score=72.5, sentiment_signal="very_positive",
                num_articles=8, news_provider="finnhub",
                combined_signal_score=65.0, signal_strength="buy",
            ),
        ]
        result = assemble_signal_result(
            run_date="2026-02-09",
            signal_results=signal_results,
        )
        sr_dicts = json.loads(result["signal_results_json"])
        assert sr_dicts[0]["sentiment_score"] == 72.5
        assert sr_dicts[0]["sentiment_signal"] == "very_positive"
        assert sr_dicts[0]["num_articles"] == 8
        assert sr_dicts[0]["news_provider"] == "finnhub"


class TestReportSentimentFields:
    """Tests for sentiment data in generate_signal_report output."""

    def test_report_shows_sentiment_column(self):
        """Report should include Sent column in per-stock table."""
        sr_dicts = [
            {"symbol": "AAPL", "run_date": "2026-02-09",
             "wf2_composite_score": 1.85, "wf2_quintile": 1,
             "technical_score": 68.0, "technical_signal": "bullish",
             "fundamental_score": 55.0, "fundamental_signal": "balanced",
             "sentiment_score": 72.5, "sentiment_signal": "very_positive",
             "combined_signal_score": 65.0, "signal_strength": "buy",
             "data_quality": "complete"},
        ]
        assembled = {
            "run_date": "2026-02-09",
            "num_symbols_analyzed": "1",
            "num_with_complete_data": "1",
            "num_with_partial_data": "0",
            "top_buy_signals": "AAPL",
            "top_sell_signals": "",
            "tech_weight": "0.3",
            "fund_weight": "0.4",
            "sent_weight": "0.3",
            "signal_results_json": json.dumps(sr_dicts),
        }
        report = generate_signal_report(assembled_result=assembled)
        assert "Sent" in report
        assert "72.5" in report

    def test_report_shows_weights(self):
        """Report should display the signal weights."""
        assembled = {
            "run_date": "2026-02-09",
            "num_symbols_analyzed": "0",
            "num_with_complete_data": "0",
            "num_with_partial_data": "0",
            "top_buy_signals": "",
            "top_sell_signals": "",
            "tech_weight": "0.3",
            "fund_weight": "0.4",
            "sent_weight": "0.3",
            "signal_results_json": "[]",
        }
        report = generate_signal_report(assembled_result=assembled)
        assert "tech=0.3" in report
        assert "fund=0.4" in report
        assert "sent=0.3" in report
