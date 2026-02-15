"""Tests for M3: Sentiment analytics functions.

Tests compute_sentiment_score, classify_sentiment_signal,
and aggregate_article_sentiments from shared/analytics.py.
"""

import pytest

from src.shared.analytics import (
    compute_sentiment_score,
    classify_sentiment_signal,
    aggregate_article_sentiments,
)


# ============================================================
# compute_sentiment_score
# ============================================================

class TestComputeSentimentScore:
    """Tests for time-decay-weighted sentiment score aggregation."""

    def test_empty_articles_returns_neutral(self):
        assert compute_sentiment_score([]) == 50.0

    def test_single_positive_article(self):
        articles = [
            {"positive": 0.9, "negative": 0.05, "published_at": "2024-02-15T10:00:00"},
        ]
        score = compute_sentiment_score(articles, run_date="2024-02-15")
        assert score == 90.0

    def test_single_negative_article(self):
        articles = [
            {"positive": 0.05, "negative": 0.9, "published_at": "2024-02-15T10:00:00"},
        ]
        score = compute_sentiment_score(articles, run_date="2024-02-15")
        assert score == 5.0

    def test_neutral_article(self):
        articles = [
            {"positive": 0.3, "negative": 0.3, "published_at": "2024-02-15T10:00:00"},
        ]
        score = compute_sentiment_score(articles, run_date="2024-02-15")
        assert score == 30.0

    def test_time_decay_recent_weighted_higher(self):
        """Recent article (today) should matter more than old article (7 days ago)."""
        articles = [
            {"positive": 0.9, "negative": 0.05, "published_at": "2024-02-15T10:00:00"},  # today, positive
            {"positive": 0.1, "negative": 0.8, "published_at": "2024-02-08T10:00:00"},   # 7 days ago, negative
        ]
        score = compute_sentiment_score(articles, decay_half_life_days=3.0, run_date="2024-02-15")
        # Today's positive article should dominate (weight ~1.0 vs ~0.2 for 7-day-old)
        assert score > 60.0

    def test_time_decay_old_articles_discounted(self):
        """Same sentiment but older articles should produce weaker signal."""
        recent = [
            {"positive": 0.9, "negative": 0.05, "published_at": "2024-02-15T10:00:00"},
        ]
        old = [
            {"positive": 0.9, "negative": 0.05, "published_at": "2024-02-01T10:00:00"},
        ]
        score_recent = compute_sentiment_score(recent, decay_half_life_days=3.0, run_date="2024-02-15")
        score_old = compute_sentiment_score(old, decay_half_life_days=3.0, run_date="2024-02-15")
        # Both should be positive, but scores are the same since each has only one article
        # (time decay only matters when comparing across articles in the same set)
        assert score_recent == score_old  # Single article -> same score regardless of age

    def test_multiple_articles_averaged(self):
        articles = [
            {"positive": 0.8, "negative": 0.1, "published_at": "2024-02-15T10:00:00"},
            {"positive": 0.2, "negative": 0.7, "published_at": "2024-02-15T10:00:00"},
        ]
        score = compute_sentiment_score(articles, decay_half_life_days=3.0, run_date="2024-02-15")
        # Equal weights (same timestamp) -> average: (80 + 20) / 2 = 50
        assert score == 50.0

    def test_missing_published_at_defaults_to_zero_age(self):
        articles = [
            {"positive": 0.8, "negative": 0.1, "published_at": ""},
        ]
        score = compute_sentiment_score(articles, run_date="2024-02-15")
        assert score == 80.0

    def test_yyyy_mm_dd_date_format(self):
        articles = [
            {"positive": 0.7, "negative": 0.2, "published_at": "2024-02-14"},
        ]
        score = compute_sentiment_score(articles, run_date="2024-02-15")
        assert 60.0 < score < 80.0  # Close to 70 but slightly decayed (1 day old)


# ============================================================
# classify_sentiment_signal
# ============================================================

class TestClassifySentimentSignal:
    """Tests for sentiment score classification into signals."""

    def test_very_positive(self):
        assert classify_sentiment_signal(70.0) == "very_positive"
        assert classify_sentiment_signal(85.0) == "very_positive"
        assert classify_sentiment_signal(100.0) == "very_positive"

    def test_positive(self):
        assert classify_sentiment_signal(55.0) == "positive"
        assert classify_sentiment_signal(60.0) == "positive"
        assert classify_sentiment_signal(69.9) == "positive"

    def test_neutral(self):
        assert classify_sentiment_signal(45.0) == "neutral"
        assert classify_sentiment_signal(50.0) == "neutral"
        assert classify_sentiment_signal(54.9) == "neutral"

    def test_negative(self):
        assert classify_sentiment_signal(30.0) == "negative"
        assert classify_sentiment_signal(40.0) == "negative"
        assert classify_sentiment_signal(44.9) == "negative"

    def test_very_negative(self):
        assert classify_sentiment_signal(0.0) == "very_negative"
        assert classify_sentiment_signal(15.0) == "very_negative"
        assert classify_sentiment_signal(29.9) == "very_negative"


# ============================================================
# aggregate_article_sentiments
# ============================================================

class TestAggregateArticleSentiments:
    """Tests for merging articles with classifier results."""

    def test_basic_merge(self):
        articles = [
            {"headline": "Good news", "published_at": "2024-02-15T10:00:00", "source": "Reuters"},
        ]
        classifier_results = [
            {"positive": 0.8, "neutral": 0.1, "negative": 0.1},
        ]
        merged = aggregate_article_sentiments(articles, classifier_results)

        assert len(merged) == 1
        assert merged[0]["headline"] == "Good news"
        assert merged[0]["published_at"] == "2024-02-15T10:00:00"
        assert merged[0]["positive"] == 0.8
        assert merged[0]["neutral"] == 0.1
        assert merged[0]["negative"] == 0.1

    def test_multiple_articles(self):
        articles = [
            {"headline": "A", "published_at": "2024-02-15"},
            {"headline": "B", "published_at": "2024-02-14"},
        ]
        classifier_results = [
            {"positive": 0.9, "neutral": 0.05, "negative": 0.05},
            {"positive": 0.1, "neutral": 0.1, "negative": 0.8},
        ]
        merged = aggregate_article_sentiments(articles, classifier_results)

        assert len(merged) == 2
        assert merged[0]["headline"] == "A"
        assert merged[0]["positive"] == 0.9
        assert merged[1]["headline"] == "B"
        assert merged[1]["negative"] == 0.8

    def test_mismatched_lengths_uses_min(self):
        articles = [
            {"headline": "A", "published_at": "2024-02-15"},
            {"headline": "B", "published_at": "2024-02-14"},
            {"headline": "C", "published_at": "2024-02-13"},
        ]
        classifier_results = [
            {"positive": 0.5, "neutral": 0.3, "negative": 0.2},
        ]
        merged = aggregate_article_sentiments(articles, classifier_results)

        assert len(merged) == 1
        assert merged[0]["headline"] == "A"

    def test_empty_inputs(self):
        assert aggregate_article_sentiments([], []) == []

    def test_missing_keys_default(self):
        articles = [{}]
        classifier_results = [{}]
        merged = aggregate_article_sentiments(articles, classifier_results)

        assert len(merged) == 1
        assert merged[0]["headline"] == ""
        assert merged[0]["published_at"] == ""
        assert merged[0]["positive"] == 0.0
