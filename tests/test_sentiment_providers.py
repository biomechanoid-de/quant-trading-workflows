"""Tests for M1: Sentiment news providers (Finnhub + Marketaux).

All API calls are mocked — no network access required.
"""

import pytest
from unittest.mock import patch, MagicMock
import sys

from src.shared.providers.sentiment_base import SentimentProvider
from src.shared.providers.finnhub_provider import FinnhubSentimentProvider
from src.shared.providers.marketaux_provider import MarketauxSentimentProvider


# ============================================================
# SentimentProvider ABC
# ============================================================

class TestSentimentProviderABC:
    """Verify abstract base class cannot be instantiated."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            SentimentProvider()

    def test_subclass_must_implement_fetch_news(self):
        class IncompleteProvider(SentimentProvider):
            pass

        with pytest.raises(TypeError):
            IncompleteProvider()


# ============================================================
# FinnhubSentimentProvider
# ============================================================

class TestFinnhubSentimentProvider:
    """Tests for Finnhub news provider with mocked API."""

    def _make_provider_with_mock(self, return_value=None, side_effect=None):
        """Create provider with mocked finnhub client."""
        mock_client = MagicMock()
        mock_finnhub = MagicMock()
        mock_finnhub.Client.return_value = mock_client

        if side_effect:
            mock_client.company_news.side_effect = side_effect
        else:
            mock_client.company_news.return_value = return_value or []

        # Inject mock into sys.modules so lazy import picks it up
        with patch.dict(sys.modules, {"finnhub": mock_finnhub}):
            provider = FinnhubSentimentProvider(api_key="test_key")
            articles = provider.fetch_news("AAPL", "2024-02-10", "2024-02-15")

        return articles, mock_client

    def test_fetch_news_returns_normalized_articles(self):
        articles, _ = self._make_provider_with_mock(return_value=[
            {
                "headline": "Apple beats earnings",
                "summary": "Strong Q4 results",
                "source": "Reuters",
                "datetime": 1707955200,  # 2024-02-15 00:00:00 UTC
                "url": "https://example.com/article1",
            },
            {
                "headline": "Apple launches new product",
                "summary": "",
                "source": "Bloomberg",
                "datetime": 1707868800,
                "url": "https://example.com/article2",
            },
        ])

        assert len(articles) == 2
        assert articles[0]["headline"] == "Apple beats earnings"
        assert articles[0]["summary"] == "Strong Q4 results"
        assert articles[0]["source"] == "Reuters"
        assert articles[0]["provider"] == "finnhub"
        assert articles[0]["url"] == "https://example.com/article1"
        assert articles[0]["published_at"] != ""

    def test_fetch_news_empty_response(self):
        articles, _ = self._make_provider_with_mock(return_value=[])
        assert articles == []

    def test_fetch_news_api_exception(self):
        mock_client = MagicMock()
        mock_finnhub = MagicMock()
        mock_finnhub.Client.return_value = mock_client
        mock_client.company_news.side_effect = Exception("API rate limit")

        with patch.dict(sys.modules, {"finnhub": mock_finnhub}):
            provider = FinnhubSentimentProvider(api_key="test_key")
            with pytest.raises(Exception, match="API rate limit"):
                provider.fetch_news("AAPL", "2024-02-10", "2024-02-15")

    def test_fetch_news_missing_fields(self):
        articles, _ = self._make_provider_with_mock(return_value=[
            {"headline": "Partial article"},
        ])

        assert len(articles) == 1
        assert articles[0]["headline"] == "Partial article"
        assert articles[0]["summary"] == ""
        assert articles[0]["source"] == ""
        assert articles[0]["provider"] == "finnhub"

    def test_fetch_news_invalid_timestamp(self):
        articles, _ = self._make_provider_with_mock(return_value=[
            {"headline": "Test", "datetime": 0},
        ])

        assert len(articles) == 1
        assert articles[0]["published_at"] == ""


# ============================================================
# MarketauxSentimentProvider
# ============================================================

class TestMarketauxSentimentProvider:
    """Tests for Marketaux news provider with mocked API."""

    def _make_response(self, json_data, raise_error=None):
        """Create mock requests.get response."""
        mock_response = MagicMock()
        mock_response.json.return_value = json_data
        if raise_error:
            mock_response.raise_for_status.side_effect = raise_error
        else:
            mock_response.raise_for_status = MagicMock()
        return mock_response

    def _make_mock_requests(self, json_data=None, raise_error=None):
        """Create mock requests module."""
        mock_requests = MagicMock()
        response = self._make_response(json_data or {"data": []}, raise_error)
        mock_requests.get.return_value = response
        return mock_requests

    def test_fetch_news_returns_normalized_articles(self):
        mock_requests = self._make_mock_requests(json_data={
            "data": [
                {
                    "title": "MSFT earnings beat expectations",
                    "description": "Strong cloud growth",
                    "source": "CNBC",
                    "published_at": "2024-02-15T10:00:00Z",
                    "url": "https://example.com/article1",
                },
            ]
        })

        with patch.dict(sys.modules, {"requests": mock_requests}):
            # Need to reimport to pick up mocked requests
            import importlib
            import src.shared.providers.marketaux_provider as mp
            importlib.reload(mp)
            provider = mp.MarketauxSentimentProvider(api_key="test_key")
            articles = provider.fetch_news("MSFT", "2024-02-10", "2024-02-15")

        assert len(articles) == 1
        assert articles[0]["headline"] == "MSFT earnings beat expectations"
        assert articles[0]["summary"] == "Strong cloud growth"
        assert articles[0]["source"] == "CNBC"
        assert articles[0]["provider"] == "marketaux"
        assert articles[0]["published_at"] == "2024-02-15T10:00:00Z"

    def test_fetch_news_empty_response(self):
        mock_requests = self._make_mock_requests(json_data={"data": []})

        with patch.dict(sys.modules, {"requests": mock_requests}):
            import importlib
            import src.shared.providers.marketaux_provider as mp
            importlib.reload(mp)
            provider = mp.MarketauxSentimentProvider(api_key="test_key")
            articles = provider.fetch_news("MSFT", "2024-02-10", "2024-02-15")

        assert articles == []

    def test_fetch_news_http_error(self):
        mock_requests = self._make_mock_requests(
            raise_error=Exception("429 Too Many Requests")
        )

        with patch.dict(sys.modules, {"requests": mock_requests}):
            import importlib
            import src.shared.providers.marketaux_provider as mp
            importlib.reload(mp)
            provider = mp.MarketauxSentimentProvider(api_key="test_key")
            with pytest.raises(Exception, match="429"):
                provider.fetch_news("MSFT", "2024-02-10", "2024-02-15")

    def test_fetch_news_missing_data_key(self):
        mock_requests = self._make_mock_requests(json_data={})

        with patch.dict(sys.modules, {"requests": mock_requests}):
            import importlib
            import src.shared.providers.marketaux_provider as mp
            importlib.reload(mp)
            provider = mp.MarketauxSentimentProvider(api_key="test_key")
            articles = provider.fetch_news("MSFT", "2024-02-10", "2024-02-15")

        assert articles == []

    def test_fetch_news_passes_correct_params(self):
        mock_requests = self._make_mock_requests(json_data={"data": []})

        with patch.dict(sys.modules, {"requests": mock_requests}):
            import importlib
            import src.shared.providers.marketaux_provider as mp
            importlib.reload(mp)
            provider = mp.MarketauxSentimentProvider(api_key="my_api_key")
            provider.fetch_news("AAPL", "2024-02-10", "2024-02-15")

        call_args = mock_requests.get.call_args
        params = call_args[1]["params"]
        assert params["symbols"] == "AAPL"
        assert params["published_after"] == "2024-02-10"
        assert params["published_before"] == "2024-02-15"
        assert params["api_token"] == "my_api_key"
        assert params["language"] == "en"
