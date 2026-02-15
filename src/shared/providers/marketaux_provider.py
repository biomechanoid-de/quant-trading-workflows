"""Marketaux news provider implementation (fallback).

Uses Marketaux REST API to fetch financial news.
Free tier: 100 requests/day, 3 articles per request.
"""

from typing import List

from src.shared.providers.sentiment_base import SentimentProvider


class MarketauxSentimentProvider(SentimentProvider):
    """News provider using Marketaux free tier (fallback).

    Fetches financial news via REST API.
    Free tier: 100 req/day with up to 3 articles per request.
    """

    BASE_URL = "https://api.marketaux.com/v1/news/all"

    def __init__(self, api_key: str):
        self._api_key = api_key

    def fetch_news(self, symbol: str, from_date: str, to_date: str) -> List[dict]:
        """Fetch news from Marketaux REST API.

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL").
            from_date: Start date (YYYY-MM-DD).
            to_date: End date (YYYY-MM-DD).

        Returns:
            List of article dicts with normalized keys.
        """
        import requests

        resp = requests.get(
            self.BASE_URL,
            params={
                "symbols": symbol,
                "published_after": from_date,
                "published_before": to_date,
                "limit": 50,
                "language": "en",
                "api_token": self._api_key,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])

        results = []
        for article in data:
            results.append({
                "headline": article.get("title", ""),
                "summary": article.get("description", ""),
                "source": article.get("source", ""),
                "published_at": article.get("published_at", ""),
                "url": article.get("url", ""),
                "provider": "marketaux",
            })

        return results
