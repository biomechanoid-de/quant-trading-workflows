"""Finnhub news provider implementation.

Uses finnhub-python SDK to fetch company-specific news.
Free tier: 60 API calls per minute — sufficient for 49 symbols.
"""

from datetime import datetime
from typing import List

from src.shared.providers.sentiment_base import SentimentProvider


class FinnhubSentimentProvider(SentimentProvider):
    """News provider using Finnhub free tier (60 calls/min).

    Fetches company news via finnhub.Client.company_news().
    Articles are normalized to the standard schema.
    """

    def __init__(self, api_key: str):
        self._api_key = api_key

    def fetch_news(self, symbol: str, from_date: str, to_date: str) -> List[dict]:
        """Fetch company news from Finnhub.

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL").
            from_date: Start date (YYYY-MM-DD).
            to_date: End date (YYYY-MM-DD).

        Returns:
            List of article dicts with normalized keys.
        """
        import finnhub

        client = finnhub.Client(api_key=self._api_key)
        raw_articles = client.company_news(symbol, _from=from_date, to=to_date)

        results = []
        for article in raw_articles:
            # Finnhub returns datetime as Unix timestamp
            ts = article.get("datetime", 0)
            published_at = ""
            if ts:
                try:
                    published_at = datetime.utcfromtimestamp(ts).isoformat()
                except (ValueError, OSError):
                    published_at = ""

            results.append({
                "headline": article.get("headline", ""),
                "summary": article.get("summary", ""),
                "source": article.get("source", ""),
                "published_at": published_at,
                "url": article.get("url", ""),
                "provider": "finnhub",
            })

        return results
