"""Abstract base class for sentiment/news data providers.

Provider-Abstraction-Layer: Build first, decide later.
Concrete providers (Finnhub, Marketaux) are pluggable without
workflow changes — just change instantiation in the task.

Phase 6: FinnhubSentimentProvider (primary) + MarketauxSentimentProvider (fallback)
"""

from abc import ABC, abstractmethod
from typing import List


class SentimentProvider(ABC):
    """Abstract provider for financial news articles.

    Subclasses implement fetch_news for a specific API.
    Switching providers requires only changing instantiation in the task.

    Article dict schema (normalized across all providers):
        - headline: str (article title)
        - summary: str (article summary/snippet, may be empty)
        - source: str (publisher name)
        - published_at: str (ISO datetime string)
        - url: str (article URL)
        - provider: str ("finnhub" or "marketaux")
    """

    @abstractmethod
    def fetch_news(self, symbol: str, from_date: str, to_date: str) -> List[dict]:
        """Fetch recent news articles for a stock symbol.

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL").
            from_date: Start date (YYYY-MM-DD).
            to_date: End date (YYYY-MM-DD).

        Returns:
            List of article dicts with normalized keys.
            Empty list if no articles found or API error.
        """
        ...
