"""Abstract base class for market data providers.

Provider-Abstraction-Layer: Build first, decide later.
All data tasks access market data through this abstract interface.
The concrete provider (yfinance, EODHD, Massive, Alpha Vantage) is
pluggable without workflow changes.

Phase 1: YFinanceProvider (free)
Later:   EODHD (EUR 19.99/Mo), Massive ($29/Mo), Alpha Vantage ($49.99/Mo)
"""

from abc import ABC, abstractmethod
from typing import List

from src.shared.models import MarketDataBatch


class DataProvider(ABC):
    """Abstract data provider for market data.

    Subclasses must implement all three methods.
    Switching providers requires only changing the instantiation
    in the task, not the workflow logic.
    """

    @abstractmethod
    def fetch_eod(self, symbols: List[str], date: str) -> MarketDataBatch:
        """Fetch end-of-day market data for symbols on a given date.

        Args:
            symbols: List of stock ticker symbols.
            date: Target date (YYYY-MM-DD). Empty string means today.

        Returns:
            MarketDataBatch with prices, volumes, spreads, and market caps.
        """
        ...

    @abstractmethod
    def fetch_fundamentals(self, symbol: str) -> dict:
        """Fetch fundamental data for a single symbol.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Dictionary with fundamental data (P/E, ROE, etc.).
        """
        ...

    @abstractmethod
    def fetch_dividends(self, symbol: str) -> list:
        """Fetch dividend history for a single symbol.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            List of dividend events.
        """
        ...
