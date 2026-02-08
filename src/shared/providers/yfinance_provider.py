"""YFinance data provider implementation (Phase 1).

Uses the free yfinance library to fetch market data from Yahoo Finance.
Sufficient for building and testing all 5 workflows. A paid provider
can be swapped in later by implementing the DataProvider ABC.

Limitations:
- Rate limiting by Yahoo Finance
- Data quality can be inconsistent
- No guaranteed SLA
"""

from typing import Dict, List

from src.shared.models import MarketDataBatch
from src.shared.providers.base import DataProvider


class YFinanceProvider(DataProvider):
    """Market data provider using yfinance (free, Phase 1).

    Implements fetch_eod fully.
    fetch_fundamentals and fetch_dividends are stubs for Phase 2/3.
    """

    def fetch_eod(self, symbols: List[str], date: str) -> MarketDataBatch:
        """Fetch end-of-day data using yfinance.

        Downloads price history for each symbol and extracts the latest
        available data point. Estimates bid-ask spread from high-low range.

        Args:
            symbols: List of stock ticker symbols.
            date: Target date (YYYY-MM-DD). Empty string means today.

        Returns:
            MarketDataBatch with available data. Missing symbols are
            reflected in the data_quality_score.
        """
        import yfinance as yf
        from datetime import datetime, timedelta

        # Resolve empty date to today
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        # Create date range for yfinance download (5 days back for safety)
        target_date = datetime.strptime(date, "%Y-%m-%d")
        start_date = (target_date - timedelta(days=5)).strftime("%Y-%m-%d")
        end_date = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")

        prices: Dict[str, float] = {}
        volumes: Dict[str, int] = {}
        spreads: Dict[str, float] = {}
        market_caps: Dict[str, float] = {}

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)

                if not hist.empty:
                    latest = hist.iloc[-1]
                    prices[symbol] = float(latest["Close"])
                    volumes[symbol] = int(latest["Volume"])

                    # Estimate spread from high-low range (in basis points)
                    if latest["Close"] > 0:
                        spreads[symbol] = round(
                            ((latest["High"] - latest["Low"]) / latest["Close"]) * 10000,
                            2,
                        )
                    else:
                        spreads[symbol] = 0.0

                    # Market cap from ticker info
                    info = ticker.info
                    market_caps[symbol] = float(info.get("marketCap", 0))
            except Exception:
                # Symbol failed - skip, quality score will reflect this
                continue

        # Calculate quality score: fraction of symbols successfully fetched
        quality = len(prices) / len(symbols) if symbols else 0.0

        return MarketDataBatch(
            symbols=symbols,
            date=date,
            prices=prices,
            volumes=volumes,
            spreads=spreads,
            market_caps=market_caps,
            data_quality_score=round(quality, 4),
        )

    def fetch_fundamentals(self, symbol: str) -> dict:
        """Fetch fundamental data - stub for Phase 2.

        Will be implemented with P/E ratio, ROE, dividend yield, etc.
        """
        return {"symbol": symbol, "status": "not_implemented", "phase": 2}

    def fetch_dividends(self, symbol: str) -> list:
        """Fetch dividend history - stub for Phase 3.

        Will be implemented with ex-dates, amounts, and payout history.
        """
        return []
