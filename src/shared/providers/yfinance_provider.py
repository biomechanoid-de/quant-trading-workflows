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
        """Fetch fundamental data from yfinance ticker.info.

        Retrieves valuation, income, efficiency, and leverage metrics.
        Missing data defaults to -1.0 (ratios) or 0.0 (yields) so
        downstream scoring can identify and handle gaps gracefully.

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL").

        Returns:
            Dict with keys: symbol, pe_ratio, forward_pe, price_to_book,
            dividend_yield, return_on_equity, debt_to_equity, current_ratio,
            trailing_eps, sector, industry.
        """
        import yfinance as yf

        defaults = {
            "symbol": symbol,
            "pe_ratio": -1.0,
            "forward_pe": -1.0,
            "price_to_book": -1.0,
            "dividend_yield": 0.0,
            "return_on_equity": -1.0,
            "debt_to_equity": -1.0,
            "current_ratio": -1.0,
            "trailing_eps": -1.0,
            "sector": "",
            "industry": "",
        }

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info:
                return defaults

            # D/E ratio from yfinance is in percentage (e.g., 150 = 1.5x)
            raw_de = info.get("debtToEquity")
            if raw_de is not None and float(raw_de) > 0:
                de_ratio = float(raw_de) / 100.0
            else:
                de_ratio = -1.0

            return {
                "symbol": symbol,
                "pe_ratio": float(info.get("trailingPE") or -1.0),
                "forward_pe": float(info.get("forwardPE") or -1.0),
                "price_to_book": float(info.get("priceToBook") or -1.0),
                "dividend_yield": float(info.get("dividendYield") or 0.0),
                "return_on_equity": float(info.get("returnOnEquity") or -1.0),
                "debt_to_equity": de_ratio,
                "current_ratio": float(info.get("currentRatio") or -1.0),
                "trailing_eps": float(info.get("trailingEps") or -1.0),
                "sector": str(info.get("sector", "")),
                "industry": str(info.get("industry", "")),
            }
        except Exception:
            return defaults

    def fetch_dividends(self, symbol: str) -> list:
        """Fetch recent dividend events for a symbol via yfinance.

        Returns dividends with ex-dates in the last 90 days.
        Uses ticker.dividends which returns a pandas Series
        (DatetimeIndex -> float amount per share).

        Args:
            symbol: Stock ticker symbol.

        Returns:
            List of dicts with keys: symbol, ex_date, amount_per_share.
        """
        import time
        from datetime import datetime, timedelta

        import yfinance as yf

        try:
            ticker = yf.Ticker(symbol)
            dividends = ticker.dividends
            if dividends is None or dividends.empty:
                return []

            cutoff = datetime.now() - timedelta(days=90)
            recent = dividends[dividends.index >= cutoff]

            results = []
            for ex_date, amount in recent.items():
                results.append({
                    "symbol": symbol,
                    "ex_date": ex_date.strftime("%Y-%m-%d"),
                    "amount_per_share": round(float(amount), 4),
                })

            time.sleep(0.2)
            return results
        except Exception:
            return []
