"""
Data provider abstractions and implementations.

Usage:
    from services import get_market_data, get_fundamentals
    price = get_market_data().get_current_price("AAPL")

Swapping implementations (e.g., to Alpha Vantage):
    Just pass a different provider to the factory.
"""

from services.base import (
    MarketDataProvider,
    FundamentalsProvider,
    CompanyProfile,
    Fundamentals,
    OHLCVBar,
)
from services.yfinance_provider import (
    YFinanceMarketData,
    YFinanceFundamentals,
)


_market_data: MarketDataProvider = YFinanceMarketData()
_fundamentals: FundamentalsProvider = YFinanceFundamentals()


def get_market_data() -> MarketDataProvider:
    """Get the active market data provider."""
    return _market_data


def get_fundamentals() -> FundamentalsProvider:
    """Get the active fundamentals provider."""
    return _fundamentals


def set_market_data(provider: MarketDataProvider) -> None:
    """Swap market data provider (useful for testing)."""
    global _market_data
    _market_data = provider


def set_fundamentals(provider: FundamentalsProvider) -> None:
    """Swap fundamentals provider (useful for testing)."""
    global _fundamentals
    _fundamentals = provider


__all__ = [
    "MarketDataProvider",
    "FundamentalsProvider",
    "CompanyProfile",
    "Fundamentals",
    "OHLCVBar",
    "get_market_data",
    "get_fundamentals",
    "set_market_data",
    "set_fundamentals",
    "YFinanceMarketData",
    "YFinanceFundamentals",
]
