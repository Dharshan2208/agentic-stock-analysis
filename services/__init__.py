"""
Data provider abstractions and implementations.

Usage:
    from services import get_market_data, get_fundamentals, get_news_provider
    price = get_market_data().get_current_price("AAPL")
    news = get_news_provider().get_news("AAPL")

Swapping implementations (e.g., to Alpha Vantage):
    Just pass a different provider to the factory.
"""

from services.base import (
    MarketDataProvider,
    FundamentalsProvider,
    NewsProvider,
    CompanyProfile,
    Fundamentals,
    NewsArticle,
    OHLCVBar,
)
from services.yfinance_provider import (
    YFinanceMarketData,
    YFinanceFundamentals,
)
from services.serper_provider import SerperNewsProvider


_market_data: MarketDataProvider = YFinanceMarketData()
_fundamentals: FundamentalsProvider = YFinanceFundamentals()
_news_provider: NewsProvider = SerperNewsProvider()


def get_market_data() -> MarketDataProvider:
    """Get the active market data provider."""
    return _market_data


def get_fundamentals() -> FundamentalsProvider:
    """Get the active fundamentals provider."""
    return _fundamentals


def get_news_provider() -> NewsProvider:
    """Get the active news provider."""
    return _news_provider


def set_market_data(provider: MarketDataProvider) -> None:
    """Swap market data provider (useful for testing)."""
    global _market_data
    _market_data = provider


def set_fundamentals(provider: FundamentalsProvider) -> None:
    """Swap fundamentals provider (useful for testing)."""
    global _fundamentals
    _fundamentals = provider


def set_news_provider(provider: NewsProvider) -> None:
    """Swap news provider (useful for testing)."""
    global _news_provider
    _news_provider = provider


__all__ = [
    "MarketDataProvider",
    "FundamentalsProvider",
    "NewsProvider",
    "CompanyProfile",
    "Fundamentals",
    "NewsArticle",
    "OHLCVBar",
    "get_market_data",
    "get_fundamentals",
    "get_news_provider",
    "set_market_data",
    "set_fundamentals",
    "set_news_provider",
    "YFinanceMarketData",
    "YFinanceFundamentals",
    "SerperNewsProvider",
]
