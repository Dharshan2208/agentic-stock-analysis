"""
Abstract interfaces for all data providers.
Every provider implements these contracts.
Agents depend on abstractions, not concrete providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OHLCVBar:
    """A single OHLCV data point."""

    time: str  # "2026-07-03 14:30" or "2026-07-03"
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass(frozen=True)
class CompanyProfile:
    """Basic company information."""

    symbol: str
    name: str
    sector: str | None = None
    industry: str | None = None
    market_cap: float | None = None
    currency: str = "USD"


@dataclass(frozen=True)
class Fundamentals:
    """Key fundamental metrics."""

    symbol: str
    trailing_pe: float | None = None
    forward_pe: float | None = None
    price_to_book: float | None = None
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    profit_margin: float | None = None
    return_on_equity: float | None = None
    earnings_growth: float | None = None
    revenue_growth: float | None = None
    free_cashflow: float | None = None
    dividend_yield: float | None = None
    beta: float | None = None
    fifty_two_week_high: float | None = None
    fifty_two_week_low: float | None = None
    analyst_target: float | None = None
    recommendation: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NewsArticle:
    """A single news article."""

    title: str
    source: str
    date: str
    url: str
    snippet: str
    full_text: str | None = None


class MarketDataProvider(ABC):
    """Contract for real-time and historical price data."""

    @abstractmethod
    def get_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        period: str = "1mo",
        bars: int = 10,
    ) -> list[OHLCVBar]:
        """Fetch OHLCV data for a symbol."""
        ...

    @abstractmethod
    def get_current_price(self, symbol: str) -> float | None:
        """Get the latest price."""
        ...


class FundamentalsProvider(ABC):
    """Contract for company fundamentals."""

    @abstractmethod
    def get_company_profile(self, symbol: str) -> CompanyProfile | None: ...

    @abstractmethod
    def get_fundamentals(self, symbol: str) -> Fundamentals | None: ...


class NewsProvider(ABC):
    """Contract for news data."""

    @abstractmethod
    def get_news(
        self,
        symbol: str,
        recency: str = "past 24 hours",
        max_articles: int = 7,
    ) -> list[NewsArticle]: ...


class TechnicalAnalysisProvider(ABC):
    """Contract for technical indicators (trend, momentum, etc.)."""

    @abstractmethod
    def get_trend_analysis(
        self,
        symbol: str,
        period: str = "7d",
    ) -> dict[str, Any]:
        """Return trend analysis metrics as a dict."""
        ...
