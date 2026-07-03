"""
Shared test fixtures and mock data providers.
"""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass, field
import pytest

from models import ResearchState, AgentAnalysis, Signal
from services.base import (
    MarketDataProvider,
    FundamentalsProvider,
    OHLCVBar,
    CompanyProfile,
    Fundamentals,
)


class MockMarketData(MarketDataProvider):
    """Returns canned data — no API calls."""

    def __init__(self):
        self.bars = [
            OHLCVBar(
                time="2026-07-01",
                open=150.0,
                high=152.0,
                low=149.0,
                close=151.5,
                volume=1000000,
            ),
            OHLCVBar(
                time="2026-07-02",
                open=151.5,
                high=153.0,
                low=150.5,
                close=152.8,
                volume=1200000,
            ),
            OHLCVBar(
                time="2026-07-03",
                open=152.8,
                high=155.0,
                low=152.0,
                close=154.2,
                volume=1500000,
            ),
        ]

    def get_ohlcv(self, symbol: str, **kwargs) -> list[OHLCVBar]:
        return self.bars

    def get_current_price(self, symbol: str) -> float | None:
        return self.bars[-1].close if self.bars else None


class MockFundamentals(FundamentalsProvider):
    """Returns canned fundamental data."""

    def get_company_profile(self, symbol: str) -> CompanyProfile | None:
        return CompanyProfile(
            symbol=symbol.upper(),
            name="Mock Corp",
            sector="Technology",
            industry="Software",
            market_cap=1_000_000_000_000,
        )

    def get_fundamentals(self, symbol: str) -> Fundamentals | None:
        return Fundamentals(
            symbol=symbol.upper(),
            trailing_pe=25.0,
            forward_pe=22.0,
            price_to_book=8.0,
            debt_to_equity=0.5,
            return_on_equity=0.35,
            earnings_growth=0.15,
            beta=1.2,
        )


@pytest.fixture
def mock_market_data():
    """Inject mock market data provider."""
    from services import set_market_data

    provider = MockMarketData()
    set_market_data(provider)
    yield provider


@pytest.fixture
def mock_fundamentals():
    """Inject mock fundamentals provider."""
    from services import set_fundamentals

    provider = MockFundamentals()
    set_fundamentals(provider)
    yield provider


@pytest.fixture
def sample_research_state() -> ResearchState:
    """A fully populated ResearchState for testing."""
    return ResearchState(
        user_query="Analyze AAPL stock",
        symbol="AAPL",
        sector="Technology",
        analyses={
            "market_analyst": AgentAnalysis(
                agent_name="market_analyst",
                symbol="AAPL",
                agent_confidence=0.82,
                summary="Price showing strong upward momentum.",
                signals=[Signal(name="uptrend", value=True, direction="bullish")],
            ),
        },
    )
