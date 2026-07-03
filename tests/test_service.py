"""Tests for data providers using mocks."""

from services.base import OHLCVBar
from tests.conftest import MockMarketData, MockFundamentals


class TestMockMarketData:
    def setup_method(self):
        self.provider = MockMarketData()

    def test_get_ohlcv_returns_bars(self):
        bars = self.provider.get_ohlcv("AAPL")
        assert len(bars) == 3
        assert all(isinstance(b, OHLCVBar) for b in bars)

    def test_current_price(self):
        price = self.provider.get_current_price("AAPL")
        assert price == 154.2


class TestMockFundamentals:
    def setup_method(self):
        self.provider = MockFundamentals()

    def test_get_profile(self):
        profile = self.provider.get_company_profile("AAPL")
        assert profile.symbol == "AAPL"
        assert profile.sector == "Technology"

    def test_get_fundamentals(self):
        fund = self.provider.get_fundamentals("AAPL")
        assert fund.trailing_pe == 25.0
        assert fund.return_on_equity == 0.35
