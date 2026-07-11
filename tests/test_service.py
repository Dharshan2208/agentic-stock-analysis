"""
Data provider tests.

Covers mock providers (from conftest) and the real yfinance / serper
implementations with HTTP mocking.
"""

from __future__ import annotations

import json
from unittest import mock

import pytest
import requests

from services.base import (
    CompanyProfile,
    Fundamentals,
    NewsArticle,
    OHLCVBar,
)
from services.serper_provider import SerperNewsProvider
from services.yfinance_provider import (
    YFinanceFundamentals,
    YFinanceMarketData,
)
from tests.conftest import MockFundamentals, MockMarketData


# Mock providers (from conftest)


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


# YFinanceMarketData – live provider tests with mocked yfinance


def _fake_history(*, empty: bool = False):
    """Return a mock pandas DataFrame mimicking yfinance history output."""
    import pandas as pd

    if empty:
        return pd.DataFrame()

    data = {
        "Open": [150.0, 151.0],
        "High": [152.0, 153.0],
        "Low": [149.0, 150.5],
        "Close": [151.5, 152.8],
        "Volume": [1_000_000, 1_200_000],
    }
    index = pd.DatetimeIndex(
        ["2026-07-01", "2026-07-02"],
        name="Date",
    )
    return pd.DataFrame(data, index=index)


class TestYFinanceMarketData:
    @pytest.fixture
    def provider(self):
        return YFinanceMarketData()

    @mock.patch("yfinance.Ticker")
    def test_get_ohlcv_returns_typed_bars(self, mock_ticker, provider):
        """get_ohlcv returns list of OHLCVBar dataclasses."""
        mock_ticker.return_value.history.return_value = _fake_history()
        bars = provider.get_ohlcv("AAPL")

        assert len(bars) == 2
        assert all(isinstance(b, OHLCVBar) for b in bars)
        assert bars[0].open == 150.0
        assert bars[0].close == 151.5
        assert bars[1].volume == 1_200_000

    @mock.patch("yfinance.Ticker")
    def test_get_ohlcv_empty_data(self, mock_ticker, provider):
        """Empty DataFrame → empty list."""
        mock_ticker.return_value.history.return_value = _fake_history(empty=True)
        bars = provider.get_ohlcv("AAPL")
        assert bars == []

    @mock.patch("yfinance.Ticker")
    def test_get_current_price_from_intraday(self, mock_ticker, provider):
        """get_current_price prefers intraday data."""
        mock_ticker.return_value.history.side_effect = [
            _fake_history(),  # intraday (1m) – first call
        ]
        price = provider.get_current_price("AAPL")
        assert price == 152.8

    @mock.patch("yfinance.Ticker")
    def test_get_current_price_falls_back_to_daily(self, mock_ticker, provider):
        """When intraday is empty, fall back to daily data."""
        mock_ticker.return_value.history.side_effect = [
            _fake_history(empty=True),  # intraday empty
            _fake_history(),  # daily fallback
        ]
        price = provider.get_current_price("AAPL")
        assert price == 152.8

    @mock.patch("yfinance.Ticker")
    def test_get_current_price_all_empty(self, mock_ticker, provider):
        """All data sources empty → None."""
        mock_ticker.return_value.history.return_value = _fake_history(empty=True)
        price = provider.get_current_price("AAPL")
        assert price is None

    def test_symbol_uppercased(self, provider):
        """Symbol should be uppercased internally."""
        with mock.patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = _fake_history()
            provider.get_ohlcv("aapl")
            # The Ticker constructor should receive "AAPL"
            assert mock_ticker.call_args[0][0] == "AAPL"


# YFinanceFundamentals – live provider tests with mocked yfinance


_FAKE_INFO: dict = {
    "longName": "Apple Inc.",
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "marketCap": 3_000_000_000_000,
    "trailingPE": 28.5,
    "forwardPE": 25.0,
    "priceToBook": 45.0,
    "debtToEquity": 1.5,
    "currentRatio": 0.9,
    "profitMargins": 0.25,
    "returnOnEquity": 1.5,
    "earningsGrowth": 0.12,
    "revenueGrowth": 0.08,
    "freeCashflow": 100_000_000_000,
    "dividendYield": 0.005,
    "beta": 1.2,
    "fiftyTwoWeekHigh": 200.0,
    "fiftyTwoWeekLow": 150.0,
    "targetMedianPrice": 210.0,
    "recommendationKey": "buy",
}


class TestYFinanceFundamentals:
    @pytest.fixture
    def provider(self):
        return YFinanceFundamentals()

    @mock.patch("yfinance.Ticker")
    def test_get_company_profile(self, mock_ticker, provider):
        """get_company_profile returns typed CompanyProfile."""
        mock_ticker.return_value.info = _FAKE_INFO
        profile = provider.get_company_profile("AAPL")

        assert isinstance(profile, CompanyProfile)
        assert profile.symbol == "AAPL"
        assert profile.name == "Apple Inc."
        assert profile.sector == "Technology"
        assert profile.market_cap == 3_000_000_000_000

    @mock.patch("yfinance.Ticker")
    def test_get_company_profile_empty_info(self, mock_ticker, provider):
        """Empty info dict → None."""
        mock_ticker.return_value.info = {}
        assert provider.get_company_profile("AAPL") is None

    @mock.patch("yfinance.Ticker")
    def test_get_fundamentals(self, mock_ticker, provider):
        """get_fundamentals returns typed Fundamentals."""
        mock_ticker.return_value.info = _FAKE_INFO
        fund = provider.get_fundamentals("AAPL")

        assert isinstance(fund, Fundamentals)
        assert fund.trailing_pe == 28.5
        assert fund.forward_pe == 25.0
        assert fund.return_on_equity == 1.5
        assert fund.beta == 1.2
        assert fund.recommendation == "buy"

    @mock.patch("yfinance.Ticker")
    def test_get_fundamentals_empty_info(self, mock_ticker, provider):
        """Empty info dict → None."""
        mock_ticker.return_value.info = {}
        assert provider.get_fundamentals("AAPL") is None

    @mock.patch("yfinance.Ticker")
    def test_fundamentals_includes_raw(self, mock_ticker, provider):
        """The raw info dict should be preserved."""
        mock_ticker.return_value.info = _FAKE_INFO
        fund = provider.get_fundamentals("AAPL")
        assert fund.raw == _FAKE_INFO


# SerperNewsProvider – live provider tests with mocked HTTP


class TestSerperNewsProvider:
    @pytest.fixture
    def provider(self):
        """Return a SerperNewsProvider with a fake API key and no-op full-text
        fetching so tests are fast and deterministic."""
        with mock.patch(
            "services.serper_provider.SerperNewsProvider._fetch_full_text",
            return_value=None,
        ):
            yield SerperNewsProvider(api_key="test-key-123")

    def test_no_api_key_returns_empty(self):
        """Without an API key the provider returns []."""
        from config import settings

        with mock.patch.object(settings, "serper_api_key", ""):
            with mock.patch(
                "services.serper_provider.SerperNewsProvider._fetch_full_text",
                return_value=None,
            ):
                provider = SerperNewsProvider(api_key=None)
                assert provider.get_news("AAPL") == []

    @mock.patch("services.serper_provider.requests.post")
    def test_get_news_returns_typed_articles(self, mock_post, provider):
        """get_news returns a list of NewsArticle dataclasses."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "news": [
                {
                    "title": "Apple Stock Rises",
                    "link": "https://example.com/1",
                    "source": "Yahoo Finance",
                    "date": "2026-07-12",
                    "snippet": "Apple shares are up today.",
                },
            ],
        }

        articles = provider.get_news("AAPL", max_articles=1)

        assert len(articles) == 1
        article = articles[0]
        assert isinstance(article, NewsArticle)
        assert article.title == "Apple Stock Rises"
        assert article.source == "Yahoo Finance"
        assert article.url == "https://example.com/1"

    @mock.patch("services.serper_provider.requests.post")
    def test_api_error_returns_empty(self, mock_post, provider):
        """HTTP errors are caught and return []."""
        mock_post.side_effect = requests.RequestException("API error")
        articles = provider.get_news("AAPL")
        assert articles == []

    @mock.patch("services.serper_provider.requests.post")
    def test_empty_news_list(self, mock_post, provider):
        """API returning no news entries → []."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"news": []}
        articles = provider.get_news("AAPL")
        assert articles == []

    @mock.patch("services.serper_provider.requests.post")
    def test_article_without_link_skipped(self, mock_post, provider):
        """An article dict without 'link' is skipped."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "news": [
                {
                    "title": "No Link Article",
                    "source": "Test",
                    "date": "2026-07-12",
                    "snippet": "This article has no link.",
                },
            ],
        }

        articles = provider.get_news("AAPL", max_articles=1)
        assert len(articles) == 0

    def test_correct_content_type_header(self, provider):
        """The request should have the correct Content-Type header."""
        with mock.patch("services.serper_provider.requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"news": []}

            provider.get_news("AAPL")

            _, kwargs = mock_post.call_args
            assert kwargs["headers"]["Content-Type"] == "application/json"
            assert kwargs["headers"]["X-API-KEY"] == "test-key-123"
