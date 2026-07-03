"""
Concrete implementation of data providers using yfinance.
"""

from __future__ import annotations

import json
from typing import Any

import yfinance as yf

from services.base import (
    MarketDataProvider,
    FundamentalsProvider,
    CompanyProfile,
    Fundamentals,
    OHLCVBar,
)


class YFinanceMarketData(MarketDataProvider):
    """yfinance implementation for price/OHLCV data."""

    def get_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        period: str = "1mo",
        bars: int = 10,
    ) -> list[OHLCVBar]:
        ticker = yf.Ticker(symbol.upper())
        data = ticker.history(interval=interval, period=period)
        if data.empty:
            return []

        last_n = data.tail(bars)
        result = []
        for idx, row in last_n.iterrows():
            idx_any: Any = idx
            ts = (
                idx_any.strftime("%Y-%m-%d %H:%M")
                if interval.endswith("m")
                else idx_any.strftime("%Y-%m-%d")
            )
            result.append(
                OHLCVBar(
                    time=ts,
                    open=round(float(row["Open"]), 4),
                    high=round(float(row["High"]), 4),
                    low=round(float(row["Low"]), 4),
                    close=round(float(row["Close"]), 4),
                    volume=int(row["Volume"]),
                )
            )
        return result

    def get_current_price(self, symbol: str) -> float | None:
        ticker = yf.Ticker(symbol.upper())
        data = ticker.history(period="1d", interval="1m")
        if data.empty:
            data = ticker.history(period="5d", interval="1d")
        if data.empty:
            return None
        return float(data["Close"].iloc[-1])


class YFinanceFundamentals(FundamentalsProvider):
    """yfinance implementation for fundamental data."""

    def get_company_profile(self, symbol: str) -> CompanyProfile | None:
        info = yf.Ticker(symbol.upper()).info
        if not info:
            return None
        return CompanyProfile(
            symbol=symbol.upper(),
            name=info.get("longName", ""),
            sector=info.get("sector"),
            industry=info.get("industry"),
            market_cap=info.get("marketCap"),
        )

    def get_fundamentals(self, symbol: str) -> Fundamentals | None:
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info
        if not info:
            return None
        return Fundamentals(
            symbol=symbol.upper(),
            trailing_pe=info.get("trailingPE"),
            forward_pe=info.get("forwardPE"),
            price_to_book=info.get("priceToBook"),
            debt_to_equity=info.get("debtToEquity"),
            current_ratio=info.get("currentRatio"),
            profit_margin=info.get("profitMargins"),
            return_on_equity=info.get("returnOnEquity"),
            earnings_growth=info.get("earningsGrowth"),
            revenue_growth=info.get("revenueGrowth"),
            free_cashflow=info.get("freeCashflow"),
            dividend_yield=info.get("dividendYield"),
            beta=info.get("beta"),
            fifty_two_week_high=info.get("fiftyTwoWeekHigh"),
            fifty_two_week_low=info.get("fiftyTwoWeekLow"),
            analyst_target=info.get("targetMedianPrice"),
            recommendation=info.get("recommendationKey"),
            raw=info,
        )
