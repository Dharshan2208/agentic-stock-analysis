"""
Market Data Analyst agent.

Responsible for:
  - Fetching real-time and historical price/volume data via the service layer
  - Computing objective price-action metrics (% change, volume ratios, gaps, ranges)
  - Using an LLM to interpret metrics into a narrative summary
  - Extracting structured buy/sell/hold signals
  - Reporting confidence based on data quality and signal strength

Does NOT do:
  - Fundamental analysis (handled by Fundamentals Analyst)
  - Long-term technical indicators (handled by Technical Analyst)
  - News/macro analysis (handled by News Intelligence Agent)
"""

from __future__ import annotations

import statistics
from typing import Any

from agents.base_agent import BaseAgent
from models import ResearchState, AgentAnalysis, Signal, ToolCallRecord
from services import get_market_data, MarketDataProvider, OHLCVBar


class MarketDataAnalyst(BaseAgent):
    """Analyzes short-term price action, volume, and intraday/daily market data."""

    def __init__(
        self,
        llm: Any,
        market_data_provider: MarketDataProvider | None = None,
        name: str = "market_data_analyst",
    ) -> None:
        super().__init__(
            name=name,
            description="Short-term price action and volume analysis",
            llm=llm,
        )
        # Dependency injection: use provided provider or global default
        self._market_data = market_data_provider or get_market_data()

    # Prompt

    def system_prompt(self) -> str:
        return (
            "You are a Market Data Analyst specializing in short-term price action. "
            "Your job is to interpret raw market metrics and produce a concise "
            "narrative about what the price and volume data indicate.\n\n"
            "Focus on:\n"
            "1. Recent price momentum (up/down/sideways)\n"
            "2. Volume confirmation or divergence\n"
            "3. Key support/resistance levels from recent range\n"
            "4. Gap fills or breakouts\n"
            "5. Volatility assessment\n\n"
            "Rules:\n"
            "- Reference specific numbers from the metrics provided.\n"
            "- Be concise (2-4 sentences max).\n"
            "- Do NOT discuss fundamentals, news, or long-term trends.\n"
            "- Stick to what the price and volume data objectively show."
        )

    # Core analysis

    def analyze(self, state: ResearchState) -> AgentAnalysis:
        tool_records: list[ToolCallRecord] = []
        symbol = state.symbol

        # ── 1. Fetch data from provider ──
        daily_bars = self._safe_fetch(
            lambda: self._market_data.get_ohlcv(
                symbol, interval="1d", period="20d", bars=15
            ),
            tool_records,
            "get_ohlcv_daily",
            {"symbol": symbol, "interval": "1d", "period": "20d"},
        )
        intraday_bars = self._safe_fetch(
            lambda: self._market_data.get_ohlcv(
                symbol, interval="1m", period="1d", bars=20
            ),
            tool_records,
            "get_ohlcv_intraday",
            {"symbol": symbol, "interval": "1m", "period": "1d"},
        )
        current_price = self._safe_fetch(
            lambda: self._market_data.get_current_price(symbol),
            tool_records,
            "get_current_price",
            {"symbol": symbol},
        )

        # ── 2. Select best available data ──
        primary_bars = daily_bars if daily_bars else intraday_bars
        if not primary_bars:
            return AgentAnalysis(
                agent_name=self.name,
                symbol=symbol,
                agent_confidence=0.0,
                summary="No price data available for analysis.",
                signals=[],
                reasoning="Market data provider returned no data.",
                tool_calls=tool_records,
                metadata={"data_source": "none"},
            )

        # ── 3. Compute objective metrics ──
        metrics = self._compute_price_metrics(primary_bars)
        metrics_str = self._format_metrics(metrics, len(primary_bars))

        # ── 4. LLM interpretation ──
        messages = self.build_messages(state) + [
            {
                "role": "user",
                "content": (
                    f"Raw market data metrics for {symbol}:\n\n"
                    f"{metrics_str}\n\n"
                    f"Data source: {'daily' if daily_bars else 'intraday'} bars "
                    f"({len(primary_bars)} candles available).\n"
                    f"Current price: {current_price or 'N/A'}\n\n"
                    "Provide a brief price-action analysis (2-4 sentences). "
                    "Then list 2-4 specific signals with direction "
                    "(bullish/bearish/neutral)."
                ),
            },
        ]
        llm_response = self.invoke_llm(messages)

        # ── 5. Build structured output ──
        signals = self._extract_signals(metrics)
        confidence = self._compute_confidence(primary_bars, metrics)
        reasoning_parts = self._build_reasoning(metrics, primary_bars)

        return AgentAnalysis(
            agent_name=self.name,
            symbol=symbol,
            agent_confidence=confidence,
            summary=llm_response,
            signals=signals,
            reasoning=" | ".join(reasoning_parts),
            tool_calls=tool_records,
            metadata={
                "bar_count": len(primary_bars),
                "data_source": "daily" if daily_bars else "intraday",
                "metrics_computed": list(metrics.keys()),
                "current_price": current_price,
            },
        )

    # Metric computation (pure math, no LLM)

    def _compute_price_metrics(self, bars: list[OHLCVBar]) -> dict[str, Any]:
        """Compute objective price-action metrics from OHLCV bars."""
        if not bars:
            return {}

        closes = [b.close for b in bars]
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        volumes = [b.volume for b in bars]
        n = len(bars)
        metrics: dict[str, Any] = {}

        #  Price changes
        if n >= 2:
            metrics["change_1d_pct"] = round(
                (closes[-1] - closes[-2]) / closes[-2] * 100, 2
            )
        if n >= 5:
            metrics["change_5d_pct"] = round(
                (closes[-1] - closes[-5]) / closes[-5] * 100, 2
            )
        if n >= 10:
            metrics["change_10d_pct"] = round(
                (closes[-1] - closes[-10]) / closes[-10] * 100, 2
            )

        #  Range
        metrics["high_10d"] = round(max(highs[-10:]), 2) if n >= 1 else None
        metrics["low_10d"] = round(min(lows[-10:]), 2) if n >= 1 else None
        if n >= 1 and metrics["low_10d"] and metrics["low_10d"] > 0:
            metrics["range_10d_pct"] = round(
                (metrics["high_10d"] - metrics["low_10d"]) / metrics["low_10d"] * 100, 2
            )
        # Position in range (0 = at low, 100 = at high)
        if (
            n >= 1
            and metrics["high_10d"] is not None
            and metrics["low_10d"] is not None
        ):
            if metrics["high_10d"] != metrics["low_10d"]:
                metrics["range_position_pct"] = round(
                    (closes[-1] - metrics["low_10d"])
                    / (metrics["high_10d"] - metrics["low_10d"])
                    * 100,
                    1,
                )

        #  Volume
        metrics["latest_volume"] = int(volumes[-1])
        if n >= 2:
            avg_vol = statistics.mean(volumes[-6:-1])
            metrics["avg_volume_5d"] = int(avg_vol)
            metrics["volume_vs_avg_5d"] = (
                round(volumes[-1] / avg_vol, 2) if avg_vol > 0 else None
            )

        #  Gap detection
        if n >= 2:
            prev_close = closes[-2]
            if prev_close > 0:
                gap = (bars[-1].open - prev_close) / prev_close * 100
                metrics["gap_pct"] = round(gap, 2)
                metrics["gap_direction"] = (
                    "up" if gap > 0.3 else ("down" if gap < -0.3 else "none")
                )

        #  Volatility
        if n >= 1:
            daily_ranges = [(h - l) / l * 100 for h, l in zip(highs, lows) if l > 0]
            if daily_ranges:
                metrics["avg_daily_range_pct"] = round(statistics.mean(daily_ranges), 2)

        #  Simple trend (first half vs second half)
        if n >= 10:
            mid = n // 2
            first_half = statistics.mean(closes[:mid])
            second_half = statistics.mean(closes[mid:])
            if first_half > 0:
                metrics["trend_strength_pct"] = round(
                    (second_half - first_half) / first_half * 100, 2
                )

        return metrics

    # Signal extraction (rule-based, no LLM)

    def _extract_signals(self, metrics: dict[str, Any]) -> list[Signal]:
        """Convert computed metrics into structured signals."""
        signals: list[Signal] = []

        # 1-day momentum
        chg_1d = metrics.get("change_1d_pct", 0) or 0
        if abs(chg_1d) > 1.0:
            signals.append(
                Signal(
                    name="price_momentum_1d",
                    value=f"{chg_1d:+.2f}%",
                    direction="bullish" if chg_1d > 0 else "bearish",
                )
            )

        # 5-day momentum
        chg_5d = metrics.get("change_5d_pct", 0) or 0
        if abs(chg_5d) > 2.0:
            signals.append(
                Signal(
                    name="price_momentum_5d",
                    value=f"{chg_5d:+.2f}%",
                    direction="bullish" if chg_5d > 0 else "bearish",
                )
            )

        # Volume spike
        vol = metrics.get("volume_vs_avg_5d")
        if vol and vol > 1.3:
            direction = "bullish" if chg_1d > 0 else "bearish"
            signals.append(
                Signal(
                    name="volume_spike",
                    value=f"{vol}x avg",
                    direction=direction,
                )
            )

        # Gap
        gap_dir = metrics.get("gap_direction", "none")
        if gap_dir != "none":
            signals.append(
                Signal(
                    name=f"gap_{gap_dir}",
                    value=f"{metrics.get('gap_pct', 0):+.2f}%",
                    direction=gap_dir,
                )
            )

        # Range position
        rp = metrics.get("range_position_pct")
        if rp is not None:
            if rp >= 80:
                signals.append(
                    Signal(
                        name="near_range_high",
                        value=f"{rp}% of range",
                        direction="bullish",
                    )
                )
            elif rp <= 20:
                signals.append(
                    Signal(
                        name="near_range_low",
                        value=f"{rp}% of range",
                        direction="bearish",
                    )
                )

        # Trend
        trend = metrics.get("trend_strength_pct")
        if trend and abs(trend) > 2.0:
            signals.append(
                Signal(
                    name="price_trend",
                    value=f"{trend:+.2f}%",
                    direction="bullish" if trend > 0 else "bearish",
                )
            )

        return signals

    # Confidence

    def _compute_confidence(
        self, bars: list[OHLCVBar], metrics: dict[str, Any]
    ) -> float:
        """Confidence (0.0–1.0) based on data quality and signal strength."""
        if not bars or len(bars) < 2:
            return 0.0

        score = 0.50

        # Data quantity
        if len(bars) >= 15:
            score += 0.15
        elif len(bars) >= 10:
            score += 0.10
        elif len(bars) >= 5:
            score += 0.05

        # Signal strength
        chg_1d = abs(metrics.get("change_1d_pct", 0) or 0)
        if chg_1d > 2.0:
            score += 0.10
        elif chg_1d > 1.0:
            score += 0.05

        vol = metrics.get("volume_vs_avg_5d")
        if vol and vol > 1.5:
            score += 0.10
        elif vol and vol > 1.2:
            score += 0.05

        return max(0.0, min(1.0, score))

    # Helpers

    def _build_reasoning(
        self, metrics: dict[str, Any], bars: list[OHLCVBar]
    ) -> list[str]:
        parts = [
            f"Computed {len(metrics)} metrics from {len(bars)} candles.",
        ]
        lo = metrics.get("low_10d")
        hi = metrics.get("high_10d")
        if lo is not None and hi is not None:
            parts.append(f"10d range: {lo} – {hi}")
        vol = metrics.get("volume_vs_avg_5d")
        if vol is not None:
            parts.append(f"Volume: {vol}x 5d avg")
        gap = metrics.get("gap_direction")
        if gap and gap != "none":
            parts.append(f"Gap: {metrics['gap_pct']:+.2f}% ({gap})")
        trend = metrics.get("trend_strength_pct")
        if trend:
            parts.append(f"Trend: {trend:+.2f}% (1st half vs 2nd half)")
        return parts

    def _safe_fetch(self, fn, records: list, name: str, inp: dict) -> Any:
        """Call provider method, record success/failure."""
        try:
            result = fn()
            records.append(
                ToolCallRecord(
                    tool_name=name,
                    input=inp,
                    output_summary=str(result)[:200] if result else "empty",
                    success=True,
                )
            )
            return result
        except Exception as e:
            records.append(
                ToolCallRecord(
                    tool_name=name,
                    input=inp,
                    output_summary="",
                    success=False,
                    error=str(e),
                )
            )
            return None

    @staticmethod
    def _format_metrics(metrics: dict[str, Any], bar_count: int) -> str:
        lines = [f"Bars: {bar_count}"]
        for k, v in metrics.items():
            if v is not None:
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)
