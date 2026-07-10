"""
Technical Analyst agent.

Responsible for:
  - Computing technical indicators from OHLCV data (SMA, RSI, MACD, volatility)
  - Using an LLM to interpret indicators into a narrative summary
  - Extracting structured buy/sell/hold signals from indicator patterns
  - Reporting confidence based on data quality and indicator clarity

Does NOT do:
  - Short-term price action tick-level analysis (handled by Market Data Analyst)
  - Fundamental analysis (handled by Fundamentals Analyst)
  - News/macro analysis (handled by News Intelligence Agent)
"""

from __future__ import annotations

import statistics
from typing import Any

from agents.base_agent import BaseAgent
from models import ResearchState, AgentAnalysis, Signal, ToolCallRecord
from services import get_market_data, MarketDataProvider, OHLCVBar


class TechnicalAnalyst(BaseAgent):
    """Analyses technical indicators (SMA, RSI, MACD, volatility)."""

    def __init__(
        self,
        llm: Any,
        market_data_provider: MarketDataProvider | None = None,
        name: str = "technical_analyst",
    ) -> None:
        super().__init__(
            name=name,
            description="Technical indicator analysis and trend/momentum assessment",
            llm=llm,
        )
        self._market_data = market_data_provider or get_market_data()

    #  Prompt

    def system_prompt(self) -> str:
        return (
            "You are a Technical Analyst specialising in chart patterns and indicators. "
            "Your job is to interpret technical metrics and produce a concise "
            "narrative about trend structure, momentum, and volatility.\n\n"
            "Focus on:\n"
            "1. Trend direction — short, medium, and long-term SMA alignment\n"
            "2. Momentum — RSI regime (overbought/oversold/neutral)\n"
            "3. MACD — signal crossovers and histogram direction\n"
            "4. Volatility regime — expanding or contracting\n"
            "5. Convergence/divergence between indicators\n\n"
            "Rules:\n"
            "- Reference specific numbers from the metrics provided.\n"
            "- Be concise (2-4 sentences max).\n"
            "- Do NOT discuss fundamentals, news, or fundamental ratios.\n"
            "- Stick to what the technical data objectively shows."
        )

    #  Core analysis

    def analyze(self, state: ResearchState) -> AgentAnalysis:
        tool_records: list[ToolCallRecord] = []
        symbol = state.symbol

        #  1. Fetch OHLCV data
        bars = self._safe_fetch(
            lambda: self._market_data.get_ohlcv(
                symbol, interval="1d", period="6mo", bars=60
            ),
            tool_records,
            "get_ohlcv",
            {"symbol": symbol, "interval": "1d", "period": "6mo"},
        )

        if not bars or len(bars) < 5:
            return AgentAnalysis(
                agent_name=self.name,
                symbol=symbol,
                agent_confidence=0.0,
                summary="Insufficient price data for technical analysis.",
                signals=[],
                reasoning=f"Only {len(bars) if bars else 0} bars available; need at least 5.",
                tool_calls=tool_records,
                metadata={"bar_count": len(bars) if bars else 0},
            )

        #  2. Compute deterministic indicators
        metrics = self._compute_indicators(bars)
        metrics_str = self._format_metrics(metrics, len(bars))

        #  3. LLM interpretation
        messages = self.build_messages(state) + [
            {
                "role": "user",
                "content": (
                    f"Technical indicators for {symbol}:\n\n"
                    f"{metrics_str}\n\n"
                    f"Data based on {len(bars)} daily bars.\n\n"
                    "Provide a technical analysis summary (2-4 sentences). "
                    "Then list 2-4 specific signals with direction "
                    "(bullish/bearish/neutral)."
                ),
            },
        ]
        llm_response = self.invoke_llm(messages)

        #  4. Build output
        signals = self._extract_signals(metrics)
        confidence = self._compute_confidence(bars, metrics)
        reasoning_parts = self._build_reasoning(metrics, bars)

        return AgentAnalysis(
            agent_name=self.name,
            symbol=symbol,
            agent_confidence=confidence,
            summary=llm_response,
            signals=signals,
            reasoning=" | ".join(reasoning_parts),
            tool_calls=tool_records,
            metadata={
                "bar_count": len(bars),
                "indicators_computed": list(metrics.keys()),
            },
        )

    #  Indicator computation (deterministic math, no LLM)

    def _compute_indicators(self, bars: list[OHLCVBar]) -> dict[str, Any]:
        """Compute technical indicators from OHLCV data."""
        closes = [b.close for b in bars]
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        n = len(bars)
        metrics: dict[str, Any] = {}

        last_close = closes[-1]

        #  Simple Moving Averages
        if n >= 5:
            metrics["sma_5"] = round(statistics.mean(closes[-5:]), 2)
        if n >= 10:
            metrics["sma_10"] = round(statistics.mean(closes[-10:]), 2)
        if n >= 20:
            metrics["sma_20"] = round(statistics.mean(closes[-20:]), 2)

        #  Price vs SMA
        for period in (5, 10, 20):
            key = f"sma_{period}"
            if key in metrics:
                pct = (last_close - metrics[key]) / metrics[key] * 100
                metrics[f"price_vs_sma{period}_pct"] = round(pct, 2)

        #  SMA crossover (bullish/bearish/neutral)
        if n >= 10 and "sma_5" in metrics and "sma_10" in metrics:
            sma5_before = statistics.mean(closes[-6:-1])  # 5 bars ending before last
            sma10_before = (
                statistics.mean(closes[-11:-1]) if n >= 11 else metrics["sma_10"]
            )
            curr_gap = metrics["sma_5"] - metrics["sma_10"]
            prev_gap = sma5_before - sma10_before
            if curr_gap > 0 and prev_gap <= 0:
                metrics["sma_crossover"] = "bullish"
            elif curr_gap < 0 and prev_gap >= 0:
                metrics["sma_crossover"] = "bearish"
            else:
                metrics["sma_crossover"] = "neutral"

        #  RSI (14-period)
        if n >= 15:
            metrics["rsi_14"] = round(self._compute_rsi(closes, 14), 1)

        #  MACD (12, 26, 9)
        if n >= 26:
            macd_line = self._ema(closes, 12) - self._ema(closes, 26)
            metrics["macd_line"] = round(macd_line, 2)

            # Build MACD series for signal line
            macd_series = []
            for i in range(25, n):
                sub = closes[: i + 1]
                e12 = self._ema(sub, 12)
                e26 = self._ema(sub, 26)
                macd_series.append(e12 - e26)

            if len(macd_series) >= 9:
                signal = self._ema(macd_series, 9)
                metrics["macd_signal"] = round(signal, 2)
                hist = macd_line - signal
                metrics["macd_histogram"] = round(hist, 2)

                # Crossover detection
                if len(macd_series) >= 10:
                    prev_signal = self._ema(macd_series[:-1], 9)
                    prev_hist = macd_series[-2] - prev_signal
                    if prev_hist <= 0 < hist:
                        metrics["macd_cross"] = "bullish"
                    elif prev_hist >= 0 > hist:
                        metrics["macd_cross"] = "bearish"
                    else:
                        metrics["macd_cross"] = "neutral"
                else:
                    metrics["macd_cross"] = "neutral"

        #  Volatility (daily returns std dev)
        if n >= 5:
            daily_returns = [
                (closes[i] - closes[i - 1]) / closes[i - 1]
                for i in range(1, n)
                if closes[i - 1] > 0
            ]
            if daily_returns:
                vol = statistics.stdev(daily_returns) * 100
                metrics["daily_volatility_pct"] = round(vol, 2)
                metrics["annualized_volatility_pct"] = round(vol * (252**0.5), 2)

        return metrics

    #  Signal extraction (rule-based, no LLM)

    def _extract_signals(self, metrics: dict[str, Any]) -> list[Signal]:
        """Convert computed indicators into structured signals."""
        signals: list[Signal] = []

        # SMA trend alignment
        price_vs_sma5 = metrics.get("price_vs_sma5_pct")
        price_vs_sma10 = metrics.get("price_vs_sma10_pct")
        price_vs_sma20 = metrics.get("price_vs_sma20_pct")

        # Count how many SMAs price is above (bullish) vs below (bearish)
        above = sum(
            1
            for v in [price_vs_sma5, price_vs_sma10, price_vs_sma20]
            if v is not None and v > 0
        )
        below = sum(
            1
            for v in [price_vs_sma5, price_vs_sma10, price_vs_sma20]
            if v is not None and v < 0
        )
        if above >= 2:
            signals.append(
                Signal(
                    name="sma_trend",
                    value=f"price above {above} SMA(s)",
                    direction="bullish",
                )
            )
        elif below >= 2:
            signals.append(
                Signal(
                    name="sma_trend",
                    value=f"price below {below} SMA(s)",
                    direction="bearish",
                )
            )
        elif above > 0 or below > 0:
            signals.append(
                Signal(
                    name="sma_trend",
                    value="price mixed vs SMAs",
                    direction="neutral",
                )
            )

        # SMA crossover
        cross = metrics.get("sma_crossover")
        if cross and cross != "neutral":
            signals.append(
                Signal(
                    name="sma_crossover",
                    value=f"SMA5 {cross} cross SMA10",
                    direction=cross,
                )
            )

        # RSI
        rsi = metrics.get("rsi_14")
        if rsi is not None:
            if rsi >= 70:
                signals.append(
                    Signal(
                        name="rsi_overbought",
                        value=f"RSI {rsi}",
                        direction="bearish",
                    )
                )
            elif rsi <= 30:
                signals.append(
                    Signal(
                        name="rsi_oversold",
                        value=f"RSI {rsi}",
                        direction="bullish",
                    )
                )
            else:
                signals.append(
                    Signal(
                        name="rsi_neutral",
                        value=f"RSI {rsi}",
                        direction="neutral",
                    )
                )

        # MACD cross
        macd_cross = metrics.get("macd_cross")
        if macd_cross and macd_cross != "neutral":
            signals.append(
                Signal(
                    name="macd_cross",
                    value=f"MACD {macd_cross} cross",
                    direction=macd_cross,
                )
            )

        # MACD histogram direction
        hist = metrics.get("macd_histogram")
        if hist is not None:
            direction = "bullish" if hist > 0 else "bearish"
            signals.append(
                Signal(
                    name="macd_momentum",
                    value=f"histogram {hist:+.2f}",
                    direction=direction,
                )
            )

        # Volatility
        vol = metrics.get("daily_volatility_pct")
        if vol is not None:
            if vol > 3.0:
                signals.append(
                    Signal(
                        name="high_volatility",
                        value=f"{vol}% daily",
                        direction="neutral",
                    )
                )
            elif vol < 0.8:
                signals.append(
                    Signal(
                        name="low_volatility",
                        value=f"{vol}% daily",
                        direction="neutral",
                    )
                )

        return signals

    #  Confidence

    def _compute_confidence(
        self, bars: list[OHLCVBar], metrics: dict[str, Any]
    ) -> float:
        """Confidence (0.0–1.0) based on data quantity and indicator diversity."""
        if not bars or len(bars) < 5:
            return 0.0

        score = 0.40

        # Data quantity
        n = len(bars)
        if n >= 50:
            score += 0.20
        elif n >= 30:
            score += 0.15
        elif n >= 20:
            score += 0.10
        elif n >= 10:
            score += 0.05

        # Indicator diversity (how many distinct indicators computed)
        indicator_count = len(metrics)
        if indicator_count >= 10:
            score += 0.20
        elif indicator_count >= 7:
            score += 0.15
        elif indicator_count >= 4:
            score += 0.10

        # Clear SMA trend
        price_vs_sma = [
            metrics.get(k) for k in ("price_vs_sma5_pct", "price_vs_sma10_pct")
        ]
        if all(v is not None and abs(v) > 1.0 for v in price_vs_sma):
            score += 0.10

        # RSI available (core indicator)
        if "rsi_14" in metrics:
            score += 0.05

        return max(0.0, min(1.0, score))

    #  Helpers

    def _build_reasoning(
        self, metrics: dict[str, Any], bars: list[OHLCVBar]
    ) -> list[str]:
        parts = [
            f"Computed {len(metrics)} indicators from {len(bars)} daily bars.",
        ]
        for sma_key in ("sma_5", "sma_10", "sma_20"):
            if sma_key in metrics:
                vs = metrics.get(f"price_vs_{sma_key}_pct")
                tag = (
                    f"price is {vs:+.2f}% vs {sma_key}"
                    if vs is not None
                    else f"{sma_key}={metrics[sma_key]}"
                )
                parts.append(tag)
        rsi = metrics.get("rsi_14")
        if rsi is not None:
            parts.append(f"RSI(14)={rsi}")
        macd_hist = metrics.get("macd_histogram")
        if macd_hist is not None:
            parts.append(f"MACD histogram={macd_hist:+.2f}")
        cross = metrics.get("macd_cross")
        if cross:
            parts.append(f"MACD cross={cross}")
        vol = metrics.get("daily_volatility_pct")
        if vol is not None:
            parts.append(f"Volatility={vol}% daily")
        return parts

    @staticmethod
    def _ema(values: list[float], period: int) -> float:
        """Exponential Moving Average."""
        multiplier = 2.0 / (period + 1)
        # Seed with SMA
        ema = statistics.mean(values[:period])
        for v in values[period:]:
            ema = (v - ema) * multiplier + ema
        return ema

    @staticmethod
    def _compute_rsi(closes: list[float], period: int) -> float:
        """Relative Strength Index."""
        gains, losses = [], []
        for i in range(len(closes) - period, len(closes)):
            change = closes[i] - closes[i - 1]
            gains.append(max(change, 0))
            losses.append(max(-change, 0))
        avg_gain = statistics.mean(gains)
        avg_loss = statistics.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

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
