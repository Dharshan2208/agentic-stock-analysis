"""
Fundamentals Analyst agent.

Responsible for:
  - Fetching company profile and fundamental financial metrics
  - Computing derived valuation, profitability, growth, and financial health scores
  - Using an LLM to interpret fundamentals into a narrative summary
  - Extracting structured signals about valuation, quality, and growth
  - Reporting confidence based on data completeness and signal clarity

Does NOT do:
  - Short-term price action (handled by Market Data Analyst)
  - Long-term technical indicators (handled by Technical Analyst)
  - News/macro analysis (handled by News Intelligence Agent)
"""

from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent
from models import ResearchState, AgentAnalysis, Signal, ToolCallRecord
from services import get_fundamentals, FundamentalsProvider, CompanyProfile, Fundamentals


# ── Rule-based thresholds ──

PE_LOW = 10.0
PE_HIGH = 30.0
DE_HIGH = 2.0
CR_LOW = 1.0
ROE_HIGH = 0.20  # 20%
PROFIT_MARGIN_HIGH = 0.15  # 15%
GROWTH_POSITIVE = 0.05  # 5%


class FundamentalsAnalyst(BaseAgent):
    """Analyzes company fundamentals: valuation, profitability, financial health, and growth."""

    def __init__(
        self,
        llm: Any,
        fundamentals_provider: FundamentalsProvider | None = None,
        name: str = "fundamentals_analyst",
    ) -> None:
        super().__init__(
            name=name,
            description="Company fundamentals, valuation, and financial health analysis",
            llm=llm,
        )
        self._fundamentals = fundamentals_provider or get_fundamentals()

    # Prompt

    def system_prompt(self) -> str:
        return (
            "You are a Fundamentals Analyst specializing in company financial health "
            "and valuation. Your job is to interpret fundamental metrics and produce "
            "a concise narrative about the company's financial position.\n\n"
            "Focus on:\n"
            "1. Valuation — is the company cheap or expensive vs earnings, book value, and growth?\n"
            "2. Profitability — how efficiently does the company generate profits?\n"
            "3. Financial health — debt levels, liquidity, cash flow adequacy\n"
            "4. Growth trajectory — are earnings and revenue expanding?\n"
            "5. Analyst consensus — what do sell-side analysts think?\n\n"
            "Rules:\n"
            "- Reference specific numbers from the metrics provided.\n"
            "- Be concise (3-5 sentences max).\n"
            "- Do NOT discuss short-term trading, news headlines, or chart patterns.\n"
            "- Stick to what the fundamentals objectively show."
        )

    # Core analysis

    def analyze(self, state: ResearchState) -> AgentAnalysis:
        tool_records: list[ToolCallRecord] = []
        symbol = state.symbol

        # ── 1. Fetch data ──
        profile = self._safe_fetch(
            lambda: self._fundamentals.get_company_profile(symbol),
            tool_records,
            "get_company_profile",
            {"symbol": symbol},
        )
        fundamentals = self._safe_fetch(
            lambda: self._fundamentals.get_fundamentals(symbol),
            tool_records,
            "get_fundamentals",
            {"symbol": symbol},
        )

        if not profile and not fundamentals:
            return AgentAnalysis(
                agent_name=self.name,
                symbol=symbol,
                agent_confidence=0.0,
                summary="No fundamental data available for analysis.",
                signals=[],
                reasoning="Fundamentals provider returned no data.",
                tool_calls=tool_records,
                metadata={"data_source": "none"},
            )

        # ── 2. Compute derived metrics ──
        metrics = self._compute_fundamental_metrics(profile, fundamentals)
        sector_info = profile.sector if profile else state.sector
        metrics_str = self._format_metrics(metrics)

        # ── 3. LLM interpretation ──
        messages = self.build_messages(state) + [
            {
                "role": "user",
                "content": (
                    f"Fundamental metrics for {symbol} "
                    f"({sector_info or 'unknown sector'}):\n\n"
                    f"{metrics_str}\n\n"
                    f"Company: {profile.name if profile else 'N/A'}\n"
                    f"Sector: {sector_info or 'N/A'}\n"
                    f"Industry: {profile.industry if profile else 'N/A'}\n\n"
                    "Provide a fundamental analysis (3-5 sentences). "
                    "Then list 3-5 specific signals with direction "
                    "(bullish/bearish/neutral)."
                ),
            },
        ]
        llm_response = self.invoke_llm(messages)

        # ── 4. Build structured output ──
        signals = self._extract_signals(metrics, fundamentals)
        confidence = self._compute_confidence(fundamentals, metrics)
        reasoning_parts = self._build_reasoning(metrics, profile, fundamentals)

        return AgentAnalysis(
            agent_name=self.name,
            symbol=symbol,
            agent_confidence=confidence,
            summary=llm_response,
            signals=signals,
            reasoning=" | ".join(reasoning_parts),
            tool_calls=tool_records,
            metadata={
                "metrics_computed": list(metrics.keys()),
                "has_profile": profile is not None,
                "has_fundamentals": fundamentals is not None,
                "sector": sector_info,
                "company_name": profile.name if profile else None,
            },
        )

    # Derived metric computation

    def _compute_fundamental_metrics(
        self,
        profile: CompanyProfile | None,
        fundamentals: Fundamentals | None,
    ) -> dict[str, Any]:
        """Compute derived valuation, health, and quality metrics."""
        metrics: dict[str, Any] = {}
        f = fundamentals
        if not f:
            return metrics

        # ── Valuation ──
        if f.trailing_pe is not None:
            metrics["trailing_pe"] = round(f.trailing_pe, 2)
        if f.forward_pe is not None:
            metrics["forward_pe"] = round(f.forward_pe, 2)

        # PE comparison
        if f.trailing_pe is not None and f.forward_pe is not None:
            pe_diff = f.forward_pe - f.trailing_pe
            metrics["pe_vs_forward_pct"] = round(
                (f.forward_pe - f.trailing_pe) / f.trailing_pe * 100, 1
            )
            metrics["pe_trend"] = (
                "contracting" if pe_diff < 0
                else "expanding" if pe_diff > 0
                else "stable"
            )

        # PE categorization
        if f.trailing_pe is not None:
            if f.trailing_pe < PE_LOW:
                metrics["pe_category"] = "low"
            elif f.trailing_pe > PE_HIGH:
                metrics["pe_category"] = "high"
            else:
                metrics["pe_category"] = "moderate"

        # Price-to-Book
        if f.price_to_book is not None:
            metrics["price_to_book"] = round(f.price_to_book, 2)

        # Analyst target
        if f.analyst_target is not None and profile and profile.market_cap:
            # We don't have current price here, so we note the target exists
            metrics["analyst_target"] = round(f.analyst_target, 2)
            metrics["has_analyst_target"] = True

        # Analyst recommendation
        if f.recommendation:
            metrics["analyst_recommendation"] = f.recommendation

        # ── Financial Health ──
        health_score = 50.0
        health_factors = []

        if f.debt_to_equity is not None:
            metrics["debt_to_equity"] = round(f.debt_to_equity, 2)
            if f.debt_to_equity < 0.5:
                health_score += 15
                health_factors.append("very_low_debt")
            elif f.debt_to_equity < 1.0:
                health_score += 10
                health_factors.append("low_debt")
            elif f.debt_to_equity > DE_HIGH:
                health_score -= 15
                health_factors.append("high_debt")
            else:
                health_factors.append("moderate_debt")

        if f.current_ratio is not None:
            metrics["current_ratio"] = round(f.current_ratio, 2)
            if f.current_ratio > 2.0:
                health_score += 10
                health_factors.append("strong_liquidity")
            elif f.current_ratio > CR_LOW:
                health_score += 5
                health_factors.append("adequate_liquidity")
            else:
                health_score -= 10
                health_factors.append("low_liquidity")

        if f.free_cashflow is not None:
            metrics["free_cashflow_b"] = round(f.free_cashflow / 1e9, 2)
            if f.free_cashflow > 0:
                health_score += 10
                health_factors.append("positive_fcf")
            else:
                health_score -= 10
                health_factors.append("negative_fcf")

        metrics["financial_health_score"] = round(max(0, min(100, health_score)))
        metrics["financial_health_factors"] = health_factors

        # ── Profitability ──
        quality_score = 50.0
        quality_factors = []

        if f.return_on_equity is not None:
            metrics["roe_pct"] = round(f.return_on_equity * 100, 1)
            if f.return_on_equity > ROE_HIGH:
                quality_score += 20
                quality_factors.append("high_roe")
            elif f.return_on_equity > 0.10:
                quality_score += 10
                quality_factors.append("moderate_roe")
            else:
                quality_score -= 10
                quality_factors.append("low_roe")

        if f.profit_margin is not None:
            metrics["profit_margin_pct"] = round(f.profit_margin * 100, 1)
            if f.profit_margin > PROFIT_MARGIN_HIGH:
                quality_score += 15
                quality_factors.append("high_margins")
            elif f.profit_margin > 0.05:
                quality_score += 5
                quality_factors.append("moderate_margins")
            else:
                quality_score -= 10
                quality_factors.append("low_margins")

        metrics["quality_score"] = round(max(0, min(100, quality_score)))
        metrics["quality_factors"] = quality_factors

        # ── Growth ──
        growth_score = 50.0
        growth_factors = []

        if f.earnings_growth is not None:
            metrics["earnings_growth_pct"] = round(f.earnings_growth * 100, 1)
            if f.earnings_growth > GROWTH_POSITIVE:
                growth_score += 20
                growth_factors.append("positive_earnings_growth")
            else:
                growth_score -= 15
                growth_factors.append("negative_earnings_growth")

        if f.revenue_growth is not None:
            metrics["revenue_growth_pct"] = round(f.revenue_growth * 100, 1)
            if f.revenue_growth > GROWTH_POSITIVE:
                growth_score += 15
                growth_factors.append("positive_revenue_growth")
            else:
                growth_score -= 10
                growth_factors.append("negative_revenue_growth")

        metrics["growth_score"] = round(max(0, min(100, growth_score)))
        metrics["growth_factors"] = growth_factors

        # ── Beta (risk) ──
        if f.beta is not None:
            metrics["beta"] = round(f.beta, 2)
            if f.beta < 0.8:
                metrics["volatility_category"] = "low"
            elif f.beta < 1.3:
                metrics["volatility_category"] = "moderate"
            else:
                metrics["volatility_category"] = "high"

        # ── Dividend ──
        if f.dividend_yield is not None and f.dividend_yield > 0:
            metrics["dividend_yield_pct"] = round(f.dividend_yield * 100, 2)

        return metrics

    # Signal extraction

    def _extract_signals(
        self,
        metrics: dict[str, Any],
        fundamentals: Fundamentals | None,
    ) -> list[Signal]:
        """Convert computed metrics into structured signals."""
        signals: list[Signal] = []

        # ── Valuation signals ──
        pe_cat = metrics.get("pe_category")
        if pe_cat == "low":
            signals.append(Signal(
                name="low_pe", value=str(metrics.get("trailing_pe", "?")),
                direction="bullish",
            ))
        elif pe_cat == "high":
            signals.append(Signal(
                name="high_pe", value=str(metrics.get("trailing_pe", "?")),
                direction="bearish",
            ))

        pe_trend = metrics.get("pe_trend")
        if pe_trend == "contracting":
            signals.append(Signal(
                name="pe_contracting",
                value=f"{metrics.get('pe_vs_forward_pct', 0):+.1f}%",
                direction="bullish",
            ))
        elif pe_trend == "expanding":
            signals.append(Signal(
                name="pe_expanding",
                value=f"{metrics.get('pe_vs_forward_pct', 0):+.1f}%",
                direction="bearish",
            ))

        # Analyst recommendation
        rec = metrics.get("analyst_recommendation")
        if rec:
            rec_map = {
                "buy": "bullish", "strong_buy": "bullish",
                "outperform": "bullish", "overweight": "bullish",
                "hold": "neutral", "neutral": "neutral",
                "underperform": "bearish", "sell": "bearish",
            }
            direction = rec_map.get(rec.lower(), "neutral")
            if direction != "neutral":
                signals.append(Signal(
                    name=f"analyst_{rec}", value=rec, direction=direction,
                ))

        # ── Financial health signals ──
        health = metrics.get("financial_health_score", 50)
        if health >= 75:
            signals.append(Signal(
                name="strong_financial_health",
                value=f"score={health}", direction="bullish",
            ))
        elif health <= 35:
            signals.append(Signal(
                name="weak_financial_health",
                value=f"score={health}", direction="bearish",
            ))

        factors = metrics.get("financial_health_factors", [])
        if "very_low_debt" in factors:
            signals.append(Signal(
                name="low_debt",
                value=str(metrics.get("debt_to_equity", "?")),
                direction="bullish",
            ))
        if "high_debt" in factors:
            signals.append(Signal(
                name="high_debt",
                value=str(metrics.get("debt_to_equity", "?")),
                direction="bearish",
            ))
        if "positive_fcf" in factors:
            signals.append(Signal(
                name="positive_free_cashflow",
                value=f"${metrics.get('free_cashflow_b', '?')}B",
                direction="bullish",
            ))
        if "negative_fcf" in factors:
            signals.append(Signal(
                name="negative_free_cashflow",
                value=f"${metrics.get('free_cashflow_b', '?')}B",
                direction="bearish",
            ))

        # ── Profitability signals ──
        quality = metrics.get("quality_score", 50)
        if quality >= 75:
            signals.append(Signal(
                name="strong_profitability",
                value=f"score={quality}", direction="bullish",
            ))
        elif quality <= 35:
            signals.append(Signal(
                name="weak_profitability",
                value=f"score={quality}", direction="bearish",
            ))

        qf = metrics.get("quality_factors", [])
        if "high_roe" in qf:
            signals.append(Signal(
                name="high_roe",
                value=f"{metrics.get('roe_pct', '?')}%",
                direction="bullish",
            ))
        if "high_margins" in qf:
            signals.append(Signal(
                name="high_profit_margins",
                value=f"{metrics.get('profit_margin_pct', '?')}%",
                direction="bullish",
            ))

        # ── Growth signals ──
        growth = metrics.get("growth_score", 50)
        if growth >= 70:
            signals.append(Signal(
                name="strong_growth",
                value=f"score={growth}", direction="bullish",
            ))
        elif growth <= 35:
            signals.append(Signal(
                name="weak_growth",
                value=f"score={growth}", direction="bearish",
            ))

        gf = metrics.get("growth_factors", [])
        if "positive_earnings_growth" in gf:
            signals.append(Signal(
                name="positive_earnings_growth",
                value=f"{metrics.get('earnings_growth_pct', '?')}%",
                direction="bullish",
            ))
        if "negative_earnings_growth" in gf:
            signals.append(Signal(
                name="negative_earnings_growth",
                value=f"{metrics.get('earnings_growth_pct', '?')}%",
                direction="bearish",
            ))
        if "positive_revenue_growth" in gf:
            signals.append(Signal(
                name="positive_revenue_growth",
                value=f"{metrics.get('revenue_growth_pct', '?')}%",
                direction="bullish",
            ))
        if "negative_revenue_growth" in gf:
            signals.append(Signal(
                name="negative_revenue_growth",
                value=f"{metrics.get('revenue_growth_pct', '?')}%",
                direction="bearish",
            ))

        # ── Dividend signal ──
        dy = metrics.get("dividend_yield_pct")
        if dy and dy > 2.0:
            signals.append(Signal(
                name="dividend_yield",
                value=f"{dy}%", direction="bullish",
            ))

        return signals

    # Confidence

    def _compute_confidence(
        self,
        fundamentals: Fundamentals | None,
        metrics: dict[str, Any],
    ) -> float:
        """Confidence (0.0–1.0) based on data completeness and signal clarity."""
        if not fundamentals:
            return 0.0

        score = 0.40  # base

        # Data completeness
        key_fields = [
            fundamentals.trailing_pe, fundamentals.forward_pe,
            fundamentals.debt_to_equity, fundamentals.return_on_equity,
            fundamentals.earnings_growth, fundamentals.revenue_growth,
        ]
        present = sum(1 for f in key_fields if f is not None)
        score += (present / len(key_fields)) * 0.25

        # Signal clarity
        health = metrics.get("financial_health_score", 50)
        if health >= 70 or health <= 30:
            score += 0.10

        quality = metrics.get("quality_score", 50)
        if quality >= 70 or quality <= 30:
            score += 0.10

        growth = metrics.get("growth_score", 50)
        if growth >= 70 or growth <= 30:
            score += 0.10

        # Analyst consensus boosts confidence
        if metrics.get("analyst_recommendation"):
            score += 0.05

        return max(0.0, min(1.0, score))

    # Helpers

    def _build_reasoning(
        self,
        metrics: dict[str, Any],
        profile: CompanyProfile | None,
        fundamentals: Fundamentals | None,
    ) -> list[str]:
        parts = [f"Computed {len(metrics)} fundamental metrics."]

        if profile:
            parts.append(f"{profile.name} ({profile.sector})")

        pe = metrics.get("trailing_pe")
        if pe:
            parts.append(f"Trailing P/E: {pe}")
        fpe = metrics.get("forward_pe")
        if fpe:
            parts.append(f"Forward P/E: {fpe}")

        health = metrics.get("financial_health_score")
        if health:
            parts.append(f"Financial health: {health}/100")
        quality = metrics.get("quality_score")
        if quality:
            parts.append(f"Quality: {quality}/100")
        growth = metrics.get("growth_score")
        if growth:
            parts.append(f"Growth: {growth}/100")

        roe = metrics.get("roe_pct")
        if roe:
            parts.append(f"ROE: {roe}%")

        return parts

    def _safe_fetch(self, fn, records: list, name: str, inp: dict) -> Any:
        """Call provider method, record success/failure."""
        try:
            result = fn()
            records.append(ToolCallRecord(
                tool_name=name, input=inp,
                output_summary=str(result)[:200] if result else "empty",
                success=True,
            ))
            return result
        except Exception as e:
            records.append(ToolCallRecord(
                tool_name=name, input=inp,
                output_summary="", success=False, error=str(e),
            ))
            return None

    @staticmethod
    def _format_metrics(metrics: dict[str, Any]) -> str:
        lines = [f"Metrics: {len(metrics)}"]
        for k, v in metrics.items():
            if v is not None and not k.endswith("_factors"):
                lines.append(f"  {k}: {v}")
        # Print factors separately
        for k, v in metrics.items():
            if k.endswith("_factors") and v:
                lines.append(f"  {k}: {', '.join(v)}")
        return "\n".join(lines)
