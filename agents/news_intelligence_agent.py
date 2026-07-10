"""
News Intelligence Agent.

Responsible for:
  - Fetching recent news articles for a symbol via the service layer
  - Deterministic extraction of event-type signals from headlines (earnings,
    analyst actions, M&A, regulatory, product news)
  - Using an LLM to summarise news sentiment and key narratives
  - Reporting confidence based on article count and signal clarity

Does NOT do:
  - Price/volume analysis (handled by Market Data Analyst)
  - Fundamental ratio computation (handled by Fundamentals Analyst)
  - Long-term technical indicators (handled by Technical Analyst)
"""

from __future__ import annotations

import re
from typing import Any

from agents.base_agent import BaseAgent
from models import ResearchState, AgentAnalysis, Signal, ToolCallRecord
from services import get_news_provider, NewsProvider, NewsArticle


#  Keyword-based event detection patterns

_EARNINGS_PATTERNS = re.compile(
    r"\b(earnings?|EPS|quarterly|Q[1-4]|fiscal|revenue|profit|loss|income)\b",
    re.IGNORECASE,
)
_ANALYST_UPGRADE_PATTERNS = re.compile(
    r"\b(upgrade|buy|outperform|overweight|raised?\s.*target|price\s*target\s*↑)\b",
    re.IGNORECASE,
)
_ANALYST_DOWNGRADE_PATTERNS = re.compile(
    r"\b(downgrade(s|d)?|sell|underperform|underweight|lowered?\s.*target|price\s*target\s*↓)\b",
    re.IGNORECASE,
)
_MA_PATTERNS = re.compile(
    r"\b(merger|acquisition|merge|acquire|takeover|buyout)\b",
    re.IGNORECASE,
)
_REGULATORY_PATTERNS = re.compile(
    r"\b(lawsuit|regulatory|probe|investigation|fine|penalty|compliance|doj|sec|ftc)\b",
    re.IGNORECASE,
)
_PRODUCT_PATTERNS = re.compile(
    r"\b(launch|announces|unveils|new\s*product|partnership|collaborat)",
    re.IGNORECASE,
)
_POSITIVE_SENTIMENT = re.compile(
    r"\b(beat?|surge|record|growth|positive|strong|profit|rally|gain|up|bullish)\b",
    re.IGNORECASE,
)
_NEGATIVE_SENTIMENT = re.compile(
    r"\b(drop(s|ped|ping)?|decline(s|d)?|fall(s|en|ing)?|loss(es)?|negative|weak(er|est)?|downturn(s)?|sell.?off(s)?|down|bearish|risk(y|s)?|downgrade(s|d)?)\b",
    re.IGNORECASE,
)


class NewsIntelligenceAgent(BaseAgent):
    """Analyses recent news to extract sentiment, event signals, and key narratives."""

    def __init__(
        self,
        llm: Any,
        news_provider: NewsProvider | None = None,
        name: str = "news_intelligence_agent",
    ) -> None:
        super().__init__(
            name=name,
            description="News sentiment, event detection, and narrative analysis",
            llm=llm,
        )
        self._news_provider = news_provider or get_news_provider()

    # Prompt

    def system_prompt(self) -> str:
        return (
            "You are a News Intelligence Analyst specialising in financial news."
            " Your job is to interpret recent news articles and produce a concise"
            " narrative about market-moving sentiment and events.\n\n"
            "Focus on:\n"
            "1. Overall sentiment — is the news broadly positive, negative, or mixed?\n"
            "2. Key events — earnings results, analyst actions, M&A, regulatory, product news\n"
            "3. Risks and catalysts — what could move the stock in either direction\n"
            "4. Numerical highlights — any EPS, revenue, price target figures mentioned\n\n"
            "Rules:\n"
            "- Reference specific article titles and numbers.\n"
            "- Be concise (3-5 sentences max).\n"
            "- Do NOT discuss price charts, technical indicators, or valuation multiples.\n"
            "- Stick to what the news objectively reports."
        )

    # Core analysis

    def analyze(self, state: ResearchState) -> AgentAnalysis:
        tool_records: list[ToolCallRecord] = []
        symbol = state.symbol

        #  1. Fetch articles
        articles = self._safe_fetch(
            lambda: self._news_provider.get_news(symbol, max_articles=7),
            tool_records,
            "get_news",
            {"symbol": symbol, "max_articles": 7},
        )
        articles = articles or []

        if not articles:
            return AgentAnalysis(
                agent_name=self.name,
                symbol=symbol,
                agent_confidence=0.0,
                summary="No recent news articles found for analysis.",
                signals=[],
                reasoning="News provider returned no articles.",
                tool_calls=tool_records,
                metadata={"article_count": 0},
            )

        #  2. Deterministic signal extraction
        signals = self._extract_news_signals(articles)
        article_summary = self._summarise_articles(articles)

        #  3. LLM narrative
        messages = self.build_messages(state) + [
            {
                "role": "user",
                "content": (
                    f"Recent news articles for {symbol}:\n\n"
                    f"{article_summary}\n\n"
                    f"Total articles: {len(articles)}\n"
                    f"Articles with full text: {sum(1 for a in articles if a.full_text)}\n\n"
                    "Provide a news intelligence summary (3-5 sentences). "
                    "Then list 2-4 specific signals with direction "
                    "(bullish/bearish/neutral)."
                ),
            },
        ]
        llm_response = self.invoke_llm(messages)

        #  4. Confidence
        confidence = self._compute_confidence(articles, signals)

        #  5. Build output
        signal_names = [s.name for s in signals]
        sentiment = self._classify_sentiment(signal_names)

        return AgentAnalysis(
            agent_name=self.name,
            symbol=symbol,
            agent_confidence=confidence,
            summary=llm_response,
            signals=signals,
            reasoning=(
                f"Fetched {len(articles)} articles | "
                f"Detected {len(signals)} signals: {', '.join(signal_names)} | "
                f"Estimated sentiment: {sentiment}"
            ),
            tool_calls=tool_records,
            metadata={
                "article_count": len(articles),
                "articles_with_text": sum(1 for a in articles if a.full_text),
                "signal_names": signal_names,
                "estimated_sentiment": sentiment,
            },
        )

    # Signal extraction (deterministic)

    def _extract_news_signals(self, articles: list[NewsArticle]) -> list[Signal]:
        """Extract event-type and sentiment signals from article titles/snippets."""
        signals: list[Signal] = []

        # Collect all text for pattern matching
        all_text = " ".join(f"{a.title} {a.snippet}" for a in articles if a.title)

        # -- Event type signals --
        if _EARNINGS_PATTERNS.search(all_text):
            signals.append(
                Signal(
                    name="earnings_news",
                    value=f"in {sum(1 for a in articles if _EARNINGS_PATTERNS.search(a.title))} articles",
                    direction="neutral",
                )
            )

        if _ANALYST_UPGRADE_PATTERNS.search(all_text):
            signals.append(
                Signal(
                    name="analyst_upgrade",
                    value="positive analyst action detected",
                    direction="bullish",
                )
            )

        if _ANALYST_DOWNGRADE_PATTERNS.search(all_text):
            signals.append(
                Signal(
                    name="analyst_downgrade",
                    value="negative analyst action detected",
                    direction="bearish",
                )
            )

        if _MA_PATTERNS.search(all_text):
            signals.append(
                Signal(
                    name="ma_activity",
                    value="merger or acquisition mentions",
                    direction="neutral",
                )
            )

        if _REGULATORY_PATTERNS.search(all_text):
            signals.append(
                Signal(
                    name="regulatory_risk",
                    value="legal or regulatory mentions",
                    direction="bearish",
                )
            )

        if _PRODUCT_PATTERNS.search(all_text):
            signals.append(
                Signal(
                    name="product_news",
                    value="product launch or announcement detected",
                    direction="bullish",
                )
            )

        # -- Sentiment counts --
        pos_count = sum(1 for a in articles if _POSITIVE_SENTIMENT.search(a.title))
        neg_count = sum(1 for a in articles if _NEGATIVE_SENTIMENT.search(a.title))
        total = len(articles)

        if total > 0:
            pos_ratio = pos_count / total
            neg_ratio = neg_count / total

            if pos_ratio >= 0.5 and pos_ratio > neg_ratio * 1.5:
                signals.append(
                    Signal(
                        name="positive_sentiment",
                        value=f"{pos_count}/{total} articles positive",
                        direction="bullish",
                    )
                )
            elif neg_ratio >= 0.5 and neg_ratio > pos_ratio * 1.5:
                signals.append(
                    Signal(
                        name="negative_sentiment",
                        value=f"{neg_count}/{total} articles negative",
                        direction="bearish",
                    )
                )
            else:
                signals.append(
                    Signal(
                        name="mixed_sentiment",
                        value=f"{pos_count} positive / {neg_count} negative / {total} total",
                        direction="neutral",
                    )
                )

        # -- High news volume --
        if total >= 4:
            signals.append(
                Signal(
                    name="high_news_volume",
                    value=f"{total} articles found",
                    direction="neutral",
                )
            )

        return signals

    def _classify_sentiment(self, signal_names: list[str]) -> str:
        """Classify overall sentiment from signal names."""
        bullish = any(
            "upgrade" in s or "positive" in s or "product" in s for s in signal_names
        )
        bearish = any(
            "downgrade" in s or "negative" in s or "regulatory" in s
            for s in signal_names
        )

        if bullish and not bearish:
            return "positive"
        if bearish and not bullish:
            return "negative"
        return "mixed"

    # Confidence

    def _compute_confidence(
        self,
        articles: list[NewsArticle],
        signals: list[Signal],
    ) -> float:
        """Confidence (0.0–1.0) based on article count, text availability, signal strength."""
        if not articles:
            return 0.0

        score = 0.30

        # Article quantity
        if len(articles) >= 5:
            score += 0.20
        elif len(articles) >= 3:
            score += 0.10

        # Articles with full text
        with_text = sum(1 for a in articles if a.full_text)
        score += min(with_text * 0.05, 0.15)

        # Signal diversity
        unique_directions = set(s.direction for s in signals)
        if len(unique_directions) >= 2:
            score += 0.10
        if len(signals) >= 3:
            score += 0.10

        # Clear sentiment
        names = [s.name for s in signals]
        has_pos = any("positive" in n for n in names)
        has_neg = any("negative" in n for n in names)
        if has_pos != has_neg:  # clear direction
            score += 0.10

        return max(0.0, min(1.0, score))

    # Helpers

    @staticmethod
    def _summarise_articles(articles: list[NewsArticle]) -> str:
        """Build a readable summary of all articles for the LLM."""
        lines = []
        for i, a in enumerate(articles, 1):
            snippet = (a.snippet or "")[:200]
            has_text = " (full text available)" if a.full_text else ""
            lines.append(
                f"Article {i}: {a.title}\n"
                f"  Source: {a.source} | {a.date}\n"
                f"  Snippet: {snippet}{has_text}\n"
            )
        return "\n".join(lines)

    def _safe_fetch(self, fn, records: list, name: str, inp: dict) -> Any:
        """Call provider method, record success/failure."""
        try:
            result = fn()
            records.append(
                ToolCallRecord(
                    tool_name=name,
                    input=inp,
                    output_summary=f"{len(result)} articles" if result else "empty",
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
