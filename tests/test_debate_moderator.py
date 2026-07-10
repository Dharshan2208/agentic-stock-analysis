"""Tests for the inter-agent debate moderator and M2.2 graph flow."""

from __future__ import annotations

import json

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage

from agents.debate_moderator import DebateModerator
from graphs.research_graph import run_research
from models import AgentAnalysis, ResearchState, Signal
from services.base import NewsArticle, NewsProvider
from tests.conftest import MockFundamentals, MockMarketData


class MockLLM(BaseLanguageModel):
    """Minimal mock LLM returning a canned response."""

    def __init__(self, response: str = "Analysis complete."):
        super().__init__()
        self._response = response

    def invoke(self, messages, **kwargs):
        return AIMessage(content=self._response)

    async def ainvoke(self, messages, **kwargs):
        return AIMessage(content=self._response)

    def generate(self, messages, **kwargs):
        raise NotImplementedError

    async def agenerate(self, messages, **kwargs):
        raise NotImplementedError

    def generate_prompt(self, prompts, stop=None, callbacks=None, **kwargs):
        raise NotImplementedError

    async def agenerate_prompt(self, prompts, stop=None, callbacks=None, **kwargs):
        raise NotImplementedError

    def get_num_tokens(self, text: str) -> int:
        return len(text.split())

    @property
    def _llm_type(self) -> str:
        return "mock"


class MockNegativeNews(NewsProvider):
    """Returns bearish news so the graph has a clear cross-agent conflict."""

    def get_news(
        self,
        symbol: str,
        recency: str = "past 24 hours",
        max_articles: int = 7,
    ) -> list[NewsArticle]:
        return [
            NewsArticle(
                title=f"{symbol} Shares Drop After Analyst Downgrade",
                source="Example",
                date="2026-07-10",
                url="https://example.com/1",
                snippet="Analysts cut their rating on weak demand.",
                full_text="Analysts cut their rating on weak demand.",
            ),
            NewsArticle(
                title=f"DOJ Investigation Creates Regulatory Risk For {symbol}",
                source="Example",
                date="2026-07-10",
                url="https://example.com/2",
                snippet="A regulatory probe increased investor concern.",
                full_text="A regulatory probe increased investor concern.",
            ),
        ]


def _analysis(
    name: str,
    confidence: float,
    directions: list[str],
) -> AgentAnalysis:
    return AgentAnalysis(
        agent_name=name,
        symbol="AAPL",
        agent_confidence=confidence,
        summary=f"{name} summary",
        signals=[
            Signal(
                name=f"{direction}_signal_{index}",
                value=True,
                direction=direction,
            )
            for index, direction in enumerate(directions)
        ],
        reasoning=f"{name} reasoning",
    )


def test_debate_detects_bullish_bearish_conflict():
    state = ResearchState(
        user_query="Analyze AAPL",
        symbol="AAPL",
        analyses={
            "market_data_analyst": _analysis(
                "market_data_analyst",
                0.8,
                ["bullish", "bullish"],
            ),
            "news_intelligence_agent": _analysis(
                "news_intelligence_agent",
                0.7,
                ["bearish", "bearish"],
            ),
        },
    )

    round_ = DebateModerator(llm=MockLLM()).run_round(state)

    assert round_.round_number == 1
    assert any(
        contribution.challenge_to == "news_intelligence_agent"
        for contribution in round_.contributions
    )
    assert any(
        "conflict" in contribution.message.lower()
        for contribution in round_.contributions
    )


def test_debate_flags_low_confidence_and_missing_signals():
    state = ResearchState(
        user_query="Analyze AAPL",
        symbol="AAPL",
        analyses={
            "technical_analyst": _analysis("technical_analyst", 0.2, []),
        },
    )

    round_ = DebateModerator(llm=MockLLM()).run_round(state)
    messages = [contribution.message.lower() for contribution in round_.contributions]

    assert any("low confidence" in message for message in messages)
    assert any("no structured signals" in message for message in messages)


def test_debate_uses_llm_fallback_when_no_rule_based_contributions():
    llm_response = json.dumps(
        {
            "contributions": [
                {
                    "agent_name": "debate_moderator",
                    "challenge_to": "market_data_analyst",
                    "message": "Momentum evidence needs volume confirmation.",
                    "supporting_evidence": ["Volume signal is absent."],
                }
            ]
        }
    )
    state = ResearchState(
        user_query="Analyze AAPL",
        symbol="AAPL",
        analyses={
            "market_data_analyst": _analysis(
                "market_data_analyst",
                0.8,
                ["bullish"],
            ),
            "technical_analyst": _analysis(
                "technical_analyst",
                0.8,
                ["bullish"],
            ),
        },
    )

    round_ = DebateModerator(llm=MockLLM(response=llm_response)).run_round(state)

    assert round_.contributions[0].challenge_to == "market_data_analyst"
    assert "volume confirmation" in round_.contributions[0].message


def test_m2_2_graph_runs_debate_without_recommendation():
    state = run_research(
        user_query="Analyze AAPL",
        symbol="AAPL",
        llm=MockLLM(),
        market_data_provider=MockMarketData(),
        fundamentals_provider=MockFundamentals(),
        news_provider=MockNegativeNews(),
    )

    assert state.phase == "complete"
    assert len(state.analyses) == 4
    assert len(state.debate_rounds) >= 1
    assert state.current_round >= 1
    assert state.recommendation is None
