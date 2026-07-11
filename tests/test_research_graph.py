"""
Research graph tests.

Tests for graph-level orchestration: node functions, routing logic,
error paths, and end-to-end invocation via ``run_research()``.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage

from agents import (
    BaseAgent,
    DebateModerator,
    MarketDataAnalyst,
)
from graphs.research_graph import (
    _error_analysis,
    _error_debate_round,
    _fallback_recommendation,
    _run_agent_node,
    _should_continue_debate,
    build_research_graph,
    run_research,
)
from models import (
    AgentAnalysis,
    DebateContribution,
    DebateRound,
    Recommendation,
    ResearchState,
)
from services import MarketDataProvider
from tests.conftest import MockFundamentals, MockMarketData


# Mock LLM


class MockLLM(BaseLanguageModel):
    """Minimal mock LLM for graph-level tests."""

    def __init__(self, response: str = "Analysis complete."):
        super().__init__()
        object.__setattr__(self, "_response", response)

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


# _error_analysis


class DummyAgent(BaseAgent):
    """Minimal agent for testing error handling."""

    def __init__(self):
        super().__init__(name="test_agent", description="Test agent", llm=MockLLM())

    def system_prompt(self) -> str:
        return "Test system prompt."

    def build_messages(self, state: ResearchState) -> list[dict[str, Any]]:
        return []

    def analyze(self, state: ResearchState) -> AgentAnalysis:
        return AgentAnalysis(
            agent_name=self.name,
            symbol=state.symbol,
            agent_confidence=0.5,
            summary="Test",
            signals=[],
        )


class TestErrorAnalysis:
    def test_creates_valid_analysis_on_exception(self):
        agent = DummyAgent()
        state = ResearchState(user_query="test", symbol="AAPL")
        error = ValueError("API failure")

        analysis = _error_analysis(agent, state, error)

        assert analysis.agent_name == "test_agent"
        assert analysis.symbol == "AAPL"
        assert analysis.agent_confidence == 0.0
        assert "API failure" in analysis.summary
        assert "ValueError" in analysis.metadata.get("error", "")

    def test_error_analysis_includes_phase(self):
        agent = DummyAgent()
        state = ResearchState(
            user_query="test", symbol="AAPL", phase="collecting_analyses"
        )
        error = RuntimeError("timeout")

        analysis = _error_analysis(agent, state, error)

        assert analysis.metadata.get("phase") == "collecting_analyses"


# _run_agent_node


class MockAgent(BaseAgent):
    """Agent that returns a canned analysis."""

    def __init__(self, *, name: str = "mock_agent", fail: bool = False):
        super().__init__(name=name, description="Mock", llm=MockLLM())
        object.__setattr__(self, "_fail", fail)

    def system_prompt(self) -> str:
        return "Test system prompt."

    def build_messages(self, state: ResearchState) -> list[dict[str, Any]]:
        return []

    def parse_analysis(self, state: ResearchState, llm_response: str) -> AgentAnalysis:
        return AgentAnalysis(
            agent_name=self.name,
            symbol=state.symbol,
            agent_confidence=0.8,
            summary="Mock analysis",
            signals=[],
        )

    def analyze(self, state: ResearchState) -> AgentAnalysis:
        if self._fail:
            raise ValueError("Intentional failure")
        return AgentAnalysis(
            agent_name=self.name,
            symbol=state.symbol,
            agent_confidence=0.8,
            summary="Mock analysis",
            signals=[],
        )


class TestRunAgentNode:
    def test_success_path(self):
        agent = MockAgent(name="a1")
        state = ResearchState(user_query="test", symbol="AAPL")
        result = _run_agent_node(agent, state)

        assert "a1" in result["analyses"]
        assert result["analyses"]["a1"].agent_confidence == 0.8
        assert result["analyses"]["a1"].summary == "Mock analysis"
        assert result["agent_execution_order"] == ["a1"]
        assert result["errors"] == []

    def test_error_path(self):
        agent = MockAgent(name="failing", fail=True)
        state = ResearchState(user_query="test", symbol="AAPL")
        result = _run_agent_node(agent, state)

        assert "failing" in result["analyses"]
        assert result["analyses"]["failing"].agent_confidence == 0.0
        assert "Intentional failure" in result["analyses"]["failing"].summary
        assert result["agent_execution_order"] == ["failing"]
        assert len(result["errors"]) == 1


# _error_debate_round


class TestErrorDebateRound:
    def test_creates_valid_debate_round(self):
        state = ResearchState(user_query="test", symbol="AAPL", current_round=2)
        error = RuntimeError("Debate crashed")

        round_ = _error_debate_round(state, error)

        assert round_.round_number == 3  # current_round + 1
        assert len(round_.contributions) == 1
        assert round_.contributions[0].agent_name == "debate_moderator"
        assert "Debate crashed" in round_.contributions[0].message

    def test_handles_zero_round(self):
        state = ResearchState(user_query="test", symbol="AAPL", current_round=0)
        error = Exception("Generic error")

        round_ = _error_debate_round(state, error)

        assert round_.round_number == 1


# _fallback_recommendation


class TestFallbackRecommendation:
    def test_creates_conservative_recommendation(self):
        state = ResearchState(user_query="test", symbol="AAPL")
        error = ValueError("Synthesis error")

        rec = _fallback_recommendation(state, error)

        assert rec.symbol == "AAPL"
        assert rec.sentiment == "neutral"
        assert rec.confidence == 0.0
        assert rec.risk_level == "high"
        assert "Synthesis error" in rec.rationale
        assert "Synthesis error" in rec.conflicting_signals[0]

    def test_preserves_timeframe(self):
        state = ResearchState(user_query="test", symbol="AAPL", timeframe="short_term")
        error = Exception("fail")

        rec = _fallback_recommendation(state, error)

        assert rec.estimated_timeframe == "short_term"


# _should_continue_debate  (routing logic)


class TestShouldContinueDebate:
    def test_max_rounds_reached(self):
        """When current_round >= max_debate_rounds → synthesize."""
        state = ResearchState(
            user_query="test",
            symbol="AAPL",
            current_round=3,
            max_debate_rounds=3,
            debate_rounds=[
                DebateRound(
                    round_number=3,
                    contributions=[
                        DebateContribution(
                            agent_name="a1",
                            message="Still challenging",
                            challenge_to="a2",
                        )
                    ],
                )
            ],
        )
        assert _should_continue_debate(state) == "synthesize"

    def test_no_debate_rounds_yet(self):
        """When there are no debate rounds → synthesize."""
        state = ResearchState(
            user_query="test",
            symbol="AAPL",
            current_round=0,
            debate_rounds=[],
        )
        assert _should_continue_debate(state) == "synthesize"

    def test_no_substantive_challenges(self):
        """When the latest round has no challenges → synthesize."""
        state = ResearchState(
            user_query="test",
            symbol="AAPL",
            current_round=1,
            max_debate_rounds=3,
            debate_rounds=[
                DebateRound(
                    round_number=1,
                    contributions=[
                        DebateContribution(
                            agent_name="a1",
                            message="I agree with everything",
                            challenge_to=None,
                        ),
                    ],
                ),
            ],
        )
        assert _should_continue_debate(state) == "synthesize"

    def test_has_substantive_challenge(self):
        """When the latest round has a challenge → continue debate."""
        state = ResearchState(
            user_query="test",
            symbol="AAPL",
            current_round=1,
            max_debate_rounds=3,
            debate_rounds=[
                DebateRound(
                    round_number=1,
                    contributions=[
                        DebateContribution(
                            agent_name="a1",
                            message="I disagree with a2's analysis",
                            challenge_to="a2",
                        ),
                    ],
                ),
            ],
        )
        assert _should_continue_debate(state) == "debate_round"


# Graph node tests (via compiled graph invocation)


def _run_graph(
    graph,
    *,
    user_query: str = "Test AAPL",
    symbol: str = "AAPL",
) -> ResearchState:
    """Invoke the compiled graph and convert the result dict back to a
    ResearchState.  LangGraph can return a plain dict when the state is
    a Pydantic model, so we normalise here."""
    raw = graph.invoke(ResearchState(user_query=user_query, symbol=symbol))
    if isinstance(raw, ResearchState):
        return raw
    return ResearchState.model_validate(raw)


class TestGraphNodes:
    """Test individual graph nodes by invoking the compiled graph with
    partial / prepared states."""

    @pytest.fixture
    def graph(self):
        from services.base import NewsProvider, NewsArticle

        class MockNews(NewsProvider):
            def get_news(self, symbol, **kwargs):
                return [
                    NewsArticle(
                        title="Test news",
                        source="Test",
                        date="2026-07-12",
                        url="https://example.com",
                        snippet="Test snippet",
                        full_text="Test full text.",
                    )
                ]

        return build_research_graph(
            llm=MockLLM(),
            market_data_provider=MockMarketData(),
            fundamentals_provider=MockFundamentals(),
            news_provider=MockNews(),
        )

    def test_prepare_node_sets_phase(self, graph):
        """The 'prepare' node sets phase and max_debate_rounds."""
        result = _run_graph(graph)

        assert result.phase in (
            "collecting_analyses",
            "debating",
            "synthesizing",
            "complete",
        )

    def test_agents_execute_in_parallel(self, graph):
        """All four agents should produce analyses."""
        result = _run_graph(graph)

        assert len(result.analyses) == 4
        agent_names = {a.agent_name for a in result.analyses.values()}
        expected = {
            "market_data_analyst",
            "fundamentals_analyst",
            "technical_analyst",
            "news_intelligence_agent",
        }
        assert agent_names == expected

    def test_debate_happens(self, graph):
        """At least one debate round should be produced."""
        result = _run_graph(graph)

        assert len(result.debate_rounds) >= 1
        assert result.current_round >= 1

    def test_final_recommendation_produced(self, graph):
        """The graph should end with a valid recommendation."""
        result = _run_graph(graph)

        assert result.recommendation is not None
        assert result.recommendation.symbol == "AAPL"
        assert result.recommendation.sentiment in ("bullish", "bearish", "neutral")
        assert 0.0 <= result.recommendation.confidence <= 1.0

    def test_phase_ends_complete(self, graph):
        """Final phase should be 'complete'."""
        result = _run_graph(graph)

        assert result.phase == "complete"

    def test_graph_execution_order_tracked(self, graph):
        """Agent execution order should contain all four agents."""
        result = _run_graph(graph)

        assert len(result.agent_execution_order) == 4


# run_research() entry point


class TestRunResearch:
    @pytest.fixture
    def mock_news(self):
        from services.base import NewsProvider, NewsArticle

        class MockNews(NewsProvider):
            def get_news(self, symbol, **kwargs):
                return [
                    NewsArticle(
                        title="Test news",
                        source="Test",
                        date="2026-07-12",
                        url="https://example.com",
                        snippet="Test snippet",
                        full_text="Test full text.",
                    )
                ]

        return MockNews()

    def test_basic_invocation(self, mock_news):
        """run_research() returns a complete ResearchState."""
        state = run_research(
            user_query="Analyze AAPL",
            symbol="AAPL",
            llm=MockLLM(),
            market_data_provider=MockMarketData(),
            fundamentals_provider=MockFundamentals(),
            news_provider=mock_news,
        )

        assert isinstance(state, ResearchState)
        assert state.phase == "complete"
        assert state.recommendation is not None
        assert state.recommendation.symbol == "AAPL"

    def test_symbol_is_uppercased(self, mock_news):
        """The symbol should be uppercased per ResearchState validation."""
        state = run_research(
            user_query="Analyze aapl",
            symbol="aapl",
            llm=MockLLM(),
            market_data_provider=MockMarketData(),
            fundamentals_provider=MockFundamentals(),
            news_provider=mock_news,
        )

        assert state.symbol == "AAPL"
        assert state.recommendation.symbol == "AAPL"
