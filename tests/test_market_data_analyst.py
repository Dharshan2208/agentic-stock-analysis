"""Tests for the Market Data Analyst agent."""

import pytest
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage

from agents import BaseAgent
from agents.market_data_analyst import MarketDataAnalyst
from models import ResearchState
from services import set_market_data, set_fundamentals
from tests.conftest import MockMarketData, MockFundamentals


class MockLLM(BaseLanguageModel):
    """Minimal mock LLM returning a canned response."""

    last_messages: list[dict[str, object]] | None = None

    def __init__(
        self,
        response: str = "Price shows steady upward momentum with above-average volume. Signals are broadly bullish.",
    ):
        super().__init__()
        self._response = response
        self.last_messages = None

    def invoke(self, messages, **kwargs):
        self.last_messages = messages
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


# ── Fixtures ──


@pytest.fixture(autouse=True)
def inject_mocks():
    """Use mock providers for every test in this module."""
    set_market_data(MockMarketData())
    set_fundamentals(MockFundamentals())
    yield
    # No cleanup needed — test isolation via fresh mocks each function


@pytest.fixture
def mock_llm():
    return MockLLM()


@pytest.fixture
def agent(mock_llm):
    return MarketDataAnalyst(llm=mock_llm)


@pytest.fixture
def state():
    return ResearchState(user_query="Analyze AAPL", symbol="AAPL", sector="Technology")


# ── Core analysis tests ──


class TestMarketDataAnalyst:
    def test_agent_name(self, agent):
        assert agent.name == "market_data_analyst"
        assert "price action" in agent.description.lower()

    def test_analyze_returns_valid_analysis(self, agent, state):
        result = agent.run(state)
        assert "market_data_analyst" in result.analyses
        analysis = result.analyses["market_data_analyst"]
        assert analysis.symbol == "AAPL"
        assert 0.0 <= analysis.agent_confidence <= 1.0
        assert len(analysis.summary) > 0

    def test_signals_are_extracted(self, agent, state):
        result = agent.run(state)
        analysis = result.analyses["market_data_analyst"]
        # With mock data (upward trend) we expect at least one signal
        assert len(analysis.signals) >= 1

    def test_signals_have_correct_directions(self, agent, state):
        """Mock data is: 150 → 151.5 → 152.8 → 154.2 — upward trend."""
        result = agent.run(state)
        analysis = result.analyses["market_data_analyst"]
        for signal in analysis.signals:
            assert signal.direction in ("bullish", "bearish", "neutral")

    def test_tool_calls_are_recorded(self, agent, state):
        result = agent.run(state)
        analysis = result.analyses["market_data_analyst"]
        assert len(analysis.tool_calls) >= 1

    def test_reasoning_is_present(self, agent, state):
        result = agent.run(state)
        analysis = result.analyses["market_data_analyst"]
        assert len(analysis.reasoning) > 0
        assert "metrics" in analysis.reasoning.lower()

    def test_execution_order_recorded(self, agent, state):
        result = agent.run(state)
        assert "market_data_analyst" in result.agent_execution_order

    def test_confidence_reasonable_range(self, agent, state):
        """With 3 mock bars (+2.7% total), confidence should be moderate."""
        result = agent.run(state)
        conf = result.analyses["market_data_analyst"].agent_confidence
        assert 0.30 <= conf <= 1.0, f"Confidence {conf} out of expected range"

    def test_empty_data_graceful(self, mock_llm):
        """If provider returns no data, agent should degrade gracefully."""

        class EmptyProvider(MockMarketData):
            def get_ohlcv(self, *args, **kwargs):
                return []

            def get_current_price(self, *args, **kwargs):
                return None

        set_market_data(EmptyProvider())
        empty_agent = MarketDataAnalyst(llm=mock_llm)
        state = ResearchState(user_query="test", symbol="EMPTY")
        result = empty_agent.run(state)
        analysis = result.analyses["market_data_analyst"]
        assert analysis.agent_confidence == 0.0
        assert (
            "no data" in analysis.summary.lower()
            or "no data" in analysis.reasoning.lower()
        )

    def test_llm_receives_metrics_in_prompt(self, agent, state, mock_llm):
        """Verify the LLM is called with metric data in the user message."""
        agent.run(state)
        assert mock_llm.last_messages is not None
        # Last user message should contain market data context
        user_msg = mock_llm.last_messages[-1]["content"]
        assert "AAPL" in user_msg
        assert any(word in user_msg for word in ["Bars:", "change", "volume", "range"])

    def test_agent_sets_metadata(self, agent, state):
        result = agent.run(state)
        meta = result.analyses["market_data_analyst"].metadata
        assert meta.get("bar_count", 0) > 0
        assert "data_source" in meta
        assert "metrics_computed" in meta

    def test_agent_works_with_registry(self, agent, state):
        """Verify the agent can be registered and run via the registry."""
        from agents import registry

        registry.clear()
        registry.register(agent)
        result = registry.run_phase(state)
        assert "market_data_analyst" in result.analyses
        assert result.analyses["market_data_analyst"].agent_confidence > 0

    def test_custom_name(self, mock_llm):
        """Allow overriding the agent name."""
        custom = MarketDataAnalyst(llm=mock_llm, name="my_price_analyst")
        assert custom.name == "my_price_analyst"
        state = ResearchState(user_query="test", symbol="TEST")
        result = custom.run(state)
        assert "my_price_analyst" in result.analyses


class TestMarketDataMetrics:
    """Unit tests for the metric computation logic."""

    def test_price_changes(self, agent):
        from services.base import OHLCVBar

        bars = [
            OHLCVBar(time="d1", open=100, high=102, low=99, close=101, volume=1000),
            OHLCVBar(time="d2", open=101, high=104, low=100, close=103, volume=1200),
            OHLCVBar(time="d3", open=103, high=105, low=102, close=104, volume=1100),
        ]
        metrics = agent._compute_price_metrics(bars)
        assert metrics["change_1d_pct"] == pytest.approx(
            0.97, rel=0.1
        )  # (104-103)/103 * 100
        assert metrics["latest_volume"] == 1100

    def test_gap_detection(self, agent):
        from services.base import OHLCVBar

        # Bar opens significantly higher than previous close
        bars = [
            OHLCVBar(time="d1", open=100, high=102, low=99, close=100, volume=1000),
            OHLCVBar(time="d2", open=105, high=107, low=104, close=106, volume=2000),
        ]
        metrics = agent._compute_price_metrics(bars)
        assert metrics["gap_direction"] == "up"
        assert metrics["gap_pct"] == pytest.approx(5.0, rel=0.1)

    def test_volume_spike(self, agent):
        from services.base import OHLCVBar

        bars = [
            OHLCVBar(time=f"d{i}", open=100, high=102, low=99, close=101, volume=1000)
            for i in range(5)
        ]
        # Last bar has 3x volume
        bars[-1] = OHLCVBar(
            time="d5", open=101, high=103, low=100, close=102, volume=3000
        )
        metrics = agent._compute_price_metrics(bars)
        assert metrics["volume_vs_avg_5d"] == pytest.approx(3.0, rel=0.1)
