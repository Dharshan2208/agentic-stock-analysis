"""Tests for agent base class and registry."""

import pytest
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage

from agents import BaseAgent, registry, AgentRegistry
from models import ResearchState, AgentAnalysis, Signal


# ── Mock LLM for testing ──


class MockLLM(BaseLanguageModel):
    """A minimal mock LLM that returns canned responses."""

    def __init__(self, response: str = "Analysis complete."):
        super().__init__()
        self._response = response

    def invoke(self, messages, **kwargs):
        return AIMessage(content=self._response)

    async def ainvoke(self, messages, **kwargs):
        return AIMessage(content=self._response)

    # Required by BaseLanguageModel ABC but not used in tests
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


# ── Concrete test agent ──


class TestAgent(BaseAgent):
    """Minimal agent for testing purposes."""

    def system_prompt(self) -> str:
        return "You are a test analyst."

    def analyze(self, state: ResearchState) -> AgentAnalysis:
        return AgentAnalysis(
            agent_name=self.name,
            symbol=state.symbol,
            agent_confidence=0.85,
            summary="Test analysis result.",
            signals=[Signal(name="test", value=True, direction="bullish")],
            reasoning="Test reasoning.",
        )


class ErrorAgent(BaseAgent):
    """Agent that simulates a failure."""

    def system_prompt(self) -> str:
        return "You will fail."

    def analyze(self, state: ResearchState) -> AgentAnalysis:
        raise RuntimeError("Simulated failure")


# ── Tests ──


class TestBaseAgent:
    def test_requires_non_empty_name(self):
        with pytest.raises(ValueError, match="non-empty"):
            TestAgent(name="", description="test", llm=MockLLM())

    def test_requires_non_empty_description(self):
        with pytest.raises(ValueError, match="non-empty"):
            TestAgent(name="test", description="", llm=MockLLM())

    def test_run_writes_to_state(self):
        agent = TestAgent(name="test_agent", description="Test", llm=MockLLM())
        state = ResearchState(user_query="test", symbol="AAPL")
        result = agent.run(state)
        assert "test_agent" in result.analyses
        assert result.analyses["test_agent"].agent_confidence == 0.85
        assert result.analyses["test_agent"].summary == "Test analysis result."

    def test_run_records_execution_order(self):
        agent = TestAgent(name="order_test", description="Test", llm=MockLLM())
        state = ResearchState(user_query="test", symbol="AAPL")
        result = agent.run(state)
        assert "order_test" in result.agent_execution_order

    def test_error_agent_graceful_degradation(self):
        agent = ErrorAgent(name="failing_agent", description="Fails", llm=MockLLM())
        state = ResearchState(user_query="test", symbol="AAPL")
        result = agent.run(state)
        # Should have an analysis (not crash)
        assert "failing_agent" in result.analyses
        assert result.analyses["failing_agent"].agent_confidence == 0.0
        assert "error" in result.analyses["failing_agent"].summary.lower()
        assert len(result.errors) == 1
        assert "Simulated failure" in result.errors[0]


class TestAgentRegistry:
    def setup_method(self):
        self.registry = AgentRegistry()

    def test_register_and_get(self):
        agent = TestAgent(name="test", description="Test", llm=MockLLM())
        self.registry.register(agent)
        assert self.registry.get("test") is agent

    def test_register_duplicate_raises(self):
        agent = TestAgent(name="dup", description="Test", llm=MockLLM())
        self.registry.register(agent)
        with pytest.raises(ValueError, match="already registered"):
            self.registry.register(agent)

    def test_list_all_returns_in_order(self):
        a1 = TestAgent(name="first", description="1", llm=MockLLM())
        a2 = TestAgent(name="second", description="2", llm=MockLLM())
        self.registry.register(a1)
        self.registry.register(a2)
        names = [a.name for a in self.registry.list_all()]
        assert names == ["first", "second"]

    def test_run_phase_runs_all_agents(self):
        a1 = TestAgent(name="agent_1", description="A", llm=MockLLM())
        a2 = TestAgent(name="agent_2", description="B", llm=MockLLM())
        self.registry.register(a1)
        self.registry.register(a2)

        state = ResearchState(user_query="test", symbol="AAPL")
        result = self.registry.run_phase(state)

        assert "agent_1" in result.analyses
        assert "agent_2" in result.analyses
        assert len(result.analyses) == 2

    def test_run_phase_continues_on_failure(self):
        good = TestAgent(name="good", description="Good", llm=MockLLM())
        bad = ErrorAgent(name="bad", description="Bad", llm=MockLLM())
        self.registry.register(good)
        self.registry.register(bad)

        state = ResearchState(user_query="test", symbol="AAPL")
        result = self.registry.run_phase(state)

        assert "good" in result.analyses
        assert "bad" in result.analyses
        # Good agent has normal confidence
        assert result.analyses["good"].agent_confidence == 0.85
        # Bad agent has zero confidence (failed)
        assert result.analyses["bad"].agent_confidence == 0.0
        assert len(result.errors) == 1

    def test_clear(self):
        agent = TestAgent(name="clear_test", description="T", llm=MockLLM())
        self.registry.register(agent)
        assert self.registry.count() == 1
        self.registry.clear()
        assert self.registry.count() == 0

    def test_global_singleton(self):
        from agents import registry as global_registry

        assert isinstance(global_registry, AgentRegistry)
