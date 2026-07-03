"""Tests for state models."""

import pytest
from models import ResearchState, AgentAnalysis, Signal, Recommendation


class TestResearchState:
    def test_symbol_is_uppercased(self):
        state = ResearchState(user_query="analyze aapl", symbol="aapl")
        assert state.symbol == "AAPL"

    def test_default_phase_is_init(self):
        state = ResearchState(user_query="test", symbol="TEST")
        assert state.phase == "init"

    def test_can_add_analysis(self):
        state = ResearchState(user_query="test", symbol="TEST")
        state.analyses["test_agent"] = AgentAnalysis(
            agent_name="test_agent",
            symbol="TEST",
            agent_confidence=0.75,
            summary="Test analysis.",
            signals=[Signal(name="test_signal", value=42, direction="bullish")],
        )
        assert len(state.analyses) == 1
        assert state.analyses["test_agent"].agent_confidence == 0.75

    def test_confidence_clamped(self):
        """Agent confidence must be 0.0–1.0."""
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            AgentAnalysis(
                agent_name="bad",
                symbol="X",
                agent_confidence=1.5,  # Invalid
                summary="bad",
            )

    def test_round_trip_serialization(self, sample_research_state):
        json_str = sample_research_state.model_dump_json()
        restored = ResearchState.model_validate_json(json_str)
        assert restored.symbol == sample_research_state.symbol
        assert len(restored.analyses) == len(sample_research_state.analyses)
        assert restored.analyses["market_analyst"].signals[0].direction == "bullish"

    def test_recommendation_defaults(self):
        rec = Recommendation(
            symbol="AAPL",
            sentiment="bullish",
            confidence=0.85,
            rationale="Strong fundamentals.",
        )
        assert rec.risk_level == "medium"
        assert rec.estimated_timeframe == "medium_term"
        assert rec.generated_at is not None
