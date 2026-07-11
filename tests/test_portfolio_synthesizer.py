"""Tests for the Portfolio Synthesizer agent (M2.3)."""

from __future__ import annotations

import json

import pytest
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage

from agents.portfolio_synthesizer import PortfolioSynthesizer
from models import (
    AgentAnalysis,
    DebateContribution,
    DebateRound,
    Recommendation,
    ResearchState,
    Signal,
)


# Helpers


def _analysis(
    name: str,
    confidence: float,
    directions: list[str] | None = None,
    signal_values: list[str | float | bool] | None = None,
    summary: str = "Test analysis summary.",
    reasoning: str = "Test reasoning.",
) -> AgentAnalysis:
    """Build an AgentAnalysis quickly."""
    signals: list[Signal] = []
    if directions:
        for i, direction in enumerate(directions):
            value = (
                signal_values[i] if signal_values and i < len(signal_values) else True
            )
            signals.append(
                Signal(
                    name=f"{direction}_signal_{i}",
                    value=value,
                    direction=direction,
                )
            )

    return AgentAnalysis(
        agent_name=name,
        symbol="AAPL",
        agent_confidence=confidence,
        summary=summary,
        signals=signals,
        reasoning=reasoning,
    )


def _state(
    analyses: dict[str, AgentAnalysis] | None = None,
    debates: list[DebateRound] | None = None,
    errors: list[str] | None = None,
    symbol: str = "AAPL",
    timeframe: str = "medium_term",
) -> ResearchState:
    return ResearchState(
        user_query=f"Analyze {symbol}",
        symbol=symbol,
        timeframe=timeframe,
        analyses=analyses or {},
        debate_rounds=debates or [],
        errors=errors or [],
    )


# Mock LLM for testing rationale generation


class MockLLM(BaseLanguageModel):
    """Mock LLM that returns a canned JSON rationale.

    Uses object.__setattr__ to avoid Pydantic v1 field conflicts on
    Python 3.14 (Core Pydantic V1 is incompatible with Python 3.14).
    """

    def __init__(
        self,
        rationale: str = (
            "The combined evidence supports a bullish outlook with moderate conviction. "
            "Price momentum is positive, fundamentals are healthy, and technical trends "
            "confirm the upward bias. However, the news sentiment is mixed, and debate "
            "highlights some uncertainty around valuation. Overall, the risk is contained."
        ),
    ):
        super().__init__()
        object.__setattr__(self, "_response", rationale)
        object.__setattr__(self, "_captured", None)

    @property
    def last_messages(self) -> list | None:
        return object.__getattribute__(self, "_captured")

    def invoke(self, messages, **kwargs):
        object.__setattr__(self, "_captured", messages)
        return AIMessage(content=self._response)

    async def ainvoke(self, messages, **kwargs):
        return self.invoke(messages)

    def generate(self, messages, **kwargs):
        raise NotImplementedError

    async def agenerate(self, messages, **kwargs):
        raise NotImplementedError

    def generate_prompt(self, prompts, **kwargs):
        raise NotImplementedError

    async def agenerate_prompt(self, prompts, **kwargs):
        raise NotImplementedError

    def get_num_tokens(self, text: str) -> int:
        return len(text.split())

    @property
    def _llm_type(self) -> str:
        return "mock"


# Synthesis integration tests


class TestPortfolioSynthesizerIntegration:
    """Full pipeline tests — from analyses to Recommendation."""

    def test_bullish_consensus_produces_bullish_recommendation(self):
        """All agents bullish → bullish recommendation."""
        state = _state(
            analyses={
                "market_data_analyst": _analysis(
                    "market_data_analyst", 0.8, ["bullish", "bullish"]
                ),
                "fundamentals_analyst": _analysis(
                    "fundamentals_analyst", 0.75, ["bullish"]
                ),
                "technical_analyst": _analysis("technical_analyst", 0.7, ["bullish"]),
            }
        )
        synth = PortfolioSynthesizer()
        rec = synth.synthesize(state)

        assert rec.symbol == "AAPL"
        assert rec.sentiment == "bullish"
        assert rec.confidence > 0.5
        assert len(rec.key_signals) >= 1
        assert len(rec.rationale) > 20

    def test_bearish_consensus_produces_bearish_recommendation(self):
        """All agents bearish → bearish recommendation."""
        state = _state(
            analyses={
                "market_data_analyst": _analysis(
                    "market_data_analyst", 0.8, ["bearish", "bearish"]
                ),
                "technical_analyst": _analysis("technical_analyst", 0.7, ["bearish"]),
            }
        )
        synth = PortfolioSynthesizer()
        rec = synth.synthesize(state)

        assert rec.sentiment == "bearish"
        assert rec.confidence > 0.5

    def test_mixed_signals_produces_neutral_sentiment(self):
        """Balanced bullish/bearish → neutral recommendation."""
        state = _state(
            analyses={
                "market_data_analyst": _analysis(
                    "market_data_analyst", 0.8, ["bullish"]
                ),
                "news_intelligence_agent": _analysis(
                    "news_intelligence_agent", 0.7, ["bearish"]
                ),
            }
        )
        synth = PortfolioSynthesizer()
        rec = synth.synthesize(state)

        assert rec.sentiment == "neutral"

    def test_empty_analyses_returns_conservative_recommendation(self):
        """No analyses → neutral, zero confidence, high risk."""
        state = _state(analyses={})
        synth = PortfolioSynthesizer()
        rec = synth.synthesize(state)

        assert rec.sentiment == "neutral"
        assert rec.confidence == 0.0
        assert rec.risk_level == "high"

    def test_all_low_confidence_agents_produces_low_confidence(self):
        """Multiple agents with moderately low confidence → low overall confidence.

        Each bullish agent gets score = direction_score * confidence, so:
          a1: 1.0 * 0.30 = 0.30
          a2: 1.0 * 0.40 = 0.40
          weighted = (0.30 + 0.40) / 2 = 0.35 ≥ +0.15 → bullish
        Confidence formula still produces < 0.4 for these low values.
        """
        state = _state(
            analyses={
                "market_data_analyst": _analysis(
                    "market_data_analyst", 0.30, ["bullish"]
                ),
                "technical_analyst": _analysis("technical_analyst", 0.40, ["bullish"]),
            }
        )
        synth = PortfolioSynthesizer()
        rec = synth.synthesize(state)

        # Sentiment still bullish (all agree), but confidence should be low
        assert rec.sentiment == "bullish"
        assert rec.confidence < 0.4

    def test_debate_conflicts_appear_in_conflicting_signals(self):
        """Debate challenges should be reflected in the recommendation."""
        state = _state(
            analyses={
                "market_data_analyst": _analysis(
                    "market_data_analyst", 0.8, ["bullish", "bullish"]
                ),
                "news_intelligence_agent": _analysis(
                    "news_intelligence_agent", 0.7, ["bearish"]
                ),
            },
            debates=[
                DebateRound(
                    round_number=1,
                    contributions=[
                        DebateContribution(
                            agent_name="market_data_analyst",
                            challenge_to="news_intelligence_agent",
                            message="Price momentum conflicts with bearish news sentiment.",
                            supporting_evidence=["Price up 5%"],
                        ),
                    ],
                ),
            ],
        )
        synth = PortfolioSynthesizer()
        rec = synth.synthesize(state)

        assert len(rec.conflicting_signals) >= 1
        assert any("challenged" in c for c in rec.conflicting_signals)

    def test_failed_agent_does_not_crash_synthesis(self):
        """An agent with 0.0 confidence and no signals is handled."""
        state = _state(
            analyses={
                "market_data_analyst": _analysis("market_data_analyst", 0.0, []),
                "fundamentals_analyst": _analysis(
                    "fundamentals_analyst", 0.8, ["bullish"]
                ),
            }
        )
        synth = PortfolioSynthesizer()
        rec = synth.synthesize(state)

        assert rec.sentiment in ("bullish", "neutral")
        assert rec.confidence > 0

    def test_timeframe_preserved(self):
        """Timeframe from state flows through to the recommendation."""
        state = _state(
            analyses={
                "market_data_analyst": _analysis(
                    "market_data_analyst", 0.8, ["bullish"]
                ),
            },
            timeframe="short_term",
        )
        synth = PortfolioSynthesizer()
        rec = synth.synthesize(state)

        assert rec.estimated_timeframe == "short_term"


# Scoring tests


class TestScoring:
    """Unit tests for deterministic signal scoring."""

    def test_score_signals_empty(self):
        synth = PortfolioSynthesizer()
        assert synth._score_signals({}) == []

    def test_score_signals_bullish(self):
        analysis = _analysis("a1", 0.8, ["bullish"])
        synth = PortfolioSynthesizer()
        scored = synth._score_signals({"a1": analysis})

        assert len(scored) == 1
        assert scored[0].score == pytest.approx(0.8, rel=0.01)
        assert scored[0].confidence == 0.8

    def test_score_signals_bearish(self):
        analysis = _analysis("a1", 0.7, ["bearish"])
        synth = PortfolioSynthesizer()
        scored = synth._score_signals({"a1": analysis})

        assert len(scored) == 1
        assert scored[0].score == pytest.approx(-0.7, rel=0.01)

    def test_score_signals_neutral(self):
        analysis = _analysis("a1", 0.9, ["neutral"])
        synth = PortfolioSynthesizer()
        scored = synth._score_signals({"a1": analysis})

        assert len(scored) == 1
        assert scored[0].score == pytest.approx(0.0, rel=0.01)  # neutral * confidence


class TestWeightedScore:
    def test_no_signals(self):
        assert PortfolioSynthesizer._weighted_score([]) == 0.0

    def test_all_neutral(self):
        from dataclasses import dataclass

        scored = [
            _FakeScoredSignal("bullish", 0.0),
            _FakeScoredSignal("neutral", 0.0),
        ]
        assert PortfolioSynthesizer._weighted_score(scored) == 0.0

    def test_mixed(self):
        scored = [
            _FakeScoredSignal("bullish", 0.8),
            _FakeScoredSignal("bearish", -0.5),
            _FakeScoredSignal("neutral", 0.0),
        ]
        result = PortfolioSynthesizer._weighted_score(scored)
        # Only directional (bullish=0.8, bearish=-0.5) count
        assert result == pytest.approx(0.15, rel=0.01)


class TestSentimentFromScore:
    def test_strong_bullish(self):
        assert PortfolioSynthesizer._sentiment_from_score(0.5) == "bullish"

    def test_weak_bullish(self):
        assert PortfolioSynthesizer._sentiment_from_score(0.15) == "bullish"

    def test_barely_bullish(self):
        assert PortfolioSynthesizer._sentiment_from_score(0.14) == "neutral"

    def test_neutral(self):
        assert PortfolioSynthesizer._sentiment_from_score(0.0) == "neutral"

    def test_bearish(self):
        assert PortfolioSynthesizer._sentiment_from_score(-0.3) == "bearish"

    def test_barely_bearish(self):
        assert PortfolioSynthesizer._sentiment_from_score(-0.14) == "neutral"

    def test_strong_bearish(self):
        assert PortfolioSynthesizer._sentiment_from_score(-0.8) == "bearish"


# Confidence tests


class TestConfidence:
    def test_empty_analyses(self):
        state = _state(analyses={})
        assert PortfolioSynthesizer._confidence(state, [], 0.0, []) == 0.0

    def test_high_confidence_scenario(self):
        state = _state(
            analyses={
                "a1": _analysis("a1", 0.9, ["bullish", "bullish"]),
                "a2": _analysis("a2", 0.85, ["bullish"]),
            }
        )
        synth = PortfolioSynthesizer()
        scored = synth._score_signals(state.analyses)
        conf = PortfolioSynthesizer._confidence(state, scored, 0.8, [])

        assert conf > 0.6
        assert conf <= 1.0

    def test_conflict_penalty(self):
        state = _state(
            analyses={
                "a1": _analysis("a1", 0.8, ["bullish"]),
            }
        )
        conf_no_conflict = PortfolioSynthesizer._confidence(
            state,
            [_FakeScoredSignal("bullish", 0.8)],
            0.8,
            [],
        )
        conf_with_conflict = PortfolioSynthesizer._confidence(
            state,
            [_FakeScoredSignal("bullish", 0.8)],
            0.8,
            ["conflict1", "conflict2", "conflict3"],
        )

        assert conf_with_conflict < conf_no_conflict

    def test_error_penalty(self):
        state_no_err = _state(analyses={"a1": _analysis("a1", 0.8, ["bullish"])})
        state_with_err = _state(
            analyses={"a1": _analysis("a1", 0.8, ["bullish"])},
            errors=["error1", "error2"],
        )
        synth = PortfolioSynthesizer()
        scored = synth._score_signals(state_no_err.analyses)

        conf_no_err = PortfolioSynthesizer._confidence(state_no_err, scored, 0.8, [])
        conf_with_err = PortfolioSynthesizer._confidence(
            state_with_err, scored, 0.8, []
        )

        assert conf_with_err < conf_no_err


# Risk level tests


class TestRiskLevel:
    def test_high_confidence_no_conflicts_low_risk(self):
        state = _state(analyses={"a1": _analysis("a1", 0.8, ["bullish"])})
        assert PortfolioSynthesizer._risk_level(state, 0.8, []) == "low"

    def test_low_confidence_high_risk(self):
        state = _state(analyses={"a1": _analysis("a1", 0.8, ["bullish"])})
        assert PortfolioSynthesizer._risk_level(state, 0.3, []) == "high"

    def test_many_conflicts_high_risk(self):
        state = _state(analyses={"a1": _analysis("a1", 0.8, ["bullish"])})
        conflicts = [f"conflict_{i}" for i in range(6)]
        assert PortfolioSynthesizer._risk_level(state, 0.8, conflicts) == "high"

    def test_failed_agents_medium_risk(self):
        state = _state(
            analyses={
                "a1": _analysis("a1", 0.0, []),
                "a2": _analysis("a2", 0.8, ["bullish"]),
            }
        )
        assert PortfolioSynthesizer._risk_level(state, 0.6, []) == "medium"

    def test_two_failed_agents_high_risk(self):
        state = _state(
            analyses={
                "a1": _analysis("a1", 0.0, []),
                "a2": _analysis("a2", 0.0, []),
                "a3": _analysis("a3", 0.8, ["bullish"]),
            }
        )
        assert PortfolioSynthesizer._risk_level(state, 0.6, []) == "high"

    def test_medium_confidence_no_conflicts(self):
        state = _state(analyses={"a1": _analysis("a1", 0.8, ["bullish"])})
        assert PortfolioSynthesizer._risk_level(state, 0.5, []) == "medium"


# Key signals tests


class TestKeySignals:
    def test_prefers_matching_direction_for_bullish(self):
        scored = [
            _FakeScoredSignal("bullish", 0.3),
            _FakeScoredSignal("bearish", 0.9),  # stronger but opposite
        ]
        signals = PortfolioSynthesizer._key_signals(scored, "bullish")
        names = [s.name for s in signals]
        assert "bullish_signal" in names  # bullish signal included despite lower score

    def test_neutral_takes_strongest(self):
        scored = [
            _FakeScoredSignal("bullish", 0.9),
            _FakeScoredSignal("bearish", 0.8),
        ]
        signals = PortfolioSynthesizer._key_signals(scored, "neutral")
        names = [s.name for s in signals]
        # Both should appear since they're different
        assert "bullish_signal" in names
        assert "bearish_signal" in names

    def test_empty_scored_signals(self):
        assert PortfolioSynthesizer._key_signals([], "bullish") == []

    def test_deduplicates_same_name_and_direction(self):
        # Two entries with same signal name and direction but different values
        from dataclasses import dataclass

        s1 = Signal(name="momentum", value="+5%", direction="bullish")
        s2 = Signal(name="momentum", value="+3%", direction="bullish")

        scored = [
            _FakeScoredSignal("bullish", 0.9, signal=s1),
            _FakeScoredSignal("bullish", 0.7, signal=s2),
        ]
        signals = PortfolioSynthesizer._key_signals(scored, "bullish")
        # Should only include one of the two (deduped by name+direction)
        assert len(signals) == 1


# Rationale tests (deterministic fallback)


class TestDeterministicRationale:
    def test_rationale_contains_sentiment_and_confidence(self):
        state = _state(
            analyses={
                "a1": _analysis("a1", 0.8, ["bullish"]),
            }
        )
        synth = PortfolioSynthesizer()
        rec = synth.synthesize(state)

        assert "bullish" in rec.rationale.lower()
        assert "confidence" in rec.rationale.lower()

    def test_rationale_mentions_conflicts_when_present(self):
        state = _state(
            analyses={
                "a1": _analysis("a1", 0.8, ["bullish"]),
            },
            debates=[
                DebateRound(
                    round_number=1,
                    contributions=[
                        DebateContribution(
                            agent_name="a2",
                            challenge_to="a1",
                            message="Risk is underpriced.",
                            supporting_evidence=[],
                        ),
                    ],
                ),
            ],
        )
        synth = PortfolioSynthesizer()
        rec = synth.synthesize(state)

        assert "conflict" in rec.rationale.lower()

    def test_rationale_no_conflicts_when_clean(self):
        state = _state(
            analyses={
                "a1": _analysis("a1", 0.8, ["bullish"]),
            },
        )
        synth = PortfolioSynthesizer()
        rec = synth.synthesize(state)

        assert "no material cross-agent conflicts" in rec.rationale.lower()


# LLM rationale tests


class TestLLMRationale:
    def test_llm_rationale_used_when_available(self):
        """When LLM returns valid JSON, its rationale is used."""
        synthetic_rationale = (
            "The technical and fundamental signals align for a bullish outlook "
            "with strong earnings momentum supporting the positive view."
        )
        mock_llm = MockLLM(rationale=json.dumps({"rationale": synthetic_rationale}))
        state = _state(
            analyses={
                "a1": _analysis("a1", 0.8, ["bullish"]),
            }
        )
        synth = PortfolioSynthesizer(llm=mock_llm)
        rec = synth.synthesize(state)

        assert (
            "technical and fundamental signals" in rec.rationale.lower()
            or "bullish outlook" in rec.rationale.lower()
        )

    def test_llm_invalid_json_falls_back_to_deterministic(self):
        """When LLM returns garbage, fall back to deterministic."""
        mock_llm = MockLLM(rationale="Not valid JSON at all!!!")
        state = _state(
            analyses={
                "a1": _analysis("a1", 0.8, ["bullish"]),
            }
        )
        synth = PortfolioSynthesizer(llm=mock_llm)
        rec = synth.synthesize(state)

        # Should still produce a valid recommendation
        assert rec.sentiment == "bullish"
        # And the rationale should be the deterministic version
        assert "confidence" in rec.rationale.lower()

    def test_llm_short_rationale_falls_back(self):
        """LLM returning < 15 characters should fall back."""
        mock_llm = MockLLM(rationale=json.dumps({"rationale": "Buy it."}))
        state = _state(
            analyses={
                "a1": _analysis("a1", 0.8, ["bullish"]),
            }
        )
        synth = PortfolioSynthesizer(llm=mock_llm)
        rec = synth.synthesize(state)

        assert "confidence" in rec.rationale.lower()

    def test_llm_none_uses_deterministic(self):
        """No LLM provided → deterministic rationale."""
        state = _state(
            analyses={
                "a1": _analysis("a1", 0.8, ["bullish"]),
            }
        )
        synth = PortfolioSynthesizer(llm=None)
        rec = synth.synthesize(state)

        assert "confidence" in rec.rationale.lower()
        assert "bullish" in rec.rationale.lower()

    def test_llm_receives_context_in_prompt(self):
        """The LLM should get analyses and debate data in the prompt."""
        mock_llm = MockLLM()
        state = _state(
            analyses={
                "test_agent": _analysis("test_agent", 0.75, ["bullish", "neutral"]),
            }
        )
        synth = PortfolioSynthesizer(llm=mock_llm)
        synth.synthesize(state)

        assert mock_llm.last_messages is not None
        full_prompt = " ".join(m.get("content", "") for m in mock_llm.last_messages)
        assert "test_agent" in full_prompt
        assert "AAPL" in full_prompt


# Conflicting signals tests


class TestConflictingSignals:
    def test_bullish_sentiment_with_bearish_signal(self):
        state = _state(
            analyses={
                "a1": _analysis("a1", 0.8, ["bullish"]),
                "a2": _analysis("a2", 0.7, ["bearish"]),
            }
        )
        synth = PortfolioSynthesizer()
        conflicts = synth._conflicting_signals(state, "bullish")

        assert len(conflicts) >= 1
        assert any("bearish" in c for c in conflicts)

    def test_neutral_sentiment_flags_all_directional(self):
        state = _state(
            analyses={
                "a1": _analysis("a1", 0.8, ["bullish"]),
                "a2": _analysis("a2", 0.7, ["bearish"]),
            }
        )
        synth = PortfolioSynthesizer()
        conflicts = synth._conflicting_signals(state, "neutral")

        # Both bullish and bearish signals are conflicts for neutral
        assert len(conflicts) >= 2

    def test_no_conflicts_when_aligned(self):
        state = _state(
            analyses={
                "a1": _analysis("a1", 0.8, ["bullish"]),
                "a2": _analysis("a2", 0.7, ["bullish"]),
            }
        )
        synth = PortfolioSynthesizer()
        conflicts = synth._conflicting_signals(state, "bullish")

        assert len(conflicts) == 0


# Edge cases


class TestEdgeCases:
    def test_analyst_without_signals(self):
        """An analysis with no signals should not crash synthesis."""
        state = _state(
            analyses={
                "a1": _analysis("a1", 0.8, []),
                "a2": _analysis("a2", 0.7, ["bullish"]),
            }
        )
        synth = PortfolioSynthesizer()
        rec = synth.synthesize(state)

        assert rec.sentiment in ("bullish", "neutral")

    def test_zero_signal_agents_but_good_confidence(self):
        """No signals across any agent → neutral with moderate confidence."""
        state = _state(
            analyses={
                "a1": _analysis("a1", 0.9, []),
                "a2": _analysis("a2", 0.85, []),
            }
        )
        synth = PortfolioSynthesizer()
        rec = synth.synthesize(state)

        assert rec.sentiment == "neutral"
        # Confidence from agent confidence alone (no evidence factor)
        assert rec.confidence > 0

    def test_all_neutral_signals(self):
        """All neutral signals → neutral sentiment."""
        state = _state(
            analyses={
                "a1": _analysis("a1", 0.8, ["neutral", "neutral"]),
                "a2": _analysis("a2", 0.7, ["neutral"]),
            }
        )
        synth = PortfolioSynthesizer()
        rec = synth.synthesize(state)

        assert rec.sentiment == "neutral"

    def test_single_directional_signal(self):
        """Single bullish signal should produce bullish sentiment."""
        state = _state(
            analyses={
                "a1": _analysis("a1", 0.8, ["bullish"]),
                "a2": _analysis("a2", 0.6, ["neutral"]),
            }
        )
        synth = PortfolioSynthesizer()
        rec = synth.synthesize(state)

        assert rec.sentiment == "bullish"


# FakeScoredSignal for testing weighted_score etc. without Signal objects
from dataclasses import dataclass


@dataclass(frozen=True)
class _FakeScoredSignal:
    """Minimal scored signal for unit tests."""

    signal_name: str = "test"
    direction: str = "bullish"
    value: str | float | bool = True
    confidence: float = 1.0
    score: float = 1.0

    def __init__(
        self,
        direction: str = "bullish",
        score: float = 1.0,
        signal: Signal | None = None,
    ):
        if signal is not None:
            object.__setattr__(self, "signal_name", signal.name)
            object.__setattr__(self, "direction", signal.direction)
            object.__setattr__(self, "value", signal.value)
        else:
            object.__setattr__(self, "signal_name", f"{direction}_signal")
            object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "score", score)

    @property
    def signal(self) -> Signal:
        return Signal(
            name=self.signal_name,
            value=self.value,
            direction=self.direction,
        )
