"""
Portfolio synthesizer agent.

The synthesizer converts analyst outputs plus debate critiques into a final
investment recommendation. It does not fetch new data.

All scoring, confidence, risk, and conflict detection is fully deterministic.
The LLM is used only for narrative rationale generation (with a deterministic
fallback when the LLM is unavailable or fails).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseLanguageModel

from models import AgentAnalysis, Recommendation, ResearchState, Signal


_DIRECTION_SCORE = {
    "bullish": 1.0,
    "bearish": -1.0,
    "neutral": 0.0,
}


@dataclass(frozen=True)
class _ScoredSignal:
    agent_name: str
    signal: Signal
    confidence: float
    score: float


class PortfolioSynthesizer:
    """Aggregates agent signals into a final recommendation."""

    def __init__(self, llm: BaseLanguageModel | None = None) -> None:
        self.llm = llm

    # Public API

    def synthesize(self, state: ResearchState) -> Recommendation:
        """Create a final recommendation from analyses and debate rounds."""
        scored_signals = self._score_signals(state.analyses)
        weighted_score = self._weighted_score(scored_signals)
        sentiment = self._sentiment_from_score(weighted_score)
        conflicts = self._conflicting_signals(state, sentiment)
        confidence = self._confidence(state, scored_signals, weighted_score, conflicts)
        risk_level = self._risk_level(state, confidence, conflicts)
        key_signals = self._key_signals(scored_signals, sentiment)

        # Try LLM-powered narrative; fall back to deterministic rationale
        rationale = self._generate_rationale(
            state=state,
            sentiment=sentiment,
            confidence=confidence,
            risk_level=risk_level,
            weighted_score=weighted_score,
            key_signals=key_signals,
            conflicts=conflicts,
            scored_signals=scored_signals,
        )

        return Recommendation(
            symbol=state.symbol,
            sentiment=sentiment,
            confidence=confidence,
            rationale=rationale,
            risk_level=risk_level,
            estimated_timeframe=state.timeframe,
            key_signals=key_signals,
            conflicting_signals=conflicts,
        )

    # Deterministic scoring

    @staticmethod
    def _score_signals(
        analyses: dict[str, AgentAnalysis],
    ) -> list[_ScoredSignal]:
        """Convert every agent signal into a scored entry.

        score = direction_score × agent_confidence
        """
        scored: list[_ScoredSignal] = []

        for agent_name, analysis in analyses.items():
            for signal in analysis.signals:
                direction_score = _DIRECTION_SCORE.get(signal.direction, 0.0)
                scored.append(
                    _ScoredSignal(
                        agent_name=agent_name,
                        signal=signal,
                        confidence=analysis.agent_confidence,
                        score=direction_score * analysis.agent_confidence,
                    )
                )

        return scored

    @staticmethod
    def _weighted_score(scored_signals: list[_ScoredSignal]) -> float:
        """Mean of all non-neutral signal scores.

        Returns 0.0 when no directional signals exist.
        """
        directional = [
            scored.score
            for scored in scored_signals
            if scored.signal.direction in ("bullish", "bearish")
        ]
        if not directional:
            return 0.0

        return sum(directional) / len(directional)

    @staticmethod
    def _sentiment_from_score(weighted_score: float) -> str:
        """Map a weighted score to a sentiment label.

        Thresholds: >= +0.15 → bullish, <= -0.15 → bearish, else neutral.
        """
        if weighted_score >= 0.15:
            return "bullish"
        if weighted_score <= -0.15:
            return "bearish"
        return "neutral"

    def _conflicting_signals(
        self,
        state: ResearchState,
        sentiment: str,
    ) -> list[str]:
        """Collect signals and debate critiques that oppose the final sentiment."""
        conflicts: list[str] = []

        for agent_name, analysis in state.analyses.items():
            for signal in analysis.signals:
                if sentiment == "bullish" and signal.direction == "bearish":
                    conflicts.append(
                        f"{agent_name}.{signal.name}: {signal.value} is bearish"
                    )
                elif sentiment == "bearish" and signal.direction == "bullish":
                    conflicts.append(
                        f"{agent_name}.{signal.name}: {signal.value} is bullish"
                    )
                elif sentiment == "neutral" and signal.direction in (
                    "bullish",
                    "bearish",
                ):
                    conflicts.append(
                        f"{agent_name}.{signal.name}: {signal.direction} signal "
                        "offset by opposing evidence"
                    )

        for debate_round in state.debate_rounds:
            for contribution in debate_round.contributions:
                if contribution.challenge_to:
                    conflicts.append(
                        f"Round {debate_round.round_number}: "
                        f"{contribution.agent_name} challenged "
                        f"{contribution.challenge_to}: {contribution.message}"
                    )

        return self._dedupe(conflicts)

    @staticmethod
    def _confidence(
        state: ResearchState,
        scored_signals: list[_ScoredSignal],
        weighted_score: float,
        conflicts: list[str],
    ) -> float:
        """Calculate overall confidence (0.0–1.0).

        Components:
          - 45 % average agent confidence
          - 30 % evidence factor (how many directional signals exist)
          - 25 % conviction (strength of directional signal)
          - Penalties for conflicts and errors
        """
        if not state.analyses:
            return 0.0

        agent_confidence = sum(
            analysis.agent_confidence for analysis in state.analyses.values()
        ) / len(state.analyses)

        directional_count = sum(
            1
            for scored in scored_signals
            if scored.signal.direction in ("bullish", "bearish")
        )
        evidence_factor = min(1.0, directional_count / 6) if directional_count else 0.25
        conviction = min(1.0, abs(weighted_score))
        conflict_penalty = min(0.35, len(conflicts) * 0.04)
        error_penalty = min(0.20, len(state.errors) * 0.05)

        confidence = (
            0.45 * agent_confidence
            + 0.30 * evidence_factor
            + 0.25 * conviction
            - conflict_penalty
            - error_penalty
        )

        return round(max(0.0, min(1.0, confidence)), 2)

    @staticmethod
    def _risk_level(
        state: ResearchState,
        confidence: float,
        conflicts: list[str],
    ) -> str:
        """Determine risk level (low / medium / high).

        High if: confidence < 0.35, or 6+ conflicts, or 2+ agents failed.
        Medium if: confidence < 0.65, or any conflicts, or any failed agents.
        Otherwise low.
        """
        failed_agents = sum(
            1
            for analysis in state.analyses.values()
            if analysis.agent_confidence == 0.0
        )

        if confidence < 0.35 or len(conflicts) >= 6 or failed_agents >= 2:
            return "high"
        if confidence < 0.65 or conflicts or failed_agents:
            return "medium"
        return "low"

    @staticmethod
    def _key_signals(
        scored_signals: list[_ScoredSignal],
        sentiment: str,
        limit: int = 8,
    ) -> list[Signal]:
        """Pick the most relevant signals for the recommendation.

        For bullish/bearish sentiment, prefer signals matching that direction.
        For neutral, take the strongest signals regardless of direction.
        """
        if sentiment in ("bullish", "bearish"):
            preferred = [
                scored
                for scored in scored_signals
                if scored.signal.direction == sentiment
            ]
        else:
            preferred = scored_signals

        ranked = sorted(preferred, key=lambda scored: abs(scored.score), reverse=True)
        key_signals: list[Signal] = []
        seen: set[tuple[str, str]] = set()

        for scored in ranked:
            key = (scored.signal.name, scored.signal.direction)
            if key in seen:
                continue
            seen.add(key)
            key_signals.append(scored.signal)
            if len(key_signals) >= limit:
                break

        return key_signals

    # Rationale generation (LLM with deterministic fallback)

    def _generate_rationale(
        self,
        state: ResearchState,
        sentiment: str,
        confidence: float,
        risk_level: str,
        weighted_score: float,
        key_signals: list[Signal],
        conflicts: list[str],
        scored_signals: list[_ScoredSignal],
    ) -> str:
        """Generate an investment narrative.

        Tries the LLM first for a richer, contextual rationale. Falls back to
        a fully deterministic formulaic rationale when the LLM is unavailable,
        returns invalid output, or raises.
        """
        if self.llm is not None:
            try:
                llm_rationale = self._invoke_llm_rationale(
                    state=state,
                    sentiment=sentiment,
                    confidence=confidence,
                    risk_level=risk_level,
                    weighted_score=weighted_score,
                    key_signals=key_signals,
                    conflicts=conflicts,
                    scored_signals=scored_signals,
                )
                if llm_rationale and len(llm_rationale) > 15:
                    return llm_rationale
            except Exception:
                pass

        return self._deterministic_rationale(
            state=state,
            sentiment=sentiment,
            confidence=confidence,
            risk_level=risk_level,
            weighted_score=weighted_score,
            key_signals=key_signals,
            conflicts=conflicts,
        )

    def _invoke_llm_rationale(
        self,
        state: ResearchState,
        sentiment: str,
        confidence: float,
        risk_level: str,
        weighted_score: float,
        key_signals: list[Signal],
        conflicts: list[str],
        scored_signals: list[_ScoredSignal],
    ) -> str | None:
        """Call the LLM for a narrative recommendation rationale.

        Expects a JSON response with a 'rationale' key. Returns None if the
        response cannot be parsed or fails validation.
        """
        prompt = self._build_synthesis_prompt(
            state=state,
            sentiment=sentiment,
            confidence=confidence,
            risk_level=risk_level,
            weighted_score=weighted_score,
            key_signals=key_signals,
            conflicts=conflicts,
            scored_signals=scored_signals,
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior portfolio strategist. Your job is to "
                    "write a concise investment rationale (3-5 sentences) that "
                    "synthesises all analyst outputs and debate critiques into "
                    "a clear, evidence-based narrative. Return JSON only:\n\n"
                    "{\n"
                    '  "rationale": "3-5 sentence investment narrative"\n'
                    "}\n\n"
                    "Rules:\n"
                    "- Reference specific signals and confidence levels.\n"
                    "- Mention any material cross-agent conflicts.\n"
                    "- Do NOT invent data or market predictions.\n"
                    "- Keep the tone neutral and professional.\n"
                    "- Return valid JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        response = self.llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

        payload: dict[str, Any] = json.loads(content)
        rationale = payload.get("rationale", "")

        if not isinstance(rationale, str) or len(rationale.strip()) < 15:
            return None

        return rationale.strip()

    def _build_synthesis_prompt(
        self,
        state: ResearchState,
        sentiment: str,
        confidence: float,
        risk_level: str,
        weighted_score: float,
        key_signals: list[Signal],
        conflicts: list[str],
        scored_signals: list[_ScoredSignal],
    ) -> str:
        """Build a structured prompt with all synthesis data for the LLM."""
        analyses_payload = {
            name: {
                "confidence": analysis.agent_confidence,
                "summary": analysis.summary,
                "signals": [
                    {
                        "name": signal.name,
                        "direction": signal.direction,
                        "value": signal.value,
                    }
                    for signal in analysis.signals
                ],
                "reasoning": analysis.reasoning,
            }
            for name, analysis in state.analyses.items()
        }

        debate_summary = []
        for debate_round in state.debate_rounds:
            for contribution in debate_round.contributions:
                if contribution.challenge_to:
                    debate_summary.append(
                        f"Round {debate_round.round_number}: "
                        f"{contribution.agent_name} challenged "
                        f"{contribution.challenge_to}: "
                        f"{contribution.message}"
                    )

        agent_breakdown = []
        for agent_name, analysis in state.analyses.items():
            signal_text = ", ".join(
                f"{s.name} ({s.direction})" for s in analysis.signals[:6]
            )
            agent_breakdown.append(
                f"{agent_name}: confidence={analysis.agent_confidence:.2f}, "
                f"signals=[{signal_text or 'none'}]"
            )

        return json.dumps(
            {
                "task": "Synthesise the following analyst outputs into a final investment narrative.",
                "symbol": state.symbol,
                "timeframe": state.timeframe,
                "computed_sentiment": sentiment,
                "computed_confidence": confidence,
                "computed_risk_level": risk_level,
                "weighted_directional_score": round(weighted_score, 3),
                "analyses": analyses_payload,
                "agent_breakdown": agent_breakdown,
                "debate_highlights": debate_summary,
                "conflicting_signals": conflicts,
                "error_count": len(state.errors),
            },
            indent=2,
        )

    # Deterministic fallback rationale

    @staticmethod
    def _deterministic_rationale(
        state: ResearchState,
        sentiment: str,
        confidence: float,
        risk_level: str,
        weighted_score: float,
        key_signals: list[Signal],
        conflicts: list[str],
    ) -> str:
        """Formulaic rationale used when the LLM is unavailable or fails."""
        agent_summaries = [
            f"{name} confidence {analysis.agent_confidence:.2f}"
            for name, analysis in state.analyses.items()
        ]
        key_signal_text = ", ".join(
            f"{signal.name} ({signal.direction})" for signal in key_signals[:4]
        )

        rationale = (
            f"The combined analyst signal for {state.symbol} is {sentiment} "
            f"with confidence {confidence:.2f} and {risk_level} risk. "
            f"The weighted directional score is {weighted_score:.2f}; "
            f"agent coverage: {', '.join(agent_summaries) or 'none'}. "
        )

        if key_signal_text:
            rationale += f"Primary supporting signals are {key_signal_text}. "

        if conflicts:
            rationale += (
                f"There are {len(conflicts)} unresolved conflicting "
                "signal(s), so the recommendation should be treated "
                "cautiously."
            )
        else:
            rationale += "No material cross-agent conflicts were detected."

        return rationale

    # Utility

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []

        for item in items:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)

        return deduped
