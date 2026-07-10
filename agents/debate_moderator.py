"""
Inter-agent debate moderator.

The moderator critiques completed analyst outputs. It does not fetch fresh data
and it does not produce a final recommendation.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models import BaseLanguageModel

from models import AgentAnalysis, DebateContribution, DebateRound, ResearchState


class DebateModerator:
    """Coordinates critique between analyst agents."""

    def __init__(
        self,
        llm: BaseLanguageModel,
        max_contributions_per_round: int = 6,
    ) -> None:
        self.llm = llm
        self.max_contributions_per_round = max_contributions_per_round

    def run_round(self, state: ResearchState) -> DebateRound:
        """Create one debate round from existing analyses."""
        next_round = state.current_round + 1

        if not state.analyses:
            return DebateRound(
                round_number=next_round,
                contributions=[
                    DebateContribution(
                        agent_name="debate_moderator",
                        message="No analyst outputs are available to debate.",
                        challenge_to=None,
                        supporting_evidence=[],
                    )
                ],
            )

        contributions = self._build_rule_based_contributions(state)

        if not contributions:
            contributions = self._build_llm_contributions(state, next_round)

        return DebateRound(
            round_number=next_round,
            contributions=contributions[: self.max_contributions_per_round],
        )

    def _build_rule_based_contributions(
        self,
        state: ResearchState,
    ) -> list[DebateContribution]:
        """
        Detect obvious conflicts before asking the LLM.

        This catches bullish/bearish disagreements, low-confidence outputs, and
        missing structured signals deterministically.
        """
        contributions: list[DebateContribution] = []
        analyses = state.analyses

        agent_signal_map = {
            agent_name: self._direction_counts(analysis)
            for agent_name, analysis in analyses.items()
        }

        for challenger_name, challenger_counts in agent_signal_map.items():
            for target_name, target_counts in agent_signal_map.items():
                if challenger_name == target_name:
                    continue

                challenger_bias = self._dominant_direction(challenger_counts)
                target_bias = self._dominant_direction(target_counts)

                if (
                    challenger_bias
                    and target_bias
                    and challenger_bias != "neutral"
                    and target_bias != "neutral"
                    and challenger_bias != target_bias
                ):
                    challenger = analyses[challenger_name]
                    target = analyses[target_name]

                    contributions.append(
                        DebateContribution(
                            agent_name=challenger_name,
                            challenge_to=target_name,
                            message=(
                                f"{challenger_name} challenges {target_name}: "
                                f"its dominant signal is {challenger_bias}, while "
                                f"{target_name} is leaning {target_bias}. Resolve "
                                "this conflict before forming a recommendation."
                            ),
                            supporting_evidence=[
                                self._summarize_analysis(challenger),
                                self._summarize_analysis(target),
                            ],
                        )
                    )

        for agent_name, analysis in analyses.items():
            if analysis.agent_confidence <= 0.35:
                contributions.append(
                    DebateContribution(
                        agent_name="debate_moderator",
                        challenge_to=agent_name,
                        message=(
                            f"{agent_name} has low confidence "
                            f"({analysis.agent_confidence:.2f}). Discount its "
                            "findings unless another analyst supports them."
                        ),
                        supporting_evidence=[
                            analysis.summary,
                            analysis.reasoning,
                        ],
                    )
                )

            if not analysis.signals:
                contributions.append(
                    DebateContribution(
                        agent_name="debate_moderator",
                        challenge_to=agent_name,
                        message=(
                            f"{agent_name} produced no structured signals, which "
                            "limits cross-agent comparison."
                        ),
                        supporting_evidence=[
                            analysis.summary,
                        ],
                    )
                )

        return self._dedupe_contributions(contributions)

    def _build_llm_contributions(
        self,
        state: ResearchState,
        round_number: int,
    ) -> list[DebateContribution]:
        """
        Ask the LLM for nuanced critique if deterministic checks find nothing.

        The model is required to return JSON. Parsing failures degrade to a
        single moderator note rather than failing the graph.
        """
        prompt = self._build_debate_prompt(state, round_number)

        try:
            response = self.llm.invoke(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a financial research debate moderator. "
                            "Identify conflicts, weak assumptions, missing "
                            "evidence, confidence problems, and timeframe "
                            "mismatches between specialist analyst outputs. "
                            "Return JSON only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ]
            )
            content = response.content if hasattr(response, "content") else str(response)
            payload: dict[str, Any] = json.loads(content)
        except Exception as exc:
            return [
                DebateContribution(
                    agent_name="debate_moderator",
                    message=(
                        "LLM debate generation failed. No additional LLM "
                        f"critique was added. Error: {type(exc).__name__}: {exc}"
                    ),
                    challenge_to=None,
                    supporting_evidence=[],
                )
            ]

        contributions: list[DebateContribution] = []
        for item in payload.get("contributions", []):
            if not isinstance(item, dict):
                continue

            message = str(item.get("message", "")).strip()
            if not message:
                continue

            contributions.append(
                DebateContribution(
                    agent_name=str(item.get("agent_name", "debate_moderator")),
                    challenge_to=item.get("challenge_to"),
                    message=message,
                    supporting_evidence=[
                        str(evidence)
                        for evidence in item.get("supporting_evidence", [])
                    ],
                )
            )

        return contributions

    def _build_debate_prompt(self, state: ResearchState, round_number: int) -> str:
        analyses_payload = {
            name: {
                "confidence": analysis.agent_confidence,
                "summary": analysis.summary,
                "reasoning": analysis.reasoning,
                "signals": [
                    {
                        "name": signal.name,
                        "value": signal.value,
                        "direction": signal.direction,
                    }
                    for signal in analysis.signals
                ],
                "metadata": analysis.metadata,
            }
            for name, analysis in state.analyses.items()
        }

        prior_debate = [
            {
                "round_number": debate_round.round_number,
                "contributions": [
                    {
                        "agent_name": contribution.agent_name,
                        "challenge_to": contribution.challenge_to,
                        "message": contribution.message,
                        "supporting_evidence": contribution.supporting_evidence,
                    }
                    for contribution in debate_round.contributions
                ],
            }
            for debate_round in state.debate_rounds
        ]

        return json.dumps(
            {
                "task": (
                    "Create debate contributions that critique analyst outputs. "
                    "Focus on conflicts, assumptions, missing evidence, confidence "
                    "issues, and timeframe mismatches."
                ),
                "symbol": state.symbol,
                "timeframe": state.timeframe,
                "round_number": round_number,
                "analyses": analyses_payload,
                "prior_debate_rounds": prior_debate,
                "output_schema": {
                    "contributions": [
                        {
                            "agent_name": (
                                "name of challenging agent or debate_moderator"
                            ),
                            "challenge_to": "name of challenged agent or null",
                            "message": "specific critique",
                            "supporting_evidence": ["evidence string 1"],
                        }
                    ]
                },
                "rules": [
                    "Return JSON only.",
                    "Do not create a final recommendation.",
                    "Do not invent market data.",
                    "Use only the supplied analyses.",
                    "Prefer concrete conflicts over generic comments.",
                ],
            },
            indent=2,
        )

    @staticmethod
    def _direction_counts(analysis: AgentAnalysis) -> dict[str, int]:
        counts = {
            "bullish": 0,
            "bearish": 0,
            "neutral": 0,
        }

        for signal in analysis.signals:
            if signal.direction in counts:
                counts[signal.direction] += 1

        return counts

    @staticmethod
    def _dominant_direction(counts: dict[str, int]) -> str | None:
        max_count = max(counts.values()) if counts else 0
        if max_count == 0:
            return None

        winners = [
            direction
            for direction, count in counts.items()
            if count == max_count
        ]

        if len(winners) != 1:
            return "neutral"

        return winners[0]

    @staticmethod
    def _summarize_analysis(analysis: AgentAnalysis) -> str:
        signal_summary = ", ".join(
            f"{signal.name}={signal.direction}"
            for signal in analysis.signals[:5]
        )

        return (
            f"{analysis.agent_name}: confidence={analysis.agent_confidence:.2f}; "
            f"signals=[{signal_summary}]; summary={analysis.summary}"
        )

    @staticmethod
    def _dedupe_contributions(
        contributions: list[DebateContribution],
    ) -> list[DebateContribution]:
        seen: set[tuple[str, str | None, str]] = set()
        deduped: list[DebateContribution] = []

        for contribution in contributions:
            key = (
                contribution.agent_name,
                contribution.challenge_to,
                contribution.message,
            )
            if key in seen:
                continue

            seen.add(key)
            deduped.append(contribution)

        return deduped
