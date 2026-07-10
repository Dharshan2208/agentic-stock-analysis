"""
State models for the multi-agent stock research system.
All state flows through these Pydantic models for validation and type safety.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping

from datetime import datetime
from typing import Any, Dict, List, Optional, Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator
from langgraph.graph.message import add_messages


# Agent-level models
class Signal(BaseModel):
    """A single signal extracted by an analyst agent."""

    name: str = Field(description="Short identifier, e.g. 'pe_vs_sector'")
    value: str | float | bool = Field(description="The signal value")
    direction: str = Field(
        description="'bullish', 'bearish', or 'neutral'",
    )


class ToolCallRecord(BaseModel):
    """Record of a tool invocation for auditability."""

    tool_name: str
    input: dict[str, Any] = Field(default_factory=dict)
    output_summary: str = Field(default="", max_length=500)
    success: bool = True
    error: str | None = None


class AgentAnalysis(BaseModel):
    """Output from a single analyst agent."""

    agent_name: str = Field(description="e.g. 'market_data_analyst'")
    symbol: str = Field(description="Ticker symbol being analyzed")
    agent_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="How confident the agent is in its own analysis",
    )
    summary: str = Field(description="2-5 sentence analysis summary")
    signals: list[Signal] = Field(default_factory=list)
    reasoning: str = Field(
        default="",
        description="Step-by-step reasoning the agent used",
    )
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# Debate models
class DebateContribution(BaseModel):
    """A single contribution in a debate round (challenge or defense)."""

    agent_name: str
    message: str = Field(description="The argument being made")
    challenge_to: str | None = Field(
        default=None,
        description="If challenging another agent, their name",
    )
    supporting_evidence: list[str] = Field(default_factory=list)


class DebateRound(BaseModel):
    """One round of debate among agents."""

    round_number: int = Field(ge=1)
    contributions: list[DebateContribution] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = None


# Recommendation models
class Recommendation(BaseModel):
    """Final investment recommendation."""

    symbol: str
    sentiment: str = Field(
        description="'bullish', 'bearish', or 'neutral'",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall confidence in the recommendation",
    )
    rationale: str = Field(description="Full explanation")
    risk_level: str = Field(
        default="medium",
        description="'low', 'medium', or 'high'",
    )
    estimated_timeframe: str = Field(
        default="medium_term",
        description="'short_term' (<1mo), 'medium_term' (1-6mo), 'long_term' (>6mo)",
    )
    key_signals: list[Signal] = Field(default_factory=list)
    conflicting_signals: list[str] = Field(
        default_factory=list,
        description="Signals that contradict the recommendation",
    )
    generated_at: datetime = Field(default_factory=datetime.now)


def merge_analyses(
    left: dict[str, AgentAnalysis] | None,
    right: dict[str, AgentAnalysis] | None,
) -> dict[str, AgentAnalysis]:
    """Merge per-agent analyses produced by parallel LangGraph branches."""
    return {**(left or {}), **(right or {})}


# Main Research State
class ResearchState(BaseModel):
    """
    The full state flowing through the LangGraph research workflow.

    This is the single source of truth for all agent interactions,
    analyses, debate rounds, and the final recommendation.
    """

    #  Input
    user_query: str = Field(description="Raw user input")
    symbol: str = Field(description="Ticker symbol extracted from query")
    sector: str | None = Field(default=None, description="Company sector")
    timeframe: str = Field(
        default="medium_term",
        description="Analysis timeframe requested",
    )

    #  Analyses (one entry per agent)
    analyses: Annotated[dict[str, AgentAnalysis], merge_analyses] = Field(
        default_factory=dict,
        description="Keyed by agent_name. Each agent writes once.",
    )

    #  Debate
    debate_rounds: Annotated[list[DebateRound], operator.add] = Field(
        default_factory=list
    )
    max_debate_rounds: int = Field(default=3, ge=1, le=10)

    #  Recommendation
    recommendation: Recommendation | None = Field(default=None)

    #  Messages (for LangGraph compatibility)
    messages: Annotated[list, add_messages] = Field(default_factory=list)

    #  Resilience
    errors: Annotated[list[str], operator.add] = Field(default_factory=list)
    agent_execution_order: Annotated[list[str], operator.add] = Field(
        default_factory=list,
        description="Tracks which agents ran and in what order",
    )

    #  Execution control
    current_round: int = Field(default=0, description="Current debate round")
    phase: str = Field(
        default="init",
        description=(
            "Current workflow phase: "
            "'init', 'collecting_analyses', 'debating', "
            "'synthesizing', 'verifying', 'complete'"
        ),
    )

    @field_validator("symbol")
    @classmethod
    def uppercase_symbol(cls, v: str) -> str:
        return v.strip().upper()

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )  # Needed for LangGraph message type
