"""
Base agent class that all specialized analyst agents extend.

Every agent:
  1. Has a unique name and description
  2. Receives the full ResearchState
  3. Writes its analysis to state.analyses[agent_name]
  4. Handles its own errors gracefully (never crashes the graph)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from models import ResearchState, AgentAnalysis, Signal, ToolCallRecord


class BaseAgent(ABC):
    """Abstract base for all financial analyst agents."""

    def __init__(
        self,
        name: str,
        description: str,
        llm: BaseLanguageModel,
        tools: list[BaseTool] | None = None,
    ) -> None:
        if not name.strip():
            raise ValueError("Agent name must be non-empty")
        if not description.strip():
            raise ValueError("Agent description must be non-empty")

        self.name = name
        self.description = description
        self.llm = llm
        self.tools = tools or []

        # Bind tools to LLM if any tools provided
        self._llm_with_tools = llm.bind_tools(self.tools) if self.tools else llm

    @abstractmethod
    def system_prompt(self) -> str:
        """
        Return the system prompt tailored to this agent's specialty.
        Called inside run() to build the message list.
        """
        ...

    @abstractmethod
    def analyze(self, state: ResearchState) -> AgentAnalysis:
        """
        Core analysis logic — implemented by each subclass.

        Args:
            state: The full research state (read what you need).

        Returns:
            AgentAnalysis with findings, signals, confidence, and reasoning.
        """
        ...

    def run(self, state: ResearchState) -> ResearchState:
        """
        Execute this agent's analysis and write results into state.

        This method handles:
          - Error wrapping (agent failures don't crash the graph)
          - Writing to state.analyses[self.name]
          - Tracking execution order

        Subclasses should NOT override this — implement analyze() instead.
        """
        state.agent_execution_order.append(self.name)

        try:
            analysis = self.analyze(state)
        except Exception as e:
            # Graceful degradation: record the error, produce empty analysis
            error_msg = f"{self.name}: {type(e).__name__}: {e}"
            state.errors.append(error_msg)
            analysis = AgentAnalysis(
                agent_name=self.name,
                symbol=state.symbol,
                agent_confidence=0.0,
                summary=f"Analysis failed: {error_msg}",
                signals=[],
                reasoning="",
                tool_calls=[],
                metadata={"error": error_msg, "phase": state.phase},
            )

        state.analyses[self.name] = analysis
        return state

    def build_messages(self, state: ResearchState) -> list[dict[str, Any]]:
        """
        Build the message list for LLM invocation.

        Override in subclass if you need custom message construction.
        """
        return [
            {"role": "system", "content": self.system_prompt()},
            {
                "role": "user",
                "content": (
                    f"Analyze {state.symbol} ({state.sector or 'unknown sector'}).\n"
                    f"User query: {state.user_query}\n"
                    f"Current phase: {state.phase}"
                ),
            },
        ]

    def invoke_llm(self, messages: list[dict[str, Any]]) -> str:
        """Invoke the LLM and return the response content."""
        response = self._llm_with_tools.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"
