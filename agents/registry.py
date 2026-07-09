"""
Agent registry — the single source of truth for all agents.

The registry:
  - Maps agent names → BaseAgent instances
  - Runs all agents in a phase (ordered execution)
  - Enables dynamic agent discovery (no hardcoded nodes)

Usage:
    registry = AgentRegistry()
    registry.register(my_agent)
    registry.register(another_agent)
    state = registry.run_phase(state)
"""

from __future__ import annotations

from typing import Any

from models import ResearchState
from agents.base_agent import BaseAgent


class AgentRegistry:
    """
    Registry of all analyst agents.

    Maintains insertion order so agents run in a deterministic sequence.
    """

    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}
        self._order: list[str] = []

    def register(self, agent: BaseAgent) -> str:
        """
        Register an agent.

        Args:
            agent: A BaseAgent instance (must have a unique name).

        Returns:
            The agent's name (for chaining or reference).

        Raises:
            ValueError: If an agent with the same name is already registered.
        """
        if agent.name in self._agents:
            raise ValueError(
                f"Agent '{agent.name}' is already registered. "
                "Each agent must have a unique name."
            )
        self._agents[agent.name] = agent
        self._order.append(agent.name)
        return agent.name

    def get(self, name: str) -> BaseAgent | None:
        """Get an agent by name. Returns None if not found."""
        return self._agents.get(name)

    def list_all(self) -> list[BaseAgent]:
        """Return all registered agents in registration order."""
        return [self._agents[name] for name in self._order]

    def list_names(self) -> list[str]:
        """Return all agent names in registration order."""
        return list(self._order)

    def count(self) -> int:
        """Number of registered agents."""
        return len(self._agents)

    def run_phase(self, state: ResearchState) -> ResearchState:
        """
        Run all registered agents sequentially.

        Each agent reads from state, writes to state.analyses[agent.name].
        If an agent fails, it writes an error analysis and execution continues
        — one agent's failure does not block others.

        Args:
            state: The current ResearchState.

        Returns:
            Updated ResearchState with all agent analyses populated.
        """
        for agent in self.list_all():
            state = agent.run(state)
        return state

    def clear(self) -> None:
        """Remove all registered agents (useful for testing)."""
        self._agents.clear()
        self._order.clear()

    def __repr__(self) -> str:
        return f"<AgentRegistry: {len(self)} agents ({', '.join(self._order)})>"

    def __len__(self) -> int:
        return self.count()


# ── Global singleton ──
# Import this from anywhere to access the shared registry.
# In tests, you can call registry.clear() and re-register mocks.
registry = AgentRegistry()
