"""
Agent definitions for the LangGraph workflow.

Each agent is a BaseAgent subclass registered in the shared registry.

Usage:
    from agents import registry, BaseAgent
    registry.register(MyAgent(...))
    state = registry.run_phase(state)
"""

from agents.base_agent import BaseAgent
from agents.registry import AgentRegistry, registry

__all__ = [
    "BaseAgent",
    "AgentRegistry",
    "registry",
]
