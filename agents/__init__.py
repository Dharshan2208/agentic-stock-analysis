"""
Agent definitions for the LangGraph workflow.

Each agent is a BaseAgent subclass registered in the shared registry.

Usage:
    from agents import registry, BaseAgent
    from agents.market_data_analyst import MarketDataAnalyst
    registry.register(MarketDataAnalyst(llm=...))
"""

from agents.base_agent import BaseAgent
from agents.registry import AgentRegistry, registry
from agents.market_data_analyst import MarketDataAnalyst

__all__ = [
    "BaseAgent",
    "AgentRegistry",
    "registry",
    "MarketDataAnalyst",
]
