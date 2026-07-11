"""
Agent definitions for the LangGraph workflow.

Each agent is a BaseAgent subclass registered in the shared registry.

Usage:
    from agents import registry, BaseAgent
    from agents.market_data_analyst import MarketDataAnalyst
    from agents.fundamentals_analyst import FundamentalsAnalyst
    from agents.news_intelligence_agent import NewsIntelligenceAgent
    from agents.technical_analyst import TechnicalAnalyst
    registry.register(MarketDataAnalyst(llm=...))
    registry.register(FundamentalsAnalyst(llm=...))
    registry.register(NewsIntelligenceAgent(llm=...))
    registry.register(TechnicalAnalyst(llm=...))
"""

from agents.base_agent import BaseAgent
from agents.registry import AgentRegistry, registry
from agents.market_data_analyst import MarketDataAnalyst
from agents.fundamentals_analyst import FundamentalsAnalyst
from agents.news_intelligence_agent import NewsIntelligenceAgent
from agents.technical_analyst import TechnicalAnalyst
from agents.debate_moderator import DebateModerator
from agents.portfolio_synthesizer import PortfolioSynthesizer

__all__ = [
    "BaseAgent",
    "AgentRegistry",
    "registry",
    "MarketDataAnalyst",
    "FundamentalsAnalyst",
    "NewsIntelligenceAgent",
    "TechnicalAnalyst",
    "DebateModerator",
    "PortfolioSynthesizer",
]
