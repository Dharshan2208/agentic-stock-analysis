"""
LangGraph research orchestration.

This graph wires the four analyst agents into a parallel execution workflow:

    prepare
      -> market_data_analyst
      -> fundamentals_analyst
      -> technical_analyst
      -> news_intelligence_agent
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langgraph.graph import END, START, StateGraph

from agents import (
    BaseAgent,
    FundamentalsAnalyst,
    MarketDataAnalyst,
    NewsIntelligenceAgent,
    TechnicalAnalyst,
)
from config import settings
from models import AgentAnalysis, ResearchState
from services import FundamentalsProvider, MarketDataProvider, NewsProvider


def _error_analysis(
    agent: BaseAgent, state: ResearchState, error: Exception
) -> AgentAnalysis:
    """Build a valid failed analysis when an agent raises."""
    error_msg = f"{agent.name}: {type(error).__name__}: {error}"

    return AgentAnalysis(
        agent_name=agent.name,
        symbol=state.symbol,
        agent_confidence=0.0,
        summary=f"Analysis failed: {error_msg}",
        signals=[],
        reasoning="",
        tool_calls=[],
        metadata={
            "error": error_msg,
            "phase": state.phase,
        },
    )


def _run_agent_node(agent: BaseAgent, state: ResearchState) -> dict[str, Any]:
    """
    Run one agent and return a partial LangGraph state update.

    This avoids mutating shared state in parallel branches. It preserves the
    graceful degradation behavior from BaseAgent.run(), but returns only the
    fields this branch owns.
    """
    try:
        analysis = agent.analyze(state)
        errors: list[str] = []
    except Exception as exc:
        analysis = _error_analysis(agent, state, exc)
        errors = [analysis.metadata["error"]]

    return {
        "analyses": {
            agent.name: analysis,
        },
        "agent_execution_order": [
            agent.name,
        ],
        "errors": errors,
    }


def build_research_graph(
    llm: BaseLanguageModel,
    market_data_provider: MarketDataProvider | None = None,
    fundamentals_provider: FundamentalsProvider | None = None,
    news_provider: NewsProvider | None = None,
):
    """
    Build the M2.1 research graph.

    Args:
        llm: Shared LLM used by all analyst agents.
        market_data_provider: Optional injected market provider.
        fundamentals_provider: Optional injected fundamentals provider.
        news_provider: Optional injected news provider.

    Returns:
        A compiled LangGraph app.
    """
    market_data_agent = MarketDataAnalyst(
        llm=llm,
        market_data_provider=market_data_provider,
    )
    fundamentals_agent = FundamentalsAnalyst(
        llm=llm,
        fundamentals_provider=fundamentals_provider,
    )
    technical_agent = TechnicalAnalyst(
        llm=llm,
        market_data_provider=market_data_provider,
    )
    news_agent = NewsIntelligenceAgent(
        llm=llm,
        news_provider=news_provider,
    )

    def prepare(state: ResearchState) -> dict[str, Any]:
        return {
            "phase": "collecting_analyses",
            "max_debate_rounds": settings.max_debate_rounds,
        }

    def run_market_data(state: ResearchState) -> dict[str, Any]:
        return _run_agent_node(market_data_agent, state)

    def run_fundamentals(state: ResearchState) -> dict[str, Any]:
        return _run_agent_node(fundamentals_agent, state)

    def run_technical(state: ResearchState) -> dict[str, Any]:
        return _run_agent_node(technical_agent, state)

    def run_news(state: ResearchState) -> dict[str, Any]:
        return _run_agent_node(news_agent, state)

    def finalize(state: ResearchState) -> dict[str, Any]:
        return {
            "phase": "complete",
        }

    graph = StateGraph(ResearchState)

    graph.add_node("prepare", prepare)
    graph.add_node("market_data_analyst", run_market_data)
    graph.add_node("fundamentals_analyst", run_fundamentals)
    graph.add_node("technical_analyst", run_technical)
    graph.add_node("news_intelligence_agent", run_news)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "prepare")

    graph.add_edge("prepare", "market_data_analyst")
    graph.add_edge("prepare", "fundamentals_analyst")
    graph.add_edge("prepare", "technical_analyst")
    graph.add_edge("prepare", "news_intelligence_agent")

    graph.add_edge(
        [
            "market_data_analyst",
            "fundamentals_analyst",
            "technical_analyst",
            "news_intelligence_agent",
        ],
        "finalize",
    )

    graph.add_edge("finalize", END)

    return graph.compile()


def run_research(
    user_query: str,
    symbol: str,
    llm: BaseLanguageModel,
    sector: str | None = None,
    timeframe: str | None = None,
    market_data_provider: MarketDataProvider | None = None,
    fundamentals_provider: FundamentalsProvider | None = None,
    news_provider: NewsProvider | None = None,
) -> ResearchState:
    """
    Convenience entry point for running M2.1 research.

    Returns a ResearchState with four populated analyses.
    No recommendation is produced in M2.1.
    """
    graph = build_research_graph(
        llm=llm,
        market_data_provider=market_data_provider,
        fundamentals_provider=fundamentals_provider,
        news_provider=news_provider,
    )

    initial_state = ResearchState(
        user_query=user_query,
        symbol=symbol,
        sector=sector,
        timeframe=timeframe or settings.default_timeframe,
    )

    result = graph.invoke(initial_state)

    if isinstance(result, ResearchState):
        return result

    return ResearchState.model_validate(result)
