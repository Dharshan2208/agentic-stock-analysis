"""
LangGraph research orchestration.

This graph wires the four analyst agents into a parallel execution workflow,
then runs inter-agent debate and synthesizes a final recommendation.

    prepare
      -> market_data_analyst
      -> fundamentals_analyst
      -> technical_analyst
      -> news_intelligence_agent
    begin_debate
      -> debate_round
    synthesize
      -> finalize
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langgraph.graph import END, START, StateGraph

from agents import (
    BaseAgent,
    DebateModerator,
    FundamentalsAnalyst,
    MarketDataAnalyst,
    NewsIntelligenceAgent,
    PortfolioSynthesizer,
    TechnicalAnalyst,
)
from config import settings
from models import (
    AgentAnalysis,
    DebateContribution,
    DebateRound,
    Recommendation,
    ResearchState,
)
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


def _error_debate_round(state: ResearchState, error: Exception) -> DebateRound:
    """Build a valid debate round when debate moderation raises."""
    error_msg = f"debate_moderator: {type(error).__name__}: {error}"

    return DebateRound(
        round_number=state.current_round + 1,
        contributions=[
            DebateContribution(
                agent_name="debate_moderator",
                challenge_to=None,
                message=f"Debate round failed: {error_msg}",
                supporting_evidence=[],
            )
        ],
    )


def _fallback_recommendation(state: ResearchState, error: Exception) -> Recommendation:
    """Build a conservative recommendation when synthesis fails."""
    error_msg = f"portfolio_synthesizer: {type(error).__name__}: {error}"

    return Recommendation(
        symbol=state.symbol,
        sentiment="neutral",
        confidence=0.0,
        rationale=f"Recommendation synthesis failed: {error_msg}",
        risk_level="high",
        estimated_timeframe=state.timeframe,
        key_signals=[],
        conflicting_signals=[error_msg],
    )


def _should_continue_debate(state: ResearchState) -> str:
    """Decide whether to run another debate round or move to synthesis.

    Extracted as a module-level function for testability.
    Returns ``"synthesize"`` when debate is complete, ``"debate_round"`` otherwise.
    """
    if state.current_round >= state.max_debate_rounds:
        return "synthesize"

    latest_round = state.debate_rounds[-1] if state.debate_rounds else None
    if latest_round is None:
        return "synthesize"

    has_substantive_challenge = any(
        contribution.challenge_to is not None
        for contribution in latest_round.contributions
    )

    if not has_substantive_challenge:
        return "synthesize"

    return "debate_round"


def build_research_graph(
    llm: BaseLanguageModel,
    market_data_provider: MarketDataProvider | None = None,
    fundamentals_provider: FundamentalsProvider | None = None,
    news_provider: NewsProvider | None = None,
):
    """
    Build the research graph.

    Args:
        llm: Shared LLM used by all analyst agents.
        market_data_provider: Optional injected market provider.
        fundamentals_provider: Optional injected fundamentals provider.
        news_provider: Optional injected news provider.

    Returns:
        A compiled LangGraph app that produces analyses, debate, and a recommendation.
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
    debate_moderator = DebateModerator(llm=llm)
    portfolio_synthesizer = PortfolioSynthesizer(llm=llm)

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

    def begin_debate(state: ResearchState) -> dict[str, Any]:
        return {
            "phase": "debating",
            "current_round": 0,
        }

    def run_debate_round(state: ResearchState) -> dict[str, Any]:
        try:
            debate_round = debate_moderator.run_round(state)
            errors: list[str] = []
        except Exception as exc:
            debate_round = _error_debate_round(state, exc)
            errors = [
                f"debate_moderator: {type(exc).__name__}: {exc}",
            ]

        return {
            "debate_rounds": [
                debate_round,
            ],
            "current_round": state.current_round + 1,
            "errors": errors,
        }

    def should_continue_debate(state: ResearchState) -> str:
        return _should_continue_debate(state)

    def synthesize(state: ResearchState) -> dict[str, Any]:
        try:
            recommendation = portfolio_synthesizer.synthesize(state)
            errors: list[str] = []
        except Exception as exc:
            recommendation = _fallback_recommendation(state, exc)
            errors = [
                f"portfolio_synthesizer: {type(exc).__name__}: {exc}",
            ]

        return {
            "phase": "synthesizing",
            "recommendation": recommendation,
            "errors": errors,
        }

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
    graph.add_node("begin_debate", begin_debate)
    graph.add_node("debate_round", run_debate_round)
    graph.add_node("synthesize", synthesize)
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
        "begin_debate",
    )

    graph.add_edge("begin_debate", "debate_round")
    graph.add_conditional_edges(
        "debate_round",
        should_continue_debate,
        {
            "debate_round": "debate_round",
            "synthesize": "synthesize",
        },
    )
    graph.add_edge("synthesize", "finalize")
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
    Entry point for running research.

    Returns a ResearchState with analyses, debate rounds, and a recommendation.
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
