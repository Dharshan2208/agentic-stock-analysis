"""Tests for the News Intelligence Agent."""

import pytest
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage

from agents.news_intelligence_agent import NewsIntelligenceAgent
from models import ResearchState
from services import set_news_provider
from services.base import NewsProvider, NewsArticle


# ── Mock LLM ──

class MockLLM(BaseLanguageModel):
    """Minimal mock LLM returning a canned response."""

    last_messages: list | None = None

    def __init__(self, response: str = "Recent news is broadly positive with earnings growth and strong product momentum."):
        super().__init__()
        self._response = response

    def invoke(self, messages, **kwargs):
        self.last_messages = messages
        return AIMessage(content=self._response)

    async def ainvoke(self, messages, **kwargs):
        return AIMessage(content=self._response)

    def generate(self, messages, **kwargs):
        raise NotImplementedError

    async def agenerate(self, messages, **kwargs):
        raise NotImplementedError

    def generate_prompt(self, prompts, **kwargs):
        raise NotImplementedError

    async def agenerate_prompt(self, prompts, **kwargs):
        raise NotImplementedError

    def get_num_tokens(self, text: str) -> int:
        return len(text.split())

    @property
    def _llm_type(self) -> str:
        return "mock"


# ── Mock News Providers ──

class MockPositiveNews(NewsProvider):
    """Returns predominantly positive articles."""

    def get_news(
        self, symbol: str, recency: str = "past 24 hours", max_articles: int = 7,
    ) -> list[NewsArticle]:
        return [
            NewsArticle(
                title="AAPL Reports Record Q3 Earnings, EPS Beats Estimates by 15%",
                source="Bloomberg", date="2026-07-10", url="https://example.com/1",
                snippet="Apple reported quarterly earnings of $1.52 per share...",
                full_text="Apple reported record earnings...",
            ),
            NewsArticle(
                title="Apple Announces New AI Features at Developers Conference",
                source="TechCrunch", date="2026-07-09", url="https://example.com/2",
                snippet="Apple unveiled new AI capabilities...",
                full_text="Apple's AI strategy includes...",
            ),
            NewsArticle(
                title="Analysts Raise AAPL Price Target to $280 on Strong Outlook",
                source="Reuters", date="2026-07-08", url="https://example.com/3",
                snippet="Several analysts raised their price targets...",
                full_text="Analysts are increasingly bullish on Apple...",
            ),
            NewsArticle(
                title="Apple Services Revenue Hits All-Time High",
                source="CNBC", date="2026-07-07", url="https://example.com/4",
                snippet="Apple's services segment grew 18% YoY...",
                full_text="Services revenue continues to be a growth driver...",
            ),
        ]


class MockNegativeNews(NewsProvider):
    """Returns predominantly negative articles."""

    def get_news(
        self, symbol: str, recency: str = "past 24 hours", max_articles: int = 7,
    ) -> list[NewsArticle]:
        return [
            NewsArticle(
                title="TSLA Shares Drop 8% After Weak Delivery Numbers",
                source="Bloomberg", date="2026-07-10", url="https://example.com/1",
                snippet="Tesla reported disappointing delivery figures...",
                full_text="Tesla deliveries fell short...",
            ),
            NewsArticle(
                title="Analyst Downgrades TSLA, Cuts Price Target to $180",
                source="Reuters", date="2026-07-09", url="https://example.com/2",
                snippet="A leading analyst downgraded Tesla...",
                full_text="The analyst cited weakening demand...",
            ),
            NewsArticle(
                title="DOJ Investigates Tesla Over Self-Driving Claims",
                source="WSJ", date="2026-07-08", url="https://example.com/3",
                snippet="The Department of Justice has opened an investigation...",
                full_text="The investigation focuses on...",
            ),
        ]


class MockMixedNews(NewsProvider):
    """Returns a mix of positive and negative articles."""

    def get_news(
        self, symbol: str, recency: str = "past 24 hours", max_articles: int = 7,
    ) -> list[NewsArticle]:
        return [
            NewsArticle(
                title="MSFT Cloud Revenue Surges 22%, Beating Estimates",
                source="Bloomberg", date="2026-07-10", url="https://example.com/1",
                snippet="Azure revenue growth accelerated...",
                full_text="Microsoft's cloud business continues to impress...",
            ),
            NewsArticle(
                title="Microsoft Faces EU Antitrust Probe Over Teams Bundling",
                source="FT", date="2026-07-09", url="https://example.com/2",
                snippet="The European Union opened a formal investigation...",
                full_text="EU regulators are investigating...",
            ),
            NewsArticle(
                title="MSFT Stock Downgraded on Valuation Concerns",
                source="Barron's", date="2026-07-08", url="https://example.com/3",
                snippet="A Wall Street analyst downgraded Microsoft...",
                full_text="The downgrade reflects valuation concerns...",
            ),
        ]


class MockEmptyNews(NewsProvider):
    """Returns no articles."""

    def get_news(
        self, symbol: str, recency: str = "past 24 hours", max_articles: int = 7,
    ) -> list[NewsArticle]:
        return []


class MockNewsNoText(NewsProvider):
    """Articles without full text (only titles/snippets)."""

    def get_news(
        self, symbol: str, recency: str = "past 24 hours", max_articles: int = 7,
    ) -> list[NewsArticle]:
        return [
            NewsArticle(
                title="AAPL Stock Rises on Positive Sentiment",
                source="Yahoo Finance", date="2026-07-10", url="https://example.com/1",
                snippet="Apple shares moved higher...",
                full_text=None,
            ),
            NewsArticle(
                title="Apple Watch Sales Remain Strong",
                source="CNBC", date="2026-07-09", url="https://example.com/2",
                snippet="Wearables segment continues to grow...",
                full_text=None,
            ),
        ]


# ── Fixtures ──

@pytest.fixture(autouse=True)
def reset_news_provider():
    """Ensure clean state before each test."""
    yield
    # No cleanup needed - each test sets its own


@pytest.fixture
def mock_llm():
    return MockLLM()


@pytest.fixture
def agent(mock_llm):
    return NewsIntelligenceAgent(llm=mock_llm)


@pytest.fixture
def state():
    return ResearchState(user_query="Analyse news", symbol="AAPL", sector="Technology")


# ── Tests ──

class TestNewsIntelligenceAgent:

    def test_agent_name(self, agent):
        assert agent.name == "news_intelligence_agent"
        assert "news" in agent.description.lower()

    def test_positive_news_produces_bullish_signals(self, mock_llm, state):
        set_news_provider(MockPositiveNews())
        agent = NewsIntelligenceAgent(llm=mock_llm)
        result = agent.run(state)
        analysis = result.analyses["news_intelligence_agent"]
        assert analysis.symbol == "AAPL"
        assert 0.0 <= analysis.agent_confidence <= 1.0
        assert len(analysis.signals) >= 3

        signal_names = [s.name for s in analysis.signals]
        assert "positive_sentiment" in signal_names, f"Got {signal_names}"
        assert "earnings_news" in signal_names, f"Got {signal_names}"

    def test_negative_news_produces_bearish_signals(self, mock_llm):
        set_news_provider(MockNegativeNews())
        agent = NewsIntelligenceAgent(llm=mock_llm)
        state = ResearchState(user_query="test", symbol="TSLA")
        result = agent.run(state)
        analysis = result.analyses["news_intelligence_agent"]
        signal_names = [s.name for s in analysis.signals]

        assert "negative_sentiment" in signal_names, f"Got {signal_names}"
        assert "analyst_downgrade" in signal_names, f"Got {signal_names}"
        assert "regulatory_risk" in signal_names, f"Got {signal_names}"

    def test_mixed_news_produces_mixed_sentiment(self, mock_llm):
        set_news_provider(MockMixedNews())
        agent = NewsIntelligenceAgent(llm=mock_llm)
        state = ResearchState(user_query="test", symbol="MSFT")
        result = agent.run(state)
        analysis = result.analyses["news_intelligence_agent"]
        signal_names = [s.name for s in analysis.signals]

        assert "mixed_sentiment" in signal_names, f"Got {signal_names}"

    def test_empty_news_graceful_degradation(self, mock_llm, state):
        set_news_provider(MockEmptyNews())
        agent = NewsIntelligenceAgent(llm=mock_llm)
        result = agent.run(state)
        analysis = result.analyses["news_intelligence_agent"]
        assert analysis.agent_confidence == 0.0
        assert "no" in analysis.summary.lower() or "no" in analysis.reasoning.lower()

    def test_articles_without_full_text(self, mock_llm, state):
        """Should still work with just titles/snippets."""
        set_news_provider(MockNewsNoText())
        agent = NewsIntelligenceAgent(llm=mock_llm)
        result = agent.run(state)
        analysis = result.analyses["news_intelligence_agent"]
        assert analysis.agent_confidence > 0.0
        assert len(analysis.signals) >= 1

    def test_tool_calls_recorded(self, mock_llm, state):
        set_news_provider(MockPositiveNews())
        agent = NewsIntelligenceAgent(llm=mock_llm)
        result = agent.run(state)
        analysis = result.analyses["news_intelligence_agent"]
        assert len(analysis.tool_calls) >= 1

    def test_execution_order_recorded(self, mock_llm, state):
        set_news_provider(MockPositiveNews())
        agent = NewsIntelligenceAgent(llm=mock_llm)
        result = agent.run(state)
        assert "news_intelligence_agent" in result.agent_execution_order

    def test_llm_receives_articles(self, mock_llm, state):
        set_news_provider(MockPositiveNews())
        agent = NewsIntelligenceAgent(llm=mock_llm)
        agent.run(state)
        assert mock_llm.last_messages is not None
        user_msg = mock_llm.last_messages[-1]["content"]
        assert "AAPL" in user_msg
        assert "Article 1" in user_msg

    def test_metadata_set(self, mock_llm, state):
        set_news_provider(MockPositiveNews())
        agent = NewsIntelligenceAgent(llm=mock_llm)
        result = agent.run(state)
        meta = result.analyses["news_intelligence_agent"].metadata
        assert meta["article_count"] >= 3
        assert "estimated_sentiment" in meta
        assert "signal_names" in meta

    def test_system_prompt_focused(self, agent):
        prompt = agent.system_prompt()
        assert "sentiment" in prompt.lower()
        assert "news" in prompt.lower()
        assert "events" in prompt.lower()

    def test_custom_name(self, mock_llm):
        custom = NewsIntelligenceAgent(llm=mock_llm, name="news_analyst")
        assert custom.name == "news_analyst"

    def test_works_with_registry(self, mock_llm, state):
        from agents import registry
        set_news_provider(MockPositiveNews())
        agent = NewsIntelligenceAgent(llm=mock_llm)
        registry.clear()
        registry.register(agent)
        result = registry.run_phase(state)
        assert "news_intelligence_agent" in result.analyses


class TestNewsSignalExtraction:
    """Direct tests for signal extraction logic (no LLM needed)."""

    def test_earnings_detection(self, mock_llm):
        set_news_provider(MockPositiveNews())
        agent = NewsIntelligenceAgent(llm=mock_llm)
        state = ResearchState(user_query="test", symbol="TEST")
        result = agent.run(state)
        assert any(s.name == "earnings_news" for s in result.analyses["news_intelligence_agent"].signals)

    def test_analyst_upgrade_detection(self, mock_llm):
        set_news_provider(MockPositiveNews())
        agent = NewsIntelligenceAgent(llm=mock_llm)
        state = ResearchState(user_query="test", symbol="TEST")
        result = agent.run(state)
        assert any(s.name == "analyst_upgrade" for s in result.analyses["news_intelligence_agent"].signals)

    def test_regulatory_detection(self, mock_llm):
        set_news_provider(MockNegativeNews())
        agent = NewsIntelligenceAgent(llm=mock_llm)
        state = ResearchState(user_query="test", symbol="TSLA")
        result = agent.run(state)
        assert any(s.name == "regulatory_risk" for s in result.analyses["news_intelligence_agent"].signals)

    def test_product_news_detection(self, mock_llm):
        """'Announces' keyword should trigger product_news signal."""
        set_news_provider(MockPositiveNews())
        agent = NewsIntelligenceAgent(llm=mock_llm)
        state = ResearchState(user_query="test", symbol="TEST")
        result = agent.run(state)
        assert any(s.name == "product_news" for s in result.analyses["news_intelligence_agent"].signals)

    def test_high_news_volume_signal(self, mock_llm):
        """4+ articles should trigger high_news_volume."""
        set_news_provider(MockPositiveNews())  # has 4 articles
        agent = NewsIntelligenceAgent(llm=mock_llm)
        state = ResearchState(user_query="test", symbol="TEST")
        result = agent.run(state)
        assert any(s.name == "high_news_volume" for s in result.analyses["news_intelligence_agent"].signals)
