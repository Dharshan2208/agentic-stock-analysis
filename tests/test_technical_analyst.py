"""Tests for the Technical Analyst agent."""

import pytest
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage

from agents.technical_analyst import TechnicalAnalyst
from models import ResearchState, Signal
from services import set_market_data
from services.base import MarketDataProvider, OHLCVBar
from tests.conftest import MockMarketData


# ── Mock LLM ──


class MockLLM(BaseLanguageModel):
    """Minimal mock LLM returning a canned response."""

    last_messages: list | None = None

    def __init__(
        self,
        response: str = "Trend is bullish with price above all SMAs. RSI is neutral, MACD positive.",
    ):
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


# ── Mock providers ──


class BullishTechnicalData(MarketDataProvider):
    """40 bars of uptrend data with small pullbacks — bullish bias."""

    _PRICES = [
        100.0, 101.5, 100.2, 102.8, 101.5,
        103.2, 104.0, 102.5, 105.0, 104.2,
        106.0, 105.5, 107.0, 106.2, 108.0,
        107.5, 109.0, 108.2, 110.0, 109.5,
        111.0, 110.2, 112.0, 111.5, 113.0,
        112.2, 114.0, 113.5, 115.0, 114.2,
        116.0, 115.0, 117.0, 116.0, 118.0,
        117.0, 119.0, 118.0, 120.0, 119.0,
    ]

    def get_ohlcv(
        self, symbol: str, interval: str = "1d", period: str = "6mo", bars: int = 60
    ) -> list[OHLCVBar]:
        result = []
        for i, p in enumerate(self._PRICES):
            result.append(OHLCVBar(
                time=f"2026-06-{i+1:02d}",
                open=round(p - 0.3, 2),
                high=round(p + 0.8, 2),
                low=round(p - 0.5, 2),
                close=p,
                volume=1_000_000 + i * 10_000,
            ))
        return result

    def get_current_price(self, symbol: str) -> float | None:
        return self._PRICES[-1]


class BearishTechnicalData(MarketDataProvider):
    """40 bars of downtrend data — bearish bias."""

    _PRICES = [
        120.0, 119.0, 120.5, 118.0, 119.2,
        117.0, 116.5, 118.0, 115.5, 116.2,
        114.0, 115.0, 113.5, 114.2, 112.0,
        113.0, 111.5, 112.2, 110.0, 111.0,
        109.5, 110.2, 108.0, 109.0, 107.0,
        108.0, 106.5, 107.5, 105.0, 106.0,
        104.5, 105.5, 103.0, 104.0, 102.0,
        103.0, 101.0, 102.0, 100.0, 101.0,
    ]

    def get_ohlcv(
        self, symbol: str, interval: str = "1d", period: str = "6mo", bars: int = 60
    ) -> list[OHLCVBar]:
        result = []
        for i, p in enumerate(self._PRICES):
            result.append(OHLCVBar(
                time=f"2026-06-{i+1:02d}",
                open=round(p + 0.3, 2),
                high=round(p + 0.5, 2),
                low=round(p - 0.8, 2),
                close=p,
                volume=1_500_000 + i * 5_000,
            ))
        return result

    def get_current_price(self, symbol: str) -> float | None:
        return self._PRICES[-1]


class ShortTechnicalData(MarketDataProvider):
    """Only 3 bars — insufficient for most indicators."""

    bars: list[OHLCVBar]

    def __init__(self):
        self.bars = [
            OHLCVBar(time="d1", open=100, high=102, low=99, close=101, volume=1000),
            OHLCVBar(time="d2", open=101, high=103, low=100, close=102, volume=1200),
            OHLCVBar(time="d3", open=102, high=104, low=101, close=103, volume=1100),
        ]

    def get_ohlcv(self, *args, **kwargs) -> list[OHLCVBar]:
        return self.bars

    def get_current_price(self, *args, **kwargs) -> float | None:
        return self.bars[-1].close if self.bars else None


class EmptyTechnicalData(MarketDataProvider):
    """No data returned."""

    def get_ohlcv(self, *args, **kwargs) -> list[OHLCVBar]:
        return []

    def get_current_price(self, *args, **kwargs) -> float | None:
        return None


# ── Fixtures ──


@pytest.fixture(autouse=True)
def inject_bullish_mocks():
    """Use bullish market data by default."""
    set_market_data(BullishTechnicalData())
    yield


@pytest.fixture
def mock_llm():
    return MockLLM()


@pytest.fixture
def agent(mock_llm):
    return TechnicalAnalyst(llm=mock_llm)


@pytest.fixture
def state():
    return ResearchState(user_query="Analyze AAPL", symbol="AAPL", sector="Technology")


# ══════════════════════════════════════════════════════════════════════
# Core agent tests
# ══════════════════════════════════════════════════════════════════════


class TestTechnicalAnalyst:
    """Integration tests for the full agent pipeline."""

    def test_agent_name(self, agent):
        assert agent.name == "technical_analyst"
        assert "technical" in agent.description.lower()

    def test_analyze_returns_valid_analysis(self, agent, state):
        result = agent.run(state)
        assert "technical_analyst" in result.analyses
        analysis = result.analyses["technical_analyst"]
        assert analysis.symbol == "AAPL"
        assert 0.0 <= analysis.agent_confidence <= 1.0
        assert len(analysis.summary) > 0

    def test_signals_are_extracted(self, agent, state):
        result = agent.run(state)
        analysis = result.analyses["technical_analyst"]
        # Bullish data should produce at least 3 signals
        assert len(analysis.signals) >= 3

    def test_signals_have_valid_directions(self, agent, state):
        result = agent.run(state)
        analysis = result.analyses["technical_analyst"]
        for signal in analysis.signals:
            assert signal.direction in ("bullish", "bearish", "neutral")

    def test_tool_calls_are_recorded(self, agent, state):
        result = agent.run(state)
        analysis = result.analyses["technical_analyst"]
        assert len(analysis.tool_calls) >= 1

    def test_execution_order_recorded(self, agent, state):
        result = agent.run(state)
        assert "technical_analyst" in result.agent_execution_order

    def test_llm_receives_metrics_in_prompt(self, agent, state, mock_llm):
        agent.run(state)
        assert mock_llm.last_messages is not None
        user_msg = mock_llm.last_messages[-1]["content"]
        assert "AAPL" in user_msg
        assert any(word in user_msg for word in ["SMA", "RSI", "MACD", "Volatility", "Bars:"])

    def test_agent_sets_metadata(self, agent, state):
        result = agent.run(state)
        meta = result.analyses["technical_analyst"].metadata
        assert meta["bar_count"] >= 5
        assert "indicators_computed" in meta
        assert len(meta["indicators_computed"]) > 0

    def test_system_prompt_focused(self, agent):
        prompt = agent.system_prompt()
        assert "trend" in prompt.lower()
        assert "RSI" in prompt or "momentum" in prompt.lower()
        assert "MACD" in prompt
        assert "volatility" in prompt.lower()

    def test_custom_name(self, mock_llm):
        custom = TechnicalAnalyst(llm=mock_llm, name="my_tech_analyst")
        assert custom.name == "my_tech_analyst"
        state = ResearchState(user_query="test", symbol="TEST")
        result = custom.run(state)
        assert "my_tech_analyst" in result.analyses

    def test_works_with_registry(self, agent, state):
        from agents import registry

        registry.clear()
        registry.register(agent)
        result = registry.run_phase(state)
        assert "technical_analyst" in result.analyses

    def test_short_data_graceful(self, mock_llm):
        """<5 bars should degrade gracefully."""
        set_market_data(ShortTechnicalData())
        short_agent = TechnicalAnalyst(llm=mock_llm)
        state = ResearchState(user_query="test", symbol="SHORT")
        result = short_agent.run(state)
        analysis = result.analyses["technical_analyst"]
        assert analysis.agent_confidence == 0.0
        assert "insufficient" in analysis.summary.lower()

    def test_empty_data_graceful(self, mock_llm):
        """No data should degrade gracefully."""
        set_market_data(EmptyTechnicalData())
        empty_agent = TechnicalAnalyst(llm=mock_llm)
        state = ResearchState(user_query="test", symbol="EMPTY")
        result = empty_agent.run(state)
        analysis = result.analyses["technical_analyst"]
        assert analysis.agent_confidence == 0.0

    def test_bearish_data_produces_bearish_signals(self, mock_llm):
        """Downtrend data should trigger bearish SMA trend and neutral RSI."""
        set_market_data(BearishTechnicalData())
        bearish_agent = TechnicalAnalyst(llm=mock_llm)
        state = ResearchState(user_query="test", symbol="WEAK")
        result = bearish_agent.run(state)
        analysis = result.analyses["technical_analyst"]
        signal_names = [s.name for s in analysis.signals]
        assert "sma_trend" in signal_names, f"Got {signal_names}"
        # Find the sma_trend signal and check direction
        sma_signal = next(s for s in analysis.signals if s.name == "sma_trend")
        assert sma_signal.direction == "bearish", f"SMA trend should be bearish, got {sma_signal.direction}"


# ══════════════════════════════════════════════════════════════════════
# Unit tests for individual indicators
# ══════════════════════════════════════════════════════════════════════


class TestTechnicalIndicators:
    """Direct tests for indicator computation (no LLM, no provider)."""

    def _make_bars(self, closes: list[float]) -> list[OHLCVBar]:
        return [
            OHLCVBar(
                time=f"d{i}", open=p - 0.2, high=p + 0.5, low=p - 0.3,
                close=p, volume=1_000_000,
            )
            for i, p in enumerate(closes)
        ]

    def test_sma_5(self, mock_llm):
        """Verify SMA-5 is the mean of the last 5 closes."""
        bars = self._make_bars([10, 11, 12, 13, 14, 15, 16])
        agent = TechnicalAnalyst(llm=mock_llm)
        metrics = agent._compute_indicators(bars)
        assert metrics["sma_5"] == pytest.approx(14.0, rel=0.01)  # mean(12,13,14,15,16)

    def test_sma_10(self, mock_llm):
        bars = self._make_bars(list(range(1, 16)))  # 1..15
        agent = TechnicalAnalyst(llm=mock_llm)
        metrics = agent._compute_indicators(bars)
        assert metrics["sma_10"] == pytest.approx(10.5, rel=0.01)  # mean(6..15)

    def test_sma_20(self, mock_llm):
        bars = self._make_bars(list(range(1, 26)))  # 1..25
        agent = TechnicalAnalyst(llm=mock_llm)
        metrics = agent._compute_indicators(bars)
        assert metrics["sma_20"] == pytest.approx(15.5, rel=0.01)  # mean(6..25)

    def test_price_vs_sma(self, mock_llm):
        """Price above SMA should give positive percentage."""
        bars = self._make_bars([10, 11, 12, 13, 14, 15])
        agent = TechnicalAnalyst(llm=mock_llm)
        metrics = agent._compute_indicators(bars)
        # SMA-5 = mean(11,12,13,14,15) = 13, price = 15 → (15-13)/13*100 ≈ 15.38
        assert metrics["price_vs_sma5_pct"] == pytest.approx(15.38, rel=0.1)

    def test_rsi_14(self, mock_llm):
        """Steady uptrend → RSI close to 100 (all gains, no losses)."""
        bars = self._make_bars([float(i) for i in range(1, 31)])  # 1..30, strictly increasing
        agent = TechnicalAnalyst(llm=mock_llm)
        metrics = agent._compute_indicators(bars)
        assert "rsi_14" in metrics
        assert metrics["rsi_14"] == 100.0  # all gains, no losses

    def test_rsi_with_noise(self, mock_llm):
        """Mixed up/down data → RSI in middle range."""
        prices = [
            100, 102, 99, 101, 98, 103, 97, 104, 96, 105,
            95, 106, 94, 107, 93, 108, 92, 109, 91, 110,
            90, 111, 89, 112, 88, 113, 87, 114, 86, 115,
        ]
        bars = self._make_bars(prices)
        agent = TechnicalAnalyst(llm=mock_llm)
        metrics = agent._compute_indicators(bars)
        assert "rsi_14" in metrics
        # With equal ups and downs (but last 14 have more ups), RSI should be ~50
        assert 30 <= metrics["rsi_14"] <= 80, f"RSI={metrics['rsi_14']} out of expected range"

    def test_macd_computed(self, mock_llm):
        """MACD line, signal, and histogram should be present with 40 bars."""
        bars = self._make_bars([float(i) for i in range(1, 41)])
        agent = TechnicalAnalyst(llm=mock_llm)
        metrics = agent._compute_indicators(bars)
        assert "macd_line" in metrics
        assert "macd_signal" in metrics
        assert "macd_histogram" in metrics
        assert "macd_cross" in metrics

    def test_macd_not_enough_data(self, mock_llm):
        """With <26 bars, MACD should not be computed."""
        bars = self._make_bars(list(range(1, 20)))  # 19 bars
        agent = TechnicalAnalyst(llm=mock_llm)
        metrics = agent._compute_indicators(bars)
        assert "macd_line" not in metrics
        assert "macd_signal" not in metrics

    def test_rsi_not_enough_data(self, mock_llm):
        """With <15 bars, RSI should not be computed."""
        bars = self._make_bars(list(range(1, 10)))  # 9 bars
        agent = TechnicalAnalyst(llm=mock_llm)
        metrics = agent._compute_indicators(bars)
        assert "rsi_14" not in metrics

    def test_volatility_computed(self, mock_llm):
        """Volatility should be computed with 5+ bars."""
        bars = self._make_bars([100, 101, 99, 102, 98, 103])
        agent = TechnicalAnalyst(llm=mock_llm)
        metrics = agent._compute_indicators(bars)
        assert "daily_volatility_pct" in metrics
        assert metrics["daily_volatility_pct"] > 0

    def test_ema_simple(self, mock_llm):
        """EMA of constant series equals the constant."""
        agent = TechnicalAnalyst(llm=mock_llm)
        result = agent._ema([10.0] * 20, 5)
        assert result == pytest.approx(10.0, rel=0.01)

    def test_ema_trend(self, mock_llm):
        """EMA of rising series is less than the latest value."""
        agent = TechnicalAnalyst(llm=mock_llm)
        result = agent._ema([float(i) for i in range(1, 21)], 5)
        # Latest = 20, EMA should be less
        assert result < 20.0


# ══════════════════════════════════════════════════════════════════════
# Unit tests for signal extraction
# ══════════════════════════════════════════════════════════════════════


class TestTechnicalSignals:
    """Rule-based signal extraction from computed metrics."""

    def _agent(self, mock_llm):
        return TechnicalAnalyst(llm=mock_llm)

    def test_sma_trend_bullish(self, mock_llm):
        """Price above multiple SMAs → bullish trend signal."""
        agent = self._agent(mock_llm)
        metrics = {
            "price_vs_sma5_pct": 2.0,   # above
            "price_vs_sma10_pct": 3.0,  # above
            "price_vs_sma20_pct": 4.0,  # above
        }
        signals = agent._extract_signals(metrics)
        sma_sigs = [s for s in signals if s.name == "sma_trend"]
        assert len(sma_sigs) == 1
        assert sma_sigs[0].direction == "bullish"

    def test_sma_trend_bearish(self, mock_llm):
        """Price below multiple SMAs → bearish trend signal."""
        agent = self._agent(mock_llm)
        metrics = {
            "price_vs_sma5_pct": -2.0,
            "price_vs_sma10_pct": -3.0,
            "price_vs_sma20_pct": -1.5,
        }
        signals = agent._extract_signals(metrics)
        sma_sigs = [s for s in signals if s.name == "sma_trend"]
        assert len(sma_sigs) == 1
        assert sma_sigs[0].direction == "bearish"

    def test_sma_trend_mixed(self, mock_llm):
        """Price above some, below others → neutral."""
        agent = self._agent(mock_llm)
        metrics = {
            "price_vs_sma5_pct": 1.0,    # above
            "price_vs_sma10_pct": -1.0,  # below
        }
        signals = agent._extract_signals(metrics)
        sma_sigs = [s for s in signals if s.name == "sma_trend"]
        assert len(sma_sigs) == 1
        assert sma_sigs[0].direction == "neutral"

    def test_rsi_overbought(self, mock_llm):
        """RSI ≥ 70 → bearish signal (overbought)."""
        agent = self._agent(mock_llm)
        signals = agent._extract_signals({"rsi_14": 75.0})
        names = [s.name for s in signals]
        assert "rsi_overbought" in names

    def test_rsi_oversold(self, mock_llm):
        """RSI ≤ 30 → bullish signal (oversold)."""
        agent = self._agent(mock_llm)
        signals = agent._extract_signals({"rsi_14": 25.0})
        names = [s.name for s in signals]
        assert "rsi_oversold" in names

    def test_rsi_neutral(self, mock_llm):
        """30 < RSI < 70 → neutral."""
        agent = self._agent(mock_llm)
        signals = agent._extract_signals({"rsi_14": 50.0})
        names = [s.name for s in signals]
        assert "rsi_neutral" in names

    def test_macd_bullish_cross(self, mock_llm):
        """MACD bullish cross."""
        agent = self._agent(mock_llm)
        signals = agent._extract_signals({
            "macd_cross": "bullish",
            "macd_histogram": 0.5,
        })
        names = [s.name for s in signals]
        assert "macd_cross" in names
        macd_sig = next(s for s in signals if s.name == "macd_cross")
        assert macd_sig.direction == "bullish"

    def test_macd_histogram_bullish(self, mock_llm):
        """Positive histogram → bullish momentum."""
        agent = self._agent(mock_llm)
        signals = agent._extract_signals({"macd_histogram": 0.5})
        macd_mom = [s for s in signals if s.name == "macd_momentum"]
        assert len(macd_mom) == 1
        assert macd_mom[0].direction == "bullish"

    def test_macd_histogram_bearish(self, mock_llm):
        """Negative histogram → bearish momentum."""
        agent = self._agent(mock_llm)
        signals = agent._extract_signals({"macd_histogram": -0.5})
        macd_mom = [s for s in signals if s.name == "macd_momentum"]
        assert len(macd_mom) == 1
        assert macd_mom[0].direction == "bearish"

    def test_high_volatility(self, mock_llm):
        """Daily vol > 3% → high_volatility signal."""
        agent = self._agent(mock_llm)
        signals = agent._extract_signals({"daily_volatility_pct": 4.5})
        names = [s.name for s in signals]
        assert "high_volatility" in names

    def test_low_volatility(self, mock_llm):
        """Daily vol < 0.8% → low_volatility signal."""
        agent = self._agent(mock_llm)
        signals = agent._extract_signals({"daily_volatility_pct": 0.5})
        names = [s.name for s in signals]
        assert "low_volatility" in names
