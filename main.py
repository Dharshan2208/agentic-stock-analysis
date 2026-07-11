"""
User-facing CLI entry point for the multi-agent research system.

Usage:
    uv run python main.py

Type a ticker symbol (e.g. AAPL) or a natural-language query
(e.g. "Analyze Microsoft") and the system runs the full
research pipeline (4 analyst agents → debate → synthesis) and
prints a formatted recommendation.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime

from colorama import Fore, Style, init as colorama_init
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from config import settings
from graphs.research_graph import run_research
from models import Recommendation, ResearchState

load_dotenv()
colorama_init(autoreset=True)


# Ticker extraction


def extract_symbol(user_input: str) -> str | None:
    """Extract a ticker symbol from user input.

    Strategies tried in order:
      1. Direct match — input is 1–5 uppercase letters (e.g. ``"AAPL"``).
      2. Bracketed ticker — e.g. ``"Apple (AAPL)"``.
      3. "Analyze X" pattern.
      4. Word-boundary uppercase token 2–5 chars long.
    Returns ``None`` when no symbol can be extracted.
    """
    cleaned = user_input.strip()

    # 1. Direct ticker — all-caps, 1-5 letters
    if re.fullmatch(r"[A-Z]{1,5}", cleaned.upper()):
        return cleaned.upper()

    # 2. Bracketed ticker: "Apple (AAPL)"
    m = re.search(r"\(([A-Za-z]{1,5})\)", cleaned)
    if m:
        return m.group(1).upper()

    # 3. "Analyze / research / check X"
    m = re.search(
        r"(?:analyze|research|check|about)\s+([A-Za-z]{1,5})\b",
        cleaned,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()

    # 4. Any uppercase word 2-5 chars (first match)
    m = re.search(r"\b([A-Z]{2,5})\b", cleaned)
    if m:
        return m.group(1).upper()

    return None


# LLM initialisation


def _build_llm() -> ChatGoogleGenerativeAI:
    """Create the Gemini LLM instance used by all agents."""
    if not settings.has_google_api_key:
        print(
            Fore.RED
            + "ERROR: GOOGLE_API_KEY not set. "
            + "Create a .env file with:\n\n"
            + "    GOOGLE_API_KEY=your-key-here\n"
            + Style.RESET_ALL
        )
        sys.exit(1)

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=0.23,
    )


# Output formatting


def _format_recommendation(rec: Recommendation) -> str:
    """Return a colour-formatted investment recommendation string."""

    # Colour helpers
    def _sentiment_colour(s: str) -> str:
        return {
            "bullish": Fore.GREEN,
            "bearish": Fore.RED,
            "neutral": Fore.YELLOW,
        }.get(s, Fore.WHITE)

    def _risk_colour(r: str) -> str:
        return {
            "low": Fore.GREEN,
            "medium": Fore.YELLOW,
            "high": Fore.RED,
        }.get(r, Fore.WHITE)

    lines: list[str] = []

    # Header
    lines.append("")
    lines.append(f"{Fore.CYAN}{'=' * 58}{Style.RESET_ALL}")
    lines.append(f"{Fore.CYAN}   INVESTMENT RECOMMENDATION{Style.RESET_ALL}")
    lines.append(f"{Fore.CYAN}{'=' * 58}{Style.RESET_ALL}")

    # Symbol & sentiment
    sentiment_colour = _sentiment_colour(rec.sentiment)
    lines.append(
        f"  {Fore.WHITE}Symbol:{Style.RESET_ALL}       "
        f"{Fore.YELLOW}{rec.symbol}{Style.RESET_ALL}"
    )
    lines.append(
        f"  {Fore.WHITE}Sentiment:{Style.RESET_ALL}     "
        f"{sentiment_colour}{rec.sentiment.upper()}{Style.RESET_ALL}"
    )
    lines.append(
        f"  {Fore.WHITE}Confidence:{Style.RESET_ALL}    "
        f"{sentiment_colour}{rec.confidence:.0%}{Style.RESET_ALL}"
    )
    lines.append(
        f"  {Fore.WHITE}Risk Level:{Style.RESET_ALL}    "
        f"{_risk_colour(rec.risk_level)}{rec.risk_level.upper()}{Style.RESET_ALL}"
    )
    lines.append(
        f"  {Fore.WHITE}Timeframe:{Style.RESET_ALL}     "
        f"{rec.estimated_timeframe.replace('_', ' ')}{Style.RESET_ALL}"
    )

    # Rationale
    lines.append("")
    lines.append(f"  {Fore.CYAN}Rationale:{Style.RESET_ALL}")
    lines.append(f"  {rec.rationale}")

    # Key signals
    if rec.key_signals:
        lines.append("")
        lines.append(f"  {Fore.CYAN}Key Signals:{Style.RESET_ALL}")
        for sig in rec.key_signals:
            icon = {"bullish": "▲", "bearish": "▼", "neutral": "◆"}.get(
                sig.direction, "•"
            )
            sig_colour = _sentiment_colour(sig.direction)
            lines.append(
                f"    {sig_colour}{icon}{Style.RESET_ALL}  {sig.name}: {sig.value}"
            )

    # Conflicting signals
    if rec.conflicting_signals:
        lines.append("")
        lines.append(f"  {Fore.CYAN}Conflicts & Caveats:{Style.RESET_ALL}")
        for c in rec.conflicting_signals[:5]:
            lines.append(f"    {Fore.RED}⚠{Style.RESET_ALL}  {c}")

    # Timestamp
    lines.append("")
    lines.append(
        f"  {Fore.WHITE}Generated:{Style.RESET_ALL} "
        f"{rec.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )
    lines.append(f"{Fore.CYAN}{'=' * 58}{Style.RESET_ALL}")
    lines.append("")

    return "\n".join(lines)


def _format_error_state(state: ResearchState) -> str:
    """Print errors that occurred during graph execution."""
    if not state.errors:
        return ""

    lines: list[str] = [
        "",
        f"{Fore.RED}{'─' * 58}{Style.RESET_ALL}",
        f"{Fore.RED}   ERRORS ENCOUNTERED{Style.RESET_ALL}",
        f"{Fore.RED}{'─' * 58}{Style.RESET_ALL}",
    ]
    for err in state.errors:
        lines.append(f"  {Fore.RED}⚠{Style.RESET_ALL}  {err}")
    lines.append(f"{Fore.RED}{'─' * 58}{Style.RESET_ALL}")
    lines.append("")
    return "\n".join(lines)


# Main loop


def main() -> None:
    """Run the interactive research CLI."""
    llm = _build_llm()

    print("")
    print(f"{Fore.CYAN}╔{'═' * 56}╗{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║  Multi-Agent Stock Research System{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║  Type a ticker (e.g. AAPL) or a question{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║  Type 'exit' or 'quit' to stop.{Style.RESET_ALL}")
    print(f"{Fore.CYAN}╚{'═' * 56}╝{Style.RESET_ALL}")
    print("")

    while True:
        try:
            raw = input(f"{Fore.GREEN}You:{Style.RESET_ALL} ").strip()
            if not raw:
                continue
            if raw.lower() in ("exit", "quit", "q"):
                print(f"{Fore.YELLOW}Exiting. Goodbye!{Style.RESET_ALL}")
                break

            symbol = extract_symbol(raw)
            if not symbol:
                print(
                    f"  {Fore.RED}Could not identify a ticker symbol in your "
                    f"input.{Style.RESET_ALL}"
                )
                print(
                    f"  {Fore.YELLOW}Examples: 'AAPL', 'Analyze MSFT', "
                    f"'What about Google (GOOGL)?'{Style.RESET_ALL}"
                )
                continue

            print(f"  {Fore.YELLOW}Researching {symbol}...{Style.RESET_ALL}")

            start = datetime.now()
            state = run_research(
                user_query=raw,
                symbol=symbol,
                llm=llm,
            )
            elapsed = (datetime.now() - start).total_seconds()

            if state.errors:
                print(_format_error_state(state))

            if state.recommendation:
                print(_format_recommendation(state.recommendation))
            else:
                print(
                    f"  {Fore.RED}No recommendation was produced. "
                    f"Check errors above.{Style.RESET_ALL}"
                )

            print(f"  {Fore.WHITE}(completed in {elapsed:.1f}s){Style.RESET_ALL}")
            print("")

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Exiting. Goodbye!{Style.RESET_ALL}")
            break
        except Exception as exc:
            print(
                f"  {Fore.RED}Unexpected error: {type(exc).__name__}: "
                f"{exc}{Style.RESET_ALL}"
            )


if __name__ == "__main__":
    main()
