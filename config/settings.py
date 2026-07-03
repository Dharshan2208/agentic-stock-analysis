"""
Application configuration.
All environment variables and secrets are centralized here.
Validated at import time — missing keys fail fast.
"""

from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    langchain_tracing_v2: bool = False
    langchain_api_key: str | None = None

    google_api_key: str = ""
    groq_api_key: str = ""

    gemini_model: str = "gemini-3.5-flash"

    serper_api_key: str = ""
    alpha_vantage_api_key: str = ""

    # ── Application ──
    max_debate_rounds: int = 3
    default_timeframe: str = "medium_term"
    data_dir: Path = Path("data")
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def has_google_api_key(self) -> bool:
        return bool(self.google_api_key)

    @property
    def has_groq_api_key(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def has_serper_api_key(self) -> bool:
        return bool(self.serper_api_key)


settings = Settings()
