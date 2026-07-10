"""
Serper.dev implementation of the NewsProvider interface.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from typing import Any

import requests
from newspaper import Article

from config import settings
from services.base import NewsProvider, NewsArticle


class SerperNewsProvider(NewsProvider):
    """Fetches news articles via Serper.dev API and extracts full text."""

    BASE_URL = "https://google.serper.dev/news"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or settings.serper_api_key

    def get_news(
        self,
        symbol: str,
        recency: str = "past 24 hours",
        max_articles: int = 7,
    ) -> list[NewsArticle]:
        """Fetch news for a symbol, download full text, return structured articles."""
        raw_articles = self._fetch_serper(symbol, recency, max_articles)
        if not raw_articles:
            return []

        enriched: list[NewsArticle] = []
        for item in raw_articles[:max_articles]:
            url = item.get("link")
            if not url:
                continue

            full_text = self._fetch_full_text(url)

            enriched.append(
                NewsArticle(
                    title=item.get("title", "Untitled"),
                    source=item.get("source", "Unknown"),
                    date=str(item.get("date", "")),
                    url=url,
                    snippet=(item.get("snippet", "") or "")[:300],
                    full_text=full_text,
                )
            )

            time.sleep(0.3)  # Be polite to article hosts

        return enriched

    # ── Internal ──

    def _fetch_serper(
        self,
        symbol: str,
        recency: str,
        max_articles: int,
    ) -> list[dict[str, Any]]:
        """Call Serper API and return raw article dicts."""
        if not self.api_key:
            return []

        tbs_map = {
            "past 24 hours": "qdr:d",
            "past week": "qdr:w",
            "past month": "qdr:m",
        }
        tbs = tbs_map.get(recency, "qdr:d")
        cutoff = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        payload = json.dumps(
            {
                "q": (
                    f'"{symbol}" (stock OR shares OR earnings OR analyst '
                    f'OR "price target" OR upgrade OR downgrade OR guidance) '
                    f"after:{cutoff} -pdf -site:pdf"
                ),
                "gl": "us",
                "hl": "en",
                "num": max_articles,
                "tbs": tbs,
            }
        )

        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                self.BASE_URL,
                headers=headers,
                data=payload,
                timeout=10,
            )
            response.raise_for_status()
            return response.json().get("news", [])
        except requests.RequestException:
            return []

    @staticmethod
    def _fetch_full_text(url: str, timeout: int = 15) -> str | None:
        """Download and parse full article text via newspaper3k."""
        try:
            article = Article(url, fetch_images=False, request_timeout=timeout)
            article.download()
            if article.download_state != 2:  # ArticleDownloadState.SUCCESS
                return None
            article.parse()
            text = article.text.strip()
            return text[:15000] if text else None
        except Exception:
            return None
