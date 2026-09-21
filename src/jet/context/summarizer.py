"""Summarizers: how a chunk is shrunk to ``short``/``long`` form.

An LLM summary is used when available; a deterministic truncation is the
fallback and the test double. Both are cached by content hash so a chunk is
summarized at most once per level per session.
"""

from __future__ import annotations

from typing import Protocol

from jet.core.text import truncate_to_tokens
from jet.core.types import Message
from jet.providers.base import LLMProvider

STYLE_PROMPTS = {
    "short": "Compress to at most {n} tokens. Keep identifiers, paths, and conclusions only.",
    "long": (
        "Compress to at most {n} tokens. Keep every fact, identifier, path, number, error, "
        "and decision that could matter for later coding work. Drop filler and repetition."
    ),
}


class Summarizer(Protocol):
    async def summarize(self, text: str, *, target_tokens: int, style: str = "long") -> str: ...


class TruncatingSummarizer:
    """Deterministic, free, offline. Used when no LLM is configured for summaries."""

    async def summarize(self, text: str, *, target_tokens: int, style: str = "long") -> str:
        return truncate_to_tokens(text, target_tokens)


class LLMSummarizer:
    """Summarize with the generation LLM, cached per (content, style, target)."""

    def __init__(self, llm: LLMProvider, *, max_cache_entries: int = 512):
        self.llm = llm
        self.max_cache_entries = max_cache_entries
        self._cache: dict[tuple[str, str, int], str] = {}

    async def summarize(self, text: str, *, target_tokens: int, style: str = "long") -> str:
        if not text.strip():
            return ""
        key = (text, style, target_tokens)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        instruction = STYLE_PROMPTS.get(style, STYLE_PROMPTS["long"]).format(n=target_tokens)
        messages = [
            Message.system(
                "You compress working memory for a coding agent. Output only the compressed text."
            ),
            Message.user(f"{instruction}\n\n---\n{text}"),
        ]
        summary = await self.llm.complete_text(messages, temperature=0.0, max_tokens=target_tokens + 64)
        summary = summary.strip() or truncate_to_tokens(text, target_tokens)
        if len(self._cache) >= self.max_cache_entries:
            self._cache.clear()
        self._cache[key] = summary
        return summary
