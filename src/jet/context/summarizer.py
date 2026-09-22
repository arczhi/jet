"""Summarizers: how a chunk is shrunk to ``short``/``long`` form.

An LLM summary is used when available; a deterministic truncation is the
fallback and the test double. Both are cached by content hash so a chunk is
summarized at most once per level per session.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any, Protocol

from jet.core.text import truncate_to_tokens
from jet.core.types import Message
from jet.errors import ProviderBadResponseError, ProviderError
from jet.providers.base import LLMProvider
from jet.tracing import Trace

STYLE_PROMPTS = {
    "short": "Compress to at most {n} tokens. Keep identifiers, paths, and conclusions only.",
    "long": (
        "Compress to at most {n} tokens. Keep every fact, identifier, path, number, error, "
        "and decision that could matter for later coding work. Drop filler and repetition."
    ),
}


class Summarizer(Protocol):
    async def summarize(self, text: str, *, target_tokens: int, style: str = "long") -> str: ...

    async def summarize_many(
        self, items: Mapping[str, str], *, target_tokens: int, style: str = "long"
    ) -> dict[str, str]: ...


class TruncatingSummarizer:
    """Deterministic, free, offline. Used when no LLM is configured for summaries."""

    async def summarize(self, text: str, *, target_tokens: int, style: str = "long") -> str:
        return truncate_to_tokens(text, target_tokens)

    async def summarize_many(
        self, items: Mapping[str, str], *, target_tokens: int, style: str = "long"
    ) -> dict[str, str]:
        return {key: truncate_to_tokens(text, target_tokens) for key, text in items.items()}


class LLMSummarizer:
    """Summarize with an LLM, cached per (content, style, target).

    Summaries are on the hot path of context assembly, so they belong to the
    fast endpoint — using a reasoning model here multiplies step latency.
    """

    def __init__(self, llm: LLMProvider, *, trace: Trace | None = None, max_cache_entries: int = 512):
        self.llm = llm
        self.trace = trace
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
        started = time.monotonic()
        summary = await self.llm.complete_text(messages, temperature=0.0, max_tokens=target_tokens + 64)
        if self.trace is not None:
            self.trace.event(
                "llm.summarize",
                style=style,
                chars=len(text),
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        summary = summary.strip() or truncate_to_tokens(text, target_tokens)
        if len(self._cache) >= self.max_cache_entries:
            self._cache.clear()
        self._cache[key] = summary
        return summary

    async def summarize_many(
        self, items: Mapping[str, str], *, target_tokens: int, style: str = "long"
    ) -> dict[str, str]:
        """Compress many chunks in one LLM round trip (per-chunk calls multiply latency)."""
        missing = [key for key, text in items.items() if not text.strip()]
        result: dict[str, str] = dict.fromkeys(missing, "")
        pending = {key: text for key, text in items.items() if text.strip()}
        if not pending:
            return result
        # Serve from the per-item cache first; only unserved items go on the wire.
        batch: dict[str, str] = {}
        for key, text in pending.items():
            cached = self._cache.get((text, style, target_tokens))
            if cached is not None:
                result[key] = cached
            else:
                batch[key] = text
        if batch:
            fetched = await self._compress_batch(batch, target_tokens=target_tokens, style=style)
            result.update(fetched)
        return result

    async def _compress_batch(
        self, batch: Mapping[str, str], *, target_tokens: int, style: str
    ) -> dict[str, str]:
        instruction = STYLE_PROMPTS.get(style, STYLE_PROMPTS["long"]).format(n=target_tokens)
        messages = [
            Message.system(
                "You compress working memory for a coding agent. "
                'Reply with ONLY one JSON object: {"summaries": {"<id>": "<compressed text>", ...}}. '
                "Keep every id; no prose, no markdown fences."
            ),
            Message.user(json.dumps({"instruction": instruction, "items": dict(batch)}, ensure_ascii=False)),
        ]
        started = time.monotonic()
        response = None
        try:
            response = await self.llm.complete(messages, temperature=0.0, max_tokens=4096)
            summaries = _parse_batch(response.text, ids=list(batch))
        except ProviderError as exc:
            # A summary must never fail a turn: degraded fallback, recorded.
            if self.trace is not None:
                degraded: dict[str, Any] = {"reason": str(exc)[:300]}
                if response is not None:
                    degraded["usage"] = response.usage.to_json()
                    degraded["excerpt"] = response.text[:200]
                self.trace.event("llm.summarize.degraded", **degraded)
            return {key: truncate_to_tokens(text, target_tokens) for key, text in batch.items()}
        if self.trace is not None:
            self.trace.event(
                "llm.summarize",
                style=style,
                items=len(batch),
                chars=sum(len(text) for text in batch.values()),
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        summaries = _parse_batch(response.text, ids=list(batch))
        for key, text in batch.items():
            summary = summaries.get(key) or truncate_to_tokens(text, target_tokens)
            if len(self._cache) >= self.max_cache_entries:
                self._cache.clear()
            self._cache[(text, style, target_tokens)] = summary
        return summaries


def _parse_batch(text: str, *, ids: list[str]) -> dict[str, str]:
    """Parse a batched-summary reply. Missing ids degrade to the caller's fallback."""
    try:
        payload: Any = json.loads(_extract_object(text))
        raw = payload.get("summaries")
    except (ValueError, AttributeError):
        raise ProviderBadResponseError(f"summarizer: no JSON object in reply: {text[:200]}") from None
    if not isinstance(raw, dict):
        raise ProviderBadResponseError("summarizer: JSON is missing a 'summaries' object")
    result: dict[str, str] = {}
    for key in ids:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            result[key] = value.strip()
    return result


def _extract_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON object in reply: {text[:200]}")
    return text[start : end + 1]
