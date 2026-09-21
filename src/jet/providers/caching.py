"""Judgment result caching.

Wraps any ``JudgmentProvider``. The cache key covers provider, model, state, and
the full question set, so a changed question can never reuse a stale answer.
``JetHome`` cache lives under ``<home>/cache/judgments.db``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jet.cache import SqliteCache
from jet.core.text import stable_hash
from jet.providers.base import JudgmentProvider
from jet.providers.questions import JudgmentResult, Question


def judgment_key(provider: str, model: str, state: Any, questions: Mapping[str, Question]) -> str:
    return stable_hash(
        provider,
        model,
        state,
        {key: question.model_dump(exclude_none=True) for key, question in questions.items()},
    )


class CachingJudgmentProvider:
    def __init__(self, inner: JudgmentProvider, cache: SqliteCache):
        self.name = inner.name
        self._inner = inner
        self._cache = cache

    async def aclose(self) -> None:
        await self._inner.aclose()

    async def ask(
        self,
        state: Any,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
    ) -> JudgmentResult:
        resolved_model = model or str(getattr(self._inner, "model", "unknown"))
        key = judgment_key(self.name, resolved_model, state, questions)
        hit = self._cache.get(key)
        if hit is not None:
            result = JudgmentResult.model_validate(hit)
            result.cached = True
            return result
        result = await self._inner.ask(state, questions, model=model)
        self._cache.set(key, result.model_dump(mode="json"), namespace="judgment")
        return result


def open_judgment_cache(path: Path) -> SqliteCache:
    return SqliteCache(path)
