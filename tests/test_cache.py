"""Judgment caching: identical questions must not be paid for twice."""

from __future__ import annotations

from pathlib import Path

import pytest

from jet.cache import SqliteCache
from jet.providers.caching import CachingJudgmentProvider, judgment_key
from jet.providers.mock import MockJudgmentProvider
from jet.providers.questions import NoulAnswer, noul
from tests.conftest import noul_answer


@pytest.fixture
def cache(tmp_path: Path) -> SqliteCache:
    return SqliteCache(tmp_path / "cache.db")


async def test_second_identical_ask_is_cached(cache: SqliteCache) -> None:
    inner = MockJudgmentProvider({"q": NoulAnswer(noul=0.42)})
    provider = CachingJudgmentProvider(inner, cache)
    first = await provider.ask("state", {"q": noul("?")})
    second = await provider.ask("state", {"q": noul("?")})
    assert first.cached is False
    assert second.cached is True
    assert noul_answer(second, "q") == 0.42
    assert len(inner.calls) == 1
    assert cache.count("judgment") == 1


async def test_changed_question_is_a_cache_miss(cache: SqliteCache) -> None:
    inner = MockJudgmentProvider({"q": NoulAnswer(noul=0.42)})
    provider = CachingJudgmentProvider(inner, cache)
    await provider.ask("state", {"q": noul("Is it A?")})
    await provider.ask("state", {"q": noul("Is it B?")})
    assert len(inner.calls) == 2
    assert cache.count("judgment") == 2


async def test_changed_state_is_a_cache_miss(cache: SqliteCache) -> None:
    inner = MockJudgmentProvider({"q": NoulAnswer(noul=0.42)})
    provider = CachingJudgmentProvider(inner, cache)
    await provider.ask("state A", {"q": noul("?")})
    await provider.ask("state B", {"q": noul("?")})
    assert len(inner.calls) == 2


def test_key_changes_with_model() -> None:
    question = {"q": noul("?")}
    assert judgment_key("typesafe", "jev-latest", "s", question) != judgment_key(
        "typesafe", "jev-old", "s", question
    )
    assert judgment_key("typesafe", "jev-latest", "s", question) == judgment_key(
        "typesafe", "jev-latest", "s", question
    )
