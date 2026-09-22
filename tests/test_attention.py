"""Meta-attention: scoring chunks, resolving levels, rendering per level."""

from __future__ import annotations

from jet.context.attention import ATTENTION_LEVELS, MetaAttention
from jet.context.store import ChunkStore
from jet.context.summarizer import TruncatingSummarizer
from jet.core.types import AttentionLevel, ChunkKind
from jet.providers.judge import Judge
from tests.conftest import make_judge


def add_chunks(store: ChunkStore, count: int, *, size: int = 40) -> list[str]:
    ids = []
    for index in range(count):
        chunk = store.add(ChunkKind.TOOL_RESULT, f"chunk {index} " + "x" * size, source=f"f{index}")
        ids.append(chunk.id)
    return ids


def attention(judge: Judge, *, batch_size: int = 24, small_pool: int = 0) -> MetaAttention:
    return MetaAttention(
        judge,
        batch_size=batch_size,
        summarizer=TruncatingSummarizer(),
        short_tokens=5,
        long_tokens=20,
        small_pool=small_pool,
    )


async def test_pinned_chunks_are_always_full(store: ChunkStore) -> None:
    pinned = store.add(ChunkKind.MEMORY, "always in context", pinned=True)
    judge = make_judge(default_score=0.0)
    result = await attention(judge).rank(task="anything", chunks=[pinned])
    assert result.views[0].level is AttentionLevel.FULL
    assert result.views[0].rendered == "always in context"
    assert result.hidden == []


async def test_score_thresholds_map_to_levels(store: ChunkStore) -> None:
    ids = add_chunks(store, 4)
    scores = {ids[0]: 3.0, ids[1]: 2.0, ids[2]: 1.0, ids[3]: 0.0}
    judge = make_judge(on_score=lambda key, question: scores[key])
    result = await attention(judge).rank(task="task", chunks=store.all(), memory_budget=30)
    levels = {view.chunk.id: view.level for view in result.views}
    assert levels[ids[0]] is AttentionLevel.FULL
    assert levels[ids[1]] is AttentionLevel.LONG
    assert levels[ids[2]] is AttentionLevel.SHORT
    assert [chunk.id for chunk in result.hidden] == [ids[3]]


async def test_summaries_are_shorter_than_full_content(store: ChunkStore) -> None:
    chunk = store.add(ChunkKind.TOOL_RESULT, "word " * 400)
    judge = make_judge(default_score=2.0)
    result = await attention(judge).rank(task="task", chunks=[chunk], memory_budget=10)
    rendered = result.views[0].rendered
    assert result.views[0].level is AttentionLevel.LONG
    assert len(rendered) < len(chunk.content)


async def test_scoring_is_batched(store: ChunkStore) -> None:
    add_chunks(store, 5)
    judge = make_judge(default_score=3.0)
    await attention(judge, batch_size=2).rank(task="task", chunks=store.all())
    provider = judge.provider
    assert len(provider.calls) == 3  # type: ignore[attr-defined]
    for _state, questions in provider.calls:  # type: ignore[attr-defined]
        assert len(questions) <= 2


async def test_small_pool_skips_scoring_and_includes_everything(store: ChunkStore) -> None:
    add_chunks(store, 5)
    judge = make_judge(default_score=0.0)
    result = await attention(judge, small_pool=8).rank(task="task", chunks=store.all())
    assert judge.provider.calls == []  # type: ignore[attr-defined]
    assert all(view.level is AttentionLevel.FULL for view in result.views)
    assert all("small_pool" in view.reason for view in result.views)
    assert result.hidden == []


async def test_small_pool_threshold_is_inclusive(store: ChunkStore) -> None:
    add_chunks(store, 3)
    judge = make_judge(default_score=0.0)
    await attention(judge, small_pool=3).rank(task="task", chunks=store.all())
    assert judge.provider.calls == []  # type: ignore[attr-defined]
    await attention(judge, small_pool=2).rank(task="task", chunks=store.all())
    assert len(judge.provider.calls) == 1  # type: ignore[attr-defined]


async def test_all_items_use_one_shared_scale(store: ChunkStore) -> None:
    add_chunks(store, 3)
    judge = make_judge(default_score=1.0)
    await attention(judge).rank(task="task", chunks=store.all())
    _state, questions = judge.provider.calls[0]  # type: ignore[attr-defined]
    for question in questions.values():
        assert question.criteria == ATTENTION_LEVELS


async def test_scores_are_reported_for_tracing(store: ChunkStore) -> None:
    ids = add_chunks(store, 2)
    judge = make_judge(default_score=2.5)
    result = await attention(judge).rank(task="task", chunks=store.all())
    assert set(result.scores) == set(ids)
    assert all(view.reason.startswith("score=") for view in result.views)
