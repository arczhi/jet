"""Context builder: budget enforcement, ordering, and message structure."""

from __future__ import annotations

import pytest

from jet.context.attention import MetaAttention
from jet.context.builder import ContextBuilder
from jet.context.store import ChunkStore
from jet.context.summarizer import TruncatingSummarizer
from jet.core.types import ChunkKind, Message
from jet.providers.judge import Judge
from tests.conftest import make_judge


def builder(judge: Judge, *, budget: int) -> ContextBuilder:
    attention = MetaAttention(
        judge,
        batch_size=10,
        summarizer=TruncatingSummarizer(),
        short_tokens=5,
        long_tokens=10,
    )
    return ContextBuilder(attention, budget_tokens=budget)


async def test_message_structure(store: ChunkStore) -> None:
    store.add(ChunkKind.TOOL_RESULT, "some tool output", source="read_file")
    plan = await builder(make_judge(default_score=3.0), budget=4000).build(
        system_prompt="SYSTEM",
        task="task",
        chunks=store.all(),
        verbatim=[Message.user("do it")],
    )
    assert plan.messages[0].role.value == "system"
    assert "Working memory" in plan.messages[1].content
    assert plan.messages[1].role.value == "user"
    assert plan.messages[-1].content == "do it"


async def test_budget_drops_the_least_relevant(store: ChunkStore) -> None:
    big = "y" * 4000
    important = store.add(ChunkKind.TOOL_RESULT, big, source="important")
    filler = store.add(ChunkKind.TOOL_RESULT, big, source="filler")
    scores = {important.id: 3.0, filler.id: 0.0}
    judge = make_judge(on_score=lambda key, question: scores[key])
    plan = await builder(judge, budget=2000).build(
        system_prompt="system",
        task="task",
        chunks=store.all(),
        verbatim=[Message.user("go")],
    )
    assert [view.chunk.id for view in plan.views] == [important.id]
    assert [chunk.id for chunk in plan.hidden] == [filler.id]
    assert plan.counts["hidden"] == 1


async def test_verbatim_is_trimmed_from_the_oldest_block(store: ChunkStore) -> None:
    """Old in-turn tool exchanges may be dropped; pairing and the goal stay."""
    from jet.core.types import ToolCall

    task = Message.user("the task")
    calls = Message.assistant("working", tool_calls=[ToolCall("c1", "read_file", {"path": "a"})])
    result = Message.tool_result("c1", "x" * 6000)
    filler = Message.user("m" * 3000)
    current = Message.user("continue")
    plan = await builder(make_judge(), budget=1400).build(
        system_prompt="system",
        task="task",
        chunks=store.all(),
        verbatim=[task, calls, result, filler, current],
    )
    kept = plan.messages
    assert kept[-1].content == "continue"
    joined = [m.content or "" for m in kept]
    # The oldest tool pair (assistant+result) is dropped as one block; the task
    # and the current message stay.
    assert calls.content not in joined and result.content not in joined
    assert "the task" in joined
    assert plan.counts["dropped_verbatim"] == 1
    # Pairing is intact: every assistant-with-calls message is followed by its results.
    for index, message in enumerate(kept):
        if message.tool_calls:
            assert index + 1 < len(kept) and kept[index + 1].role.value == "tool"


async def test_budget_too_small_is_an_explicit_error(store: ChunkStore) -> None:
    store.add(ChunkKind.TOOL_RESULT, "x" * 4000)
    with pytest.raises(ValueError, match="cannot fit the current turn"):
        await builder(make_judge(), budget=500).build(
            system_prompt="system",
            task="task",
            chunks=store.all(),
            verbatim=[Message.user("u" * 2000)],
        )


async def test_goal_is_always_rendered_into_memory(store: ChunkStore) -> None:
    store.add(ChunkKind.TOOL_RESULT, "z" * 8000)
    judge = make_judge(default_score=0.0)  # everything else hidden
    plan = await builder(judge, budget=1200).build(
        system_prompt="system",
        task="find the treasure",
        chunks=store.all(),
        verbatim=[Message.user("u" * 500)],
    )
    memory = plan.messages[1].content
    assert "### [goal · full]" in memory and "find the treasure" in memory


async def test_pinned_memory_always_included(store: ChunkStore) -> None:
    store.add(ChunkKind.MEMORY, "PROJECT RULE: run tests", pinned=True)
    store.add(ChunkKind.TOOL_RESULT, "z" * 8000)
    judge = make_judge(default_score=0.0)  # everything else judged irrelevant
    plan = await builder(judge, budget=3000).build(
        system_prompt="system",
        task="task",
        chunks=store.all(),
        verbatim=[Message.user("go")],
    )
    rendered = "\n".join(view.rendered for view in plan.views)
    assert "PROJECT RULE" in rendered


async def test_views_are_ordered_by_sequence(store: ChunkStore) -> None:
    first = store.add(ChunkKind.TOOL_RESULT, "first")
    second = store.add(ChunkKind.TOOL_RESULT, "second")
    third = store.add(ChunkKind.TOOL_RESULT, "third")
    random_scores = {first.id: 1.0, second.id: 3.0, third.id: 2.0}
    judge = make_judge(on_score=lambda key, question: random_scores[key])
    plan = await builder(judge, budget=4000).build(
        system_prompt="system",
        task="task",
        chunks=store.all(),
        verbatim=[],
    )
    assert [view.chunk.id for view in plan.views] == [first.id, second.id, third.id]
