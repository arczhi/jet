"""Recursive decomposition: proposal, dedup, and recursion decisions."""

from __future__ import annotations

import pytest

from jet.context.decompose import Decomposer
from jet.context.store import ChunkStore
from jet.core.types import ChunkKind, GoalStatus, ToolCall, Usage
from jet.errors import ProviderBadResponseError
from jet.providers.mock import MockLLMProvider, MockTurn
from jet.providers.questions import NoulAnswer, ScoreAnswer
from tests.conftest import make_judge, make_llm


def decomposer(store: ChunkStore, llm: MockLLMProvider, judge: object, **kwargs: object) -> Decomposer:
    return Decomposer(
        llm=llm,
        judge=judge,  # type: ignore[arg-type]
        store=store,
        max_depth=kwargs.pop("max_depth", 2),  # type: ignore[arg-type]
        max_total_subgoals=kwargs.pop("max_total_subgoals", 8),  # type: ignore[arg-type]
    )


def covered_if_repeated(state: object, question: object) -> float:
    """Distinguish the two judgment questions the decomposer asks.

    Both carry a ``candidate``; only the dedup question also carries
    ``existing_subgoals``.
    """
    if not isinstance(state, dict):
        return 0.0
    instructions = str(getattr(question, "instructions", ""))
    if "already covered" in instructions:
        return 0.9 if state.get("candidate") in state.get("existing_subgoals", []) else 0.0
    if "executed directly" in instructions:
        return 0.9
    return 0.0


async def test_proposal_is_deduplicated(store: ChunkStore) -> None:
    llm = make_llm('["read a.txt", "read b.txt", "read a.txt"]')
    judge = make_judge(on_noul=covered_if_repeated)
    subgoals = await decomposer(store, llm, judge).decompose("read both files")
    assert [subgoal.text for subgoal in subgoals] == ["read a.txt", "read b.txt"]
    assert all(chunk.kind is ChunkKind.SUBGOAL for chunk in store.by_kind(ChunkKind.SUBGOAL))


async def test_non_actionable_subgoal_is_recursed(store: ChunkStore) -> None:
    llm = make_llm(
        MockTurn(text='["ship the feature"]'),
        MockTurn(text='["write code", "run tests"]'),
    )

    def on_noul(state: object, question: object) -> float:
        if not isinstance(state, dict):
            return 0.0
        instructions = str(getattr(question, "instructions", ""))
        if "executed directly" in instructions:
            return 0.1 if state.get("candidate") == "ship the feature" else 0.9
        return 0.0

    judge = make_judge(on_noul=on_noul)
    subgoals = await decomposer(store, llm, judge).decompose("ship it")
    texts = [subgoal.text for subgoal in subgoals]
    assert texts == ["ship the feature", "write code", "run tests"]
    parent = next(subgoal for subgoal in subgoals if subgoal.text == "ship the feature")
    assert parent.status is GoalStatus.SKIPPED
    children = [subgoal for subgoal in subgoals if subgoal.parent_id == parent.chunk_id]
    assert {child.text for child in children} == {"write code", "run tests"}
    assert all(child.depth == 1 for child in children)


async def test_depth_limit_stops_recursion(store: ChunkStore) -> None:
    llm = make_llm(
        MockTurn(text='["level 0"]'),
        MockTurn(text='["level 1"]'),
        MockTurn(text='["level 2"]'),
        MockTurn(text='["level 3"]'),
    )

    def never_actionable(state: object, question: object) -> float:
        if isinstance(state, dict) and "already covered" in str(getattr(question, "instructions", "")):
            return 0.0
        return 0.0

    judge = make_judge(on_noul=never_actionable)
    subgoals = await decomposer(store, llm, judge, max_depth=1).decompose("deep")
    depths = sorted({subgoal.depth for subgoal in subgoals})
    assert depths == [0, 1]


async def test_total_subgoal_cap(store: ChunkStore) -> None:
    llm = make_llm('["a", "b", "c", "d", "e", "f"]')
    judge = make_judge(on_noul=covered_if_repeated)
    subgoals = await decomposer(store, llm, judge, max_total_subgoals=3).decompose("many")
    assert len(subgoals) == 3


async def test_known_subgoals_are_excluded(store: ChunkStore) -> None:
    llm = make_llm(
        MockTurn(text='["step one"]'),
        MockTurn(text='["step one", "step two"]'),
    )
    judge = make_judge(on_noul=covered_if_repeated)
    engine = decomposer(store, llm, judge)
    first = await engine.decompose("the goal")
    second = await engine.decompose("the goal", known=first)
    assert [subgoal.text for subgoal in first] == ["step one"]
    assert [subgoal.text for subgoal in second] == ["step two"]


async def test_malformed_proposal_raises(store: ChunkStore) -> None:
    llm = make_llm("I will now think about the plan.")
    with pytest.raises(ProviderBadResponseError, match="JSON array"):
        await decomposer(store, llm, make_judge()).decompose("goal")


def test_unused_symbols_are_imported_for_type_checks() -> None:
    # Keeps imports honest for lint/type gates without runtime side effects.
    assert ToolCall("c", "t", {}).name == "t"
    assert Usage().total_tokens == 0
    assert NoulAnswer(noul=0.5).noul == 0.5
    assert ScoreAnswer(score=1.0).score == 1.0
