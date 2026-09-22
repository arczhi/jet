"""Recursive decomposition: gating, proposal, dedup, and recursion decisions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from jet.context.decompose import Decomposer
from jet.context.store import ChunkStore
from jet.core.types import ChunkKind, GoalStatus
from jet.errors import ProviderBadResponseError
from jet.providers.mock import MockLLMProvider, MockTurn
from tests.conftest import make_judge, make_llm

GATE_MARK = "broken into multiple subtasks"
COVERED_MARK = "already covered"
ACTIONABLE_MARK = "executed directly"


def judge_answers(
    *,
    gate: float = 0.9,
    covered: Callable[[Any, object], float] | None = None,
    actionable: Callable[[Any, object], float] | float = 0.9,
) -> object:
    """Route the decomposer's three judgment questions by their instructions."""

    def on_noul(state: object, question: object) -> float:
        instructions = str(getattr(question, "instructions", ""))
        if GATE_MARK in instructions:
            return gate
        if COVERED_MARK in instructions:
            if callable(covered):
                return covered(state, question)
            return 0.0
        if ACTIONABLE_MARK in instructions:
            if callable(actionable):
                return actionable(state, question)
            return actionable
        return 0.0

    return make_judge(on_noul=on_noul)


def decomposer(store: ChunkStore, llm: MockLLMProvider, judge: object, **kwargs: object) -> Decomposer:
    return Decomposer(
        llm=llm,
        judge=judge,  # type: ignore[arg-type]
        store=store,
        max_depth=kwargs.pop("max_depth", 2),  # type: ignore[arg-type]
        max_total_subgoals=kwargs.pop("max_total_subgoals", 8),  # type: ignore[arg-type]
        gating=kwargs.pop("gating", True),  # type: ignore[arg-type]
    )


async def test_simple_goal_skips_the_planning_call(store: ChunkStore) -> None:
    llm = make_llm()  # unscripted: any planning call would raise
    judge = judge_answers(gate=0.0)  # single action, no plan needed
    subgoals = await decomposer(store, llm, judge).decompose("read a.txt")
    assert subgoals == []
    assert llm.calls == []


async def test_gating_can_be_disabled(store: ChunkStore) -> None:
    llm = make_llm('["do it"]')
    judge = judge_answers(gate=0.0)
    subgoals = await decomposer(store, llm, judge, gating=False).decompose("read a.txt")
    assert [subgoal.text for subgoal in subgoals] == ["do it"]


async def test_proposal_is_deduplicated(store: ChunkStore) -> None:
    llm = make_llm('["read a.txt", "read b.txt", "read a.txt"]')

    def covered(state: Any, question: object) -> float:
        return 0.9 if state.get("candidate") in state.get("existing_subgoals", []) else 0.0

    judge = judge_answers(covered=covered)
    subgoals = await decomposer(store, llm, judge).decompose("read both files")
    assert [subgoal.text for subgoal in subgoals] == ["read a.txt", "read b.txt"]
    assert all(chunk.kind is ChunkKind.SUBGOAL for chunk in store.by_kind(ChunkKind.SUBGOAL))


async def test_non_actionable_subgoal_is_recursed(store: ChunkStore) -> None:
    llm = make_llm(
        MockTurn(text='["ship the feature"]'),
        MockTurn(text='["write code", "run tests"]'),
    )

    def actionable(state: Any, question: object) -> float:
        return 0.1 if state.get("candidate") == "ship the feature" else 0.9

    judge = judge_answers(actionable=actionable)
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
    judge = judge_answers(actionable=0.0)  # nothing is ever actionable
    subgoals = await decomposer(store, llm, judge, max_depth=1).decompose("deep")
    depths = sorted({subgoal.depth for subgoal in subgoals})
    assert depths == [0, 1]


async def test_total_subgoal_cap(store: ChunkStore) -> None:
    llm = make_llm('["a", "b", "c", "d", "e", "f"]')
    judge = judge_answers()
    subgoals = await decomposer(store, llm, judge, max_total_subgoals=3).decompose("many")
    assert len(subgoals) == 3


async def test_known_subgoals_are_excluded(store: ChunkStore) -> None:
    llm = make_llm(
        MockTurn(text='["step one"]'),
        MockTurn(text='["step one", "step two"]'),
    )
    judge = judge_answers(
        covered=lambda state, question: (
            0.9
            if isinstance(state, dict) and state.get("candidate") in state.get("existing_subgoals", [])
            else 0.0
        )
    )
    engine = decomposer(store, llm, judge)
    first = await engine.decompose("the goal")
    second = await engine.decompose("the goal", known=first)
    assert [subgoal.text for subgoal in first] == ["step one"]
    assert [subgoal.text for subgoal in second] == ["step two"]


async def test_malformed_proposal_raises(store: ChunkStore) -> None:
    llm = make_llm("I will now think about the plan.")
    with pytest.raises(ProviderBadResponseError, match="JSON array"):
        await decomposer(store, llm, judge_answers()).decompose("goal")
