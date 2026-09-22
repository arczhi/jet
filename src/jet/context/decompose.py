"""Recursive decomposition of goals into subgoals.

Code owns the recursion, the depth limits, and the dedup decision procedure; the
LLM only proposes candidate decompositions and the judgment model answers the two
questions code cannot answer deterministically:

* is this subgoal already covered by one we have?
* is this subgoal directly actionable, or does it need another level?

Every subgoal becomes a chunk with a parent link, so a session's plan is a tree
that survives restarts and can be re-entered at any node.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Protocol

from jet.context.store import ChunkStore
from jet.core.types import ChunkKind, GoalStatus, Message, SubGoal
from jet.errors import ProviderBadResponseError
from jet.providers.base import LLMProvider
from jet.providers.judge import Judge
from jet.tracing import Trace


class Planner(Protocol):
    """Contract the agent loop depends on; ``Decomposer`` is the production impl."""

    async def decompose(
        self,
        goal: str,
        *,
        parent_id: str | None = None,
        depth: int = 0,
        known: Sequence[SubGoal] | None = None,
    ) -> list[SubGoal]: ...


DECOMPOSE_SYSTEM = """You are the planning component of a coding agent.
Decompose the goal into the smallest set of concrete, independently actionable subtasks.
Reply with ONLY a JSON array of strings. No prose, no markdown fences.
Rules:
- Each subtask must name a concrete action or outcome (search, read, edit, run, verify).
- Do not repeat subtasks or split into trivia.
- Prefer 1-6 subtasks. If the goal is already a single action, return exactly one item.
- Do NOT include meta commentary, plans to plan, or requests for confirmation."""

COVERED_INSTRUCTION = (
    "Is `candidate` already covered by one of `existing_subgoals`? "
    "Covered means the same work with the same outcome, so doing both would be redundant."
)
COVERED_CRITERIA = {
    "true": "The same intent and outcome as an existing subgoal.",
    "false": "Distinct work not covered by any existing subgoal.",
}

ACTIONABLE_INSTRUCTION = (
    "Can `candidate` be executed directly now with one concrete action "
    "(one search, read, edit, run, or answer), without being decomposed further first?"
)
ACTIONABLE_CRITERIA = {
    "true": "A single concrete action completes it.",
    "false": "It bundles several steps and needs a further breakdown.",
}

NEEDS_PLAN_INSTRUCTION = (
    "Does `goal` need to be broken into multiple subtasks before a coding agent can act on it? "
    "Answer no if the goal is a single action (one search, read, edit, run, or question), "
    "even if that action needs tool calls."
)
NEEDS_PLAN_CRITERIA = {
    "true": "Multi-part work that needs sequencing before anything can be executed.",
    "false": "A single action the agent can take immediately.",
}


class Decomposer:
    def __init__(
        self,
        *,
        llm: LLMProvider,
        judge: Judge,
        store: ChunkStore,
        trace: Trace | None = None,
        max_depth: int = 2,
        max_total_subgoals: int = 16,
        gating: bool = True,
    ):
        self.llm = llm
        self.judge = judge
        self.store = store
        self.trace = trace
        self.max_depth = max_depth
        self.max_total_subgoals = max_total_subgoals
        self.gating = gating

    async def decompose(
        self,
        goal: str,
        *,
        parent_id: str | None = None,
        depth: int = 0,
        known: Sequence[SubGoal] | None = None,
    ) -> list[SubGoal]:
        if depth > self.max_depth:
            return []
        if depth == 0 and self.gating and await self._plan_not_needed(goal):
            # A fast judgment call replaces an expensive LLM planning round trip
            # for goals that are single actions anyway.
            if self.trace is not None:
                self.trace.event("decompose.skipped", goal=goal)
            return []
        context = self._context_note(parent_id)
        proposed = await self._propose(goal, context=context, known=known or [])
        created: list[SubGoal] = []
        existing = list(known or [])
        for text in proposed:
            if len(existing) + len(created) >= self.max_total_subgoals:
                break
            if await self._is_covered(text, existing + created):
                continue
            subgoal = await self._persist(text, parent_id=parent_id, depth=depth, goal=goal)
            created.append(subgoal)
            if not await self._is_actionable(text):
                children = await self.decompose(
                    text,
                    parent_id=subgoal.chunk_id,
                    depth=depth + 1,
                    known=existing + created,
                )
                if children:
                    created.extend(children)
                    subgoal.status = GoalStatus.SKIPPED
                    self.store.update_status(subgoal.chunk_id, GoalStatus.SKIPPED)
        return created

    async def _plan_not_needed(self, goal: str) -> bool:
        probability = await self.judge.noul(
            {"goal": goal},
            NEEDS_PLAN_INSTRUCTION,
            criteria=NEEDS_PLAN_CRITERIA,
            purpose="plan_gating",
        )
        return probability < 0.5

    async def _propose(self, goal: str, *, context: str, known: Sequence[SubGoal]) -> list[str]:
        payload = {
            "goal": goal,
            "context": context,
            "already_known_subgoals": [subgoal.text for subgoal in known],
        }
        messages = [
            Message.system(DECOMPOSE_SYSTEM),
            Message.user(json.dumps(payload, ensure_ascii=False)),
        ]
        response = await self.llm.complete(messages, temperature=0.0, max_tokens=600)
        if self.trace is not None:
            self.trace.event("decompose.propose", goal=goal, subgoal_count=len(response.text))
        items = _parse_string_array(response.text)
        return items

    async def _is_covered(self, candidate: str, existing: Sequence[SubGoal]) -> bool:
        if not existing:
            return False
        probability = await self.judge.noul(
            {
                "candidate": candidate,
                "existing_subgoals": [subgoal.text for subgoal in existing],
            },
            COVERED_INSTRUCTION,
            criteria=COVERED_CRITERIA,
            purpose="subgoal_dedup",
        )
        return probability >= 0.5

    async def _is_actionable(self, candidate: str) -> bool:
        probability = await self.judge.noul(
            {"candidate": candidate},
            ACTIONABLE_INSTRUCTION,
            criteria=ACTIONABLE_CRITERIA,
            purpose="subgoal_actionable",
        )
        return probability >= 0.5

    async def _persist(self, text: str, *, parent_id: str | None, depth: int, goal: str) -> SubGoal:
        chunk = self.store.add(
            ChunkKind.SUBGOAL,
            text,
            parent_id=parent_id,
            status=GoalStatus.PENDING,
            meta={"depth": depth, "goal": goal},
        )
        return SubGoal(
            id=chunk.id,
            text=text,
            parent_id=parent_id,
            depth=depth,
            status=GoalStatus.PENDING,
            chunk_id=chunk.id,
        )

    def _context_note(self, parent_id: str | None) -> str:
        if parent_id is None:
            return ""
        parent = self.store.get(parent_id)
        return parent.content if parent else ""


def _parse_string_array(text: str) -> list[str]:
    payload = text.strip()
    if payload.startswith("```"):
        payload = payload.strip("`")
        if payload.startswith("json"):
            payload = payload[4:]
    start = payload.find("[")
    end = payload.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ProviderBadResponseError(f"decomposer: LLM reply contains no JSON array: {text[:200]}")
    try:
        parsed: Any = json.loads(payload[start : end + 1])
    except ValueError as exc:
        raise ProviderBadResponseError(
            f"decomposer: malformed JSON array: {payload[start : end + 1][:200]}"
        ) from exc
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ProviderBadResponseError("decomposer: expected a JSON array of strings")
    cleaned = [item.strip() for item in parsed if item.strip()]
    if not cleaned:
        raise ProviderBadResponseError("decomposer: LLM proposed no usable subgoals")
    return cleaned
