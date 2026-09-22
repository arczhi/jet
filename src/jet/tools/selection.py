"""Judgment-based tool routing.

The system prompt always carries compact snippets of every tool, so the model
knows what exists. Full schemas are only sent for tools a fast judgment pass
selects as relevant to the current step. This keeps the context small without
hiding capabilities.
"""

from __future__ import annotations

from collections.abc import Sequence

from jet.core.types import ToolSpec
from jet.errors import JetError
from jet.providers.judge import Judge
from jet.tools.registry import ToolRegistry
from jet.tracing import Trace

RELEVANCE_LEVELS = [
    "not relevant to this step",
    "marginally useful",
    "relevant and likely needed",
    "essential for this step",
]

RELEVANCE_INSTRUCTION = (
    "How relevant is `item` (a tool available to a coding agent) for making progress on `task`? "
    "Judge whether the tool would actually be used, not whether it sounds related."
)

DEFAULT_ALWAYS = ("read_file", "list_files", "glob")


class ToolSelector:
    def __init__(
        self,
        judge: Judge,
        *,
        top_k: int = 6,
        threshold: float = 1.5,
        always_include: Sequence[str] = DEFAULT_ALWAYS,
        trace: Trace | None = None,
    ):
        self.judge = judge
        self.top_k = top_k
        self.threshold = threshold
        self.always_include = tuple(always_include)
        self.trace = trace

    async def select(self, *, task: str, registry: ToolRegistry, context: str = "") -> list[ToolSpec]:
        specs = registry.specs()
        if not specs:
            return []
        items = {
            spec.name: {
                "description": spec.description,
                "read_only": spec.read_only,
                "danger": spec.danger.value,
            }
            for spec in specs
        }
        try:
            scores = await self.judge.score_many(
                state={"task": task, "context": context},
                items=items,
                levels=RELEVANCE_LEVELS,
                instruction=RELEVANCE_INSTRUCTION,
                purpose="tool_selection",
            )
        except JetError as exc:
            # Judge outage: send every schema rather than blind the model.
            if self.trace is not None:
                self.trace.event("tools.selected.degraded", reason=str(exc)[:200])
            return specs
        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        selected: list[str] = [name for name, score in ranked if score >= self.threshold][: self.top_k]
        for name in self.always_include:
            if name in scores and name not in selected and registry.get(name) is not None:
                selected.append(name)
        if not selected and ranked:
            selected = [ranked[0][0]]
        if self.trace is not None:
            self.trace.event(
                "tools.selected",
                scores={name: round(score, 3) for name, score in ranked},
                selected=selected,
            )
        return [registry.require(name).spec for name in selected]
