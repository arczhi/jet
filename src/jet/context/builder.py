"""Context builder: turn selected chunk views plus recent turns into a message list.

The prompt is assembled fresh for every step:

1. system prompt (identity, invariants, tool snippets)
2. working memory — the selected chunk views, most relevant first
3. verbatim recent turns — kept as real chat messages so tool-call pairing stays valid
4. the current user message

A token budget bounds the whole thing. Chunks that do not fit are dropped from the
window (never from the store) and reported as hidden, so context decisions are
inspectable instead of implicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jet.context.attention import AttentionView, MetaAttention, summarize_views
from jet.core.text import estimate_tokens
from jet.core.types import AttentionLevel, Chunk, Message

LEVEL_TITLES = {
    AttentionLevel.FULL: "full",
    AttentionLevel.LONG: "summary",
    AttentionLevel.SHORT: "note",
}

VIEW_ORDER = {AttentionLevel.FULL: 0, AttentionLevel.LONG: 1, AttentionLevel.SHORT: 2}


@dataclass
class ContextPlan:
    messages: list[Message]
    views: list[AttentionView]
    hidden: list[Chunk]
    tokens: int
    counts: dict[str, int] = field(default_factory=dict)


class ContextBuilder:
    def __init__(
        self, attention: MetaAttention, *, budget_tokens: int = 24_000, memory_header: str = "Working memory"
    ):
        self.attention = attention
        self.budget_tokens = budget_tokens
        self.memory_header = memory_header

    async def build(
        self,
        *,
        system_prompt: str,
        task: str,
        chunks: list[Chunk],
        verbatim: list[Message],
    ) -> ContextPlan:
        system_tokens = estimate_tokens(system_prompt)
        verbatim_tokens = sum(estimate_tokens(message.content) for message in verbatim)
        available = self.budget_tokens - system_tokens - verbatim_tokens
        if available <= 0:
            raise ValueError(
                "context budget is smaller than the fixed prompt and recent turns; "
                "raise JET_CONTEXT_BUDGET_TOKENS"
            )

        result = await self.attention.rank(task=task, chunks=chunks)
        selected: list[AttentionView] = []
        used = 0
        ordered = sorted(result.views, key=lambda view: (VIEW_ORDER[view.level], view.chunk.seq))
        for view in ordered:
            cost = estimate_tokens(view.rendered) + 8
            if used + cost > available:
                continue
            selected.append(view)
            used += cost

        selected.sort(key=lambda view: view.chunk.seq)
        memory = _render_memory(self.memory_header, selected, hidden_count=len(result.hidden))
        messages: list[Message] = []
        if system_prompt:
            messages.append(Message.system(system_prompt))
        if memory:
            messages.append(Message.user(memory))
        messages.extend(verbatim)
        return ContextPlan(
            messages=messages,
            views=selected,
            hidden=result.hidden,
            tokens=system_tokens + estimate_tokens(memory) + verbatim_tokens,
            counts={**summarize_views(selected), "hidden": len(result.hidden)},
        )


def _render_memory(header: str, views: list[AttentionView], *, hidden_count: int) -> str:
    if not views and not hidden_count:
        return ""
    lines: list[str] = [f"## {header}", ""]
    for view in views:
        chunk = view.chunk
        title = LEVEL_TITLES[view.level]
        label = f"[{chunk.kind.value}"
        if chunk.source:
            label += f" · {chunk.source}"
        label += f" · {title}]"
        lines.append(f"### {label} (seq {chunk.seq})")
        lines.append(view.rendered)
        lines.append("")
    if hidden_count:
        lines.append(
            f"({hidden_count} older items were judged irrelevant to this step and left out. "
            "They remain retrievable on request.)"
        )
    return "\n".join(lines).strip()


def render_view_table(views: list[AttentionView]) -> list[dict[str, Any]]:
    """Compact, JSON-safe view summary for traces and UI."""
    return [
        {
            "chunk_id": view.chunk.id,
            "kind": view.chunk.kind.value,
            "source": view.chunk.source,
            "seq": view.chunk.seq,
            "level": view.level.value,
            "score": round(view.score, 3),
            "tokens": estimate_tokens(view.rendered),
        }
        for view in views
    ]
