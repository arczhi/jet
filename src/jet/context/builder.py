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
    verbatim_messages: int = 0
    verbatim_tokens: int = 0
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
        if system_tokens >= self.budget_tokens:
            raise ValueError(
                "context budget is smaller than the system prompt; raise JET_CONTEXT_BUDGET_TOKENS"
            )
        available = self.budget_tokens - system_tokens

        blocks = _message_blocks(verbatim)
        kept, dropped = _pack_from_end(blocks, available)
        verbatim_tokens = sum(estimate_tokens(message.content) for message in kept)
        memory_budget = max(available - verbatim_tokens, 0)

        result = await self.attention.rank(task=task, chunks=chunks, memory_budget=memory_budget)
        selected: list[AttentionView] = []
        used = 0
        ordered = sorted(result.views, key=lambda view: (VIEW_ORDER[view.level], view.chunk.seq))
        for view in ordered:
            cost = estimate_tokens(view.rendered) + 8
            if used + cost > available - verbatim_tokens:
                continue
            selected.append(view)
            used += cost

        selected.sort(key=lambda view: view.chunk.seq)
        memory = _render_memory(self.memory_header, selected, hidden_count=len(result.hidden), task=task)
        messages: list[Message] = []
        if system_prompt:
            messages.append(Message.system(system_prompt))
        if memory:
            messages.append(Message.user(memory))
        messages.extend(kept)
        return ContextPlan(
            messages=messages,
            views=selected,
            hidden=result.hidden,
            tokens=system_tokens + estimate_tokens(memory) + verbatim_tokens,
            verbatim_messages=len(kept),
            verbatim_tokens=verbatim_tokens,
            counts={**summarize_views(selected), "hidden": len(result.hidden), "dropped_verbatim": dropped},
        )


def _message_blocks(messages: list[Message]) -> list[list[Message]]:
    """Group messages into protocol-safe blocks.

    An assistant message carrying tool_calls opens a block that stays open
    through its tool results; the pair is kept or dropped together.
    """
    blocks: list[list[Message]] = []
    current: list[Message] = []
    for message in messages:
        if message.role.value == "tool":
            current.append(message)
            continue
        if current:
            blocks.append(current)
        current = [message]
    if current:
        blocks.append(current)
    return blocks


def _pack_from_end(blocks: list[list[Message]], budget: int) -> tuple[list[Message], int]:
    """Keep the most recent blocks that fit; the newest block is mandatory."""
    if not blocks:
        return [], 0
    last_tokens = sum(estimate_tokens(m.content) for m in blocks[-1])
    if last_tokens > budget:
        raise ValueError("context budget cannot fit the current turn; raise JET_CONTEXT_BUDGET_TOKENS")
    kept: list[Message] = []
    used = 0
    dropped = 0
    for block in reversed(blocks):
        cost = sum(estimate_tokens(m.content) for m in block)
        if not kept:
            kept = list(block)
            used = cost
            continue
        if used + cost <= budget:
            kept = list(block) + kept
            used += cost
        else:
            dropped += 1
    return kept, dropped


def _render_memory(header: str, views: list[AttentionView], *, hidden_count: int, task: str = "") -> str:
    if not views and not hidden_count and not task:
        return ""
    lines: list[str] = [f"## {header}", ""]
    if task:
        lines.append(f"### [goal · full]\n{task}")
        lines.append("")
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
