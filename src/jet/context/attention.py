"""Meta-attention: deciding how much of each chunk the next step needs.

Static context is a KV-cache artifact, not a requirement. For every step, jet
re-asks: of all known chunks, which are irrelevant, which need a line, which need
a summary, and which are essential in full? A fast judgment model answers for all
chunks in batched calls on one shared scale, so scores are comparable and the
code owns the thresholds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from jet.context.summarizer import Summarizer
from jet.core.types import AttentionLevel, Chunk, ChunkKind
from jet.providers.judge import Judge

#: Ordered levels. Index doubles as the score scale, so scores are comparable.
ATTENTION_LEVELS: list[str] = [
    "irrelevant to the task — omit entirely",
    "background only — worth at most a one-line note",
    "relevant — a summary is enough",
    "essential — the full content is needed",
]

ATTENTION_INSTRUCTION = (
    "How much of `item` does a coding agent need in its next working context to make progress "
    "on `task`? Judge only usefulness for `task`, not whether `item` is correct."
)

NON_SCORABLE_KINDS = {ChunkKind.PLAN, ChunkKind.SUBGOAL, ChunkKind.MEMORY}


@dataclass
class AttentionView:
    chunk: Chunk
    level: AttentionLevel
    score: float
    rendered: str
    reason: str = ""


@dataclass
class AttentionResult:
    views: list[AttentionView]
    hidden: list[Chunk] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)


class MetaAttention:
    def __init__(
        self,
        judge: Judge,
        *,
        batch_size: int = 24,
        full_threshold: float = 2.5,
        long_threshold: float = 1.5,
        short_threshold: float = 0.4,
        summarizer: Summarizer | None = None,
        char_limit: int = 6000,
        short_tokens: int = 60,
        long_tokens: int = 320,
    ):
        self.judge = judge
        self.batch_size = batch_size
        self.full_threshold = full_threshold
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.summarizer = summarizer
        self.char_limit = char_limit
        self.short_tokens = short_tokens
        self.long_tokens = long_tokens

    async def rank(self, *, task: str, chunks: Sequence[Chunk]) -> AttentionResult:
        """Score every candidate chunk and resolve each to a level with rendered text."""
        scorable = [c for c in chunks if c.kind not in NON_SCORABLE_KINDS]
        if scorable:
            scores = await self._score(task, scorable)
        else:
            scores = {}
        views: list[AttentionView] = []
        hidden: list[Chunk] = []
        for chunk in chunks:
            if chunk.pinned or chunk.kind in NON_SCORABLE_KINDS:
                views.append(
                    AttentionView(
                        chunk=chunk,
                        level=AttentionLevel.FULL,
                        score=3.0,
                        rendered=chunk.content,
                        reason="pinned",
                    )
                )
                continue
            score = scores.get(chunk.id, 0.0)
            level = self._level(score)
            if level is AttentionLevel.HIDE:
                hidden.append(chunk)
                continue
            views.append(
                AttentionView(
                    chunk=chunk,
                    level=level,
                    score=score,
                    rendered=await self._render(chunk, level),
                    reason=f"score={score:.2f}",
                )
            )
        return AttentionResult(views=views, hidden=hidden, scores=scores)

    def _level(self, score: float) -> AttentionLevel:
        if score >= self.full_threshold:
            return AttentionLevel.FULL
        if score >= self.long_threshold:
            return AttentionLevel.LONG
        if score >= self.short_threshold:
            return AttentionLevel.SHORT
        return AttentionLevel.HIDE

    async def _score(self, task: str, chunks: Sequence[Chunk]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            items: dict[str, dict[str, Any]] = {
                chunk.id: {
                    "kind": chunk.kind.value,
                    "source": chunk.source,
                    "content": _clip(chunk.content, self.char_limit),
                }
                for chunk in batch
            }
            scores.update(
                await self.judge.score_many(
                    state={"task": task},
                    items=items,
                    levels=ATTENTION_LEVELS,
                    instruction=ATTENTION_INSTRUCTION,
                    purpose="meta_attention",
                )
            )
        return scores

    async def _render(self, chunk: Chunk, level: AttentionLevel) -> str:
        if level is AttentionLevel.FULL:
            return chunk.content
        if self.summarizer is None:
            from jet.core.text import truncate_to_tokens

            target = self.long_tokens if level is AttentionLevel.LONG else self.short_tokens
            return truncate_to_tokens(chunk.content, target)
        style = "long" if level is AttentionLevel.LONG else "short"
        target = self.long_tokens if level is AttentionLevel.LONG else self.short_tokens
        return await self.summarizer.summarize(chunk.content, target_tokens=target, style=style)


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 40] + f"\n…[{len(text) - limit + 40} chars omitted for scoring]"


def summarize_views(views: Sequence[AttentionView]) -> Mapping[str, int]:
    counts: dict[str, int] = {level.value: 0 for level in AttentionLevel}
    for view in views:
        counts[view.level.value] += 1
    return counts
