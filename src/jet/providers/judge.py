"""Judge: the semantic-decision service used by the agent.

All fast decisions flow through here — relevance scoring, routing, dedup,
permission advice, verification. Code owns thresholds and policy; the judgment
model only supplies probabilities. Every call is traced, and result metadata
(latency, tokens, cache hit) is returned so the loop can report cost.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from jet.core.text import stable_hash
from jet.providers.base import JudgmentProvider
from jet.providers.questions import (
    ChoiceAnswer,
    ChoiceQuestion,
    JudgmentResult,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)
from jet.tracing import Trace

Json = Any


class Judge:
    def __init__(self, provider: JudgmentProvider, *, trace: Trace | None = None):
        self.provider = provider
        self.trace = trace

    @property
    def name(self) -> str:
        return getattr(self.provider, "name", "unknown")

    async def ask(
        self,
        state: Json,
        questions: Mapping[str, Question],
        *,
        purpose: str = "judgment",
        model: str | None = None,
    ) -> JudgmentResult:
        if not questions:
            raise ValueError("judge.ask requires at least one question")
        if self.trace is None:
            return await self.provider.ask(state, questions, model=model)
        with self.trace.span(
            "judgment",
            purpose=purpose,
            provider=self.name,
            question_ids=sorted(questions),
            state_hash=stable_hash(state)[:16],
        ) as extra:
            result = await self.provider.ask(state, questions, model=model)
            extra["model"] = result.model
            extra["cached"] = result.cached
            extra["latency_ms"] = round(result.latency_ms)
            extra["input_tokens"] = result.usage.input_tokens
            extra["output_tokens"] = result.usage.output_tokens
            return result

    async def noul(
        self,
        state: Json,
        instructions: Json,
        *,
        criteria: dict[str, Json] | None = None,
        purpose: str = "noul",
    ) -> float:
        result = await self.ask(
            state, {"q": NoulQuestion(instructions=instructions, criteria=criteria)}, purpose=purpose
        )
        answer = result.answers["q"]
        if not isinstance(answer, NoulAnswer):
            raise TypeError(f"expected a noul answer, got {type(answer).__name__}")
        return answer.noul

    async def choose(
        self,
        state: Json,
        instructions: Json,
        options: Mapping[str, Json] | Sequence[str],
        *,
        purpose: str = "choice",
    ) -> str:
        criteria = dict(options) if isinstance(options, Mapping) else dict.fromkeys(options)
        result = await self.ask(
            state,
            {"q": ChoiceQuestion(instructions=instructions, criteria=criteria)},
            purpose=purpose,
        )
        answer = result.answers["q"]
        if not isinstance(answer, ChoiceAnswer):
            raise TypeError(f"expected a choice answer, got {type(answer).__name__}")
        return answer.choice

    async def score(
        self,
        state: Json,
        instructions: Json,
        levels: Sequence[Json],
        *,
        purpose: str = "score",
    ) -> float:
        result = await self.ask(
            state, {"q": ScoreQuestion(instructions=instructions, criteria=list(levels))}, purpose=purpose
        )
        answer = result.answers["q"]
        if not isinstance(answer, ScoreAnswer):
            raise TypeError(f"expected a score answer, got {type(answer).__name__}")
        return answer.score

    async def score_many(
        self,
        *,
        state: Json,
        items: Mapping[str, Json],
        levels: Sequence[Json],
        instruction: str,
        purpose: str = "score_many",
    ) -> dict[str, float]:
        """Score many items on one shared, comparable scale in a single call.

        This is the meta-attention primitive: because every item uses the same
        levels and is scored in one request, scores can be compared and ranked.
        """
        questions: dict[str, Question] = {
            key: ScoreQuestion(
                instructions={"task": instruction, "item": item},
                criteria=list(levels),
            )
            for key, item in items.items()
        }
        result = await self.ask(state, questions, purpose=purpose)
        scores: dict[str, float] = {}
        for key, answer in result.answers.items():
            if isinstance(answer, ScoreAnswer):
                scores[key] = answer.score
        missing = set(items) - set(scores)
        if missing:
            raise TypeError(f"judgment returned non-score answers for: {sorted(missing)}")
        return scores

    async def aclose(self) -> None:
        await self.provider.aclose()
