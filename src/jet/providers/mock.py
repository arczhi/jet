"""Deterministic mock providers for tests and the ``--provider mock`` smoke path.

Mocks are strict: an unscripted call raises instead of inventing an answer. That
keeps tests honest about what the code under test actually asked for.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from jet.core.types import (
    LLMResponse,
    Message,
    ReasoningDelta,
    StreamDone,
    StreamEvent,
    TextDelta,
    ToolCall,
    ToolCallsReady,
    ToolSpec,
    Usage,
)
from jet.errors import ProviderBadResponseError, ProviderUnavailableError
from jet.providers.base import BaseLLMProvider
from jet.providers.questions import (
    Answer,
    JudgeUsage,
    JudgmentResult,
    NoulAnswer,
    Question,
    ScoreAnswer,
)


class MockJudgmentProvider:
    """Answers from a fixed mapping, a script, or a callable.

    ``answers`` is consulted first, then ``default_noul``/``default_score``.
    Everything unscripted raises ``ProviderBadResponseError``.
    """

    name = "mock"

    def __init__(
        self,
        answers: Mapping[str, Answer] | None = None,
        *,
        script: Iterable[Mapping[str, Answer]] | None = None,
        handler: Callable[[Any, Mapping[str, Question]], Mapping[str, Answer]] | None = None,
        default_noul: float | None = None,
        default_score: float | None = None,
    ):
        self.answers: dict[str, Answer] = dict(answers or {})
        self.script = [dict(entry) for entry in script] if script else []
        self.handler = handler
        self.default_noul = default_noul
        self.default_score = default_score
        self.calls: list[tuple[Any, dict[str, Question]]] = []
        self.model = "mock-judge"

    async def aclose(self) -> None:
        return None

    async def ask(
        self,
        state: Any,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
    ) -> JudgmentResult:
        self.calls.append((state, dict(questions)))
        if self.script:
            source = self.script.pop(0)
        elif self.handler is not None:
            source = dict(self.handler(state, questions))
        else:
            source = self.answers
        resolved: dict[str, Answer] = {}
        for key, question in questions.items():
            if key in source:
                resolved[key] = source[key]
            elif self.default_noul is not None and question.type == "noul":
                resolved[key] = NoulAnswer(noul=self.default_noul)
            elif self.default_score is not None and question.type == "score":
                resolved[key] = ScoreAnswer(score=self.default_score)
            else:
                raise ProviderBadResponseError(
                    f"mock judge has no answer for question id {key!r} ({question.type})"
                )
        return JudgmentResult(
            answers=resolved,
            model=model or self.model,
            provider=self.name,
            usage=JudgeUsage(),
            latency_ms=0.0,
        )


@dataclass
class MockTurn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None
    reasoning: str = ""
    delay_s: float = 0.0


class MockLLMProvider(BaseLLMProvider):
    """Replays scripted turns. Records every request for assertions."""

    name = "mock"
    model = "mock-llm"

    def __init__(self, turns: Sequence[MockTurn | str] | None = None):
        self.turns: list[MockTurn] = [
            MockTurn(text=turn) if isinstance(turn, str) else turn for turn in (turns or [])
        ]
        self.calls: list[list[Message]] = []
        self.tool_specs_seen: list[list[str]] = []

    async def aclose(self) -> None:
        return None

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        self.calls.append(list(messages))
        self.tool_specs_seen.append([spec.name for spec in tools])
        if not self.turns:
            raise ProviderUnavailableError("mock llm script is exhausted")
        turn = self.turns.pop(0)
        if turn.delay_s:
            import asyncio

            await asyncio.sleep(turn.delay_s)
        if turn.reasoning:
            yield ReasoningDelta(turn.reasoning)
        if turn.text:
            yield TextDelta(turn.text)
        if turn.tool_calls:
            yield ToolCallsReady(turn.tool_calls)
        yield StreamDone(
            LLMResponse(
                text=turn.text,
                tool_calls=turn.tool_calls,
                usage=turn.usage,
                model=self.model,
                finish_reason=turn.finish_reason,
                reasoning=turn.reasoning,
            )
        )
