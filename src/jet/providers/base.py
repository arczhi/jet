"""Provider contracts.

Two plugin seams:

* ``JudgmentProvider`` — fast System One judgments (Jev, laya-mlx, mocks).
* ``LLMProvider`` — token generation and verification (deepseek-v4.1-flash, mocks).

Both are resolved from settings in ``jet.providers.registry``. Adding a provider
means implementing the protocol and registering a factory; nothing in the agent
loop changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from jet.core.types import LLMResponse, Message, StreamDone, StreamEvent, ToolSpec
from jet.providers.questions import JudgmentResult, Question


@runtime_checkable
class JudgmentProvider(Protocol):
    name: str

    async def ask(
        self,
        state: Any,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
    ) -> JudgmentResult: ...

    async def aclose(self) -> None: ...


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    model: str

    def stream(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]: ...

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...

    async def complete_text(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str: ...

    async def aclose(self) -> None: ...


class BaseLLMProvider(ABC):
    """Implements ``complete`` by draining ``stream`` so providers only write one path."""

    name: str = "base"
    model: str = "unknown"

    @abstractmethod
    def stream(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]: ...

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        response: LLMResponse | None = None
        async for event in self.stream(messages, tools=tools, temperature=temperature, max_tokens=max_tokens):
            if isinstance(event, StreamDone):
                response = event.response
        if response is None:
            raise RuntimeError(f"{self.name} stream ended without a response")
        return response

    async def complete_text(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        response = await self.complete(messages, temperature=temperature, max_tokens=max_tokens)
        return response.text
