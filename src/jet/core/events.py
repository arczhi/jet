"""Events emitted by the agent loop.

The loop never prints. It emits typed events; the CLI/TUI (or a test) decides how
to render them. This keeps the execution path observable and the engine testable
without a terminal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from jet.core.types import (
    Chunk,
    LLMResponse,
    PolicyVerdict,
    ToolCall,
    ToolOutcome,
    ToolSpec,
    TurnResult,
    Verification,
)


@dataclass(frozen=True)
class TurnStarted:
    task: str


@dataclass(frozen=True)
class StepStarted:
    step: int


@dataclass(frozen=True)
class PlanReady:
    goal: str
    subgoals: list[str]


@dataclass(frozen=True)
class ContextBuilt:
    messages: int
    tokens: int
    hidden_chunks: int
    dropped_verbatim: int = 0
    views: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class AssistantDelta:
    text: str


@dataclass(frozen=True)
class AssistantThinking:
    """Reasoning-model output (``reasoning_content``); never fed back into context."""

    text: str


@dataclass(frozen=True)
class AssistantMessage:
    text: str
    response: LLMResponse


@dataclass(frozen=True)
class ToolProposed:
    call: ToolCall
    spec: ToolSpec


@dataclass(frozen=True)
class ApprovalRequested:
    call: ToolCall
    verdict: PolicyVerdict


@dataclass(frozen=True)
class ToolFinished:
    outcome: ToolOutcome


@dataclass(frozen=True)
class ChunkAdded:
    chunk: Chunk


@dataclass(frozen=True)
class Verified:
    verification: Verification


@dataclass(frozen=True)
class TurnFinished:
    result: TurnResult


@dataclass(frozen=True)
class Notice:
    level: Literal["info", "warning", "error"]
    text: str


AgentEvent = (
    TurnStarted
    | StepStarted
    | PlanReady
    | ContextBuilt
    | AssistantDelta
    | AssistantThinking
    | AssistantMessage
    | ToolProposed
    | ApprovalRequested
    | ToolFinished
    | ChunkAdded
    | Verified
    | TurnFinished
    | Notice
)
