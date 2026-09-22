"""Shared domain types: messages, tool calls, usage, chunks, plans.

These are provider-independent. Wire formats (OpenAI JSON, TypeSafe JSON) stay in
``jet.providers``; domain code never depends on a vendor shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

Json = Any


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


class Danger(StrEnum):
    SAFE = "safe"
    MODERATE = "moderate"
    DANGEROUS = "dangerous"


@dataclass(frozen=True)
class ToolSpec:
    """A callable the model may invoke.

    ``description`` is a short snippet that can always be in context; the full
    ``parameters`` schema is only sent when a judgment pass selects the tool.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    read_only: bool = True
    danger: Danger = Danger.SAFE


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    @staticmethod
    def system(content: str) -> Message:
        return Message(role=Role.SYSTEM, content=content)

    @staticmethod
    def user(content: str) -> Message:
        return Message(role=Role.USER, content=content)

    @staticmethod
    def assistant(content: str = "", tool_calls: list[ToolCall] | None = None) -> Message:
        return Message(role=Role.ASSISTANT, content=content, tool_calls=tool_calls or [])

    @staticmethod
    def tool_result(tool_call_id: str, content: str, name: str | None = None) -> Message:
        return Message(role=Role.TOOL, content=content, tool_call_id=tool_call_id, name=name)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_json(self) -> dict[str, int]:
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens}


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    model: str
    finish_reason: str | None = None
    reasoning: str = ""


# --- streaming events -------------------------------------------------------


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ReasoningDelta:
    text: str


@dataclass(frozen=True)
class ToolCallProgress:
    """Streaming tool-call build-up, so long writes show live progress."""

    index: int
    name: str
    chars: int


@dataclass(frozen=True)
class ToolCallsReady:
    calls: list[ToolCall]


@dataclass(frozen=True)
class StreamDone:
    response: LLMResponse


StreamEvent = TextDelta | ReasoningDelta | ToolCallProgress | ToolCallsReady | StreamDone


# --- RLCD context -----------------------------------------------------------


class ChunkKind(StrEnum):
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    PLAN = "plan"
    SUBGOAL = "subgoal"
    SUMMARY = "summary"
    MEMORY = "memory"
    VERIFICATION = "verification"
    SYSTEM_NOTE = "system_note"


class GoalStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Chunk:
    """One unit of session state. The unit of meta-attention scoring."""

    id: str
    session_id: str
    seq: int
    kind: ChunkKind
    content: str
    created_at: float
    parent_id: str | None = None
    source: str | None = None
    pinned: bool = False
    token_estimate: int = 0
    status: GoalStatus | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class AttentionLevel(StrEnum):
    """How much of a chunk should enter the next context window."""

    HIDE = "hide"
    SHORT = "short"
    LONG = "long"
    FULL = "full"


@dataclass
class SubGoal:
    id: str
    text: str
    parent_id: str | None
    depth: int
    status: GoalStatus
    chunk_id: str


@dataclass
class Plan:
    goal: str
    subgoals: list[SubGoal]

    def pending(self) -> list[SubGoal]:
        return [g for g in self.subgoals if g.status == GoalStatus.PENDING]


class ApprovalDecision(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class ApprovalMode(StrEnum):
    ASK = "ask"
    AUTO = "auto"
    DENY = "deny"


@dataclass
class PolicyVerdict:
    decision: ApprovalDecision
    reason: str
    rule: str | None = None
    judged: bool = False

    @property
    def allowed(self) -> bool:
        return self.decision is ApprovalDecision.ALLOW


@dataclass
class ToolOutcome:
    tool_call: ToolCall
    ok: bool
    output: str
    duration_ms: int
    error: str | None = None


@dataclass
class Verification:
    satisfied: bool
    confidence: float
    reason: str
    verifier: str
    conclusive: bool = True


@dataclass
class TurnResult:
    text: str
    steps: int
    usage: Usage
    verification: Verification | None
    stopped_reason: Literal["done", "budget", "max_steps", "error", "canceled"]
    tool_calls: int = 0
