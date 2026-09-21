"""Wire contract for agent events: every payload must be JSON and typed."""

from __future__ import annotations

import json
from typing import Any

from jet.core.events import (
    ApprovalRequested,
    AssistantDelta,
    AssistantMessage,
    AssistantThinking,
    ChunkAdded,
    ContextBuilt,
    Notice,
    PlanReady,
    StepStarted,
    ToolFinished,
    ToolProposed,
    TurnFinished,
    TurnStarted,
    Verified,
)
from jet.core.types import (
    ApprovalDecision,
    ChunkKind,
    Danger,
    LLMResponse,
    PolicyVerdict,
    ToolCall,
    ToolOutcome,
    ToolSpec,
    TurnResult,
    Usage,
    Verification,
)
from jet.providers.mock import MockTurn
from jet.providers.openai_llm import OpenAICompatibleLLM
from jet.server.serialize import serialize_event


def _json_roundtrip(payload: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(json.dumps(payload, ensure_ascii=False))
    return result


def test_turn_started() -> None:
    payload = serialize_event(TurnStarted(task="do it"))
    assert payload == {"type": "turn_started", "task": "do it"}


def test_step_and_plan_and_notice() -> None:
    assert serialize_event(StepStarted(step=2))["type"] == "step_started"  # type: ignore[index]
    plan = serialize_event(PlanReady(goal="g", subgoals=["a", "b"]))
    assert plan is not None and plan["subgoals"] == ["a", "b"]
    notice = serialize_event(Notice(level="warning", text="careful"))
    assert notice is not None and notice["level"] == "warning"


def test_context_built_shape() -> None:
    event = ContextBuilt(
        messages=4,
        tokens=1234,
        hidden_chunks=2,
        views=[{"chunk_id": "c1", "kind": "tool_result", "level": "full", "tokens": 10}],
    )
    payload = serialize_event(event)
    assert payload is not None
    assert _json_roundtrip(payload)["views"][0]["level"] == "full"


def test_streaming_events() -> None:
    assert serialize_event(AssistantDelta(text="hi"))["type"] == "assistant_delta"  # type: ignore[index]
    thinking = serialize_event(AssistantThinking(text="hmm"))
    assert thinking is not None and thinking["text"] == "hmm"


def test_assistant_message_carries_usage() -> None:
    response = LLMResponse(text="done", tool_calls=[], usage=Usage(10, 4), model="m", finish_reason="stop")
    payload = serialize_event(AssistantMessage(text="done", response=response))
    assert payload is not None
    assert payload["usage"] == {"input_tokens": 10, "output_tokens": 4}


def test_tool_events() -> None:
    spec = ToolSpec(
        name="run_command",
        description="run",
        parameters={"type": "object", "properties": {}},
        read_only=False,
        danger=Danger.DANGEROUS,
    )
    proposed = serialize_event(ToolProposed(call=ToolCall("c1", "run_command", {"command": "ls"}), spec=spec))
    assert proposed is not None and proposed["danger"] == "dangerous"
    outcome = ToolOutcome(
        tool_call=ToolCall("c1", "run_command", {"command": "ls"}),
        ok=False,
        output="exit 1",
        duration_ms=12,
        error="denied",
    )
    finished = serialize_event(ToolFinished(outcome=outcome))
    assert finished is not None
    assert finished["ok"] is False and finished["error"] == "denied"


def test_chunk_added_hides_content_except_subgoals() -> None:
    from jet.core.types import Chunk

    tool_chunk = Chunk(
        id="c1", session_id="s", seq=1, kind=ChunkKind.TOOL_RESULT, content="x" * 500, created_at=0.0
    )
    payload = serialize_event(ChunkAdded(chunk=tool_chunk))
    assert payload is not None and "content" not in payload["chunk"]
    subgoal_chunk = Chunk(
        id="c2", session_id="s", seq=2, kind=ChunkKind.SUBGOAL, content="read a.txt", created_at=0.0
    )
    subgoal_payload = serialize_event(ChunkAdded(chunk=subgoal_chunk))
    assert subgoal_payload is not None
    assert subgoal_payload["chunk"]["content"] == "read a.txt"


def test_verified_and_turn_finished() -> None:
    verified = serialize_event(
        Verified(verification=Verification(satisfied=True, confidence=0.9, reason="ok", verifier="judge"))
    )
    assert verified is not None and verified["satisfied"] is True
    result = TurnResult(
        text="done",
        steps=3,
        usage=Usage(100, 20),
        verification=None,
        stopped_reason="done",
        tool_calls=2,
    )
    finished = serialize_event(TurnFinished(result=result))
    assert finished is not None
    assert finished["stopped_reason"] == "done" and finished["tool_calls"] == 2
    assert _json_roundtrip(finished)["usage"]["input_tokens"] == 100


def test_approval_requested_is_not_client_serialized() -> None:
    event = ApprovalRequested(
        call=ToolCall("c1", "write_file", {"path": "a"}),
        verdict=PolicyVerdict(decision=ApprovalDecision.ASK, reason="writes"),
    )
    assert serialize_event(event) is None


def test_unknown_event_returns_none() -> None:
    assert serialize_event(object()) is None


def test_all_mock_llm_events_serialize() -> None:
    turn = MockTurn(text="x", reasoning="r")
    assert turn.reasoning == "r"
    assert OpenAICompatibleLLM is not None
