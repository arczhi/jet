"""Agent events to wire JSON.

One function owns the mapping so the client contract is explicit and testable.
Serialized payloads always carry a ``type`` discriminator; nothing else about the
UI is decided here.
"""

from __future__ import annotations

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
    ToolWriting,
    TurnFinished,
    TurnStarted,
    Verified,
)


def serialize_event(event: Any) -> dict[str, Any] | None:
    if isinstance(event, TurnStarted):
        return {"type": "turn_started", "task": event.task}
    if isinstance(event, StepStarted):
        return {"type": "step_started", "step": event.step}
    if isinstance(event, PlanReady):
        return {"type": "plan_ready", "goal": event.goal, "subgoals": list(event.subgoals)}
    if isinstance(event, ContextBuilt):
        return {
            "type": "context_built",
            "messages": event.messages,
            "tokens": event.tokens,
            "hidden_chunks": event.hidden_chunks,
            "dropped_verbatim": event.dropped_verbatim,
            "verbatim_messages": event.verbatim_messages,
            "verbatim_tokens": event.verbatim_tokens,
            "views": event.views,
        }
    if isinstance(event, AssistantDelta):
        return {"type": "assistant_delta", "text": event.text}
    if isinstance(event, AssistantThinking):
        return {"type": "assistant_thinking", "text": event.text}
    if isinstance(event, AssistantMessage):
        return {
            "type": "assistant_message",
            "text": event.text,
            "model": event.response.model,
            "finish_reason": event.response.finish_reason,
            "usage": event.response.usage.to_json(),
        }
    if isinstance(event, ToolProposed):
        return {
            "type": "tool_proposed",
            "name": event.call.name,
            "arguments": event.call.arguments,
            "danger": event.spec.danger.value,
            "read_only": event.spec.read_only,
        }
    if isinstance(event, ToolWriting):
        return {"type": "tool_writing", "name": event.name, "chars": event.chars}
    if isinstance(event, ToolFinished):
        outcome = event.outcome
        return {
            "type": "tool_finished",
            "name": outcome.tool_call.name,
            "arguments": outcome.tool_call.arguments,
            "ok": outcome.ok,
            "output": outcome.output,
            "error": outcome.error,
            "duration_ms": outcome.duration_ms,
        }
    if isinstance(event, ChunkAdded):
        chunk = event.chunk
        payload: dict[str, Any] = {
            "type": "chunk_added",
            "chunk": {
                "id": chunk.id,
                "seq": chunk.seq,
                "kind": chunk.kind.value,
                "source": chunk.source,
                "pinned": chunk.pinned,
                "status": chunk.status.value if chunk.status else None,
                "tokens": chunk.token_estimate,
            },
        }
        if chunk.kind.value == "subgoal":
            # Subgoal text is short and the client uses it to label plan rows.
            payload["chunk"]["content"] = chunk.content
        return payload
    if isinstance(event, Verified):
        return {
            "type": "verified",
            "satisfied": event.verification.satisfied,
            "confidence": event.verification.confidence,
            "reason": event.verification.reason,
            "verifier": event.verification.verifier,
        }
    if isinstance(event, TurnFinished):
        result = event.result
        return {
            "type": "turn_finished",
            "text": result.text,
            "steps": result.steps,
            "usage": result.usage.to_json(),
            "verification": None
            if result.verification is None
            else {
                "satisfied": result.verification.satisfied,
                "confidence": result.verification.confidence,
                "reason": result.verification.reason,
                "verifier": result.verification.verifier,
            },
            "stopped_reason": result.stopped_reason,
            "tool_calls": result.tool_calls,
        }
    if isinstance(event, Notice):
        return {"type": "notice", "level": event.level, "text": event.text}
    # Approval prompts are produced by the turn manager, which owns the approval id.
    if isinstance(event, ApprovalRequested):
        return None
    return None
