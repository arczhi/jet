"""End-to-end agent loop tests with deterministic providers.

These pin the behaviors the architecture promises: RLCD context assembly, tool
routing, policy gating, verification-driven retries, session persistence, and
usage accounting.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from jet.agent.session import create_agent
from jet.config import Settings
from jet.context.summarizer import TruncatingSummarizer
from jet.core.events import (
    AssistantMessage,
    ContextBuilt,
    Notice,
    StepStarted,
    ToolFinished,
    TurnFinished,
    TurnStarted,
    Verified,
)
from jet.core.types import ChunkKind, ToolCall
from jet.providers.mock import MockTurn
from tests.conftest import NullDecomposer, collect_events, make_judge, make_llm


def build(
    settings: Settings,
    *,
    llm: Any = None,
    judge: Any = None,
    emit: Callable[[Any], None] | None = None,
    approve: Callable[[ToolCall, Any], Any] | None = None,
    session_id: str | None = None,
) -> Any:
    return create_agent(
        settings,
        session_id=session_id,
        judge=judge or make_judge(),
        llm=llm or make_llm("done"),
        summarizer=TruncatingSummarizer(),
        decomposer=NullDecomposer(),
        emit=emit,
        approve=approve,
    )


async def test_simple_answer_is_verified_and_persisted(settings: Settings) -> None:
    events, sink = collect_events()
    agent = build(settings, llm=make_llm(MockTurn(text="All done.")), emit=sink)
    result = await agent.run_turn("say hi")
    assert result.stopped_reason == "done"
    assert result.verification is not None and result.verification.satisfied
    assert result.steps == 1
    kinds = [chunk.kind for chunk in agent.store.all()]
    assert ChunkKind.USER_MESSAGE in kinds
    assert ChunkKind.ASSISTANT_MESSAGE in kinds
    assert any(isinstance(event, TurnStarted) for event in events)
    assert any(isinstance(event, ContextBuilt) for event in events)
    assert any(isinstance(event, AssistantMessage) for event in events)
    assert any(isinstance(event, Verified) for event in events)
    assert isinstance(events[-1], TurnFinished)
    await agent.aclose()


async def test_tool_call_executes_and_result_enters_state(settings: Settings) -> None:
    llm = make_llm(
        MockTurn(tool_calls=[ToolCall("c1", "read_file", {"path": "a.txt"})]),
        MockTurn(text="a.txt contains alpha, beta, gamma."),
    )
    events, sink = collect_events()
    agent = build(settings, llm=llm, emit=sink)
    result = await agent.run_turn("summarize a.txt")
    assert result.stopped_reason == "done"
    assert result.tool_calls == 1
    tool_chunks = agent.store.by_kind(ChunkKind.TOOL_RESULT)
    assert len(tool_chunks) == 1
    assert "alpha" in tool_chunks[0].content
    assert tool_chunks[0].source == "read_file"
    finished = [event for event in events if isinstance(event, ToolFinished)]
    assert finished and finished[0].outcome.ok
    await agent.aclose()


async def test_denied_tool_keeps_protocol_pairing(settings: Settings, workspace: Path) -> None:
    """A denied call must never wedge a note between tool_calls and its result."""
    ask_settings = settings.model_copy(update={"approval_mode": "ask"})
    llm = make_llm(
        MockTurn(tool_calls=[ToolCall("c1", "write_file", {"path": "blocked.txt", "content": "x"})]),
        MockTurn(text="The write was denied, so I stopped."),
    )
    agent = build(ask_settings, llm=llm)
    result = await agent.run_turn("write blocked.txt")
    assert result.stopped_reason == "done"
    second_call = llm.calls[1]
    for index, message in enumerate(second_call):
        if message.tool_calls:
            following = second_call[index + 1]
            assert following.role.value == "tool", "note interleaved into tool pairing"
            assert following.tool_call_id == message.tool_calls[0].id
    await agent.aclose()


async def test_duplicate_tool_call_is_refused(settings: Settings) -> None:
    """A repeated identical call gets an explicit refusal, not another run."""
    llm = make_llm(
        MockTurn(tool_calls=[ToolCall("c1", "read_file", {"path": "a.txt"})]),
        MockTurn(tool_calls=[ToolCall("c2", "read_file", {"path": "a.txt"})]),
        MockTurn(text="done"),
    )
    agent = build(settings, llm=llm)
    result = await agent.run_turn("read a.txt")
    assert result.stopped_reason == "done"
    tool_chunks = agent.store.by_kind(ChunkKind.TOOL_RESULT)
    assert any("duplicate call refused" in chunk.content for chunk in tool_chunks)
    notes = agent.store.by_kind(ChunkKind.SYSTEM_NOTE)
    assert any(note.meta.get("kind") == "duplicate_tool_call" for note in notes)
    await agent.aclose()


async def test_unknown_tool_is_reported_not_crashed(settings: Settings) -> None:
    llm = make_llm(
        MockTurn(tool_calls=[ToolCall("c1", "teleport", {"where": "mars"})]),
        MockTurn(text="I cannot teleport."),
    )
    agent = build(settings, llm=llm)
    result = await agent.run_turn("teleport me")
    assert result.stopped_reason == "done"
    tool_chunks = agent.store.by_kind(ChunkKind.TOOL_RESULT)
    assert "unknown tool" in tool_chunks[0].content
    assert tool_chunks[0].meta["ok"] is False
    await agent.aclose()


async def test_write_is_denied_without_an_approver(settings: Settings, workspace: Path) -> None:
    ask_settings = settings.model_copy(update={"approval_mode": "ask"})
    llm = make_llm(
        MockTurn(tool_calls=[ToolCall("c1", "write_file", {"path": "out.txt", "content": "x"})]),
        MockTurn(text="The write was denied, so I stopped."),
    )
    events, sink = collect_events()
    agent = build(ask_settings, llm=llm, emit=sink)
    result = await agent.run_turn("write out.txt")
    assert result.stopped_reason == "done"
    assert not (workspace / "out.txt").exists()
    tool_chunks = agent.store.by_kind(ChunkKind.TOOL_RESULT)
    assert "denied by policy" in tool_chunks[0].content
    assert any(isinstance(event, Notice) and "step budget" in event.text for event in events) is False
    notes = agent.store.by_kind(ChunkKind.SYSTEM_NOTE)
    assert any(note.meta.get("kind") == "approval_denied" for note in notes)
    await agent.aclose()


async def test_write_runs_when_the_approver_agrees(settings: Settings, workspace: Path) -> None:
    ask_settings = settings.model_copy(update={"approval_mode": "ask"})
    approvals: list[ToolCall] = []

    async def approve(call: ToolCall, verdict: Any) -> bool:
        approvals.append(call)
        return True

    llm = make_llm(
        MockTurn(tool_calls=[ToolCall("c1", "write_file", {"path": "out.txt", "content": "x"})]),
        MockTurn(text="Wrote out.txt."),
    )
    agent = build(ask_settings, llm=llm, approve=approve)
    result = await agent.run_turn("write out.txt")
    assert result.stopped_reason == "done"
    assert (workspace / "out.txt").read_text() == "x"
    assert approvals and approvals[0].name == "write_file"
    await agent.aclose()


async def test_verification_failure_feeds_back_and_retries(settings: Settings, workspace: Path) -> None:
    verification_calls = {"n": 0}

    def on_noul(state: Any, question: Any) -> float:
        if isinstance(state, dict) and "evidence" in state:
            verification_calls["n"] += 1
            return 0.1 if verification_calls["n"] == 1 else 0.9
        return 0.9

    judge = make_judge(on_noul=on_noul)
    llm = make_llm(MockTurn(text="I think it is fine."), MockTurn(text="Fixed and verified now."))
    retry_settings = settings.model_copy(update={"max_steps": 4, "verifier_provider": "judge"})
    agent = build(retry_settings, llm=llm, judge=judge)
    result = await agent.run_turn("fix the thing")
    assert result.stopped_reason == "done"
    assert verification_calls["n"] == 2
    assert result.verification is not None and result.verification.satisfied
    notes = agent.store.by_kind(ChunkKind.SYSTEM_NOTE)
    assert any(note.meta.get("kind") == "verification_retry" for note in notes)
    assert len(agent.store.by_kind(ChunkKind.ASSISTANT_MESSAGE)) == 2
    await agent.aclose()


async def test_max_steps_stops_the_turn(settings: Settings) -> None:
    llm = make_llm(
        MockTurn(tool_calls=[ToolCall("c1", "read_file", {"path": "a.txt"})]),
        MockTurn(tool_calls=[ToolCall("c2", "read_file", {"path": "a.txt"})]),
        MockTurn(tool_calls=[ToolCall("c3", "read_file", {"path": "a.txt"})]),
    )
    limited = settings.model_copy(update={"max_steps": 2})
    events, sink = collect_events()
    agent = build(limited, llm=llm, emit=sink)
    result = await agent.run_turn("loop forever")
    assert result.stopped_reason == "max_steps"
    assert result.tool_calls == 2
    assert len([event for event in events if isinstance(event, StepStarted)]) == 2
    assert any(isinstance(event, Notice) and event.level == "warning" for event in events)
    await agent.aclose()


async def test_context_is_rebuilt_from_chunks_on_later_steps(settings: Settings) -> None:
    llm = make_llm(
        MockTurn(tool_calls=[ToolCall("c1", "read_file", {"path": "a.txt"})]),
        MockTurn(text="done reading"),
    )
    events, sink = collect_events()
    agent = build(settings, llm=llm, emit=sink)
    await agent.run_turn("read a.txt")
    contexts = [event for event in events if isinstance(event, ContextBuilt)]
    assert len(contexts) == 2
    # Fresh session: only pinned memory is pre-turn state; this turn's own
    # messages never re-enter memory and nothing was dropped.
    assert all(view["kind"] == "memory" for view in contexts[1].views)
    assert contexts[1].dropped_verbatim == 0
    await agent.aclose()


async def test_memory_covers_prior_turns_on_a_follow_up(settings: Settings) -> None:
    """A second turn sees the first turn's chunks through RLCD attention."""
    first_llm = make_llm(
        MockTurn(tool_calls=[ToolCall("c1", "read_file", {"path": "a.txt"})]), MockTurn(text="read it")
    )
    agent = build(settings, llm=first_llm)
    await agent.run_turn("read a.txt")
    await agent.aclose()

    events, sink = collect_events()
    second_llm = make_llm(
        MockTurn(tool_calls=[ToolCall("c2", "read_file", {"path": "a.txt"})]),
        MockTurn(text="done"),
    )
    agent2 = build(settings, llm=second_llm, emit=sink, session_id=agent.session_id)
    try:
        await agent2.run_turn("read a.txt again")
        contexts = [event for event in events if isinstance(event, ContextBuilt)]
        assert any(view["kind"] == "tool_result" for view in contexts[0].views)
        # Contexts within turn 2 must carry only pre-turn state (turn 1's
        # chunks) plus pinned memory; turn 2's own results travel in verbatim.
        seqs = {view["seq"] for view in contexts[1].views}
        assert 8 not in seqs, "turn 2's own tool result leaked into memory"
        assert 4 in seqs, "turn 1's tool result should be in memory"
    finally:
        await agent2.aclose()


async def test_resumed_session_reuses_prior_chunks(settings: Settings) -> None:
    session_id = "ses_resume_test"
    first_llm = make_llm(MockTurn(text="First turn done."))
    first = build(settings, llm=first_llm, session_id=session_id)
    await first.run_turn("remember the number 41")
    first_chunk_ids = {chunk.id for chunk in first.store.all()}
    await first.aclose()

    events, sink = collect_events()
    second_llm = make_llm(MockTurn(text="Second turn done."))
    second = build(settings, llm=second_llm, emit=sink, session_id=session_id)
    result = await second.run_turn("what number did I mention?")
    assert result.stopped_reason == "done"
    contexts = [event for event in events if isinstance(event, ContextBuilt)]
    visible_ids = {view["chunk_id"] for view in contexts[0].views}
    assert visible_ids & first_chunk_ids, "prior session chunks must be considered"
    assert second.store.count() > len(first_chunk_ids)
    await second.aclose()


async def test_usage_is_accounted(settings: Settings) -> None:
    from jet.core.types import Usage

    llm = make_llm(
        MockTurn(tool_calls=[ToolCall("c1", "read_file", {"path": "a.txt"})], usage=Usage(10, 2)),
        MockTurn(text="done", usage=Usage(20, 3)),
    )
    agent = build(settings, llm=llm)
    result = await agent.run_turn("read")
    assert result.usage.input_tokens == 30
    assert result.usage.output_tokens == 5
    assert agent.state.total_usage.total_tokens == 35
    await agent.aclose()


async def test_only_selected_tool_schemas_are_sent(settings: Settings) -> None:
    """The LLM must not receive every tool schema, only the judged subset."""
    llm = make_llm(MockTurn(text="done"))
    agent = build(settings, llm=llm)
    await agent.run_turn("just answer")
    sent = llm.tool_specs_seen[0]
    assert 0 < len(sent) <= settings.tool_top_k + len(("read_file", "list_files", "glob"))
    assert set(sent).issubset(
        {"read_file", "list_files", "glob", "grep", "write_file", "edit_file", "run_command"}
    )
    await agent.aclose()


async def test_trace_records_decisions(settings: Settings) -> None:
    llm = make_llm(
        MockTurn(tool_calls=[ToolCall("c1", "read_file", {"path": "a.txt"})]), MockTurn(text="done")
    )
    agent = build(settings, llm=llm)
    await agent.run_turn("read it")
    trace_text = agent.trace.path.read_text()
    assert '"event": "llm.turn"' in trace_text
    assert '"event": "judgment"' in trace_text
    assert '"event": "tools.selected"' in trace_text
    assert '"event": "policy.verdict"' in trace_text
    await agent.aclose()


async def test_empty_task_is_rejected(settings: Settings) -> None:
    agent = build(settings)
    with pytest.raises(ValueError, match="non-empty"):
        await agent.run_turn("   ")
    await agent.aclose()
