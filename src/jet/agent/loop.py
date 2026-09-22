"""The agent loop.

One user task is one *turn*. A turn is a bounded sequence of steps:

    build context (RLCD) -> route tools (judgment) -> stream the LLM
      -> approve + execute tools -> persist chunks -> verify

Context is rebuilt from chunks on every step, so the window is always shaped by
the current goal instead of whatever happened to be said. Verification gates the
completion claim; failed verification feeds back as an explicit note and the
loop continues until the step budget runs out.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from jet.agent.prompts import (
    APPROVAL_DENIED,
    CONTEXT_NOTE_HEADER,
    VERIFICATION_RETRY,
    system_prompt,
)
from jet.agent.session import SessionState
from jet.agent.verify import Verifier
from jet.config import Settings
from jet.context.builder import ContextBuilder, render_view_table
from jet.context.decompose import Planner
from jet.context.memory import MemoryLoader
from jet.context.store import ChunkStore
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
    GoalStatus,
    LLMResponse,
    Message,
    PolicyVerdict,
    ReasoningDelta,
    StreamDone,
    SubGoal,
    TextDelta,
    ToolCall,
    ToolOutcome,
    ToolSpec,
    TurnResult,
    Usage,
    Verification,
)
from jet.errors import JetError, ProviderError
from jet.policy.permissions import PermissionPolicy
from jet.providers.base import LLMProvider
from jet.providers.judge import Judge
from jet.tools.base import ToolContext
from jet.tools.registry import ToolRegistry
from jet.tools.selection import ToolSelector
from jet.tracing import Trace, cost_usd

EventSink = Callable[[Any], None]
Approver = Callable[[ToolCall, PolicyVerdict], Awaitable[bool]]
StopReason = Literal["done", "budget", "max_steps", "error", "canceled"]

EVIDENCE_CHUNKS = 12
EVIDENCE_CHARS_PER_CHUNK = 1_200
EVIDENCE_TOTAL_CHARS = 8_000

TOOL_RESULT_MAX_CHARS = 24_000


class Agent:
    def __init__(
        self,
        *,
        settings: Settings,
        state: SessionState,
        judge: Judge,
        llm: LLMProvider,
        verifier_llm: LLMProvider | None = None,
        registry: ToolRegistry,
        selector: ToolSelector,
        policy: PermissionPolicy,
        builder: ContextBuilder,
        decomposer: Planner,
        verifier: Verifier,
        memory: MemoryLoader,
        emit: EventSink | None = None,
        approve: Approver | None = None,
        tool_context: ToolContext | None = None,
    ):
        self.settings = settings
        self.state = state
        self.judge = judge
        self.llm = llm
        self.verifier_llm = verifier_llm or llm
        self.registry = registry
        self.selector = selector
        self.policy = policy
        self.builder = builder
        self.decomposer = decomposer
        self.verifier = verifier
        self.memory = memory
        self.emit = emit or (lambda event: None)
        self.approve = approve
        self.tool_context = tool_context or ToolContext(
            workspace=settings.workspace,
            store=state.store,
            memory=memory,
        )

    # -- public API --------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self.state.session_id

    @property
    def store(self) -> ChunkStore:
        return self.state.store

    @property
    def trace(self) -> Trace:
        return self.state.trace

    async def run_turn(self, task: str) -> TurnResult:
        task = task.strip()
        if not task:
            raise ValueError("run_turn requires a non-empty task")
        self.emit(TurnStarted(task=task))
        turn_start_count = self.state.store.count()
        user_chunk = self.state.store.add(ChunkKind.USER_MESSAGE, task)
        self.emit(ChunkAdded(chunk=user_chunk))
        loaded = self.memory.load_root()
        if loaded:
            self.emit(Notice(level="info", text=f"loaded memory: {', '.join(loaded)}"))
        self.state.turn_messages = [Message.user(task)]

        subgoals = await self._plan(task)
        usage = Usage()
        final_text = ""
        verification: Verification | None = None
        stopped: StopReason = "max_steps"
        tool_count = 0
        steps = 0

        while steps < self.settings.max_steps:
            steps += 1
            self.emit(StepStarted(step=steps))
            current_goal = self._current_goal(task, subgoals)
            # Memory carries pre-turn state plus anything accumulated outside the
            # conversation (memory files, subgoals); this turn's own messages stay
            # verbatim and are never duplicated into memory.
            candidates = [
                chunk
                for chunk in self.state.store.all()
                if chunk.seq <= turn_start_count or chunk.kind in (ChunkKind.MEMORY, ChunkKind.SUBGOAL)
            ]
            plan, selected = await asyncio.gather(
                self.builder.build(
                    system_prompt=self._system_prompt(),
                    task=current_goal,
                    chunks=candidates,
                    verbatim=list(self.state.turn_messages),
                ),
                self.selector.select(task=current_goal, registry=self.registry, context=task),
            )
            self.emit(
                ContextBuilt(
                    messages=len(plan.messages),
                    tokens=plan.tokens,
                    hidden_chunks=len(plan.hidden),
                    dropped_verbatim=plan.counts.get("dropped_verbatim", 0),
                    views=render_view_table(plan.views),
                )
            )
            try:
                response = await self._stream(plan.messages, selected)
            except ProviderError as exc:
                self.emit(Notice(level="error", text=f"generation failed: {exc}"))
                stopped = "error"
                break
            usage = usage + response.usage
            self.state.total_usage = self.state.total_usage + response.usage
            self.state.total_cost_usd += cost_usd(
                response.usage.input_tokens,
                response.usage.output_tokens,
                float(getattr(self.llm, "input_price", 0.0)),
                float(getattr(self.llm, "output_price", 0.0)),
            )
            self.state.turn_messages.append(Message.assistant(response.text, tool_calls=response.tool_calls))
            self.emit(AssistantMessage(text=response.text, response=response))
            self._persist_assistant(response)

            if not response.tool_calls:
                final_text = response.text
                verification = await self.verifier.verify(
                    goal=task, answer=final_text, evidence=self._evidence()
                )
                self.emit(Verified(verification=verification))
                if verification.satisfied:
                    self._complete_subgoals(subgoals)
                    stopped = "done"
                    break
                if steps >= self.settings.max_steps:
                    stopped = "max_steps"
                    break
                self._add_note(
                    VERIFICATION_RETRY.format(goal=task, reason=verification.reason),
                    meta={"kind": "verification_retry"},
                )
                continue

            for call in response.tool_calls:
                tool_count += 1
                self._mark_running(subgoals)
                outcome = await self._execute_tool(call, current_goal)
                self._persist_tool(outcome)
                self.state.turn_messages.append(Message.tool_result(call.id, outcome.output, name=call.name))
                self.emit(ToolFinished(outcome=outcome))

        if stopped == "max_steps" and verification is None:
            self.emit(
                Notice(
                    level="warning",
                    text=f"step budget exhausted after {steps} steps without a verified answer",
                )
            )
        result = TurnResult(
            text=final_text,
            steps=steps,
            usage=usage,
            verification=verification,
            stopped_reason=stopped,
            tool_calls=tool_count,
        )
        self.emit(TurnFinished(result=result))
        return result

    async def aclose(self) -> None:
        await self.judge.aclose()
        await self.llm.aclose()
        if self.verifier_llm is not self.llm:
            await self.verifier_llm.aclose()
        self.tool_context.store.close()

    # -- planning ----------------------------------------------------------

    async def _plan(self, task: str) -> list[SubGoal]:
        known = self._active_subgoals()
        try:
            subgoals = await self.decomposer.decompose(task, known=known)
        except JetError as exc:
            self.emit(Notice(level="warning", text=f"decomposition skipped: {exc}"))
            subgoals = []
        if subgoals:
            self.emit(PlanReady(goal=task, subgoals=[subgoal.text for subgoal in subgoals]))
        return subgoals

    def _active_subgoals(self) -> list[SubGoal]:
        chunks = self.state.store.by_kind(ChunkKind.SUBGOAL)
        active: list[SubGoal] = []
        for chunk in chunks:
            status = chunk.status or GoalStatus.PENDING
            if status in (GoalStatus.PENDING, GoalStatus.RUNNING):
                active.append(
                    SubGoal(
                        id=chunk.id,
                        text=chunk.content,
                        parent_id=chunk.parent_id,
                        depth=int(chunk.meta.get("depth", 0)),
                        status=status,
                        chunk_id=chunk.id,
                    )
                )
        return active

    def _current_goal(self, task: str, subgoals: list[SubGoal]) -> str:
        for subgoal in subgoals:
            if subgoal.status in (GoalStatus.PENDING, GoalStatus.RUNNING):
                return f"{task}\nCurrent step: {subgoal.text}"
        return task

    def _mark_running(self, subgoals: list[SubGoal]) -> None:
        for subgoal in subgoals:
            if subgoal.status is GoalStatus.PENDING:
                subgoal.status = GoalStatus.RUNNING
                self.state.store.update_status(subgoal.chunk_id, GoalStatus.RUNNING)
                return

    def _complete_subgoals(self, subgoals: list[SubGoal]) -> None:
        for subgoal in subgoals:
            if subgoal.status in (GoalStatus.PENDING, GoalStatus.RUNNING):
                subgoal.status = GoalStatus.DONE
                self.state.store.update_status(subgoal.chunk_id, GoalStatus.DONE)

    # -- generation --------------------------------------------------------

    async def _stream(self, messages: list[Message], tools: Sequence[ToolSpec]) -> LLMResponse:
        response: LLMResponse | None = None
        with self.state.trace.span(
            "llm.turn",
            provider=getattr(self.llm, "name", "unknown"),
            model=self.llm.model,
            message_count=len(messages),
            tool_names=[spec.name for spec in tools],
        ) as extra:
            async for event in self.llm.stream(messages, tools=tools):
                if isinstance(event, TextDelta):
                    self.emit(AssistantDelta(text=event.text))
                elif isinstance(event, ReasoningDelta):
                    self.emit(AssistantThinking(text=event.text))
                elif isinstance(event, StreamDone):
                    response = event.response
            if response is None:
                raise ProviderError("LLM stream ended without a response")
            extra["input_tokens"] = response.usage.input_tokens
            extra["output_tokens"] = response.usage.output_tokens
            extra["reasoning_chars"] = len(response.reasoning)
            extra["cost_usd"] = round(
                cost_usd(
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                    float(getattr(self.llm, "input_price", 0.0)),
                    float(getattr(self.llm, "output_price", 0.0)),
                ),
                6,
            )
        return response

    def _persist_assistant(self, response: LLMResponse) -> None:
        if not response.text and not response.tool_calls:
            return
        chunk = self.state.store.add(
            ChunkKind.ASSISTANT_MESSAGE,
            response.text,
            meta={"tool_calls": [call.to_json() for call in response.tool_calls]},
        )
        self.emit(ChunkAdded(chunk=chunk))

    def _persist_tool(self, outcome: ToolOutcome) -> None:
        content = outcome.output[:TOOL_RESULT_MAX_CHARS]
        chunk = self.state.store.add(
            ChunkKind.TOOL_RESULT,
            content,
            source=outcome.tool_call.name,
            meta={
                "tool": outcome.tool_call.name,
                "arguments": outcome.tool_call.arguments,
                "ok": outcome.ok,
                "duration_ms": outcome.duration_ms,
                "error": outcome.error,
            },
        )
        self.emit(ChunkAdded(chunk=chunk))

    def _add_note(self, text: str, *, meta: dict[str, Any] | None = None) -> None:
        chunk = self.state.store.add(ChunkKind.SYSTEM_NOTE, text, meta=meta or {})
        self.emit(ChunkAdded(chunk=chunk))
        self.state.turn_messages.append(Message.user(f"{CONTEXT_NOTE_HEADER}\n{text}"))

    # -- tools -------------------------------------------------------------

    async def _execute_tool(self, call: ToolCall, goal: str) -> ToolOutcome:
        started = time.monotonic()
        tool = self.registry.get(call.name)
        if tool is None:
            return ToolOutcome(
                tool_call=call,
                ok=False,
                output=f"unknown tool {call.name!r}; available: {', '.join(self.registry.names())}",
                duration_ms=0,
                error="unknown_tool",
            )
        self.emit(ToolProposed(call=call, spec=tool.spec))
        verdict = await self.policy.evaluate(tool=tool.spec, arguments=call.arguments, goal=goal)
        allowed = verdict.allowed
        if verdict.decision is ApprovalDecision.ASK:
            self.emit(ApprovalRequested(call=call, verdict=verdict))
            if self.approve is None:
                allowed = False
                verdict = PolicyVerdict(
                    decision=ApprovalDecision.DENY,
                    reason=f"{verdict.reason}; no approver available (non-interactive session)",
                    rule="no_approver",
                )
            else:
                allowed = await self.approve(call, verdict)
        if not allowed:
            note = f"denied by policy: {verdict.reason}"
            self._add_note(
                APPROVAL_DENIED.format(tool=call.name, reason=verdict.reason),
                meta={"kind": "approval_denied", "tool": call.name},
            )
            return ToolOutcome(
                tool_call=call,
                ok=False,
                output=note,
                duration_ms=int((time.monotonic() - started) * 1000),
                error="denied",
            )
        try:
            result = await tool.run(call.arguments, self.tool_context)
            ok, output, error = result.ok, result.output, (None if result.ok else "tool reported failure")
        except JetError as exc:
            ok, output, error = False, f"{type(exc).__name__}: {exc}", type(exc).__name__
        except Exception as exc:  # noqa: BLE001 - unexpected tool crashes must not kill the turn
            ok, output, error = False, f"unexpected tool failure: {type(exc).__name__}: {exc}", "unexpected"
            self.state.trace.event("tool.crash", tool=call.name, error=repr(exc))
        duration_ms = int((time.monotonic() - started) * 1000)
        self._touch_paths(call)
        return ToolOutcome(
            tool_call=call,
            ok=ok,
            output=output,
            duration_ms=duration_ms,
            error=error,
        )

    def _touch_paths(self, call: ToolCall) -> None:
        for key in ("path", "file", "target"):
            value = call.arguments.get(key)
            if isinstance(value, str):
                try:
                    loaded = self.memory.load_for_path(Path(value))
                except (OSError, ValueError):
                    return
                if loaded:
                    self.emit(Notice(level="info", text=f"loaded memory: {', '.join(loaded)}"))
                return

    # -- support -----------------------------------------------------------

    def _system_prompt(self) -> str:
        memory_present = bool(self.state.store.by_kind(ChunkKind.MEMORY))
        return system_prompt(self.settings.workspace, self.registry.snippets(), memory_present=memory_present)

    def _evidence(self) -> str:
        chunks = [
            chunk
            for chunk in self.state.store.recent(EVIDENCE_CHUNKS * 2)
            if chunk.kind in (ChunkKind.TOOL_RESULT, ChunkKind.ASSISTANT_MESSAGE)
        ][-EVIDENCE_CHUNKS:]
        parts: list[str] = []
        used = 0
        for chunk in chunks:
            text = chunk.content[:EVIDENCE_CHARS_PER_CHUNK]
            if used + len(text) > EVIDENCE_TOTAL_CHARS:
                break
            label = f"[{chunk.kind.value}{' · ' + chunk.source if chunk.source else ''}]"
            parts.append(f"{label}\n{text}")
            used += len(text)
        return "\n\n".join(parts) if parts else "(no evidence recorded)"
