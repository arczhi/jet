"""Turn manager: owns the agent, routes events to subscribers, brokers approvals.

The desktop client drives turns over HTTP/SSE. This module is the only place that
knows how a turn's event stream and approval prompts map onto the wire. The agent
itself stays UI-agnostic: it emits events and awaits approval callbacks.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from jet.config import Settings
from jet.core.types import ApprovalMode, PolicyVerdict, ToolCall
from jet.errors import JetError
from jet.server.serialize import serialize_event
from jet.tracing import cost_usd

EventSink = Callable[[Any], None]
Approver = Callable[[ToolCall, PolicyVerdict], Awaitable[bool]]
AgentFactory = Callable[[EventSink, Approver], Any]

APPROVAL_TIMEOUT_S = 1800
HEARTBEAT_S = 15.0


class TurnBusyError(JetError):
    """A turn is already running; the client must wait or cancel it."""


class SetupRequiredError(JetError):
    """Provider credentials are missing; the client must show the setup dialog."""


@dataclass
class TurnRecord:
    id: str
    task: str
    events: list[dict[str, Any]] = field(default_factory=list)
    queues: list[asyncio.Queue[dict[str, Any] | None]] = field(default_factory=list)
    approvals: dict[str, asyncio.Future[bool]] = field(default_factory=dict)
    runner: asyncio.Task[None] | None = None
    finished: bool = False
    cancel_requested: bool = False


class TurnManager:
    def __init__(
        self,
        factory_builder: Callable[[Any], AgentFactory],
        settings: Settings,
    ):
        """``factory_builder`` maps (possibly reloaded) settings to an agent factory.

        It is re-invoked on ``reload`` so a first-run credentials save produces
        a working agent — the original settings object must never be reused.
        """
        self.settings = settings
        self._factory_builder = factory_builder
        self._agent: Any = None
        self._turn: TurnRecord | None = None
        self._closed = False
        self._setup_error: str | None = None

    # -- lifecycle ---------------------------------------------------------

    @property
    def agent(self) -> Any:
        if self._agent is None:
            raise RuntimeError("TurnManager.startup() must run before serving requests")
        return self._agent

    async def startup(self) -> None:
        if self._agent is None:
            from jet.errors import ConfigError

            try:
                self._agent = self._factory_builder(self.settings)(self._emit, self._approve)
            except ConfigError as exc:
                # First-run without credentials: boot into the setup state
                # instead of failing the whole service.
                self._setup_error = str(exc)
                self._agent = None

    async def reload(self) -> dict[str, Any]:
        """Rebuild the engine after a settings change (e.g. first-run setup).

        Settings are re-resolved from every source so the freshly saved
        credentials layer actually applies; runtime choices (workspace,
        approval mode) carry over.
        """
        from jet.config import load_settings

        fresh = load_settings(workspace=self.settings.workspace)
        self.settings = fresh.model_copy(update={"approval_mode": self.settings.approval_mode})
        if self._agent is not None:
            await self._agent.aclose()
            self._agent = None
        self._setup_error = None
        await self.startup()
        return self.state()

    @property
    def setup_error(self) -> str | None:
        return self._setup_error

    def missing_setup(self) -> list[str]:
        """Which provider credentials the user still needs to enter."""
        missing: list[str] = []
        if self.settings.judge_provider == "typesafe" and not self.settings.typesafe_api_key:
            missing.append("jev")
        try:
            profile = self.settings.active_llm
        except JetError:
            missing.append("deepseek")
        else:
            if profile.model != "mock" and not profile.base_url:
                missing.append("deepseek")
        return missing

    async def shutdown(self) -> None:
        self._closed = True
        if self._turn and not self._turn.finished and self._turn.runner:
            self._turn.runner.cancel()
            await asyncio.gather(self._turn.runner, return_exceptions=True)
        if self._agent is not None:
            await self._agent.aclose()
            self._agent = None

    async def new_session(self) -> str:
        if self._turn and not self._turn.finished:
            raise TurnBusyError("cancel the running turn before starting a new session")
        if self._agent is not None:
            await self._agent.aclose()
        self._agent = self._factory_builder(self.settings)(self._emit, self._approve)
        return str(self._agent.session_id)

    async def switch_workspace(self, raw_path: str) -> dict[str, Any]:
        """Point the whole engine at a new directory and start a fresh session.

        The choice is persisted so the next app launch opens the same directory.
        """
        from pathlib import Path

        from jet.errors import JetError
        from jet.state import update_state

        resolved = Path(raw_path).expanduser()
        if not resolved.is_dir():
            raise JetError(f"not a directory: {raw_path}")

        if self._turn and not self._turn.finished:
            raise TurnBusyError("cancel the running turn before switching directories")
        self.settings.workspace = resolved.resolve()
        await self.new_session()
        update_state(self.settings.home, last_workspace=str(self.settings.workspace))
        return self.state()

    # -- turns -------------------------------------------------------------

    @property
    def current_turn(self) -> TurnRecord | None:
        return self._turn

    def turn(self, turn_id: str) -> TurnRecord | None:
        if self._turn is not None and self._turn.id == turn_id:
            return self._turn
        return None

    async def start_turn(self, task: str) -> TurnRecord:
        if self._agent is None:
            raise SetupRequiredError(
                self._setup_error or "provider credentials are missing; complete the setup dialog first"
            )
        if self._turn is not None and not self._turn.finished:
            raise TurnBusyError("a turn is already running")
        record = TurnRecord(id=_new_turn_id(), task=task)
        self._turn = record

        async def run() -> None:
            try:
                await self.agent.run_turn(task)
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001 - client must see the failure, not a dead stream
                self._push({"type": "notice", "level": "error", "text": f"{type(exc).__name__}: {exc}"})
                self._push(
                    {
                        "type": "turn_finished",
                        "text": "",
                        "steps": 0,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                        "verification": None,
                        "stopped_reason": "error",
                        "tool_calls": 0,
                        "cost_usd": 0.0,
                    }
                )
                self._trace_payload("turn.failed", error=f"{type(exc).__name__}: {exc}")
            finally:
                self._finalize(record, canceled=record.cancel_requested)

        record.runner = asyncio.create_task(run())
        return record

    async def cancel_turn(self, turn_id: str) -> bool:
        record = self.turn(turn_id)
        if record is None or record.finished or record.runner is None:
            return False
        record.cancel_requested = True
        record.runner.cancel()
        # A task canceled before its first step never runs its own finally block,
        # so finalization happens here too. ``_finalize`` is idempotent.
        self._finalize(record, canceled=True)
        return True

    def _finalize(self, record: TurnRecord, *, canceled: bool) -> None:
        if record.finished:
            return
        if canceled:
            self._push({"type": "turn_canceled", "reason": "canceled by client"})
            self._trace_payload("turn.canceled")
        record.finished = True
        for queue in list(record.queues):
            queue.put_nowait(None)

    def _trace_payload(self, kind: str, **fields: Any) -> None:
        """Client-only events (cancellations, crashes) must still be traceable."""
        agent = self._agent
        if agent is not None:
            agent.trace.event(kind, **fields)

    async def subscribe(self, record: TurnRecord) -> AsyncIterator[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        for payload in list(record.events):
            yield payload
        if record.finished:
            return
        record.queues.append(queue)
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_S)
                except TimeoutError:
                    yield {"type": "heartbeat", "ts": time.time()}
                    continue
                if item is None:
                    return
                yield item
        finally:
            if queue in record.queues:
                record.queues.remove(queue)

    # -- approvals ---------------------------------------------------------

    async def resolve_approval(self, turn_id: str, approval_id: str, allow: bool) -> bool:
        record = self.turn(turn_id)
        if record is None:
            return False
        future = record.approvals.get(approval_id)
        if future is None or future.done():
            return False
        future.set_result(allow)
        return True

    async def _approve(self, call: ToolCall, verdict: PolicyVerdict) -> bool:
        record = self._turn
        if record is None or record.finished:
            return False
        approval_id = f"apr_{int(time.time() * 1000)}_{len(record.approvals)}"
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        record.approvals[approval_id] = future
        self._push(
            {
                "type": "approval_requested",
                "approval_id": approval_id,
                "turn_id": record.id,
                "tool": call.name,
                "arguments": call.arguments,
                "reason": verdict.reason,
                "rule": verdict.rule,
            }
        )
        try:
            allowed = await asyncio.wait_for(future, timeout=APPROVAL_TIMEOUT_S)
        except TimeoutError:
            allowed = False
            self._push(
                {
                    "type": "notice",
                    "level": "warning",
                    "text": f"approval timed out after {APPROVAL_TIMEOUT_S}s; denied by default",
                }
            )
        finally:
            record.approvals.pop(approval_id, None)
        self._push({"type": "approval_resolved", "approval_id": approval_id, "allowed": allowed})
        return allowed

    # -- inspection --------------------------------------------------------

    def state(self) -> dict[str, Any]:
        agent = self._agent
        settings = self.settings
        from jet.state import read_state

        ui_state = read_state(settings.home)
        try:
            llm_base_url = settings.active_llm.base_url or ""
            llm_model = settings.llm_model
            verifier_model = settings.verifier_llm().model
        except JetError:
            llm_base_url, llm_model, verifier_model = "", "", ""
        payload: dict[str, Any] = {
            "workspace": str(settings.workspace),
            "theme": ui_state.get("theme") or "light",
            "lang": ui_state.get("lang") or "en",
            "setup_required": bool(self.missing_setup()),
            "setup_missing": self.missing_setup(),
            "setup_error": self._setup_error,
            "typesafe_base_url": settings.typesafe_base_url,
            "llm_base_url": llm_base_url,
            "judge": settings.judge_provider,
            "llm_profile": settings.llm_profile,
            "llm_model": llm_model,
            "verifier_profile": settings.verifier_llm_profile,
            "verifier_model": verifier_model,
            "approval_mode": settings.approval_mode,
            "busy": bool(self._turn and not self._turn.finished),
        }
        if agent is not None:
            payload.update(
                {
                    "session_id": str(agent.session_id),
                    "trace_path": str(agent.trace.path),
                    "chunks": agent.store.count(),
                    "usage": agent.state.total_usage.to_json(),
                    "cost_usd": round(agent.state.total_cost_usd, 4),
                }
            )
            if self._turn is not None:
                payload["approval_mode"] = str(agent.policy.mode.value)
                payload["current_turn"] = self._turn.id
        return payload

    def chunks(self, limit: int = 50) -> dict[str, Any]:
        if self._agent is None:
            return {"total": 0, "chunks": []}
        total = self._agent.store.count()
        chunks = self._agent.store.recent(limit)
        return {
            "total": total,
            "chunks": [
                {
                    "id": chunk.id,
                    "seq": chunk.seq,
                    "kind": chunk.kind.value,
                    "source": chunk.source,
                    "pinned": chunk.pinned,
                    "status": chunk.status.value if chunk.status else None,
                    "tokens": chunk.token_estimate,
                    "created_at": chunk.created_at,
                    "content": chunk.content[:4_000],
                }
                for chunk in chunks
            ],
        }

    def set_approval_mode(self, mode: ApprovalMode) -> str:
        self.settings.approval_mode = mode.value
        if self._agent is not None:
            self._agent.policy.mode = mode
        return mode.value

    # -- event plumbing ----------------------------------------------------

    def _emit(self, event: Any) -> None:
        payload = serialize_event(event)
        if payload is None:
            return
        if payload["type"] == "turn_finished":
            payload["cost_usd"] = cost_usd(
                payload["usage"]["input_tokens"],
                payload["usage"]["output_tokens"],
                float(getattr(self.agent.llm, "input_price", 0.0)),
                float(getattr(self.agent.llm, "output_price", 0.0)),
            )
        self._push(payload)

    def _push(self, payload: dict[str, Any]) -> None:
        record = self._turn
        if record is None or record.finished:
            return
        payload.setdefault("ts", time.time())
        record.events.append(payload)
        for queue in list(record.queues):
            queue.put_nowait(payload)


def _new_turn_id() -> str:
    from jet.ids import new_id

    return new_id("turn")
