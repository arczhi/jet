"""HTTP service tests.

Two layers:

* ASGI tests for request/response endpoints (state, chunks, sessions, errors).
* Manager-level tests for streaming, approvals, and cancellation. SSE is not
  exercised through ``ASGITransport`` because it buffers responses until the app
  finishes, which turns an approval deadlock into a test hang; the real-wire SSE
  contract lives in ``test_server_integration.py``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import httpx

from jet.agent.loop import Agent
from jet.agent.session import create_agent
from jet.config import Settings, load_settings
from jet.context.summarizer import TruncatingSummarizer
from jet.core.types import ToolCall
from jet.providers.mock import MockTurn
from jet.server.app import create_app
from jet.server.manager import Approver, EventSink, TurnManager
from tests.conftest import NullDecomposer, make_judge, make_llm


def build(settings: Settings, *, llm: Any = None, judge: Any = None) -> TurnManager:
    judge = judge or make_judge()
    llm = llm or make_llm("All done.")

    def factory(emit: EventSink, approve: Approver) -> Agent:
        return cast(
            Agent,
            create_agent(
                settings,
                judge=judge,
                llm=llm,
                summarizer=TruncatingSummarizer(),
                decomposer=NullDecomposer(),
                emit=emit,
                approve=approve,
            ),
        )

    return TurnManager(lambda _settings: factory, settings)


async def collect(
    manager: TurnManager, turn_id: str, *, until: str = "turn_finished", timeout: float = 5.0
) -> list[dict[str, Any]]:
    """Consume a turn's event stream until ``until`` arrives."""
    record = manager.turn(turn_id)
    assert record is not None
    events: list[dict[str, Any]] = []

    async def consume() -> None:
        async for event in manager.subscribe(record):
            events.append(event)
            if event["type"] == until:
                return

    await asyncio.wait_for(consume(), timeout=timeout)
    return events


async def wait_for(manager: TurnManager, predicate: Any, *, timeout: float = 5.0) -> dict[str, Any]:
    """Poll the current turn's buffered events for a predicate match."""
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        record = manager.current_turn
        if record is not None:
            for event in list(record.events):
                if predicate(event):
                    return event
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError("event did not arrive")
        await asyncio.sleep(0.01)


async def test_health_state_and_index(settings: Settings) -> None:
    manager = build(settings)
    await manager.startup()
    app = create_app(manager)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://jet.test") as http:
        assert (await http.get("/health")).json() == {"ok": True}
        index = await http.get("/")
        assert index.status_code == 200 and "jet" in index.text
        state = (await http.get("/api/state")).json()
        assert state["session_id"]
        assert state["llm_model"] == "mock"
        assert state["workspace"] == str(settings.workspace)
        assert state["busy"] is False
    await manager.shutdown()


async def test_turn_completes_and_is_inspectable(settings: Settings) -> None:
    manager = build(settings, llm=make_llm("All done."))
    await manager.startup()
    app = create_app(manager)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://jet.test") as http:
        response = await http.post("/api/turns", json={"task": "say hi"})
        assert response.status_code == 202
        assert response.json()["turn_id"]
        finished = await wait_for(manager, lambda event: event["type"] == "turn_finished")
        assert finished["stopped_reason"] == "done"
        assert finished["cost_usd"] == 0.0
        payload = (await http.get("/api/chunks?limit=10")).json()
        assert payload["total"] >= 2
        kinds = [chunk["kind"] for chunk in payload["chunks"]]
        assert "user_message" in kinds and "assistant_message" in kinds
        state = (await http.get("/api/state")).json()
        assert state["busy"] is False and state["chunks"] > 0
    await manager.shutdown()


async def test_second_turn_while_busy_is_rejected(settings: Settings) -> None:
    manager = build(settings, llm=make_llm(MockTurn(text="slow", delay_s=0.3), MockTurn(text="x")))
    await manager.startup()
    app = create_app(manager)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://jet.test") as http:
        first = await http.post("/api/turns", json={"task": "one"})
        assert first.status_code == 202
        second = await http.post("/api/turns", json={"task": "two"})
        assert second.status_code == 409
        await wait_for(manager, lambda event: event["type"] == "turn_finished")
        third = await http.post("/api/turns", json={"task": "three"})
        assert third.status_code == 202
    await manager.shutdown()


async def test_blank_task_and_unknown_turn(settings: Settings) -> None:
    manager = build(settings)
    await manager.startup()
    app = create_app(manager)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://jet.test") as http:
        assert (await http.post("/api/turns", json={"task": "   "})).status_code == 422
        assert (await http.get("/api/turns/nope/events")).status_code == 404
        assert (await http.post("/api/turns/nope/approvals/x", json={"allow": True})).status_code == 404
    await manager.shutdown()


async def test_approval_allow_writes_the_file(settings: Settings, workspace: Path) -> None:
    ask_settings = settings.model_copy(update={"approval_mode": "ask"})
    llm = make_llm(
        MockTurn(tool_calls=[ToolCall("c1", "write_file", {"path": "out.txt", "content": "x"})]),
        MockTurn(text="Wrote out.txt."),
    )
    manager = build(ask_settings, llm=llm)
    await manager.startup()
    record = await manager.start_turn("write out.txt")

    async def drive() -> list[str]:
        types: list[str] = []
        async for event in manager.subscribe(record):
            types.append(event["type"])
            if event["type"] == "approval_requested":
                assert event["tool"] == "write_file"
                assert event["reason"]
                resolved = await manager.resolve_approval(record.id, event["approval_id"], True)
                assert resolved
            if event["type"] == "turn_finished":
                return types
        return types

    types = await asyncio.wait_for(drive(), timeout=5.0)
    assert "approval_requested" in types and "approval_resolved" in types
    assert "turn_finished" in types
    assert (workspace / "out.txt").read_text() == "x"
    await manager.shutdown()


async def test_approval_deny_blocks_the_file(settings: Settings, workspace: Path) -> None:
    ask_settings = settings.model_copy(update={"approval_mode": "ask"})
    llm = make_llm(
        MockTurn(tool_calls=[ToolCall("c1", "write_file", {"path": "no.txt", "content": "x"})]),
        MockTurn(text="Denied."),
    )
    manager = build(ask_settings, llm=llm)
    await manager.startup()
    record = await manager.start_turn("write no.txt")
    denied = False

    async def drive() -> None:
        nonlocal denied
        async for event in manager.subscribe(record):
            if event["type"] == "approval_requested":
                await manager.resolve_approval(record.id, event["approval_id"], False)
            if event["type"] == "tool_finished" and not event["ok"]:
                denied = True
            if event["type"] == "turn_finished":
                return

    await asyncio.wait_for(drive(), timeout=5.0)
    assert denied
    assert not (workspace / "no.txt").exists()
    await manager.shutdown()


async def test_cancel_stops_the_turn(settings: Settings) -> None:
    manager = build(settings, llm=make_llm(MockTurn(text="slow", delay_s=0.5), MockTurn(text="x")))
    await manager.startup()
    record = await manager.start_turn("long task")
    assert await manager.cancel_turn(record.id) is True
    canceled = await wait_for(manager, lambda event: event["type"] == "turn_canceled")
    assert canceled["reason"]
    assert record.finished
    await manager.shutdown()


async def test_approval_mode_can_be_changed(settings: Settings) -> None:
    manager = build(settings.model_copy(update={"approval_mode": "ask"}))
    await manager.startup()
    app = create_app(manager)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://jet.test") as http:
        response = await http.post("/api/approval-mode", json={"mode": "auto"})
        assert response.json() == {"mode": "auto"}
        assert (await http.get("/api/state")).json()["approval_mode"] == "auto"
    await manager.shutdown()


async def test_new_session_replaces_state(settings: Settings) -> None:
    manager = build(settings)
    await manager.startup()
    app = create_app(manager)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://jet.test") as http:
        before = (await http.get("/api/state")).json()["session_id"]
        response = await http.post("/api/session/new")
        assert response.status_code == 200
        assert response.json()["session_id"] != before
        after = (await http.get("/api/state")).json()["session_id"]
        assert after != before
    await manager.shutdown()


async def test_workspace_switch_updates_engine_and_persists(settings: Settings) -> None:
    manager = build(settings)
    await manager.startup()
    app = create_app(manager)
    other_dir = settings.workspace.parent / "other-ws"
    other_dir.mkdir()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://jet.test") as http:
        before = (await http.get("/api/state")).json()
        response = await http.post("/api/workspace", json={"path": str(other_dir)})
        assert response.status_code == 200
        after = response.json()
        assert after["workspace"] == str(other_dir)
        assert after["session_id"] != before["session_id"]
    from jet.state import read_state

    assert read_state(settings.home)["last_workspace"] == str(other_dir)
    await manager.shutdown()


async def test_workspace_switch_rejects_non_directories(settings: Settings) -> None:
    manager = build(settings)
    await manager.startup()
    app = create_app(manager)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://jet.test") as http:
        response = await http.post("/api/workspace", json={"path": "no/such/dir"})
        assert response.status_code == 422
        file_path = settings.workspace / "file.txt"
        file_path.write_text("x")
        response = await http.post("/api/workspace", json={"path": str(file_path)})
        assert response.status_code == 422
    await manager.shutdown()


async def test_workspace_switch_while_busy_is_rejected(settings: Settings) -> None:
    manager = build(settings, llm=make_llm(MockTurn(text="slow", delay_s=0.3), MockTurn(text="x")))
    await manager.startup()
    app = create_app(manager)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://jet.test") as http:
        await http.post("/api/turns", json={"task": "busy"})
        response = await http.post("/api/workspace", json={"path": str(settings.workspace)})
        assert response.status_code == 409
        await wait_for(manager, lambda event: event["type"] == "turn_finished")
    await manager.shutdown()


async def test_lang_persists_and_surfaces_in_state(settings: Settings) -> None:
    manager = build(settings)
    await manager.startup()
    app = create_app(manager)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://jet.test") as http:
        assert (await http.get("/api/state")).json()["lang"] == "en"
        response = await http.post("/api/lang", json={"lang": "zh"})
        assert response.json() == {"lang": "zh"}
        assert (await http.get("/api/state")).json()["lang"] == "zh"
        assert (await http.post("/api/lang", json={"lang": "fr"})).status_code == 422
    from jet.state import read_state

    assert read_state(settings.home)["lang"] == "zh"
    await manager.shutdown()


async def test_theme_persists_and_surfaces_in_state(settings: Settings) -> None:
    manager = build(settings)
    await manager.startup()
    app = create_app(manager)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://jet.test") as http:
        state = (await http.get("/api/state")).json()
        assert state["theme"] == "light"
        response = await http.post("/api/theme", json={"theme": "dark"})
        assert response.json() == {"theme": "dark"}
        assert (await http.get("/api/state")).json()["theme"] == "dark"
        assert (await http.post("/api/theme", json={"theme": "midnight"})).status_code == 422
    from jet.state import read_state

    assert read_state(settings.home)["theme"] == "dark"
    await manager.shutdown()


async def test_fs_list_returns_directories(settings: Settings) -> None:
    manager = build(settings)
    await manager.startup()
    app = create_app(manager)
    nested = settings.workspace / "sub"
    nested.mkdir()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://jet.test") as http:
        payload = (await http.get("/api/fs/list", params={"path": str(settings.workspace)})).json()
        names = [entry["name"] for entry in payload["entries"]]
        assert "sub" in names
        assert all(entry["path"] for entry in payload["entries"])
        assert payload["parent"] is not None and payload["home"]
        assert (await http.get("/api/fs/list", params={"path": "/no/such/dir"})).status_code == 404
    await manager.shutdown()


async def test_buffered_events_replay_to_late_subscribers(settings: Settings) -> None:
    manager = build(settings, llm=make_llm("done"))
    await manager.startup()
    record = await manager.start_turn("hi")
    await wait_for(manager, lambda event: event["type"] == "turn_finished")
    replay = await collect(manager, record.id, until="turn_finished")
    assert replay[0]["type"] == "turn_started"
    assert any(event["type"] == "turn_finished" for event in replay)
    await manager.shutdown()


async def test_startup_workspace_restores_last_directory(tmp_path: Path) -> None:
    from jet.desktop import resolve_startup_workspace
    from jet.state import update_state

    home = tmp_path / "home"
    last_dir = tmp_path / "last"
    last_dir.mkdir()
    update_state(home, last_workspace=str(last_dir))
    settings = load_settings(workspace=tmp_path / "fresh", home=home)
    resolved = resolve_startup_workspace(settings, explicit=False)
    assert resolved.workspace == last_dir.resolve()
    # An explicit choice always wins over memory.
    kept = resolve_startup_workspace(settings, explicit=True)
    assert kept.workspace == (tmp_path / "fresh").resolve()
    # A stale remembered directory is ignored, not fatal.
    update_state(home, last_workspace="/no/such/dir")
    fallback = resolve_startup_workspace(settings, explicit=False)
    assert fallback.workspace == (tmp_path / "fresh").resolve()
