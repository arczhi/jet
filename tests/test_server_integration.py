"""Real-wire integration tests: uvicorn + HTTP/SSE, exactly as the client sees it.

``ASGITransport`` buffers responses, so approval and cancellation flows — which
depend on incremental SSE delivery — are only meaningful against a real server.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from jet.agent.loop import Agent
from jet.agent.session import create_agent
from jet.config import Settings
from jet.context.summarizer import TruncatingSummarizer
from jet.core.types import ToolCall
from jet.desktop import DesktopServer
from jet.providers.mock import MockTurn
from jet.server.manager import Approver, EventSink
from tests.conftest import NullDecomposer, make_judge, make_llm

pytestmark = pytest.mark.integration


def agent_factory(settings: Settings, llm: Any, judge: Any) -> Any:
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

    return factory


@contextmanager
def serve(settings: Settings, llm: Any, judge: Any | None = None) -> Iterator[str]:
    server = DesktopServer(
        settings,
        port=0,
        agent_factory=agent_factory(settings, llm, judge or make_judge()),
    )
    server.start()
    try:
        yield server.url
    finally:
        server.stop()


async def events(url: str, turn_id: str) -> AsyncIterator[dict[str, Any]]:
    async with (
        httpx.AsyncClient(base_url=url, timeout=10.0) as http,
        http.stream("GET", f"/api/turns/{turn_id}/events") as response,
    ):
        assert response.status_code == 200
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                yield json.loads(line[6:])


async def test_approval_round_trip_over_http(settings: Settings, workspace: Path) -> None:
    ask_settings = settings.model_copy(update={"approval_mode": "ask"})
    llm = make_llm(
        MockTurn(tool_calls=[ToolCall("c1", "write_file", {"path": "wire.txt", "content": "ok"})]),
        MockTurn(text="Wrote wire.txt."),
    )
    with serve(ask_settings, llm) as url:
        async with httpx.AsyncClient(base_url=url, timeout=10.0) as http:
            turn_id = (await http.post("/api/turns", json={"task": "write wire.txt"})).json()["turn_id"]
            approved = False
            async for event in events(url, turn_id):
                if event["type"] == "approval_requested":
                    response = await http.post(
                        f"/api/turns/{turn_id}/approvals/{event['approval_id']}",
                        json={"allow": True},
                    )
                    assert response.status_code == 200
                    approved = True
                if event["type"] == "turn_finished":
                    assert event["stopped_reason"] == "done"
                    break
            assert approved
        assert (workspace / "wire.txt").read_text() == "ok"


async def test_approval_deny_over_http(settings: Settings, workspace: Path) -> None:
    ask_settings = settings.model_copy(update={"approval_mode": "ask"})
    llm = make_llm(
        MockTurn(tool_calls=[ToolCall("c1", "write_file", {"path": "blocked.txt", "content": "x"})]),
        MockTurn(text="Denied."),
    )
    with serve(ask_settings, llm) as url:
        async with httpx.AsyncClient(base_url=url, timeout=10.0) as http:
            turn_id = (await http.post("/api/turns", json={"task": "write blocked.txt"})).json()["turn_id"]
            async for event in events(url, turn_id):
                if event["type"] == "approval_requested":
                    await http.post(
                        f"/api/turns/{turn_id}/approvals/{event['approval_id']}",
                        json={"allow": False},
                    )
                if event["type"] == "turn_finished":
                    break
        assert not (workspace / "blocked.txt").exists()


async def test_cancel_over_http(settings: Settings) -> None:
    llm = make_llm(MockTurn(text="slow", delay_s=0.6), MockTurn(text="later"))
    with serve(settings, llm) as url:
        async with httpx.AsyncClient(base_url=url, timeout=10.0) as http:
            turn_id = (await http.post("/api/turns", json={"task": "long"})).json()["turn_id"]
            canceled = False
            async for event in events(url, turn_id):
                if event["type"] == "turn_started":
                    result = await http.post(f"/api/turns/{turn_id}/cancel")
                    assert result.json() == {"canceled": True}
                    canceled = True
                if event["type"] == "turn_canceled":
                    break
            assert canceled
            # The session survives cancellation and accepts a new turn.
            again = await http.post("/api/turns", json={"task": "second"})
            assert again.status_code == 202


async def test_stream_replays_history_after_reconnect(settings: Settings) -> None:
    llm = make_llm("done")
    with serve(settings, llm) as url:
        async with httpx.AsyncClient(base_url=url, timeout=10.0) as http:
            turn_id = (await http.post("/api/turns", json={"task": "hi"})).json()["turn_id"]
            await asyncio.sleep(0.4)
            types = [event["type"] async for event in events(url, turn_id)]
            assert types[0] == "stream_open"
            assert "turn_started" in types and "turn_finished" in types
