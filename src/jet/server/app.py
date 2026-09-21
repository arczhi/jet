"""FastAPI app: session state, turn submission, SSE event stream, approvals.

Routes stay thin; the TurnManager owns turn lifecycle and approval futures.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from jet.core.types import ApprovalMode
from jet.server.manager import TurnBusyError, TurnManager

STATIC_DIR = Path(__file__).parent / "static"


class TurnRequest(BaseModel):
    task: str = Field(min_length=1)


class ApprovalDecision(BaseModel):
    allow: bool


class ApprovalModeRequest(BaseModel):
    mode: Literal["ask", "auto", "deny"]


def create_app(manager: TurnManager) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await manager.startup()
        try:
            yield
        finally:
            await manager.shutdown()

    app = FastAPI(title="jet", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.manager = manager
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/api/state")
    async def state() -> dict[str, object]:
        return manager.state()

    @app.get("/api/chunks")
    async def chunks(limit: int = 60) -> dict[str, object]:
        return manager.chunks(max(1, min(500, limit)))

    @app.post("/api/turns", status_code=202)
    async def start_turn(body: TurnRequest) -> dict[str, str]:
        task = body.task.strip()
        if not task:
            raise HTTPException(status_code=422, detail="task must not be blank")
        try:
            record = await manager.start_turn(task)
        except TurnBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"turn_id": record.id}

    @app.get("/api/turns/{turn_id}/events")
    async def events(turn_id: str) -> StreamingResponse:
        record = manager.turn(turn_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown turn")

        async def stream() -> AsyncIterator[str]:
            yield f"data: {json.dumps({'type': 'stream_open', 'turn_id': turn_id})}\n\n"
            async for payload in manager.subscribe(record):
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
        )

    @app.post("/api/turns/{turn_id}/approvals/{approval_id}")
    async def decide_approval(turn_id: str, approval_id: str, body: ApprovalDecision) -> dict[str, bool]:
        resolved = await manager.resolve_approval(turn_id, approval_id, body.allow)
        if not resolved:
            raise HTTPException(status_code=404, detail="approval not pending")
        return {"resolved": True}

    @app.post("/api/turns/{turn_id}/cancel")
    async def cancel(turn_id: str) -> dict[str, bool]:
        return {"canceled": await manager.cancel_turn(turn_id)}

    @app.post("/api/session/new")
    async def new_session() -> dict[str, str]:
        try:
            session_id = await manager.new_session()
        except TurnBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"session_id": session_id}

    @app.post("/api/approval-mode")
    async def set_approval_mode(body: ApprovalModeRequest) -> dict[str, str]:
        return {"mode": manager.set_approval_mode(ApprovalMode(body.mode))}

    return app
