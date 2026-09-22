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
from jet.errors import JetError
from jet.server.manager import SetupRequiredError, TurnBusyError, TurnManager
from jet.state import update_state

STATIC_DIR = Path(__file__).parent / "static"


class TurnRequest(BaseModel):
    task: str = Field(min_length=1)


class ApprovalDecision(BaseModel):
    allow: bool


class ApprovalModeRequest(BaseModel):
    mode: Literal["ask", "auto", "deny"]


class ThemeRequest(BaseModel):
    theme: Literal["light", "dark"]


class LangRequest(BaseModel):
    lang: Literal["en", "zh"]


class SetupRequest(BaseModel):
    typesafe_api_key: str = ""
    typesafe_base_url: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""


class WorkspaceRequest(BaseModel):
    path: str = Field(min_length=1)


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
        except SetupRequiredError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
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

    @app.post("/api/theme")
    async def set_theme(body: ThemeRequest) -> dict[str, str]:
        update_state(manager.settings.home, theme=body.theme)
        return {"theme": body.theme}

    @app.post("/api/lang")
    async def set_lang(body: LangRequest) -> dict[str, str]:
        update_state(manager.settings.home, lang=body.lang)
        return {"lang": body.lang}

    @app.post("/api/setup")
    async def setup(body: SetupRequest) -> dict[str, object]:
        from jet.config import load_credentials, save_credentials
        from jet.server.probe import probe_deepseek, probe_jev

        if not body.llm_api_key.strip() or not body.llm_base_url.strip():
            raise HTTPException(status_code=422, detail="DeepSeek API key 和端点地址都要填写")
        llm_url = body.llm_base_url.strip()
        llm_key = body.llm_api_key.strip()
        llm_model = body.llm_model.strip() or "deepseek-flash"
        jev_key = body.typesafe_api_key.strip()
        jev_url = body.typesafe_base_url.strip() or "https://api.typesafe.ai"

        jev_probe = await probe_jev(jev_url, jev_key) if jev_key else None
        deepseek_probe = await probe_deepseek(llm_url, llm_key, llm_model)
        report: dict[str, object] = {"deepseek": deepseek_probe.to_json()}
        if jev_probe is not None:
            report["jev"] = jev_probe.to_json()
        if not deepseek_probe.ok or (jev_probe is not None and not jev_probe.ok):
            # Do not persist credentials that are known-bad; the dialog shows
            # which key failed and why.
            raise HTTPException(status_code=422, detail=report)

        credentials = load_credentials()
        credentials["llm_profile"] = "official"
        credentials["llm_profiles"] = {
            "official": {
                "base_url": body.llm_base_url.strip(),
                "api_key": body.llm_api_key.strip(),
                "model": body.llm_model.strip() or "deepseek-flash",
            }
        }
        if body.typesafe_api_key.strip():
            credentials["typesafe_api_key"] = body.typesafe_api_key.strip()
            if body.typesafe_base_url.strip():
                credentials["typesafe_base_url"] = body.typesafe_base_url.strip()
        else:
            # No judge key entered: fall back to the offline mock judge so the
            # app still works; the setup dialog will reappear until a key is set.
            credentials["judge_provider"] = "mock"
        save_credentials(credentials)
        return await manager.reload()

    @app.post("/api/workspace")
    async def set_workspace(body: WorkspaceRequest) -> dict[str, object]:
        try:
            return await manager.switch_workspace(body.path)
        except TurnBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except JetError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/fs/list")
    async def list_directory(path: str = "") -> dict[str, object]:
        target = Path(path).expanduser().resolve() if path.strip() else Path.home().resolve()
        if not target.is_dir():
            raise HTTPException(status_code=404, detail=f"not a directory: {path}")
        entries: list[dict[str, str]] = []
        try:
            children = sorted(target.iterdir(), key=lambda child: child.name.lower())
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=f"not readable: {target}") from exc
        for child in children[:500]:
            if child.is_dir():
                entries.append({"name": child.name, "path": str(child)})
        parent = target.parent if target.parent != target else None
        return {
            "path": str(target),
            "parent": str(parent) if parent else None,
            "home": str(Path.home()),
            "entries": entries,
        }

    return app
