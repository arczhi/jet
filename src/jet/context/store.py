"""Chunk store: the durable, explicit state of a session.

Everything the agent has seen or produced is a typed chunk in a tree. Meta-
attention scores chunks; the context builder assembles a window from them. No
lossy compaction step exists — old detail stays addressable forever.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from jet.core.text import estimate_tokens
from jet.core.types import Chunk, ChunkKind, GoalStatus
from jet.ids import new_id

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at REAL NOT NULL,
    parent_id TEXT,
    source TEXT,
    pinned INTEGER NOT NULL DEFAULT 0,
    token_estimate INTEGER NOT NULL,
    status TEXT,
    meta TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_chunks_session_seq ON chunks(session_id, seq);
CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_id);
CREATE INDEX IF NOT EXISTS idx_chunks_kind ON chunks(session_id, kind);
"""


class ChunkStore:
    """SQLite-backed chunk tree for one session."""

    def __init__(self, path: Path | str, session_id: str):
        self.session_id = session_id
        self.path = Path(path) if path != ":memory:" else Path(":memory:")
        if path != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- writes ------------------------------------------------------------

    def add(
        self,
        kind: ChunkKind,
        content: str,
        *,
        parent_id: str | None = None,
        source: str | None = None,
        pinned: bool = False,
        status: GoalStatus | None = None,
        meta: Mapping[str, Any] | None = None,
        chunk_id: str | None = None,
    ) -> Chunk:
        with self._lock:
            seq_row = self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM chunks WHERE session_id = ?",
                (self.session_id,),
            ).fetchone()
            seq = int(seq_row[0])
            chunk = Chunk(
                id=chunk_id or new_id("chk"),
                session_id=self.session_id,
                seq=seq,
                kind=kind,
                content=content,
                created_at=time.time(),
                parent_id=parent_id,
                source=source,
                pinned=pinned,
                token_estimate=estimate_tokens(content),
                status=status,
                meta=dict(meta or {}),
            )
            self._conn.execute(
                """
                INSERT INTO chunks
                (id, session_id, seq, kind, content, created_at, parent_id, source,
                 pinned, token_estimate, status, meta)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.id,
                    chunk.session_id,
                    chunk.seq,
                    chunk.kind.value,
                    chunk.content,
                    chunk.created_at,
                    chunk.parent_id,
                    chunk.source,
                    int(chunk.pinned),
                    chunk.token_estimate,
                    chunk.status.value if chunk.status else None,
                    json.dumps(chunk.meta, ensure_ascii=False, default=str),
                ),
            )
            self._conn.commit()
        return chunk

    def update_content(
        self, chunk_id: str, content: str, *, meta_extra: Mapping[str, Any] | None = None
    ) -> Chunk:
        chunk = self.get(chunk_id)
        if chunk is None:
            raise KeyError(f"unknown chunk {chunk_id}")
        meta = {**chunk.meta, **dict(meta_extra or {})}
        with self._lock:
            self._conn.execute(
                "UPDATE chunks SET content = ?, token_estimate = ?, meta = ? WHERE id = ?",
                (
                    content,
                    estimate_tokens(content),
                    json.dumps(meta, ensure_ascii=False, default=str),
                    chunk_id,
                ),
            )
            self._conn.commit()
        updated = self.get(chunk_id)
        assert updated is not None
        return updated

    def update_status(self, chunk_id: str, status: GoalStatus) -> None:
        with self._lock:
            self._conn.execute("UPDATE chunks SET status = ? WHERE id = ?", (status.value, chunk_id))
            self._conn.commit()

    # -- reads -------------------------------------------------------------

    def get(self, chunk_id: str) -> Chunk | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        return _row_to_chunk(row) if row else None

    def all(self) -> list[Chunk]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM chunks WHERE session_id = ? ORDER BY seq", (self.session_id,)
            ).fetchall()
        return [_row_to_chunk(row) for row in rows]

    def by_kind(self, kind: ChunkKind) -> list[Chunk]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM chunks WHERE session_id = ? AND kind = ? ORDER BY seq",
                (self.session_id, kind.value),
            ).fetchall()
        return [_row_to_chunk(row) for row in rows]

    def children(self, parent_id: str) -> list[Chunk]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM chunks WHERE parent_id = ? ORDER BY seq", (parent_id,)
            ).fetchall()
        return [_row_to_chunk(row) for row in rows]

    def recent(self, limit: int, *, kinds: Iterable[ChunkKind] | None = None) -> list[Chunk]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM chunks WHERE session_id = ? ORDER BY seq DESC LIMIT ?",
                (self.session_id, limit),
            ).fetchall()
        chunks = [_row_to_chunk(row) for row in rows]
        if kinds is not None:
            allowed = {kind.value for kind in kinds}
            chunks = [chunk for chunk in chunks if chunk.kind.value in allowed]
        return list(reversed(chunks))

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE session_id = ?", (self.session_id,)
            ).fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        id=row["id"],
        session_id=row["session_id"],
        seq=int(row["seq"]),
        kind=ChunkKind(row["kind"]),
        content=row["content"],
        created_at=float(row["created_at"]),
        parent_id=row["parent_id"],
        source=row["source"],
        pinned=bool(row["pinned"]),
        token_estimate=int(row["token_estimate"]),
        status=GoalStatus(row["status"]) if row["status"] else None,
        meta=json.loads(row["meta"]) if row["meta"] else {},
    )
