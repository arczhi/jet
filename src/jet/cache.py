"""Small durable key/value cache backed by SQLite.

Used for judgment results: identical state+questions+model must not be paid for
twice. Values are stored as JSON so they stay inspectable with any sqlite tool.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    key TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    value TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entries_namespace ON entries(namespace);
"""


class SqliteCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM entries WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        value = json.loads(row[0])
        return value if isinstance(value, dict) else None

    def set(self, key: str, value: dict[str, Any], *, namespace: str) -> None:
        payload = json.dumps(value, ensure_ascii=False, default=str)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO entries (key, namespace, value, created_at) VALUES (?, ?, ?, ?)",
                (key, namespace, payload, time.time()),
            )
            self._conn.commit()

    def count(self, namespace: str | None = None) -> int:
        with self._lock:
            if namespace is None:
                row = self._conn.execute("SELECT COUNT(*) FROM entries").fetchone()
            else:
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM entries WHERE namespace = ?", (namespace,)
                ).fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()
