"""Session tracing.

Every judgment, LLM turn, tool call, and policy decision is appended to a JSONL
file under ``<home>/sessions/<session>/trace.jsonl``. One line per event, flushed
immediately so a crashed session is still inspectable. Spans record duration and
status so slow paths are visible without a profiler.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from jet.core.text import stable_hash

_REDACT_KEYS = {"api_key", "authorization", "key", "secret", "password", "token"}


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: ("***" if k.lower() in _REDACT_KEYS else _redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def cost_usd(usage_input: int, usage_output: int, input_price: float, output_price: float) -> float:
    return (usage_input * input_price + usage_output * output_price) / 1_000_000


class Trace:
    """Append-only JSONL trace writer. Safe to share across threads/tasks."""

    def __init__(self, path: Path, session_id: str, enabled: bool = True):
        self.path = path
        self.session_id = session_id
        self.enabled = enabled
        self._lock = threading.Lock()
        if enabled:
            path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, kind: str, **fields: Any) -> None:
        if not self.enabled:
            return
        record = {
            "ts": time.time(),
            "session_id": self.session_id,
            "event": kind,
            **_redact(fields),
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    @contextmanager
    def span(self, kind: str, **fields: Any) -> Iterator[dict[str, Any]]:
        """Record ``kind.start`` / ``kind.end`` with duration and outcome."""
        started = time.monotonic()
        extra: dict[str, Any] = {}
        error: str | None = None
        try:
            yield extra
        except Exception as exc:  # re-raised; recorded for diagnosis
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            duration_ms = int((time.monotonic() - started) * 1000)
            payload = {**fields, **extra, "duration_ms": duration_ms}
            if error is not None:
                payload["error"] = error
            self.event(kind, **payload)


def cache_key(*parts: Any) -> str:
    return stable_hash(*parts)
