"""Durable UI state (small JSON kv) under ``<home>/state.json``.

This is for client preferences that outlive sessions — e.g. the last opened
workspace directory. Session data belongs to the chunk store; this file only
holds a handful of keys and is rewritten wholesale on change.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def state_path(home: Any) -> Any:
    return Path(home) / "state.json"


def read_state(home: Any) -> dict[str, Any]:
    path = state_path(home)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_state(home: Any, data: dict[str, Any]) -> None:
    path = state_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def update_state(home: Any, **fields: Any) -> dict[str, Any]:
    data = {**read_state(home), **fields}
    write_state(home, data)
    return data
