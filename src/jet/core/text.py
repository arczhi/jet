"""Token estimation and text helpers.

Character-based estimation is deliberate: it is deterministic, free, and only
used for budget planning. Vendors report exact usage and jet accounts for that
exactly in traces.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    limit = max_tokens * CHARS_PER_TOKEN
    if len(text) <= limit:
        return text
    head = max(0, limit - 1)
    return text[:head] + "…"


def stable_hash(*parts: Any) -> str:
    """Hash JSON-serializable parts deterministically (used for judgment caches)."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(json.dumps(part, sort_keys=True, ensure_ascii=False, default=str).encode())
        digest.update(b"\x00")
    return digest.hexdigest()


def join_nonempty(parts: Iterable[str], sep: str = "\n") -> str:
    return sep.join(p for p in parts if p)
