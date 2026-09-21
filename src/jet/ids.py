"""Identifier helpers. IDs are time-sortable so logs and stores read chronologically."""

from __future__ import annotations

import secrets
import time


def new_id(prefix: str) -> str:
    """Return a sortable, prefixed id such as ``ses_m8x2k3_a1b2c3``."""
    stamp = f"{time.time_ns():x}"
    return f"{prefix}_{stamp}_{secrets.token_hex(3)}"
