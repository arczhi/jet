"""Durable UI state file (last opened directory)."""

from __future__ import annotations

from pathlib import Path

from jet.state import read_state, state_path, update_state, write_state


def test_roundtrip(tmp_path: Path) -> None:
    update_state(tmp_path, last_workspace="/tmp/x")
    assert read_state(tmp_path) == {"last_workspace": "/tmp/x"}


def test_update_preserves_other_keys(tmp_path: Path) -> None:
    write_state(tmp_path, {"a": 1})
    update_state(tmp_path, b=2)
    assert read_state(tmp_path) == {"a": 1, "b": 2}


def test_missing_or_corrupt_file_reads_empty(tmp_path: Path) -> None:
    assert read_state(tmp_path) == {}
    state_path(tmp_path).write_text("not json")
    assert read_state(tmp_path) == {}
