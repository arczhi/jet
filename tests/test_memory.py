"""Conditional project memory loading."""

from __future__ import annotations

from pathlib import Path

from jet.context.memory import MemoryLoader
from jet.context.store import ChunkStore
from jet.core.types import ChunkKind


def test_root_memory_is_loaded_and_pinned(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("rule one")
    store = ChunkStore(tmp_path / "chunks.db", "ses")
    loader = MemoryLoader(workspace, store)
    loaded = loader.load_root()
    assert loaded == ["AGENTS.md"]
    chunks = store.by_kind(ChunkKind.MEMORY)
    assert len(chunks) == 1
    assert chunks[0].pinned is True
    assert chunks[0].meta["scope"] == "."
    store.close()


def test_memory_is_not_loaded_twice(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("rule one")
    store = ChunkStore(tmp_path / "chunks.db", "ses")
    loader = MemoryLoader(workspace, store)
    loader.load_root()
    assert loader.load_root() == []
    assert len(store.by_kind(ChunkKind.MEMORY)) == 1
    store.close()


def test_nested_memory_loads_on_touch(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    (workspace / "src").mkdir(parents=True)
    (workspace / "AGENTS.md").write_text("root rule")
    (workspace / "src" / "AGENTS.md").write_text("src rule")
    (workspace / "src" / "mod.py").write_text("x = 1")
    store = ChunkStore(tmp_path / "chunks.db", "ses")
    loader = MemoryLoader(workspace, store)
    loader.load_root()
    loaded = loader.load_for_path(Path("src/mod.py"))
    assert loaded == ["src/AGENTS.md"]
    scopes = {chunk.meta["scope"] for chunk in store.by_kind(ChunkKind.MEMORY)}
    assert scopes == {".", "src"}
    store.close()


def test_touch_outside_workspace_loads_nothing(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    store = ChunkStore(tmp_path / "chunks.db", "ses")
    loader = MemoryLoader(workspace, store)
    assert loader.load_for_path(Path("/etc/hosts")) == []
    store.close()


def test_empty_and_duplicate_content_is_skipped(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("")
    (workspace / "JET.md").write_text("same text")
    (workspace / "src").mkdir()
    (workspace / "src" / "AGENTS.md").write_text("same text")
    store = ChunkStore(tmp_path / "chunks.db", "ses")
    loader = MemoryLoader(workspace, store)
    loaded = loader.load_root()
    assert loaded == ["JET.md"]
    nested = loader.load_for_path(Path("src"))
    assert nested == []
    store.close()
