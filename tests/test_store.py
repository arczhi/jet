"""Chunk store: durable, ordered, tree-structured session state."""

from __future__ import annotations

from pathlib import Path

from jet.context.store import ChunkStore
from jet.core.types import ChunkKind, GoalStatus


def test_add_assigns_increasing_seq_and_defaults(store: ChunkStore) -> None:
    first = store.add(ChunkKind.USER_MESSAGE, "hello")
    second = store.add(ChunkKind.TOOL_RESULT, "output", source="read_file")
    assert second.seq == first.seq + 1
    assert second.token_estimate > 0
    assert second.source == "read_file"
    assert store.count() == 2


def test_parent_child_links(store: ChunkStore) -> None:
    parent = store.add(ChunkKind.SUBGOAL, "root")
    child = store.add(ChunkKind.SUBGOAL, "child", parent_id=parent.id)
    grandchildren = store.children(parent.id)
    assert [chunk.id for chunk in grandchildren] == [child.id]
    assert store.get(child.id).parent_id == parent.id  # type: ignore[union-attr]


def test_update_content_recomputes_tokens(store: ChunkStore) -> None:
    chunk = store.add(ChunkKind.TOOL_RESULT, "short")
    updated = store.update_content(chunk.id, "much longer content " * 50, meta_extra={"summarized": True})
    assert updated.token_estimate > chunk.token_estimate
    assert updated.meta["summarized"] is True


def test_status_transitions(store: ChunkStore) -> None:
    chunk = store.add(ChunkKind.SUBGOAL, "do it", status=GoalStatus.PENDING)
    store.update_status(chunk.id, GoalStatus.RUNNING)
    assert store.get(chunk.id).status is GoalStatus.RUNNING  # type: ignore[union-attr]


def test_recent_respects_order_and_kind_filters(store: ChunkStore) -> None:
    store.add(ChunkKind.USER_MESSAGE, "one")
    store.add(ChunkKind.TOOL_RESULT, "two")
    store.add(ChunkKind.TOOL_RESULT, "three")
    recent = store.recent(2, kinds=[ChunkKind.TOOL_RESULT])
    assert [chunk.content for chunk in recent] == ["two", "three"]


def test_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "chunks.db"
    first = ChunkStore(path, "ses_persist")
    chunk = first.add(ChunkKind.USER_MESSAGE, "remember me")
    first.close()
    second = ChunkStore(path, "ses_persist")
    restored = second.get(chunk.id)
    assert restored is not None
    assert restored.content == "remember me"
    second.close()


def test_sessions_are_isolated(tmp_path: Path) -> None:
    path = tmp_path / "chunks.db"
    a = ChunkStore(path, "ses_a")
    a.add(ChunkKind.USER_MESSAGE, "a")
    a.close()
    b = ChunkStore(path, "ses_b")
    assert b.count() == 0
    b.close()
