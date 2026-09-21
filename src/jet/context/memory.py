"""Conditional project memory (AGENTS.md and friends).

Memory files are pinned chunks: they are always in context, never compacted away.
They are also *conditional* — a nested file only enters context when the session
actually touches that directory. This is the native version of "load the
frontend style guide only when frontend work happens".
"""

from __future__ import annotations

from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path

from jet.context.store import ChunkStore
from jet.core.types import ChunkKind

DEFAULT_MEMORY_FILES = ("AGENTS.md", "JET.md", ".jet/AGENTS.md")
MAX_MEMORY_BYTES = 64_000


class MemoryLoader:
    def __init__(
        self, workspace: Path, store: ChunkStore, *, filenames: Iterable[str] = DEFAULT_MEMORY_FILES
    ):
        self.workspace = workspace.resolve()
        self.store = store
        self.filenames = tuple(filenames)
        self._loaded: set[Path] = set()

    def _hashes(self) -> set[str]:
        return {
            str(chunk.meta.get("sha"))
            for chunk in self.store.by_kind(ChunkKind.MEMORY)
            if chunk.meta.get("sha")
        }

    def load_root(self) -> list[str]:
        """Load memory files at the workspace root. Returns loaded paths."""
        return self._load_files(self._candidates(self.workspace))

    def load_for_path(self, path: Path) -> list[str]:
        """Load the nearest not-yet-loaded memory file for a touched path."""
        target = path if path.is_absolute() else (self.workspace / path)
        target = target.resolve()
        if not target.is_relative_to(self.workspace):
            return []
        directory = target if target.is_dir() else target.parent
        while True:
            loaded = self._load_files(self._candidates(directory))
            if loaded:
                return loaded
            if directory == self.workspace:
                return []
            directory = directory.parent

    def _candidates(self, directory: Path) -> list[Path]:
        return [directory / name for name in self.filenames]

    def _load_files(self, candidates: list[Path]) -> list[str]:
        loaded: list[str] = []
        known = self._hashes()
        for path in candidates:
            resolved = path.resolve()
            if resolved in self._loaded or not resolved.is_file():
                continue
            if not resolved.is_relative_to(self.workspace):
                continue
            try:
                raw = resolved.read_bytes()[:MAX_MEMORY_BYTES]
            except OSError:
                continue
            text = raw.decode("utf-8", "replace").strip()
            if not text:
                continue
            digest = sha256(text.encode()).hexdigest()
            self._loaded.add(resolved)
            if digest in known:
                continue
            relative = resolved.relative_to(self.workspace)
            self.store.add(
                ChunkKind.MEMORY,
                text,
                source=str(relative),
                pinned=True,
                meta={
                    "sha": digest,
                    "scope": "." if resolved.parent == self.workspace else str(relative.parent),
                    "origin": "memory_file",
                },
            )
            loaded.append(str(relative))
        return loaded
