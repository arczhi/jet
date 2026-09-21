"""File tools: read, write, edit, list. All paths are workspace-confined."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jet.core.types import Danger, ToolSpec
from jet.errors import ToolExecutionError
from jet.tools.base import (
    Tool,
    ToolContext,
    ToolResult,
    int_argument,
    relative_display,
    resolve_in_workspace,
)

MAX_READ_LINES = 4_000
MAX_LIST_ENTRIES = 800
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", "dist", "build", ".mypy_cache", ".ruff_cache"}


class ReadFileTool(Tool):
    spec = ToolSpec(
        name="read_file",
        description="Read a file in the workspace, optionally a line range.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace-relative file path"},
                "offset": {"type": "integer", "description": "1-based first line (optional)"},
                "limit": {"type": "integer", "description": "Maximum lines to return (optional)"},
            },
            "required": ["path"],
        },
        read_only=True,
        danger=Danger.SAFE,
    )

    async def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raw = self.required(arguments, "path", str)
        offset = int_argument(arguments, "offset", 1)
        limit = int_argument(arguments, "limit", MAX_READ_LINES)
        path = resolve_in_workspace(ctx.workspace, raw)
        if not path.is_file():
            raise ToolExecutionError(f"read_file: not a file: {raw}")
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ToolExecutionError(f"read_file: {exc}") from exc
        lines = text.splitlines()
        window = lines[offset - 1 : offset - 1 + limit]
        header = (
            f"# {relative_display(ctx.workspace, path)} "
            f"(lines {offset}-{offset + len(window) - 1} of {len(lines)})"
        )
        body = "\n".join(f"{offset + i:>6}\t{line}" for i, line in enumerate(window))
        return ToolResult(
            output=self.clip(f"{header}\n{body}", ctx),
            meta={"path": raw, "lines": len(lines), "bytes": len(text)},
        )


class WriteFileTool(Tool):
    spec = ToolSpec(
        name="write_file",
        description="Create or overwrite a file with full content.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace-relative file path"},
                "content": {"type": "string", "description": "Full file content"},
            },
            "required": ["path", "content"],
        },
        read_only=False,
        danger=Danger.MODERATE,
    )

    async def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raw = self.required(arguments, "path", str)
        content = self.required(arguments, "content", str)
        path = resolve_in_workspace(ctx.workspace, raw)
        existed = path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise ToolExecutionError(f"write_file: {exc}") from exc
        action = "overwrote" if existed else "created"
        lines = content.count("\n") + (1 if content else 0)
        return ToolResult(
            output=f"{action} {relative_display(ctx.workspace, path)} ({lines} lines, {len(content)} bytes)",
            meta={"path": raw, "existed": existed, "bytes": len(content)},
        )


class EditFileTool(Tool):
    spec = ToolSpec(
        name="edit_file",
        description="Replace an exact string in a file. Fails if the match is absent or ambiguous.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace-relative file path"},
                "old_string": {"type": "string", "description": "Exact text to replace"},
                "new_string": {"type": "string", "description": "Replacement text"},
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace every occurrence (default: only if unique)",
                },
            },
            "required": ["path", "old_string", "new_string"],
        },
        read_only=False,
        danger=Danger.MODERATE,
    )

    async def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raw = self.required(arguments, "path", str)
        old = self.required(arguments, "old_string", str)
        new = self.required(arguments, "new_string", str)
        replace_all = bool(arguments.get("replace_all", False))
        if old == new:
            raise ToolExecutionError("edit_file: old_string and new_string are identical")
        path = resolve_in_workspace(ctx.workspace, raw)
        if not path.is_file():
            raise ToolExecutionError(f"edit_file: not a file: {raw}")
        text = path.read_text(encoding="utf-8", errors="replace")
        occurrences = text.count(old)
        if occurrences == 0:
            raise ToolExecutionError(f"edit_file: old_string not found in {raw}")
        if occurrences > 1 and not replace_all:
            raise ToolExecutionError(
                f"edit_file: old_string occurs {occurrences} times in {raw}; "
                "add more context or set replace_all"
            )
        updated = text.replace(old, new, -1 if replace_all else 1)
        try:
            path.write_text(updated, encoding="utf-8")
        except OSError as exc:
            raise ToolExecutionError(f"edit_file: {exc}") from exc
        replaced = occurrences if replace_all else 1
        plural = "s" if replaced != 1 else ""
        return ToolResult(
            output=f"edited {relative_display(ctx.workspace, path)} ({replaced} replacement{plural})",
            meta={"path": raw, "replacements": replaced},
        )


class ListFilesTool(Tool):
    spec = ToolSpec(
        name="list_files",
        description="List files under a directory (workspace-relative), directory-first.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory to list (default '.')"},
                "depth": {"type": "integer", "description": "Recursion depth (default 2)"},
            },
        },
        read_only=True,
        danger=Danger.SAFE,
    )

    async def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raw = str(arguments.get("path", ".") or ".")
        depth = int_argument(arguments, "depth", 2)
        root = resolve_in_workspace(ctx.workspace, raw)
        if not root.is_dir():
            raise ToolExecutionError(f"list_files: not a directory: {raw}")
        entries: list[str] = []
        self._walk(root, root, depth, entries)
        entries.sort(key=lambda entry: (not entry.endswith("/"), entry))
        body = "\n".join(entries[:MAX_LIST_ENTRIES])
        if len(entries) > MAX_LIST_ENTRIES:
            body += f"\n…[{len(entries) - MAX_LIST_ENTRIES} more entries omitted]"
        header = f"# {relative_display(ctx.workspace, root) or '.'} ({len(entries)} entries)"
        return ToolResult(output=self.clip(f"{header}\n{body}", ctx), meta={"path": raw})

    def _walk(self, root: Path, directory: Path, depth: int, out: list[str]) -> None:
        if depth < 0:
            return
        try:
            children = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name))
        except OSError:
            return
        for child in children:
            if child.name in SKIP_DIRS or child.name.startswith("."):
                continue
            display = str(child.relative_to(root)) if child != root else child.name
            if child.is_dir():
                out.append(f"{display}/")
                self._walk(root, child, depth - 1, out)
            else:
                out.append(display)
        return
