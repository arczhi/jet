"""Search tools: regex content search and globbing.

``grep`` prefers ripgrep when installed and falls back to a deterministic Python
scan. Both paths emit the same ``path:line:text`` format so the model sees one
contract regardless of the host.
"""

from __future__ import annotations

import asyncio
import fnmatch
import re
import shutil
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

SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", "dist", "build", ".mypy_cache", ".ruff_cache"}
MAX_FILE_BYTES = 2_000_000


class GrepTool(Tool):
    spec = ToolSpec(
        name="grep",
        description="Search file contents by regular expression; returns path:line:text.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Python/RE2 regular expression"},
                "path": {"type": "string", "description": "File or directory (default '.')"},
                "glob": {"type": "string", "description": "Only search files matching this glob"},
                "ignore_case": {"type": "boolean", "description": "Case-insensitive search"},
                "max_results": {"type": "integer", "description": "Maximum matches (default 200)"},
            },
            "required": ["pattern"],
        },
        read_only=True,
        danger=Danger.SAFE,
    )

    async def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        pattern = self.required(arguments, "pattern", str)
        raw_path = str(arguments.get("path", ".") or ".")
        glob = arguments.get("glob")
        if glob is not None and not isinstance(glob, str):
            raise ToolExecutionError("grep: glob must be a string")
        ignore_case = bool(arguments.get("ignore_case", False))
        max_results = int_argument(arguments, "max_results", 200)
        try:
            compiled = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as exc:
            raise ToolExecutionError(f"grep: invalid regex: {exc}") from exc
        target = resolve_in_workspace(ctx.workspace, raw_path)
        if not target.exists():
            raise ToolExecutionError(f"grep: path does not exist: {raw_path}")

        rg = shutil.which("rg") if ctx.prefer_rg else None
        if rg:
            matches, used = await self._rg(rg, pattern, target, glob, ignore_case, max_results, ctx)
        else:
            matches, used = self._python(compiled, target, glob, max_results, ctx)
        header = (
            f"# grep {pattern!r} in {relative_display(ctx.workspace, target) or '.'} ({len(matches)} matches"
        )
        header += ", ripgrep)" if used else ")"
        body = "\n".join(matches) if matches else "(no matches)"
        return ToolResult(
            output=self.clip(f"{header}\n{body}", ctx),
            meta={"pattern": pattern, "matches": len(matches), "engine": "rg" if used else "python"},
        )

    async def _rg(
        self,
        rg: str,
        pattern: str,
        target: Path,
        glob: str | None,
        ignore_case: bool,
        max_results: int,
        ctx: ToolContext,
    ) -> tuple[list[str], bool]:
        args = [rg, "--line-number", "--no-heading", "--color=never", "--max-count", str(max_results)]
        if ignore_case:
            args.append("--ignore-case")
        if glob:
            args.extend(["--glob", glob])
        args.extend(["--", pattern, "."])
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=str(target if target.is_dir() else target.parent),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
        except TimeoutError as exc:
            proc.kill()
            raise ToolExecutionError("grep: ripgrep timed out after 60s") from exc
        lines = stdout.decode("utf-8", "replace").splitlines()[:max_results]
        if proc.returncode not in (0, 1):
            raise ToolExecutionError(f"grep: ripgrep failed with exit code {proc.returncode}")
        if not target.is_dir():
            prefix = relative_display(ctx.workspace, target)
            lines = [line.replace("./", prefix + ":", 1) if line.startswith("./") else line for line in lines]
        return lines, True

    def _python(
        self,
        compiled: re.Pattern[str],
        target: Path,
        glob: str | None,
        max_results: int,
        ctx: ToolContext,
    ) -> tuple[list[str], bool]:
        files = [target] if target.is_file() else list(self._iter_files(target, glob))
        matches: list[str] = []
        for file in files:
            if len(matches) >= max_results:
                break
            try:
                if file.stat().st_size > MAX_FILE_BYTES:
                    continue
                raw = file.read_bytes()
            except OSError:
                continue
            if b"\x00" in raw[:1024]:
                continue
            text = raw.decode("utf-8", "replace")
            for number, line in enumerate(text.splitlines(), start=1):
                if compiled.search(line):
                    matches.append(f"{relative_display(ctx.workspace, file)}:{number}:{line}")
                    if len(matches) >= max_results:
                        break
        return matches, False

    def _iter_files(self, root: Path, glob: str | None) -> list[Path]:
        found: list[Path] = []
        for path in root.rglob(glob or "*"):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if path.is_file() and (
                glob is None or fnmatch.fnmatch(path.name, glob) or fnmatch.fnmatch(str(path), glob)
            ):
                found.append(path)
        return found


class GlobTool(Tool):
    spec = ToolSpec(
        name="glob",
        description="Find files by glob pattern (e.g. 'src/**/*.py').",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern relative to the workspace"},
                "max_results": {"type": "integer", "description": "Maximum paths (default 200)"},
            },
            "required": ["pattern"],
        },
        read_only=True,
        danger=Danger.SAFE,
    )

    async def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        pattern = self.required(arguments, "pattern", str)
        max_results = int_argument(arguments, "max_results", 200)
        if Path(pattern).is_absolute():
            raise ToolExecutionError("glob: pattern must be relative to the workspace")
        found: list[str] = []
        for path in sorted(ctx.workspace.glob(pattern)):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            found.append(relative_display(ctx.workspace, path) + ("/" if path.is_dir() else ""))
            if len(found) >= max_results:
                break
        header = f"# glob {pattern!r} ({len(found)} paths)"
        body = "\n".join(found) if found else "(no paths)"
        return ToolResult(
            output=self.clip(f"{header}\n{body}", ctx), meta={"pattern": pattern, "count": len(found)}
        )
