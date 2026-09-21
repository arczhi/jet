"""Tool contract and shared tool context.

A tool is a named, schema-described action with an explicit danger level. Tools
never decide whether they may run — that is the policy layer's job.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jet.context.memory import MemoryLoader
from jet.context.store import ChunkStore
from jet.core.types import ToolSpec
from jet.errors import ToolExecutionError


@dataclass
class ToolContext:
    workspace: Path
    store: ChunkStore
    memory: MemoryLoader
    max_output_bytes: int = 48_000
    prefer_rg: bool = True


@dataclass
class ToolResult:
    output: str
    ok: bool = True
    meta: dict[str, Any] = field(default_factory=dict)


class Tool(ABC):
    spec: ToolSpec

    @abstractmethod
    async def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult: ...

    def clip(self, text: str, ctx: ToolContext) -> str:
        if len(text) <= ctx.max_output_bytes:
            return text
        kept = ctx.max_output_bytes
        return text[:kept] + f"\n…[output truncated: {len(text) - kept} of {len(text)} chars omitted]"

    def required(self, arguments: dict[str, Any], key: str, kind: type) -> Any:
        if key not in arguments:
            raise ToolExecutionError(f"{self.spec.name}: missing required argument {key!r}")
        value = arguments[key]
        if not isinstance(value, kind):
            raise ToolExecutionError(
                f"{self.spec.name}: argument {key!r} must be {kind.__name__}, got {type(value).__name__}"
            )
        return value


def resolve_in_workspace(workspace: Path, raw: str) -> Path:
    """Resolve a user/model-supplied path and refuse to escape the workspace."""
    candidate = Path(raw)
    resolved = candidate.resolve() if candidate.is_absolute() else (workspace / candidate).resolve()
    workspace_resolved = workspace.resolve()
    if resolved != workspace_resolved and not resolved.is_relative_to(workspace_resolved):
        raise ToolExecutionError(f"path {raw!r} is outside the workspace ({workspace_resolved})")
    return resolved


def relative_display(workspace: Path, path: Path) -> str:
    try:
        return str(path.relative_to(workspace.resolve()))
    except ValueError:
        return str(path)


def int_argument(arguments: dict[str, Any], key: str, default: int, *, minimum: int = 1) -> int:
    """Read an integer argument, distinguishing "absent" from a falsy 0.

    ``arguments.get(key) or default`` would silently turn an explicit 0 into the
    default, hiding a bad call. This helper rejects it instead.
    """
    raw = arguments.get(key)
    if raw is None:
        return default
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ToolExecutionError(f"{key} must be an integer, got {type(raw).__name__}")
    if raw < minimum:
        raise ToolExecutionError(f"{key} must be >= {minimum}, got {raw}")
    return raw
