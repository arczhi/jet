"""Tool registry and catalog rendering.

The registry is the single source of truth for what jet can do. The catalog is
rendered in two sizes: compact snippets for the system prompt (always cheap) and
full JSON schemas injected only for tools a judgment pass selects.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from jet.core.types import ToolSpec
from jet.tools.base import Tool


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()):
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        name = tool.spec.name
        if name in self._tools:
            raise ValueError(f"duplicate tool name: {name}")
        self._tools[name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def require(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"unknown tool: {name}")
        return tool

    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def by_names(self, names: Sequence[str]) -> list[Tool]:
        return [self._tools[name] for name in names if name in self._tools]

    def snippets(self) -> str:
        lines = ["Available actions (request one by name; details load on demand):"]
        for tool in self._tools.values():
            flags = "read-only" if tool.spec.read_only else "write"
            lines.append(f"- {tool.spec.name} ({flags}): {tool.spec.description}")
        return "\n".join(lines)
