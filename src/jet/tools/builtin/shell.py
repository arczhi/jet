"""Shell execution tool. Dangerous by definition; the policy layer governs it."""

from __future__ import annotations

import asyncio
from typing import Any

from jet.core.types import Danger, ToolSpec
from jet.errors import ToolExecutionError
from jet.tools.base import Tool, ToolContext, ToolResult, int_argument

DEFAULT_TIMEOUT_S = 60
MAX_TIMEOUT_S = 600


class RunCommandTool(Tool):
    spec = ToolSpec(
        name="run_command",
        description="Run a shell command in the workspace and return its output and exit code.",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "timeout_s": {"type": "integer", "description": "Timeout in seconds (default 60)"},
            },
            "required": ["command"],
        },
        read_only=False,
        danger=Danger.DANGEROUS,
    )

    async def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        command = self.required(arguments, "command", str)
        timeout_s = int_argument(arguments, "timeout_s", DEFAULT_TIMEOUT_S, minimum=1)
        if timeout_s > MAX_TIMEOUT_S:
            raise ToolExecutionError(f"run_command: timeout_s must be <= {MAX_TIMEOUT_S}")
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(ctx.workspace),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except TimeoutError as exc:
            proc.kill()
            await proc.wait()
            raise ToolExecutionError(
                f"run_command: command exceeded {timeout_s}s and was killed: {command}"
            ) from exc
        output = stdout.decode("utf-8", "replace")
        ok = proc.returncode == 0
        header = f"$ {command}\nexit {proc.returncode}"
        return ToolResult(
            output=self.clip(f"{header}\n{output}", ctx),
            ok=ok,
            meta={"command": command, "exit_code": proc.returncode},
        )
