"""Interactive terminal client (macOS-first).

One asyncio event loop for the whole session, so provider HTTP connections stay
warm. Input is read on a worker thread via prompt_toolkit; agent events render on
the main loop. Approval prompts are explicit and fail closed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.prompt import Prompt

from jet.agent.loop import Agent
from jet.core.events import ApprovalRequested, Notice
from jet.core.types import PolicyVerdict, ToolCall
from jet.ui.render import EventRenderer, approval_panel


class Repl:
    def __init__(self, agent: Agent, console: Console, *, history_path: Path, verbose: bool = False):
        self.agent = agent
        self.console = console
        self.verbose = verbose
        self.renderer = EventRenderer(console, verbose=verbose, llm=agent.llm)
        self.session: PromptSession[str] = PromptSession(history=FileHistory(str(history_path)))
        self._auto = False

    async def run(self) -> None:
        self._banner()
        while True:
            try:
                task = await asyncio.to_thread(self._read_input)
            except (EOFError, KeyboardInterrupt):
                self.console.print("[dim]bye[/dim]")
                return
            task = task.strip()
            if not task:
                continue
            if await self._handle_command(task):
                continue
            try:
                await self.agent.run_turn(task)
            except KeyboardInterrupt:
                self.console.print("[yellow]turn interrupted[/yellow]")
            except Exception as exc:  # noqa: BLE001 - keep the client alive, surface the error
                self.console.print(f"[red]{type(exc).__name__}: {exc}[/red]")
                self.agent.emit(Notice(level="error", text=f"{type(exc).__name__}: {exc}"))

    def _read_input(self) -> str:
        return str(self.session.prompt(HTML("<b>jet ›</b> ")))

    def _banner(self) -> None:
        self.console.print(
            f"[bold]jet[/bold] · session [cyan]{self.agent.session_id}[/cyan] · "
            f"workspace [cyan]{self.agent.settings.workspace}[/cyan]"
        )
        self.console.print(
            f"[dim]judge {self.agent.settings.judge_provider} · "
            f"llm {self.agent.settings.llm_model} · approvals {self._mode()} · /help for commands[/dim]\n"
        )

    def _mode(self) -> str:
        return "auto" if self._auto else self.agent.settings.approval_mode

    async def _handle_command(self, text: str) -> bool:
        if not text.startswith("/"):
            return False
        command, _, argument = text.partition(" ")
        command = command.lower()
        if command in ("/quit", "/exit", "/q"):
            raise EOFError
        if command == "/help":
            self.console.print(
                "/help · /session · /trace · /cost · /auto [on|off] · /quit\n"
                "Anything else is a task for the agent."
            )
            return True
        if command == "/session":
            self.console.print(f"session: [cyan]{self.agent.session_id}[/cyan]")
            self.console.print(f"chunks: {self.agent.store.count()}")
            return True
        if command == "/trace":
            self.console.print(f"trace: {self.agent.trace.path}")
            return True
        if command == "/cost":
            usage = self.agent.state.total_usage
            self.console.print(
                f"tokens: {usage.input_tokens} in / {usage.output_tokens} out ({usage.total_tokens} total)"
            )
            return True
        if command == "/auto":
            self._auto = argument.strip().lower() not in ("off", "false", "0")
            self.console.print(f"approvals: [cyan]{self._mode()}[/cyan]")
            return True
        self.console.print(f"[yellow]unknown command: {command}[/yellow]")
        return True

    async def approve(self, call: ToolCall, verdict: PolicyVerdict) -> bool:
        if self._auto:
            return True
        approval_panel(self.console, ApprovalRequested(call=call, verdict=verdict))
        answer = await asyncio.to_thread(
            Prompt.ask,
            "[yellow]allow?[/yellow]",
            choices=["y", "n"],
            default="n",
        )
        return answer == "y"
