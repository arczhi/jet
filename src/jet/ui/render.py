"""Event rendering for the terminal client.

Rendering is a pure function of agent events; the engine never prints. Streaming
text is written inline so responses appear as they are generated.
"""

from __future__ import annotations

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from jet.core.events import (
    ApprovalRequested,
    AssistantDelta,
    AssistantMessage,
    AssistantThinking,
    ChunkAdded,
    ContextBuilt,
    Notice,
    PlanReady,
    StepStarted,
    ToolFinished,
    ToolProposed,
    TurnFinished,
    TurnStarted,
    Verified,
)
from jet.tracing import cost_usd


class EventRenderer:
    def __init__(self, console: Console, *, verbose: bool = False, llm: object | None = None):
        self.console = console
        self.verbose = verbose
        self.llm = llm
        self._streaming = False

    def __call__(self, event: object) -> None:
        if isinstance(event, TurnStarted):
            self._end_stream()
            return
        if isinstance(event, StepStarted):
            if self.verbose:
                self.console.print(f"[dim]· step {event.step}[/dim]")
            return
        if isinstance(event, PlanReady):
            steps = " | ".join(event.subgoals[:6])
            suffix = " …" if len(event.subgoals) > 6 else ""
            self.console.print(f"[dim]· plan ({len(event.subgoals)}): {escape(steps)}{suffix}[/dim]")
            return
        if isinstance(event, ContextBuilt):
            counts = (event.views and _counts(event.views)) or ""
            self.console.print(
                f"[dim]· context {event.messages} msgs · {event.tokens / 1000:.1f}k tok"
                f"{counts} · {event.hidden_chunks} hidden[/dim]"
            )
            return
        if isinstance(event, AssistantDelta):
            if not self._streaming:
                self._streaming = True
            self.console.print(event.text, end="", soft_wrap=True)
            return
        if isinstance(event, AssistantThinking):
            if self.verbose:
                self.console.print(f"[dim italic]{escape(event.text)}[/dim italic]", end="")
            return
        if isinstance(event, AssistantMessage):
            self._end_stream()
            if self.verbose and event.response.usage.total_tokens:
                self.console.print(
                    f"[dim]  ({event.response.usage.input_tokens} in / "
                    f"{event.response.usage.output_tokens} out)[/dim]"
                )
            return
        if isinstance(event, ToolProposed):
            return
        if isinstance(event, ToolFinished):
            outcome = event.outcome
            args = _arg_hint(outcome.tool_call.arguments)
            mark = "[green]✓[/green]" if outcome.ok else "[red]✗[/red]"
            self.console.print(
                f"{mark} [bold]{outcome.tool_call.name}[/bold]({escape(args)}) "
                f"[dim]{outcome.duration_ms}ms[/dim]"
            )
            if not outcome.ok and outcome.output:
                self.console.print(f"[red]  {escape(outcome.output[:400])}[/red]")
            return
        if isinstance(event, ApprovalRequested):
            return
        if isinstance(event, ChunkAdded):
            if self.verbose:
                self.console.print(
                    f"[dim]  + chunk {event.chunk.kind.value} #{event.chunk.seq} "
                    f"({event.chunk.token_estimate} tok)[/dim]"
                )
            return
        if isinstance(event, Verified):
            verification = event.verification
            mark = "[green]verified[/green]" if verification.satisfied else "[yellow]unverified[/yellow]"
            self.console.print(f"[dim]· {mark} · {escape(verification.reason[:200])}[/dim]")
            return
        if isinstance(event, Notice):
            colors = {"info": "dim", "warning": "yellow", "error": "red"}
            self.console.print(f"[{colors[event.level]}]{escape(event.text)}[/{colors[event.level]}]")
            return
        if isinstance(event, TurnFinished):
            self._end_stream()
            result = event.result
            cost = cost_usd(
                result.usage.input_tokens,
                result.usage.output_tokens,
                float(getattr(self.llm, "input_price", 0.0)),
                float(getattr(self.llm, "output_price", 0.0)),
            )
            cost_note = f" · ${cost:.4f}" if cost else ""
            self.console.print(
                f"[dim]— {result.stopped_reason} · {result.steps} steps · "
                f"{result.tool_calls} tools · {result.usage.total_tokens} tok{cost_note}[/dim]"
            )
            self.console.print()
            return

    def _end_stream(self) -> None:
        if self._streaming:
            self.console.print()
            self._streaming = False


def _counts(views: list[dict[str, object]]) -> str:
    counts: dict[str, int] = {}
    for view in views:
        level = str(view.get("level", "?"))
        counts[level] = counts.get(level, 0) + 1
    parts = [f"{level} {count}" for level, count in sorted(counts.items())]
    return " (" + ", ".join(parts) + ")" if parts else ""


def _arg_hint(arguments: dict[str, object]) -> str:
    for key in ("command", "path", "pattern", "query"):
        value = arguments.get(key)
        if isinstance(value, str):
            return value if len(value) <= 90 else value[:87] + "…"
    return ", ".join(f"{key}={value}" for key, value in list(arguments.items())[:2])


def approval_panel(console: Console, event: ApprovalRequested) -> None:
    args = event.call.arguments
    body = Text()
    body.append(f"tool: {event.call.name}\n", style="bold")
    for key, value in args.items():
        rendered = str(value)
        if len(rendered) > 400:
            rendered = rendered[:400] + "…"
        body.append(f"{key}: {rendered}\n")
    body.append(f"reason: {event.verdict.reason}", style="dim")
    console.print(Panel(body, title="approval needed", border_style="yellow", expand=False))
