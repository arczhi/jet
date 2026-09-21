"""jet command-line entrypoint.

Thin by design: parse options into Settings, build an Agent, hand it to the
interactive client or a single-shot runner. All behavior lives in the engine.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from jet.config import Settings, load_settings
from jet.ids import new_id
from jet.ui.render import EventRenderer

app = typer.Typer(add_completion=False, help="jet — a TypeSafe-native coding agent")
console = Console()


@dataclass
class CliContext:
    settings: Settings
    session_id: str | None
    verbose: bool


_CONTEXT: CliContext | None = None


def _context() -> CliContext:
    if _CONTEXT is None:
        raise RuntimeError("CLI context has not been initialized")
    return _CONTEXT


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    workspace: Path = typer.Option(Path("."), "--workspace", "-w", help="Workspace root"),
    judge: str = typer.Option(None, "--judge", help="Judge provider: typesafe|openai_compat|mock"),
    approval: str = typer.Option(None, "--approval", help="Approval mode: ask|auto|deny"),
    session: str = typer.Option(None, "--session", "-s", help="Resume an existing session id"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show steps, chunks, token usage"),
) -> None:
    global _CONTEXT
    settings = load_settings(
        workspace=workspace,
        judge_provider=judge,
        approval_mode=approval,
    )
    _CONTEXT = CliContext(settings=settings, session_id=session, verbose=verbose)
    ctx.obj = _CONTEXT
    if ctx.invoked_subcommand is None:
        _start_repl(_CONTEXT)


def main() -> None:
    app()


def _build_agent(context: CliContext) -> Any:
    from jet.agent.session import create_agent

    return create_agent(context.settings, session_id=context.session_id)


def _start_repl(context: CliContext) -> None:
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_repl(context))


async def _repl(context: CliContext) -> None:
    from jet.ui.repl import Repl

    agent = _build_agent(context)
    repl = Repl(
        agent,
        console,
        history_path=context.settings.home / "history",
        verbose=context.verbose,
    )
    agent.emit = repl.renderer
    agent.approve = repl.approve
    try:
        await repl.run()
    finally:
        await agent.aclose()


@app.command()
def run(
    task: str = typer.Argument(..., help="Task for the agent"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto-approve non-denied actions"),
) -> None:
    """Run one task non-interactively. Without --yes, approvals are denied."""
    ctx = _context()
    if yes:
        ctx.settings = ctx.settings.model_copy(update={"approval_mode": "auto"})
    exit_code = asyncio.run(_run_task(ctx, task))
    raise typer.Exit(code=exit_code)


@app.command("app")
def app_client(
    port: int = typer.Option(8765, "--port", help="Preferred local port"),
    browser: bool = typer.Option(False, "--browser", help="Use the default browser instead of a window"),
) -> None:
    """Open the native macOS client (local service + window)."""
    ctx = _context()
    from jet.desktop import run_desktop

    raise typer.Exit(code=run_desktop(ctx.settings, port=port, open_browser=browser))


async def _run_task(context: CliContext, task: str) -> int:
    agent = _build_agent(context)
    agent.emit = EventRenderer(console, verbose=context.verbose, llm=agent.llm)
    try:
        result = await agent.run_turn(task)
    except Exception as exc:  # noqa: BLE001 - CLI boundary translates errors to exit codes
        console.print(f"[red]{type(exc).__name__}: {exc}[/red]")
        return 1
    finally:
        await agent.aclose()
    if result.text and not result.verification:
        console.print(result.text)
    return 0 if result.stopped_reason == "done" else 2


@app.command()
def doctor(
    offline: bool = typer.Option(False, "--offline", help="Skip provider network probes"),
) -> None:
    """Verify configuration, directories, and provider reachability."""
    ctx = _context()
    exit_code = asyncio.run(_doctor(ctx.settings, offline=offline))
    raise typer.Exit(code=exit_code)


async def _doctor(settings: Settings, *, offline: bool) -> int:
    failures = 0
    console.print("[bold]config[/bold]")
    for key, value in settings.redacted().items():
        console.print(f"  {key} = {value}")
    console.print("\n[bold]checks[/bold]")
    if not settings.workspace.is_dir():
        console.print(f"  [red]✗ workspace missing: {settings.workspace}[/red]")
        failures += 1
    else:
        console.print(f"  ✓ workspace {settings.workspace}")
        memory = settings.workspace / "AGENTS.md"
        console.print(
            f"  {'✓' if memory.is_file() else '·'} AGENTS.md {'found' if memory.is_file() else 'absent'}"
        )
    for directory in (settings.home, settings.session_dir, settings.cache_dir):
        try:
            directory.mkdir(parents=True, exist_ok=True)
            console.print(f"  ✓ {directory}")
        except OSError as exc:
            console.print(f"  [red]✗ cannot create {directory}: {exc}[/red]")
            failures += 1

    if offline:
        console.print("  · provider probes skipped (--offline)")
        return 1 if failures else 0

    import time

    from jet.providers.registry import build_judge

    probe_session = new_id("doctor")

    try:
        judge = build_judge(settings, session_id=probe_session)
        started = time.monotonic()
        probability = await judge.noul("2 + 2 equals 4.", "Is the statement true?")
        latency = (time.monotonic() - started) * 1000
        console.print(
            f"  ✓ judge {settings.judge_provider} ({judge.name}) "
            f"answered p={probability:.2f} in {latency:.0f}ms"
        )
        await judge.aclose()
    except Exception as exc:  # noqa: BLE001 - doctor reports, never crashes
        console.print(f"  [red]✗ judge {settings.judge_provider}: {exc}[/red]")
        failures += 1

    probe_profiles = [settings.llm_profile]
    verifier_profile = settings.verifier_llm_profile
    if verifier_profile and verifier_profile not in probe_profiles:
        probe_profiles.append(verifier_profile)
    for name in probe_profiles:
        failures += await _probe_llm(settings, name, probe_session)
    return 1 if failures else 0


async def _probe_llm(settings: Settings, profile_name: str, session_id: str) -> int:
    import time

    from jet.core.types import Message
    from jet.providers.registry import build_llm_provider

    try:
        profile = settings.llm_profile_named(profile_name)
    except Exception as exc:  # noqa: BLE001 - doctor reports, never crashes
        console.print(f"  [red]✗ llm profile {profile_name}: {exc}[/red]")
        return 1
    if profile.model == "mock":
        console.print(f"  · llm profile {profile_name} is mock (offline)")
        return 0
    try:
        llm = build_llm_provider(settings, session_id=session_id, profile_name=profile_name)
        started = time.monotonic()
        response = await llm.complete([Message.user("Reply with the single word: ok")], max_tokens=256)
        latency = (time.monotonic() - started) * 1000
        text = response.text.strip()
        if text:
            console.print(
                f"  ✓ llm [{profile_name}] {profile.model} replied "
                f"{text[:20]!r} in {latency:.0f}ms ({profile.base_url})"
            )
            await llm.aclose()
            return 0
        if response.reasoning.strip():
            console.print(
                f"  ✓ llm [{profile_name}] {profile.model} is a reasoning model "
                f"(reasoning output, no final text at 256 tokens) in {latency:.0f}ms "
                f"({profile.base_url})"
            )
            await llm.aclose()
            return 0
        console.print(
            f"  [red]✗ llm [{profile_name}] {profile.model} returned an empty completion "
            f"in {latency:.0f}ms[/red]"
        )
        await llm.aclose()
        return 1
    except Exception as exc:  # noqa: BLE001 - doctor reports, never crashes
        console.print(f"  [red]✗ llm [{profile_name}] {profile.model}: {exc}[/red]")
        return 1


@app.command()
def sessions() -> None:
    """List local sessions."""
    ctx = _context()
    root = ctx.settings.session_dir
    if not root.is_dir():
        console.print("[dim]no sessions yet[/dim]")
        return
    entries = sorted(root.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True)
    for entry in entries[:50]:
        db = entry / "chunks.db"
        count = "?"
        if db.is_file():
            try:
                conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
                row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
                count = str(row[0]) if row else "0"
                conn.close()
            except sqlite3.Error:
                count = "?"
        console.print(f"  {entry.name}  chunks={count}")


@app.command()
def trace(
    session_id: str = typer.Argument(..., help="Session id"),
    tail: int = typer.Option(40, "--tail", "-n", help="Number of trailing events"),
) -> None:
    """Print the tail of a session trace."""
    ctx = _context()
    path = ctx.settings.session_dir / session_id / "trace.jsonl"
    if not path.is_file():
        console.print(f"[red]no trace at {path}[/red]")
        raise typer.Exit(code=1)
    lines = path.read_text(encoding="utf-8").splitlines()[-tail:]
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            console.print(line)
            continue
        event = record.pop("event", "?")
        record.pop("ts", None)
        record.pop("session_id", None)
        console.print(f"[bold]{event}[/bold] {json.dumps(record, ensure_ascii=False)[:300]}")


@app.command()
def resume(session_id: str = typer.Argument(..., help="Session id")) -> None:
    """Resume a session interactively. Prior state is reloaded through RLCD."""
    ctx = _context()
    ctx.session_id = session_id
    _start_repl(ctx)
