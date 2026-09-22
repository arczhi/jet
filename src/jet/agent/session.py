"""Session state and the composition root.

``create_agent`` is the only place concrete providers, tools, and policies are
wired together. Everything else receives its dependencies, which keeps the loop
testable with mocks and makes provider swaps a config change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jet.config import Settings
from jet.context.attention import MetaAttention
from jet.context.builder import ContextBuilder
from jet.context.decompose import Decomposer, Planner
from jet.context.memory import MemoryLoader
from jet.context.store import ChunkStore
from jet.context.summarizer import LLMSummarizer, Summarizer
from jet.core.types import ApprovalMode, Message, Usage
from jet.ids import new_id
from jet.policy.permissions import PermissionPolicy
from jet.providers.base import LLMProvider
from jet.providers.judge import Judge
from jet.providers.registry import build_judge, build_llm_provider
from jet.tools.builtin import (
    EditFileTool,
    GlobTool,
    GrepTool,
    ListFilesTool,
    ReadFileTool,
    RunCommandTool,
    WriteFileTool,
)
from jet.tools.registry import ToolRegistry
from jet.tools.selection import ToolSelector
from jet.tracing import Trace


@dataclass
class SessionState:
    session_id: str
    store: ChunkStore
    trace: Trace
    turn_messages: list[Message] = field(default_factory=list)
    total_usage: Usage = field(default_factory=Usage)
    total_cost_usd: float = 0.0

    @property
    def session_dir(self) -> Path:
        return Path(self.store.path).parent


def default_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ReadFileTool(),
            ListFilesTool(),
            GlobTool(),
            GrepTool(),
            WriteFileTool(),
            EditFileTool(),
            RunCommandTool(),
        ]
    )


def create_agent(
    settings: Settings,
    *,
    session_id: str | None = None,
    emit: Any = None,
    approve: Any = None,
    judge: Judge | None = None,
    llm: LLMProvider | None = None,
    summarizer: Summarizer | None = None,
    registry: ToolRegistry | None = None,
    decomposer: Planner | None = None,
) -> Any:
    """Build an Agent with all dependencies wired. Returns ``jet.agent.loop.Agent``."""
    from jet.agent.loop import Agent
    from jet.agent.verify import Verifier

    resolved_session = session_id or new_id("ses")
    session_dir = settings.session_dir / resolved_session
    session_dir.mkdir(parents=True, exist_ok=True)
    store = ChunkStore(session_dir / "chunks.db", resolved_session)
    trace = Trace(session_dir / "trace.jsonl", resolved_session, enabled=settings.trace_enabled)

    resolved_judge = judge or build_judge(settings, trace, session_id=resolved_session)
    if resolved_judge.trace is None:
        resolved_judge.trace = trace
    resolved_llm = llm or build_llm_provider(settings, session_id=resolved_session)
    verifier_llm: LLMProvider | None = None
    if llm is None and settings.verifier_llm_profile:
        candidate = build_llm_provider(
            settings,
            session_id=resolved_session,
            profile_name=settings.verifier_llm_profile,
        )
        if candidate.model != resolved_llm.model:
            verifier_llm = candidate
    # Summaries sit on the hot path of context assembly: use the fast verifier
    # endpoint, never the reasoning generator.
    resolved_summarizer = summarizer or LLMSummarizer(verifier_llm or resolved_llm, trace=trace)
    attention = MetaAttention(
        resolved_judge,
        batch_size=settings.attention_batch_size,
        full_threshold=settings.attention_full_threshold,
        long_threshold=settings.attention_long_threshold,
        short_threshold=settings.attention_short_threshold,
        summarizer=resolved_summarizer,
        small_pool=settings.attention_small_pool,
    )
    builder = ContextBuilder(attention, budget_tokens=settings.context_budget_tokens)
    memory = MemoryLoader(settings.workspace, store)
    resolved_decomposer: Planner = decomposer or Decomposer(
        llm=resolved_llm,
        judge=resolved_judge,
        store=store,
        trace=trace,
        max_depth=settings.decompose_max_depth,
        max_total_subgoals=settings.decompose_max_subgoals,
    )
    policy = PermissionPolicy.from_config(
        settings.permission_rules,
        mode=ApprovalMode(settings.approval_mode),
        judge=resolved_judge,
        judge_enabled=settings.permission_judge,
        trace=trace,
    )
    verifier = Verifier(
        judge=resolved_judge,
        llm=verifier_llm or resolved_llm,
        mode=settings.verifier_provider,
        threshold=settings.verifier_threshold,
        trace=trace,
    )
    selector = ToolSelector(
        resolved_judge,
        top_k=settings.tool_top_k,
        trace=trace,
    )
    resolved_registry = registry or default_tool_registry()
    return Agent(
        settings=settings,
        state=SessionState(resolved_session, store, trace),
        judge=resolved_judge,
        llm=resolved_llm,
        verifier_llm=verifier_llm,
        registry=resolved_registry,
        selector=selector,
        policy=policy,
        builder=builder,
        decomposer=resolved_decomposer,
        verifier=verifier,
        memory=memory,
        emit=emit,
        approve=approve,
    )
