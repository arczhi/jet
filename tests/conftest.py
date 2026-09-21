"""Shared fixtures. Tests are isolated from the developer's real jet config."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from jet.config import LLMProfile, Settings, load_settings
from jet.context.memory import MemoryLoader
from jet.context.store import ChunkStore
from jet.core.types import Message, SubGoal, Usage
from jet.providers.judge import Judge
from jet.providers.mock import MockJudgmentProvider, MockLLMProvider, MockTurn
from jet.providers.questions import (
    ChoiceAnswer,
    JudgmentResult,
    NoulAnswer,
    Question,
    ScoreAnswer,
)
from jet.tools.base import ToolContext
from jet.tracing import Trace


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Never read the developer's ~/.config/jet/config.toml, .env, or JET_* vars."""
    monkeypatch.setattr("jet.config.USER_CONFIG_PATH", tmp_path / "no-user-config.toml")
    monkeypatch.setenv("JET_CONFIG", str(tmp_path / "no-project-config.toml"))
    for key in list(os.environ):
        if key.startswith("JET_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "a.txt").write_text("alpha\nbeta\ngamma\n")
    (root / "src").mkdir()
    (root / "src" / "mod.py").write_text("def add(a, b):\n    return a + b\n")
    (root / "AGENTS.md").write_text("Always run tests before claiming success.\n")
    return root


@pytest.fixture
def settings(tmp_path: Path, workspace: Path) -> Settings:
    return load_settings(
        workspace=workspace,
        home=tmp_path / "jet-home",
        judge_provider="mock",
        llm_profiles={"default": LLMProfile(model="mock")},
        approval_mode="auto",
        verifier_provider="judge",
        context_budget_tokens=6000,
        trace_enabled=True,
    )


@pytest.fixture
def store(tmp_path: Path) -> Iterator[ChunkStore]:
    chunk_store = ChunkStore(tmp_path / "chunks.db", "ses_test")
    yield chunk_store
    chunk_store.close()


@pytest.fixture
def trace(tmp_path: Path) -> Trace:
    return Trace(tmp_path / "trace.jsonl", "ses_test")


@pytest.fixture
def tool_context(workspace: Path, store: ChunkStore) -> ToolContext:
    return ToolContext(workspace=workspace, store=store, memory=MemoryLoader(workspace, store))


def make_judge(
    *,
    default_noul: float = 0.9,
    default_score: float = 3.0,
    on_noul: Callable[[Any, Question], float] | None = None,
    on_score: Callable[[str, Question], float] | None = None,
    answers: Mapping[str, Any] | None = None,
) -> Judge:
    """Build a Judge over a deterministic mock provider.

    ``on_noul`` receives the full state so tests can distinguish decision kinds
    (verification states contain ``evidence``, dedup states contain
    ``existing_subgoals``, and so on).
    """

    def handler(state: Any, questions: Mapping[str, Question]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, question in questions.items():
            if question.type == "noul":
                value = on_noul(state, question) if on_noul else default_noul
                result[key] = NoulAnswer(noul=value)
            elif question.type == "score":
                value = on_score(key, question) if on_score else default_score
                result[key] = ScoreAnswer(score=value)
            elif question.type == "choice":
                options = list(question.criteria)
                result[key] = ChoiceAnswer(
                    choice=options[0],
                    probabilities={option: 1.0 / len(options) for option in options},
                    confidence=0.5,
                )
        return result

    provider = MockJudgmentProvider(answers=answers, handler=None if answers else handler)
    return Judge(provider)


def make_llm(*turns: MockTurn | str) -> MockLLMProvider:
    return MockLLMProvider(list(turns))


def noul_answer(result: JudgmentResult, key: str) -> float:
    """Read a noul probability with an explicit type check (no guessing)."""
    answer = result.answers[key]
    assert isinstance(answer, NoulAnswer), f"expected a noul answer, got {type(answer).__name__}"
    return answer.noul


class NullDecomposer:
    """Test double: pretend the task needs no decomposition."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def decompose(
        self,
        goal: str,
        *,
        parent_id: str | None = None,
        depth: int = 0,
        known: Sequence[SubGoal] | None = None,
    ) -> list[SubGoal]:
        self.calls.append(goal)
        return []


def collect_events() -> tuple[list[Any], Callable[[Any], None]]:
    events: list[Any] = []

    def sink(event: Any) -> None:
        events.append(event)

    return events, sink


def usage(input_tokens: int, output_tokens: int) -> Usage:
    return Usage(input_tokens=input_tokens, output_tokens=output_tokens)


def user_message(text: str) -> Message:
    return Message.user(text)
