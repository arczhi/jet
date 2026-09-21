"""Permissions and approval policy.

Deterministic rules run first and are the only source of permission grants.
The judgment model may only make a decision *stricter*, never more permissive:
its role is to catch a dangerous command that no rule pattern anticipated.

Three modes:

* ``ask``  — rules decide allow/ask/deny; ``ask`` goes to the human.
* ``auto`` — rule ``ask`` resolves to allow unless judgment flags it unsafe.
* ``deny`` — only read-only tools run without a human; everything else is denied.

Nothing here prompts anybody. ``evaluate`` returns a verdict; the agent loop
asks the user when the verdict is ``ask``, and fails closed if no approver exists.
"""

from __future__ import annotations

import fnmatch
import json
import re
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, field_validator

from jet.core.types import ApprovalDecision, ApprovalMode, PolicyVerdict, ToolSpec
from jet.errors import PolicyDeniedError
from jet.providers.judge import Judge
from jet.tracing import Trace

UNSAFE_THRESHOLD = 0.2
UNSAFE_INSTRUCTION = (
    "Is `call` safe to execute in the user's workspace and plausibly aligned with `goal`? "
    "Unsafe means it could destroy data, leak secrets, escalate privileges, "
    "or run untrusted remote code."
)
UNSAFE_CRITERIA = {
    "true": "Safe: reversible or read-only, contained to the workspace, aligned with the goal.",
    "false": "Unsafe: could destroy data, leak secrets, escalate privileges, or run remote code.",
}


class PermissionRule(BaseModel):
    """One rule. ``tool`` is a glob over tool names; ``match`` is a regex over the
    call's subject (command for shell, path for file tools, JSON otherwise)."""

    tool: str = "*"
    action: Literal["allow", "ask", "deny"]
    match: str | None = None
    reason: str = ""
    _compiled: re.Pattern[str] | None = None

    @field_validator("match")
    @classmethod
    def _valid_regex(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"invalid rule regex {value!r}: {exc}") from exc
        return value

    def matches(self, tool_name: str, subject: str) -> bool:
        if not fnmatch.fnmatch(tool_name, self.tool):
            return False
        if self.match is None:
            return True
        return re.search(self.match, subject) is not None


DEFAULT_RULES: list[PermissionRule] = [
    PermissionRule(tool="read_file", action="allow", reason="read-only"),
    PermissionRule(tool="list_files", action="allow", reason="read-only"),
    PermissionRule(tool="glob", action="allow", reason="read-only"),
    PermissionRule(tool="grep", action="allow", reason="read-only"),
    PermissionRule(
        tool="run_command",
        action="deny",
        match=r"rm\s+(-[a-zA-Z]*\s+)*(-[a-zA-Z]*r[a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*r)[a-zA-Z]*\s+(/|\$HOME|~)(\s|$)",
        reason="recursive delete of a root or home path",
    ),
    PermissionRule(tool="run_command", action="deny", match=r"\bsudo\b", reason="privilege escalation"),
    PermissionRule(
        tool="run_command",
        action="deny",
        match=r"(curl|wget|fetch)\b[^\n|]*\|\s*(sh|bash|zsh|python[0-9.]*)\b",
        reason="piping remote content into a shell",
    ),
    PermissionRule(tool="run_command", action="deny", match=r":\(\)\s*\{.*\};\s*:", reason="fork bomb"),
    PermissionRule(tool="write_file", action="ask", reason="writes to the workspace"),
    PermissionRule(tool="edit_file", action="ask", reason="edits the workspace"),
    PermissionRule(tool="run_command", action="ask", reason="shell execution"),
    PermissionRule(tool="*", action="ask", reason="not covered by an explicit rule"),
]


def subject_for(tool: ToolSpec, arguments: dict[str, Any]) -> str:
    if tool.name == "run_command":
        command = arguments.get("command")
        if isinstance(command, str):
            return command
    for key in ("path", "file", "target"):
        value = arguments.get(key)
        if isinstance(value, str):
            return value
    return json.dumps(arguments, ensure_ascii=False, default=str)


class PermissionPolicy:
    def __init__(
        self,
        rules: Sequence[PermissionRule] | None = None,
        *,
        mode: ApprovalMode = ApprovalMode.ASK,
        judge: Judge | None = None,
        judge_enabled: bool = True,
        trace: Trace | None = None,
        unsafe_threshold: float = UNSAFE_THRESHOLD,
    ):
        self.rules = list(rules) if rules else list(DEFAULT_RULES)
        self.mode = mode
        self.judge = judge
        self.judge_enabled = judge_enabled
        self.trace = trace
        self.unsafe_threshold = unsafe_threshold

    @classmethod
    def from_config(cls, raw_rules: Sequence[dict[str, Any]], **kwargs: Any) -> PermissionPolicy:
        rules = [PermissionRule.model_validate(rule) for rule in raw_rules] if raw_rules else None
        return cls(rules, **kwargs)

    async def evaluate(
        self,
        *,
        tool: ToolSpec,
        arguments: dict[str, Any],
        goal: str,
    ) -> PolicyVerdict:
        subject = subject_for(tool, arguments)
        verdict = self._rule_verdict(tool.name, subject)
        if verdict.decision is ApprovalDecision.DENY:
            self._trace(tool, subject, verdict)
            return verdict

        if self.mode is ApprovalMode.DENY and not tool.read_only:
            verdict = PolicyVerdict(
                decision=ApprovalDecision.DENY,
                reason="approval mode 'deny' allows only read-only tools",
                rule="mode:deny",
            )
            self._trace(tool, subject, verdict)
            return verdict

        if self.judge_enabled and self.judge is not None and not tool.read_only:
            unsafe = await self.judge.noul(
                {
                    "call": {"tool": tool.name, "arguments": arguments},
                    "goal": goal,
                },
                UNSAFE_INSTRUCTION,
                criteria=UNSAFE_CRITERIA,
                purpose="permission_advice",
            )
            if unsafe < self.unsafe_threshold:
                verdict = PolicyVerdict(
                    decision=ApprovalDecision.DENY,
                    reason=f"judgment flagged this call as unsafe (safe-probability {unsafe:.2f})",
                    rule="judge",
                    judged=True,
                )
                self._trace(tool, subject, verdict)
                return verdict
            verdict.judged = True

        if self.mode is ApprovalMode.AUTO and verdict.decision is ApprovalDecision.ASK:
            verdict = PolicyVerdict(
                decision=ApprovalDecision.ALLOW,
                reason=f"auto-approved: {verdict.reason}",
                rule=verdict.rule,
                judged=verdict.judged,
            )
        self._trace(tool, subject, verdict)
        return verdict

    def _rule_verdict(self, tool_name: str, subject: str) -> PolicyVerdict:
        for rule in self.rules:
            if rule.matches(tool_name, subject):
                decision = ApprovalDecision(rule.action)
                return PolicyVerdict(
                    decision=decision,
                    reason=rule.reason or f"matched rule {rule.tool}/{rule.action}",
                    rule=f"{rule.tool}:{rule.action}",
                )
        raise PolicyDeniedError(
            "no permission rule matched; refusing to run (add a catch-all rule to allow asking)"
        )

    def _trace(self, tool: ToolSpec, subject: str, verdict: PolicyVerdict) -> None:
        if self.trace is None:
            return
        self.trace.event(
            "policy.verdict",
            tool=tool.name,
            subject=subject[:400],
            decision=verdict.decision.value,
            rule=verdict.rule,
            judged=verdict.judged,
            reason=verdict.reason,
        )
