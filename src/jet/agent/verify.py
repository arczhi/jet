"""Verification: did the work actually satisfy the goal?

Two independent verifiers are available and combinable:

* judgment model — fast noul over the goal, answer, and evidence
* generation LLM — strict JSON verdict with a reason

Verifier failures return a structured ``Verification`` saying so; they never
crash the turn and never silently pass.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Literal

from jet.core.types import Message, Verification
from jet.errors import JetError, ProviderError
from jet.providers.base import LLMProvider
from jet.providers.judge import Judge
from jet.tracing import Trace

SATISFIED_INSTRUCTION = (
    "Given `goal`, `answer`, and `evidence`, does the work shown actually satisfy the goal "
    "without unsupported claims or missing steps?"
)
SATISFIED_CRITERIA = {
    "true": "The goal is met and the evidence supports the claims.",
    "false": "The goal is not met, unproven, or the answer claims more than the evidence shows.",
}

LLM_SYSTEM = """You verify whether a coding agent completed a goal.
Reply with ONLY one JSON object:
{"satisfied": <true|false>, "confidence": <0..1>, "reason": "<one or two sentences>"}
Judge only what the evidence shows. Unsupported claims are not satisfied. No prose, no fences."""


class Verifier:
    def __init__(
        self,
        *,
        judge: Judge | None,
        llm: LLMProvider | None,
        mode: Literal["llm", "judge", "both"] = "both",
        threshold: float = 0.6,
        trace: Trace | None = None,
    ):
        self.judge = judge
        self.llm = llm
        self.mode = mode
        self.threshold = threshold
        self.trace = trace

    async def verify(self, *, goal: str, answer: str, evidence: str, mode: str | None = None) -> Verification:
        effective = mode or self.mode
        if effective == "judge":
            return await self._judge(goal, answer, evidence)
        if effective == "llm":
            return await self._llm(goal, answer, evidence)
        judge_result, llm_result = await asyncio.gather(
            self._judge(goal, answer, evidence),
            self._llm(goal, answer, evidence),
        )
        if not llm_result.conclusive:
            # An inconclusive cross-check must not fail the turn and force a retry
            # loop; the judgment verdict stands, with the degradation recorded.
            return Verification(
                satisfied=judge_result.satisfied,
                confidence=judge_result.confidence,
                reason=f"llm verifier inconclusive ({llm_result.reason}); judgment verdict used",
                verifier="judge+llm",
                conclusive=False,
            )
        satisfied = judge_result.satisfied and llm_result.satisfied
        reason = (
            f"judge: {judge_result.reason} | llm: {llm_result.reason}"
            if not satisfied
            else judge_result.reason
        )
        return Verification(
            satisfied=satisfied,
            confidence=min(judge_result.confidence, llm_result.confidence),
            reason=reason,
            verifier="judge+llm",
        )

    async def _judge(self, goal: str, answer: str, evidence: str) -> Verification:
        if self.judge is None:
            return Verification(
                satisfied=False,
                confidence=0.0,
                reason="no judgment model configured for verification",
                verifier="judge",
            )
        try:
            probability = await self.judge.noul(
                {"goal": goal, "answer": answer, "evidence": evidence},
                SATISFIED_INSTRUCTION,
                criteria=SATISFIED_CRITERIA,
                purpose="verification_judge",
            )
        except JetError as exc:
            # Judge outage: report an inconclusive verdict — the caller decides
            # how to degrade without spinning.
            return Verification(
                satisfied=False,
                confidence=0.0,
                reason=f"judgment verifier unavailable ({str(exc)[:120]})",
                verifier="judge",
                conclusive=False,
            )
        return Verification(
            satisfied=probability >= self.threshold,
            confidence=probability,
            reason=f"judgment probability {probability:.2f} (threshold {self.threshold:.2f})",
            verifier="judge",
        )

    async def _llm(self, goal: str, answer: str, evidence: str) -> Verification:
        if self.llm is None:
            return Verification(
                satisfied=False,
                confidence=0.0,
                reason="no generation model configured for verification",
                verifier="llm",
            )
        payload = {"goal": goal, "answer": answer, "evidence": evidence}
        messages = [
            Message.system(LLM_SYSTEM),
            Message.user(json.dumps(payload, ensure_ascii=False)),
        ]
        started = time.monotonic()
        try:
            response = await self.llm.complete(messages, temperature=0.0, max_tokens=2048)
        except ProviderError as exc:
            return Verification(
                satisfied=False,
                confidence=0.0,
                reason=f"verifier provider failed: {exc}",
                verifier="llm",
                conclusive=False,
            )
        if not response.text.strip() and response.reasoning.strip():
            # A reasoning model spent its whole budget thinking and never answered;
            # mark the verdict inconclusive so it never triggers a retry loop.
            if self.trace is not None:
                self.trace.event("verification.llm.reasoning_only", reasoning_chars=len(response.reasoning))
            return Verification(
                satisfied=False,
                confidence=0.0,
                reason="verifier produced reasoning but no verdict before its token budget",
                verifier="llm",
                conclusive=False,
            )
        try:
            parsed = json.loads(_extract_object(response.text))
            satisfied = bool(parsed["satisfied"])
            confidence = float(parsed.get("confidence", 0.5))
            reason = str(parsed.get("reason", ""))[:500]
        except (ValueError, KeyError, TypeError) as exc:
            if self.trace is not None:
                self.trace.event("verification.llm.malformed", excerpt=response.text[:300])
            return Verification(
                satisfied=False,
                confidence=0.0,
                reason=f"verifier returned malformed output ({exc})",
                verifier="llm",
                conclusive=False,
            )
        if self.trace is not None:
            self.trace.event(
                "verification.llm",
                satisfied=satisfied,
                confidence=confidence,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        return Verification(
            satisfied=satisfied,
            confidence=min(1.0, max(0.0, confidence)),
            reason=reason or "llm verdict",
            verifier="llm",
        )


def _extract_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON object in verifier reply: {text[:200]}")
    return text[start : end + 1]
