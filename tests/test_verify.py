"""Verifier behavior: cross-model agreement and graceful degradation."""

from __future__ import annotations

from jet.agent.verify import Verifier
from jet.providers.mock import MockTurn
from tests.conftest import make_judge, make_llm


async def test_inconclusive_llm_verifier_degrades_to_judge() -> None:
    """A reasoning model that never answers must not fail the turn."""
    judge = make_judge(default_noul=0.9)
    llm = make_llm(
        # reasoning-only: the model hit its budget before producing a verdict
        MockTurn(text="", reasoning="pondering the answer at length")
    )
    verifier = Verifier(judge=judge, llm=llm, mode="both")
    result = await verifier.verify(goal="g", answer="a", evidence="e")
    assert result.satisfied is True
    assert result.conclusive is False
    assert "inconclusive" in result.reason
    assert result.verifier == "judge+llm"


async def test_malformed_llm_verifier_degrades_to_judge() -> None:
    judge = make_judge(default_noul=0.2)
    llm = make_llm("I think it's probably fine, honestly.")
    verifier = Verifier(judge=judge, llm=llm, mode="both")
    result = await verifier.verify(goal="g", answer="a", evidence="e")
    assert result.satisfied is False
    assert result.conclusive is False
    assert "malformed" in result.reason


async def test_conclusive_llm_disagreement_fails_the_verification() -> None:
    judge = make_judge(default_noul=0.9)
    llm = make_llm('{"satisfied": false, "confidence": 0.8, "reason": "claims unproven"}')
    verifier = Verifier(judge=judge, llm=llm, mode="both")
    result = await verifier.verify(goal="g", answer="a", evidence="e")
    assert result.satisfied is False
    assert result.conclusive is True
    assert "llm" in result.reason


async def test_agreement_between_verifiers_passes() -> None:
    judge = make_judge(default_noul=0.9)
    llm = make_llm('{"satisfied": true, "confidence": 0.9, "reason": "evidence checks out"}')
    verifier = Verifier(judge=judge, llm=llm, mode="both")
    result = await verifier.verify(goal="g", answer="a", evidence="e")
    assert result.satisfied is True
    assert result.conclusive is True


async def test_llm_only_mode_still_reports_inconclusive() -> None:
    llm = make_llm(MockTurn(text="", reasoning="thinking"))
    verifier = Verifier(judge=make_judge(), llm=llm, mode="llm")
    result = await verifier.verify(goal="g", answer="a", evidence="e")
    assert result.satisfied is False
    assert result.conclusive is False
