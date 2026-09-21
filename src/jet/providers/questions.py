"""Typed questions and answers — jet's internal mirror of the TypeSafe wire contract.

The HTTP layer serializes these models; the domain consumes them. Validation is
strict on shape and loose on values (probabilities are normalized, not rejected)
so one malformed float cannot sink a whole judgment call.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator

Json = Any


class NoulQuestion(BaseModel):
    """Yes/no question. Answer is the probability of yes."""

    type: Literal["noul"] = "noul"
    instructions: Json
    criteria: dict[str, Json] | None = None


class ChoiceQuestion(BaseModel):
    """Select one option from a defined set."""

    type: Literal["choice"] = "choice"
    instructions: Json
    criteria: dict[str, Json]

    @field_validator("criteria")
    @classmethod
    def _non_empty(cls, value: dict[str, Json]) -> dict[str, Json]:
        if not value:
            raise ValueError("choice criteria must define at least one option")
        return value


class ScoreQuestion(BaseModel):
    """Rate along ordered, described levels (2 to 10)."""

    type: Literal["score"] = "score"
    instructions: Json
    criteria: list[Json]

    @field_validator("criteria")
    @classmethod
    def _levels(cls, value: list[Json]) -> list[Json]:
        if not 2 <= len(value) <= 10:
            raise ValueError("score criteria must have between 2 and 10 levels")
        return value


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float

    @field_validator("noul")
    @classmethod
    def _clamp(cls, value: float) -> float:
        return min(1.0, max(0.0, value))


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float = 0.0

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, value: float) -> float:
        return min(1.0, max(0.0, value))


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    legend: dict[str, str] = Field(default_factory=dict)
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float = 0.0

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, value: float) -> float:
        return min(1.0, max(0.0, value))


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class JudgeUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class JudgmentResult(BaseModel):
    """One provider round-trip: answers plus enough metadata to trace and cache."""

    answers: dict[str, Answer]
    model: str
    provider: str
    usage: JudgeUsage = Field(default_factory=JudgeUsage)
    latency_ms: float = 0.0
    cached: bool = False


def noul(instructions: Json, criteria: dict[str, Json] | None = None) -> NoulQuestion:
    return NoulQuestion(instructions=instructions, criteria=criteria)


def choice(instructions: Json, criteria: dict[str, Json]) -> ChoiceQuestion:
    return ChoiceQuestion(instructions=instructions, criteria=criteria)


def score(instructions: Json, criteria: list[Json]) -> ScoreQuestion:
    return ScoreQuestion(instructions=instructions, criteria=criteria)
