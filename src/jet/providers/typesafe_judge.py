"""TypeSafe HTTP judgment provider (Jev).

Contract: POST ``{base_url}/v1/systemone`` with ``{state, model, questions}``.
Works against the hosted API and any compatible endpoint.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from jet.errors import ProviderBadResponseError
from jet.providers.http import HttpClient, HttpTransport
from jet.providers.questions import JudgmentResult, Question, ScoreAnswer


class TypeSafeJudgmentProvider:
    name = "typesafe"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str = "jev-latest",
        timeout_s: float = 120.0,
        max_retries: int = 2,
        idle_timeout_s: float = 90.0,
        http_client: HttpClient | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ):
        self.model = model
        self._transport = HttpTransport(
            base_url=base_url,
            api_key=api_key,
            timeout_s=timeout_s,
            max_retries=max_retries,
            idle_timeout_s=idle_timeout_s,
            provider="typesafe",
            client=http_client,
            extra_headers=extra_headers,
        )

    async def aclose(self) -> None:
        await self._transport.aclose()

    async def ask(
        self,
        state: Any,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
    ) -> JudgmentResult:
        payload = {
            "state": state,
            "model": model or self.model,
            "questions": {key: question.model_dump(exclude_none=True) for key, question in questions.items()},
        }
        started = time.monotonic()
        data = await self._transport.post_json("/v1/systemone", payload)
        latency_ms = (time.monotonic() - started) * 1000
        return self._parse(data, latency_ms, requested=questions)

    def _parse(
        self, data: dict[str, Any], latency_ms: float, requested: Mapping[str, Question]
    ) -> JudgmentResult:
        raw_answers = data.get("answers")
        if not isinstance(raw_answers, dict):
            raise ProviderBadResponseError("typesafe: response is missing an 'answers' object")
        usage = data.get("usage") or {}
        try:
            result = JudgmentResult.model_validate(
                {
                    "answers": raw_answers,
                    "model": data.get("model", self.model),
                    "provider": self.name,
                    "usage": {
                        "input_tokens": int(usage.get("input_tokens", 0)),
                        "output_tokens": int(usage.get("output_tokens", 0)),
                    },
                    "latency_ms": latency_ms,
                }
            )
        except ValueError as exc:
            raise ProviderBadResponseError(f"typesafe: invalid answers payload: {exc}") from exc
        missing = set(requested) - set(result.answers)
        if missing:
            raise ProviderBadResponseError(f"typesafe: no answer for question ids: {sorted(missing)}")
        return self._fill_legend(result, requested)

    @staticmethod
    def _fill_legend(result: JudgmentResult, requested: Mapping[str, Question]) -> JudgmentResult:
        """Attach score legends when a compatible endpoint omits them."""
        for key, answer in result.answers.items():
            if isinstance(answer, ScoreAnswer) and not answer.legend:
                question = requested.get(key)
                if question is not None and question.type == "score":
                    levels = question.criteria
                    answer.legend = {str(i): str(level) for i, level in enumerate(levels)}
        return result
