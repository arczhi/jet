"""OpenAI-compatible judgment provider (laya-mlx and friends).

A System One judgment is not a native Chat Completions concept, so this provider
carries the contract explicitly: one structured prompt, JSON-only output, strict
validation back into jet's typed answers. If a model cannot honor the contract,
the provider raises ``ProviderBadResponseError`` — it never guesses.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any

from jet.errors import ProviderBadResponseError
from jet.providers.http import HttpClient, HttpTransport
from jet.providers.questions import JudgmentResult, Question, ScoreAnswer

SYSTEM_PROMPT = """You are a fast judgment model used by software.
You answer questions about a provided state. You do not chat.

Respond with ONLY one JSON object. No prose, no markdown fences.

Shape:
{"answers": {"<question_id>": <answer>, ...}}

Answer shapes by question type:
- noul:   {"type": "noul", "noul": <probability the answer is yes, 0..1>}
- choice: {"type": "choice", "choice": "<option key>",
           "probabilities": {"<option key>": <probability>, ...}, "confidence": <0..1>}
- score:  {"type": "score", "score": <probability-weighted level index, float>,
           "probabilities": {"<level index>": <probability>, ...}, "confidence": <0..1>}

Rules:
- Answer every question id exactly once, using the same ids.
- Probabilities are numbers that sum to 1.
- Answer only what was asked. Do not add extra keys, explanations, or fields."""


class OpenAICompatibleJudge:
    """Judgment provider over an OpenAI-compatible /chat/completions endpoint."""

    name = "openai_compat"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str,
        json_mode: bool = True,
        timeout_s: float = 120.0,
        max_retries: int = 2,
        http_client: HttpClient | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ):
        self.model = model
        self.json_mode = json_mode
        self._transport = HttpTransport(
            base_url=base_url,
            api_key=api_key,
            timeout_s=timeout_s,
            max_retries=max_retries,
            provider="judge_openai",
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
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "state": state,
                            "questions": {
                                key: question.model_dump(exclude_none=True)
                                for key, question in questions.items()
                            },
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "temperature": 0,
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}

        started = time.monotonic()
        data = await self._transport.post_json("/chat/completions", payload)
        latency_ms = (time.monotonic() - started) * 1000

        try:
            choices = data["choices"]
            content = choices[0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderBadResponseError(
                f"judge_openai: unexpected completion shape: {str(data)[:200]}"
            ) from exc

        answers = self._parse_content(content)
        usage = data.get("usage") or {}
        try:
            result = JudgmentResult.model_validate(
                {
                    "answers": answers,
                    "model": data.get("model", self.model),
                    "provider": self.name,
                    "usage": {
                        "input_tokens": int(usage.get("prompt_tokens") or 0),
                        "output_tokens": int(usage.get("completion_tokens") or 0),
                    },
                    "latency_ms": latency_ms,
                }
            )
        except ValueError as exc:
            raise ProviderBadResponseError(f"judge_openai: invalid answers: {exc}") from exc

        missing = set(questions) - set(result.answers)
        if missing:
            raise ProviderBadResponseError(f"judge_openai: no answer for question ids: {sorted(missing)}")
        return self._fill_legend(result, questions)

    @staticmethod
    def _parse_content(content: Any) -> dict[str, Any]:
        if not isinstance(content, str) or not content.strip():
            raise ProviderBadResponseError("judge_openai: empty completion content")
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ProviderBadResponseError(f"judge_openai: completion contains no JSON object: {text[:200]}")
        try:
            parsed = json.loads(text[start : end + 1])
        except ValueError as exc:
            raise ProviderBadResponseError(
                f"judge_openai: malformed JSON in completion: {text[start : end + 1][:200]}"
            ) from exc
        answers = parsed.get("answers") if isinstance(parsed, dict) else None
        if not isinstance(answers, dict):
            raise ProviderBadResponseError("judge_openai: JSON is missing an 'answers' object")
        return answers

    @staticmethod
    def _fill_legend(result: JudgmentResult, requested: Mapping[str, Question]) -> JudgmentResult:
        for key, answer in result.answers.items():
            if isinstance(answer, ScoreAnswer) and not answer.legend:
                question = requested.get(key)
                if question is not None and question.type == "score":
                    answer.legend = {str(i): str(level) for i, level in enumerate(question.criteria)}
        return result
