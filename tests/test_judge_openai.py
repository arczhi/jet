"""OpenAI-compatible judgment provider tests (the laya-mlx path)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from jet.errors import ProviderBadResponseError
from jet.providers.http import HttpClient
from jet.providers.openai_judge import SYSTEM_PROMPT, OpenAICompatibleJudge
from jet.providers.questions import noul, score
from tests.conftest import noul_answer


def client(handler: Any) -> HttpClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def make_judge(handler: Any, **kwargs: Any) -> OpenAICompatibleJudge:
    return OpenAICompatibleJudge(
        base_url="http://127.0.0.1:9/v1",
        api_key=None,
        model="laya",
        http_client=client(handler),
        **kwargs,
    )


def completion(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "laya",
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8},
        },
    )


async def test_noul_roundtrip_and_prompt_contract() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return completion(json.dumps({"answers": {"q": {"type": "noul", "noul": 0.73}}}))

    judge = make_judge(handler)
    result = await judge.ask("some state", {"q": noul("Is it so?")})
    assert noul_answer(result, "q") == 0.73
    assert result.usage.input_tokens == 20
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert captured["body"]["messages"][0]["content"] == SYSTEM_PROMPT
    user_payload = json.loads(captured["body"]["messages"][1]["content"])
    assert user_payload["state"] == "some state"
    assert user_payload["questions"]["q"]["type"] == "noul"
    await judge.aclose()


async def test_json_mode_can_be_disabled() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return completion(json.dumps({"answers": {"q": {"type": "noul", "noul": 0.1}}}))

    judge = make_judge(handler, json_mode=False)
    await judge.ask("s", {"q": noul("?")})
    assert "response_format" not in captured["body"]
    await judge.aclose()


async def test_code_fenced_json_is_recovered() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return completion('```json\n{"answers": {"q": {"type": "noul", "noul": 0.4}}}\n```')

    judge = make_judge(handler)
    result = await judge.ask("s", {"q": noul("?")})
    assert noul_answer(result, "q") == 0.4
    await judge.aclose()


async def test_score_legend_is_filled() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return completion(
            json.dumps(
                {
                    "answers": {
                        "r": {
                            "type": "score",
                            "score": 0.5,
                            "probabilities": {"0": 0.5, "1": 0.5},
                            "confidence": 0.5,
                        }
                    }
                }
            )
        )

    judge = make_judge(handler)
    result = await judge.ask("s", {"r": score("how relevant?", ["irrelevant", "essential"])})
    assert result.answers["r"].legend == {"0": "irrelevant", "1": "essential"}  # type: ignore[union-attr]
    await judge.aclose()


async def test_missing_answers_object_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return completion(json.dumps({"nope": 1}))

    judge = make_judge(handler)
    with pytest.raises(ProviderBadResponseError, match="answers"):
        await judge.ask("s", {"q": noul("?")})
    await judge.aclose()


async def test_unanswered_question_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return completion(json.dumps({"answers": {}}))

    judge = make_judge(handler)
    with pytest.raises(ProviderBadResponseError, match="no answer"):
        await judge.ask("s", {"q": noul("?")})
    await judge.aclose()


async def test_non_json_reply_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return completion("I think the answer is probably yes.")

    judge = make_judge(handler)
    with pytest.raises(ProviderBadResponseError, match="no JSON object"):
        await judge.ask("s", {"q": noul("?")})
    await judge.aclose()
