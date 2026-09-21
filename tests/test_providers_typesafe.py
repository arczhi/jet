"""TypeSafe judgment provider contract tests over a mocked HTTP transport."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from jet.errors import ProviderAuthError, ProviderBadResponseError, ProviderRateLimitError
from jet.providers.http import HttpClient
from jet.providers.questions import noul, score
from jet.providers.typesafe_judge import TypeSafeJudgmentProvider
from tests.conftest import noul_answer


def client(handler: Any) -> HttpClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_request_shape_is_the_documented_contract() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {"urgent": {"type": "noul", "noul": 0.87}},
                "usage": {"input_tokens": 11, "output_tokens": 3},
            },
        )

    provider = TypeSafeJudgmentProvider(
        base_url="https://api.example.com/", api_key="secret", http_client=client(handler)
    )
    result = await provider.ask("state text", {"urgent": noul("Is it urgent?")})
    assert noul_answer(result, "urgent") == 0.87
    assert result.usage.input_tokens == 11
    assert result.model == "jev-1.13.0"
    assert result.provider == "typesafe"
    await provider.aclose()

    assert captured["url"] == "https://api.example.com/v1/systemone"
    assert captured["auth"] == "Bearer secret"
    assert captured["body"]["model"] == "jev-latest"
    assert captured["body"]["state"] == "state text"
    assert captured["body"]["questions"]["urgent"]["type"] == "noul"


async def test_score_legend_is_filled_from_question_when_absent() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {
                    "r": {
                        "type": "score",
                        "score": 1.5,
                        "probabilities": {"0": 0.5, "1": 0.5},
                        "confidence": 0.6,
                    }
                },
            },
        )

    provider = TypeSafeJudgmentProvider(
        base_url="https://api.example.com", api_key="k", http_client=client(handler)
    )
    result = await provider.ask("s", {"r": score("how relevant?", ["no", "yes"])})
    answer = result.answers["r"]
    assert answer.legend == {"0": "no", "1": "yes"}  # type: ignore[union-attr]
    await provider.aclose()


async def test_missing_answer_is_a_hard_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "m", "answers": {}})

    provider = TypeSafeJudgmentProvider(
        base_url="https://api.example.com", api_key="k", http_client=client(handler)
    )
    with pytest.raises(ProviderBadResponseError, match="no answer"):
        await provider.ask("s", {"q": noul("?")})
    await provider.aclose()


async def test_auth_error_is_not_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"error": "bad key"})

    provider = TypeSafeJudgmentProvider(
        base_url="https://api.example.com",
        api_key="k",
        max_retries=2,
        http_client=client(handler),
    )
    with pytest.raises(ProviderAuthError):
        await provider.ask("s", {"q": noul("?")})
    assert calls["n"] == 1
    await provider.aclose()


async def test_rate_limit_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("jet.providers.http.backoff_seconds", lambda attempt: 0.0)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json={"model": "m", "answers": {"q": {"type": "noul", "noul": 0.5}}})

    provider = TypeSafeJudgmentProvider(
        base_url="https://api.example.com",
        api_key="k",
        max_retries=2,
        http_client=client(handler),
    )
    result = await provider.ask("s", {"q": noul("?")})
    assert noul_answer(result, "q") == 0.5
    assert calls["n"] == 2
    await provider.aclose()


async def test_rate_limit_exhaustion_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="slow down")

    provider = TypeSafeJudgmentProvider(
        base_url="https://api.example.com",
        api_key="k",
        max_retries=0,
        http_client=client(handler),
    )
    with pytest.raises(ProviderRateLimitError):
        await provider.ask("s", {"q": noul("?")})
    await provider.aclose()
