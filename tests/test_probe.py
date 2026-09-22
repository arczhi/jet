"""Probe contract tests over a mocked HTTP transport."""

from __future__ import annotations

import httpx

from jet.server.probe import probe_deepseek, probe_jev


async def test_probe_deepseek_accepts_valid_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer k1"
        return httpx.Response(200, json={"choices": []})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://api.example.com"
    ) as client:
        result = await probe_deepseek("https://api.example.com", "k1", "m", client=client)
    assert result.ok and result.status == 200


async def test_probe_jev_reports_rejected_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "bad key"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await probe_jev("https://api.example.com", "bad", client=client)
    assert not result.ok and result.status == 401 and "rejected" in result.message


async def test_probe_rate_limit_counts_as_accepted() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="slow down")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await probe_deepseek("https://api.example.com", "k", "m", client=client)
    assert result.ok and "rate limited" in result.message


async def test_probe_unreachable_endpoint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await probe_jev("https://api.example.com", "k", client=client)
    assert not result.ok and result.status is None and "unreachable" in result.message
