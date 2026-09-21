"""Shared HTTP transport for model providers.

One place owns timeouts, retries, status-to-error mapping, and SSE parsing so
every provider fails in the same, diagnosable way. Retries only apply to
retryable statuses and only before a stream has produced data.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx

from jet.errors import (
    ConfigError,
    ProviderAuthError,
    ProviderBadResponseError,
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)

HttpClient = httpx.AsyncClient


def jet_version() -> str:
    from jet import __version__

    return __version__


def raise_for_status(status: int, body: str, provider: str) -> None:
    excerpt = body.strip()[:400]
    if status in (401, 403):
        raise ProviderAuthError(f"{provider}: HTTP {status}: {excerpt}", status=status)
    if status == 429:
        raise ProviderRateLimitError(f"{provider}: HTTP 429: {excerpt}")
    if status >= 500:
        raise ProviderUnavailableError(f"{provider}: HTTP {status}: {excerpt}", status=status)
    raise ProviderBadResponseError(f"{provider}: HTTP {status}: {excerpt}")


def backoff_seconds(attempt: int) -> float:
    ceiling: float = min(8.0, 0.5 * (2.0**attempt))
    return float(ceiling * (0.5 + random.random() / 2))


class HttpTransport:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        timeout_s: float = 120.0,
        max_retries: int = 2,
        provider: str = "provider",
        extra_headers: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.provider = provider
        self.extra_headers = dict(extra_headers or {})
        self._client = client
        self._owns_client = client is None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_s)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    def headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"jet/{jet_version()}",
            **self.extra_headers,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_error: ProviderError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client.post(url, json=payload, headers=self.headers())
            except httpx.HTTPError as exc:
                last_error = ProviderUnavailableError(f"{self.provider}: {exc}")
                if attempt < self.max_retries:
                    await asyncio.sleep(backoff_seconds(attempt))
                    continue
                raise last_error from exc
            if response.status_code >= 400:
                try:
                    raise_for_status(response.status_code, response.text, self.provider)
                except ProviderError as exc:
                    if exc.retryable and attempt < self.max_retries:
                        last_error = exc
                        await asyncio.sleep(backoff_seconds(attempt))
                        continue
                    raise
            try:
                data = response.json()
            except ValueError as exc:
                raise ProviderBadResponseError(
                    f"{self.provider}: response is not JSON: {response.text[:200]}"
                ) from exc
            if not isinstance(data, dict):
                raise ProviderBadResponseError(
                    f"{self.provider}: expected a JSON object, got {type(data).__name__}"
                )
            return data
        raise last_error or ProviderError(f"{self.provider}: request failed")

    async def stream_sse(self, path: str, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Yield decoded ``data:`` objects from a Server-Sent Events response."""
        url = f"{self.base_url}{path}"
        last_error: ProviderError | None = None
        for attempt in range(self.max_retries + 1):
            started = False
            try:
                async with self.client.stream("POST", url, json=payload, headers=self.headers()) as response:
                    if response.status_code >= 400:
                        body = (await response.aread()).decode("utf-8", "replace")
                        raise_for_status(response.status_code, body, self.provider)
                    async for line in response.aiter_lines():
                        if not line or line.startswith(":"):
                            continue
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            return
                        try:
                            decoded = json.loads(data)
                        except ValueError as exc:
                            raise ProviderBadResponseError(
                                f"{self.provider}: bad SSE payload: {data[:200]}"
                            ) from exc
                        started = True
                        yield decoded
                return
            except ProviderError as exc:
                if isinstance(exc, ProviderBadResponseError) or started or not exc.retryable:
                    raise
                last_error = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(backoff_seconds(attempt))
                    continue
                raise
            except httpx.HTTPError as exc:
                last_error = ProviderUnavailableError(f"{self.provider}: {exc}")
                if started or attempt >= self.max_retries:
                    raise last_error from exc
                await asyncio.sleep(backoff_seconds(attempt))
        raise last_error or ProviderError(f"{self.provider}: stream failed")


def require_api_key(api_key: str | None, provider: str, env_name: str) -> str:
    if not api_key:
        raise ConfigError(f"{provider} requires an API key. Set {env_name} (see .env.example).")
    return api_key
