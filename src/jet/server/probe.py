"""Save-time credential probing for the setup dialog.

Two tiny requests (one per provider) tell the user which key is broken at the
moment they click save, instead of a raw 401 later on the first turn.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

PROBE_TIMEOUT_S = 15.0


@dataclass
class ProbeResult:
    ok: bool
    status: int | None
    message: str

    def to_json(self) -> dict[str, object]:
        return {"ok": self.ok, "status": self.status, "message": self.message}


def _judge(status: int | None, body: str) -> ProbeResult:
    if status == 200:
        return ProbeResult(ok=True, status=status, message="key accepted")
    if status in (401, 403):
        return ProbeResult(False, status, "key rejected by the endpoint")
    if status == 429:
        # Rate limiting proves the credentials were accepted.
        return ProbeResult(True, status, "key accepted (rate limited right now)")
    if status == 404:
        return ProbeResult(False, status, "endpoint path not found — check the URL")
    return ProbeResult(False, status, body.strip()[:200] or f"HTTP {status}")


async def probe_deepseek(
    base_url: str, api_key: str, model: str, *, client: httpx.AsyncClient | None = None
) -> ProbeResult:
    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=PROBE_TIMEOUT_S)
    try:
        response = await client.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Reply with: ok"}],
                "max_tokens": 1,
            },
        )
    except httpx.HTTPError as exc:
        return ProbeResult(False, None, f"endpoint unreachable: {type(exc).__name__}")
    finally:
        if owns_client:
            await client.aclose()
    return _judge(response.status_code, response.text)


async def probe_jev(base_url: str, api_key: str, *, client: httpx.AsyncClient | None = None) -> ProbeResult:
    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=PROBE_TIMEOUT_S)
    try:
        response = await client.post(
            f"{base_url.rstrip('/')}/v1/systemone",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "state": "2 + 2 equals 4.",
                "model": "jev-latest",
                "questions": {"probe": {"type": "noul", "instructions": "Is the statement true?"}},
            },
        )
    except httpx.HTTPError as exc:
        return ProbeResult(False, None, f"endpoint unreachable: {type(exc).__name__}")
    finally:
        if owns_client:
            await client.aclose()
    return _judge(response.status_code, response.text)
