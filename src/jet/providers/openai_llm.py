"""OpenAI-compatible LLM provider (chat completions with SSE streaming).

This is the generation/verification path. Default model: ``deepseek-v4.1-flash``.
Any OpenAI-compatible endpoint works: OpenRouter, vLLM, Ollama, llama.cpp, MLX
servers, etc. The base URL must include ``/v1``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from jet.core.types import (
    LLMResponse,
    Message,
    ReasoningDelta,
    Role,
    StreamDone,
    StreamEvent,
    TextDelta,
    ToolCall,
    ToolCallProgress,
    ToolCallsReady,
    ToolSpec,
    Usage,
)
from jet.errors import ProviderBadResponseError
from jet.providers.base import BaseLLMProvider
from jet.providers.http import HttpClient, HttpTransport

#: Emit one ToolCallProgress roughly per this many streamed argument chars,
#: so a long file write shows live progress instead of a silent caret.
TOOL_PROGRESS_EVERY_CHARS = 1200


def to_openai_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for message in messages:
        if message.role is Role.ASSISTANT:
            entry: dict[str, Any] = {"role": "assistant", "content": message.content or None}
            if message.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, ensure_ascii=False),
                        },
                    }
                    for call in message.tool_calls
                ]
            if entry["content"] is None and not message.tool_calls:
                entry["content"] = ""
            payload.append(entry)
        elif message.role is Role.TOOL:
            payload.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "content": message.content,
                    **({"name": message.name} if message.name else {}),
                }
            )
        else:
            payload.append({"role": message.role.value, "content": message.content})
    return payload


def to_openai_tools(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        }
        for spec in tools
    ]


class OpenAICompatibleLLM(BaseLLMProvider):
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        include_usage: bool = True,
        timeout_s: float = 120.0,
        max_retries: int = 2,
        idle_timeout_s: float = 90.0,
        name: str = "openai_compat",
        http_client: HttpClient | None = None,
        extra_headers: Mapping[str, str] | None = None,
        input_price: float = 0.0,
        output_price: float = 0.0,
    ):
        self.name = name
        self.model = model
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self.include_usage = include_usage
        self.input_price = input_price
        self.output_price = output_price
        self._transport = HttpTransport(
            base_url=base_url,
            api_key=api_key,
            timeout_s=timeout_s,
            max_retries=max_retries,
            idle_timeout_s=idle_timeout_s,
            provider=name,
            client=http_client,
            extra_headers=extra_headers,
        )

    async def aclose(self) -> None:
        await self._transport.aclose()

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_messages(messages),
            "stream": True,
            "temperature": self.default_temperature if temperature is None else temperature,
        }
        if tools:
            payload["tools"] = to_openai_tools(tools)
            payload["tool_choice"] = "auto"
        resolved_max = self.default_max_tokens if max_tokens is None else max_tokens
        if resolved_max:
            payload["max_tokens"] = resolved_max
        if self.include_usage:
            payload["stream_options"] = {"include_usage": True}

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage = Usage()
        finish_reason: str | None = None
        call_buffers: dict[int, dict[str, Any]] = {}
        progress_emitted: dict[int, int] = {}

        async for chunk in self._transport.stream_sse("/chat/completions", payload):
            raw_usage = chunk.get("usage")
            if isinstance(raw_usage, dict):
                usage = Usage(
                    input_tokens=int(raw_usage.get("prompt_tokens") or 0),
                    output_tokens=int(raw_usage.get("completion_tokens") or 0),
                )
            choices = chunk.get("choices") or []
            for choice in choices:
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    text_parts.append(content)
                    yield TextDelta(content)
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if reasoning:
                    reasoning_parts.append(reasoning)
                    yield ReasoningDelta(reasoning)
                for tool_delta in delta.get("tool_calls") or []:
                    name = self._accumulate_tool_call(call_buffers, tool_delta)
                    if name:
                        args_len = len(call_buffers[int(tool_delta.get("index", 0))]["arguments"])
                        emitted = progress_emitted.get(int(tool_delta.get("index", 0)), 0)
                        if args_len - emitted >= TOOL_PROGRESS_EVERY_CHARS:
                            progress_emitted[int(tool_delta.get("index", 0))] = args_len
                            yield ToolCallProgress(
                                index=int(tool_delta.get("index", 0)), name=name, chars=args_len
                            )

        calls = self._finalize_tool_calls(call_buffers, finish_reason)
        if calls:
            yield ToolCallsReady(calls)
        response = LLMResponse(
            text="".join(text_parts),
            tool_calls=calls,
            usage=usage,
            model=self.model,
            finish_reason=finish_reason,
            reasoning="".join(reasoning_parts),
        )
        yield StreamDone(response)

    @staticmethod
    def _accumulate_tool_call(buffers: dict[int, dict[str, Any]], delta: dict[str, Any]) -> str:
        """Accumulate one tool-call delta; returns the known name (if any)."""
        index = int(delta.get("index", 0))
        buffer = buffers.setdefault(index, {"id": None, "name": None, "arguments": ""})
        if delta.get("id"):
            buffer["id"] = delta["id"]
        function = delta.get("function") or {}
        if function.get("name"):
            buffer["name"] = function["name"]
        if function.get("arguments"):
            buffer["arguments"] += function["arguments"]
        return buffer["name"] or ""

    @staticmethod
    def _finalize_tool_calls(buffers: dict[int, dict[str, Any]], finish_reason: str | None) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for index in sorted(buffers):
            buffer = buffers[index]
            name = buffer["name"]
            if not name:
                raise ProviderBadResponseError(f"streamed tool call at index {index} has no function name")
            raw_arguments = buffer["arguments"].strip()
            if raw_arguments:
                try:
                    arguments = json.loads(raw_arguments)
                except ValueError as exc:
                    hint = (
                        " — the completion hit max_tokens before the tool call finished; "
                        "raise max_tokens on this profile"
                        if finish_reason == "length"
                        else ""
                    )
                    raise ProviderBadResponseError(
                        f"tool call {name} has truncated JSON arguments{hint}"
                    ) from exc
                if not isinstance(arguments, dict):
                    raise ProviderBadResponseError(f"tool call {name} arguments must be a JSON object")
            else:
                arguments = {}
            calls.append(
                ToolCall(
                    id=buffer["id"] or f"call_{index}",
                    name=name,
                    arguments=arguments,
                )
            )
        return calls
