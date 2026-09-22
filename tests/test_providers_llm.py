"""OpenAI-compatible LLM provider tests: message mapping, SSE streaming, tool calls."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from jet.core.types import Message, ToolCall, ToolSpec
from jet.errors import ProviderAuthError, ProviderBadResponseError
from jet.providers.http import HttpClient
from jet.providers.openai_llm import OpenAICompatibleLLM, to_openai_messages, to_openai_tools


def client(handler: Any) -> HttpClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def sse(chunks: list[dict[str, Any]]) -> bytes:
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    return (body + "data: [DONE]\n\n").encode()


def make_llm(handler: Any, **kwargs: Any) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        base_url="http://127.0.0.1:9/v1",
        api_key=None,
        model="deepseek-v4.1-flash",
        http_client=client(handler),
        **kwargs,
    )


async def test_stream_text_and_usage() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=sse(
                [
                    {"choices": [{"delta": {"content": "Hel"}}]},
                    {"choices": [{"delta": {"content": "lo"}}]},
                    {
                        "choices": [{"delta": {}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 7, "completion_tokens": 2},
                    },
                ]
            ),
        )

    llm = make_llm(handler)
    response = await llm.complete([Message.user("hi")])
    assert response.text == "Hello"
    assert response.usage.input_tokens == 7
    assert response.usage.output_tokens == 2
    assert response.finish_reason == "stop"
    assert captured["body"]["stream"] is True
    assert captured["body"]["model"] == "deepseek-v4.1-flash"
    assert captured["body"]["stream_options"] == {"include_usage": True}
    await llm.aclose()


async def test_include_usage_flag_is_honored() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=sse([{"choices": [{"delta": {"content": "x"}}]}]))

    llm = make_llm(handler, include_usage=False)
    await llm.complete([Message.user("hi")])
    assert "stream_options" not in captured["body"]
    await llm.aclose()


async def test_tool_call_arguments_accumulate_across_chunks() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse(
                [
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_1",
                                            "function": {"name": "read_file", "arguments": '{"pa'},
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [{"index": 0, "function": {"arguments": 'th": "a.txt"}'}}]
                                }
                            }
                        ]
                    },
                    {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
                ]
            ),
        )

    llm = make_llm(handler)
    response = await llm.complete([Message.user("read a.txt")])
    assert response.tool_calls == [ToolCall(id="call_1", name="read_file", arguments={"path": "a.txt"})]
    await llm.aclose()


async def test_empty_arguments_become_empty_dict() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse(
                [
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {"index": 0, "id": "c", "function": {"name": "list_files"}}
                                    ]
                                }
                            }
                        ]
                    }
                ]
            ),
        )

    llm = make_llm(handler)
    response = await llm.complete([Message.user("list")])
    assert response.tool_calls[0].arguments == {}
    await llm.aclose()


async def test_malformed_tool_arguments_raise() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse(
                [
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "c",
                                            "function": {"name": "x", "arguments": "{oops"},
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                ]
            ),
        )

    llm = make_llm(handler)
    with pytest.raises(ProviderBadResponseError, match="truncated JSON"):
        await llm.complete([Message.user("go")])
    await llm.aclose()


async def test_truncated_tool_call_mentions_max_tokens() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse(
                [
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "c",
                                            "function": {
                                                "name": "write_file",
                                                "arguments": '{"path": "a.txt", "cont',
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                    {"choices": [{"delta": {}, "finish_reason": "length"}]},
                ]
            ),
        )

    llm = make_llm(handler)
    with pytest.raises(ProviderBadResponseError, match="raise max_tokens"):
        await llm.complete([Message.user("go")])
    await llm.aclose()


async def test_reasoning_content_is_captured_but_kept_out_of_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse(
                [
                    {"choices": [{"delta": {"reasoning_content": "let me think"}}]},
                    {"choices": [{"delta": {"content": "ok"}}]},
                    {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                ]
            ),
        )

    llm = make_llm(handler)
    response = await llm.complete([Message.user("hi")])
    assert response.text == "ok"
    assert response.reasoning == "let me think"
    await llm.aclose()


async def test_headers_identify_jet_and_allow_gateway_requirements() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["user_agent"] = request.headers.get("user-agent")
        captured["session"] = request.headers.get("x-opencode-session")
        return httpx.Response(200, content=sse([{"choices": [{"delta": {"content": "x"}}]}]))

    llm = OpenAICompatibleLLM(
        base_url="http://127.0.0.1:9/v1",
        api_key=None,
        model="m",
        http_client=client(handler),
        extra_headers={"x-opencode-session": "ses_abc"},
    )
    await llm.complete([Message.user("hi")])
    assert captured["user_agent"].startswith("jet/")
    assert captured["session"] == "ses_abc"
    await llm.aclose()


async def test_auth_error_surfaces() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad key"})

    llm = make_llm(handler, max_retries=0)
    with pytest.raises(ProviderAuthError):
        await llm.complete([Message.user("hi")])
    await llm.aclose()


def test_message_mapping_preserves_tool_protocol() -> None:
    messages = [
        Message.system("sys"),
        Message.user("u"),
        Message.assistant("thinking", tool_calls=[ToolCall("c1", "read_file", {"path": "a"})]),
        Message.tool_result("c1", "contents", name="read_file"),
    ]
    payload = to_openai_messages(messages)
    assert payload[0] == {"role": "system", "content": "sys"}
    assert payload[2]["tool_calls"][0]["function"]["name"] == "read_file"
    assert json.loads(payload[2]["tool_calls"][0]["function"]["arguments"]) == {"path": "a"}
    assert payload[3] == {"role": "tool", "tool_call_id": "c1", "content": "contents", "name": "read_file"}


def test_tool_spec_mapping() -> None:
    spec = ToolSpec(
        name="read_file",
        description="read",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    )
    payload = to_openai_tools([spec])
    assert payload[0]["type"] == "function"
    assert payload[0]["function"]["name"] == "read_file"
