"""Provider registry: session-bound header templates and LLM profile building."""

from __future__ import annotations

import pytest

from jet.config import LLMProfile, Settings
from jet.errors import ConfigError
from jet.providers.mock import MockLLMProvider
from jet.providers.openai_llm import OpenAICompatibleLLM
from jet.providers.registry import build_llm_provider, resolve_headers


def test_session_placeholder_is_substituted() -> None:
    headers = resolve_headers({"x-opencode-session": "{session_id}", "X-Fixed": "v"}, "ses_123")
    assert headers == {"x-opencode-session": "ses_123", "X-Fixed": "v"}


def test_missing_session_is_explicit_not_empty() -> None:
    headers = resolve_headers({"x-opencode-session": "{session_id}"}, None)
    assert headers == {"x-opencode-session": "unbound"}


def test_no_headers_configured() -> None:
    assert resolve_headers({}, "ses_1") == {}


def test_build_llm_provider_uses_the_selected_profile() -> None:
    settings = Settings(
        llm_profile="official",
        llm_profiles={
            "zen": LLMProfile(base_url="http://zen", model="deepseek-v4.1-flash"),
            "official": LLMProfile(base_url="http://official", model="deepseek-flash", api_key="k"),
        },
    )
    provider = build_llm_provider(settings, session_id="ses_1")
    assert isinstance(provider, OpenAICompatibleLLM)
    assert provider.model == "deepseek-flash"
    assert provider._transport.base_url == "http://official"


def test_build_llm_provider_can_target_a_named_profile() -> None:
    settings = Settings(
        llm_profile="zen",
        llm_profiles={
            "zen": LLMProfile(base_url="http://zen", model="deepseek-v4.1-flash"),
            "official": LLMProfile(base_url="http://official", model="deepseek-flash"),
        },
    )
    provider = build_llm_provider(settings, session_id="ses_1", profile_name="official")
    assert isinstance(provider, OpenAICompatibleLLM)
    assert provider.model == "deepseek-flash"


def test_mock_profile_needs_no_base_url() -> None:
    settings = Settings(llm_profiles={"default": LLMProfile(model="mock")})
    assert isinstance(build_llm_provider(settings), MockLLMProvider)


def test_missing_base_url_is_an_explicit_error() -> None:
    settings = Settings(llm_profiles={"default": LLMProfile(model="deepseek-flash")})
    with pytest.raises(ConfigError, match="no base_url"):
        build_llm_provider(settings)


def test_profile_headers_expand_session_id() -> None:
    settings = Settings(
        llm_profiles={
            "default": LLMProfile(
                base_url="http://x",
                model="m",
                extra_headers={"x-opencode-session": "{session_id}"},
            )
        }
    )
    provider = build_llm_provider(settings, session_id="ses_42")
    assert isinstance(provider, OpenAICompatibleLLM)
    assert provider._transport.extra_headers == {"x-opencode-session": "ses_42"}


def test_profile_prices_reach_the_provider() -> None:
    settings = Settings(
        llm_profiles={
            "default": LLMProfile(base_url="http://x", model="m", input_price=1.5, output_price=7.0)
        }
    )
    provider = build_llm_provider(settings)
    assert provider.input_price == 1.5  # type: ignore[attr-defined]
    assert provider.output_price == 7.0  # type: ignore[attr-defined]
