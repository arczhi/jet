"""Provider factories. Configuration is the only place provider choice lives."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from jet.config import Settings
from jet.errors import ConfigError
from jet.providers.base import JudgmentProvider, LLMProvider
from jet.providers.caching import CachingJudgmentProvider, open_judgment_cache
from jet.providers.judge import Judge
from jet.providers.mock import MockJudgmentProvider, MockLLMProvider
from jet.providers.openai_judge import OpenAICompatibleJudge
from jet.providers.openai_llm import OpenAICompatibleLLM
from jet.providers.typesafe_judge import TypeSafeJudgmentProvider
from jet.tracing import Trace


def resolve_headers(template: Mapping[str, str], session_id: str | None) -> dict[str, str]:
    """Expand ``{session_id}`` in configured header values.

    Gateways such as OpenCode Go require a stable per-conversation session header;
    a template keeps that vendor-specific detail in configuration, not in code.
    """
    resolved: dict[str, str] = {}
    for key, value in template.items():
        resolved[key] = value.replace("{session_id}", session_id or "unbound")
    return resolved


def build_judgment_provider(settings: Settings, session_id: str | None = None) -> JudgmentProvider:
    headers = resolve_headers(settings.judge_extra_headers, session_id)
    if settings.judge_provider == "mock":
        # Offline demo mode: permissive judgments so a mock session can complete.
        return MockJudgmentProvider(default_noul=0.8, default_score=2.0)
    if settings.judge_provider == "typesafe":
        if not settings.typesafe_api_key:
            raise ConfigError(
                "JET_TYPESAFE_API_KEY is required for the typesafe judge provider "
                "(or set JET_JUDGE_PROVIDER=mock)"
            )
        return TypeSafeJudgmentProvider(
            base_url=settings.typesafe_base_url,
            api_key=settings.typesafe_api_key,
            model=settings.typesafe_model,
            timeout_s=settings.provider_timeout_s,
            max_retries=settings.provider_max_retries,
            extra_headers=headers,
        )
    if settings.judge_provider == "openai_compat":
        return OpenAICompatibleJudge(
            base_url=settings.judge_openai_base_url,
            api_key=settings.judge_openai_api_key,
            model=settings.judge_openai_model,
            json_mode=settings.judge_openai_json_mode,
            timeout_s=settings.provider_timeout_s,
            max_retries=settings.provider_max_retries,
            extra_headers=headers,
        )
    raise ConfigError(f"unknown judge provider: {settings.judge_provider}")


def build_llm_provider(
    settings: Settings, session_id: str | None = None, *, profile_name: str | None = None
) -> LLMProvider:
    name = profile_name or settings.llm_profile
    profile = settings.llm_profile_named(name)
    headers = resolve_headers(profile.extra_headers, session_id)
    if profile.model == "mock":
        return MockLLMProvider()
    if not profile.base_url:
        raise ConfigError(
            f"LLM profile {name!r} has no base_url. Configure JET_LLM_PROFILES "
            '(e.g. {"default": {"base_url": "https://api.deepseek.com", '
            '"api_key": "...", "model": "deepseek-flash"}}) or set '
            'JET_LLM_PROFILES default model to "mock" for offline runs'
        )
    return OpenAICompatibleLLM(
        base_url=profile.base_url,
        api_key=profile.api_key,
        model=profile.model,
        temperature=profile.temperature,
        max_tokens=profile.max_tokens,
        include_usage=profile.include_usage,
        timeout_s=settings.provider_timeout_s,
        max_retries=settings.provider_max_retries,
        extra_headers=headers,
        input_price=profile.input_price,
        output_price=profile.output_price,
    )


def build_judge(
    settings: Settings,
    trace: Trace | None = None,
    *,
    cache_path: Path | None = None,
    session_id: str | None = None,
) -> Judge:
    provider: JudgmentProvider = build_judgment_provider(settings, session_id=session_id)
    if settings.judge_cache:
        path = cache_path or (settings.cache_dir / "judgments.db")
        provider = CachingJudgmentProvider(provider, open_judgment_cache(path))
    return Judge(provider, trace=trace)
