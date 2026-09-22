"""Configuration precedence, LLM profiles, TOML support, and secret redaction."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from jet.config import LLMProfile, Settings, load_settings
from jet.errors import ConfigError


def write_toml(path: Path, body: str) -> Path:
    path.write_text(body)
    return path


def test_defaults_are_usable() -> None:
    settings = Settings()
    assert settings.judge_provider == "typesafe"
    assert settings.context_budget_tokens == 24_000
    assert settings.approval_mode == "ask"
    assert settings.llm_profile == "default"
    assert settings.llm_model == "deepseek-flash"


def test_env_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JET_CONTEXT_BUDGET_TOKENS", "12345")
    monkeypatch.setenv(
        "JET_LLM_PROFILES",
        '{"default": {"base_url": "http://x", "model": "deepseek-v4.1-flash"}}',
    )
    settings = Settings()
    assert settings.context_budget_tokens == 12345
    assert settings.llm_model == "deepseek-v4.1-flash"


def test_typesafe_key_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "apikey_alias")
    assert Settings().typesafe_api_key == "apikey_alias"
    monkeypatch.setenv("JET_TYPESAFE_API_KEY", "apikey_prefixed")
    assert Settings().typesafe_api_key == "apikey_prefixed"


def test_toml_file_is_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = write_toml(
        tmp_path / "jet.toml",
        'max_steps = 7\n\n[llm_profiles.default]\nmodel = "from-toml"\n',
    )
    monkeypatch.setenv("JET_CONFIG", str(config))
    settings = Settings()
    assert settings.llm_model == "from-toml"
    assert settings.max_steps == 7


def test_env_beats_toml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = write_toml(tmp_path / "jet.toml", '[llm_profiles.default]\nmodel = "from-toml"\n')
    monkeypatch.setenv("JET_CONFIG", str(config))
    monkeypatch.setenv("JET_LLM_PROFILES", '{"default": {"model": "from-env"}}')
    assert Settings().llm_model == "from-env"


def test_explicit_overrides_beat_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JET_LLM_PROFILES", '{"default": {"model": "from-env"}}')
    settings = load_settings(llm_profiles={"default": LLMProfile(model="from-code")})
    assert settings.llm_model == "from-code"


def test_none_overrides_are_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JET_LLM_PROFILES", '{"default": {"model": "from-env"}}')
    settings = load_settings(llm_profiles=None)
    assert settings.llm_model == "from-env"


def test_dict_fields_merge_across_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pinned behavior: dict config merges key-by-key; scalars are replaced."""
    monkeypatch.setenv("JET_LLM_PROFILES", '{"zen": {"model": "zen-model"}}')
    monkeypatch.setenv("JET_LLM_PROFILE", "zen")
    settings = load_settings(llm_profiles={"default": LLMProfile(model="local")})
    assert set(settings.llm_profiles) == {"default", "zen"}
    assert settings.llm_model == "zen-model"


def test_profile_selection_and_lookup() -> None:
    settings = Settings(
        llm_profile="zen",
        llm_profiles={
            "zen": LLMProfile(base_url="http://zen", model="deepseek-v4.1-flash"),
            "official": LLMProfile(base_url="http://official", model="deepseek-flash"),
        },
    )
    assert settings.llm_model == "deepseek-v4.1-flash"
    assert settings.llm_profile_named("official").model == "deepseek-flash"
    with pytest.raises(ConfigError, match="unknown LLM profile"):
        settings.llm_profile_named("nope")


def test_verifier_profile_defaults_to_the_generator() -> None:
    settings = Settings(
        llm_profile="zen",
        llm_profiles={
            "zen": LLMProfile(model="deepseek-v4.1-flash"),
            "official": LLMProfile(model="deepseek-flash"),
        },
    )
    assert settings.verifier_llm().model == "deepseek-v4.1-flash"
    with_verifier = settings.model_copy(update={"verifier_llm_profile": "official"})
    assert with_verifier.verifier_llm().model == "deepseek-flash"


def test_unknown_active_profile_fails_at_use_time() -> None:
    settings = Settings(llm_profile="ghost", llm_profiles={"default": LLMProfile()})
    with pytest.raises(ConfigError, match="ghost"):
        _ = settings.llm_model


def test_redacted_masks_secrets() -> None:
    settings = Settings(
        typesafe_api_key="sk-abcdef123456",
        llm_profiles={"default": LLMProfile(api_key="secret", base_url="http://x")},
    )
    redacted = settings.redacted()
    assert redacted["typesafe_api_key"] == "***3456"
    assert redacted["llm_profiles"]["default"]["api_key"] == "***cret"
    assert redacted["llm_profiles"]["default"]["base_url"] == "http://x"


def test_invalid_values_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(context_budget_tokens=10)
    with pytest.raises(ValidationError):
        Settings(approval_mode="yolo")  # type: ignore[arg-type]


def test_paths_are_expanded(tmp_path: Path) -> None:
    settings = Settings(home=tmp_path / "jet-home")
    assert settings.home.is_absolute()
    assert settings.session_dir == tmp_path / "jet-home" / "sessions"
