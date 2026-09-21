"""Typed configuration.

Resolution order (highest priority first):
CLI overrides -> environment -> project .env -> project jet.toml ->
``~/.config/jet/config.toml`` -> defaults.

Scalar fields are replaced by the highest-priority source. Dict fields
(``llm_profiles``, ``permission_rules``, header maps) are merged key-by-key
across sources — a profile defined in ``jet.toml`` is still visible when the
environment adds another one. Select one with ``llm_profile`` explicitly;
selecting a profile that exists in no source fails loudly.

Secrets only ever come from the environment or .env (gitignored); TOML files are
for non-secret settings. ``jet doctor`` prints this config with secrets redacted.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from jet.errors import ConfigError

USER_CONFIG_PATH = Path("~/.config/jet/config.toml")
PROJECT_CONFIG_NAME = "jet.toml"


def _discover_config_files() -> list[Path]:
    files: list[Path] = []
    explicit = os.environ.get("JET_CONFIG")
    if explicit:
        files.append(Path(explicit).expanduser())
    files.append(USER_CONFIG_PATH.expanduser())
    files.append(Path.cwd() / PROJECT_CONFIG_NAME)
    return files


def load_toml_config(paths: list[Path] | None = None) -> dict[str, Any]:
    """Merge TOML files in order; later files win. Missing files are skipped."""
    merged: dict[str, Any] = {}
    for path in paths if paths is not None else _discover_config_files():
        if not path.is_file():
            continue
        with path.open("rb") as handle:
            data = tomllib.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"config file {path} must contain a TOML table")
        merged.update(data)
    return merged


class _TomlSource(PydanticBaseSettingsSource):
    def __init__(self, settings_cls: type[BaseSettings], data: dict[str, Any]):
        super().__init__(settings_cls)
        self._data = data

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        return self._data


class LLMProfile(BaseModel):
    """One generation/verification endpoint.

    Named profiles are the plugin seam for LLMs: point jet at OpenCode Go, the
    official DeepSeek API, a local vLLM server, or anything OpenAI-compatible by
    adding a profile and selecting it with ``JET_LLM_PROFILE``.
    """

    base_url: str | None = None
    api_key: str | None = None
    model: str = "deepseek-flash"
    temperature: float = 0.0
    max_tokens: int | None = None
    include_usage: bool = True
    extra_headers: dict[str, str] = Field(default_factory=dict)
    input_price: float = 0.0
    output_price: float = 0.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="JET_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_default=True,
    )

    # judgment model (fast System One decisions)
    judge_provider: Literal["typesafe", "openai_compat", "mock"] = "typesafe"
    typesafe_base_url: str = "https://api.typesafe.ai"
    typesafe_api_key: str | None = None
    typesafe_model: str = "jev-latest"
    judge_openai_base_url: str = "http://127.0.0.1:8080/v1"
    judge_openai_api_key: str | None = None
    judge_openai_model: str = "laya"
    judge_openai_json_mode: bool = True
    judge_cache: bool = True
    judge_extra_headers: dict[str, str] = Field(default_factory=dict)
    provider_timeout_s: float = Field(default=120.0, gt=0)
    provider_max_retries: int = Field(default=2, ge=0)

    # generation + verification LLMs (named profiles; see LLMProfile)
    llm_profile: str = "default"
    llm_profiles: dict[str, LLMProfile] = Field(default_factory=lambda: {"default": LLMProfile()})
    verifier_llm_profile: str | None = None
    verifier_provider: Literal["llm", "judge", "both"] = "both"

    # session / storage
    home: Path = Path("~/.jet")
    workspace: Path = Path(".")
    max_steps: int = 24
    context_budget_tokens: int = Field(default=24_000, ge=1_000)
    attention_batch_size: int = Field(default=24, ge=1)
    attention_full_threshold: float = Field(default=2.5, ge=0.0)
    attention_long_threshold: float = Field(default=1.5, ge=0.0)
    attention_short_threshold: float = Field(default=0.4, ge=0.0)
    tool_top_k: int = Field(default=6, ge=1)
    approval_mode: Literal["ask", "auto", "deny"] = "ask"
    permission_judge: bool = True
    permission_rules: list[dict[str, Any]] = Field(default_factory=list)
    verifier_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    decompose_max_depth: int = Field(default=2, ge=0)
    decompose_max_subgoals: int = Field(default=16, ge=1)
    trace_enabled: bool = True

    # pricing per million tokens for cost accounting (judgment model only;
    # LLM prices live on each LLMProfile)
    judge_input_price: float = 0.0
    judge_output_price: float = 0.0

    @field_validator("home", "workspace", mode="after")
    @classmethod
    def _expand(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _TomlSource(settings_cls, load_toml_config()),
            file_secret_settings,
        )

    @property
    def session_dir(self) -> Path:
        return self.home / "sessions"

    @property
    def cache_dir(self) -> Path:
        return self.home / "cache"

    def llm_profile_named(self, name: str) -> LLMProfile:
        profile = self.llm_profiles.get(name)
        if profile is None:
            raise ConfigError(
                f"unknown LLM profile {name!r}; configured profiles: "
                f"{', '.join(sorted(self.llm_profiles)) or '(none)'}"
            )
        return profile

    @property
    def active_llm(self) -> LLMProfile:
        return self.llm_profile_named(self.llm_profile)

    @property
    def llm_model(self) -> str:
        return self.active_llm.model

    def verifier_llm(self) -> LLMProfile:
        """Verification profile: a different model by default is cross-model review."""
        if self.verifier_llm_profile and self.verifier_llm_profile != self.llm_profile:
            return self.llm_profile_named(self.verifier_llm_profile)
        return self.active_llm

    def redacted(self) -> dict[str, Any]:
        """Config as a plain dict with secrets masked, for doctor/journal output."""
        data = self.model_dump(mode="json")

        def mask(value: Any) -> Any:
            text = str(value)
            return f"***{text[-4:]}" if len(text) > 4 else "***"

        for key in ("typesafe_api_key", "judge_openai_api_key"):
            if data.get(key):
                data[key] = mask(data[key])
        for name, profile in data.get("llm_profiles", {}).items():
            if profile.get("api_key"):
                profile["api_key"] = mask(profile["api_key"])
            data["llm_profiles"][name] = profile
        return data


def load_settings(**overrides: Any) -> Settings:
    """Build Settings, dropping None overrides so env/TOML still win."""
    clean = {k: v for k, v in overrides.items() if v is not None}
    return Settings(**clean)
