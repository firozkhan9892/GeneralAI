"""Configuration model for LLM provider loading.

Defines which providers are available at runtime and how they are
selected.  Credentials are never hardcoded here — they come from
environment variables via :func:`build_llm_settings_from_env` or an
explicitly constructed :class:`LLMSettings`.
"""

from __future__ import annotations

import os
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config.defaults import (
    DEFAULT_API_MODE,
    ENV_API_MODE,
    LLM_API_MODES,
    ENV_GEMINI_API_KEY,
    ENV_GEMINI_MODEL,
    ENV_OLLAMA_MODEL,
    ENV_OLLAMA_URL,
    ENV_OPENAI_API_KEY,
    ENV_OPENAI_MODEL,
    ENV_OPENAI_URL,
    ENV_OPENROUTER_API_KEY,
    ENV_OPENROUTER_MODEL,
    ENV_OPENROUTER_URL,
)

APIMode = Literal["mock", "real"]

# Providers whose configuration must be present in ``real`` mode before
# they are registered.  ``requires_api_key`` marks providers that need a
# credential to be useful.
_REAL_PROVIDERS: tuple[tuple[str, bool], ...] = (
    ("openai", True),
    ("openrouter", True),
    ("gemini", True),
    ("ollama", False),
)


class LLMProviderConfig(BaseModel):
    """Runtime configuration for a single LLM provider.

    Attributes:
        name: Provider name (matches the ``ProviderFactory`` builder).
        api_key: Optional credential.  Never logged or exposed.
        model: Optional model override.  Falls back to the provider default.
        base_url: Optional endpoint override.  Falls back to the provider default.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Provider name registered in the factory")
    api_key: str | None = Field(default=None, description="Provider API key")
    model: str | None = Field(default=None, description="Model override")
    base_url: str | None = Field(default=None, description="Endpoint override")


class LLMSettings(BaseModel):
    """Application-level LLM provider selection.

    Attributes:
        api_mode: ``mock`` runs entirely offline with no credentials;
            ``real`` registers providers whose configuration is present.
        providers: Per-provider ``real``-mode configuration.
    """

    model_config = ConfigDict(frozen=True)

    api_mode: APIMode = Field(
        default=cast(APIMode, DEFAULT_API_MODE), description="Provider mode"
    )
    providers: tuple[LLMProviderConfig, ...] = Field(
        default_factory=tuple, description="Real-mode provider configurations"
    )

    @field_validator("api_mode", mode="before")
    @classmethod
    def _validate_api_mode(cls, value: str) -> str:
        lowered = value.lower()
        if lowered not in LLM_API_MODES:
            raise ValueError(
                f"Invalid API mode '{value}'. Must be one of {sorted(LLM_API_MODES)}."
            )
        return lowered


def _env(name: str | None) -> str | None:
    """Return the environment variable *name* or ``None``."""
    if name is None:
        return None
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return value.strip()


def _read_provider(
    name: str,
    api_key_env: str | None,
    url_env: str | None,
    model_env: str | None,
) -> LLMProviderConfig:
    """Build a provider config from environment variables."""
    return LLMProviderConfig(
        name=name,
        api_key=_env(api_key_env),
        base_url=_env(url_env),
        model=_env(model_env),
    )


def build_llm_settings_from_env() -> LLMSettings:
    """Build :class:`LLMSettings` from ``GENERAL_AI_*`` environment variables.

    Never raises for missing credentials; providers with no configuration
    are simply omitted from ``real``-mode registration.

    Returns:
        A frozen :class:`LLMSettings` instance.
    """
    api_mode: APIMode = cast(APIMode, (_env(ENV_API_MODE) or DEFAULT_API_MODE).lower())

    providers: tuple[LLMProviderConfig, ...] = ()
    if api_mode == "real":
        provider_configs = [
            _read_provider(
                "openai", ENV_OPENAI_API_KEY, ENV_OPENAI_URL, ENV_OPENAI_MODEL
            ),
            _read_provider(
                "openrouter",
                ENV_OPENROUTER_API_KEY,
                ENV_OPENROUTER_URL,
                ENV_OPENROUTER_MODEL,
            ),
            _read_provider("gemini", ENV_GEMINI_API_KEY, None, ENV_GEMINI_MODEL),
            _read_provider("ollama", None, ENV_OLLAMA_URL, ENV_OLLAMA_MODEL),
        ]
        providers = tuple(
            config for config in provider_configs if is_provider_configured(config)
        )

    return LLMSettings(api_mode=api_mode, providers=providers)


def is_provider_configured(config: LLMProviderConfig) -> bool:
    """Return ``True`` when a provider has its required configuration.

    OpenAI, OpenRouter, and Gemini need an API key.  Ollama is a local
    server and is only registered when a base URL or model is configured.

    Unknown provider names are always considered unconfigured so that
    ``register_default_llm_providers`` never attempts to build a builder
    that does not exist.
    """
    known_providers = {name for name, _ in _REAL_PROVIDERS}
    if config.name not in known_providers:
        return False
    requires_api_key = dict(_REAL_PROVIDERS).get(config.name, True)
    if requires_api_key and not config.api_key:
        return False
    if config.name == "ollama" and config.base_url is None and config.model is None:
        return False
    return True
