"""Tests for config-driven LLM provider loading (Phase 14.6).

Covers the LLMSettings config model, the environment builder, the
``register_default_llm_providers`` helper, and its integration into
``create_app`` — including mock mode, real mode with mocked
credentials, missing credentials, invalid API_MODE, idempotency, and
secret-leakage guards.  No external network calls are made.
"""

from __future__ import annotations

import os
from typing import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.llm.bootstrap import register_default_llm_providers
from app.llm.config import (
    LLMProviderConfig,
    LLMSettings,
    build_llm_settings_from_env,
)
from app.llm.factory import ProviderFactory
from app.llm.providers import MockProvider, OpenAIProvider, OpenRouterProvider
from app.llm.providers.ollama import OllamaProvider
from app.llm.registry import ProviderRegistry
from app.server.app import create_app
from app.server.config import ServerSettings


@pytest.fixture
def registry() -> ProviderRegistry:
    return ProviderRegistry()


@pytest.fixture
def factory() -> ProviderFactory:
    return ProviderFactory()


@pytest.fixture
def mock_llm_settings() -> LLMSettings:
    return LLMSettings(api_mode="mock")


@pytest.fixture
def real_llm_settings() -> LLMSettings:
    return LLMSettings(
        api_mode="real",
        providers=(
            LLMProviderConfig(name="openai", api_key="sk-test", model="gpt-4o-mini"),
            LLMProviderConfig(name="ollama"),
        ),
    )


@pytest.fixture
def clear_llm_env() -> Iterator[None]:
    """Snapshot and clear LLM-related env vars for the duration of a test."""
    env_vars = [
        "GENERAL_AI_API_MODE",
        "GENERAL_AI_OPENAI_API_KEY",
        "GENERAL_AI_OPENAI_MODEL",
        "GENERAL_AI_OPENAI_URL",
        "GENERAL_AI_OPENROUTER_API_KEY",
        "GENERAL_AI_OPENROUTER_MODEL",
        "GENERAL_AI_OPENROUTER_URL",
        "GENERAL_AI_GEMINI_API_KEY",
        "GENERAL_AI_GEMINI_MODEL",
        "GENERAL_AI_OLLAMA_MODEL",
        "GENERAL_AI_OLLAMA_URL",
    ]
    saved = {name: os.environ.get(name) for name in env_vars}
    for name in env_vars:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class TestLLMSettingsModel:
    def test_default_mode_is_mock(self) -> None:
        settings = LLMSettings()
        assert settings.api_mode == "mock"

    def test_real_mode_accepts_providers(self, real_llm_settings: LLMSettings) -> None:
        assert real_llm_settings.api_mode == "real"
        names = [p.name for p in real_llm_settings.providers]
        assert names == ["openai", "ollama"]

    def test_frozen(self) -> None:
        settings = LLMSettings()
        with pytest.raises(Exception):
            settings.api_mode = "real"  # type: ignore[misc]

    def test_invalid_api_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="API mode"):
            LLMSettings(api_mode="banana")  # type: ignore[arg-type]  # intentional

    def test_provider_config_carries_credentials(self) -> None:
        config = LLMProviderConfig(name="openai", api_key="sk-secret")
        assert config.api_key == "sk-secret"


class TestBuildSettingsFromEnv:
    def test_defaults_to_mock(self, clear_llm_env: None) -> None:
        settings = build_llm_settings_from_env()
        assert settings.api_mode == "mock"
        assert settings.providers == ()

    def test_real_mode_without_credentials_registers_none(
        self, clear_llm_env: None
    ) -> None:
        os.environ["GENERAL_AI_API_MODE"] = "real"
        settings = build_llm_settings_from_env()
        assert settings.api_mode == "real"
        assert settings.providers == ()

    def test_real_mode_with_openai_credentials(self, clear_llm_env: None) -> None:
        os.environ["GENERAL_AI_API_MODE"] = "real"
        os.environ["GENERAL_AI_OPENAI_API_KEY"] = "sk-real"
        settings = build_llm_settings_from_env()
        assert settings.api_mode == "real"
        assert [p.name for p in settings.providers] == ["openai"]
        assert settings.providers[0].api_key == "sk-real"

    def test_real_mode_case_insensitive(self, clear_llm_env: None) -> None:
        os.environ["GENERAL_AI_API_MODE"] = "REAL"
        os.environ["GENERAL_AI_GEMINI_API_KEY"] = "gem-test"
        settings = build_llm_settings_from_env()
        assert settings.api_mode == "real"
        assert [p.name for p in settings.providers] == ["gemini"]


class TestRegisterDefaultProviders:
    def test_mock_mode_registers_mock(
        self, registry: ProviderRegistry, factory: ProviderFactory
    ) -> None:
        register_default_llm_providers(registry, factory, LLMSettings(api_mode="mock"))
        assert isinstance(registry.get_or_raise("mock"), MockProvider)
        assert registry.count == 1

    def test_mock_mode_is_idempotent(
        self, registry: ProviderRegistry, factory: ProviderFactory
    ) -> None:
        register_default_llm_providers(registry, factory, LLMSettings(api_mode="mock"))
        register_default_llm_providers(registry, factory, LLMSettings(api_mode="mock"))
        assert registry.count == 1

    def test_mock_mode_does_not_override_existing(
        self, registry: ProviderRegistry, factory: ProviderFactory
    ) -> None:
        existing = MockProvider(model="custom")
        registry.register(existing)
        register_default_llm_providers(registry, factory, LLMSettings(api_mode="mock"))
        assert registry.get("mock") is existing

    def test_real_mode_registers_configured_only(
        self, registry: ProviderRegistry, factory: ProviderFactory
    ) -> None:
        register_default_llm_providers(registry, factory, self._real_settings())
        provider = registry.get_or_raise("openai")
        assert isinstance(provider, OpenAIProvider)
        assert registry.has("ollama")
        assert not registry.has("gemini")

    def test_real_mode_skips_missing_credentials(
        self, registry: ProviderRegistry, factory: ProviderFactory
    ) -> None:
        settings = LLMSettings(
            api_mode="real", providers=(LLMProviderConfig(name="openai"),)
        )
        register_default_llm_providers(registry, factory, settings)
        assert registry.count == 0

    def test_real_mode_registers_ollama_without_key(
        self, registry: ProviderRegistry, factory: ProviderFactory
    ) -> None:
        settings = LLMSettings(
            api_mode="real",
            providers=(
                LLMProviderConfig(name="ollama", base_url="http://localhost:11434"),
            ),
        )
        register_default_llm_providers(registry, factory, settings)
        assert isinstance(registry.get_or_raise("ollama"), OllamaProvider)

    def test_real_mode_forwards_kwargs(
        self, registry: ProviderRegistry, factory: ProviderFactory
    ) -> None:
        settings = LLMSettings(
            api_mode="real",
            providers=(LLMProviderConfig(name="openai", api_key="k", model="m"),),
        )
        register_default_llm_providers(registry, factory, settings)
        provider = registry.get_or_raise("openai")
        assert isinstance(provider, OpenAIProvider)
        assert provider.default_model == "m"

    def test_real_mode_is_idempotent(
        self, registry: ProviderRegistry, factory: ProviderFactory
    ) -> None:
        settings = self._real_settings()
        register_default_llm_providers(registry, factory, settings)
        register_default_llm_providers(registry, factory, settings)
        assert registry.count == 2

    def test_no_secret_leakage_in_registry(
        self, registry: ProviderRegistry, factory: ProviderFactory
    ) -> None:
        settings = LLMSettings(
            api_mode="real",
            providers=(LLMProviderConfig(name="openrouter", api_key="sk-leak-check"),),
        )
        register_default_llm_providers(registry, factory, settings)
        provider = registry.get_or_raise("openrouter")
        assert isinstance(provider, OpenRouterProvider)
        assert "sk-leak-check" not in repr(provider)

    @staticmethod
    def _real_settings() -> LLMSettings:
        return LLMSettings(
            api_mode="real",
            providers=(
                LLMProviderConfig(name="openai", api_key="sk-test"),
                LLMProviderConfig(name="ollama", base_url="http://localhost:11434"),
            ),
        )


class TestCreateAppIntegration:
    def test_mock_mode_starts_without_credentials(self) -> None:
        app = create_app()
        assert isinstance(app, FastAPI)
        container = app.state.container
        registry = container.resolve(ProviderRegistry)
        assert isinstance(registry.get_or_raise("mock"), MockProvider)

    def test_mock_mode_starts_with_empty_env(self, clear_llm_env: None) -> None:
        app = create_app(llm_settings=LLMSettings(api_mode="mock"))
        registry = app.state.container.resolve(ProviderRegistry)
        assert isinstance(registry.get_or_raise("mock"), MockProvider)

    def test_real_mode_with_credentials(self) -> None:
        settings = LLMSettings(
            api_mode="real",
            providers=(LLMProviderConfig(name="openai", api_key="sk-test"),),
        )
        app = create_app(llm_settings=settings)
        registry = app.state.container.resolve(ProviderRegistry)
        assert isinstance(registry.get_or_raise("openai"), OpenAIProvider)

    def test_real_mode_without_credentials_starts(self, clear_llm_env: None) -> None:
        app = create_app(llm_settings=LLMSettings(api_mode="real"))
        registry = app.state.container.resolve(ProviderRegistry)
        assert registry.count == 0

    def test_backward_compatible_signature(self) -> None:
        app = create_app(settings=ServerSettings(api_key="x"), discover_tools=False)
        assert isinstance(app, FastAPI)

    def test_health_endpoint_still_works(self) -> None:
        app = create_app()
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
