"""OpenRouter provider — OpenAI-compatible chat completions API."""

from __future__ import annotations

from app.llm.providers.openai import OpenAIProvider


class OpenRouterProvider(OpenAIProvider):
    """Provider for OpenRouter's OpenAI-compatible endpoints.

    OpenRouter mirrors the OpenAI chat completions contract, so this
    provider reuses :class:`OpenAIProvider` and only changes the
    default URL, model, and auth header.
    """

    name = "openrouter"
    base_url = "https://openrouter.ai/api/v1"

    def _default_model_name(self) -> str:
        return "openrouter/auto"

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers
