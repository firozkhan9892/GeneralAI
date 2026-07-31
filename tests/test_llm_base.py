"""Tests for BaseLLMProvider abstraction."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from app.llm.base import BaseLLMProvider
from app.llm.models import (
    ChatRequest,
    ChatResponse,
    Message,
    ModelInfo,
    Role,
    StreamChunk,
)


class _StubProvider(BaseLLMProvider):
    """A minimal concrete provider for exercising the base class."""

    name = "stub"

    def __init__(self, fail: bool = False) -> None:
        self._fail = fail

    @property
    def default_model(self) -> str:
        return "stub-1"

    def model_info(self, model: str | None = None) -> ModelInfo:
        return ModelInfo(name=model or "stub-1", provider=self.name)

    def generate(self, request: ChatRequest) -> ChatResponse:
        if self._fail:
            raise RuntimeError("boom")
        return ChatResponse(
            content=f"generated:{request.messages[-1].content}",
            model=request.model,
            provider=self.name,
        )

    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        if self._fail:
            raise RuntimeError("boom")
        text = request.messages[-1].content
        yield StreamChunk(content=text[:2], model=request.model, provider=self.name)
        yield StreamChunk(
            content=text[2:],
            finish_reason="stop",
            model=request.model,
            provider=self.name,
        )


def _request(text: str = "hello") -> ChatRequest:
    return ChatRequest(
        messages=(Message(role=Role.USER, content=text),),
        model="stub-1",
    )


@pytest.fixture
def provider() -> _StubProvider:
    return _StubProvider()


class TestBaseProviderContract:
    def test_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseLLMProvider()  # type: ignore[abstract]

    def test_name_defined(self, provider: _StubProvider) -> None:
        assert provider.name == "stub"

    def test_default_model(self, provider: _StubProvider) -> None:
        assert provider.default_model == "stub-1"

    def test_model_info(self, provider: _StubProvider) -> None:
        info = provider.model_info()
        assert info.name == "stub-1"
        assert info.provider == "stub"


class TestBaseProviderAsync:
    @pytest.mark.asyncio
    async def test_generate_async_delegates(self, provider: _StubProvider) -> None:
        response = await provider.generate_async(_request("hi"))
        assert response.content == "generated:hi"

    @pytest.mark.asyncio
    async def test_stream_async_yields_all_chunks(
        self, provider: _StubProvider
    ) -> None:
        chunks = [c async for c in provider.stream_async(_request("hello"))]
        assert "".join(c.content for c in chunks) == "hello"
        assert chunks[-1].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_generate_async_propagates_errors(self) -> None:
        failing = _StubProvider(fail=True)
        with pytest.raises(RuntimeError):
            await failing.generate_async(_request("hi"))


class TestBaseProviderHelpers:
    def test_get_model_uses_request_when_present(self) -> None:
        request = _request()
        assert BaseLLMProvider._get_model(request, "default") == "stub-1"

    def test_get_model_falls_back_to_default(self) -> None:
        request = ChatRequest(messages=(), model="")
        assert BaseLLMProvider._get_model(request, "default") == "default"
