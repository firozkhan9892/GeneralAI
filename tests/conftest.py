"""Shared test fixtures for the LLM provider layer."""

from __future__ import annotations

from typing import Any, Iterator

import pytest

from app.llm.transport import HttpTransport, HttpResponse


class FakeHttpTransport(HttpTransport):
    """Deterministic transport that returns canned responses."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.responses: list[HttpResponse] = []
        self.stream_responses: list[list[bytes]] = []

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_s: float,
    ) -> HttpResponse:
        self.requests.append(
            {
                "url": url,
                "headers": headers,
                "payload": payload,
                "timeout_s": timeout_s,
            }
        )
        return self._next_response()

    def post_stream(
        self,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_s: float,
    ) -> Iterator[bytes]:
        self.requests.append(
            {
                "url": url,
                "headers": headers,
                "payload": payload,
                "timeout_s": timeout_s,
            }
        )
        if not self.stream_responses:
            return
        yield from self.stream_responses.pop(0)

    def _next_response(self) -> HttpResponse:
        if not self.responses:
            return HttpResponse(status_code=200, body=b"{}")
        return self.responses.pop(0)


@pytest.fixture
def fake_transport() -> FakeHttpTransport:
    """A fresh fake transport for each test."""
    return FakeHttpTransport()
