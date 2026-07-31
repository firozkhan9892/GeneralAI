"""Tests for HTTP transport, SSE helpers, and error mapping."""

from __future__ import annotations

import json
import urllib.error
from email.message import Message
from unittest import mock

import pytest

from app.llm.exceptions import (
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from app.llm.transport import (
    HttpResponse,
    UrllibHttpTransport,
    iter_lines,
    iter_sse_payloads,
)


class TestHttpResponse:
    def test_json(self) -> None:
        response = HttpResponse(status_code=200, body=b'{"a": 1}')
        assert response.json() == {"a": 1}

    def test_json_invalid(self) -> None:
        response = HttpResponse(status_code=200, body=b"not json")
        with pytest.raises(ProviderResponseError):
            response.json()

    def test_json_non_object(self) -> None:
        response = HttpResponse(status_code=200, body=b"[1,2]")
        with pytest.raises(ProviderResponseError):
            response.json()


class TestIterLines:
    def test_basic(self) -> None:
        assert list(iter_lines(iter([b"a\nb\n"]))) == ["a", "b"]

    def test_chunked_lines(self) -> None:
        chunks = [b"hel", b"lo\nwo", b"rld\n"]
        assert list(iter_lines(iter(chunks))) == ["hello", "world"]

    def test_skips_blank_lines(self) -> None:
        assert list(iter_lines(iter([b"\n\n\na\n\n"]))) == ["a"]

    def test_trailing_without_newline(self) -> None:
        assert list(iter_lines(iter([b"abc"]))) == ["abc"]

    def test_empty(self) -> None:
        assert list(iter_lines(iter([b"", b""]))) == []


class TestIterSsePayloads:
    def test_extracts_data(self) -> None:
        lines = ["data: {a}", "data:  {b}", "event: foo"]
        assert list(iter_sse_payloads(iter(lines))) == ["{a}", "{b}"]

    def test_skips_done(self) -> None:
        lines = ["data: x", "data: [DONE]"]
        assert list(iter_sse_payloads(iter(lines))) == ["x"]

    def test_skips_non_data(self) -> None:
        lines = ["plain line", ": comment", "data: x"]
        assert list(iter_sse_payloads(iter(lines))) == ["x"]

    def test_empty_payload_skipped(self) -> None:
        assert list(iter_sse_payloads(iter(["data:", "data: "]))) == []


class TestUrllibTransport:
    def test_post_builds_request(self) -> None:
        payload = {"model": "gpt"}
        transport = UrllibHttpTransport()
        with mock.patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.status = 200
            urlopen.return_value.__enter__.return_value.read.return_value = (
                b'{"ok": true}'
            )
            urlopen.return_value.__enter__.return_value.headers = {}
            response = transport.post(
                "https://example.com/chat",
                headers={"Authorization": "Bearer x"},
                payload=payload,
                timeout_s=10,
            )
        request = urlopen.call_args.args[0]
        assert request.get_method() == "POST"
        assert request.get_header("Authorization") == "Bearer x"
        assert request.get_header("Content-type") == "application/json"
        assert json.loads(request.data) == payload
        assert response.status_code == 200
        assert response.json() == {"ok": True}

    def test_post_auth_error(self) -> None:
        transport = UrllibHttpTransport()
        with mock.patch("urllib.request.urlopen") as urlopen:
            urlopen.side_effect = urllib.error.HTTPError(
                "https://example.com", 401, "Unauthorized", Message(), None
            )
            with pytest.raises(ProviderAuthenticationError):
                transport.post(
                    "https://example.com/chat",
                    headers={},
                    payload={},
                    timeout_s=10,
                )

    def test_post_rate_limit(self) -> None:
        transport = UrllibHttpTransport()
        with mock.patch("urllib.request.urlopen") as urlopen:
            urlopen.side_effect = urllib.error.HTTPError(
                "https://example.com", 429, "Too Many Requests", Message(), None
            )
            with pytest.raises(ProviderRateLimitError):
                transport.post(
                    "https://example.com/chat",
                    headers={},
                    payload={},
                    timeout_s=10,
                )

    def test_post_connection_error(self) -> None:
        transport = UrllibHttpTransport()
        with mock.patch("urllib.request.urlopen") as urlopen:
            urlopen.side_effect = urllib.error.URLError("no route to host")
            with pytest.raises(ProviderConnectionError):
                transport.post(
                    "https://example.com/chat",
                    headers={},
                    payload={},
                    timeout_s=10,
                )

    def test_post_timeout(self) -> None:
        transport = UrllibHttpTransport()
        with mock.patch("urllib.request.urlopen") as urlopen:
            urlopen.side_effect = TimeoutError("timed out")
            with pytest.raises(ProviderTimeoutError):
                transport.post(
                    "https://example.com/chat",
                    headers={},
                    payload={},
                    timeout_s=10,
                )

    def test_post_response_error(self) -> None:
        transport = UrllibHttpTransport()
        with mock.patch("urllib.request.urlopen") as urlopen:
            urlopen.side_effect = urllib.error.HTTPError(
                "https://example.com", 400, "Bad Request", Message(), None
            )
            with pytest.raises(ProviderResponseError):
                transport.post(
                    "https://example.com/chat",
                    headers={},
                    payload={},
                    timeout_s=10,
                )

    def test_post_stream_yields_chunks(self) -> None:
        transport = UrllibHttpTransport()
        with mock.patch("urllib.request.urlopen") as urlopen:
            context = urlopen.return_value.__enter__.return_value
            context.read.side_effect = [b"data: {", b'"a":1}\n\n', b""]
            chunks = list(
                transport.post_stream(
                    "https://example.com/chat",
                    headers={},
                    payload={},
                    timeout_s=10,
                )
            )
        assert b"".join(chunks) == b'data: {"a":1}\n\n'

    def test_post_stream_auth_error(self) -> None:
        transport = UrllibHttpTransport()
        with mock.patch("urllib.request.urlopen") as urlopen:
            urlopen.side_effect = urllib.error.HTTPError(
                "https://example.com", 401, "Unauthorized", Message(), None
            )
            with pytest.raises(ProviderAuthenticationError):
                list(
                    transport.post_stream(
                        "https://example.com/chat",
                        headers={},
                        payload={},
                        timeout_s=10,
                    )
                )
