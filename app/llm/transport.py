"""Injectable HTTP transport for LLM providers.

Providers depend on the abstract :class:`HttpTransport` rather than a
specific HTTP client so that tests can inject fakes and the application
can supply a shared implementation.  A default stdlib implementation
(:class:`UrllibHttpTransport`) is provided for out-of-the-box use.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator

from app.llm.exceptions import (
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HttpResponse:
    """A minimal, provider-agnostic HTTP response."""

    status_code: int
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)

    def json(self) -> dict[str, Any]:
        """Parse the response body as JSON.

        Returns:
            The parsed JSON object.

        Raises:
            ProviderResponseError: If the body is not valid JSON.
        """
        try:
            data = json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderResponseError(
                f"Provider returned invalid JSON: {self.body[:200]!r}",
                module="llm.transport",
            ) from exc
        if not isinstance(data, dict):
            raise ProviderResponseError(
                "Provider returned a non-object JSON body",
                module="llm.transport",
            )
        return data


class HttpTransport(ABC):
    """Abstract synchronous HTTP transport used by providers."""

    @abstractmethod
    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_s: float,
    ) -> HttpResponse:
        """POST a JSON payload and return the response.

        Args:
            url: Destination URL.
            headers: Request headers.
            payload: JSON-serialisable request body.
            timeout_s: Request timeout in seconds.

        Returns:
            The HTTP response.

        Raises:
            ProviderTimeoutError: On timeout.
            ProviderConnectionError: On connection failure.
            ProviderRateLimitError: On HTTP 429.
            ProviderAuthenticationError: On HTTP 401/403.
            ProviderResponseError: On other non-2xx responses.
        """

    @abstractmethod
    def post_stream(
        self,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_s: float,
    ) -> Iterator[bytes]:
        """POST a JSON payload and stream raw response bytes.

        Args:
            url: Destination URL.
            headers: Request headers.
            payload: JSON-serialisable request body.
            timeout_s: Request timeout in seconds.

        Returns:
            Iterator over raw response chunks.

        Raises:
            Same error hierarchy as :meth:`post`.
        """


def iter_lines(chunks: Iterator[bytes]) -> Iterator[str]:
    """Split raw byte chunks into newline-delimited text lines.

    Args:
        chunks: Iterator over raw byte chunks.

    Returns:
        Iterator over decoded, stripped text lines.
    """
    buffer = b""
    for chunk in chunks:
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            line = line.strip()
            if line:
                yield line.decode("utf-8", errors="replace")
    if buffer.strip():
        yield buffer.strip().decode("utf-8", errors="replace")


def iter_sse_payloads(lines: Iterator[str]) -> Iterator[str]:
    """Extract ``data:`` payloads from an SSE line stream.

    Skips blank lines, event metadata, and the ``[DONE]`` sentinel.

    Args:
        lines: Iterator over decoded text lines.

    Returns:
        Iterator over the raw ``data:`` payload strings.
    """
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("data:"):
            continue
        payload = stripped[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            continue
        yield payload


class UrllibHttpTransport(HttpTransport):
    """HTTP transport built on the standard library ``urllib``."""

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_s: float,
    ) -> HttpResponse:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={**headers, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                status = int(response.status)
                raw = response.read()
                response_headers = {
                    key: value for key, value in response.headers.items()
                }
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            raw = exc.read()
            response_headers = {key: value for key, value in exc.headers.items()}
            self._raise_for_status(status, raw, url)
        except TimeoutError as exc:
            raise ProviderTimeoutError(
                f"Request to {url} timed out after {timeout_s}s",
                module="llm.transport",
            ) from exc
        except urllib.error.URLError as exc:
            reason = exc.reason
            raise ProviderConnectionError(
                f"Failed to reach {url}: {reason}",
                module="llm.transport",
                cause=reason if isinstance(reason, Exception) else None,
            ) from exc
        self._raise_for_status(status, raw, url)
        return HttpResponse(
            status_code=status,
            body=raw,
            headers=response_headers,
        )

    def post_stream(
        self,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_s: float,
    ) -> Iterator[bytes]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={**headers, "Content-Type": "application/json"},
        )
        try:
            response = urllib.request.urlopen(request, timeout=timeout_s)
        except urllib.error.HTTPError as exc:
            self._raise_for_status(int(exc.code), exc.read(), url)
            return
        except TimeoutError as exc:
            raise ProviderTimeoutError(
                f"Request to {url} timed out after {timeout_s}s",
                module="llm.transport",
            ) from exc
        except urllib.error.URLError as exc:
            reason = exc.reason
            raise ProviderConnectionError(
                f"Failed to reach {url}: {reason}",
                module="llm.transport",
                cause=reason if isinstance(reason, Exception) else None,
            ) from exc
        with response as stream:
            while True:
                chunk = stream.read(4096)
                if not chunk:
                    break
                yield chunk

    @staticmethod
    def _raise_for_status(status: int, raw: bytes, url: str) -> None:
        """Translate a non-2xx status code into a provider exception."""
        if 200 <= status < 300:
            return
        message = raw[:300].decode("utf-8", errors="replace")
        if status == 401 or status == 403:
            raise ProviderAuthenticationError(
                f"Authentication failed for {url} ({status}): {message}",
                module="llm.transport",
            )
        if status == 429:
            raise ProviderRateLimitError(
                f"Rate limited by {url} ({status}): {message}",
                module="llm.transport",
            )
        if status >= 500:
            raise ProviderConnectionError(
                f"Provider server error at {url} ({status}): {message}",
                module="llm.transport",
            )
        raise ProviderResponseError(
            f"Provider returned status {status} for {url}: {message}",
            module="llm.transport",
        )
