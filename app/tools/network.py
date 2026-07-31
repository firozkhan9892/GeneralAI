"""Minimal HTTP client used by web and HTTP tools.

Kept deliberately small and injectable so tools that perform network
I/O can be unit-tested without touching the real network.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.tools.exceptions import ToolExecutionError


class HttpResponse(BaseModel):
    """A de-serialised HTTP response."""

    model_config = ConfigDict(frozen=True)

    status_code: int = Field(..., description="HTTP status code")
    body: str = Field(default="", description="Decoded response body")
    headers: dict[str, str] = Field(
        default_factory=dict, description="Response headers"
    )

    @property
    def ok(self) -> bool:
        """Return ``True`` for 2xx status codes."""
        return 200 <= self.status_code < 300


class HttpClient(ABC):
    """Abstract client for issuing HTTP requests."""

    @abstractmethod
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        payload: Any = None,
        timeout_s: float = 10.0,
    ) -> HttpResponse:
        """Issue an HTTP request.

        Args:
            method: HTTP method (``GET``, ``POST``, ...).
            url: Target URL.
            headers: Optional request headers.
            payload: Optional JSON-serialisable body.
            timeout_s: Request timeout in seconds.

        Returns:
            The response.

        Raises:
            ToolExecutionError: If the request fails.
        """


class UrllibHttpClient(HttpClient):
    """:class:`HttpClient` backed by the standard library ``urllib``."""

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        payload: Any = None,
        timeout_s: float = 10.0,
    ) -> HttpResponse:
        data: bytes | None = None
        request_headers = dict(headers or {})
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json")
        request = urllib.request.Request(
            url,
            data=data,
            headers=request_headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                raw = response.read()
                response_headers = {key: value for key, value in response.getheaders()}
        except urllib.error.URLError as exc:
            raise ToolExecutionError(
                f"HTTP request to '{url}' failed: {exc}",
                module="tools.network",
                cause=exc,
            ) from exc
        except OSError as exc:
            raise ToolExecutionError(
                f"HTTP request to '{url}' failed: {exc}",
                module="tools.network",
                cause=exc,
            ) from exc
        try:
            body = raw.decode("utf-8")
        except UnicodeDecodeError:
            body = raw.decode("utf-8", errors="replace")
        return HttpResponse(
            status_code=response.status,
            body=body,
            headers=response_headers,
        )
