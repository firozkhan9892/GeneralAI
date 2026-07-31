"""HTTP tools: issue arbitrary HTTP requests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.exceptions import ToolValidationError
from app.tools.models import ToolCategory, ToolParameter
from app.tools.network import HttpClient, UrllibHttpClient


class HttpRequestTool(Tool):
    """Issue an HTTP request and return the raw response."""

    name = "http_request"
    description = "Issue an HTTP request (method, headers, JSON payload)"
    category = ToolCategory.HTTP
    requires_confirmation = True
    sandboxable = True
    parameters = (
        ToolParameter(
            name="url",
            description="Target URL",
            param_type="string",
            required=True,
        ),
        ToolParameter(
            name="method",
            description="HTTP method",
            param_type="string",
            default="GET",
        ),
        ToolParameter(
            name="headers",
            description="Request headers",
            param_type="object",
        ),
        ToolParameter(
            name="payload",
            description="JSON-serialisable request body",
            param_type="object",
        ),
        ToolParameter(
            name="timeout_s",
            description="Request timeout in seconds",
            param_type="number",
            default=10.0,
        ),
    )

    def __init__(self, client: HttpClient | None = None) -> None:
        self._client: HttpClient = client or UrllibHttpClient()

    def run(
        self, arguments: Mapping[str, Any], context: ToolContext | None = None
    ) -> Any:
        url = arguments["url"]
        if not url.startswith(("http://", "https://")):
            raise ToolValidationError(
                f"Invalid URL: {url}",
                module="tools.categories.http",
            )
        method = arguments.get("method", "GET").upper()
        headers = arguments.get("headers") or {}
        payload = arguments.get("payload")
        timeout_s = float(arguments.get("timeout_s", 10.0))
        response = self._client.request(
            method,
            url,
            headers=headers,
            payload=payload,
            timeout_s=timeout_s,
        )
        return {
            "status": response.status_code,
            "headers": response.headers,
            "body": response.body,
        }
