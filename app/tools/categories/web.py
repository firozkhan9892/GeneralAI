"""Web tools: fetch remote resources."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.exceptions import ToolValidationError
from app.tools.models import ToolCategory, ToolParameter
from app.tools.network import HttpClient, UrllibHttpClient


class WebFetchTool(Tool):
    """Fetch the body of a URL as text."""

    name = "web_fetch"
    description = "Fetch the text content of a URL"
    category = ToolCategory.WEB
    requires_confirmation = True
    sandboxable = True
    parameters = (
        ToolParameter(
            name="url",
            description="URL to fetch",
            param_type="string",
            required=True,
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
                module="tools.categories.web",
            )
        timeout_s = float(arguments.get("timeout_s", 10.0))
        response = self._client.request("GET", url, timeout_s=timeout_s)
        return {
            "status": response.status_code,
            "body": response.body,
        }
