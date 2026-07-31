"""Server configuration.

Independent of the root :class:`AppSettings` — the FastAPI layer is
configured directly via :class:`ServerSettings`, which can be supplied
programmatically to :func:`app.server.app.create_app`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app import __version__, __app_name__


class ServerSettings(BaseModel):
    """Configuration for the GeneralAI FastAPI server.

    Args:
        title: OpenAPI title.
        version: OpenAPI version string.
        host: Bind host (informational; used by ``uvicorn``).
        port: Bind port (informational; used by ``uvicorn``).
        api_key: Shared API key.  When set, every non-public endpoint
            requires it via the ``X-API-Key`` header.  When ``None``,
            authentication is disabled (development convenience).
        rate_limit_enabled: Whether fixed-window rate limiting applies.
        rate_limit_per_minute: Maximum requests per identity per minute.
        cors_origins: Origins allowed for CORS (empty disables CORS).
    """

    model_config = ConfigDict(frozen=True)

    title: str = Field(default=f"{__app_name__} API", description="OpenAPI title")
    version: str = Field(default=__version__, description="OpenAPI version")
    host: str = Field(default="127.0.0.1", description="Bind host")
    port: int = Field(default=8000, ge=1, le=65535, description="Bind port")
    api_key: str | None = Field(
        default=None, description="Shared API key; None disables auth"
    )
    rate_limit_enabled: bool = Field(
        default=True, description="Enable fixed-window rate limiting"
    )
    rate_limit_per_minute: int = Field(
        default=60, ge=1, le=100000, description="Requests per minute per identity"
    )
    cors_origins: tuple[str, ...] = Field(
        default_factory=tuple, description="Allowed CORS origins"
    )
