"""GeneralAI FastAPI server.

This package provides the REST API layer for the GeneralAI platform.

Key features:
- FastAPI-powered REST API with OpenAPI documentation
- WebSocket support for real-time streaming
- Server-Sent Events (SSE) for background notifications
- API key authentication
- Rate limiting
- Dependency injection integration
- Comprehensive request/response models

The primary entry point is :func:`create_app` in the :mod:`app.server.app`
module, which builds a fully configured FastAPI application.

All routers are organized by domain:
- :mod:`app.server.routers.chat` - chat and streaming endpoints
- :mod:`app.server.routers.agent` - agent management and WebSocket
- :mod:`app.server.routers.health` - health checks
- :mod:`app.server.routers.memory` - memory search
- :mod:`app.server.routers.tools` - tool execution
"""

from app.server.app import create_app
from app.server.config import ServerSettings

__all__ = ["create_app", "ServerSettings"]
