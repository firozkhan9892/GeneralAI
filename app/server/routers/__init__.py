"""Server routers."""

from __future__ import annotations

from app.server.routers.chat import router as chat_router
from app.server.routers.agent import router as agent_router
from app.server.routers.health import router as health_router
from app.server.routers.memory import router as memory_router
from app.server.routers.tools import router as tools_router

__all__ = [
    "chat_router",
    "agent_router",
    "health_router",
    "memory_router",
    "tools_router",
]
