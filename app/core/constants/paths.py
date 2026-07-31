"""Path and directory constants."""

from __future__ import annotations

from typing import Final

# Reserved registry keys
REGISTRY_MODELS: Final[str] = "models"
REGISTRY_TOOLS: Final[str] = "tools"
REGISTRY_AGENTS: Final[str] = "agents"
REGISTRY_PLUGINS: Final[str] = "plugins"
REGISTRY_WORKFLOWS: Final[str] = "workflows"
REGISTRY_EVENTS: Final[str] = "events"

# Default capacity limits
REGISTRY_DEFAULT_MAX_ITEMS: Final[int] = 10_000
PLUGIN_MAX_DEPENDENCY_DEPTH: Final[int] = 20

# Container defaults
CONTAINER_DEFAULT_MAX_RESOLVE_DEPTH: Final[int] = 32
