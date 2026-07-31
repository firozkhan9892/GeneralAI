"""Event-related constants and standard event type names."""

from __future__ import annotations

from typing import Final

# Standard event type names used across the platform
EVENT_APP_STARTING: Final[str] = "app.starting"
EVENT_APP_STARTED: Final[str] = "app.started"
EVENT_APP_STOPPING: Final[str] = "app.stopping"
EVENT_APP_STOPPED: Final[str] = "app.stopped"
EVENT_APP_ERROR: Final[str] = "app.error"

EVENT_PLUGIN_LOADING: Final[str] = "plugin.loading"
EVENT_PLUGIN_LOADED: Final[str] = "plugin.loaded"
EVENT_PLUGIN_UNLOADED: Final[str] = "plugin.unloaded"
EVENT_PLUGIN_ERROR: Final[str] = "plugin.error"

EVENT_CONFIG_CHANGED: Final[str] = "config.changed"
EVENT_CONFIG_ERROR: Final[str] = "config.error"

EVENT_SERVICE_REGISTERED: Final[str] = "service.registered"
EVENT_SERVICE_UNREGISTERED: Final[str] = "service.unregistered"

# Reserved prefixes — event types starting with these are system events
SYSTEM_EVENT_PREFIX: Final[str] = "app."
PLUGIN_EVENT_PREFIX: Final[str] = "plugin."

# Event bus configuration defaults
EVENT_BUS_MAX_HANDLERS_PER_EVENT: Final[int] = 100
EVENT_BUS_HANDLER_TIMEOUT: Final[float] = 30.0
