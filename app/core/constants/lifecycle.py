"""Lifecycle stage constants and enums."""

from __future__ import annotations

from enum import Enum


class LifecycleStage(str, Enum):
    """Ordered stages of the application lifecycle.

    Stages execute sequentially during startup and in reverse
    during shutdown.
    """

    CREATED = "created"
    CONFIG_LOADING = "config_loading"
    CONFIG_LOADED = "config_loaded"
    SERVICES_INITIALIZING = "services_initializing"
    SERVICES_INITIALIZED = "services_initialized"
    PLUGINS_LOADING = "plugins_loading"
    PLUGINS_LOADED = "plugins_loaded"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"


# Map of forward lifecycle transitions (includes reverse paths for shutdown)
LIFECYCLE_TRANSITIONS: dict[LifecycleStage, list[LifecycleStage]] = {
    LifecycleStage.CREATED: [LifecycleStage.CONFIG_LOADING],
    LifecycleStage.CONFIG_LOADING: [LifecycleStage.CONFIG_LOADED],
    LifecycleStage.CONFIG_LOADED: [LifecycleStage.SERVICES_INITIALIZING],
    LifecycleStage.SERVICES_INITIALIZING: [LifecycleStage.SERVICES_INITIALIZED],
    LifecycleStage.SERVICES_INITIALIZED: [LifecycleStage.PLUGINS_LOADING],
    LifecycleStage.PLUGINS_LOADING: [LifecycleStage.PLUGINS_LOADED],
    LifecycleStage.PLUGINS_LOADED: [LifecycleStage.STARTING],
    LifecycleStage.STARTING: [LifecycleStage.RUNNING],
    LifecycleStage.RUNNING: [LifecycleStage.STOPPING],
    LifecycleStage.STOPPING: [LifecycleStage.STOPPED],
    LifecycleStage.STOPPED: [],
}

# Shutdown transitions — allow stopping from any stage past CREATED
SHUTDOWN_TRANSITIONS: dict[LifecycleStage, list[LifecycleStage]] = {
    LifecycleStage.CONFIG_LOADED: [LifecycleStage.STOPPING],
    LifecycleStage.SERVICES_INITIALIZED: [LifecycleStage.STOPPING],
    LifecycleStage.PLUGINS_LOADED: [LifecycleStage.STOPPING],
}

# Reverse ordering for shutdown
SHUTDOWN_ORDER: list[LifecycleStage] = [
    LifecycleStage.RUNNING,
    LifecycleStage.STARTING,
    LifecycleStage.PLUGINS_LOADED,
    LifecycleStage.PLUGINS_LOADING,
    LifecycleStage.SERVICES_INITIALIZED,
    LifecycleStage.SERVICES_INITIALIZING,
    LifecycleStage.CONFIG_LOADED,
    LifecycleStage.CONFIG_LOADING,
    LifecycleStage.CREATED,
]

# Hook point names (one per stage transition)
HOOK_BEFORE_INIT = "before_init"
HOOK_AFTER_CONFIG = "after_config_loaded"
HOOK_AFTER_SERVICES = "after_services_initialized"
HOOK_AFTER_PLUGINS = "after_plugins_loaded"
HOOK_BEFORE_START = "before_start"
HOOK_AFTER_START = "after_start"
HOOK_BEFORE_STOP = "before_stop"
HOOK_AFTER_STOP = "after_stop"
