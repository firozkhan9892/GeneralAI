"""Plugin-related constants."""

from __future__ import annotations

from typing import Final

# Entry point group used for plugin discovery
PLUGIN_ENTRYPOINT_GROUP: Final[str] = "generalai.plugins"

# Plugin metadata keys
PLUGIN_META_NAME: Final[str] = "name"
PLUGIN_META_VERSION: Final[str] = "version"
PLUGIN_META_DESCRIPTION: Final[str] = "description"
PLUGIN_META_AUTHOR: Final[str] = "author"
PLUGIN_META_DEPENDENCIES: Final[str] = "dependencies"
PLUGIN_META_ENABLED: Final[str] = "enabled"
PLUGIN_META_MODULE: Final[str] = "module"
PLUGIN_META_PACKAGE: Final[str] = "package"

# Plugin state constants
PLUGIN_STATE_DISCOVERED: Final[str] = "discovered"
PLUGIN_STATE_LOADED: Final[str] = "loaded"
PLUGIN_STATE_ENABLED: Final[str] = "enabled"
PLUGIN_STATE_DISABLED: Final[str] = "disabled"
PLUGIN_STATE_ERROR: Final[str] = "error"

# Plugin directory scanning
PLUGIN_SCAN_DIRS: Final[list[str]] = ["plugins"]
PLUGIN_MANIFEST_FILENAME: Final[str] = "plugin.json"
