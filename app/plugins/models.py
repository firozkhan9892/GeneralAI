"""Plugin domain models.

Extends the lightweight metadata in :mod:`app.core.plugins.plugin_metadata`
with richer lifecycle, dependency and registration models used by the
:class:`PluginManager`.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.plugins.plugin_metadata import PluginMetadata


class PluginType(str, Enum):
    """Category of capability a plugin provides."""

    TOOL = "tool"
    AGENT = "agent"
    WORKFLOW = "workflow"
    API_ROUTE = "api_route"
    MEMORY_PROVIDER = "memory_provider"
    LLM_PROVIDER = "llm_provider"
    MIXED = "mixed"


class PluginLifecycleState(str, Enum):
    """Lifecycle stages a plugin can be in."""

    INSTALLED = "installed"
    LOADED = "loaded"
    ENABLED = "enabled"
    DISABLED = "disabled"
    UNLOADED = "unloaded"
    ERROR = "error"


class PluginDependency(BaseModel):
    """A versioned dependency on another plugin.

    Args:
        name: The dependency plugin name.
        version_spec: PEP 440 version specifier set (e.g. ``>=1.0.0``,``~=2.1``).
            Empty string means any version is acceptable.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Dependency plugin name")
    version_spec: str = Field(
        default="",
        description="PEP 440 version specifier (e.g. '>=1.0.0'); empty = any",
    )

    def matches(self, version: str) -> bool:
        """Return ``True`` when *version* satisfies the constraint."""
        if not self.version_spec:
            return True
        try:
            from packaging.specifiers import SpecifierSet
            from packaging.version import Version

            spec = SpecifierSet(self.version_spec)
            return Version(version) in spec
        except Exception:
            return False


class PluginManifest(BaseModel):
    """Extended manifest with richer dependency and type information.

    Backward-compat: accepts the same fields as :class:`PluginMetadata`
    (including ``dependencies: list[str]``) plus the new ``plugin_type``
    and structured ``plugin_dependencies``.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Unique plugin name")
    version: str = Field(default="0.1.0", description="Semantic version")
    description: str = Field(default="", description="Human-readable description")
    author: str = Field(default="", description="Plugin author")
    plugin_type: PluginType = Field(
        default=PluginType.MIXED, description="Capability category"
    )
    enabled: bool = Field(default=True, description="Whether the plugin is enabled")
    module: str = Field(default="", description="Python module path")
    package: str = Field(default="", description="Python package name (if any)")
    dependencies: list[str] = Field(
        default_factory=list,
        description="Legacy: list of dependency plugin names (any version)",
    )
    plugin_dependencies: list[PluginDependency] = Field(
        default_factory=list,
        description="Structured dependencies with version constraints",
    )
    generalai_version: str = Field(
        default="",
        description="Minimum GeneralAI version required (e.g. '>=1.0.0')",
    )

    @classmethod
    def from_metadata(cls, metadata: PluginMetadata) -> PluginManifest:
        """Build a manifest from a legacy :class:`PluginMetadata`."""
        return cls(
            name=metadata.name,
            version=metadata.version,
            description=metadata.description,
            author=metadata.author,
            enabled=metadata.enabled,
            module=metadata.module,
            package=metadata.package,
            dependencies=list(metadata.dependencies),
        )

    @property
    def effective_dependencies(self) -> list[PluginDependency]:
        """Return merged dependency list (structured + legacy)."""
        result = list(self.plugin_dependencies)
        for dep_name in self.dependencies:
            if not any(d.name == dep_name for d in result):
                result.append(PluginDependency(name=dep_name))
        return result


class PluginRegistration(BaseModel):
    """Tracks a single capability registration made by a plugin.

    Args:
        plugin_name: Name of the plugin that owns this registration.
        plugin_type: What kind of capability was registered.
        registration_id: Identifier used to remove the registration.
        registry_target: Human-readable target (e.g. ``tool:echo``).
        extra: Additional context about the registration.
    """

    model_config = ConfigDict(frozen=True)

    plugin_name: str = Field(..., description="Owning plugin name")
    plugin_type: PluginType = Field(..., description="Capability type")
    registration_id: str = Field(..., description="Identifier for unregistration")
    registry_target: str = Field(..., description="Target registry identifier")
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Additional registration metadata"
    )


class PluginState(BaseModel):
    """Mutable runtime state of a loaded plugin.

    Args:
        name: Plugin name.
        version: Plugin version string.
        plugin_type: Capability category.
        lifecycle_state: Current lifecycle stage.
        installed_at: Timestamp when install completed.
        loaded_at: Timestamp when the module was loaded.
        enabled_at: Timestamp when enabled.
        disabled_at: Timestamp when disabled.
        manifest: The source manifest (None before install completes).
        error: Error message if in ERROR state.
    """

    model_config = ConfigDict(frozen=False)

    name: str = Field(..., description="Plugin name")
    version: str = Field(default="0.1.0", description="Plugin version")
    plugin_type: PluginType = Field(
        default=PluginType.MIXED, description="Capability category"
    )
    lifecycle_state: PluginLifecycleState = Field(
        default=PluginLifecycleState.INSTALLED, description="Current lifecycle state"
    )
    installed_at: datetime | None = Field(default=None, description="Install timestamp")
    loaded_at: datetime | None = Field(default=None, description="Load timestamp")
    enabled_at: datetime | None = Field(default=None, description="Enable timestamp")
    disabled_at: datetime | None = Field(default=None, description="Disable timestamp")
    manifest: PluginManifest | None = Field(default=None, description="Source manifest")
    error: str | None = Field(
        default=None, description="Error message if in ERROR state"
    )
