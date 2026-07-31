"""Plugin metadata model."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PluginMetadata(BaseModel):
    """Structured metadata for a discovered plugin.

    This model is populated from the plugin's manifest (``plugin.json``)
    or from package entry-point metadata.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Unique plugin name")
    version: str = Field(default="0.1.0", description="Semantic version")
    description: str = Field(default="", description="Human-readable description")
    author: str = Field(default="", description="Plugin author")
    dependencies: list[str] = Field(
        default_factory=list,
        description="List of plugin names this plugin depends on",
    )
    enabled: bool = Field(default=True, description="Whether the plugin is enabled")
    module: str = Field(default="", description="Python module path")
    package: str = Field(default="", description="Python package name (if any)")
