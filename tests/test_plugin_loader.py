"""Tests for PluginLoader."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from app.core.plugins import PluginLoader


class TestPluginLoader:
    """Suite for plugin discovery and loading."""

    @pytest.fixture
    def temp_plugin_dir(self) -> Path:
        tmp = Path(tempfile.mkdtemp())
        plugin_dir = tmp / "plugins"
        plugin_dir.mkdir()
        return plugin_dir

    def _create_manifest(
        self, directory: Path, name: str, deps: list[str] | None = None
    ) -> Path:
        manifest = directory / "plugin.json"
        data = {
            "name": name,
            "version": "1.0.0",
            "dependencies": deps or [],
            "enabled": True,
        }
        manifest.write_text(json.dumps(data), encoding="utf-8")
        return manifest

    def test_initial_state(self) -> None:
        loader = PluginLoader(plugin_dirs=[])
        assert loader.discovered == {}
        assert loader.loaded == {}
        assert loader.failed == {}

    def test_discover_directory_plugins(self, temp_plugin_dir: Path) -> None:
        plugin_a = temp_plugin_dir / "plugin_a"
        plugin_a.mkdir()
        self._create_manifest(plugin_a, "plugin-a")

        loader = PluginLoader(plugin_dirs=[str(temp_plugin_dir)])
        discovered = loader.discover()
        assert "plugin-a" in discovered
        assert discovered["plugin-a"].version == "1.0.0"

    def test_discover_skips_missing_manifest(self, temp_plugin_dir: Path) -> None:
        empty_dir = temp_plugin_dir / "no_manifest"
        empty_dir.mkdir()

        loader = PluginLoader(plugin_dirs=[str(temp_plugin_dir)])
        discovered = loader.discover()
        # Should discover nothing (no valid manifests)
        assert len(discovered) == 0

    def test_discover_skips_invalid_manifest(self, temp_plugin_dir: Path) -> None:
        bad = temp_plugin_dir / "bad_plugin"
        bad.mkdir()
        (bad / "plugin.json").write_text("not json", encoding="utf-8")

        loader = PluginLoader(plugin_dirs=[str(temp_plugin_dir)])
        discovered = loader.discover()
        # Should silently skip invalid JSON
        assert len(discovered) == 0

    def test_validate_dependencies(self, temp_plugin_dir: Path) -> None:
        plugin_a = temp_plugin_dir / "plugin_a"
        plugin_a.mkdir()
        self._create_manifest(plugin_a, "plugin-a")

        plugin_b = temp_plugin_dir / "plugin_b"
        plugin_b.mkdir()
        self._create_manifest(plugin_b, "plugin-b", deps=["plugin-a"])

        loader = PluginLoader(plugin_dirs=[str(temp_plugin_dir)])
        loader.discover()
        # Should not raise — deps exist
        loader._validate_dependencies()

    def test_validate_dependencies_missing(self, temp_plugin_dir: Path) -> None:
        plugin = temp_plugin_dir / "orphan"
        plugin.mkdir()
        self._create_manifest(plugin, "orphan", deps=["nonexistent"])

        loader = PluginLoader(plugin_dirs=[str(temp_plugin_dir)])
        loader.discover()
        with pytest.raises(Exception, match="depends on 'nonexistent'"):
            loader.load_all()

    def test_resolve_load_order(self, temp_plugin_dir: Path) -> None:
        """Topological sort: deps come first."""
        a = temp_plugin_dir / "a"
        a.mkdir()
        self._create_manifest(a, "a")
        b = temp_plugin_dir / "b"
        b.mkdir()
        self._create_manifest(b, "b", deps=["a"])
        c = temp_plugin_dir / "c"
        c.mkdir()
        self._create_manifest(c, "c", deps=["a", "b"])

        loader = PluginLoader(plugin_dirs=[str(temp_plugin_dir)])
        loader.discover()
        order = loader._resolve_load_order()
        # a must come before b, and b before c
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")

    def test_get_metadata_nonexistent(self) -> None:
        loader = PluginLoader(plugin_dirs=[])
        assert loader.get_metadata("missing") is None

    def test_get_plugin_nonexistent(self) -> None:
        loader = PluginLoader(plugin_dirs=[])
        assert loader.get_plugin("missing") is None
