"""Tests for application settings."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config.settings import AppSettings


class TestAppSettings:
    """Suite for AppSettings validation and behaviour."""

    def test_defaults(self) -> None:
        """Settings should load with sensible defaults."""
        settings = AppSettings()
        assert settings.app_name == "GeneralAI"
        assert settings.app_version == "0.1.0"
        assert settings.log_level == "INFO"
        assert settings.environment == "development"
        assert settings.debug is False

    def test_environment_validation(self) -> None:
        """Invalid environments should be rejected."""
        with pytest.raises(ValueError, match="Invalid environment"):
            AppSettings(environment="invalid_env")

    def test_valid_environments(self) -> None:
        """All valid environments should be accepted."""
        for env in ("development", "staging", "production"):
            settings = AppSettings(environment=env)
            assert settings.environment == env

    def test_log_level_validation(self) -> None:
        """Invalid log levels should be rejected."""
        with pytest.raises(ValueError, match="Invalid log level"):
            AppSettings(log_level="TRACE")

    def test_valid_log_levels(self) -> None:
        """All standard log levels should be accepted."""
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            settings = AppSettings(log_level=level)
            assert settings.log_level == level

    def test_is_development(self) -> None:
        """is_development should return True only for development."""
        assert AppSettings(environment="development").is_development is True
        assert AppSettings(environment="production").is_development is False

    def test_is_production(self) -> None:
        """is_production should return True only for production."""
        assert AppSettings(environment="production").is_production is True
        assert AppSettings(environment="development").is_production is False

    def test_resolve_path_absolute(self) -> None:
        """resolve_path should pass through absolute paths unchanged."""
        settings = AppSettings()
        path = Path("C:/absolute/path")
        assert settings.resolve_path(path) == path

    def test_resolve_path_relative(self) -> None:
        """resolve_path should resolve relative paths against project_root."""
        settings = AppSettings()
        relative = Path("some/relative/path")
        expected = (settings.project_root / relative).resolve()
        assert settings.resolve_path(relative) == expected

    def test_ensure_directories_creates_folders(self, tmp_path: Path) -> None:
        """ensure_directories should create declared directories."""
        settings = AppSettings(
            log_dir=tmp_path / "logs",
            data_dir=tmp_path / "data",
            models_dir=tmp_path / "models",
            project_root=tmp_path,
        )
        settings.ensure_directories()
        assert (tmp_path / "logs").is_dir()
        assert (tmp_path / "data").is_dir()
        assert (tmp_path / "models").is_dir()
