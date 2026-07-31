"""Application settings management.

Uses Pydantic's BaseSettings to load configuration from environment
variables, .env files, and programmatic defaults.  Every configurable
value is typed and validated at startup.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config.defaults import (
    APP_NAME,
    APP_VERSION,
    APP_DESCRIPTION,
    ENV_PREFIX,
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOG_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_MODELS_DIR,
    DEFAULT_ENVIRONMENT,
    ENVIRONMENTS,
)


class AppSettings(BaseSettings):
    """Root configuration for the GeneralAI application.

    Every field can be overridden via an environment variable prefixed
    with ``GENERAL_AI_`` (e.g. ``GENERAL_AI_LOG_LEVEL=DEBUG``) or by
    placing a ``.env`` file in the project root.
    """

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Application metadata
    # ------------------------------------------------------------------
    app_name: str = Field(default=APP_NAME, frozen=True)
    app_version: str = Field(default=APP_VERSION, frozen=True)
    app_description: str = Field(default=APP_DESCRIPTION, frozen=True)

    # ------------------------------------------------------------------
    # Runtime environment
    # ------------------------------------------------------------------
    environment: str = Field(default=DEFAULT_ENVIRONMENT)
    debug: bool = Field(default=False)

    @field_validator("environment")
    @classmethod
    def _validate_environment(cls, value: str) -> str:
        lowered = value.lower()
        if lowered not in ENVIRONMENTS:
            raise ValueError(
                f"Invalid environment '{value}'. Must be one of {ENVIRONMENTS}."
            )
        return lowered

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    project_root: Path = Field(default_factory=lambda: Path.cwd().resolve())
    log_dir: Path = Field(default=DEFAULT_LOG_DIR)
    data_dir: Path = Field(default=DEFAULT_DATA_DIR)
    models_dir: Path = Field(default=DEFAULT_MODELS_DIR)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: str = Field(default=DEFAULT_LOG_LEVEL)
    log_to_console: bool = Field(default=True)
    log_to_file: bool = Field(default=True)

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = value.upper()
        if upper not in allowed:
            raise ValueError(f"Invalid log level '{value}'. Must be one of {allowed}.")
        return upper

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------
    @property
    def is_development(self) -> bool:
        """Return True when running in development mode."""
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        """Return True when running in production mode."""
        return self.environment == "production"

    def resolve_path(self, sub_path: str | Path) -> Path:
        """Resolve *sub_path* relative to the project root.

        Args:
            sub_path: Relative or absolute path to resolve.

        Returns:
            Absolute :class:`Path` resolved against the project root.
        """
        candidate = Path(sub_path)
        if candidate.is_absolute():
            return candidate
        return (self.project_root / candidate).resolve()

    def ensure_directories(self) -> None:
        """Create all required directories if they do not exist."""
        for directory in (self.log_dir, self.data_dir, self.models_dir):
            self.resolve_path(directory).mkdir(parents=True, exist_ok=True)
