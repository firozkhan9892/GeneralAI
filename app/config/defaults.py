"""Default configuration values and environment variable names.

Centralises all hardcoded defaults to a single location so that
modules remain configurable and testable.
"""

from pathlib import Path
from typing import Final

# ------------------------------------------------------------------
# Application metadata
# ------------------------------------------------------------------
APP_NAME: Final[str] = "GeneralAI"
APP_VERSION: Final[str] = "0.1.0"
APP_DESCRIPTION: Final[str] = "Autonomous AI Platform"

# ------------------------------------------------------------------
# Environment variable names
# ------------------------------------------------------------------
ENV_PREFIX: Final[str] = "GENERAL_AI_"
ENV_CONFIG_PATH: Final[str] = f"{ENV_PREFIX}CONFIG_PATH"
ENV_LOG_LEVEL: Final[str] = f"{ENV_PREFIX}LOG_LEVEL"
ENV_LOG_DIR: Final[str] = f"{ENV_PREFIX}LOG_DIR"
ENV_DEBUG: Final[str] = f"{ENV_PREFIX}DEBUG"
ENV_ENVIRONMENT: Final[str] = f"{ENV_PREFIX}ENVIRONMENT"

# ------------------------------------------------------------------
# Path defaults (relative to project root)
# ------------------------------------------------------------------
DEFAULT_CONFIG_DIR: Final[str] = "config"
DEFAULT_LOG_DIR: Final[Path] = Path("logs")
DEFAULT_DATA_DIR: Final[Path] = Path("data")
DEFAULT_MODELS_DIR: Final[Path] = Path("models")

# ------------------------------------------------------------------
# Logging defaults
# ------------------------------------------------------------------
DEFAULT_LOG_LEVEL: Final[str] = "INFO"
DEFAULT_LOG_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
DEFAULT_LOG_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
LOG_MAX_BYTES: Final[int] = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT: Final[int] = 5

# ------------------------------------------------------------------
# Runtime
# ------------------------------------------------------------------
ENVIRONMENTS: Final[list[str]] = ["development", "staging", "production"]
DEFAULT_ENVIRONMENT: Final[str] = "development"
