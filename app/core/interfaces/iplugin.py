"""Plugin interface."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from app.core.interfaces.base import IModule


class IPlugin(IModule):
    """Contract that every plugin must satisfy.

    Plugins extend the platform with new capabilities.  Each plugin
    exposes metadata and implements the standard module lifecycle.
    """

    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return this plugin's metadata dict.

        Must include at least:
        - ``name``: str
        - ``version``: str
        - ``description``: str
        - ``dependencies``: list[str]
        """

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """Return whether this plugin is currently enabled."""
