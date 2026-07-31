"""Capability manager — stage 7 of the cognitive pipeline."""

from __future__ import annotations

import logging
from typing import Any

from app.kernel.capability.models import Capability, ProviderType

log = logging.getLogger(__name__)


class CapabilityManager:
    """Resolves actions to capability providers.

    Determines if and how an action can be performed, which provider
    to use, and whether approval or a plugin is required.
    """

    async def resolve(
        self, action_type: str, parameters: dict[str, Any] | None = None
    ) -> Capability:
        """Resolve an action to a capability.

        Args:
            action_type: The type of action to resolve.
            parameters: Optional action parameters.

        Returns:
            Capability resolution with provider info.

        Raises:
            NotImplementedError: Always — placeholder.
        """
        raise NotImplementedError("CapabilityManager.resolve not yet implemented")

    def register_capability(
        self, name: str, provider_type: ProviderType, provider_name: str
    ) -> None:
        """Register a capability provider.

        Args:
            name: Capability name.
            provider_type: Type of provider.
            provider_name: Name of the provider.
        """
        log.debug(
            "Registered capability '%s' -> %s:%s", name, provider_type, provider_name
        )
