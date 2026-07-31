"""Brain interface placeholder."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from app.core.interfaces.base import IModule


class IBrain(IModule):
    """Contract for cognitive core implementations.

    Future implementations will process input, maintain context,
    and produce reasoning output.  The interface deliberately
    stays minimal until the cognitive architecture is designed.
    """

    @abstractmethod
    async def process(
        self, input_data: Any, context: dict[str, Any] | None = None
    ) -> Any:
        """Process *input_data* and return a result.

        Args:
            input_data: The input to process (type depends on the
                concrete brain implementation).
            context: Optional contextual key-value pairs.

        Returns:
            The processing result.
        """
