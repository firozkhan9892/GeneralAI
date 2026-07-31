"""Base abstraction for all tools.

A :class:`Tool` declares its own metadata and parameters, validates and
coerces its arguments, and exposes a synchronous ``run`` plus a default
asynchronous ``arun`` that offloads to a worker thread.  Concrete tools
only need to declare metadata and implement ``run``.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from app.tools.context import ToolContext
from app.tools.exceptions import ToolValidationError
from app.tools.models import ToolCategory, ToolMetadata, ToolParameter

# JSON-schema style type names mapped to Python coercers.
_COERCERS: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}


class Tool(ABC):
    """Abstract contract for executable tools.

    Subclasses declare class-level metadata (``name``, ``description``,
    ``category``, ``parameters``) and implement :meth:`run`.  The base
    class provides argument validation/coercion, metadata construction,
    and a thread-offloaded async variant.
    """

    name: str = ""
    description: str = ""
    category: ToolCategory = ToolCategory.BUILTIN
    version: str = "1.0.0"
    parameters: tuple[ToolParameter, ...] = ()
    timeout_s: float = 30.0
    requires_confirmation: bool = False
    sandboxable: bool = False

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> ToolMetadata:
        """Return a descriptor describing this tool."""
        return ToolMetadata(
            name=self.name,
            description=self.description,
            category=self.category,
            version=self.version,
            parameters=self.parameters,
            timeout_s=self.timeout_s,
            requires_confirmation=self.requires_confirmation,
            sandboxable=self.sandboxable,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and coerce *arguments* against declared parameters.

        Args:
            arguments: Raw invocation arguments.

        Returns:
            A cleaned mapping with coerced values and defaults applied.

        Raises:
            ToolValidationError: If a required parameter is missing, an
                unknown parameter is supplied, or coercion fails.
        """
        if not arguments:
            arguments = {}

        declared = {param.name: param for param in self.parameters}

        unknown = set(arguments) - set(declared)
        if unknown:
            raise ToolValidationError(
                f"Unknown parameter(s): {', '.join(sorted(unknown))}",
                module="tools.base",
            )

        cleaned: dict[str, Any] = {}
        for param in self.parameters:
            if param.name in arguments:
                cleaned[param.name] = self._coerce(param, arguments[param.name])
            elif param.required:
                raise ToolValidationError(
                    f"Missing required parameter '{param.name}'",
                    module="tools.base",
                )
            elif param.default is not None:
                cleaned[param.name] = param.default
        return cleaned

    @staticmethod
    def _coerce(param: ToolParameter, value: Any) -> Any:
        """Coerce *value* to the declared type of *param*."""
        coercer = _COERCERS.get(param.param_type)
        if coercer is None:
            return value
        if isinstance(value, coercer):
            return value
        try:
            if param.param_type == "boolean":
                return _coerce_boolean(value)
            return coercer(value)
        except (TypeError, ValueError) as exc:
            raise ToolValidationError(
                f"Parameter '{param.name}' must be {param.param_type}",
                module="tools.base",
            ) from exc

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    @abstractmethod
    def run(
        self, arguments: Mapping[str, Any], context: ToolContext | None = None
    ) -> Any:
        """Execute the tool synchronously.

        Args:
            arguments: Validated invocation arguments.
            context: Optional execution context.

        Returns:
            The tool output.

        Raises:
            ToolError: On any tool-level failure.
        """

    async def arun(
        self, arguments: Mapping[str, Any], context: ToolContext | None = None
    ) -> Any:
        """Execute the tool asynchronously.

        The default implementation offloads :meth:`run` to a worker
        thread.  Tools with native async implementations should override.
        """
        return await asyncio.to_thread(self.run, arguments, context)


def _coerce_boolean(value: Any) -> bool:
    """Coerce a value to ``bool`` with sensible string handling."""
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
    return bool(value)
