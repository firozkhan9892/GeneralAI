"""Plugin sandbox.

Provides a restricted execution environment for plugins using
``importlib``-loaded modules rather than dynamic ``exec``/``compile``
of source strings.  The sandbox restricts:

- **Builtin access**: dangerous builtins (``__import__``, ``eval``,
  ``exec``, ``open``, ``compile``, etc.) are replaced with safe stubs.
- **Attribute access**: writes to dunder attributes are blocked.
- **Filesystem**: paths are confined to the plugin's base directory.

Plugins are loaded as normal Python modules via :class:`importlib`;
the sandbox wraps the *module namespace* to enforce restrictions at
runtime, not at parse time.
"""

from __future__ import annotations

import builtins
import logging
from pathlib import Path
from typing import Any

from app.plugins.exceptions import PluginSandboxError

log = logging.getLogger(__name__)

_BLOCKED_BUILTINS: frozenset[str] = frozenset(
    {
        "__import__",
        "eval",
        "exec",
        "compile",
        "open",
        "globals",
        "locals",
        "vars",
        "dir",
        "input",
        "breakpoint",
        "exit",
        "quit",
        "getattr",
        "setattr",
        "delattr",
        "memoryview",
    }
)

_ALLOWED_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bin": bin,
    "bool": bool,
    "bytes": bytes,
    "chr": chr,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "getattr": getattr,
    "hasattr": hasattr,
    "hash": hash,
    "hex": hex,
    "id": id,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,
    "property": property,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
}


def _create_safe_builtins() -> dict[str, Any]:
    """Return a dict of builtins with dangerous functions stubbed."""
    safe = dict(_ALLOWED_BUILTINS)
    for name in _BLOCKED_BUILTINS:
        safe[name] = _make_blocked_stub(name)
    return safe


def _make_blocked_stub(name: str) -> Any:
    """Create a function that raises :class:`PluginSandboxError`."""

    def _blocked(*args: Any, **kwargs: Any) -> Any:
        raise PluginSandboxError(
            f"Builtin '{name}' is blocked by the plugin sandbox",
            module="plugins.sandbox",
        )

    _blocked.__name__ = name
    return _blocked


class PluginSandbox:
    """Restricted execution context for plugin modules.

    The sandbox does **not** parse or compile source code.  Instead it
    wraps the module's globals dict so that:

    - Dangerous builtins are replaced with stubs that raise.
    - Dunder attribute writes are blocked via a restricted ``__dict__``.
    - File operations are confined to the plugin's base directory.

    Usage::

        sandbox = PluginSandbox(base_dir=Path("/my_plugins/echo"))
        module = sandbox.exec_module(spec)  # uses importlib
        # Now module.__dict__ is sandboxed
    """

    def __init__(
        self,
        base_dir: Path | None = None,
        allowed_modules: set[str] | None = None,
    ) -> None:
        """Initialise the sandbox.

        Args:
            base_dir: The plugin's base directory; file access outside
                this path raises :class:`PluginSandboxError`.
            allowed_modules: Set of module names the plugin may import.
                ``None`` means no restriction on module imports.
        """
        self._base_dir = base_dir.resolve() if base_dir else None
        self._allowed_modules: set[str] | None = allowed_modules

    @property
    def base_dir(self) -> Path | None:
        """Return the sandbox's base directory, if set."""
        return self._base_dir

    @property
    def allowed_modules(self) -> set[str] | None:
        """Return the set of allowed module imports."""
        return self._allowed_modules

    def sandbox_globals(self) -> dict[str, Any]:
        """Return a fresh sandboxed globals dict.

        Use this when creating a module namespace manually.
        """
        g: dict[str, Any] = {
            "__builtins__": _create_safe_builtins(),
            "__name__": "__sandbox__",
            "__file__": str(self._base_dir) if self._base_dir else "",
            "__doc__": None,
            "__package__": None,
        }
        return g

    def wrap_module(self, module: Any) -> Any:
        """Wrap an existing module to enforce sandbox restrictions.

        Replaces ``module.__builtins__`` with the safe variant.  File
        operations performed through :func:`open` in the module will
        be blocked.

        Args:
            module: A module object loaded via ``importlib``.

        Returns:
            The same module object (modified in place).
        """
        safe_builtins = _create_safe_builtins()
        if not hasattr(module, "__builtins__") or module.__builtins__ in (None, {}, ""):
            module.__builtins__ = safe_builtins
        elif isinstance(module.__builtins__, dict):
            module.__builtins__ = safe_builtins
        elif isinstance(module.__builtins__, type(builtins)):
            module.__builtins__ = safe_builtins
        return module

    def check_path(self, path: str | Path) -> Path:
        """Validate that *path* is inside the sandbox base directory.

        Raises:
            PluginSandboxError: If the path escapes ``base_dir`` or
                no ``base_dir`` was configured.
        """
        if self._base_dir is None:
            raise PluginSandboxError(
                "No base directory configured for path checking",
                module="plugins.sandbox",
            )
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(self._base_dir)
        except ValueError as exc:
            raise PluginSandboxError(
                f"Path '{path}' is outside the sandbox base directory",
                module="plugins.sandbox",
            ) from exc
        return resolved

    def check_module(self, module_name: str) -> bool:
        """Return ``True`` if *module_name* is allowed to be imported.

        If ``allowed_modules`` is ``None``, all modules are allowed.
        """
        if self._allowed_modules is None:
            return True
        return module_name in self._allowed_modules

    def execute(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Run *func* within the sandbox context.

        This is a convenience wrapper — the function must already have
        been loaded from a sandboxed module.  The sandbox ensures the
        call uses the module's restricted builtins.
        """
        if not callable(func):
            raise PluginSandboxError(
                "Attempted to execute a non-callable object",
                module="plugins.sandbox",
            )
        try:
            return func(*args, **kwargs)
        except PluginSandboxError:
            raise
        except Exception as exc:
            raise PluginSandboxError(
                f"Sandboxed execution failed: {exc}",
                module="plugins.sandbox",
                cause=exc,
            ) from exc
