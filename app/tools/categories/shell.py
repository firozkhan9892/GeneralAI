"""Shell tools: run commands in a subprocess."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from typing import Any

from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.exceptions import ToolExecutionError, ToolValidationError
from app.tools.models import ToolCategory, ToolParameter


class ShellRunTool(Tool):
    """Execute a command in a subprocess and capture its output."""

    name = "shell_run"
    description = "Run a shell command and return stdout, stderr, and exit code"
    category = ToolCategory.SHELL
    requires_confirmation = True
    sandboxable = True
    parameters = (
        ToolParameter(
            name="command",
            description="Command line to execute",
            param_type="string",
            required=True,
        ),
        ToolParameter(
            name="timeout_s",
            description="Execution timeout in seconds",
            param_type="number",
            default=30.0,
        ),
    )

    def run(
        self, arguments: Mapping[str, Any], context: ToolContext | None = None
    ) -> Any:
        command = arguments["command"].strip()
        if not command:
            raise ToolValidationError(
                "command parameter is required",
                module="tools.categories.shell",
            )
        timeout_s = float(arguments.get("timeout_s", 30.0))
        try:
            completed = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolExecutionError(
                f"Command timed out after {timeout_s}s",
                module="tools.categories.shell",
                cause=exc,
            ) from exc
        except OSError as exc:
            raise ToolExecutionError(
                f"Failed to run command: {exc}",
                module="tools.categories.shell",
                cause=exc,
            ) from exc
        return {
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
