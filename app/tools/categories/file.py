"""File-system tools: read, write, and list."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.exceptions import ToolExecutionError, ToolValidationError
from app.tools.models import ToolCategory, ToolParameter


class FileReadTool(Tool):
    """Read the contents of a text file."""

    name = "file_read"
    description = "Read a text file from the filesystem"
    category = ToolCategory.FILE
    sandboxable = True
    parameters = (
        ToolParameter(
            name="path",
            description="Absolute path of the file to read",
            param_type="string",
            required=True,
        ),
        ToolParameter(
            name="encoding",
            description="Text encoding",
            param_type="string",
            default="utf-8",
        ),
    )

    def run(
        self, arguments: Mapping[str, Any], context: ToolContext | None = None
    ) -> Any:
        path = Path(arguments["path"])
        encoding = arguments.get("encoding", "utf-8")
        try:
            return path.read_text(encoding=encoding)
        except OSError as exc:
            raise ToolExecutionError(
                f"Failed to read '{path}': {exc}",
                module="tools.categories.file",
                cause=exc,
            ) from exc


class FileWriteTool(Tool):
    """Write content to a text file, creating parent directories."""

    name = "file_write"
    description = "Write text content to a file"
    category = ToolCategory.FILE
    sandboxable = True
    parameters = (
        ToolParameter(
            name="path",
            description="Absolute path of the file to write",
            param_type="string",
            required=True,
        ),
        ToolParameter(
            name="content",
            description="Text content to write",
            param_type="string",
            required=True,
        ),
        ToolParameter(
            name="encoding",
            description="Text encoding",
            param_type="string",
            default="utf-8",
        ),
    )

    def run(
        self, arguments: Mapping[str, Any], context: ToolContext | None = None
    ) -> Any:
        path = Path(arguments["path"])
        encoding = arguments.get("encoding", "utf-8")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(arguments["content"], encoding=encoding)
        except OSError as exc:
            raise ToolExecutionError(
                f"Failed to write '{path}': {exc}",
                module="tools.categories.file",
                cause=exc,
            ) from exc
        return {"path": str(path), "bytes": len(arguments["content"])}


class FileListTool(Tool):
    """List the entries in a directory."""

    name = "file_list"
    description = "List entries inside a directory"
    category = ToolCategory.FILE
    sandboxable = True
    parameters = (
        ToolParameter(
            name="path",
            description="Directory to list",
            param_type="string",
            required=True,
        ),
    )

    def run(
        self, arguments: Mapping[str, Any], context: ToolContext | None = None
    ) -> Any:
        path = Path(arguments["path"])
        if not path.exists():
            raise ToolValidationError(
                f"Path does not exist: {path}",
                module="tools.categories.file",
            )
        if not path.is_dir():
            raise ToolValidationError(
                f"Path is not a directory: {path}",
                module="tools.categories.file",
            )
        try:
            return sorted(entry.name for entry in path.iterdir())
        except OSError as exc:
            raise ToolExecutionError(
                f"Failed to list '{path}': {exc}",
                module="tools.categories.file",
                cause=exc,
            ) from exc
