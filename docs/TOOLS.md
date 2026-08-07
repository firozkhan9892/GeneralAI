# Tool System

## Overview

The tool system provides a permission-based execution environment for extensible tool operations. Tools can be built-in, plugin-provided, or custom-registered.

## Architecture

```mermaid
graph TB
    subgraph "Registry"
        ToolRegistry[ToolRegistry]
        Categories[Tool Categories]
        Discovery[Auto Discovery]
    end

    subgraph "Execution"
        Executor[ToolExecutor]
        Context[ToolContext]
        Permissions[PermissionSystem]
        Cancellation[CancellationToken]
    end

    subgraph "Built-in Tools"
        Calculator[Calculator]
        File[File Tools]
        Http[HTTP Tools]
        Shell[Shell Tools]
        Python[Python Eval]
        Web[Web Tools]
    end

    ToolRegistry --> Discovery
    Discovery --> Categories
    ToolRegistry --> Executor
    Executor --> Context
    Executor --> Permissions
    Executor --> Cancellation
    Built-in Tools --> Categories
```

## Tool Categories

| Category | Tools | Description |
|---|---|---|
| `builtin` | echo, uuid, clock, calculator | Core utilities |
| `file` | read, write, list, delete | File operations |
| `http` | get, post, put, delete | HTTP requests |
| `shell` | execute | Shell command execution |
| `python` | evaluate | Python expression evaluation |
| `web` | search, scrape | Web operations |
| `planning` | plan, step, reflect | Agent planning |

## Using Tools

```python
from app.tools.registry import ToolRegistry
from app.tools.executor import ToolExecutor

registry = ToolRegistry()
registry.discover()

executor = ToolExecutor(registry=registry)

result = await executor.run(
    tool_name="calculator",
    parameters={"expression": "2 + 2"},
    context=ToolContext(...),
)

if result.success:
    print(result.data)  # 4
```

## Creating a Tool

```python
from app.tools.base import Tool
from app.tools.models import ToolMetadata, ToolResult

class MyTool(Tool):
    metadata = ToolMetadata(
        name="my-tool",
        description="Does something useful",
        category="custom",
        parameters=[
            ToolParameter(
                name="input",
                param_type="string",
                description="Input value",
                required=True,
            ),
        ],
    )

    async def execute(self, **kwargs) -> ToolResult:
        input_value = kwargs.get("input")
        result = process(input_value)
        return ToolResult(success=True, data=result)
```

## Permission System

Tools require explicit permission grants:

```python
from app.tools.permissions import PermissionSystem

permissions = PermissionSystem()
permissions.allow("user-123", "calculator")
permissions.deny("user-123", "shell")
```

## Execution Features

| Feature | Description |
|---|---|
| Timeout | Configurable per-tool timeout |
| Retry | Automatic retry on transient failure |
| Cancellation | Mid-execution cancellation |
| Streaming | Progressive result delivery |

## Built-in Tools

### Calculator

```python
result = await executor.run("calculator", {"expression": "sqrt(144) * 2"})
# result.data = 24.0
```

### File Reader

```python
result = await executor.run("file_read", {"path": "/data/input.txt"})
# result.data = "file contents..."
```

### HTTP Request

```python
result = await executor.run("http_get", {"url": "https://api.example.com/data"})
# result.data = response body
```
