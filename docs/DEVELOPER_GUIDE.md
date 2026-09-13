# Developer Onboarding

## Project Setup

### Prerequisites

- Python 3.10+
- Git
- Virtual environment (venv recommended)

### Initial Setup

```bash
# Clone repository
git clone <repo-url>
cd generalai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import app; print('OK')"
python -m pytest tests/ -q --tb=no
```

## Code Conventions

### Architecture Principles

1. **Zero global state** — Everything goes through DI container
2. **Idempotent bootstrap** — Registration functions can be called multiple times
3. **Frozen Pydantic models** — All domain models are immutable
4. **Async-first APIs** — Core methods are async with sync offloads
5. **Thread-safe registries** — All registries use RLock protection

### Module Pattern

```python
# 1. Define ABC in base.py
class MyService(ABC):
    @abstractmethod
    async def process(self, data: str) -> str: ...

# 2. Implement in services/
class MyServiceImpl(MyService):
    async def process(self, data: str) -> str:
        return data.upper()

# 3. Register in bootstrap.py
def register_my_components(container: DependencyContainer) -> None:
    if not container.has(MyService):
        container.register_singleton(MyService, factory=MyServiceImpl)

# 4. Export in __init__.py
from app.mymodule.service import MyServiceImpl
```

### Naming Conventions

| Element | Convention | Example |
|---|---|---|
| ABCs | `I` prefix or `Base` suffix | `ITool`, `BaseProvider` |
| Implementations | Descriptive name | `InMemoryVectorStore` |
| Registries | `XxxRegistry` | `ToolRegistry` |
| Bootstraps | `register_xxx_components()` | `register_tool_components()` |
| Models | Frozen Pydantic | `KnowledgeDocument` |
| Exceptions | `XxxError` | `KnowledgeIndexError` |

### Testing Pattern

```python
import pytest
from app.mymodule import MyService

class TestMyService:
    def test_basic_operation(self):
        service = MyService()
        result = service.process("input")
        assert result == "expected"

    def test_edge_case(self):
        service = MyService()
        with pytest.raises(ValueError):
            service.process("")
```

### Quality Gates

The exact commands CI runs (see [Continuous Integration](#continuous-integration)):

```bash
# Run all tests
python -m pytest -q

# Check types
python -m mypy . --no-error-summary

# Lint
python -m ruff check .

# Format check
python -m ruff format --check .
```

## Continuous Integration

GeneralAI uses GitHub Actions (`.github/workflows/ci.yml`) to run all quality
gates automatically on every push to `main` and every pull request targeting
`main`.

### What CI Runs

The pipeline has three isolated jobs:

| Job | Purpose |
|---|---|
| `test` | Installs the package in editable mode with dev extras, then runs `pytest`, `mypy`, `ruff check`, and `ruff format --check` |
| `package` | Builds the sdist and wheel with `python -m build`, verifies artifact names and metadata (name `generalai`, version `1.0.0`, `numpy` runtime dep), installs the wheel into a clean environment, and verifies `import app` and `create_app()` work |
| `optional-imports` | Installs only the base package (no optional extras), asserts `faiss`, `chromadb`, `pypdf`, `bs4`, and `sentence_transformers` are absent, and verifies the app still imports and `create_app()` starts |

All jobs run on a clean `ubuntu-latest` runner with Python 3.10 (the declared
minimum), use pip dependency caching, and run from a directory outside the
repository checkout so the local `app/` package cannot shadow the installed one.

### Reproducing CI Locally

```bash
# 1. Install the package (editable) with dev extras
pip install -e '.[dev]'

# 2. Quality gates
python -m pytest -q
python -m mypy . --no-error-summary
python -m ruff check .
python -m ruff format --check .

# 3. Packaging verification
python -m build

# 4. Optional-dependency isolation (base install only)
python -m venv /tmp/generalai-base
/tmp/generalai-base/bin/pip install .
cd /tmp
/tmp/generalai-base/bin/python -c "import app; print(app.__version__)"
/tmp/generalai-base/bin/python -c "from app.server.app import create_app; create_app()"
```

If any step fails locally, it will fail in CI too, so fix it before pushing.

## Key Entry Points

| Entry Point | File | Purpose |
|---|---|---|
| CLI | `main.py` | Command-line interface |
| HTTP Server | `app/server/app.py` | FastAPI factory |
| DI Container | `app/core/container/` | Service registration |
| Event Bus | `app/core/events/` | Inter-module communication |
| Cognitive Kernel | `app/kernel/` | AI processing pipeline |
| LLM Router | `app/llm/` | Multi-provider routing |
| Workflow Engine | `app/automation/` | DAG automation |
| Knowledge/RAG | `app/knowledge/` | Document retrieval |

## Common Patterns

### Resolving Dependencies

```python
from app.core.container import DependencyContainer

container = DependencyContainer()
service = container.resolve(MyService)
```

### Publishing Events

```python
from app.core.events import EventBus, Event

event_bus = container.resolve(EventBus)
await event_bus.publish(Event(
    event_type="knowledge.document.ingested",
    source="knowledge",
    context={"doc_id": "abc123"},
))
```

### Registering Tools

```python
from app.tools.registry import ToolRegistry
from app.tools.models import Tool, ToolMetadata, ToolResult

registry = ToolRegistry()
registry.register(MyTool())
```
