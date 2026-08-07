# Quick Start Guide

## CLI Mode

Run GeneralAI directly from the command line:

```bash
# Basic usage
python main.py --prompt "What is the meaning of life?"

# With debug logging
python main.py --debug --prompt "Explain quantum computing"

# Specify environment
python main.py --env production --log-level WARNING --prompt "Hello"
```

### CLI Flags

| Flag | Description | Default |
|---|---|---|
| `--prompt <text>` | Run a single prompt and exit | — |
| `--debug` | Enable debug logging | `false` |
| `--env <name>` | Environment (development/staging/production) | `development` |
| `--log-level <level>` | Logging level | `INFO` |
| `--session-id <id>` | Session identifier for persistence | auto-generated |
| `--version` | Print version and exit | — |

## HTTP Server Mode

Start the REST API server:

```bash
# Development (with auto-reload)
uvicorn app.server.app:create_app --factory --reload

# Production
uvicorn app.server.app:create_app --factory --host 0.0.0.0 --port 8000 --workers 4
```

### Verify Server

```bash
# Health check
curl http://localhost:8000/health

# Metrics
curl http://localhost:8000/metrics
```

### API Key Authentication

Set the API key via environment variable:

```bash
export GENERAL_AI_API_KEY="your-secret-key"
```

Then include it in requests:

```bash
curl -H "X-API-Key: your-secret-key" http://localhost:8000/agent/run \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, AI"}'
```

## First Workflow

Create and run a simple workflow:

```python
from app.automation.workflow import WorkflowService
from app.automation.models import WorkflowDefinition, WorkflowStep

# Create a workflow definition
definition = WorkflowDefinition(
    id="hello-workflow",
    name="Hello Workflow",
    steps=[
        WorkflowStep(
            id="step1",
            type="tool",
            name="Say Hello",
            config={"tool": "echo", "input": "Hello, World!"},
        ),
    ],
)

# Execute
service = WorkflowService()
service.register(definition)
result = await service.run("hello-workflow", inputs={})
```

## Knowledge Ingestion

Ingest documents into the knowledge base:

```python
from app.knowledge import KnowledgeSettings
from app.knowledge.bootstrap import register_knowledge_components

# Initialize
container = DependencyContainer()
register_knowledge_components(container)

# Ingest a document
pipeline = container.resolve(IndexingPipeline)
pipeline.ingest(
    b"Document content here...",
    source_uri="document.txt",
    collection_id="my-docs",
    namespace="production",
)

# Search
results = pipeline.search(
    query="search query",
    collection_id="my-docs",
    namespace="production",
    top_k=5,
)
```

## Next Topics

- [REST API Reference](ENDPOINTS.md) — Full API documentation
- [Workflow Engine](WORKFLOW.md) — Workflow creation and execution
- [Knowledge/RAG System](KNOWLEDGE_RAG.md) — Document ingestion and retrieval
- [Plugin System](PLUGINS.md) — Extending GeneralAI
- [Configuration](CONFIGURATION.md) — All settings and environment variables
