# GeneralAI

Autonomous AI Platform — production-grade foundation with multi-LLM routing, workflow automation, plugin system, and enterprise RAG.

## Features

- **Multi-LLM Intelligence Layer** — Route requests across OpenAI, Gemini, Ollama, and OpenRouter with health monitoring, circuit breakers, and automatic fallback
- **Cognitive Kernel** — 18-stage cognitive pipeline (Perception → Intent → Goals → Planning → Reasoning → Decision → Response)
- **Workflow Engine** — DAG-based workflow automation with scheduling, persistence, and version control
- **Plugin System** — Extensible plugin architecture with sandboxed execution
- **Tool Framework** — Permission-based tool execution with timeout, retry, and cancellation
- **Enterprise RAG** — Knowledge ingestion, hybrid retrieval (BM25 + vector), and citation generation
- **Agent System** — Multi-agent orchestration with session persistence and recovery
- **REST API** — FastAPI-based server with authentication, rate limiting, and streaming

## Architecture

```
GeneralAI/
├── app/
│   ├── core/           # DI container, event bus, lifecycle, registry
│   ├── server/         # FastAPI application, routers, security
│   ├── kernel/         # Cognitive pipeline (18 stages)
│   ├── llm/            # Multi-LLM routing and providers
│   ├── agents/         # Agent management and orchestration
│   ├── automation/     # Workflow engine and scheduler
│   ├── plugins/        # Plugin system and sandbox
│   ├── tools/          # Tool framework and execution
│   ├── knowledge/      # RAG pipeline and retrieval
│   ├── config/         # Settings and defaults
│   └── utils/          # Logging and helpers
├── tests/              # 2,497 tests across all modules
├── docs/               # Architecture and API documentation
├── main.py             # CLI entry point
└── requirements.txt    # Python dependencies
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run CLI mode
python main.py --prompt "Hello, AI"

# 3. Run HTTP server
uvicorn app.server.app:create_app --factory --reload

# 4. Test the API
curl http://localhost:8000/health
```

## Documentation

| Document | Description |
|---|---|
| [INSTALL.md](docs/INSTALL.md) | Installation and setup guide |
| [QUICKSTART.md](docs/QUICKSTART.md) | Getting started tutorial |
| [ENDPOINTS.md](docs/ENDPOINTS.md) | REST API reference |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture with diagrams |
| [WORKFLOW.md](docs/WORKFLOW.md) | Workflow engine guide |
| [KNOWLEDGE_RAG.md](docs/KNOWLEDGE_RAG.md) | Knowledge/RAG system guide |
| [PLUGINS.md](docs/PLUGINS.md) | Plugin system guide |
| [TOOLS.md](docs/TOOLS.md) | Tool system guide |
| [CONFIGURATION.md](docs/CONFIGURATION.md) | Configuration reference |
| [DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) | Developer onboarding |
| [CI.md](docs/CI.md) | Continuous integration & quality gates |
| [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Troubleshooting guide |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Deployment guide |

## Configuration

Settings are loaded from (in order of precedence):

1. CLI arguments
2. Environment variables prefixed with `GENERAL_AI_`
3. `.env` file in the project root
4. Defaults defined in `app/config/defaults.py`

Key environment variables:

| Variable | Default | Description |
|---|---|---|
| `GENERAL_AI_ENVIRONMENT` | `development` | Runtime environment |
| `GENERAL_AI_LOG_LEVEL` | `INFO` | Logging level |
| `GENERAL_AI_DEBUG` | `false` | Enable debug mode |

See [CONFIGURATION.md](docs/CONFIGURATION.md) for the full reference.

## Project Status

| Phase | Status | Description |
|---|---|---|
| Phase 8 | ✅ Complete | Foundation, DI, events, lifecycle |
| Phase 9 | ✅ Complete | FastAPI server, security |
| Phase 10 | ✅ Complete | Plugin system |
| Phase 11 | ✅ Complete | Multi-LLM intelligence layer |
| Phase 12 | ✅ Complete | Workflow engine |
| Phase 13a | ✅ Complete | Knowledge foundation |
| Phase 13b | ✅ Complete | Knowledge pipeline core |
| Phase 13c | ✅ Complete | Embeddings & vector storage |
| Phase 13d | ✅ Complete | Retrieval engine & RAG core |
| Phase 14 | ✅ Complete | Production readiness audit |
| Phase 14.1 | ✅ Complete | Production polish & documentation |
| Phase 14.4 | ✅ Complete | GitHub Actions CI/CD & automated quality gates |

## License

MIT
