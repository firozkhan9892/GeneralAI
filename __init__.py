"""GeneralAI — Autonomous AI Platform.

This package contains the core application modules following clean architecture
and SOLID principles. Each submodule is independently extensible.

Key components:
- app.agents — Agent management, session orchestration, persistence
- app.core — Framework infrastructure (DI, events, lifecycle, plugins)
- app.kernel — Cognitive pipeline (perception, intent, goals, planning, etc.)
- app.server — REST API (FastAPI) with WebSocket and SSE support
- app.tools — Tool registry, execution, and categories
- app.llm — Language model integrations (OpenAI, Ollama, etc.)
"""

__version__ = "0.1.0"
__app_name__ = "GeneralAI"

__all__ = ["__version__", "__app_name__"]
