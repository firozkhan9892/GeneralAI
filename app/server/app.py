"""FastAPI application factory for the GeneralAI server.

``create_app`` wires the DI container, lifespan, middleware,
exception handlers, and all routers into a ready-to-serve
:class:`fastapi.FastAPI` instance.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

from app.agents.bootstrap import register_agent_manager_components
from app.agents.manager import AgentManager
from app.automation.bootstrap import register_automation_components
from app.knowledge.bootstrap import register_knowledge_components
from app.knowledge.collection_registry import CollectionRegistry
from app.knowledge.analytics import KnowledgeAnalytics
from app.knowledge.indexing.pipeline import IndexingPipeline
from app.knowledge.namespace_registry import NamespaceRegistry
from app.knowledge.retrieval.pipeline import RetrievalPipeline
from app.automation.exceptions import (
    WorkflowApprovalError,
    WorkflowConcurrencyError,
    WorkflowNotFoundError,
    WorkflowSchedulerError,
    WorkflowValidationError,
    WorkflowVersionError,
)
from app.automation.workflow import WorkflowService
from app.core.container import DependencyContainer
from app.core.exceptions import GeneralAIError
from app.llm.bootstrap import (
    register_default_llm_providers,
    register_llm_components,
)
from app.llm.config import LLMSettings
from app.llm.factory import ProviderFactory
from app.llm.registry import ProviderRegistry
from app.server.config import ServerSettings
from app.server.metrics import MetricsCollector
from app.server.routers.chat import router as chat_router
from app.server.routers.health import router as health_router
from app.server.routers.knowledge import router as knowledge_router
from app.server.routers.memory import router as memory_router
from app.server.routers.tools import router as tools_router
from app.server.routers.workflows import (
    router as workflows_router,
    schedule_router as workflows_schedule_router,
)
from app.tools.categories.planning import plan_tools
from app.tools.registry import ToolRegistry

log = logging.getLogger(__name__)


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(404)
    async def _not_found(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={"detail": str(exc)},
        )

    @app.exception_handler(422)
    async def _unprocessable(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": str(exc)},
        )

    @app.exception_handler(GeneralAIError)
    async def _general_error(request: Request, exc: GeneralAIError) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"detail": exc.message},
        )

    # Workflow automation errors map to REST semantics.  Registered
    # after the base handler so Starlette resolves the most specific
    # handler for each exception type.
    @app.exception_handler(WorkflowNotFoundError)
    async def _workflow_not_found(
        request: Request, exc: WorkflowNotFoundError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={"detail": exc.message},
        )

    @app.exception_handler(WorkflowValidationError)
    async def _workflow_validation(
        request: Request, exc: WorkflowValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "detail": exc.message,
                "violations": exc.context.get("violations", []),
            },
        )

    @app.exception_handler(WorkflowVersionError)
    async def _workflow_version(
        request: Request, exc: WorkflowVersionError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"detail": exc.message},
        )

    @app.exception_handler(WorkflowApprovalError)
    async def _workflow_approval(
        request: Request, exc: WorkflowApprovalError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"detail": exc.message},
        )

    @app.exception_handler(WorkflowSchedulerError)
    async def _workflow_scheduler(
        request: Request, exc: WorkflowSchedulerError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"detail": exc.message},
        )

    @app.exception_handler(WorkflowConcurrencyError)
    async def _workflow_concurrency(
        request: Request, exc: WorkflowConcurrencyError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"detail": exc.message},
        )


def _build_lifespan(_app: FastAPI):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        container: DependencyContainer = app.state.container
        register_agent_manager_components(container)
        register_automation_components(container)

        manager: AgentManager = container.resolve(AgentManager)
        app.state.agent_manager = manager

        registry: ToolRegistry = container.resolve(ToolRegistry)
        if registry.count == 0:
            registry.discover()
            for tool in plan_tools():
                registry.register(tool)

        workflow_service: WorkflowService = container.resolve(WorkflowService)
        app.state.workflow_service = workflow_service

        await manager.restore()
        await workflow_service.startup()
        log.info("Server lifespan started")
        try:
            yield
        finally:
            await workflow_service.shutdown()
            await manager.shutdown()
            log.info("Server lifespan ended")

    return lifespan


def create_app(
    *,
    container: DependencyContainer | None = None,
    settings: ServerSettings | None = None,
    llm_settings: LLMSettings | None = None,
    discover_tools: bool = True,
) -> FastAPI:
    """Build and return a configured :class:`FastAPI` application.

    Args:
        container: Optional pre-built ``DependencyContainer``.  When
            omitted a fresh one is created and wired with the
            default kernel + agent components.
        settings: Optional :class:`ServerSettings`.  Defaults to a
            permissive dev configuration (no API key, rate limiting
            enabled at 60 req/min).
        llm_settings: Optional :class:`LLMSettings`.  Defaults to the
            environment (``API_MODE``, provider credentials).  In mock
            mode no credentials are required.
        discover_tools: When ``True`` (default) the built-in tool
            catalogue plus planning tools are discovered on startup
            if the registry is empty.

    Returns:
        A ready-to-serve FastAPI app.
    """
    settings = settings or ServerSettings()
    container = container or DependencyContainer()

    register_agent_manager_components(container)
    register_llm_components(container)
    register_automation_components(container)
    register_knowledge_components(container)

    # Register default LLM providers from configuration (mock by default).
    register_default_llm_providers(
        container.resolve(ProviderRegistry),
        container.resolve(ProviderFactory),
        settings=llm_settings,
    )

    metrics = MetricsCollector()
    from app.server.security import RateLimiter

    rate_limiter = RateLimiter(settings.rate_limit_per_minute)

    # Create a placeholder FastAPI instance for lifespan builder
    placeholder = FastAPI(title=settings.title, version=settings.version)
    lifespan = _build_lifespan(placeholder)

    app = FastAPI(
        title=settings.title,
        version=settings.version,
        lifespan=lifespan,
    )

    # Middleware
    @app.middleware("http")
    async def _metrics_middleware(request: Request, call_next):
        response = await call_next(request)
        metrics.record(request.url.path, response.status_code)
        return response

    app.state.settings = settings
    app.state.container = container
    app.state.metrics = metrics
    app.state.rate_limiter = rate_limiter

    # Resolve shared singletons eagerly so dependencies can read them
    # from app.state without depending on the container at request time.
    app.state.agent_manager = container.resolve(AgentManager)
    app.state.llm_router = container.resolve(
        __import__("app.llm.llm_router", fromlist=["LLMRouter"]).LLMRouter
    )
    app.state.memory_engine = container.resolve(
        __import__("app.kernel.memory.engine", fromlist=["MemoryEngine"]).MemoryEngine
    )
    app.state.tool_registry = container.resolve(ToolRegistry)
    app.state.tool_executor = container.resolve(
        __import__("app.tools.executor", fromlist=["ToolExecutor"]).ToolExecutor
    )
    app.state.workflow_service = container.resolve(WorkflowService)

    # Knowledge / RAG components
    app.state.collection_registry = container.resolve(CollectionRegistry)
    app.state.namespace_registry = container.resolve(NamespaceRegistry)
    app.state.knowledge_analytics = container.resolve(KnowledgeAnalytics)

    # Build knowledge pipelines from registered components
    from app.knowledge.embeddings.cache import EmbeddingCache
    from app.knowledge.documents.chunkers.recursive import RecursiveChunker
    from app.knowledge.documents.loaders.text import TextLoader
    from app.knowledge.retrieval.bm25 import BM25Retriever, BM25Index
    from app.knowledge.registry import EmbeddingProviderRegistry, VectorStoreRegistry

    _cache = EmbeddingCache()
    _provider_registry = container.resolve(EmbeddingProviderRegistry)
    _store_registry = container.resolve(VectorStoreRegistry)
    _provider = _provider_registry.get_or_raise("mock")
    _store = _store_registry.get_or_raise("in_memory")

    app.state.indexing_pipeline = IndexingPipeline(
        loader=TextLoader(),
        chunker=RecursiveChunker(),
        embedding_provider=_provider,
        vector_store=_store,
        cache=_cache,
        analytics=app.state.knowledge_analytics,
    )

    bm25 = BM25Retriever(index=BM25Index())
    app.state.retrieval_pipeline = RetrievalPipeline(
        retriever=bm25,
        embedding_provider=_provider,
        vector_store=_store,
        analytics=app.state.knowledge_analytics,
    )

    # CORS (opt-in)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Exception handlers
    _register_exception_handlers(app)

    # Public router (health — no auth)
    app.include_router(health_router)

    # Protected routers (auth + rate limit)
    from app.server.security import require_api_key, rate_limit

    protected_deps = [Depends(require_api_key), Depends(rate_limit)]
    app.include_router(chat_router, dependencies=protected_deps)
    app.include_router(memory_router, dependencies=protected_deps)
    app.include_router(tools_router, dependencies=protected_deps)
    app.include_router(workflows_router, dependencies=protected_deps)
    app.include_router(workflows_schedule_router, dependencies=protected_deps)
    app.include_router(knowledge_router, dependencies=protected_deps)
    # Register agent routes individually — WS route must NOT use HTTP
    # dependencies, so we don't include the whole agent_router with deps.
    from app.server.routers.agent import (
        agent_cancel,
        agent_run,
        agent_status,
        list_sessions,
        agent_ws,
    )

    app.add_api_route(
        "/agent/run",
        agent_run,
        methods=["POST"],
        dependencies=protected_deps,
        tags=["agent"],
    )
    app.add_api_route(
        "/agent/cancel",
        agent_cancel,
        methods=["POST"],
        dependencies=protected_deps,
        tags=["agent"],
    )
    app.add_api_route(
        "/agent/status/{session_id}",
        agent_status,
        methods=["GET"],
        dependencies=protected_deps,
        tags=["agent"],
    )
    app.add_api_route(
        "/agents",
        list_sessions,
        methods=["GET"],
        dependencies=protected_deps,
        tags=["agent"],
    )
    app.add_api_websocket_route("/agent/ws", agent_ws)

    log.info("GeneralAI server created (version=%s)", settings.version)
    return app
