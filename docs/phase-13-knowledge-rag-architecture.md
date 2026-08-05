# Phase 13 — Enterprise Knowledge & RAG System — Architecture Design

Status: **Proposed** (no implementation yet)
Author: Architecture audit
Date: 2026-08-02
Scope: Design document only. No code is implemented by this document.

---

## 1. Objectives & Scope

Phase 13 introduces an **Enterprise Knowledge & RAG (Retrieval-Augmented Generation)** subsystem to
GeneralAI, delivered as a new top-level module `app/knowledge/`. It provides:

1. **Document ingestion** — parse, chunk, embed, and index documents from 7 formats (PDF, DOCX, TXT,
   Markdown, HTML, CSV, JSON).
2. **Retrieval** — hybrid vector + lexical (BM25) search with ranking, metadata filtering, query
   rewriting, multi-query expansion, context compression, and reranking.
3. **Enterprise concerns** — knowledge collections, namespaces, versioned documents, incremental
   indexing, background indexing workers, embedding caching, retrieval analytics, and full citation /
   source attribution.
4. **Extensibility** — plugin support for custom loaders, embedding providers, vector stores, and
   retrievers; DI integration; REST APIs; streaming retrieval.

Constraints (from the brief):
- **No breaking changes.** Everything additive; existing REST API, workflows, agents, tools, memory,
  LLM layer, plugins, DI, and tests must keep working unchanged.
- **Enterprise architecture matching previous phases** — same patterns: idempotent module
  `bootstrap.py`, DI container singletons, `BaseRegistry`-backed thread-safe registries, module-local
  exception hierarchy rooted at `GeneralAIError`, event constants in a module-local `events.py`,
  FastAPI router under `protected_deps`, `app.state` singletons, `sse_format` streaming, isolated
  test fixtures, full Pydantic v2 typing (frozen models).
- **Quality gate** (mandatory, must stay green): `pytest -q` (2186 currently passing), `mypy .`,
  `ruff check .`, `ruff format --check .`.
- **No feature creep** outside the Knowledge/RAG system.

Out of scope for Phase 13:
- MCP (Model Context Protocol) integration — previously designed in
  `docs/phase-13-mcp-architecture.md`; the Knowledge system is standalone and MCP can bridge it later.
- Durable persistent stores (Postgres, disk-backed vectors) — Phase 13 uses in-memory + FAISS/Chroma
  with the same store-abstraction pattern the automation module uses (swap via DI).

---

## 2. Existing Architecture Audit (as-built)

Verified against the current source tree.

### 2.1 What exists today (relevant surfaces)

- **Memory (`app/kernel/memory/engine.py`)**: deterministic keyword/tag-overlap retrieval
  (`_score_record` = 0.7 keyword + 0.2 tag + 0.1 importance). `MemoryStore` interface
  (save/get/delete/query/all/count/clear) + `InMemoryMemoryStore`. `MemoryEngine` public API:
  `remember`, `forget`, `clear`, `touch`, `get`, `retrieve`, `search`, `summarize`, `consolidate`,
  `prune`, `count`. **No embeddings, no vectors.**
- **`app/memory/`** is a placeholder (`__init__.py` only). The real memory engine lives in
  `app/kernel/memory/`.
- **LLM layer (`app/llm/`)**: provider-agnostic. `BaseLLMProvider` requires **sync** `generate` /
  `stream`; async variants (`generate_async`/`stream_async`) default to `asyncio.to_thread` offload
  (base.py:90-123). `BaseHttpProvider` template hooks. `ProviderRegistry` wraps
  `BaseRegistry[BaseLLMProvider]`. `ProviderFactory` builds providers from registered builders
  (`mock`, `openai`, `openrouter`, `gemini`, `ollama`). `CapabilityMatrix` tracks
  `ProviderCapabilities`; **`CapabilityFlag.EMBEDDINGS` and `TaskType.EMBEDDING` already exist but are
  dormant** (router_models.py:33,72) — no provider sets it, no embedding method exists anywhere.
  `ModelInfo` (models.py:92-116) has **no** `supports_embeddings` field.
- **Tools (`app/tools/`)**: `ToolRegistry` (wraps `BaseRegistry`), `ToolExecutor` (resolve →
  permission → validate → execute with retry/timeout; **never raises**, returns `ToolResult`),
  `PermissionSystem` (`check` via fnmatch rules, `allow`/`deny`/`confirm`), `ToolMetadata` /
  `ToolParameter` (JSON-schema-style `param_type`). `register_tool_components(container)` idempotent
  bootstrap.
- **Workflow automation (`app/automation/`)**: `WorkflowService` façade over versioned, immutable
  definitions (published versions never mutated); `WorkflowRegistry` (register/publish/deprecate/
  get/list_versions), `WorkflowRunRegistry`, `ScheduleStore`, `WorkflowRunStore`, `EventStore`;
  `StepTypeRegistry` (register/unregister/get by step type); `WorkflowValidator`;
  `WorkflowGraphExporter`; `WorkflowExecutor` (DAG engine); `WorkflowScheduler` (async, thread-safe,
  injectable `Clock` for deterministic tests — never calls `datetime.utcnow()` directly);
  `register_automation_components(container)` idempotent, with `_try_resolve_tool_executor` /
  `_try_resolve_agent_manager` / `_try_resolve_llm_router` lazy-collaborator pattern.
- **Events (`app/core/events/event_bus.py`)**: `EventBus.publish(event)` (async, fans out with
  `asyncio.gather(return_exceptions=True)`), `subscribe(event_type, handler)` (async coroutine
  required), `unsubscribe`, `clear`. `Event` frozen pydantic (`event_id`, `timestamp`, `source`,
  `event_type`). Constants use dotted lowercase domains (`workflow.*` in
  `app/automation/events.py`, `app.*`/`plugin.*` in `app/core/constants/events.py`).
- **Lifecycle (`app/core/lifecycle/lifecycle_manager.py`)**: `LifecycleManager.run()` =
  initialize → start → idle → shutdown; `register_hook(hook_point, hook)` with hook constants
  (`HOOK_AFTER_SERVICES`, `HOOK_AFTER_PLUGINS`, etc.). The **FastAPI server uses its own lifespan**
  (`_build_lifespan` in app.py:131-160): registers components, resolves singletons onto `app.state`,
  calls `workflow_service.startup()` / `workflow_service.shutdown()`.
- **DI (`app/core/container/dependency_container.py`)**: thread-safe `DependencyContainer`;
  `register_singleton(interface, instance=None, factory=None)`, `register_factory`, `resolve`
  (auto constructor injection via `typing.get_type_hints`, circular-detection via `_resolve_stack`),
  `unregister`, `has`, `clear`. Key = `module.qualname`.
- **FastAPI (`app/server/app.py`)**: `create_app(*, container=None, settings=None,
  discover_tools=True)`. Lifespan registers agent/llm/automation components. `protected_deps =
  [Depends(require_api_key), Depends(rate_limit)]` applied to routers. `app.state` holds
  `settings`, `container`, `metrics`, `rate_limiter`, `agent_manager`, `llm_router`, `memory_engine`,
  `tool_registry`, `tool_executor`, `workflow_service`. Exception handlers: generic `GeneralAIError →
  500` registered first, then specific domain handlers (404/422/409/400) registered after so the most
  specific wins.
- **Auth/streaming (`app/server/security.py`, `app/server/streaming.py`)**: `require_api_key`
  (X-API-Key header or `api_key` query), `rate_limit` (fixed-window `RateLimiter`);
  `sse_format(event, data)` → `"event: {event}\ndata: {json}\n\n"`; `StreamingResponse` framing
  pattern in `chat.py` and `workflows.py`.
- **Server wiring (`app/server/dependencies.py`)**: `get_*` providers read `request.app.state`.
  `app/server/schemas.py` holds request/response models; responses embed domain models; list
  responses use `{total, <items>}` pattern.
- **Plugins (`app/plugins/`)**: `PluginBase` lifecycle (initialize/install/load/enable/disable/
  unregister/unload/uninstall/cleanup). `PluginContext` is a **frozen dataclass** of optional service
  handles (`tool_registry`, `agent_manager`, `provider_registry`, `provider_factory`, `memory_engine`,
  `fastapi_app`, `container`, `workflow_service`, `workflow_registry`, `step_type_registry`, `logger`).
  `PluginType` enum: TOOL, AGENT, WORKFLOW, API_ROUTE, MEMORY_PROVIDER, LLM_PROVIDER, MIXED.
  `PluginManager` auto-wires **LLM_PROVIDER** plugins (`_register_llm_provider`, manager.py:777) but
  **no** MEMORY_PROVIDER auto-wiring exists. Context is populated ad hoc (tests construct filled
  contexts); production `register_plugin_components` registers with `context=None`.
- **Config (`app/config/settings.py`, `app/config/defaults.py`)**: pydantic-settings `AppSettings`
  (env prefix `GENERAL_AI_`); defaults include `DEFAULT_DATA_DIR = Path("data")`,
  `DEFAULT_MODELS_DIR = Path("models")`, `ENVIRONMENTS`. `ServerSettings` (app/server/config.py) is a
  frozen BaseModel: `api_key`, `rate_limit_enabled`, `rate_limit_per_minute`, `cors_origins`, `host`,
  `port`, `title`, `version`.
- **Exceptions (`app/core/exceptions/base.py`)**: `GeneralAIError(message, *, module, cause,
  context)`; `module` attribute; server maps to 500 unless a specific handler is registered.
- **Testing**: root `tests/conftest.py` has `FakeHttpTransport`/`fake_transport`; `tests/automation/
  conftest.py` has `fake_clock`/`system_clock`/`linear_definition`/`diamond_definition`/`utc_now`;
  server tests build isolated `create_app(settings=ServerSettings(rate_limit_enabled=False))` +
  `TestClient`, assert routes via `/openapi.json`. **Baseline: 2186 tests pass.**

### 2.2 Greenfield gaps (verified)

- **No embedding provider, no vector store, no retrieval, no RAG** anywhere. `EMBEDDINGS` capability
  flag is dormant. No `supports_embeddings` on `ModelInfo`.
- **No `app/knowledge/` package.**
- `requirements.txt` has no FAISS/Chroma/numpy/docx/markdown/HTML parsing deps.
  **Environment check (installed locally)**: `faiss-cpu 1.13.2`, `chromadb 1.1.1`, `numpy 1.26.4`,
  `pypdf 6.11`, `pdfplumber 0.11.9`, `pypdfium2`, `pdfminer.six`, `beautifulsoup4 4.12.3`,
  `markdown-it-py`, `scikit-learn 1.7.2`, `sentence-transformers 2.7.0`, `tiktoken 0.12`,
  `transformers 5.12`, `langchain*`. **`python-docx` is NOT installed** (required for DOCX).
- `mypy.ini` = `explicit_package_bases = True` only. No `pyproject.toml`.
- `docs/phase-13-mcp-architecture.md` exists (previous scope; superseded for implementation, but
  stored for reference).

---

## 3. Integration Points (mapped)

| Integration | Existing surface | Phase 13 role |
|---|---|---|
| DI | `DependencyContainer`, idempotent module `bootstrap.py` pattern | New `app/knowledge/bootstrap.py` registering all knowledge components |
| FastAPI | `create_app()`, routers, `protected_deps`, `app.state`, dependencies | New `app/knowledge/router.py` mounted under `protected_deps`; `app.state.knowledge_service`; `get_knowledge_service` dep |
| Events | `EventBus`, `app/automation/events.py` constant pattern | New `app/knowledge/events.py` (`knowledge.*` domain) + publish on ingest/index/retrieve |
| Lifecycle | `_build_lifespan` in `app/server/app.py` | `knowledge_service.startup()`/`shutdown()`, index worker lifecycle |
| Tools | `ToolRegistry`/`ToolExecutor`/`ToolMetadata` | Register `kb_search`/`kb_ingest` tools (via `knowledge_tools()`); workflow TASK steps can call them |
| LLM | `LLMRouter`, `BaseLLMProvider`, `CapabilityMatrix`, `Message`/`ToolCall` | Embedding provider abstraction reuses `BaseLLMProvider`-style sync+async pattern; optional LLM-based query rewrite / rerank / compression via `_try_resolve_llm_router` |
| Memory | `MemoryEngine` (keyword retrieval) | Keep separate; optionally knowledge retrieval may cite memory records later (out of scope) |
| Plugins | `PluginContext` frozen dataclass, `PluginType`, `PluginManager` | Add optional knowledge handles to `PluginContext`; plugins register custom loaders/embedding/vector/retriever via new registries |
| Exceptions | `GeneralAIError`, server handlers | New `app/knowledge/exceptions.py`; add specific handlers to `create_app` |
| Config | `AppSettings` / `ServerSettings` | Add optional knowledge settings (chunk defaults, store type, worker concurrency) |
| Testing | `tests/`, fixtures, isolated app | New `tests/knowledge/` with the same fixture conventions |

---

## 4. Proposed Module Layout

New top-level module `app/knowledge/` (sibling of `automation`, `tools`, `llm`). Each file follows
the repo's conventions (frozen pydantic models, module-local exceptions/events, structured logging).

```
app/knowledge/
  __init__.py                  # public exports
  bootstrap.py                 # register_knowledge_components(container) — idempotent
  constants.py                 # default chunk sizes, cache sizes, event-agnostic constants
  exceptions.py                # KnowledgeError hierarchy
  events.py                    # EVENT_KNOWLEDGE_* constants
  models.py                    # domain models (documents, chunks, collections, queries, results)
  config.py                    # KnowledgeSettings (pydantic) + defaults
  documents/
    __init__.py
    loader.py                  # DocumentLoader ABC + registry
    loaders/
      __init__.py
      text.py                  # TXT, Markdown, CSV, JSON (stdlib + json)
      pdf.py                   # pypdf (lazy import)
      docx.py                  # python-docx (lazy import)
      html.py                  # beautifulsoup4 (lazy import)
    chunker.py                 # Chunker ABC + registry
    chunkers/
      __init__.py
      fixed.py                 # Fixed-size (by char/token, with overlap)
      recursive.py             # Recursive structural splitting
      semantic.py              # Semantic (embedding-similarity boundaries)
      sliding.py               # Sliding window
    parser.py                  # format dispatch (extension -> loader)
  embeddings/
    __init__.py
    provider.py                # EmbeddingProvider ABC
    registry.py                # EmbeddingProviderRegistry (thread-safe)
    factory.py                 # EmbeddingProviderFactory (builders)
    providers/
      __init__.py
      mock.py                  # deterministic hash-based embeddings (tests/default)
      deterministic.py         # lightweight local provider (no deps)
      openai.py                # OpenAI-compatible embeddings (lazy; reuse HttpTransport)
    cache.py                   # EmbeddingCache (thread-safe LRU)
  vectorstores/
    __init__.py
    base.py                    # VectorStore ABC
    registry.py                # VectorStoreRegistry (thread-safe)
    factories.py               # build in-memory/faiss/chroma by name
    in_memory.py               # numpy-based brute-force cosine
    faiss.py                   # faiss-cpu IndexFlatIP (lazy import)
    chroma.py                  # chromadb persistent/in-memory (lazy import)
  retrieval/
    __init__.py
    query.py                   # RetrievalQuery model + normalization
    vector.py                  # VectorRetriever
    bm25.py                    # BM25Retriever (pure-python BM25, tokenize + idf)
    hybrid.py                  # HybridRetriever (RRF + score normalization)
    filter.py                  # MetadataFilter model + evaluation
    rewrite.py                 # QueryRewriter ABC + identity/LLM impls
    multiquery.py              # MultiQueryRetriever (expands into N sub-queries)
    compress.py                # ContextCompressor ABC + identity/LLM impls
    rerank.py                  # Reranker ABC + identity/LLM impls
    citations.py               # CitationBuilder + SourceReference
    registry.py                # RetrieverRegistry (thread-safe, pluggable retrievers)
  indexing/
    __init__.py
    pipeline.py                # IngestPipeline (parse -> chunk -> embed -> store)
    worker.py                  # IndexWorker (background async queue consumer)
    incremental.py             # content-hash change detection, chunk diffing
    versions.py                # DocumentVersion tracking (immutable versions)
  service.py                   # KnowledgeService façade (collections/namespaces/docs/retrieve)
  analytics.py                 # RetrievalAnalytics (thread-safe counters + summaries)
  router.py                    # FastAPI router (/knowledge/...)
  schemas.py                   # REST request/response models (server layer)
  deps.py                      # get_knowledge_service(request) dependency
```

---

## 5. Domain Models (`app/knowledge/models.py`)

All frozen Pydantic v2 models (`ConfigDict(frozen=True)`), matching `app/llm/models.py` /
`app/tools/models.py` conventions.

### 5.1 Documents & chunks

- `DocumentFormat(str, Enum)` — `TXT`, `MARKDOWN`, `HTML`, `PDF`, `DOCX`, `CSV`, `JSON`.
- `DocumentStatus(str, Enum)` — `DRAFT`, `INDEXED`, `FAILED`, `DELETED`.
- `KnowledgeDocument` — `doc_id: str`, `collection_id: str`, `namespace: str`, `title: str`,
  `source_uri: str` (origin; file/URL/id), `format: DocumentFormat`, `content: str` (raw text),
  `content_hash: str` (sha256 of normalized text), `version: int` (monotonic), `created_at`,
  `updated_at`, `status: DocumentStatus`, `metadata: dict[str, object]` (arbitrary, filterable),
  `chunk_ids: tuple[str, ...]`.
- `DocumentChunk` — `chunk_id: str`, `doc_id: str`, `collection_id: str`, `namespace: str`,
  `content: str`, `chunk_index: int`, `token_count: int`, `metadata: dict[str, object]` (inherited +
  page/offset where known), `hash: str`.
- `DocumentVersion` — `doc_id: str`, `version: int`, `content_hash: str`, `created_at`,
  `chunk_ids: tuple[str, ...]`. **Versions are immutable** (workflow-definition pattern); editing a
  document creates a new version and re-indexes only changed chunks.
- `ChunkingStrategy(str, Enum)` — `FIXED`, `SEMANTIC`, `RECURSIVE`, `SLIDING`.
- `ChunkingConfig` — `strategy: ChunkingStrategy`, `chunk_size: int`, `overlap: int`,
  `separators: tuple[str, ...]`, `max_tokens_per_chunk: int | None`.

### 5.2 Collections & namespaces

- `KnowledgeNamespace` — `name: str`, `description: str = ""`, `created_at`. Namespaces isolate
  collections (e.g. `prod`, `staging`, `team-a`).
- `KnowledgeCollection` — `collection_id: str`, `name: str`, `namespace: str`,
  `description: str = ""`, `created_at`, `updated_at`, `document_count: int = 0`,
  `chunk_count: int = 0`, `embedding_model: str = ""` (model used to index), `metadata:
  dict[str, object]`.
- `CollectionIdentity` helper — `(namespace, collection_name)` composite key used by the registry.

### 5.3 Queries & results

- `MetadataFilter` — operator model: `field: str`, `op: str` (`eq`, `neq`, `in`, `not_in`, `gt`,
  `gte`, `lt`, `lte`, `exists`, `contains`), `value: Any`. Support **AND of multiple filters**
  (a filter set evaluates to the conjunction); OR is out of scope (add later via `FilterGroup`).
- `RetrievalQuery` — `collection_id` or `namespace + collection_name`, `query: str` (raw),
  `rewritten_query: str = ""`, `namespace: str`, `filters: tuple[MetadataFilter, ...] = ()`,
  `strategy: str = "hybrid"` (vector|bm25|hybrid), `top_k: int`, `vector_weight: float = 0.5`,
  `bm25_weight: float = 0.5`, `rerank: bool = False`, `compression: bool = False`,
  `multi_query: bool = False`, `include_sources: bool = True`, `stream: bool = False`.
- `RetrievalHit` — `chunk_id: str`, `doc_id: str`, `collection_id: str`, `namespace: str`,
  `content: str`, `score: float`, `ranks: dict[str, float]` (per-strategy scores),
  `metadata: dict[str, object]`, `citations: tuple[Citation, ...]`.
- `RetrievalResult` — `query: str`, `rewritten_query: str`, `total: int`, `hits: tuple[RetrievalHit,
  ...]`, `sources: tuple[SourceReference, ...]`, `latency_ms: float`, `strategy: str`,
  `citations: tuple[Citation, ...]`, `analytics: dict[str, object] | None`.

### 5.4 Citations & sources

- `Citation` — `citation_id: str` (stable hash of doc_id+chunk_ids), `doc_id: str`,
  `doc_title: str`, `chunk_ids: tuple[str, ...]`, `page: int | None`, `source_uri: str`,
  `snippet: str`, `namespace: str`, `collection_id: str`.
- `SourceReference` — `doc_id`, `doc_title`, `source_uri`, `namespace`, `collection_id`,
  `version: int`, `confidence: float`.

### 5.5 Indexing / analytics

- `IngestionJob` — `job_id: str`, `collection_id: str`, `namespace: str`, `status: str`
  (`queued|running|succeeded|failed`), `doc_ids: tuple[str, ...]`, `total_chunks: int`,
  `processed_chunks: int`, `error: str | None`, `created_at`, `updated_at`.
- `RetrievalAnalyticsEntry` — `query: str`, `collection_id`, `namespace`, `latency_ms: float`,
  `hit_count: int`, `top_score: float`, `avg_score: float`, `strategy: str`, `reranked: bool`,
  `timestamp`.
- `RetrievalAnalyticsSummary` — `total_queries: int`, `avg_latency_ms: float`,
  `avg_hit_count: float`, `collections: dict[str, int]`, `top_queries: tuple[str, int]`,
  `recent: tuple[RetrievalAnalyticsEntry, ...]`.

---

## 6. Document Ingestion Pipeline

### 6.1 Architecture

Pipeline is a chain of **pluggable stages**, orchestrated by `IngestPipeline`:

```
raw bytes/file  →  DocumentLoader  →  KnowledgeDocument (text)  →  Chunker  →  chunks
      →  EmbeddingProvider.embed(chunks)  →  VectorStore.add(chunks, vectors)  →  index (BM25 + metadata)
```

### 6.2 Document loaders

- `DocumentLoader(ABC)`:
  ```python
  class DocumentLoader(ABC):
      format: DocumentFormat
      def load(self, content: bytes, *, source_uri: str, metadata: dict[str, object] | None = None) -> KnowledgeDocument
      async def load_async(self, ...) -> KnowledgeDocument   # default: asyncio.to_thread(load)
  ```
- `LoaderRegistry` — thread-safe map `format -> loader` (wraps `BaseRegistry`).
- Built-in loaders (each file lazy-imports its heavy dependency only when used):
  - `TextLoader` — TXT, Markdown, CSV, JSON (stdlib; CSV via `csv`, JSON via `json`).
  - `MarkdownLoader` — same as text but preserves headings for recursive chunking (reuses text
    parsing; headings retained in content).
  - `HtmlLoader` — `beautifulsoup4` → `get_text` with block separation (`\n\n` between block tags).
  - `PdfLoader` — `pypdf` → per-page extraction, page number recorded in chunk metadata.
  - `DocxLoader` — `python-docx` → paragraph/table extraction.
  - `JsonLoader` — parses JSON; either treats the whole payload as one document or, when
    `metadata["json_path"]` is given, extracts a text field.
- `FormatParser` — dispatches by extension/MIME to the loader registry; raises
  `KnowledgeUnsupportedFormatError` for unknown formats.

### 6.3 Chunking strategies (`Chunker(ABC)` + `ChunkerRegistry`)

- `Chunker.chunk(document) -> list[DocumentChunk]`, chunk index assigned, metadata copied.
- `FixedChunker` — fixed character size with configurable overlap; safe word boundaries.
- `SlidingWindowChunker` — overlapping windows (same family as fixed but configured via window/stride
  and used for streaming ingestion).
- `RecursiveChunker` — recursive structural split on separator hierarchy
  (`\n\n\n`, `\n\n`, `\n`, `. `) until chunks fit `chunk_size` (markdown headings preferred when
  present).
- `SemanticChunker` — splits on embedding-similarity drops between consecutive sentences using the
  configured `EmbeddingProvider` (sentences → embeddings → merge where adjacent similarity above
  threshold). Requires an embedding provider; falls back to recursive if unavailable.
- `ChunkerRegistry` — thread-safe `strategy -> chunker`.

### 6.4 Format → loader → chunker dispatch

`KnowledgeService.ingest(collection, bytes, *, format, chunking=ChunkingConfig())`:
1. `parser.load(...)` → `KnowledgeDocument` (content_hash computed on normalized text).
2. If content_hash unchanged and version exists → **incremental no-op** (skip re-embedding).
3. Else bump version, chunk, embed, store, update BM25 + vector indexes.

---

## 7. Embedding Provider Abstraction

### 7.1 Interface (mirrors `BaseLLMProvider` philosophy)

```python
class EmbeddingProvider(ABC):
    name: str
    dimensions: int
    def embed(self, texts: list[str]) -> list[list[float]]          # sync abstract
    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed, texts)            # base offload
    def model_info(self) -> EmbeddingModelInfo                      # name/dimensions/model
```

- Same sync-core + async-offload contract as `BaseLLMProvider` (base.py:90-123), so a single provider
  implementation serves both sync tool paths and async server paths.
- `EmbeddingModelInfo` — `name`, `provider`, `dimensions: int`, `model: str`, `max_input_tokens:
  int | None`.

### 7.2 Built-in providers

- `DeterministicEmbeddingProvider` — pure-python feature-hash embedding (no dependencies). Produces
  fixed-dimension normalized vectors; **deterministic for tests** (the MemoryEngine-style guarantee).
  Used as the default when no external provider is configured.
- `MockEmbeddingProvider` — alias of deterministic (kept for parity with `MockProvider`).
- `OpenAICompatibleEmbeddingProvider` — optional; reuses `app/llm/transport.py` `HttpTransport` and
  the `BaseHttpProvider` hook pattern (`_url`, `_headers`, `_build_payload`, `_parse_response`).
  Lazy import so it never breaks environments without the dep. Advertises via `CapabilityFlag.
  EMBEDDINGS` when registered with the LLM `CapabilityMatrix` (see §8.5).

### 7.3 Registry & factory

- `EmbeddingProviderRegistry` — thread-safe `name -> EmbeddingProvider` (wraps `BaseRegistry`),
  same shape as `ProviderRegistry`.
- `EmbeddingProviderFactory` — registered builder callables, same shape as `ProviderFactory`
  (`register`/`has`/`names`/`create`). Built-ins: `deterministic`, `mock`, `openai`.

---

## 8. Vector Stores

### 8.1 Interface

```python
class VectorStore(ABC):
    name: str
    dimensions: int
    def add(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None        # sync
    def delete(self, chunk_ids: Iterable[str]) -> None
    def delete_by_document(self, doc_id: str, namespace: str) -> None
    def search(self, vector: list[float], *, top_k: int,
               filters: tuple[MetadataFilter, ...] = ()) -> list[VectorSearchHit]
    async def add_async(...) / delete_async(...) / search_async(...)                     # thread-offload
    def count(self) -> int
    def clear(self) -> None
```

- `VectorSearchHit` — `chunk_id`, `doc_id`, `namespace`, `collection_id`, `score`, `metadata`.
- Metadata filtering is evaluated **post-search on the candidate set** (vector index returns
  top-N*oversample, filters applied, re-ranked) — keeps the store interface small and works for all
  three backends. Oversample factor (`filter_oversample=8`) is a `VectorStore` config.

### 8.2 Implementations

- `InMemoryVectorStore` — numpy brute-force cosine similarity over a dict of vectors; RLock-protected.
  Default, zero-dependency (numpy is a required dep of the knowledge module). Deterministic.
- `FaissVectorStore` — `faiss-cpu` `IndexFlatIP` (normalized vectors → cosine). Lazy import
  (`import faiss` inside `__init__`); if unavailable, the store refuses to construct with a
  `KnowledgeIndexError`. Thread-safe writes via a lock (faiss index add is not thread-safe).
- `ChromaVectorStore` — `chromadb`; in-memory or persistent client (`Settings(persist_directory=...)`
  from `KnowledgeSettings`). Lazy import; maps collection→chroma collection; stores chunk metadata in
  chroma metadata.
- `VectorStoreRegistry` — thread-safe `name -> factory` (not instances), so stores are built per
  namespace/collection on demand. Factories return instances; the registry is pluggable (plugins can
  register a factory).

### 8.3 Namespace/collection isolation

- Store keys include `namespace` + `collection_id` so multiple collections never collide. For
  Chroma, one chroma collection per knowledge collection; for FAISS/numpy, a separate index instance
  per `(namespace, collection)` kept in a registry map (built lazily, evicted on collection delete).

### 8.4 Default store selection

`KnowledgeSettings.default_vector_store` ∈ `{in_memory, faiss, chroma}`. `in_memory` is default;
`faiss`/`chroma` selected by config or per-collection. Falls back to `in_memory` with a warning if the
configured store's dependency is missing.

### 8.5 LLM capability integration

When an embedding provider is registered and the LLM `CapabilityMatrix` is available (resolved lazily
via `_try_resolve`), register the provider under `CapabilityFlag.EMBEDDINGS` so the router can see
embedding-capable providers. **No change to `LLMRouter` logic** — this is additive metadata only.
`ModelInfo` gains no field (avoid touching it); the capability flag suffices for Phase 13.

---

## 9. Hybrid Retrieval

### 9.1 Retrievers

- `Retriever(ABC)`:
  ```python
  class Retriever(ABC):
      name: str
      async def retrieve(self, query: RetrievalQuery, *, context: RetrievalContext) -> list[RetrievalHit]
  ```
  - `RetrievalContext` — bundle of collection, vector store, BM25 index, filters, clock.
- `VectorRetriever` — embeds the (rewritten) query via the collection's embedding provider, calls
  `VectorStore.search`, applies filters, returns hits with `ranks={"vector": score}`.
- `Bm25Retriever` — **pure-python BM25** (no external dep): tokenize (lowercase, strip punctuation,
  optional stopword list), build inverted index with `idf` (Robertson–Spark Jones), score with the
  BM25 formula (`k1=1.5, b=0.75`). Returns hits with `ranks={"bm25": score}`. Deterministic.
  Maintained per collection/namespace; updated incrementally on ingest/delete (§11).
- `HybridRetriever` — runs vector + BM25, fuses with **Reciprocal Rank Fusion (RRF)**:
  `score = w_v * rrf_v + w_b * rrf_b` where `rrf = Σ 1/(k + rank)` over both lists (k=60 default),
  weights from `RetrievalQuery.vector_weight`/`bm25_weight`. Removes duplicates by chunk_id; merges
  per-strategy ranks. Optionally min-max normalizes raw scores into `ranks` for transparency.

### 9.2 Metadata filtering

- `MetadataFilter` (§5.3) evaluated by `evaluate_filter(metadata, filter) -> bool` in
  `app/knowledge/retrieval/filter.py`. Filters apply post-vector-search on the oversampled candidate
  set and pre/post-BM25 (BM25 filters directly at index scan). All filters must pass (AND semantics).
- Filterable fields come from document/collection `metadata` plus intrinsic fields (`doc_id`,
  `namespace`, `format`, `version`, `page` when present).

### 9.3 RetrievalResult assembly

`RetrievalResult` includes normalized scores, per-strategy ranks, deduped hits, source references,
citations (§13), latency, and strategy. Streaming mode (§18) yields hits incrementally over SSE.

---

## 10. Query Rewriting & Multi-Query

### 10.1 Query rewriting

- `QueryRewriter(ABC)` — `async def rewrite(self, raw: str, *, context) -> str`.
- `IdentityQueryRewriter` — default, returns input unchanged (zero risk).
- `LlmQueryRewriter` — optional; uses `LLMRouter` (resolved lazily via `_try_resolve_llm_router`,
  same pattern as `app/automation/bootstrap.py`). Sends a system prompt asking for a normalized,
  self-contained, search-optimized query; never runs if no router is available (falls back to
  identity). Requires `RetrievalQuery.rewrite=True`.

### 10.2 Multi-query retrieval

- `MultiQueryRetriever` — wraps any base retriever. When `multi_query=True`, expands the query into
  N variants (`n_queries`, default 3) via the optional LLM rewriter (or by generating deterministic
  surface variants when no LLM is available), runs retrieval per variant, fuses via RRF, dedupes.
- Designed so that in the default configuration (no LLM) multi-query still works via deterministic
  variant generation (e.g., keyword-focused, phrase-focused, noun-phrase) — keeping the feature
  testable without network access.

---

## 11. Incremental Indexing & Versioned Documents

### 11.1 Content-hash change detection

- Each document stores `content_hash = sha256(normalize(content))`.
- On ingest:
  1. Compute hash.
  2. If `(doc_id, version)` exists and hash matches → **no-op** (skip parse/chunk/embed entirely).
  3. If hash differs → new version; diff old vs new chunk hashes:
     - unchanged chunks (same chunk hash + same index) → keep vectors (no re-embed).
     - changed/removed → delete old vectors; embed only new/changed chunks.
- This makes re-ingesting an unchanged document effectively free (the "cache embeddings" goal at the
  index level).

### 11.2 Versioned documents

- `DocumentVersion` snapshots are immutable; `KnowledgeDocument.version` is monotonic.
- Update = create new version + re-index delta; delete = remove latest version's chunks from index,
  keep earlier versions (or drop entirely per `keep_versions` config).
- `KnowledgeService.get_document(doc_id, version=None)` returns the requested (default latest)
  version; `list_versions(doc_id)` returns all.

### 11.3 Document update/delete

- `update_document(doc_id, *, content=None, metadata=None)` → new version, delta re-index.
- `delete_document(doc_id)` → remove chunks from vector store + BM25 + version store; document marked
  `DELETED` (soft delete) unless `hard=True`.
- BM25 index maintained incrementally (add/remove terms for changed chunks only).

---

## 12. Background Indexing Workers

- `IndexWorker` — async, thread-safe, model:
  - Owns a bounded `asyncio.Queue` of `IngestionJob`s (max size from `KnowledgeSettings`).
  - `enqueue(job)` returns job_id immediately; the worker drains the queue, processing each job
    through `IngestPipeline` (parse/chunk/embed/store), updating job status and emitting events.
  - Concurrent workers count from `KnowledgeSettings.index_workers` (default 2), gated by a semaphore.
  - `start()` / `stop()` called from the server lifespan (parity with `WorkflowService.startup()/
    shutdown()`). `stop()` drains gracefully with a timeout, then cancels.
  - Uses an injectable `Clock` (from `app/automation/time.py`) so timing is deterministic in tests —
    never calls `datetime.utcnow()` directly.
- Server ingest endpoints default to **async enqueue** (`wait=false` returns job_id; `wait=true`
  awaits completion) — mirroring the `AgentRunRequest.wait` pattern in schemas.py.

---

## 13. Context Compression & Reranking

### 13.1 Context compression

- `ContextCompressor(ABC)` — `async def compress(self, hits: list[RetrievalHit], *, query: str) ->
  list[RetrievalHit]`.
- `IdentityCompressor` — default, returns hits unchanged.
- `LlmCompressor` — optional; batches hits, asks the LLM (via `LLMRouter`) to drop irrelevant chunks
  and/or produce concise summaries, returns a reduced hit set with `metadata["compressed"]=True`.
  Falls back to identity without an LLM router. Enabled by `RetrievalQuery.compression=True`.

### 13.2 Reranking

- `Reranker(ABC)` — `async def rerank(self, query: str, hits: list[RetrievalHit]) ->
  list[RetrievalHit]`.
- `IdentityReranker` — default (no-op, preserves fusion order).
- `CrossEncoderReranker` — optional; `sentence-transformers` cross-encoder (lazy import). Reranks with
  a dedicated cross-encoder model, overriding the fused score. Falls back to identity when the dep or
  model is unavailable.
- Enabled by `RetrievalQuery.rerank=True`. Reranking runs after fusion and before citation assembly.

---

## 14. Citation Generation & Source Attribution

- `CitationBuilder` — after retrieval/rerank, builds:
  - One `Citation` per hit: `citation_id = sha256(doc_id + "|" + "|".join(chunk_ids))[:16]`, doc title,
    chunk ids, page (from chunk metadata when present), source_uri, snippet (first N chars), and
    collection/namespace.
  - `sources` — deduped `SourceReference` list across hits (unique by doc_id+version), each with
    confidence = max hit score.
  - Result citations are returned in the `RetrievalResult` and available for prompt construction by
    consumers (agents, workflows, REST clients).
- Every retrieved chunk carries `citations` back to its `KnowledgeDocument` so attribution is
  traceable end-to-end. This is a **pure function of the hits** — deterministic and unit-testable.

---

## 15. Knowledge Collections & Namespaces

- `CollectionRegistry` — thread-safe `(namespace, name) -> KnowledgeCollection` map + metadata;
  same shape as `WorkflowRegistry` but keyed by composite identity. Stores collection state (document
  count, chunk count, embedding model, vector store type).
- `NamespaceRegistry` — thread-safe `name -> KnowledgeNamespace`.
- `KnowledgeService` API (façade, mirroring `WorkflowService`):
  - Namespaces: `create_namespace(name)`, `list_namespaces()`, `delete_namespace(name)`.
  - Collections: `create_collection(name, namespace, *, chunking, embedding_provider, vector_store)`,
    `get_collection(name, namespace)`, `list_collections(namespace=None)`, `delete_collection(name,
    namespace)`.
  - Documents: `ingest(...)` (sync, returns doc), `ingest_async(job, wait)`, `update_document`,
    `delete_document`, `get_document`, `list_documents(collection)`, `list_versions(doc_id)`.
  - Retrieval: `retrieve(query) -> RetrievalResult`, `retrieve_stream(query) -> AsyncIterator[event]`.
  - Analytics: `get_analytics(collection=None, namespace=None) -> RetrievalAnalyticsSummary`.
- Deleting a namespace/collection clears its vector store, BM25 index, versions, and analytics.

---

## 16. Embedding Cache

- `EmbeddingCache` — thread-safe LRU keyed by `(provider_name, model, sha256(text)) -> vector`; max
  entries from `KnowledgeSettings.embedding_cache_size` (default 10_000).
- Wraps `EmbeddingProvider.embed_async` via `cached_embed(provider, texts)`: dedupes within a batch
  and across batches. LRU eviction is RLock-protected; pure dict + `collections.OrderedDict` (no extra
  dependency).
- Cache is consulted **before** the provider and populated **after**; on provider failure the cache
  is untouched. Clearable via `EmbeddingCache.clear()`.
- Combined with content-hash delta indexing (§11), re-indexing costs are minimized twice: unchanged
  chunks skip embed, and repeated text across documents reuses cached vectors.

---

## 17. Retrieval Analytics

- `RetrievalAnalytics` — thread-safe (RLock) recorder:
  - `record(entry: RetrievalAnalyticsEntry) -> None` — appended on every retrieval; maintains running
    totals (`total_queries`, sum latency, sum hits, top scores) and a bounded recent window.
  - `summary(collection=None, namespace=None) -> RetrievalAnalyticsSummary` — aggregate statistics.
  - `clear()`.
- Wired into `KnowledgeService.retrieve` (records after each retrieval) and exposed via REST
  (`GET /knowledge/analytics`) plus optionally folded into `MetricsResponse` (additive field) — the
  server metrics collector remains unchanged.

---

## 18. Server REST APIs

### 18.1 Router mounting

- New `app/knowledge/router.py` with `router = APIRouter(prefix="/knowledge", tags=["knowledge"])`.
- Mounted in `create_app()` **after** the workflow routers, with `protected_deps` (auth + rate limit),
  matching the other protected routers (app.py:253-258).
- `app.state.knowledge_service` eagerly resolved in `create_app` (parity with
  `app.state.workflow_service`). `get_knowledge_service(request)` added to `app/server/dependencies.py`.
- New REST schemas in `app/knowledge/schemas.py` (server-layer request/response models, reusing the
  `{total, items}` list pattern).

### 18.2 Endpoints

Namespaces:
- `POST /knowledge/namespaces` — create namespace.
- `GET /knowledge/namespaces` — list.
- `DELETE /knowledge/namespaces/{name}` — delete (requires no collections).

Collections:
- `POST /knowledge/collections` — create collection (`name`, `namespace`, optional
  `chunking`, `embedding_provider`, `vector_store`, `metadata`).
- `GET /knowledge/collections` — list (optional `namespace` filter).
- `GET /knowledge/collections/{name}` — get (requires `namespace` query; default
  `KnowledgeSettings.default_namespace`).
- `DELETE /knowledge/collections/{name}` — delete collection and indexes.

Documents / ingestion:
- `POST /knowledge/collections/{name}/documents` — ingest (multipart `file` upload + `format` +
  `chunking` + `wait` query). `wait=true` returns the document; `wait=false` returns an
  `IngestionJob` (worker).
- `GET /knowledge/collections/{name}/documents` — list documents.
- `GET /knowledge/collections/{name}/documents/{doc_id}` — get document (with version query).
- `GET /knowledge/collections/{name}/documents/{doc_id}/versions` — list versions.
- `PATCH /knowledge/collections/{name}/documents/{doc_id}` — update content/metadata.
- `DELETE /knowledge/collections/{name}/documents/{doc_id}` — delete.

Retrieval:
- `POST /knowledge/collections/{name}/search` — full retrieval (query, filters, strategy, top_k,
  rewrite/multi_query/compression/rerank flags) → `RetrievalResult`.
- `GET /knowledge/collections/{name}/search?q=...` — convenience GET variant.
- `POST /knowledge/collections/{name}/search/stream` — **streaming retrieval** (SSE).

Analytics:
- `GET /knowledge/analytics` — summary (optional collection/namespace filter).

### 18.3 Error mapping

Add handlers in `create_app._register_exception_handlers` (registered **after** the generic
`GeneralAIError` handler):
- `KnowledgeCollectionNotFoundError` / `KnowledgeNamespaceNotFoundError` /
  `KnowledgeDocumentNotFoundError` → 404.
- `KnowledgeValidationError` → 422.
- `KnowledgeDuplicateError` → 409.
- `KnowledgeUnsupportedFormatError` → 415.
- `KnowledgeIngestionError` / `KnowledgeIndexError` / `KnowledgeVersionError` → 500 (or 422 for
  version conflicts).

---

## 19. Streaming Retrieval

- `POST /knowledge/collections/{name}/search/stream` returns a `StreamingResponse` with
  `media_type="text/event-stream"` + the standard headers (`Cache-Control: no-cache`,
  `X-Accel-Buffering: no`, `Connection: keep-alive`) — exactly the `chat.py` / `workflows.py` framing.
- Events emitted via `sse_format` (reusing `app/server/streaming.py`):
  1. `knowledge.query.started` — `{query, collection, namespace, strategy, top_k}`.
  2. `knowledge.hit` — one event per retrieved hit (deduped, ranked): `{index, total, RetrievalHit}`.
  3. `knowledge.source` — a `SourceReference` event (attribution as it becomes known).
  4. `knowledge.completed` — final `RetrievalResult` summary `{total, latency_ms, citations, sources}`.
  5. `knowledge.error` — on failure.
- The stream is produced by `KnowledgeService.retrieve_stream(query)` which yields
  `(event_name, payload)` tuples (same contract as `poll_session` in `app/server/streaming.py:26`),
  keeping the router thin.

---

## 20. Plugin Integration

- Extend the **frozen** `PluginContext` dataclass (`app/plugins/base.py:20`) with **optional** fields
  (default `None` — backward compatible):
  - `knowledge_service: Any | None`
  - `loader_registry: Any | None`
  - `embedding_provider_registry: Any | None`
  - `embedding_provider_factory: Any | None`
  - `vector_store_registry: Any | None`
  - `chunker_registry: Any | None`
  - `retriever_registry: Any | None`
- Plugins register custom components in their `load`/`enable` hooks:
  - **custom loader**: subclass `DocumentLoader`, register with `context.loader_registry`.
  - **custom embedding provider**: subclass `EmbeddingProvider`, register with the provider registry;
    optionally add a builder to the factory.
  - **custom vector store**: implement `VectorStore` ABC + register a factory with
    `vector_store_registry`.
  - **custom retriever**: implement `Retriever` ABC, register with `retriever_registry`.
- New `PluginType` values (`KNOWLEDGE_LOADER`, `KNOWLEDGE_EMBEDDING`, `KNOWLEDGE_VECTOR_STORE`,
  `KNOWLEDGE_RETRIEVER`) — additive enum members, no existing code breaks. (Optional: reuse `MIXED`
  to avoid enum churn; the design proposes explicit types for observability but this is a decision
  point in §27.)
- `PluginManager` auto-wiring: add a `_register_knowledge_provider` hook analogous to
  `_register_llm_provider` (manager.py:777) **only if** a plugin exposes a known knowledge
  capability attribute; otherwise plugin knowledge components are wired through the registries the
  plugin already receives in its context. Plugin teardown follows the "must never raise" convention
  (`_unregister_plugin_workflows`).

---

## 21. Dependency Injection

`app/knowledge/bootstrap.py` — `register_knowledge_components(container)` idempotent, guarded by
`container.has(...)` (mirrors `app/automation/bootstrap.py`):

1. **Settings**: `KnowledgeSettings` (frozen pydantic, `default_namespace`, `default_vector_store`,
   `default_chunking`, `index_workers`, `embedding_cache_size`, `bm25_k1/b`).
2. **Registries (singletons)**: `LoaderRegistry`, `ChunkerRegistry`, `EmbeddingProviderRegistry`,
   `EmbeddingProviderFactory`, `VectorStoreRegistry`, `RetrieverRegistry`, `CollectionRegistry`,
   `NamespaceRegistry`.
3. **Providers (singletons)**: default embedding provider (deterministic) registered with the
   registry + factory; built-in chunkers and loaders registered.
4. **Index infra**: `EmbeddingCache`, `IndexWorker` (built by a factory resolving
   `IngestPipeline`).
5. **Service**: `KnowledgeService` built by `_make_knowledge_service(container)` factory, lazily
   resolving `LLMRouter` (for rewrite/compress/rerank) and `ToolRegistry` (for tool registration)
   when present — the `_try_resolve_*` pattern.
6. **Tools**: `knowledge_tools()` returns `Tool` instances (`kb_search`, `kb_ingest`) registered with
   the `ToolRegistry` during `create_app` discovery, enabling workflow TASK/AGENT steps to call
   knowledge retrieval.

Called from `create_app()` (after `register_automation_components`) and the lifespan — idempotent,
safe both ways. Eagerly resolved onto `app.state.knowledge_service`.

---

## 22. Lifecycle Integration

- In `_build_lifespan` (app/server/app.py:131-160):
  - Startup (after `register_knowledge_components`): resolve `KnowledgeService`, call
    `knowledge_service.startup()` (restores namespaces/collections from registry state, registers
    knowledge tools with `ToolRegistry`, starts `IndexWorker`).
  - Shutdown: `knowledge_service.shutdown()` (stop worker, drain queue, flush analytics).
- `KnowledgeService.startup/shutdown` mirror `WorkflowService.startup/shutdown` (workflow.py:459-486)
  so the lifespan symmetry is preserved.

---

## 23. Events & Observability

New `app/knowledge/events.py` (dotted lowercase `knowledge.*` domain, matching
`app/automation/events.py`):

- `EVENT_KNOWLEDGE_NAMESPACE_CREATED = "knowledge.namespace.created"`
- `EVENT_KNOWLEDGE_COLLECTION_CREATED = "knowledge.collection.created"`
- `EVENT_KNOWLEDGE_COLLECTION_DELETED = "knowledge.collection.deleted"`
- `EVENT_KNOWLEDGE_DOCUMENT_INGESTED = "knowledge.document.ingested"`
- `EVENT_KNOWLEDGE_DOCUMENT_UPDATED = "knowledge.document.updated"`
- `EVENT_KNOWLEDGE_DOCUMENT_DELETED = "knowledge.document.deleted"`
- `EVENT_KNOWLEDGE_INDEX_COMPLETED = "knowledge.index.completed"`
- `EVENT_KNOWLEDGE_RETRIEVED = "knowledge.retrieved"`
- `EVENT_KNOWLEDGE_WORKER_STARTED = "knowledge.worker.started"` / `_STOPPED`

`KnowledgeService` publishes via the `EventBus` (resolved lazily) with `Event(source="knowledge",
event_type=...)`; event payloads in `Event.context` or structured attributes. Structured logging with
`log = logging.getLogger(__name__)` and correlation extras (`collection_id`, `namespace`, `doc_id`,
`job_id`) throughout — matching the executor/scheduler logging style from Phase 12f.

---

## 24. Exceptions (`app/knowledge/exceptions.py`)

`KnowledgeError(GeneralAIError)` with `module="knowledge"`; hierarchy:
- `KnowledgeNamespaceNotFoundError`
- `KnowledgeCollectionNotFoundError`
- `KnowledgeDocumentNotFoundError`
- `KnowledgeValidationError`
- `KnowledgeDuplicateError`
- `KnowledgeUnsupportedFormatError`
- `KnowledgeIngestionError`
- `KnowledgeIndexError`
- `KnowledgeVersionError`
- `KnowledgeCacheError`

Each carries `message`, optional `cause`, and `context` dict (the `GeneralAIError` contract).
Registered with specific handlers in `create_app` (§18.3).

---

## 25. Testing Strategy

New `tests/knowledge/` following repo conventions (fixtures, isolated app):

- `conftest.py` — `knowledge_settings`, `collection_factory` (creates namespace+collection with
  deterministic provider), `ingest_fixture` (ingests a canned markdown/JSON doc), `fake_clock`/`system_clock`
  (reuse `app/automation/time.py`), isolated `create_app(settings=ServerSettings(rate_limit_enabled=False))`.
- `test_loaders.py` — each format (PDF/DOCX/TXT/MD/HTML/CSV/JSON) → expected text; unknown format →
  `KnowledgeUnsupportedFormatError`. Use tiny generated fixtures (build a minimal PDF via pypdf writer,
  minimal DOCX via `Document()` API if python-docx present — skip with `pytest.importorskip` when the
  optional dep is absent).
- `test_chunkers.py` — fixed/recursive/sliding/semantic produce expected chunk counts/boundaries;
  determinism; overlap handling.
- `test_embeddings.py` — deterministic provider: fixed dimensions, normalized, identical input → equal
  vectors; `EmbeddingCache` hit/miss/LRU eviction.
- `test_vectorstores.py` — in-memory + (importorskip) faiss + chroma: add/search/delete/clear;
  metadata filtering post-search; namespace isolation.
- `test_bm25.py` — known corpus → expected ranked order; incremental add/delete.
- `test_hybrid.py` — RRF fusion order, weights, dedupe, per-strategy ranks.
- `test_incremental.py` — unchanged content → no-op; changed content → version bump + delta chunk
  re-embedding.
- `test_versions.py` — immutable versions, update/delete semantics.
- `test_worker.py` — enqueue → job lifecycle → succeeded/failed; concurrent workers; graceful stop.
- `test_service.py` — full pipeline round-trip (ingest → retrieve → citations/sources), namespaces,
  delete cascade.
- `test_router.py` — OpenAPI route existence; auth (with/without API key); 404/422/415/409 mapping;
  multipart ingest; streaming SSE events sequence.
- `test_analytics.py` — counters, summary aggregation, clear.
- `test_plugins.py` — plugin registers a custom loader/embedding/vector/retriever and it is used.
- `test_quality_gate.py` (optional helper) — asserts the four gate commands remain green.

Tests must run **without** faiss/chroma/LLM deps being configured (defaults exercise in-memory +
deterministic paths) so the suite stays green in a minimal environment; optional-dep tests use
`pytest.importorskip`.

---

## 26. Migration Plan & Backward Compatibility

Purely additive — no data migrations, no API changes:

1. **Phase 13a — Foundations**: `app/knowledge/` scaffold, models, exceptions, events, constants,
   `KnowledgeSettings`, loaders (TXT/MD/CSV/JSON + lazy PDF/DOCX/HTML), chunkers, deterministic
   embedding provider, in-memory vector store, BM25, registries, DI bootstrap, `KnowledgeService`
   core (collections/namespaces/ingest/retrieve), lifecycle wiring. Tests green.
2. **Phase 13b — Enterprise features**: incremental indexing, versions, update/delete, embedding
   cache, background `IndexWorker`, retrieval analytics, citations/sources, hybrid retrieval + RRF,
   metadata filters. Tests green.
3. **Phase 13c — Advanced retrieval**: query rewriting (identity + LLM), multi-query, context
   compression, reranking (identity + cross-encoder). Tests green.
4. **Phase 13d — Integration**: REST router + schemas + exception handlers + streaming retrieval,
   knowledge tools (`kb_search`/`kb_ingest`) into `ToolRegistry`, `PluginContext` extension +
   knowledge plugin types + auto-wiring, FAISS/Chroma stores, optional LLM-based rewriterrer/reranker
   via `_try_resolve_llm_router`. Tests green.
5. **Quality gate** run after each sub-phase: pytest, mypy, ruff check, ruff format.

Dependencies: add to `requirements.txt` — `numpy` (required), and **optional** `pypdf`, `python-docx`,
`beautifulsoup4`, `faiss-cpu`, `chromadb`, `sentence-transformers`. All optional deps are imported
lazily so the core module works with only stdlib + numpy + pydantic. (List them as comment-grouped
optional requirements; the lazy-import design means tests pass even without them.)

No existing module, router, model, or test is modified except:
- `app/server/app.py` — add `register_knowledge_components` call, mount `knowledge_router`, add
  `app.state.knowledge_service`, add knowledge exception handlers.
- `app/server/dependencies.py` — add `get_knowledge_service`.
- `app/plugins/base.py` — add optional `PluginContext` fields.
- `app/tools/catalog.py` / `app/tools/categories/*` — add `kb_search`/`kb_ingest` to the discovery
  set (or register from knowledge bootstrap; decision in §27).

---

## 27. Open Design Decisions (confirm before implementation)

1. **DOCX dependency**: `python-docx` is **not installed**. Options: (a) add to requirements.txt and
   install now, or (b) implement a minimal DOCX reader via stdlib `zipfile` + XML parsing (no dep),
   or (c) `pytest.importorskip` the DOCX loader tests and ship the loader behind a lazy import.
   Recommendation: (a) add `python-docx` to requirements (it is a common, small dep) — confirm.
2. **LLM-dependent features** (rewrite/multi-query/compress/rerank): default to identity/no-op when no
   `LLMRouter` is configured, making the suite deterministic offline. Confirm this is acceptable
   rather than requiring a live LLM for tests.
3. **Knowledge tools into `ToolRegistry`**: register `kb_search`/`kb_ingest` automatically at
   `create_app`/lifecycle (via `knowledge_tools()`), or leave tool registration to explicit config.
   Recommendation: auto-register (consistent with `plan_tools()` in app.py:144-145).
4. **PluginType enum**: add explicit `KNOWLEDGE_*` enum members (more introspectable) vs reuse
   `MIXED`. Recommendation: add explicit members (additive).
5. **MetricsResponse extension**: fold `knowledge_collections` / `knowledge_documents` counts into
   `GET /metrics` (additive fields) or leave analytics under `/knowledge/analytics` only.
   Recommendation: add the fields (cheap, consistent with `memory_records`).
6. **Persistent state**: Phase 13 collections/documents are **in-memory** (like workflow stores).
   Confirm no disk persistence is required now (FAISS/Chroma persistence can be added via config
   without API change).

---

## 28. Summary

Phase 13 adds a self-contained **Enterprise Knowledge & RAG system** in `app/knowledge/`: a pluggable
ingestion pipeline (7 formats, 4 chunking strategies, embedding provider abstraction, 3 vector
stores), hybrid retrieval (vector + pure-python BM25 + RRF fusion) with metadata filtering, query
rewriting, multi-query, context compression, reranking, citations, source attribution, collections,
namespaces, incremental indexing, background workers, embedding caching, versioned documents,
analytics, and full plugin support — exposed through REST APIs (including SSE streaming retrieval) and
registered tools. Everything is additive, reuses the DI/bootstrap/FastAPI/exception/lifecycle/testing
patterns established in Phases 8–12, is fully typed (Pydantic v2), thread-safe, and implementable in
four independently-mergeable sub-phases without any breaking change.
