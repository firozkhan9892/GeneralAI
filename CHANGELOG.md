# Changelog

All notable changes to GeneralAI are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-09-13

Production-grade foundation — the first stable release.

### Added

- `CHANGELOG.md`.
- Pydantic v2 warning regression tests (`TestPydanticSchemaWarnings`) guarding
  against `UnsupportedFieldAttributeWarning` (no `Field(alias=...)` usage exists).
- Complete `.env.example` template covering every `GENERAL_AI_*` variable.

### Changed

- Version bumped to `1.0.0`; PyPI classifier promoted from `3 - Alpha` to
  `5 - Production/Stable`; CI packaging assertions updated accordingly.
- `docs/CONFIGURATION.md` rewritten to match the implemented settings exactly:
  env-driven `AppSettings` and LLM provider variables, and programmatic
  (non-env) `ServerSettings`/`KnowledgeSettings`.
- Docker guidance now reflects the repository (no `Dockerfile` /
  `docker-compose.yml` are shipped); the ineffectual `GENERAL_AI_API_KEY`
  environment variable was removed from all documentation.
- `docs/DEVELOPER_GUIDE.md` and `docs/INSTALL.md` updated to mirror CI commands.
- Root and `app` package `__version__` unified at `1.0.0`.

### Fixed

- Removed development artifacts (`pytest_output.txt`, `tests/audit_report.txt`)
  and dead placeholder modules (`app.planner`, `app.brain`) from the release tree.
- Eliminated all `RuntimeWarning: coroutine ... was never awaited` leaks in the
  test suite by awaiting manager shutdown; orchestrator attributions were
  confirmed as GC-time bleed, not genuine leaks. Production lifecycle already
  shuts down agents correctly.
- Alignment of documentation with actual server defaults (`host` 127.0.0.1,
  `title` "GeneralAI API", CORS disabled by default).

## [0.1.0] - 2026-08-01

Foundational development series (phases 8–14), released incrementally.

### Added (Phase 8) — Foundation

- Clean-architecture core: DI container, event bus, lifecycle manager,
  module and plugin registries.

### Added (Phase 9) — FastAPI Server & Security

- FastAPI application factory with lifespan management, WebSocket and SSE
  support, health/metrics endpoints.
- Shared API-key authentication (`X-API-Key`), fixed-window rate limiting,
  CORS support.

### Added (Phase 10) — Plugin & Extension System

- Declarative plugin manifests, plugin loading via entry points, dependency
  resolution, and sandboxed execution.

### Added (Phase 11) — Multi-LLM Intelligence Layer

- Multi-provider routing with health monitoring, circuit breakers, fallback
  chains, request queueing, caching, capability matrix, and cost optimizer.

### Added (Phase 12) — Workflow Engine

- DAG-based workflow definitions, validation, registries, persistence with
  versioning, scheduler, and executor with step-level retries.

### Added (Phase 13) — Enterprise RAG

- Knowledge ingestion pipeline, chunking with versioning, embeddings and
  vector stores (in-memory reference implementation plus optional FAISS and
  ChromaDB), BM25, hybrid retrieval with RRF, and citation generation.

### Added (Phase 14) — Production Readiness

- 14.1 Production documentation and developer guides.
- 14.2 Knowledge/RAG server integration.
- 14.3 Packaging foundation (`pyproject.toml`, `python -m build`).
- 14.4 GitHub Actions CI with automated quality gates.
- 14.5 Production architecture refactor.
- 14.6 Config-driven LLM provider integration (`mock`/`real` modes via
  `GENERAL_AI_*` environment variables).
- 14.7 LLM provider runtime hardening.
- 14.8 Provider output and format integrity.
- 14.9 Provider health and failure management.
- 14.10 Verified LLM failure and fallback paths (mypy-clean tests).
- 14.11 Fallback chain wired into `LLMRouter` generation paths.

### Fixed

- LLM bootstrap idempotency (repeat registration no longer duplicates providers).
- Minor bug fixes throughout the foundation, server, and workflow layers.