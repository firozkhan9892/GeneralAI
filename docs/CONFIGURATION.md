# Configuration Reference

This reference documents every runtime setting exactly as implemented.

## Settings Precedence

Settings are loaded from (highest to lowest priority):

1. **CLI arguments** — `python main.py --env --log-level --debug`
2. **Environment variables** — prefixed with `GENERAL_AI_`
3. **`.env` file** — in the project root (utf-8)
4. **Defaults** — defined in `app/config/defaults.py`

`AppSettings` (Pydantic `BaseSettings`) reads environment variables and the
`.env` file. LLM provider settings are read from the same `GENERAL_AI_*`
variables by `build_llm_settings_from_env()` in `app/llm/config.py`. Server
and knowledge settings are **programmatic** models passed at construction
time — they are **not** loaded from the environment.

## Application Settings (`AppSettings`)

| Variable | Default | Type | Description |
|---|---|---|---|
| `GENERAL_AI_ENVIRONMENT` | `development` | str | One of `development`, `staging`, `production` |
| `GENERAL_AI_LOG_LEVEL` | `INFO` | str | One of `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `GENERAL_AI_DEBUG` | `false` | bool | Enable debug mode |
| `GENERAL_AI_LOG_TO_CONSOLE` | `true` | bool | Write logs to stdout |
| `GENERAL_AI_LOG_TO_FILE` | `true` | bool | Write logs to `logs/generalai.log` (rotating, 10 MB × 5) |
| `GENERAL_AI_LOG_DIR` | `logs` | Path | Log directory |
| `GENERAL_AI_DATA_DIR` | `data` | Path | Data directory |
| `GENERAL_AI_MODELS_DIR` | `models` | Path | Models directory |

## LLM Settings (environment variables)

Loaded via `build_llm_settings_from_env()` in `app/llm/config.py`.

| Variable | Default | Description |
|---|---|---|
| `GENERAL_AI_API_MODE` | `mock` | `mock` (offline, deterministic, no credentials) or `real` |
| `GENERAL_AI_OPENAI_API_KEY` | — | OpenAI credential |
| `GENERAL_AI_OPENAI_MODEL` | — | Model override (falls back to provider default) |
| `GENERAL_AI_OPENAI_URL` | — | Endpoint override |
| `GENERAL_AI_OPENROUTER_API_KEY` | — | OpenRouter credential |
| `GENERAL_AI_OPENROUTER_MODEL` | — | Model override |
| `GENERAL_AI_OPENROUTER_URL` | — | Endpoint override |
| `GENERAL_AI_GEMINI_API_KEY` | — | Google Gemini credential |
| `GENERAL_AI_GEMINI_MODEL` | — | Model override |
| `GENERAL_AI_OLLAMA_URL` | — | Local Ollama endpoint |
| `GENERAL_AI_OLLAMA_MODEL` | — | Ollama model |

### Mock vs. Real Mode

- **`mock` (default)** — a deterministic offline provider is registered.
  No credentials are read and **no external calls are made**. Ideal for
  development, tests, and CI.
- **`real`** — only providers whose required configuration is present are
  registered:
  - **OpenAI / OpenRouter / Gemini** — require their API key, otherwise they
    are skipped.
  - **Ollama** — no API key; registered only when `GENERAL_AI_OLLAMA_URL` or
    `GENERAL_AI_OLLAMA_MODEL` is set.
  - Missing configuration never raises; unconfigured providers are omitted.

### Multi-LLM Resilience (real mode)

The LLM layer continuously monitors providers and routes traffic around
failures:

- **Health monitoring** — each provider tracks success rate and average
  latency from live calls (`app/llm/health_monitor.py`).
- **Circuit breaker** — a provider that exceeds the failure threshold opens
  its circuit and is skipped until it recovers (half-open probing).
- **Routing** — weighted selection across healthy providers using health,
  latency, capability, and cost signals (`app/llm/llm_router.py`).
- **Fallback chains** — failed calls cascade to the next healthy provider
  (`app/llm/fallback_manager.py`).
- **Request queue + caching** — per-provider rate limits and response
  caching reduce redundant traffic.

## Server Settings (`ServerSettings` — programmatic)

The FastAPI layer is configured with a frozen `ServerSettings` model passed
to `create_app(settings=...)` (`app/server/config.py`). These fields are
**not** read from environment variables. There is **no**
`GENERAL_AI_API_KEY` environment variable.

| Field | Default | Description |
|---|---|---|
| `title` | `GeneralAI API` | OpenAPI title |
| `version` | `1.0.0` | API version |
| `host` | `127.0.0.1` | Bind host (informational) |
| `port` | `8000` | Bind port (informational) |
| `api_key` | `None` | Shared API key; `None` disables authentication |
| `rate_limit_enabled` | `true` | Enable fixed-window rate limiting |
| `rate_limit_per_minute` | `60` | Max requests per identity per minute |
| `cors_origins` | `()` | Allowed CORS origins (empty disables CORS) |

### Authentication

When `api_key` is set, every non-public endpoint requires it via the
`X-API-Key` **header**. Requests without it are rejected with `401`.

> **Security note:** a query-parameter fallback (`?api_key=...`) also
> accepts the key (see `app/server/security.py`). Prefer the header: query
> parameters can leak into logs, proxy metrics, and browser history. The
> fallback exists for convenience and should not be used in production.

### Rate Limiting

A fixed-window limiter keyed by identity (API key if presented, otherwise
client IP). Excess requests receive `429 Too Many Requests` with a
`Retry-After` header.

### CORS

CORS middleware is created only when `cors_origins` is non-empty. An empty
tuple disables CORS entirely.

## Knowledge Settings (`KnowledgeSettings` — programmatic)

Frozen models loaded once at startup and shared via the DI container
(`app/knowledge/config.py`):

| Field | Default | Description |
|---|---|---|
| `default_namespace` | `default` | Namespace used when none is specified |
| `default_chunk_size` | `1000` | Default max characters per chunk |
| `default_chunk_overlap` | `200` | Overlap between consecutive chunks |
| `embedding_cache_size` | `10000` | Max entries in the embedding LRU cache |
| `index_workers` | `2` | Concurrent background indexing workers |
| `keep_versions` | `3` | Document versions retained |

## Example `.env` File

```bash
# Environment
GENERAL_AI_ENVIRONMENT=development
GENERAL_AI_LOG_LEVEL=INFO
GENERAL_AI_DEBUG=false

# Logging toggles
GENERAL_AI_LOG_TO_CONSOLE=true
GENERAL_AI_LOG_TO_FILE=true

# LLM mode: mock (default) or real
GENERAL_AI_API_MODE=mock

# Providers (only required in real mode)
GENERAL_AI_OPENAI_API_KEY=sk-...
GENERAL_AI_OPENAI_MODEL=gpt-4o
GENERAL_AI_GEMINI_API_KEY=...
GENERAL_AI_OPENROUTER_API_KEY=...
GENERAL_AI_OLLAMA_URL=http://localhost:11434
GENERAL_AI_OLLAMA_MODEL=llama3
```

See `.env.example` in the repository root for the complete template.