# Configuration Reference

## Settings Precedence

Settings are loaded from (highest to lowest priority):

1. **CLI arguments** — override everything
2. **Environment variables** — prefixed with `GENERAL_AI_`
3. **`.env` file** — in project root
4. **Defaults** — defined in `app/config/defaults.py`

## App Settings

| Variable | Default | Type | Description |
|---|---|---|---|
| `GENERAL_AI_ENVIRONMENT` | `development` | str | Runtime environment |
| `GENERAL_AI_LOG_LEVEL` | `INFO` | str | Logging level |
| `GENERAL_AI_DEBUG` | `false` | bool | Enable debug mode |
| `GENERAL_AI_LOG_TO_CONSOLE` | `true` | bool | Log to stdout |
| `GENERAL_AI_LOG_TO_FILE` | `true` | bool | Log to file |
| `GENERAL_AI_DATA_DIR` | `./data` | Path | Data directory |
| `GENERAL_AI_LOG_DIR` | `./logs` | Path | Log directory |
| `GENERAL_AI_MODELS_DIR` | `./models` | Path | Models directory |

## Server Settings

| Variable | Default | Type | Description |
|---|---|---|---|
| `api_key` | `None` | str | Shared API key (None = disabled) |
| `host` | `0.0.0.0` | str | Bind address |
| `port` | `8000` | int | Bind port |
| `title` | `GeneralAPI` | str | OpenAPI title |
| `version` | `0.1.0` | str | API version |
| `rate_limit_enabled` | `true` | bool | Enable rate limiting |
| `rate_limit_per_minute` | `60` | int | Requests per minute |
| `cors_origins` | `()` | tuple | Allowed CORS origins |

## Knowledge Settings

| Variable | Default | Type | Description |
|---|---|---|---|
| `default_namespace` | `default` | str | Default namespace |
| `default_chunk_size` | `1000` | int | Characters per chunk |
| `default_chunk_overlap` | `200` | int | Overlap between chunks |
| `embedding_cache_size` | `10000` | int | LRU cache entries |
| `index_workers` | `2` | int | Background workers |
| `keep_versions` | `3` | int | Document versions retained |

## LLM Settings

| Setting | Default | Description |
|---|---|---|
| Provider priority | `mock > ollama > openai > gemini > openrouter` | Fallback order |
| Health check interval | 30s | Provider health monitoring |
| Circuit breaker threshold | 5 failures | Before opening circuit |
| Request timeout | 60s | Per-request timeout |
| Max retries | 3 | Automatic retry attempts |

## Example `.env` File

```bash
# Environment
GENERAL_AI_ENVIRONMENT=production
GENERAL_AI_LOG_LEVEL=WARNING
GENERAL_AI_DEBUG=false

# Server
API_KEY=your-secret-key-here
HOST=0.0.0.0
PORT=8000
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=120
CORS_ORIGINS=http://localhost:3000,https://app.example.com

# Knowledge
DEFAULT_NAMESPACE=production
EMBEDDING_CACHE_SIZE=50000
INDEX_WORKERS=4
```
