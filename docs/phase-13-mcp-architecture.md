# Phase 13 — MCP (Model Context Protocol) & External Integrations — Architecture Design

Status: **Proposed** (no implementation yet)
Author: Architecture audit
Date: 2026-08-02
Scope: Design document only. No code is implemented by this document.

---

## 1. Objectives & Scope

Phase 13 introduces **Model Context Protocol (MCP)** support and a general **external-connector**
foundation to GeneralAI. The goals, in priority order:

1. **Expose GeneralAI as an MCP server** so any MCP-capable client (Claude Desktop, Cursor, IDE agents,
   other servers) can use GeneralAI's tools, prompts, resources, workflows, memory, and LLM routing.
2. **Consume external MCP servers as a client** so GeneralAI's LLM router, agents, and workflows can call
   tools/resources hosted by third-party MCP servers (e.g. GitHub MCP, database MCP).
3. **Provide the external-connector foundation** (transport, auth, sessions, capability negotiation)
   that both the MCP server and MCP client share, and that future non-MCP connectors can reuse.

Constraints (from the brief):
- **No breaking changes.** Everything additive; existing HTTP API, workflows, agents, tools, plugins,
  DI, and tests must keep working unchanged.
- **Follow existing patterns** (Phases 8–12): DI container registration via `bootstrap.py`, FastAPI
  routers with `protected_deps`, JSON-schema-style tool metadata, `app.state` singletons, isolated
  test fixtures, exception hierarchy rooted at `GeneralAIError`.
- **Design only** — implementation is a separate follow-up.

Out of scope for Phase 13 core (listed for later phases):
- Non-MCP external connectors (webhooks, OAuth provider APIs) — the connector foundation is built
  so they can be added, but concrete non-MCP connectors are not implemented.
- Full replacement of the HTTP REST API with MCP.

---

## 2. Existing Architecture Audit (as-built)

Audited against the current source tree. Everything below is verified; integration points reference
real modules and types.

### 2.1 Module map (`app/`)

| Module | Responsibility | Key types |
|---|---|---|
| `app/core/` | Kernel infra: DI container, events, exceptions, interfaces, lifecycle, config | `DependencyContainer`, `EventBus`, `GeneralAIError`, `IModule`/`IWorkflow`/`ITool` |
| `app/server/` | FastAPI app factory, routers, auth, rate limiting, SSE streaming | `create_app()`, `ServerSettings`, `require_api_key`, `sse_format` |
| `app/automation/` | Workflow engine (DAG executor, scheduler, registries, stores, events) | `WorkflowService`, `WorkflowExecutor`, `WorkflowStep`, `register_automation_components` |
| `app/plugins/` | Plugin lifecycle (discovery, enable/disable/unload) | `PluginBase`, `PluginContext`, `PluginManager` |
| `app/tools/` | Tool registry, executor, permissions, catalog, models | `ToolRegistry`, `ToolExecutor`, `ToolMetadata`, `ToolParameter` |
| `app/llm/` | Provider-agnostic LLM layer: router, fallback, streaming, capabilities | `LLMRouter`, `BaseLLMProvider`, `CapabilityMatrix`, `Message`/`ToolCall`/`ToolDefinition` |
| `app/agents/` | Agent manager + session orchestration | `AgentManager` |
| `app/kernel/` | Agent internals (deterministic cognitive pipeline) and memory engine | `MemoryEngine` |
| `app/memory/` | Memory stores and retrieval | (memory engine) |
| `app/planner/` | Planning | (planning tools) |
| `app/brain/` | Brain/context | (module-level) |
| `app/config/` | App settings (pydantic-settings) | `AppSettings` |

### 2.2 Dependency injection (`app/core/container/dependency_container.py`)

- `DependencyContainer` — thread-safe, RLock-protected.
- `register_singleton(interface, instance=None, factory=None)` — three forms: pre-built instance,
  zero-arg factory called once, or (no args) the interface itself built via `_build` with automatic
  constructor injection from `typing.get_type_hints`.
- `register_factory(interface, factory)` — transient; factory called on every `resolve`.
- `resolve(interface)` — key is `f"{interface.__module__}.{interface.__qualname__}"`; circular
  dependency detection via `_resolve_stack`.
- **Pattern**: each module owns a `bootstrap.py` with an **idempotent** `register_<module>_components(container)`
  (guarded by `container.has(...)`), safe to call from both the server factory and the lifespan.
  See `app/automation/bootstrap.py` (registers stores → registries → validator/exporter → executor →
  `WorkflowService` factory, lazily resolving `ToolExecutor`/`AgentManager`/`LLMRouter` when present).

### 2.3 FastAPI integration (`app/server/app.py`)

- `create_app(*, container=None, settings=None, discover_tools=True)`.
- Lifespan: `register_agent_manager_components`, `register_automation_components`, then eagerly
  resolves singletons onto `app.state` (`agent_manager`, `llm_router`, `memory_engine`, `tool_registry`,
  `tool_executor`, `workflow_service`).
- Middleware: metrics only (URL path → status). CORS opt-in via `settings.cors_origins`.
- Exception handlers: `_register_exception_handlers(app)` — generic `GeneralAIError → 500` registered
  first, then specific handlers for domain exceptions (409, 404, etc.).
- Routers: `health_router` public; `chat/memory/tools/workflows/workflows_schedule` behind
  `protected_deps = [Depends(require_api_key), Depends(rate_limit)]`. Agent routes registered
  individually because the WS route must NOT get HTTP deps.

### 2.4 Auth & streaming (`app/server/security.py`, `app/server/streaming.py`)

- `ServerSettings` (frozen BaseModel, not pydantic-settings): `api_key`, `rate_limit_enabled`,
  `rate_limit_per_minute`, `cors_origins`.
- `require_api_key` (X-API-Key), `rate_limit` (fixed-window `RateLimiter`).
- `sse_format(...)` helper + `StreamingResponse` used for chat streaming.

### 2.5 Tools (`app/tools/models.py`, registry/executor/permissions)

- `ToolParameter` — JSON-schema-style `param_type` ∈ {`string`, `integer`, `number`, `boolean`,
  `object`, `array`}, `description`, `required`, `default`, `enum`, `items` (for arrays),
  `properties` (for objects).
- `ToolMetadata` — `name`, `description`, `parameters` (list of `ToolParameter`), `category`
  (`ToolCategory`), `authorization` info.
- `ToolRegistry` — registration, discovery (`discover()` for built-ins), `count`, lookups.
- `ToolExecutor` — executes a tool by name with validated args; `ToolPermission` gate.
- **This is a near-1:1 mapping to MCP tool schemas** (MCP uses JSON Schema for `inputSchema`).

### 2.6 LLM layer (`app/llm/`)

- Two-tier architecture (verified):
  - `app/llm/` — provider-agnostic **chat router** with tool-calling protocol.
  - `app/kernel/agent/` + `app/agents/` — **deterministic, rule-based** cognitive pipeline that
    **never calls the LLM**. There is currently **no LLM-driven tool dispatch loop** (no ReAct/function-call
    loop that iterates tool results back into the model).
- `BaseLLMProvider` (`app/llm/base.py`): `name`, `default_model`, `model_info() -> ModelInfo`,
  `generate(request: ChatRequest) -> ChatResponse` (sync abstract), `stream(...) -> Iterator[StreamChunk]`
  (sync abstract), `generate_async`/`stream_async` default offload via `asyncio.to_thread`.
- `BaseHttpProvider` (`app/llm/providers/_base.py`): template hooks `_chat_url`, `_headers`,
  `_build_payload`, `_parse_response`, `_parse_stream_chunk`.
- `app/llm/llm_router.py` — routing/fallback/load-balancing; `capability_matrix.py` — `CapabilityMatrix`
  (register/unregister/get/supports/can_handle over `CapabilityFlag`); `unified_streamer.py`; `registry.py`;
  `factory.py`; `transport.py` (`HttpTransport`).
- Models (`app/llm/models.py`): `Role` (SYSTEM/USER/ASSISTANT/TOOL), `Message` (role, content, name,
  tool_call_id), `ToolCall` (id, name, arguments), `ToolDefinition`. This is already the MCP-shaped
  message/tool vocabulary.

### 2.7 Workflow engine (`app/automation/`)

- `WorkflowStepType` enum (11): TASK, AGENT, LLM, SUBWORKFLOW, TRANSFORM, CONDITIONAL, LOOP, PARALLEL,
  DELAY, APPROVAL, CALLBACK.
- `WorkflowStep` (frozen pydantic): id, type, name, description, `depends_on` (DAG edges), `timeout_s`,
  `retry_policy`, `error_policy` (ABORT/SKIP/RETRY/IGNORE), `input_bindings` (template expressions like
  `${step.X.output.Y}`), `output_mapping`, `condition`, `metadata`; kind-specific fields.
- `WorkflowService` façade: registry/run_registry/executor/validator/exporter/stores + lazily-optional
  `tool_executor`, `agent_manager`, `llm_router`.
- Events (`app/automation/events.py`): `EVENT_WORKFLOW_*` constants; event bus in `app/core/events/`.
- Exceptions (`app/automation/exceptions.py`): `WorkflowError` hierarchy — `WorkflowNotFoundError`,
  `WorkflowVersionError`, `WorkflowValidationError`, `WorkflowApprovalError`, `WorkflowSchedulerError`,
  `WorkflowConcurrencyError`, `WorkflowExecutionError`, `WorkflowStepError`, `WorkflowPause`,
  `WorkflowOutputConflictError` (added in 12f).
- Note: Phase 12f fixed the `step.X.output.Y` template consistency (`_traverse_output` helper in
  `app/automation/context.py` supports both `{"output": ...}` wrapper and legacy raw output).

### 2.8 Plugins (`app/plugins/`)

- `PluginBase`, `PluginContext`, `PluginManager`; plugin types include WORKFLOW; plugins register
  tools/agents/workflows/providers via `PluginContext`. `_unregister_plugin_workflows` uses a broad
  except (teardown must never propagate — documented in 12f).

### 2.9 Config & interfaces

- `app/config/settings.py`: pydantic-settings `AppSettings`; `app/config/defaults.py`.
- `app/core/interfaces/`: placeholder interfaces `IModule`, `IAgent`, `IBrain`, `IEvent`, `IMemory`,
  `IPlanner`, `IPlugin`, `ITool`, `IWorkflow` — thin contracts, not yet wired everywhere.

### 2.10 Testing conventions

- `tests/`, `tests/automation/conftest.py` fixtures: `linear_definition`, `diamond_definition`,
  executor fixture, `system_clock`.
- Server tests: isolated `create_app(settings=ServerSettings(rate_limit_enabled=False))`.
- Quality gates: pytest, mypy (`mypy.ini`), `ruff check`, `ruff format`. Python 3.10. No `pyproject.toml`;
  dependencies in `requirements.txt` only.

### 2.11 Greenfield gaps (verified)

- **No MCP code exists anywhere** in the repo (searched).
- **No external connector / webhook / OAuth / outbound-HTTP integration** infrastructure exists.
- `requirements.txt` has **no MCP SDK** dependency.

---

## 3. Integration Points (mapped)

| Integration | Existing surface | Phase 13 role |
|---|---|---|
| FastAPI | `create_app()`, routers, `protected_deps`, `app.state` | Host MCP Server over Streamable HTTP; share auth/rate-limit; register new module |
| DI | `DependencyContainer`, module `bootstrap.py` pattern | New `app/mcp/bootstrap.py` registering server/client/session/registry singletons |
| Tools | `ToolMetadata`/`ToolParameter`, `ToolRegistry`, `ToolExecutor`, permissions | Expose as MCP tools 1:1; execute MCP-invoked tools via `ToolExecutor` |
| LLM | `LLMRouter`, `BaseLLMProvider`, `CapabilityMatrix`, `Message`/`ToolCall`/`ToolDefinition` | MCP client tools flow into LLM tool-calling; prompts map to LLM requests |
| Workflow | `WorkflowService`, `WorkflowExecutor`, step types | Expose `workflow.execute/list/get` MCP tools; allow MCP-hosted tools as a step source |
| Memory | `MemoryEngine` (app/kernel/memory), `app/memory/` | Expose as MCP resources (`memory://`) |
| Events | `EventBus`, `EVENT_WORKFLOW_*` | MCP client session events → bus; server lifecycle events |
| Auth | `require_api_key`, `ServerSettings.api_key` | MCP HTTP requests reuse header auth; per-request token validation |
| Plugins | `PluginContext` registration | Plugins may register MCP servers/clients; plugin-provided tools auto-exposed |
| Exceptions | `GeneralAIError` hierarchy, server handlers | Map to JSON-RPC error codes |
| Interfaces | `app/core/interfaces/` placeholders | Optional new `IMcp` placeholder if interfaces are touched |

---

## 4. Proposed Module Layout

New module: **`app/mcp/`** (top-level sibling of `automation`, `tools`, `llm`). Follows the module
conventions: own `bootstrap.py`, own `exceptions.py`, own `models.py`, own `__init__.py` exports,
own `tests/mcp/`.

```
app/mcp/
  __init__.py              # public exports
  bootstrap.py             # register_mcp_components(container) — idempotent
  models.py                # JSON-RPC envelope, MCP protocol messages, sessions
  constants.py             # protocol version, JSON-RPC error codes, URI schemes
  exceptions.py            # McpError hierarchy rooted at GeneralAIError
  transport/
    base.py                # McpTransport ABC (server + client)
    streamable_http.py     # Streamable HTTP transport (server + client)
    stdio.py               # stdio transport (client, for local MCP servers) [later phase]
  server/
    __init__.py
    protocol.py            # JSON-RPC dispatch, request handlers
    capabilities.py        # MCP capabilities/version negotiation
    sessions.py            # session registry, state, lifecycle
    resources.py           # resource URI -> content resolver
    tool_bridge.py         # ToolRegistry/ToolExecutor -> MCP tools
    prompt_bridge.py       # prompts -> LLM router
    workflow_bridge.py     # WorkflowService -> MCP tools
  client/
    __init__.py
    client.py              # McpClient (session, initialize, call_tool, read_resource)
    transport.py           # client-side transport selection
    registry.py            # McpServerRegistry (external servers)
    capability_negotiation.py
  connector/
    connector.py           # Connector base (shared foundation)
    auth.py                # credential store, token refresh hooks
  server_router.py         # FastAPI router mounting /mcp
  deps.py                  # FastAPI dependency providers for session/state
```

---

## 5. MCP Server — Streamable HTTP Transport

### 5.1 Transport choice

**Streamable HTTP** (2026-07-28 spec) is the target. Rationale:
- Reuses the existing FastAPI app, auth (`protected_deps`), rate limiting, and SSE streaming
  (`sse_format`) with no new server process.
- Native support for streaming (SSE responses) — required for chat-style tool loops.
- The repo's HTTP layer is already the transport of record (chat/memory/tools/workflows routers).

**stdio** transport is **deferred** to a later phase (useful for local agents; not needed for the
HTTP-first integration).

### 5.2 Protocol contract (Streamable HTTP)

- **One JSON-RPC message per HTTP POST**; no batched requests initially (support batch in a later phase).
- Client MUST send `Accept: application/json, text/event-stream`; server may respond with either.
- `POST /mcp` — main endpoint. Accepts JSON-RPC 2.0 requests/notifications.
- **GET** (optional, SSE stream) — deferred; the spec's server-initiated-message feature is not needed
  in Phase 13 (all flows are request/response or per-request SSE).
- **Origin header validation**: for browser-based clients, validate `Origin` against
  `settings.cors_origins`; mismatch → JSON-RPC error response with code 403. For non-browser clients
  (no `Origin`) validation is skipped. This follows the spec's security requirement and reuses the
  existing CORS allowlist.
- Server SHOULD NOT bind to 0.0.0.0 by default for local MCP usage (DNS-rebinding protection); the
  deployment config may override.

### 5.3 Request flow

1. HTTP POST with JSON-RPC body → FastAPI router `/mcp` (protected by `protected_deps` unless configured
   otherwise; see §11 Auth).
2. Parse envelope (`jsonrpc: "2.0"`, `id`, `method`, `params`). Malformed → error response `-32700`
   parse error / `-32600` invalid request.
3. Dispatch to handler by `method`:
   - `initialize` → capability/version negotiation (§10).
   - `notifications/initialized` → mark session initialized.
   - `tools/list`, `tools/call` (§7).
   - `resources/list`, `resources/read`, `resources/templates/list` (§8).
   - `prompts/list`, `prompts/get` (§9).
   - `ping` → `{}`.
   - Unknown → `-32601` method not found.
4. Response envelope: `{jsonrpc, id, result | error}`. For streaming-capable methods, the handler may
   return an SSE stream (`application/json` response with `text/event-stream` Accept when the request
   opts in).

### 5.4 Server protocol handler

Design: stateless **dispatch table** `method -> handler`, mirroring `app/automation/registries.py`
`StepTypeRegistry` pattern (register/call by key). Handlers are thin adapters over existing services
(`ToolExecutor`, `WorkflowService`, `MemoryEngine`, `LLMRouter`); no new business logic lives in the
MCP server layer.

---

## 6. MCP Client

### 6.1 Role

GeneralAI consumes **external MCP servers** so that:
- The **LLM router** can offer external-server tools as `ToolDefinition`s in chat requests and execute
  `ToolCall`s through them.
- **Workflows** can use TASK/AGENT/LLM steps that reference external tools via a new input binding.
- **Agents** (future) can use MCP tools in a tool-dispatch loop.

### 6.2 Client design

- `McpClient` — one session per remote server; owns `initialize` handshake, capability table
  (`ProtocolVersion`, `capabilities`), `tools/list`, `resources/list`, `call_tool`, `read_resource`,
  `ping`, and JSON-RPC request/notification plumbing over the selected transport.
- `McpServerRegistry` — thread-safe registry of configured external MCP servers
  (id → `McpServerConfig` + connected `McpClient`), mirroring `CapabilityMatrix` shape and the
  `ToolRegistry` registration pattern.
- **Capability negotiation** on connect: store per-server `ProtocolVersion` and capability flags
  (`tools`, `resources`, `prompts`, `streaming`, `sampling`, etc.). `supports(server_id, flag)`
  mirrors `CapabilityMatrix.supports`.
- **Tool discovery → `ToolDefinition`**: on connect, `tools/list` results are translated to
  `app.llm.models.ToolDefinition` (name prefixed `mcp_<server>.<tool>` to avoid collisions) and the
  existing tool-calling vocabulary is reused.

### 6.3 Client transport

- Phase 13: **Streamable HTTP client transport** over the existing `HttpTransport` infrastructure
  (`app/llm/transport.py`) — reuse request/retry/header plumbing.
- stdio client transport is deferred (later phase) but the `McpTransport` ABC (see §12) leaves a seam.

### 6.4 Lifecycle

- Client connections are **lazy and event-driven**: connect on first use, reconnect with backoff,
  refresh capability table. Managed by a singleton `McpClientManager` resolved from DI and surfaced on
  `app.state`; `shutdown()` closes all client sessions (lifespan parity with `workflow_service.shutdown()`).

---

## 7. Tool Exposure (Server side)

### 7.1 Tool list

`ToolMetadata` → MCP `Tool`:
```
{
  "name": metadata.name,
  "description": metadata.description,
  "inputSchema": {
    "type": "object",
    "properties": {param.name: <json-schema for ToolParameter>},
    "required": [p.name for p in parameters if p.required]
  }
}
```
Mapping is mechanical because `ToolParameter` is already JSON-schema-shaped:
- `string` → `{"type": "string"}`
- `integer` → `{"type": "integer"}`
- `number` → `{"type": "number"}`
- `boolean` → `{"type": "boolean"}`
- `object` → `{"type": "object", "properties": <ToolParameter.properties>}`
- `array` → `{"type": "array", "items": <ToolParameter.items>}`
Plus `description`, `enum`, `default`.

### 7.2 Tool call

`tools/call` → `ToolExecutor.execute(name, arguments)`:
- Validate args against `ToolMetadata` (existing validator), enforce `ToolPermission` (existing
  permissions gate).
- Result → MCP `CallToolResult` (`{content: [{type: "text", text: <json>}], isError: false}`).
- Tool errors (permission denied, not found, execution failure) → `isError: true` result with the
  message, **not** a JSON-RPC error, per MCP convention. Registry/validation failures stay structured:
  `isError` carries the tool-level outcome; JSON-RPC `error` is reserved for protocol-level failures.

### 7.3 Scope

Which tools are exposed is governed by **explicit configuration** (allowlist) — never implicit
"everything is public." Default: built-in + planning tools + registered plugin tools, subject to the
MCP auth policy (§11). No tool requiring secrets is exposed until the connector auth layer exists.

---

## 8. Resource System

### 8.1 Model

MCP resources = named content addressable by URI. GeneralAI exposes:

| URI scheme | Backing store | Example |
|---|---|---|
| `memory://sessions/<id>` | `MemoryEngine` session context | `memory://sessions/abc123` |
| `memory://collections/<name>` | `app/memory/` store | `memory://collections/kb` |
| `workflow://<id>` | `WorkflowService` definition | `workflow://wf_42` |
| `workflow://runs/<run_id>` | `WorkflowRunRegistry`/store | `workflow://runs/run_7` |
| `context://brain/<topic>` | `app/brain/` (later) | `context://brain/goals` |

### 8.2 Resource registry & templates

- `ResourceRegistry` — thread-safe URI → resolver mapping (register/query), mirroring the
  `ToolRegistry`/`CapabilityMatrix` shape.
- `resources/templates/list` returns templates like `memory://sessions/{session_id}`,
  `workflow://runs/{run_id}`.
- `resources/list` + `resources/read` resolve via the registry. Read results are text/JSON MIME-typed.
- **Permission-aware**: resource resolution checks the same auth/token used for the request (§11).
  Sensitive stores (e.g. memory containing secrets) are gated by the allowlist.

---

## 9. Prompt Exposure

- Prompts in MCP are **server-side reusable prompt templates** the client can render and send to an LLM.
- GeneralAI **has no server-side prompt catalog today** — chat requests are constructed by the LLM
  router/clients. Phase 13 introduces a small `PromptRegistry` (id → template string + optional
  parameter schema + optional `Message`-build function).
- Phase 13 scope: **minimal** prompt catalog derived from existing agent system prompts, e.g.
  `generalai.plan`, `generalai.analyze`. `prompts/list` returns registry contents; `prompts/get`
  renders with the client-supplied arguments into `Message[]` via the existing `Message`/`Role` models.
- This is deliberately light — the primary Phase 13 value is tools + resources + workflows.

---

## 10. Capability Negotiation

### 10.1 MCP server `initialize`

Response must advertise:
- `protocolVersion` — the MCP protocol version this server implements (constants.py).
- `capabilities.tools` — `listChanged` optional; set `listChanged: false` initially.
- `capabilities.resources` — with `subscribe` disabled initially.
- `capabilities.prompts` — present.
- `capabilities.logging` — present (maps to existing structured logging).
- `serverInfo` — `{name: "generalai", version: <settings.version>}`.

### 10.2 MCP client negotiation

On connecting to an external server: select highest mutually supported `protocolVersion`, read the
server's `capabilities`, and store them in `McpServerRegistry` keyed by server id — same shape as
`CapabilityMatrix` (`register`/`get`/`supports(server_id, flag)`). Later client calls that require an
unsupported capability short-circuit with a structured error.

### 10.3 Unified capability view

`app/llm/capability_matrix.py` keeps tracking **LLM provider** capabilities (unchanged). A new
`McpServerRegistry` (or a thin adapter registered in the container) tracks **MCP server** capabilities.
No merging of the two tables; the LLM router asks the MCP registry only when it needs MCP-hosted tools.

---

## 11. Authentication & Authorization

### 11.1 Server side (MCP served by GeneralAI)

- Reuse `protected_deps = [Depends(require_api_key), Depends(rate_limit)]` by default — an MCP client
  is authenticated the same way as a REST client: `X-API-Key`.
- The `/mcp` router is mounted with the **same dependency policy** as other protected routers.
  A future config flag (`ServerSettings.mcp_public`) can mount it without auth for local-only
  deployments (default off; documented security risk).
- Origin validation (§5.2) for browser clients.
- **Per-request token context**: the authenticated API key (or session token) is carried on the request
  into tool/resource handlers so `ToolPermission` and resource ACLs evaluate against the caller, not
  a global policy. This reuses `require_api_key` and extends it with a lightweight request-scoped
  principal (stored on the request/state, mirroring how `app.state` carries services).

### 11.2 Client side (GeneralAI consuming external MCP servers)

- `McpServerConfig` holds credential material: `{type, token | client_id+client_secret, scopes}`.
- **Credential store** (`app/mcp/connector/auth.py`) keeps credentials out of code/config-in-repo:
  loaded from environment/secret provider, with an `AuthProvider` hook for token refresh (OAuth-style)
  deferred to the connector phase.
- Transport attaches auth headers via the existing `HttpTransport` header hook.

### 11.3 No new auth protocol

No OAuth2.1/OAuth2.0 server implementation in Phase 13. The server supports static API-key auth
(native) and the client supports static token/bearer with a refresh hook seam.

---

## 12. Transport Abstraction

`McpTransport` ABC (server + client):
```
class McpTransport(ABC):
    name: str
    async def send(self, message: dict) -> dict          # request/response
    async def stream(self, message: dict) -> AsyncIterator[dict]
    async def close(self) -> None
```
- `StreamableHttpServerTransport` — wraps FastAPI request/response; exposes `handle(request) -> Response`
  so the router stays thin.
- `StreamableHttpClientTransport` — wraps `HttpTransport`.
- stdio client transport stub (raises `NotImplementedError`) left as a seam.

This mirrors `app/llm/transport.py`'s `HttpTransport` and `BaseHttpProvider` hook philosophy: transports
are swappable, protocol logic lives above them.

---

## 13. Session Management

- `McpSession` — per-MCP-client-server session: `session_id`, `protocol_version`, `capabilities`,
  `initialized` flag, `last_seen`, `server_info`. Created on first `initialize`, upgraded on
  `notifications/initialized`.
- `McpSessionStore` — thread-safe `session_id -> McpSession` map with TTL idle eviction (reuse
  `app/automation/scheduler.py`'s time/clock injection pattern via `system_clock` for testability).
- Sessions are **transient** (in-memory) in Phase 13; no persistence. Stateless JSON-RPC methods
  (`tools/call`, `resources/read`) work without a session — session is required only for
  negotiated/streaming interactions, matching Streamable HTTP semantics.
- Client-side sessions are owned by `McpClientManager` (one per remote server) and re-established on
  failure with backoff.

---

## 14. Streaming

- **Outbound (server → client)**: streaming-capable MCP methods (future chat/tool loops) respond via
  SSE, reusing `app/server/streaming.py` (`sse_format`, `StreamingResponse`). Phase 13 does **not**
  implement server-initiated messages (no GET SSE channel) — per-request `text/event-stream` responses
  only.
- **Inbound (client)**: `call_tool`/`read_resource` on external servers are request/response; if a
  server advertises `streaming`, `McpClient.stream(...)` consumes an SSE response through
  `unified_streamer`-compatible iteration.
- LLM streaming is **unchanged** — MCP does not wrap the existing chat SSE.

---

## 15. Error Model

### 15.1 JSON-RPC error codes (server)

Standard JSON-RPC 2.0 + MCP application-level codes, centralized in `app/mcp/constants.py`:

| Code | Meaning | Maps to |
|---|---|---|
| `-32700` | Parse error | `McpParseError` |
| `-32600` | Invalid request | `McpInvalidRequestError` |
| `-32601` | Method not found | `McpMethodNotFoundError` |
| `-32602` | Invalid params | `McpInvalidParamsError` (from tool arg validation) |
| `-32603` | Internal error | any unhandled `GeneralAIError` |
| `-32000` | Invalid server error | protocol contract violations |
| `403` | Origin header validation failed | `McpOriginDeniedError` |

### 15.2 Mapping from domain exceptions

- `WorkflowNotFoundError`/`WorkflowVersionError` → tool-level `isError` result (they surface from
  `workflow_*` MCP tools), not JSON-RPC errors.
- `WorkflowValidationError` → JSON-RPC `-32602` when a `workflow.execute` call is malformed.
- `GeneralAIError` base → `-32603` internal, logged via existing structured logging.
- Tool execution failures → `CallToolResult.isError` (§7.2).

### 15.3 Hierarchy

`McpError(GeneralAIError)` base with a `code` attribute; specific subclasses carry `code` + `data`.
Server exception handlers translate `McpError` to JSON-RPC responses; unknown exceptions are caught and
mapped to `-32603` so a JSON-RPC error is **always** returned (never a raw HTML 500).

---

## 16. DI Integration

New `app/mcp/bootstrap.py` with idempotent `register_mcp_components(container)` following
`app/automation/bootstrap.py` exactly:
- Guard every registration with `container.has(...)`.
- Register singletons for: `McpSessionStore`, `ResourceRegistry`, `PromptRegistry`,
  `McpServerRegistry`, `McpClientManager`, and (lazily built) `McpServer` protocol dispatcher.
- `McpClientManager` factory resolves existing singletons (`LLMRouter`, `HttpTransport`,
  `ToolExecutor`, `WorkflowService`) when present; returns a manager with `None` collaborators
  otherwise — same `_try_resolve_*` pattern as `app/automation/bootstrap.py`.
- Called from `create_app()` (after the other modules) and the lifespan — idempotent, safe both ways.
- Optional new placeholder `app/core/interfaces/imcp.py` (`IMcp`) if interfaces are touched, but the
  concrete classes are the primary integration (matching `IWorkflow`'s status).

---

## 17. FastAPI Integration

- New `app/mcp/server_router.py` exposing a single router:
  - `POST /mcp` → `McpServer.handle(request)` (Streamable HTTP).
  - Optional `GET /mcp` SSE stream — **deferred** (not needed for per-request flows).
- Mounted in `create_app()`:
  - With `protected_deps` by default (API key + rate limit) — consistent with other protected routers.
  - `app.state.mcp_server` resolved eagerly (parity with `app.state.workflow_service`).
- Lifespan additions:
  - `await mcp_client_manager.connect_all()` on startup (optional; lazy is default) and
    `await mcp_client_manager.shutdown()` in the finalizer.
- Exception handler: `_register_exception_handlers` gains a `McpError → JSON-RPC error` handler
  (registered **after** the generic `GeneralAIError` handler so the more specific one wins).
- No changes to existing routers/deps; the MCP router is additive.

---

## 18. Workflow Integration

### 18.1 Workflows as MCP tools (server side)

`workflow_bridge.py` maps `WorkflowService` methods to MCP tools:

| MCP tool | Backing call |
|---|---|
| `workflow.list` | `WorkflowService.list()` |
| `workflow.get` | `WorkflowService.get_definition(id)` |
| `workflow.execute` | `WorkflowService.execute(...)` |
| `workflow.status` | run-status lookup |
| `workflow.cancel` | cancel/abort a run |

Arguments are JSON-schema-built from the definition/input shapes; outputs reuse `WorkflowRunContext`
outputs (the 12f `_traverse_output` normalization means `step.X.output.Y` references keep working).

### 18.2 External MCP tools inside workflows

- A workflow `input_bindings` expression gains support for `mcp:<server>.<tool>?{params}` (or a
  dedicated new step field) so TASK steps can invoke external MCP tools. Design detail to finalize in
  implementation; backward-compatible because unknown binding prefixes are **not** interpreted today
  and would have errored — we define the new prefix explicitly.
- `WorkflowExecutor`'s TASK/AGENT execution path consults `McpClientManager` when the binding prefix
  matches; otherwise behaves exactly as before.
- No change to `WorkflowStepType` enum (no new step type required); the binding namespace is the seam.

### 18.3 Events

`EventBus` is reused; MCP client session lifecycle and external-tool failures emit events using the
existing event constants pattern (`EVENT_*`) for observability parity.

---

## 19. Agent Integration

- `app/agents/AgentManager` + `app/kernel/agent/` pipeline stays **deterministic** — unchanged.
- Phase 13 introduces **no** LLM-driven agent tool loop. Agents expose themselves as MCP tools
  (`agent.run`, `agent.status`) through the same tool bridge, so an MCP client can drive an agent.
- The **future** LLM tool-dispatch loop (the gap noted in §2.6) can consume MCP-hosted tools via
  `ToolDefinition`/`ToolCall` once built — but that loop is explicitly out of Phase 13 scope.

---

## 20. LLM Integration

- **Chat + tool calling** (`app/llm/`): external MCP tools are translated to `ToolDefinition` and
  surfaced to `LLMRouter` when requested (opt-in via `McpServerRegistry` inclusion set). When the LLM
  emits a `ToolCall`, execution routes through the MCP client (`call_tool`) and the result returns as
  a `TOOL`-role `Message` — reusing the existing `Role`/`Message`/`ToolCall` models unchanged.
- **Router layering**: `LLMRouter` delegates tool availability to a `ToolProvider` adapter that can
  serve both local (`ToolRegistry`) and MCP-hosted tools. This keeps `app/llm/` free of MCP imports.
- **Prompts**: `PromptRegistry` renders into `Message[]` via `app/llm/models.py`; rendering itself
  never calls an LLM — it only builds messages.
- No changes to `BaseLLMProvider`/`BaseHttpProvider`; the LLM layer treats MCP tools exactly like local
  tools once they are `ToolDefinition`s.

---

## 21. External Connector Architecture (foundation)

Shared foundation in `app/mcp/connector/` reused by MCP client and future non-MCP connectors:
- `Connector` ABC — `name`, `connect()`, `close()`, `is_connected`; `McpClient` implements it.
- `AuthProvider` ABC — `credentials() -> dict`, `refresh()`; default `StaticTokenAuthProvider`
  (env/secret-backed), OAuth refresh seam for later.
- `CredentialStore` — keeps secrets out of the repo; `settings`/env driven.
- `EndpointConfig` — base config (`base_url`, `timeout_s`, `retry_policy`, `auth`).

Rationale: the auth/session/transport/retry seams are shared; concrete non-MCP connectors (REST
webhooks, OAuth services) become small `Connector` implementations in later phases without reworking
the foundation.

---

## 22. Plugin Integration

- `PluginContext` gains **registration hooks**: `register_mcp_server(config)` and
  `register_mcp_client(config)`.
- **Server side**: plugin-registered MCP servers (e.g. a plugin exposing a custom tool set) register
  tool/resource/prompt entries into the same registries the core MCP server serves, so a plugin's tools
  automatically appear in `tools/list` without new code.
- **Client side**: plugin-registered external MCP servers are added to `McpServerRegistry`; the plugin's
  `shutdown()` closes its client sessions (reusing the teardown-must-not-raise convention from
  `_unregister_plugin_workflows`).
- No change to `PluginBase`; `PluginContext` extension is additive.

---

## 23. Testing Strategy

Follow existing conventions (`tests/`, fixtures, isolated app):

- `tests/mcp/`
  - `test_protocol.py` — JSON-RPC envelope parsing, method dispatch, error codes
    (`-32601`, `-32602`, `-32700`), unknown-method handling.
  - `test_tool_bridge.py` — `ToolMetadata → inputSchema` mapping (all 6 param types),
    `tools/call` → `ToolExecutor`, `isError` semantics, permission enforcement.
  - `test_resource_bridge.py` — `resources/list`/`read`, templates, URI resolution,
    permission gating.
  - `test_prompt_bridge.py` — `prompts/list`/`get`, template rendering into `Message[]`.
  - `test_session.py` — initialize/initialized lifecycle, idle TTL eviction (uses `system_clock`
    fixture), session-required vs stateless methods.
  - `test_capabilities.py` — negotiation, unsupported-capability short-circuit.
  - `test_server_http.py` — `POST /mcp` through `create_app(settings=ServerSettings(
    rate_limit_enabled=False))`, auth policy (with/without API key), Origin validation (403),
    error mapping to JSON-RPC responses (never raw HTML 500).
  - `test_client.py` — client `initialize`, `tools/list → ToolDefinition`, `call_tool`, reconnect
    with backoff, `McpServerRegistry` thread-safety.
  - `test_workflow_bridge.py` — `workflow.execute` MCP tool runs a real workflow
    (`linear_definition`/`diamond_definition` fixtures), outputs normalized via `_traverse_output`.
  - `test_connector.py` — `Connector`/`AuthProvider` seams, credential store, refresh hook.
- Conformance: a small fixture serving canned MCP messages verifies the server against the Streamable
  HTTP contract (Accept header handling, per-POST message, error envelope).
- Quality gates: pytest + mypy + ruff check + ruff format, all green.

---

## 24. Migration Plan

Purely additive; no data or API migrations.

1. **Phase 13a — Server (inbound)**: `app/mcp/` scaffold, Streamable HTTP transport,
   `POST /mcp` router behind `protected_deps`, JSON-RPC dispatch, `initialize`/capabilities,
   tool bridge (local tools + `workflow.*` tools), resource bridge, session store, error mapping,
   DI bootstrap, FastAPI wiring. Tests green. Feature toggled **on** (additive) but scoped to
   allowlisted tools.
2. **Phase 13b — Client (outbound)**: `McpClient`, `McpServerRegistry`, client transport,
   capability negotiation, tool discovery → `ToolDefinition`, `call_tool`, `McpClientManager`
   lifecycle + shutdown. Tests green. Opt-in config only (no external server is auto-connected).
3. **Phase 13c — Workflow + LLM binding**: `mcp:` binding support in workflow steps, `ToolProvider`
   adapter for `LLMRouter`, `EventBus` integration. Tests green.
4. **Phase 13d — Connector foundation + plugins**: `Connector`/`AuthProvider`/`CredentialStore`,
   `PluginContext` registration hooks, plugin shutdown teardown. Tests green.
5. **Later phases (out of scope)**: stdio transports, server-initiated SSE, batch JSON-RPC,
   LLM tool-dispatch agent loop, non-MCP connectors (webhooks, OAuth services), prompt catalog
   expansion.

Each sub-phase leaves the codebase green (pytest/mypy/ruff) and merges independently; no sub-phase
changes existing behavior.

---

## 25. Backward Compatibility & Config

- All MCP features are additive and off-by-default where they could change observable behavior:
  - External MCP clients only connect when configured.
  - `mcp:` bindings only affect steps that use them (unknown prefixes were already errors).
  - No changes to existing `WorkflowStepType`, `WorkflowStep`, `ToolMetadata`, `Message`,
    `ServerSettings` (adds **new optional** fields only: `mcp_enabled`, `mcp_public`,
    `mcp_tool_allowlist`).
- New optional `ServerSettings` fields:
  - `mcp_enabled: bool = False` — master switch.
  - `mcp_public: bool = False` — mount `/mcp` without `protected_deps` (local-only; security risk
    documented).
  - `mcp_tool_allowlist: list[str] = []` — tools exposed via MCP (empty = none; explicit opt-in).
  - `mcp_external_servers: list[...] = []` — configured outbound servers (empty = no connections).
- `requirements.txt`: no MCP SDK dependency is **required**; the design implements the small JSON-RPC
  + Streamable HTTP surface directly (reusing `HttpTransport`, `sse_format`, pydantic). Decision point:
  optionally add the official `mcp` package for client conformance testing only, never at runtime.

---

## 26. Open Design Decisions (to confirm before implementation)

1. **SDK vs self-contained**: runtime JSON-RPC/Streamable HTTP implementation vs official `mcp` SDK.
   Recommendation: **self-contained** (matches repo's no-heavy-dependency ethos; reuses
   `HttpTransport`/pydantic); use the SDK only in optional conformance tests.
2. **Tool naming**: prefix outbound MCP tools (`mcp_<server>.<tool>`) — confirm no tool name
   collisions with local names are acceptable.
3. **`mcp:` binding syntax** for workflow steps — confirm the `mcp:<server>.<tool>` prefix approach
   (vs. a new step field) before implementing.
4. **Session persistence**: in-memory transient sessions confirmed acceptable for Phase 13
   (no multi-node/restart durability).
5. **Auth for external servers**: static token/bearer + refresh hook only — confirm no OAuth2 client
   flow is needed in Phase 13.
6. **Interfaces**: whether to add an `IMcp` placeholder to `app/core/interfaces/` now (consistency)
   or skip until the interfaces module is actually wired.

---

## 27. Summary

Phase 13 adds a self-contained **MCP server** (Streamable HTTP, `POST /mcp`, JSON-RPC 2.0) exposing
existing tools, workflows, memory, and minimal prompts; an **MCP client** consuming external MCP
servers into the existing LLM tool-calling vocabulary; and a **connector foundation**
(auth/session/transport seams) shared with future external integrations. Everything is additive,
reuses the DI/bootstrap/FastAPI/exception/testing patterns established in Phases 8–12, and is
implementable in four independently-mergeable sub-phases without any breaking change.
