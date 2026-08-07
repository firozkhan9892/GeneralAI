# REST API Reference

## Authentication

All protected endpoints require an API key via the `X-API-Key` header:

```
X-API-Key: your-secret-key
```

Or as a query parameter: `?api_key=your-secret-key`

## Health & Metrics

### GET /health

Returns server health status.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "uptime_seconds": 1234.5,
  "modules": {
    "kernel": "ready",
    "llm": "ready",
    "workflow": "ready"
  }
}
```

### GET /metrics

Returns request metrics.

**Response:**
```json
{
  "total_requests": 1000,
  "requests_per_endpoint": {
    "/health": 500,
    "/agent/run": 300,
    "/workflows": 200
  },
  "status_codes": {
    "200": 950,
    "401": 25,
    "429": 25
  }
}
```

## Agent Endpoints

### POST /agent/run

Execute an agent with a prompt.

**Request:**
```json
{
  "prompt": "What is the weather?",
  "session_id": "optional-session-id",
  "stream": false
}
```

**Response:**
```json
{
  "session_id": "abc123",
  "status": "completed",
  "response": "The weather is...",
  "tokens_used": 150,
  "duration_ms": 1234.5
}
```

### POST /agent/cancel

Cancel a running agent session.

**Request:**
```json
{
  "session_id": "abc123"
}
```

### GET /agent/status/{session_id}

Get the status of an agent session.

**Response:**
```json
{
  "session_id": "abc123",
  "status": "running",
  "current_step": "reasoning",
  "progress": 0.65
}
```

### GET /agents

List all active agent sessions.

**Response:**
```json
{
  "sessions": [
    {
      "session_id": "abc123",
      "status": "running",
      "created_at": "2026-01-01T00:00:00Z"
    }
  ],
  "total": 1
}
```

### WS /agent/ws

WebSocket endpoint for streaming agent responses.

**Message Format:**
```json
{
  "type": "token",
  "data": "Hello"
}
```

## Chat Endpoints

### POST /chat/stream

Stream a chat response (SSE).

**Request:**
```json
{
  "message": "Hello, AI",
  "session_id": "optional"
}
```

**Response:** Server-Sent Events stream

## Memory Endpoints

### GET /memory/search

Search the memory engine.

**Query Parameters:**
- `q` (required): Search query
- `limit` (optional): Maximum results (default: 10)
- `session_id` (optional): Filter by session

**Response:**
```json
{
  "results": [
    {
      "content": "Previous conversation...",
      "score": 0.85,
      "timestamp": "2026-01-01T00:00:00Z"
    }
  ],
  "total": 1
}
```

## Tool Endpoints

### POST /tools/run

Execute a tool.

**Request:**
```json
{
  "tool": "calculator",
  "parameters": {
    "expression": "2 + 2"
  },
  "timeout_ms": 5000
}
```

**Response:**
```json
{
  "success": true,
  "result": 4,
  "duration_ms": 12.5
}
```

## Workflow Endpoints

### GET /workflows

List all workflow definitions.

### POST /workflows

Create a new workflow definition.

**Request:**
```json
{
  "name": "My Workflow",
  "description": "Does things",
  "steps": [
    {
      "id": "step1",
      "type": "tool",
      "name": "Do Something",
      "config": {"tool": "echo", "input": "hello"},
      "depends_on": []
    }
  ]
}
```

### POST /{workflow_id}/run

Execute a workflow.

**Request:**
```json
{
  "inputs": {"key": "value"},
  "idempotency_key": "optional-unique-key"
}
```

### GET /{workflow_id}/graph

Get the workflow graph structure.

### GET /runs

List workflow runs.

### GET /runs/{run_id}

Get workflow run details.

### GET /runs/{run_id}/events

Get events for a workflow run.

## Error Responses

All errors follow a consistent format:

```json
{
  "detail": "Error description",
  "type": "error_type"
}
```

| Status | Meaning |
|---|---|
| 400 | Bad request (invalid input) |
| 401 | Unauthorized (missing/invalid API key) |
| 404 | Not found |
| 409 | Conflict (version conflict, duplicate) |
| 422 | Validation error |
| 429 | Rate limit exceeded |
| 500 | Internal server error |
