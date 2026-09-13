# Deployment Guide

## Deployment Options

### 1. Local Development

```bash
# Run with auto-reload
uvicorn app.server.app:create_app --factory --reload --port 8000

# Run with debug
python main.py --debug --log-level DEBUG
```

> **Note:** `create_app()` with no arguments builds a permissive development
> configuration (no API key, rate limiting enabled at 60 req/min). This is
> fine locally; see [Enabling API Authentication](#enabling-api-authentication)
> before exposing the service on a network.

### 2. Production (Linux)

```bash
# Create dedicated user
sudo useradd -r -s /bin/false generalai

# Create directories
sudo mkdir -p /opt/generalai/{data,logs}
sudo chown -R generalai:generalai /opt/generalai

# Create virtual environment
sudo -u generalai python3 -m venv /opt/generalai/venv
sudo -u generalai /opt/generalai/venv/bin/pip install -r requirements.txt
```

Enable authentication and moderate the CORS policy by wrapping
`create_app` in a small module with an explicit `ServerSettings`:

```bash
# /opt/generalai/serve.py
sudo tee /opt/generalai/serve.py << 'EOF'
from app.server.app import create_app
from app.server.config import ServerSettings

app = create_app(
    settings=ServerSettings(
        api_key="your-secret-key",
        cors_origins=("https://app.example.com",),
    ),
)
EOF
```

```bash
# Create systemd service
sudo tee /etc/systemd/system/generalai.service << EOF
[Unit]
Description=GeneralAI Platform
After=network.target

[Service]
Type=simple
User=generalai
Group=generalai
WorkingDirectory=/opt/generalai
Environment=GENERAL_AI_ENVIRONMENT=production
ExecStart=/opt/generalai/venv/bin/uvicorn serve:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable generalai
sudo systemctl start generalai
```

### 3. Production (Windows)

```bash
# Run as a scheduled task or service
# Using NSSM (Non-Sucking Service Manager):
```

First create a wrapper module that applies your `ServerSettings` (see
[Enabling API Authentication](#enabling-api-authentication)):

```powershell
# serve.py in the project root:
#   from app.server.app import create_app
#   from app.server.config import ServerSettings
#   app = create_app(settings=ServerSettings(api_key="your-secret-key"))

nssm install GeneralAI "C:\path\to\venv\Scripts\uvicorn.exe" "serve:app --host 0.0.0.0 --port 8000"
nssm set GeneralAI AppDirectory "C:\path\to\generalai"
nssm set GeneralAI DisplayName "GeneralAI Platform"
nssm set GeneralAI Description "Autonomous AI Platform"
nssm set GeneralAI Start SERVICE_AUTO_START
nssm start GeneralAI
```

### 4. Container Deployment (optional)

The distribution ships **without** a `Dockerfile` or
`docker-compose.yml` — the repository has no built-in container build. If you
want to containerize GeneralAI, add a `Dockerfile` to your fork:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p data logs

# Environment
ENV GENERAL_AI_ENVIRONMENT=production
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run
CMD ["uvicorn", "app.server.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run (from your fork)
docker build -t generalai:latest .
docker run -d \
  --name generalai \
  -p 8000:8000 \
  -e GENERAL_AI_ENVIRONMENT=production \
  -v generalai-data:/app/data \
  -v generalai-logs:/app/logs \
  generalai:latest

# View logs
docker logs -f generalai
```

> **Remember:** the server API key is set **in code** via `ServerSettings`,
> not through an environment variable. There is no `GENERAL_AI_API_KEY`
> environment variable in GeneralAI — setting one has no effect.

### 5. Railway / Render (PaaS)

```bash
# Set environment variables in dashboard:
# GENERAL_AI_ENVIRONMENT=production
# PORT=8000

# Start command:
uvicorn app.server.app:create_app --factory --host 0.0.0.0 --port $PORT
```

To require an API key on a PaaS, create the app with an explicit
`ServerSettings` in a small module (see below) and point your start command
at that module's `app` object.

## Enabling API Authentication

Pass an explicit `ServerSettings` when creating the application:

```python
from app.server.app import create_app
from app.server.config import ServerSettings

app = create_app(
    settings=ServerSettings(
        api_key="your-secret-key",
        rate_limit_per_minute=120,
        cors_origins=("https://app.example.com",),
    ),
)
```

When `api_key` is set, every non-public endpoint requires the `X-API-Key`
header. A `?api_key=` query-parameter fallback also works but is not
recommended in production (it can leak into logs and proxy metrics).

## Production Checklist

- [ ] Set a strong API key via `ServerSettings(api_key=...)`
- [ ] Enable rate limiting (default 60 req/min per identity)
- [ ] Set log level to `WARNING` or `ERROR`
- [ ] Configure CORS origins (empty tuple disables CORS)
- [ ] Set up log rotation (`logs/*.log` rotates at 10 MB, 5 backups)
- [ ] Configure health check monitoring (`GET /health`)
- [ ] Set up backup for `data/` directory
- [ ] Configure SSL/TLS termination
- [ ] Set resource limits (memory, CPU)

## Environment Variables

See [CONFIGURATION.md](CONFIGURATION.md) for the complete list of environment
variables. Server `ServerSettings` fields are configured programmatically,
not via the environment.

## Monitoring

```bash
# Health check
curl http://localhost:8000/health

# Metrics
curl http://localhost:8000/metrics

# Logs
tail -f logs/generalai.log
```

## Scaling

### Horizontal Scaling

GeneralAI is stateless (all state is in-memory or persistent stores). For horizontal scaling:

1. Use external persistent stores (Redis, PostgreSQL)
2. Add load balancer
3. Run multiple instances
4. Share data volume (NFS, S3)

### Vertical Scaling

```bash
# Increase workers
uvicorn app.server.app:create_app --factory --workers 8

# Increase connections
uvicorn app.server.app:create_app --factory --limit-concurrency 1000
```