# Troubleshooting Guide

## Common Issues

### Installation Issues

| Symptom | Cause | Solution |
|---|---|---|
| `ModuleNotFoundError: No module named 'app'` | Wrong working directory | Run from project root |
| `Permission denied` on data/logs | Missing directories | `mkdir -p data logs` |
| `pip install` fails | Python version | Use Python 3.10+ |
| Import errors with optional deps | Missing packages | `pip install pypdf beautifulsoup4` |

### Runtime Issues

| Symptom | Cause | Solution |
|---|---|---|
| `RegistrationError: Type already registered` | Duplicate bootstrap call | Ensure idempotency guards |
| `CircularDependencyError` | Circular DI references | Break the cycle with lazy resolution |
| `TypeNotRegisteredError` | Missing bootstrap call | Register the type before resolving |
| `TimeoutError` on tool execution | Tool hung | Check timeout configuration |

### Server Issues

| Symptom | Cause | Solution |
|---|---|---|
| Port 8000 already in use | Another process | `lsof -i :8000` then kill, or change port |
| 401 Unauthorized | Missing/invalid API key | Set `X-API-Key` header |
| 429 Too Many Requests | Rate limit exceeded | Wait or increase limit |
| 500 Internal Server Error | Unhandled exception | Check logs |

### Knowledge/RAG Issues

| Symptom | Cause | Solution |
|---|---|---|
| Empty search results | No documents indexed | Ingest documents first |
| Wrong embedding dimensions | Model mismatch | Use same model for index and search |
| PDF ingestion fails | Missing pypdf | `pip install pypdf` |
| HTML ingestion fails | Missing beautifulsoup4 | `pip install beautifulsoup4` |
| Method routing returns wrong mock | Multiple methods on same endpoint | Use unique endpoints per method |

### Workflow Issues

| Symptom | Cause | Solution |
|---|---|---|
| Workflow not found | Not registered | Call `service.register(definition)` |
| Step fails | Invalid config | Check step type and config |
| Schedule not firing | Scheduler not started | Call `service.startup()` |
| Version conflict | Concurrent publish | Use idempotency keys |

## Debug Mode

Enable debug logging for detailed diagnostics:

```bash
python main.py --debug --log-level DEBUG
```

## Log Locations

| Log | Location |
|---|---|
| Application log | `logs/generalai.log` |
| Test output | `pytest_output.txt` |
| Audit report | `tests/audit_report.txt` |

## Health Checks

```bash
# Server health
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "version": "0.1.0", ...}
```

## Getting Help

1. Check logs: `tail -f logs/generalai.log`
2. Run tests: `python -m pytest tests/ -q --tb=short`
3. Check types: `python -m mypy app/ --ignore-missing-imports`
4. Check lint: `python -m ruff check app/ tests/`
