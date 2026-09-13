# Installation Guide

## Prerequisites

- Python 3.10 or higher
- pip package manager

## Standard Installation

### 1. Clone the Repository

```bash
git clone https://github.com/firozkhan9892/GeneralAI.git
cd GeneralAI
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import app; print('GeneralAI', app.__version__, 'installed successfully')"
python -m pytest tests/ -q --tb=no
```

## Optional Dependencies

Some features require additional packages:

```bash
# For PDF document ingestion
pip install pypdf

# For HTML document ingestion
pip install beautifulsoup4

# For FAISS vector store
pip install faiss-cpu

# For ChromaDB vector store
pip install chromadb

# For sentence-transformers embeddings
pip install sentence-transformers

# For numpy-based operations (required)
pip install numpy
```

Or install them all at once with the package extras:

```bash
pip install -e '.[all]'
```

## Run the Server

```bash
# Development server (auto-reload, permissive defaults)
uvicorn app.server.app:create_app --factory --reload --port 8000

# Health check
curl http://localhost:8000/health
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for production authentication, Docker,
and process-manager guidance.

## Development Installation

```bash
# Install the package (editable) with development tools
pip install -e '.[dev]'

# Run quality gates (identical to CI)
python -m pytest -q
python -m mypy . --no-error-summary
python -m ruff check .
python -m ruff format --check .
```

## Troubleshooting

### Common Issues

| Issue | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'app'` | Run from project root directory |
| `Permission denied` on data/logs directories | Create directories: `mkdir -p data logs` |
| Port 8000 already in use | Change port: `uvicorn ... --port 8080` |
| Import errors with optional deps | Install optional dependencies as needed |