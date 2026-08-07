# Installation Guide

## Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional) Docker for containerized deployment

## Standard Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/generalai.git
cd generalai
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
python -c "import app; print('GeneralAI installed successfully')"
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

## Docker Installation

```bash
# Build image
docker build -t generalai:latest .

# Run container
docker run -p 8000:8000 \
  -e GENERAL_AI_API_KEY=your-secret-key \
  -v generalai-data:/app/data \
  -v generalai-logs:/app/logs \
  generalai:latest
```

## Development Installation

```bash
# Install with development tools
pip install -r requirements.txt
pip install pytest pytest-asyncio pytest-cov mypy ruff

# Run quality gates
python -m pytest tests/ -q
python -m ruff check app/ tests/
python -m ruff format --check app/ tests/
```

## Troubleshooting

### Common Issues

| Issue | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'app'` | Run from project root directory |
| `Permission denied` on data/logs directories | Create directories: `mkdir -p data logs` |
| Port 8000 already in use | Change port: `--port 8080` |
| Import errors with optional deps | Install optional dependencies as needed |
