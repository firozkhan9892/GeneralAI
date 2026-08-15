# Continuous Integration

GeneralAI uses a single GitHub Actions workflow, `.github/workflows/ci.yml`,
to enforce production-grade quality gates automatically.

## Triggers

| Event | Scope |
|---|---|
| `push` | any push to `main` |
| `pull_request` | any PR targeting `main` |

The workflow runs with `permissions: contents: read` (least privilege) and
uses no secrets, API keys, or tokens.

## Jobs

The pipeline contains three isolated jobs that run in parallel on a clean
`ubuntu-latest` runner using Python **3.10** (the project's declared minimum,
see `requires-python = ">=3.10"` in `pyproject.toml`). All jobs use pip
dependency caching.

### `test` — Quality Gates

1. Check out the repository.
2. Set up Python 3.10 (pip cache keyed on `requirements.txt`).
3. Install the package in editable mode with development extras via the
   documented path: `pip install -r requirements.txt` (resolves to
   `-e .[dev]`).
4. Run the quality gates:

   ```bash
   python -m pytest -q
   python -m mypy . --no-error-summary
   python -m ruff check .
   python -m ruff format --check .
   ```

No optional dependencies (`faiss`, `chromadb`, `pypdf`, `beautifulsoup4`,
`sentence-transformers`) are installed, matching the documented runtime
setup.

### `package` — Packaging Verification

1. Check out the repository and set up Python 3.10.
2. Install only the build frontend (`pip install build`).
3. Build the sdist and wheel with `python -m build`.
4. Verify the artifacts and metadata:
   - exactly one `.whl` and one `.tar.gz` are produced,
   - wheel `METADATA` declares `Name: generalai`, `Version: 0.1.0`, and a
     `numpy` runtime dependency,
   - the sdist contains `generalai-0.1.0/app/__init__.py`.
5. Install the generated wheel into a clean virtual environment.
6. From a directory **outside** the checkout (`/tmp`) confirm the installed
   package imports correctly:

   ```python
   import app                      # resolves to the installed wheel
   from app.server.app import create_app; create_app()
   ```

The build artifacts are uploaded as the `generalai-dist` workflow artifact
for inspection.

### `optional-imports` — Imports Without Optional Dependencies

1. Check out the repository and set up Python 3.10.
2. Install **only** the base package (`pip install .`, no extras) into a
   clean virtual environment.
3. Assert that `faiss`, `chromadb`, `pypdf`, `bs4`, and
   `sentence_transformers` are **not** importable.
4. From a directory **outside** the checkout confirm the base install still
   imports and `create_app()` starts.

This guarantees the platform does not hard-depend on optional feature
packages.

## Failure Behaviour

Every gate is a hard step: any failure in `pytest`, `mypy`, `ruff check`,
`ruff format --check`, or packaging verification fails the whole workflow.
There are no mock or forced-pass conditions.

## Reproducing CI Locally

```bash
# 1. Install the package (editable) with dev extras — the documented path
pip install -r requirements.txt

# 2. Quality gates
python -m pytest -q
python -m mypy . --no-error-summary
python -m ruff check .
python -m ruff format --check .

# 3. Packaging verification
pip install build
python -m build

# 4. Optional-dependency isolation (base install only)
python -m venv /tmp/generalai-base
/tmp/generalai-base/bin/pip install .
cd /tmp
/tmp/generalai-base/bin/python -c "import app; print(app.__version__)"
/tmp/generalai-base/bin/python -c "from app.server.app import create_app; create_app()"
```

If any step fails locally, it will also fail in CI, so fix it before pushing.
