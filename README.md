# GeneralAI

Autonomous AI Platform — production-grade foundation.

## Architecture

```
GeneralAI/
├── app/
│   ├── brain/        # Cognitive core (placeholder)
│   ├── memory/       # Storage & retrieval (placeholder)
│   ├── planner/      # Task decomposition (placeholder)
│   ├── agents/       # Multi-agent orchestration (placeholder)
│   ├── tools/        # Tool-calling interface (placeholder)
│   ├── automation/   # Workflow engine (placeholder)
│   ├── config/       # Settings, defaults, env vars
│   └── utils/        # Logging, helpers
├── tests/
├── docs/
├── logs/
├── data/
├── models/
├── main.py
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

CLI flags:
- `--debug` — enable debug logging
- `--env <name>` — development / staging / production
- `--log-level <level>` — DEBUG / INFO / WARNING / ERROR / CRITICAL
- `--version` — print version

## Configuration

Settings are loaded from (in order of precedence):
1. CLI arguments
2. Environment variables prefixed with `GENERAL_AI_`
3. `.env` file in the project root
4. Defaults defined in `app/config/defaults.py`

## Project Status

Foundation phase — only the scaffolding is in place.
