# Workflow Engine

## Overview

The workflow engine provides DAG-based automation with scheduling, persistence, and version control. Workflows are composed of steps that execute in topological order based on their dependencies.

## Core Concepts

| Concept | Description |
|---|---|
| **WorkflowDefinition** | Immutable workflow template with steps and settings |
| **WorkflowRun** | A single execution instance of a workflow |
| **WorkflowStep** | An individual unit of work within a workflow |
| **ScheduleSpec** | Cron-like schedule for recurring workflow execution |
| **WorkflowSnapshot** | Immutable copy of definition state at run time |

## Architecture

```mermaid
graph TB
    subgraph "Registries"
        WReg[WorkflowRegistry]
        RReg[WorkflowRunRegistry]
    end

    subgraph "Stores"
        WStore[WorkflowStore]
        RStore[WorkflowRunStore]
        SStore[ScheduleStore]
        EStore[EventStore]
    end

    subgraph "Execution"
        Executor[WorkflowExecutor]
        Scheduler[WorkflowScheduler]
        Validator[WorkflowValidator]
        Graph[WorkflowGraph]
    end

    subgraph "Service"
        Service[WorkflowService]
    end

    Service --> WReg
    Service --> RReg
    Service --> Executor
    Service --> Scheduler
    Executor --> Graph
    Executor --> Validator
    Scheduler --> SStore
    WReg --> WStore
    RReg --> RStore
    Executor --> EStore
```

## Step Types

| Type | Description | Config |
|---|---|---|
| `tool` | Execute a registered tool | `tool`, `parameters` |
| `workflow` | Run a nested workflow | `workflow_id`, `inputs` |
| `approval` | Require human approval | `approvers`, `timeout` |
| `condition` | Conditional branching | `condition`, `then`, `else` |
| `parallel` | Execute steps in parallel | `steps` |

## Creating a Workflow

```python
from app.automation.models import (
    WorkflowDefinition,
    WorkflowStep,
    WorkflowStatus,
)

definition = WorkflowDefinition(
    id="data-pipeline",
    name="Data Processing Pipeline",
    description="Process and analyze data",
    steps=[
        WorkflowStep(
            id="extract",
            type="tool",
            name="Extract Data",
            config={"tool": "file_reader", "path": "/data/input.csv"},
            depends_on=[],
        ),
        WorkflowStep(
            id="transform",
            type="tool",
            name="Transform Data",
            config={"tool": "transform", "operation": "normalize"},
            depends_on=["extract"],
        ),
        WorkflowStep(
            id="load",
            type="tool",
            name="Load Results",
            config={"tool": "file_writer", "path": "/data/output.json"},
            depends_on=["transform"],
        ),
    ],
    status=WorkflowStatus.ACTIVE,
)
```

## Executing a Workflow

```python
from app.automation.workflow import WorkflowService

service = WorkflowService()
service.register(definition)

# Run synchronously
run = await service.run("data-pipeline", inputs={"source": "production"})

# Run with idempotency key (prevents duplicate runs)
run = await service.run(
    "data-pipeline",
    inputs={"source": "production"},
    idempotency_key="run-2026-01-01",
)
```

## Scheduling

```python
from app.automation.models import ScheduleSpec, ScheduleTriggerType

schedule = ScheduleSpec(
    workflow_id="data-pipeline",
    trigger_type=ScheduleTriggerType.CRON,
    cron_expression="0 6 * * *",  # Daily at 6 AM
    inputs={"source": "production"},
    enabled=True,
)

service.create_schedule(schedule)
```

## Recovery

The workflow engine supports automatic recovery:

1. **On startup**: Restores persisted definitions and runs
2. **Pending runs**: Resumes runs that were interrupted
3. **Scheduler**: Recomputes timer states after restart

```python
# Startup automatically handles recovery
await workflow_service.startup()

# Shutdown drains in-flight runs
await workflow_service.shutdown()
```

## Error Handling

| Error | HTTP | Description |
|---|---|---|
| `WorkflowNotFoundError` | 404 | Workflow ID not found |
| `WorkflowValidationError` | 422 | Invalid step configuration |
| `WorkflowVersionError` | 409 | Version conflict |
| `WorkflowApprovalError` | 409 | Approval required/rejected |
| `WorkflowSchedulerError` | 400 | Invalid schedule |
| `WorkflowConcurrencyError` | 409 | Concurrent modification |
