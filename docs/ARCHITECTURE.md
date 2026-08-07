# Architecture

## System Overview

GeneralAI is a modular, plugin-extensible AI platform built around a cognitive kernel, multi-LLM routing, and enterprise RAG capabilities.

## High-Level Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        CLI[CLI Interface]
        REST[REST API]
        WS[WebSocket]
    end

    subgraph "Server Layer"
        FastAPI[FastAPI Server]
        Auth[API Key Auth]
        RateLimit[Rate Limiter]
        MetricsCollector[Metrics]
    end

    subgraph "Core Framework"
        DI[DI Container]
        Events[Event Bus]
        Lifecycle[Lifecycle Manager]
        Registry[Registries]
    end

    subgraph "Cognitive Kernel"
        Orchestrator[Cognitive Orchestrator]
        Pipeline[18-Stage Pipeline]
    end

    subgraph "Intelligence Layer"
        LLMRouter[LLM Router]
        Agents[Agent Manager]
        Memory[Memory Engine]
    end

    subgraph "Automation Layer"
        Workflow[Workflow Service]
        Scheduler[Workflow Scheduler]
        Executor[Workflow Executor]
    end

    subgraph "Extension Layer"
        Plugins[Plugin Manager]
        Tools[Tool Executor]
        Knowledge[RAG Pipeline]
    end

    CLI --> FastAPI
    REST --> FastAPI
    WS --> FastAPI
    FastAPI --> Auth --> RateLimit --> MetricsCollector
    FastAPI --> Orchestrator
    FastAPI --> Workflow
    FastAPI --> Knowledge
    FastAPI --> Tools
    Orchestrator --> Pipeline
    Orchestrator --> LLMRouter
    Orchestrator --> Agents
    Orchestrator --> Memory
    Pipeline --> Events
    LLMRouter --> Agents
    Workflow --> Scheduler --> Executor
    Plugins --> Tools
    Plugins --> Knowledge
    DI --> Registry
    Lifecycle --> DI
```

## Cognitive Pipeline

The cognitive kernel processes requests through 18 stages:

```mermaid
graph LR
    Input[User Input] --> Perception[Perception]
    Perception --> Intent[Intent]
    Intent --> Goals[Goals]
    Goals --> Planning[Planning]
    Planning --> Reasoning[Reasoning]
    Reasoning --> Decision[Decision]
    Decision --> Capability[Capability]
    Capability --> Policy[Policy]
    Policy --> Skills[Skill Selection]
    Skills --> ToolResolve[Tool Resolution]
    ToolResolve --> ToolExec[Tool Execution]
    ToolExec --> Reflection[Reflection]
    Reflection --> Experience[Experience]
    Experience --> Context[Context]
    Context --> State[State]
    State --> Model[Model Router]
    Model --> Response[Response Builder]
    Response --> Output[Output]
```

## Dependency Injection Flow

```mermaid
sequenceDiagram
    participant Main as main.py
    participant Container as DependencyContainer
    participant Bootstrap as Bootstrap
    participant Service as Service

    Main->>Container: create_app()
    Container->>Bootstrap: register_kernel_components()
    Container->>Bootstrap: register_llm_components()
    Container->>Bootstrap: register_agent_components()
    Container->>Bootstrap: register_automation_components()
    Container->>Bootstrap: register_tool_components()
    Container->>Bootstrap: register_plugin_components()
    Note over Container: All singletons registered
    Container->>Service: resolve(T)
    Service-->>Container: Instance
    Container-->>Main: FastAPI app
```

## Event Bus Architecture

```mermaid
graph LR
    subgraph Publishers
        Kernel[Kernel Events]
        Agent[Agent Events]
        Workflow[Workflow Events]
        Plugin[Plugin Events]
    end

    subgraph EventBus
        Pub[Publish]
        Sub[Subscribe]
        Dispatch[Dispatch]
    end

    subgraph Subscribers
        Analytics[Analytics]
        Logging[Logging]
        Metrics[Metrics]
        Notifications[Notifications]
    end

    Kernel --> Pub
    Agent --> Pub
    Workflow --> Pub
    Plugin --> Pub
    Pub --> Sub --> Dispatch
    Dispatch --> Analytics
    Dispatch --> Logging
    Dispatch --> Metrics
    Dispatch --> Notifications
```

## Module Dependencies

```mermaid
graph TD
    Config[config] --> Core[core]
    Core --> Server[server]
    Core --> Kernel[kernel]
    Core --> LLM[llm]
    Core --> Agents[agents]
    Core --> Automation[automation]
    Core --> Plugins[plugins]
    Core --> Tools[tools]
    Core --> Knowledge[knowledge]
    Kernel --> LLM
    Kernel --> Agents
    Kernel --> Memory[memory]
    Agents --> Tools
    Automation --> Tools
    Automation --> Agents
    Plugins --> Tools
    Plugins --> Knowledge
    Server --> Kernel
    Server --> Automation
    Server --> Tools
```

## Threading Model

```mermaid
graph TB
    subgraph "Main Thread"
        App[Application]
        LifecycleMgr[Lifecycle Manager]
    end

    subgraph "Async Event Loop"
        AsyncOps[Async Operations]
        EventBusDispatch[Event Bus Dispatch]
    end

    subgraph "Thread Pool"
        SyncOps[Synchronous Operations]
        ToolExec[Tool Execution]
    end

    subgraph "RLock Protected"
        Container[DI Container]
        Registries[Registries]
        Stores[Stores]
        Caches[Caches]
    end

    App --> AsyncOps
    App --> SyncOps
    AsyncOps --> EventBusDispatch
    SyncOps --> Container
    Container --> Registries
    Container --> Stores
    Container --> Caches
```

## Data Flow

```mermaid
graph LR
    subgraph Input
        User[User Prompt]
        Files[Document Upload]
        API[API Call]
    end

    subgraph Processing
        Parse[Parse & Normalize]
        Route[LLM Route]
        Retrieve[Knowledge Retrieve]
        Execute[Tool Execute]
        Reason[Reasoning Chain]
    end

    subgraph Storage
        VStore[(Vector Store)]
        BM25[(BM25 Index)]
        Cache[(Embedding Cache)]
        Session[(Session Store)]
        WorkflowDB[(Workflow Store)]
    end

    subgraph Output
        Response[Text Response]
        Citations[Citations]
        Sources[Source References]
    end

    User --> Parse --> Route
    Files --> Parse
    API --> Parse
    Route --> Retrieve --> VStore
    Retrieve --> BM25
    Retrieve --> Cache
    Route --> Execute
    Execute --> Reason
    Reason --> Response
    Retrieve --> Citations
    Retrieve --> Sources
    Route --> Session
    Execute --> WorkflowDB
```
