"""Agent Runtime — the execution brain of GeneralAI.

The AgentRuntime integrates every completed cognitive engine (Perception,
Intent, Goal, Planning, Reasoning, Decision, Policy, Memory, Experience,
Reflection, Response) with the Phase-5 ToolExecutor into a deterministic,
fully offline agent that perceives, reasons, plans, acts, and responds.
"""

from __future__ import annotations

from app.kernel.agent.bootstrap import register_agent_components
from app.kernel.agent.loop import AgentLoop
from app.kernel.agent.models import (
    AgentRequest,
    AgentResponse,
    AgentRunConfig,
    AgentRunSummary,
    AgentStatus,
    AgentStep,
    AgentStepStatus,
)
from app.kernel.agent.policies import FallbackPolicy, RetryPolicy
from app.kernel.agent.runtime import AgentRuntime

__all__ = [
    "AgentLoop",
    "AgentRequest",
    "AgentResponse",
    "AgentRunConfig",
    "AgentRunSummary",
    "AgentRuntime",
    "AgentStatus",
    "AgentStep",
    "AgentStepStatus",
    "FallbackPolicy",
    "RetryPolicy",
    "register_agent_components",
]
