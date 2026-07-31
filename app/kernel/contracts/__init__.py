"""Inter-engine communication contracts for the Cognitive Kernel.

Every message between engines is wrapped in a ``MessageEnvelope`` carrying
routing, tracing, and versioning metadata. The domain-specific payload is
one of the typed request/response pairs defined in this package.

Usage
-----
>>> from app.kernel.contracts import (
...     MessageEnvelope, EngineType,
...     PerceptionToIntentRequest,
... )
>>> req = PerceptionToIntentRequest(...)
>>> envelope = MessageEnvelope.from_payload(
...     payload=req,
...     source_engine=EngineType.PERCEPTION,
...     target_engine=EngineType.INTENT,
... )
"""

from app.kernel.contracts.base import (
    ContractRequest,
    ContractResponse,
    ContractResult,
    EngineType,
    ErrorInfo,
    MessageEnvelope,
    ResultStatus,
)
from app.kernel.contracts.capability import (
    CapabilityToPolicyRequest,
    CapabilityToPolicyResponse,
)
from app.kernel.contracts.decision import (
    DecisionToCapabilityRequest,
    DecisionToCapabilityResponse,
)
from app.kernel.contracts.events import (
    EVENT_CAPABILITY_RESOLVED,
    EVENT_CAPABILITY_UNAVAILABLE,
    EVENT_DECISION_MADE,
    EVENT_EXECUTION_COMPLETED,
    EVENT_EXECUTION_FAILED,
    EVENT_EXECUTION_STARTED,
    EVENT_EXPERIENCE_RECORDED,
    EVENT_GOAL_COMPLETED,
    EVENT_GOAL_CREATED,
    EVENT_GOAL_DECOMPOSED,
    EVENT_INTENT_IDENTIFIED,
    EVENT_PERCEPTION_COMPLETED,
    EVENT_PERCEPTION_STARTED,
    EVENT_PLAN_CREATED,
    EVENT_PLAN_REVISED,
    EVENT_POLICY_CONFIRMATION_REQUIRED,
    EVENT_POLICY_DENIED,
    EVENT_POLICY_EVALUATED,
    EVENT_REASONING_COMPLETED,
    EVENT_REASONING_STARTED,
    EVENT_REASONING_STEP,
    EVENT_REFLECTION_COMPLETED,
    EVENT_REFLECTION_REFINEMENT,
    EVENT_REFLECTION_STARTED,
    EVENT_STATE_CHANGED,
    EVENT_STATE_ERROR,
    EventSeverity,
    EVENT_SEVERITY,
    PipelineEvent,
)
from app.kernel.contracts.experience import (
    ExperienceToMemoryRequest,
    ExperienceToMemoryResponse,
)
from app.kernel.contracts.goals import (
    GoalToPlannerRequest,
    GoalToPlannerResponse,
)
from app.kernel.contracts.intent import (
    IntentToGoalRequest,
    IntentToGoalResponse,
)
from app.kernel.contracts.memory import (
    MemoryToResponseRequest,
    MemoryToResponseResponse,
)
from app.kernel.contracts.perception import (
    PerceptionToIntentRequest,
    PerceptionToIntentResponse,
)
from app.kernel.contracts.planning import (
    PlannerToReasoningRequest,
    PlannerToReasoningResponse,
)
from app.kernel.contracts.policy import (
    PolicyToTaskRequest,
    PolicyToTaskResponse,
)
from app.kernel.contracts.reasoning import (
    ReasoningToDecisionRequest,
    ReasoningToDecisionResponse,
)
from app.kernel.contracts.reflection import (
    ReflectionToExperienceRequest,
    ReflectionToExperienceResponse,
)
from app.kernel.contracts.task import (
    TaskToToolRequest,
    TaskToToolResponse,
)
from app.kernel.contracts.tool import (
    ToolToReflectionRequest,
    ToolToReflectionResponse,
)

__all__ = [
    # Base primitives
    "EngineType",
    "ResultStatus",
    "ErrorInfo",
    "ContractResult",
    "MessageEnvelope",
    "ContractRequest",
    "ContractResponse",
    # Events
    "PipelineEvent",
    "EventSeverity",
    "EVENT_SEVERITY",
    # Backward-compat string constants
    "EVENT_PERCEPTION_STARTED",
    "EVENT_PERCEPTION_COMPLETED",
    "EVENT_INTENT_IDENTIFIED",
    "EVENT_GOAL_CREATED",
    "EVENT_GOAL_COMPLETED",
    "EVENT_GOAL_DECOMPOSED",
    "EVENT_PLAN_CREATED",
    "EVENT_PLAN_REVISED",
    "EVENT_REASONING_STARTED",
    "EVENT_REASONING_STEP",
    "EVENT_REASONING_COMPLETED",
    "EVENT_DECISION_MADE",
    "EVENT_CAPABILITY_RESOLVED",
    "EVENT_CAPABILITY_UNAVAILABLE",
    "EVENT_POLICY_EVALUATED",
    "EVENT_POLICY_DENIED",
    "EVENT_POLICY_CONFIRMATION_REQUIRED",
    "EVENT_EXECUTION_STARTED",
    "EVENT_EXECUTION_COMPLETED",
    "EVENT_EXECUTION_FAILED",
    "EVENT_REFLECTION_STARTED",
    "EVENT_REFLECTION_COMPLETED",
    "EVENT_REFLECTION_REFINEMENT",
    "EVENT_EXPERIENCE_RECORDED",
    "EVENT_STATE_CHANGED",
    "EVENT_STATE_ERROR",
    # Perception <-> Intent
    "PerceptionToIntentRequest",
    "PerceptionToIntentResponse",
    # Intent <-> Goal
    "IntentToGoalRequest",
    "IntentToGoalResponse",
    # Goal <-> Planner
    "GoalToPlannerRequest",
    "GoalToPlannerResponse",
    # Planner <-> Reasoning
    "PlannerToReasoningRequest",
    "PlannerToReasoningResponse",
    # Reasoning <-> Decision
    "ReasoningToDecisionRequest",
    "ReasoningToDecisionResponse",
    # Decision <-> Capability
    "DecisionToCapabilityRequest",
    "DecisionToCapabilityResponse",
    # Capability <-> Policy
    "CapabilityToPolicyRequest",
    "CapabilityToPolicyResponse",
    # Policy <-> Task
    "PolicyToTaskRequest",
    "PolicyToTaskResponse",
    # Task <-> Tool
    "TaskToToolRequest",
    "TaskToToolResponse",
    # Tool <-> Reflection
    "ToolToReflectionRequest",
    "ToolToReflectionResponse",
    # Reflection <-> Experience
    "ReflectionToExperienceRequest",
    "ReflectionToExperienceResponse",
    # Experience <-> Memory
    "ExperienceToMemoryRequest",
    "ExperienceToMemoryResponse",
    # Memory <-> Response
    "MemoryToResponseRequest",
    "MemoryToResponseResponse",
]
