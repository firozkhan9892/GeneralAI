"""Comprehensive tests for all Cognitive Kernel domain models.

Covers validation, serialization, equality, immutability,
JSON schema, and enum correctness for every model.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

# ── Core models ──────────────────────────────────────────────────────────────
from app.kernel.models.core import Conversation, InputMessage, Session, SessionStatus
from app.kernel.models.models import (
    Message,
    ModelCapability,
    ModelRequest,
    ModelResponse,
    ResponseFormat,
    RoleType,
    ToolDefinition,
)

# ── Perception ───────────────────────────────────────────────────────────────
from app.kernel.perception.models import (
    Entity,
    ModalityType,
    Percept,
    QualityScore,
    RawMessage,
)

# ── Intent ───────────────────────────────────────────────────────────────────
from app.kernel.intent.models import (
    ClarificationRequest,
    Intent,
    IntentClassification,
    IntentConfidence,
    IntentType,
)

# ── Goals ────────────────────────────────────────────────────────────────────
from app.kernel.goals.models import (
    Goal,
    GoalHierarchy,
    GoalPriority,
    GoalStatus,
    GoalType,
)

# ── Planning ─────────────────────────────────────────────────────────────────
from app.kernel.planning.models import (
    DependencyGraph,
    Plan,
    PlanningStrategy,
    SkillStep,
)

# ── Reasoning ────────────────────────────────────────────────────────────────
from app.kernel.reasoning.models import (
    ReasoningRequest,
    ReasoningStep,
    ReasoningStrategy,
    ReasoningTrace,
    StepType,
)

# ── Decision ─────────────────────────────────────────────────────────────────
from app.kernel.decision.models import (
    ActionCandidate,
    Decision,
    DecisionReason,
    DecisionScore,
)

# ── Capability ───────────────────────────────────────────────────────────────
from app.kernel.capability.models import (
    Capability,
    CapabilityProvider,
    CapabilityRequirement,
    CapabilityResult,
    ProviderType,
    ResourceEstimate,
)

# ── Policy ───────────────────────────────────────────────────────────────────
from app.kernel.policy.models import (
    AppliedPolicy,
    PolicyAction,
    PolicyDecision,
    PolicyRule,
    SandboxConfig,
    VerdictType,
)

# ── Skills ───────────────────────────────────────────────────────────────────
from app.kernel.skills.models import (
    Skill,
    SkillDescriptor,
    SkillRequirement,
    SkillResult,
    ToolRequirement,
)

# ── Tools ────────────────────────────────────────────────────────────────────
from app.kernel.tools.models import (
    ExecutionType,
    RateLimit,
    ToolBinding,
    ToolDescriptor,
    ToolRequest,
    ToolResult,
)

# ── Tasks ────────────────────────────────────────────────────────────────────
from app.kernel.tasks.models import Task, TaskResult, TaskStatus

# ── Reflection ───────────────────────────────────────────────────────────────
from app.kernel.reflection.models import (
    ErrorDetail,
    ErrorType,
    Refinement,
    ReflectionReport,
    ReflectionRequest,
    ReflectionScore,
)

# ── Experience ───────────────────────────────────────────────────────────────
from app.kernel.experience.models import (
    DecisionSummary,
    Experience,
    ExperienceQuery,
    Insight,
    LessonCategory,
    LessonLearned,
)

# ── Pipeline ─────────────────────────────────────────────────────────────────
from app.kernel.pipeline.models import (
    ErrorPolicy,
    PipelineContext,
    PipelineDefinition,
    PipelineMetadata,
    PipelineStep,
)

# ── State ────────────────────────────────────────────────────────────────────
from app.kernel.state.models import CognitiveState, SessionState, Transition

# ── Context ──────────────────────────────────────────────────────────────────
from app.kernel.context.models import (
    CognitiveContext,
    ContextDelta,
    ContextSnapshot,
    ContextSource,
    TokenBudget,
)

# ── Response ─────────────────────────────────────────────────────────────────
from app.kernel.response.models import OutputMessage, StreamChunk

# =============================================================================
# Validation
# =============================================================================


class TestValidation:
    """Field constraints — ge, le, required, type enforcement."""

    @pytest.mark.parametrize(
        "cls,kwargs,expect_pass",
        [
            (Entity, {"type": "url", "value": "https://x.com"}, True),
            (Entity, {"type": "url", "value": "x", "confidence": 1.5}, False),
            (Entity, {"type": "url", "value": "x", "confidence": -0.1}, False),
            (QualityScore, {"overall": 0.5}, True),
            (QualityScore, {"overall": 1.5}, False),
            (IntentConfidence, {"primary": 0.8}, True),
            (IntentConfidence, {"primary": 1.5}, False),
            (IntentConfidence, {"primary": -0.1}, False),
            (SkillStep, {"order": 0, "skill_name": "test"}, True),
            (SkillStep, {"order": -1, "skill_name": "test"}, False),
            (DecisionScore, {"criterion_name": "c1", "score": 0.5}, True),
            (DecisionScore, {"criterion_name": "c1", "score": 1.5}, False),
            (
                DecisionScore,
                {"criterion_name": "c1", "score": 0.5, "weight": -1},
                False,
            ),
            (Task, {"action_type": "call", "action_name": "foo"}, True),
            (
                Task,
                {"action_type": "call", "action_name": "foo", "retry_count": -1},
                False,
            ),
            (ReflectionScore, {"dimension": "accuracy", "score": 0.5}, True),
            (ReflectionScore, {"dimension": "accuracy", "score": 1.5}, False),
            (ToolDescriptor, {"name": "t"}, True),
            (ToolDescriptor, {"name": "t", "timeout_s": 0}, False),
            (ExperienceQuery, {"limit": 10}, True),
            (ExperienceQuery, {"limit": 0}, False),
            (ExperienceQuery, {"limit": 101}, False),
            (ReasoningRequest, {"problem": "solve x"}, True),
            (ReasoningRequest, {"problem": "solve x", "max_steps": 101}, False),
            (ReasoningRequest, {"problem": "solve x", "max_steps": 0}, False),
            (Goal, {"description": "do something"}, True),
            (Goal, {"description": "do something", "progress": 1.5}, False),
            (Goal, {"description": "do something", "progress": -0.1}, False),
        ],
    )
    def test_field_constraints(
        self, cls: type, kwargs: dict[str, Any], expect_pass: bool
    ) -> None:
        if expect_pass:
            instance = cls(**kwargs)
            assert instance is not None
        else:
            with pytest.raises(ValidationError):
                cls(**kwargs)

    def test_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            Entity()  # type: ignore[call-arg]

    def test_intent_requires_confidence(self) -> None:
        with pytest.raises(ValidationError):
            Intent(primary=IntentType.ASK_QUESTION)  # type: ignore[call-arg]


# =============================================================================
# Serialization
# =============================================================================


class TestSerialization:
    """Round-trip dict / JSON serialization for every model type."""

    @pytest.mark.parametrize(
        "model",
        [
            InputMessage(content="hello"),
            Entity(type="email", value="a@b.com"),
            Percept(raw=RawMessage(content="hi"), modality=ModalityType.TEXT),
            Intent(
                primary=IntentType.ASK_QUESTION,
                confidence=IntentConfidence(primary=0.9),
            ),
            Goal(description="test"),
            Plan(goal_id="g1"),
            SkillStep(order=1, skill_name="search"),
            ReasoningRequest(problem="p"),
            ActionCandidate(action_type="call"),
            Decision(selected_action=ActionCandidate(action_type="call")),
            Capability(
                name="search",
                provider=CapabilityProvider(
                    name="web", provider_type=ProviderType.TOOL
                ),
            ),
            PolicyDecision(
                action=PolicyAction(action_type="call"), verdict=VerdictType.ALLOW
            ),
            Skill(name="s", descriptor=SkillDescriptor(name="s")),
            ToolRequest(tool_name="t"),
            Task(action_type="call", action_name="search"),
            ReflectionRequest(output="test"),
            Experience(session_id="s1"),
            CognitiveContext(),
            OutputMessage(content="ok"),
            StreamChunk(content="chunk"),
            SessionState(session_id="s1"),
            Transition(
                from_state=CognitiveState.IDLE, to_state=CognitiveState.PERCEIVING
            ),
            TokenBudget(),
            Message(role=RoleType.USER, content="hello"),
            ModelCapability(),
            PipelineStep(name="perception", order=0),
        ],
    )
    def test_dict_roundtrip(self, model: BaseModel) -> None:
        data = model.model_dump()
        restored = model.__class__.model_validate(data)
        assert restored == model
        assert restored.model_dump() == data

    @pytest.mark.parametrize(
        "model",
        [
            InputMessage(content="hello"),
            Entity(type="url", value="https://x.com"),
            IntentConfidence(primary=0.85),
        ],
    )
    def test_json_roundtrip(self, model: BaseModel) -> None:
        raw = model.model_dump_json()
        data = json.loads(raw)
        restored = model.__class__.model_validate(data)
        assert restored == model

    def test_enum_values_in_serialization(self) -> None:
        r = RawMessage(content="hi", modality=ModalityType.IMAGE)
        data = r.model_dump()
        assert data["modality"] == "image"
        assert isinstance(data["modality"], str)

    def test_datetime_serialization(self) -> None:
        now = datetime.utcnow()
        m = InputMessage(content="hi", timestamp=now)
        data = m.model_dump()
        assert "timestamp" in data
        restored = InputMessage.model_validate(data)
        assert restored.timestamp == now


# =============================================================================
# Equality
# =============================================================================


class TestEquality:
    """Value equality for frozen models."""

    def test_equal_models_are_equal(self) -> None:
        a = Entity(type="url", value="https://x.com")
        b = Entity(type="url", value="https://x.com")
        assert a == b
        assert hash(a) == hash(b)

    def test_different_models_are_not_equal(self) -> None:
        a = Entity(type="url", value="https://x.com")
        b = Entity(type="email", value="a@b.com")
        assert a != b

    def test_model_not_equal_to_dict(self) -> None:
        e = Entity(type="url", value="https://x.com")
        assert e != {
            "type": "url",
            "value": "https://x.com",
            "confidence": 1.0,
            "position": None,
        }

    def test_same_with_different_metadata(self) -> None:
        a = InputMessage(content="hi", metadata={"a": 1})
        b = InputMessage(content="hi", metadata={"a": 2})
        assert a != b


# =============================================================================
# Immutability
# =============================================================================


class TestImmutability:
    """Frozen models raise TypeError on attribute assignment."""

    @pytest.mark.parametrize(
        "model,attr,value",
        [
            (Entity(type="url", value="x"), "type", "email"),
            (Entity(type="url", value="x"), "value", "new"),
            (IntentConfidence(primary=0.5), "primary", 0.9),
            (DecisionScore(criterion_name="c", score=0.5), "score", 1.0),
            (SkillRequirement(skill_name="s"), "skill_name", "other"),
            (
                Task(action_type="call", action_name="search"),
                "status",
                TaskStatus.RUNNING,
            ),
            (ReflectionScore(dimension="acc", score=0.5), "score", 0.9),
            (RateLimit(), "max_calls_per_minute", 10),
            (PipelineStep(name="p", order=0), "order", 1),
            (
                Transition(
                    from_state=CognitiveState.IDLE, to_state=CognitiveState.PERCEIVING
                ),
                "reason",
                "new",
            ),
        ],
    )
    def test_frozen_raises(self, model: Any, attr: str, value: Any) -> None:
        with pytest.raises((TypeError, ValidationError)):
            setattr(model, attr, value)


class TestMutableModels:
    """SessionState and CognitiveContext are intentionally mutable."""

    def test_session_state_mutable(self) -> None:
        s = SessionState(session_id="s1")
        s.state = CognitiveState.PERCEIVING
        assert s.state == CognitiveState.PERCEIVING

    def test_cognitive_context_mutable(self) -> None:
        c = CognitiveContext(id="ctx1")
        c.version = 5
        assert c.version == 5


# =============================================================================
# JSON Schema
# =============================================================================


class TestJsonSchema:
    """Every model produces a valid JSON Schema."""

    @pytest.mark.parametrize(
        "cls",
        [
            InputMessage,
            Entity,
            Percept,
            Intent,
            Goal,
            GoalHierarchy,
            Plan,
            ReasoningRequest,
            ReasoningTrace,
            Decision,
            Capability,
            PolicyDecision,
            Skill,
            ToolRequest,
            ToolResult,
            Task,
            TaskResult,
            ReflectionReport,
            Experience,
            PipelineDefinition,
            PipelineContext,
            ModelRequest,
            ModelResponse,
            CognitiveContext,
            OutputMessage,
            Session,
            SessionState,
            Transition,
        ],
    )
    def test_json_schema_generates(self, cls: type[BaseModel]) -> None:
        schema = cls.model_json_schema()
        assert schema is not None
        assert "title" in schema or "$ref" in schema

    def test_schema_contains_fields(self) -> None:
        schema = Entity.model_json_schema()
        props = schema.get("properties", {})
        assert "type" in props
        assert "value" in props
        assert "confidence" in props

    def test_schema_contains_constraints(self) -> None:
        schema = Entity.model_json_schema()
        confidence = schema["properties"]["confidence"]
        assert confidence.get("minimum") == 0.0
        assert confidence.get("maximum") == 1.0


# =============================================================================
# Enum Correctness
# =============================================================================


class TestEnumCorrectness:
    """All enums have correct values and no magic strings."""

    @pytest.mark.parametrize(
        "enum_cls,expected",
        [
            (
                ModalityType,
                [
                    "text",
                    "image",
                    "audio",
                    "tool_result",
                    "system_event",
                    "multimodal",
                    "unknown",
                ],
            ),
            (
                IntentType,
                [
                    "ask_question",
                    "solve_problem",
                    "execute_task",
                    "plan_project",
                    "learn",
                    "create_content",
                    "explore",
                    "debug",
                    "meta",
                    "clarify",
                    "unknown",
                ],
            ),
            (
                GoalStatus,
                [
                    "proposed",
                    "active",
                    "blocked",
                    "completed",
                    "failed",
                    "abandoned",
                    "superseded",
                    "paused",
                ],
            ),
            (GoalPriority, [0, 25, 50, 75, 100]),
            (
                GoalType,
                [
                    "question",
                    "task",
                    "project",
                    "learning",
                    "exploration",
                    "debugging",
                    "system",
                ],
            ),
            (
                PlanningStrategy,
                ["top_down", "bottom_up", "means_end", "case_based", "reactive"],
            ),
            (
                ReasoningStrategy,
                [
                    "chain_of_thought",
                    "tree_of_thought",
                    "react",
                    "reflexion",
                    "straw_man",
                    "first_principles",
                    "analogical",
                    "decomposition",
                ],
            ),
            (StepType, ["think", "observe", "act", "evaluate", "search", "calculate"]),
            (ProviderType, ["skill", "tool", "plugin", "agent", "system"]),
            (VerdictType, ["allow", "deny", "confirm", "sandbox"]),
            (ErrorPolicy, ["abort", "skip", "retry", "ignore"]),
            (
                ErrorType,
                [
                    "hallucination",
                    "contradiction",
                    "incomplete",
                    "irrelevant",
                    "logical_gap",
                    "factual_error",
                    "style",
                    "safety",
                ],
            ),
            (
                LessonCategory,
                ["strategy", "avoidance", "preference", "optimization", "safety"],
            ),
            (
                TaskStatus,
                [
                    "pending",
                    "running",
                    "completed",
                    "failed",
                    "cancelled",
                    "blocked",
                    "skipped",
                ],
            ),
            (
                CognitiveState,
                [
                    "idle",
                    "perceiving",
                    "intent_analyzing",
                    "clarifying",
                    "goal_resolving",
                    "plan_creating",
                    "reasoning",
                    "deciding",
                    "capability_checking",
                    "policy_evaluating",
                    "confirming",
                    "skill_selecting",
                    "tool_resolving",
                    "executing",
                    "reflecting",
                    "recording",
                    "responding",
                    "error",
                    "terminated",
                ],
            ),
            (ExecutionType, ["local", "remote", "sandboxed"]),
            (RoleType, ["system", "user", "assistant", "tool"]),
            (SessionStatus, ["active", "idle", "paused", "expired", "terminated"]),
        ],
    )
    def test_enum_values(self, enum_cls: type[Enum], expected: list[Any]) -> None:
        values = [v.value for v in enum_cls]
        assert sorted(values if isinstance(expected[0], str) else values) == sorted(
            expected
        )

    def test_goal_priority_ordering(self) -> None:
        assert GoalPriority.CRITICAL.value > GoalPriority.HIGH.value
        assert GoalPriority.HIGH.value > GoalPriority.NORMAL.value
        assert GoalPriority.NORMAL.value > GoalPriority.LOW.value
        assert GoalPriority.LOW.value > GoalPriority.BACKGROUND.value


# =============================================================================
# Immutable Composition (Conversation.append)
# =============================================================================


class TestImmutableComposition:
    """Conversation.append returns a new instance."""

    def test_append_returns_new(self) -> None:
        c1 = Conversation(id="conv1")
        msg = InputMessage(content="hello")
        c2 = c1.append(msg)
        assert c1 is not c2
        assert len(c1.messages) == 0
        assert len(c2.messages) == 1
        assert c2.messages[0].content == "hello"
        assert c1.id == c2.id

    def test_conversation_is_frozen(self) -> None:
        c = Conversation()
        with pytest.raises((TypeError, ValidationError)):
            c.session_id = "new"  # type: ignore[misc]


# =============================================================================
# Model Instantiation — all models construct successfully
# =============================================================================


class TestModelInstantiation:
    """Every model can be constructed with minimal required args."""

    @pytest.mark.parametrize(
        "cls,kwargs",
        [
            (InputMessage, {"content": "hi"}),
            (Conversation, {}),
            (Session, {}),
            (Entity, {"type": "url", "value": "x"}),
            (QualityScore, {}),
            (RawMessage, {"content": "hi"}),
            (Percept, {"raw": RawMessage(content="hi"), "modality": ModalityType.TEXT}),
            (IntentConfidence, {"primary": 0.5}),
            (
                IntentClassification,
                {
                    "primary": IntentType.ASK_QUESTION,
                    "confidence": IntentConfidence(primary=0.5),
                },
            ),
            (ClarificationRequest, {"ambiguity_description": "unclear"}),
            (
                Intent,
                {
                    "primary": IntentType.ASK_QUESTION,
                    "confidence": IntentConfidence(primary=0.9),
                },
            ),
            (Goal, {"description": "test"}),
            (GoalHierarchy, {"root": Goal(description="test")}),
            (SkillStep, {"order": 0, "skill_name": "s"}),
            (DependencyGraph, {}),
            (Plan, {}),
            (ReasoningStep, {"id": "s1"}),
            (ReasoningRequest, {"problem": "p"}),
            (ReasoningTrace, {}),
            (DecisionScore, {"criterion_name": "c", "score": 0.5}),
            (DecisionReason, {}),
            (ActionCandidate, {"action_type": "call"}),
            (Decision, {"selected_action": ActionCandidate(action_type="call")}),
            (
                CapabilityRequirement,
                {"action_type": "call", "required_capability": "search"},
            ),
            (ResourceEstimate, {}),
            (CapabilityProvider, {"name": "p", "provider_type": ProviderType.TOOL}),
            (
                Capability,
                {
                    "name": "c",
                    "provider": CapabilityProvider(
                        name="p", provider_type=ProviderType.TOOL
                    ),
                },
            ),
            (CapabilityResult, {}),
            (
                PolicyRule,
                {"name": "r", "action_pattern": "*", "verdict": VerdictType.ALLOW},
            ),
            (PolicyAction, {"action_type": "call"}),
            (AppliedPolicy, {"policy_name": "r", "verdict": VerdictType.ALLOW}),
            (SandboxConfig, {}),
            (
                PolicyDecision,
                {
                    "action": PolicyAction(action_type="call"),
                    "verdict": VerdictType.ALLOW,
                },
            ),
            (SkillRequirement, {"skill_name": "s"}),
            (ToolRequirement, {"tool_name": "t"}),
            (SkillDescriptor, {"name": "s"}),
            (Skill, {"name": "s", "descriptor": SkillDescriptor(name="s")}),
            (SkillResult, {"skill_name": "s"}),
            (RateLimit, {}),
            (ToolDescriptor, {"name": "t"}),
            (ToolRequest, {"tool_name": "t"}),
            (ToolBinding, {"tool_name": "t", "descriptor": ToolDescriptor(name="t")}),
            (ToolResult, {"tool_name": "t"}),
            (Task, {"action_type": "call", "action_name": "search"}),
            (TaskResult, {"task_id": "t1", "status": TaskStatus.COMPLETED}),
            (ReflectionScore, {"dimension": "acc", "score": 0.5}),
            (ErrorDetail, {"type": ErrorType.HALLUCINATION}),
            (Refinement, {}),
            (ReflectionRequest, {"output": "test"}),
            (ReflectionReport, {}),
            (DecisionSummary, {"action_type": "call"}),
            (LessonLearned, {"description": "lesson"}),
            (Insight, {"description": "insight"}),
            (ExperienceQuery, {}),
            (Experience, {}),
            (ErrorPolicy, ErrorPolicy.ABORT),  # enums
            (PipelineStep, {"name": "p", "order": 0}),
            (
                PipelineDefinition,
                {"name": "p", "steps": (PipelineStep(name="p", order=0),)},
            ),
            (PipelineMetadata, {}),
            (PipelineContext, {}),
            (Message, {"role": RoleType.USER, "content": "hi"}),
            (ToolDefinition, {"name": "t"}),
            (ResponseFormat, {}),
            (ModelRequest, {}),
            (ModelResponse, {}),
            (ModelCapability, {}),
            (
                Transition,
                {
                    "from_state": CognitiveState.IDLE,
                    "to_state": CognitiveState.PERCEIVING,
                },
            ),
            (SessionState, {}),
            (TokenBudget, {}),
            (CognitiveContext, {}),
            (ContextSource, {}),
            (ContextDelta, {"field": "name", "value": "v"}),
            (ContextSnapshot, {"context": CognitiveContext(), "version": 1}),
            (OutputMessage, {}),
            (StreamChunk, {}),
        ],
    )
    def test_construct(self, cls: type, kwargs: dict[str, Any] | Any) -> None:
        if isinstance(kwargs, dict):
            instance = cls(**kwargs)
        else:
            instance = cls(kwargs)
        assert instance is not None


# =============================================================================
# Nested validation — recursive models
# =============================================================================


class TestNestedValidation:
    """Recursive / nested model validation."""

    def test_percept_contains_raw_message(self) -> None:
        raw = RawMessage(content="hello", modality=ModalityType.TEXT)
        p = Percept(raw=raw, modality=ModalityType.TEXT)
        assert p.raw.content == "hello"

    def test_goal_hierarchy_flat_map(self) -> None:
        root = Goal(description="root")
        child = Goal(description="child", parent_id="root")
        gh = GoalHierarchy(
            root=root, children=(child,), all_goals={"root": root, "child": child}
        )
        assert gh.all_goals["child"].description == "child"

    def test_decision_contains_selected_action(self) -> None:
        action = ActionCandidate(action_type="respond", confidence=0.9)
        d = Decision(selected_action=action)
        assert d.selected_action.action_type == "respond"
        assert d.selected_action.confidence == 0.9

    def test_policy_decision_chain(self) -> None:
        action = PolicyAction(action_type="tool_call", tool_name="search")
        verdict = PolicyDecision(action=action, verdict=VerdictType.ALLOW)
        assert verdict.action.tool_name == "search"
        assert verdict.verdict == VerdictType.ALLOW

    def test_capability_bidirectional(self) -> None:
        provider = CapabilityProvider(
            name="web_search", provider_type=ProviderType.TOOL
        )
        cap = Capability(name="search", provider=provider, available=True)
        assert cap.provider.name == "web_search"
        assert cap.available is True

    def test_session_contains_conversation(self) -> None:
        msg = InputMessage(content="hello")
        conv = Conversation(id="c1", messages=(msg,))
        sess = Session(id="s1", conversation=conv)
        assert sess.conversation.id == "c1"
        assert sess.conversation.messages[0].content == "hello"


# =============================================================================
# Type enforcement — wrong types rejected
# =============================================================================


class TestTypeEnforcement:
    """Pydantic type validation rejects wrong types."""

    def test_string_in_enum_field(self) -> None:
        with pytest.raises(ValidationError):
            Entity(type="url", value="x", confidence="not-a-float")  # type: ignore[arg-type]

    def test_wrong_enum_value(self) -> None:
        with pytest.raises(ValidationError):
            RawMessage(content="hi", modality="invalid_modality")  # type: ignore[arg-type]

    def test_list_instead_of_tuple(self) -> None:
        r = Entity(type="url", value="x")
        with pytest.raises(ValidationError):
            Percept(raw=r, modality=ModalityType.TEXT, entities=["not", "entities"])  # type: ignore[arg-type]
