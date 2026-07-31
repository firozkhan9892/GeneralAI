"""Tests for inter-engine communication contracts (Phase 3.3)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
from pydantic import ValidationError

from app.kernel.contracts import (
    CapabilityToPolicyRequest,
    CapabilityToPolicyResponse,
    ContractRequest,
    ContractResponse,
    ContractResult,
    DecisionToCapabilityRequest,
    DecisionToCapabilityResponse,
    EngineType,
    ErrorInfo,
    EventSeverity,
    ExperienceToMemoryRequest,
    ExperienceToMemoryResponse,
    GoalToPlannerRequest,
    GoalToPlannerResponse,
    IntentToGoalRequest,
    IntentToGoalResponse,
    MemoryToResponseRequest,
    MemoryToResponseResponse,
    MessageEnvelope,
    PerceptionToIntentRequest,
    PerceptionToIntentResponse,
    PipelineEvent,
    PlannerToReasoningRequest,
    PlannerToReasoningResponse,
    PolicyToTaskRequest,
    PolicyToTaskResponse,
    ReasoningToDecisionRequest,
    ReasoningToDecisionResponse,
    ReflectionToExperienceRequest,
    ReflectionToExperienceResponse,
    ResultStatus,
    TaskToToolRequest,
    TaskToToolResponse,
    ToolToReflectionRequest,
    ToolToReflectionResponse,
    EVENT_SEVERITY,
)
from app.kernel.contracts.events import PipelineEvent as PipelineEventDirect
from app.kernel.intent.models import IntentConfidence, IntentType
from app.kernel.perception.models import ModalityType, Percept, RawMessage


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_envelope() -> MessageEnvelope:
    return MessageEnvelope(
        source_engine=EngineType.PERCEPTION,
        target_engine=EngineType.INTENT,
        correlation_id="corr-001",
        session_id="sess-001",
        payload={"key": "value"},
    )


# ── Base primitives ─────────────────────────────────────────────────────────


class TestEngineType:
    def test_values(self) -> None:
        assert EngineType.PERCEPTION.value == "perception"
        assert EngineType.INTENT.value == "intent"
        assert EngineType.ORCHESTRATOR.value == "orchestrator"
        assert EngineType.UNKNOWN.value == "unknown"

    def test_all_engines_covered(self) -> None:
        expected = {
            "perception",
            "intent",
            "goal",
            "planner",
            "reasoning",
            "decision",
            "capability",
            "policy",
            "task",
            "tool",
            "skill",
            "reflection",
            "experience",
            "memory",
            "response",
            "orchestrator",
            "pipeline",
            "unknown",
        }
        assert {e.value for e in EngineType} == expected


class TestResultStatus:
    def test_values(self) -> None:
        assert ResultStatus.SUCCESS.value == "success"
        assert ResultStatus.FAILURE.value == "failure"
        assert ResultStatus.RETRY.value == "retry"
        assert ResultStatus.CANCELLED.value == "cancelled"
        assert ResultStatus.TIMEOUT.value == "timeout"


class TestErrorInfo:
    def test_minimal(self) -> None:
        err = ErrorInfo(code="test.error")
        assert err.code == "test.error"
        assert err.message == ""
        assert err.details is None
        assert err.retryable is False
        assert err.source_engine == EngineType.UNKNOWN

    def test_full(self) -> None:
        err = ErrorInfo(
            code="cap.unavailable",
            message="Capability not found",
            details={"cap_id": "abc"},
            retryable=True,
            source_engine=EngineType.CAPABILITY,
        )
        assert err.code == "cap.unavailable"
        assert err.retryable is True
        assert err.details == {"cap_id": "abc"}

    def test_frozen(self) -> None:
        err = ErrorInfo(code="x")
        with pytest.raises(ValidationError):
            err.code = "y"


class TestContractResult:
    def test_default_success(self) -> None:
        r: ContractResult = ContractResult()
        assert r.status == ResultStatus.SUCCESS
        assert r.error is None
        assert r.correlation_id == ""
        assert r.duration_ms == 0

    def test_failure(self) -> None:
        err = ErrorInfo(code="fail")
        r: ContractResult = ContractResult(
            status=ResultStatus.FAILURE, error=err, correlation_id="c1"
        )
        assert r.status == ResultStatus.FAILURE
        assert r.error is not None
        assert r.error.code == "fail"

    def test_frozen(self) -> None:
        r: ContractResult = ContractResult()
        with pytest.raises(ValidationError):
            r.status = ResultStatus.FAILURE

    def test_duration_ge_zero(self) -> None:
        with pytest.raises(ValidationError):
            ContractResult(duration_ms=-1)


class TestMessageEnvelope:
    def test_minimal(self) -> None:
        env = MessageEnvelope(
            source_engine=EngineType.PLANNER,
            target_engine=EngineType.REASONING,
        )
        assert env.source_engine == EngineType.PLANNER
        assert env.target_engine == EngineType.REASONING
        assert env.payload == {}
        assert isinstance(env.timestamp, datetime)

    def test_full(self) -> None:
        env = MessageEnvelope(
            correlation_id="c1",
            session_id="s1",
            source_engine=EngineType.PERCEPTION,
            target_engine=EngineType.INTENT,
            contract_version="2.0.0",
            metadata={"route": "fast"},
            context_ref="ctx://abc",
            payload={"query": "hello"},
        )
        assert env.correlation_id == "c1"
        assert env.contract_version == "2.0.0"
        assert env.context_ref == "ctx://abc"

    def test_from_payload(self) -> None:
        class FakePayload(ContractRequest):
            query: str

        payload = FakePayload(
            query="hello",
            source_engine=EngineType.PERCEPTION,
            target_engine=EngineType.INTENT,
        )
        env = MessageEnvelope.from_payload(
            payload=payload,
            source_engine=EngineType.PERCEPTION,
            target_engine=EngineType.INTENT,
            session_id="s1",
            correlation_id="c1",
        )
        assert env.source_engine == EngineType.PERCEPTION
        assert env.target_engine == EngineType.INTENT
        assert env.payload["query"] == "hello"
        assert env.payload["source_engine"] == "perception"
        assert env.session_id == "s1"

    def test_frozen(self) -> None:
        env = MessageEnvelope(
            source_engine=EngineType.PERCEPTION, target_engine=EngineType.INTENT
        )
        with pytest.raises(ValidationError):
            env.correlation_id = "new"

    def test_serialize_roundtrip(self, sample_envelope: MessageEnvelope) -> None:
        data = sample_envelope.model_dump()
        restored = MessageEnvelope.model_validate(data)
        assert restored.correlation_id == sample_envelope.correlation_id
        assert restored.source_engine == sample_envelope.source_engine


class TestContractRequest:
    def test_defaults(self) -> None:
        r = ContractRequest()
        assert r.correlation_id == ""
        assert r.session_id == ""
        assert r.source_engine == EngineType.UNKNOWN
        assert r.contract_version == "1.0.0"
        assert r.metadata == {}
        assert r.context_ref is None

    def test_extra_fields_ignored(self) -> None:
        r = ContractRequest.model_validate({"extra_field": "should_ignore"})
        assert not hasattr(r, "extra_field")

    def test_frozen(self) -> None:
        r = ContractRequest()
        with pytest.raises(ValidationError):
            r.correlation_id = "x"


class TestContractResponse:
    def test_defaults(self) -> None:
        r = ContractResponse()
        assert r.result.status == ResultStatus.SUCCESS
        assert r.correlation_id == ""

    def test_extra_fields_ignored(self) -> None:
        r = ContractResponse.model_validate({"future_field": "should_ignore"})
        assert not hasattr(r, "future_field")

    def test_frozen(self) -> None:
        r = ContractResponse()
        with pytest.raises(ValidationError):
            r.correlation_id = "x"


# ── Event tests ─────────────────────────────────────────────────────────────


class TestPipelineEvent:
    def test_values(self) -> None:
        assert PipelineEvent.PERCEPTION_STARTED.value == "perception.started"
        assert PipelineEvent.PERCEPTION_FAILED.value == "perception.failed"
        assert PipelineEvent.PIPELINE_CANCELLED.value == "pipeline.cancelled"
        assert PipelineEvent.PIPELINE_TIMEOUT.value == "pipeline.timeout"

    def test_enum_is_str_enum(self) -> None:
        assert isinstance(PipelineEvent.PERCEPTION_STARTED, str)

    def test_event_names_are_unique(self) -> None:
        values = [e.value for e in PipelineEvent]
        assert len(values) == len(set(values))

    def test_all_events_have_severity_or_default_debug(self) -> None:
        for event in PipelineEvent:
            if event in EVENT_SEVERITY:
                assert isinstance(EVENT_SEVERITY[event], EventSeverity)

    def test_import_from_contracts_events(self) -> None:
        assert PipelineEvent is PipelineEventDirect

    def test_reimported_via_init(self) -> None:
        from app.kernel.contracts import PipelineEvent as PE

        assert PE is PipelineEvent


class TestEventSeverity:
    def test_values(self) -> None:
        assert EventSeverity.INFO.value == "info"
        assert EventSeverity.ERROR.value == "error"
        assert EventSeverity.CRITICAL.value == "critical"


class TestEventSeverityMapping:
    def test_error_events_severity(self) -> None:
        for event, sev in EVENT_SEVERITY.items():
            if "failed" in event.value or "error" in event.value:
                assert sev in (EventSeverity.ERROR, EventSeverity.WARNING)

    def test_info_events_severity(self) -> None:
        for event, sev in EVENT_SEVERITY.items():
            if "completed" in event.value or "created" in event.value:
                assert sev == EventSeverity.INFO

    def test_warning_events(self) -> None:
        assert EVENT_SEVERITY[PipelineEvent.PIPELINE_CANCELLED] == EventSeverity.WARNING
        assert EVENT_SEVERITY[PipelineEvent.POLICY_DENIED] == EventSeverity.WARNING
        assert (
            EVENT_SEVERITY[PipelineEvent.INTENT_CLARIFICATION_REQUESTED]
            == EventSeverity.WARNING
        )


# ── Contract pair helper ────────────────────────────────────────────────────


def _check_request(req: ContractRequest, src: EngineType, tgt: EngineType) -> None:
    assert req.source_engine == src
    assert req.target_engine == tgt
    assert req.contract_version == "1.0.0"
    assert isinstance(req.timestamp, datetime)


def _check_response(resp: ContractResponse, src: EngineType, tgt: EngineType) -> None:
    assert resp.source_engine == src
    assert resp.target_engine == tgt
    assert isinstance(resp.result, ContractResult)


def _check_roundtrip(req: ContractRequest, resp: ContractResponse) -> None:
    req_data = req.model_dump()
    restored_req = req.__class__.model_validate(req_data)
    assert restored_req.model_dump() == req_data

    resp_data = resp.model_dump()
    restored_resp = resp.__class__.model_validate(resp_data)
    assert restored_resp.model_dump() == resp_data


def _check_forward_compat(
    model_class: type[ContractRequest | ContractResponse], **required: Any
) -> None:
    data: dict[str, Any] = {"unknown_field": "should_be_ignored"}
    data.update(required)
    instance = model_class.model_validate(data)
    assert not hasattr(instance, "unknown_field")


# ── Perception <-> Intent ───────────────────────────────────────────────────


class TestPerceptionToIntentContract:
    def test_request_defaults(self) -> None:
        from app.kernel.perception.models import Percept

        raw_msg = RawMessage(content="hi", modality=ModalityType.TEXT)
        p = Percept(raw=raw_msg, modality=ModalityType.TEXT)
        req = PerceptionToIntentRequest(percept=p)
        _check_request(req, EngineType.PERCEPTION, EngineType.INTENT)
        assert req.percept.raw.content == "hi"

    def test_response_defaults(self) -> None:
        resp = PerceptionToIntentResponse()
        _check_response(resp, EngineType.INTENT, EngineType.PERCEPTION)

    def test_roundtrip(self) -> None:
        from app.kernel.perception.models import Percept

        raw_msg = RawMessage(content="hello", modality=ModalityType.TEXT)
        p = Percept(raw=raw_msg, modality=ModalityType.TEXT)
        req = PerceptionToIntentRequest(percept=p)
        resp = PerceptionToIntentResponse()
        _check_roundtrip(req, resp)

    def test_forward_compat(self) -> None:
        raw_msg = RawMessage(content="x", modality=ModalityType.TEXT)
        percept = Percept(raw=raw_msg, modality=ModalityType.TEXT)
        _check_forward_compat(PerceptionToIntentRequest, percept=percept)
        _check_forward_compat(PerceptionToIntentResponse)

    def test_percept_required(self) -> None:
        with pytest.raises(ValidationError):
            PerceptionToIntentRequest()


# ── Intent <-> Goal ─────────────────────────────────────────────────────────


class TestIntentToGoalContract:
    def test_request_defaults(self) -> None:
        from app.kernel.intent.models import Intent

        intent = Intent(
            primary=IntentType.ASK_QUESTION, confidence=IntentConfidence(primary=0.9)
        )
        req = IntentToGoalRequest(intent=intent)
        _check_request(req, EngineType.INTENT, EngineType.GOAL)

    def test_roundtrip(self) -> None:
        from app.kernel.intent.models import Intent

        intent = Intent(
            primary=IntentType.ASK_QUESTION, confidence=IntentConfidence(primary=0.95)
        )
        req = IntentToGoalRequest(intent=intent)
        resp = IntentToGoalResponse()
        _check_roundtrip(req, resp)

    def test_forward_compat(self) -> None:
        from app.kernel.intent.models import Intent

        _check_forward_compat(
            IntentToGoalRequest,
            intent=Intent(
                primary=IntentType.UNKNOWN, confidence=IntentConfidence(primary=0.5)
            ),
        )
        _check_forward_compat(IntentToGoalResponse)


# ── Goal <-> Planner ────────────────────────────────────────────────────────


class TestGoalToPlannerContract:
    def test_request_defaults(self) -> None:
        from app.kernel.goals.models import Goal

        g = Goal(description="test_goal")
        req = GoalToPlannerRequest(goal=g)
        _check_request(req, EngineType.GOAL, EngineType.PLANNER)

    def test_roundtrip(self) -> None:
        from app.kernel.goals.models import Goal, GoalPriority

        g = Goal(description="parse file", priority=GoalPriority.NORMAL)
        req = GoalToPlannerRequest(goal=g)
        resp = GoalToPlannerResponse()
        _check_roundtrip(req, resp)

    def test_forward_compat(self) -> None:
        from app.kernel.goals.models import Goal

        _check_forward_compat(GoalToPlannerRequest, goal=Goal(description="g"))
        _check_forward_compat(GoalToPlannerResponse)


# ── Planner <-> Reasoning ───────────────────────────────────────────────────


class TestPlannerToReasoningContract:
    def test_request_defaults(self) -> None:
        from app.kernel.planning.models import Plan

        plan = Plan(steps=())
        req = PlannerToReasoningRequest(plan=plan)
        _check_request(req, EngineType.PLANNER, EngineType.REASONING)

    def test_roundtrip(self) -> None:
        from app.kernel.planning.models import Plan, SkillStep

        step = SkillStep(order=0, skill_name="analyze", description="step 1")
        plan = Plan(steps=(step,))
        req = PlannerToReasoningRequest(plan=plan)
        resp = PlannerToReasoningResponse()
        _check_roundtrip(req, resp)

    def test_forward_compat(self) -> None:
        from app.kernel.planning.models import Plan

        _check_forward_compat(PlannerToReasoningRequest, plan=Plan(steps=()))
        _check_forward_compat(PlannerToReasoningResponse)


# ── Reasoning <-> Decision ──────────────────────────────────────────────────


class TestReasoningToDecisionContract:
    def test_request_defaults(self) -> None:
        from app.kernel.reasoning.models import ReasoningTrace

        trace = ReasoningTrace(steps=())
        req = ReasoningToDecisionRequest(trace=trace)
        _check_request(req, EngineType.REASONING, EngineType.DECISION)

    def test_roundtrip(self) -> None:
        from app.kernel.reasoning.models import ReasoningStep, ReasoningTrace

        step = ReasoningStep(id="r1", content="therefore...")
        trace = ReasoningTrace(steps=(step,))
        req = ReasoningToDecisionRequest(trace=trace)
        resp = ReasoningToDecisionResponse()
        _check_roundtrip(req, resp)

    def test_forward_compat(self) -> None:
        from app.kernel.reasoning.models import ReasoningTrace

        _check_forward_compat(
            ReasoningToDecisionRequest, trace=ReasoningTrace(steps=())
        )
        _check_forward_compat(ReasoningToDecisionResponse)


# ── Decision <-> Capability ─────────────────────────────────────────────────


class TestDecisionToCapabilityContract:
    def test_request_defaults(self) -> None:
        from app.kernel.decision.models import ActionCandidate, Decision

        d = Decision(selected_action=ActionCandidate(action_type="run"))
        req = DecisionToCapabilityRequest(decision=d)
        _check_request(req, EngineType.DECISION, EngineType.CAPABILITY)

    def test_roundtrip(self) -> None:
        from app.kernel.decision.models import ActionCandidate, Decision

        d = Decision(selected_action=ActionCandidate(action_type="run", confidence=0.9))
        req = DecisionToCapabilityRequest(decision=d)
        resp = DecisionToCapabilityResponse()
        _check_roundtrip(req, resp)

    def test_forward_compat(self) -> None:
        from app.kernel.decision.models import ActionCandidate, Decision

        _check_forward_compat(
            DecisionToCapabilityRequest,
            decision=Decision(selected_action=ActionCandidate(action_type="a")),
        )
        _check_forward_compat(DecisionToCapabilityResponse)


# ── Capability <-> Policy ────────────────────────────────────────────────────


class TestCapabilityToPolicyContract:
    def test_request_defaults(self) -> None:
        from app.kernel.capability.models import CapabilityResult

        cr = CapabilityResult()
        req = CapabilityToPolicyRequest(capability_result=cr)
        _check_request(req, EngineType.CAPABILITY, EngineType.POLICY)

    def test_roundtrip(self) -> None:
        from app.kernel.capability.models import CapabilityResult

        cr = CapabilityResult()
        req = CapabilityToPolicyRequest(capability_result=cr)
        resp = CapabilityToPolicyResponse()
        _check_roundtrip(req, resp)

    def test_forward_compat(self) -> None:
        from app.kernel.capability.models import CapabilityResult

        _check_forward_compat(
            CapabilityToPolicyRequest,
            capability_result=CapabilityResult(),
        )
        _check_forward_compat(CapabilityToPolicyResponse)


# ── Policy <-> Task ──────────────────────────────────────────────────────────


class TestPolicyToTaskContract:
    def test_request_defaults(self) -> None:
        from app.kernel.policy.models import PolicyAction, PolicyDecision, VerdictType

        pd = PolicyDecision(
            action=PolicyAction(action_type="tool_call"), verdict=VerdictType.ALLOW
        )
        req = PolicyToTaskRequest(policy_decision=pd)
        _check_request(req, EngineType.POLICY, EngineType.TASK)

    def test_roundtrip(self) -> None:
        from app.kernel.policy.models import PolicyAction, PolicyDecision, VerdictType

        pd = PolicyDecision(
            action=PolicyAction(action_type="tool_call"),
            verdict=VerdictType.ALLOW,
            rationale="ok",
        )
        req = PolicyToTaskRequest(policy_decision=pd)
        resp = PolicyToTaskResponse()
        _check_roundtrip(req, resp)

    def test_forward_compat(self) -> None:
        from app.kernel.policy.models import PolicyAction, PolicyDecision, VerdictType

        _check_forward_compat(
            PolicyToTaskRequest,
            policy_decision=PolicyDecision(
                action=PolicyAction(action_type="a"), verdict=VerdictType.ALLOW
            ),
        )
        _check_forward_compat(PolicyToTaskResponse)


# ── Task <-> Tool ────────────────────────────────────────────────────────────


class TestTaskToToolContract:
    def test_request_defaults(self) -> None:
        from app.kernel.tasks.models import Task

        t = Task(action_type="test", action_name="x")
        req = TaskToToolRequest(task=t)
        _check_request(req, EngineType.TASK, EngineType.TOOL)

    def test_roundtrip(self) -> None:
        from app.kernel.tasks.models import Task

        t = Task(action_type="read_file", action_name="read_config", plan_step_order=5)
        req = TaskToToolRequest(task=t)
        resp = TaskToToolResponse()
        _check_roundtrip(req, resp)

    def test_forward_compat(self) -> None:
        from app.kernel.tasks.models import Task

        _check_forward_compat(
            TaskToToolRequest, task=Task(action_type="a", action_name="b")
        )
        _check_forward_compat(TaskToToolResponse)


# ── Tool <-> Reflection ──────────────────────────────────────────────────────


class TestToolToReflectionContract:
    def test_request_defaults(self) -> None:
        from app.kernel.tools.models import ToolResult

        tr = ToolResult(tool_name="echo", success=True)
        req = ToolToReflectionRequest(tool_result=tr)
        _check_request(req, EngineType.TOOL, EngineType.REFLECTION)

    def test_roundtrip(self) -> None:
        from app.kernel.tools.models import ToolResult

        tr = ToolResult(tool_name="echo", success=True, output="hello")
        req = ToolToReflectionRequest(tool_result=tr)
        resp = ToolToReflectionResponse()
        _check_roundtrip(req, resp)

    def test_forward_compat(self) -> None:
        from app.kernel.tools.models import ToolResult

        _check_forward_compat(
            ToolToReflectionRequest, tool_result=ToolResult(tool_name="t", success=True)
        )
        _check_forward_compat(ToolToReflectionResponse)


# ── Reflection <-> Experience ────────────────────────────────────────────────


class TestReflectionToExperienceContract:
    def test_request_defaults(self) -> None:
        from app.kernel.reflection.models import ReflectionReport

        rr = ReflectionReport(overall_score=0.9)
        req = ReflectionToExperienceRequest(reflection_report=rr)
        _check_request(req, EngineType.REFLECTION, EngineType.EXPERIENCE)
        assert req.session_data == {}

    def test_response_defaults(self) -> None:
        resp = ReflectionToExperienceResponse()
        _check_response(resp, EngineType.EXPERIENCE, EngineType.REFLECTION)

    def test_roundtrip(self) -> None:
        from app.kernel.reflection.models import ReflectionReport

        rr = ReflectionReport(overall_score=0.85)
        req = ReflectionToExperienceRequest(
            reflection_report=rr, session_data={"key": "val"}
        )
        resp = ReflectionToExperienceResponse(experience_id="exp-001")
        _check_roundtrip(req, resp)

    def test_forward_compat(self) -> None:
        from app.kernel.reflection.models import ReflectionReport

        _check_forward_compat(
            ReflectionToExperienceRequest,
            reflection_report=ReflectionReport(overall_score=0.9),
        )
        _check_forward_compat(ReflectionToExperienceResponse)


# ── Experience <-> Memory ────────────────────────────────────────────────────


class TestExperienceToMemoryContract:
    def test_request_defaults(self) -> None:
        req = ExperienceToMemoryRequest()
        _check_request(req, EngineType.EXPERIENCE, EngineType.MEMORY)
        assert req.experience is None
        assert req.query is None

    def test_response_defaults(self) -> None:
        resp = ExperienceToMemoryResponse()
        _check_response(resp, EngineType.MEMORY, EngineType.EXPERIENCE)
        assert resp.experiences == ()
        assert resp.total_count == 0

    def test_roundtrip(self) -> None:
        from app.kernel.experience.models import Experience

        exp = Experience(goal_description="test", outcome_score=0.8, success=True)
        req = ExperienceToMemoryRequest(experience=exp)
        resp = ExperienceToMemoryResponse(stored_id="mem-001")
        _check_roundtrip(req, resp)

    def test_stored_id_roundtrip(self) -> None:
        resp = ExperienceToMemoryResponse(stored_id="mem-abc", total_count=1)
        data = resp.model_dump()
        restored = ExperienceToMemoryResponse.model_validate(data)
        assert restored.stored_id == "mem-abc"
        assert restored.total_count == 1

    def test_forward_compat(self) -> None:
        _check_forward_compat(ExperienceToMemoryRequest)
        _check_forward_compat(ExperienceToMemoryResponse)


# ── Memory <-> Response ──────────────────────────────────────────────────────


class TestMemoryToResponseContract:
    def test_request_defaults(self) -> None:
        req = MemoryToResponseRequest()
        _check_request(req, EngineType.MEMORY, EngineType.RESPONSE)
        assert req.session_data == {}
        assert req.stream is False

    def test_response_defaults(self) -> None:
        resp = MemoryToResponseResponse()
        _check_response(resp, EngineType.RESPONSE, EngineType.MEMORY)

    def test_roundtrip(self) -> None:
        req = MemoryToResponseRequest(session_data={"key": "val"}, stream=True)
        resp = MemoryToResponseResponse(stream_id="stream-001")
        _check_roundtrip(req, resp)

    def test_forward_compat(self) -> None:
        _check_forward_compat(MemoryToResponseRequest)
        _check_forward_compat(MemoryToResponseResponse)


# ── Cross-cutting ────────────────────────────────────────────────────────────


class TestCrossCutting:
    def test_all_contract_request_classes_inherit(self) -> None:
        classes = [
            PerceptionToIntentRequest,
            IntentToGoalRequest,
            GoalToPlannerRequest,
            PlannerToReasoningRequest,
            ReasoningToDecisionRequest,
            DecisionToCapabilityRequest,
            CapabilityToPolicyRequest,
            PolicyToTaskRequest,
            TaskToToolRequest,
            ToolToReflectionRequest,
            ReflectionToExperienceRequest,
            ExperienceToMemoryRequest,
            MemoryToResponseRequest,
        ]
        for cls in classes:
            assert issubclass(cls, ContractRequest), (
                f"{cls.__name__} not subclass of ContractRequest"
            )

    def test_all_contract_response_classes_inherit(self) -> None:
        classes = [
            PerceptionToIntentResponse,
            IntentToGoalResponse,
            GoalToPlannerResponse,
            PlannerToReasoningResponse,
            ReasoningToDecisionResponse,
            DecisionToCapabilityResponse,
            CapabilityToPolicyResponse,
            PolicyToTaskResponse,
            TaskToToolResponse,
            ToolToReflectionResponse,
            ReflectionToExperienceResponse,
            ExperienceToMemoryResponse,
            MemoryToResponseResponse,
        ]
        for cls in classes:
            assert issubclass(cls, ContractResponse), (
                f"{cls.__name__} not subclass of ContractResponse"
            )

    def test_all_contracts_serializable_to_json(self) -> None:
        from app.kernel.capability.models import CapabilityResult
        from app.kernel.decision.models import ActionCandidate, Decision
        from app.kernel.experience.models import Experience
        from app.kernel.goals.models import Goal
        from app.kernel.intent.models import Intent
        from app.kernel.perception.models import Percept
        from app.kernel.planning.models import Plan, SkillStep
        from app.kernel.policy.models import PolicyAction, PolicyDecision, VerdictType
        from app.kernel.reasoning.models import ReasoningStep, ReasoningTrace
        from app.kernel.reflection.models import ReflectionReport
        from app.kernel.tasks.models import Task
        from app.kernel.tools.models import ToolResult

        requests = [
            PerceptionToIntentRequest(
                percept=Percept(
                    raw=RawMessage(content="hi", modality=ModalityType.TEXT),
                    modality=ModalityType.TEXT,
                )
            ),
            IntentToGoalRequest(
                intent=Intent(
                    primary=IntentType.UNKNOWN, confidence=IntentConfidence(primary=0.5)
                )
            ),
            GoalToPlannerRequest(goal=Goal(description="g")),
            PlannerToReasoningRequest(
                plan=Plan(steps=(SkillStep(order=0, skill_name="a", description="d"),))
            ),
            ReasoningToDecisionRequest(
                trace=ReasoningTrace(steps=(ReasoningStep(id="r1", content="c"),))
            ),
            DecisionToCapabilityRequest(
                decision=Decision(selected_action=ActionCandidate(action_type="run"))
            ),
            CapabilityToPolicyRequest(
                capability_result=CapabilityResult(),
            ),
            PolicyToTaskRequest(
                policy_decision=PolicyDecision(
                    action=PolicyAction(action_type="tool_call"),
                    verdict=VerdictType.ALLOW,
                )
            ),
            TaskToToolRequest(task=Task(action_type="run", action_name="a")),
            ToolToReflectionRequest(
                tool_result=ToolResult(tool_name="t", success=True)
            ),
            ReflectionToExperienceRequest(
                reflection_report=ReflectionReport(overall_score=0.9)
            ),
            ExperienceToMemoryRequest(experience=Experience()),
            MemoryToResponseRequest(session_data={"k": "v"}),
        ]
        for req in requests:
            data = req.model_dump_json()
            assert '"correlation_id"' in data
            assert '"source_engine"' in data
