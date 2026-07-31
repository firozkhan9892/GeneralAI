"""Comprehensive tests for the Response Builder (Phase 3.5J)."""

from __future__ import annotations

import pytest

from app.kernel.response.builder import ResponseBuilder
from app.kernel.response.models import OutputMessage, StreamChunk


# ──────────────────────────────────────────────
# Builder creation & defaults
# ──────────────────────────────────────────────


class TestBuilderInit:
    """Builder construction and default state."""

    def test_create_builder(self) -> None:
        builder = ResponseBuilder()
        assert builder is not None

    def test_default_format_text(self) -> None:
        builder = ResponseBuilder()
        assert builder._default_format == "text"

    def test_custom_format(self) -> None:
        builder = ResponseBuilder(default_format="markdown")
        assert builder._default_format == "markdown"


# ──────────────────────────────────────────────
# build — basic / empty
# ──────────────────────────────────────────────


class TestBuildBasic:
    """build() with minimal or empty context."""

    @pytest.mark.asyncio
    async def test_empty_context(self) -> None:
        builder = ResponseBuilder()
        msg = await builder.build({})
        assert isinstance(msg, OutputMessage)
        assert msg.content == "(no output)"
        assert msg.success is True
        assert msg.format == "text"

    @pytest.mark.asyncio
    async def test_none_context(self) -> None:
        builder = ResponseBuilder()
        msg = await builder.build(None)
        assert isinstance(msg, OutputMessage)
        assert msg.success is True

    @pytest.mark.asyncio
    async def test_empty_object(self) -> None:
        builder = ResponseBuilder()
        msg = await builder.build(object())
        assert isinstance(msg, OutputMessage)

    @pytest.mark.asyncio
    async def test_session_id_from_context(self) -> None:
        builder = ResponseBuilder()
        msg = await builder.build({"session_id": "sess_123"})
        assert msg.session_id == "sess_123"

    @pytest.mark.asyncio
    async def test_session_id_from_object(self) -> None:
        builder = ResponseBuilder()
        ctx = type("Ctx", (), {"session_id": "sess_abc"})()
        msg = await builder.build(ctx)
        assert msg.session_id == "sess_abc"

    @pytest.mark.asyncio
    async def test_format_from_context(self) -> None:
        builder = ResponseBuilder()
        msg = await builder.build({"response_format": "markdown"})
        assert msg.format == "markdown"

    @pytest.mark.asyncio
    async def test_deterministic_empty(self) -> None:
        builder = ResponseBuilder()
        m1 = await builder.build({})
        m2 = await builder.build({})
        assert m1.content == m2.content
        assert m1.success == m2.success

    @pytest.mark.asyncio
    async def test_success_default_true(self) -> None:
        builder = ResponseBuilder()
        msg = await builder.build({})
        assert msg.success is True

    @pytest.mark.asyncio
    async def test_error_none_when_successful(self) -> None:
        builder = ResponseBuilder()
        msg = await builder.build({})
        assert msg.error is None


# ──────────────────────────────────────────────
# build — with decision
# ──────────────────────────────────────────────


class TestBuildDecision:
    """Content and metadata from decision stage."""

    @pytest.mark.asyncio
    async def test_decision_action_in_content(self) -> None:
        builder = ResponseBuilder()
        ctx = {
            "decision": type(
                "Dec",
                (),
                {
                    "selected_action": type(
                        "Act",
                        (),
                        {
                            "action_type": "respond",
                            "description": "Reply",
                            "confidence": 0.9,
                        },
                    )()
                },
            )()
        }
        msg = await builder.build(ctx)
        assert "Action: respond" in msg.content

    @pytest.mark.asyncio
    async def test_decision_confidence_in_metadata(self) -> None:
        builder = ResponseBuilder()
        ctx = {
            "decision": type(
                "Dec",
                (),
                {
                    "selected_action": type(
                        "Act",
                        (),
                        {"action_type": "noop", "description": "", "confidence": 0.85},
                    )()
                },
            )()
        }
        msg = await builder.build(ctx)
        assert msg.metadata["decision_confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_decision_action_type_in_metadata(self) -> None:
        builder = ResponseBuilder()
        ctx = {
            "decision": type(
                "Dec",
                (),
                {
                    "selected_action": type(
                        "Act",
                        (),
                        {
                            "action_type": "escalate",
                            "description": "",
                            "confidence": 0.5,
                        },
                    )()
                },
            )()
        }
        msg = await builder.build(ctx)
        assert msg.metadata["decision_action"] == "escalate"

    @pytest.mark.asyncio
    async def test_decision_with_description(self) -> None:
        builder = ResponseBuilder()
        ctx = {
            "decision": type(
                "Dec",
                (),
                {
                    "selected_action": type(
                        "Act",
                        (),
                        {
                            "action_type": "compute",
                            "description": "Run calculation",
                            "confidence": 0.7,
                        },
                    )()
                },
            )()
        }
        msg = await builder.build(ctx)
        assert "Description: Run calculation" in msg.content

    @pytest.mark.asyncio
    async def test_decision_missing_action(self) -> None:
        builder = ResponseBuilder()
        ctx = {"decision": type("Dec", (), {"selected_action": None})()}
        msg = await builder.build(ctx)
        assert msg.success is True


# ──────────────────────────────────────────────
# build — with policy
# ──────────────────────────────────────────────


class TestBuildPolicy:
    """Policy verdict handling."""

    @pytest.mark.asyncio
    async def test_policy_allowed(self) -> None:
        builder = ResponseBuilder()
        verdict = type("V", (), {"value": "allow"})()
        ctx = {"policy_verdict": type("Pol", (), {"verdict": verdict})()}
        msg = await builder.build(ctx)
        assert msg.success is True
        assert msg.metadata.get("policy_allowed") is True

    @pytest.mark.asyncio
    async def test_policy_denied_sets_error(self) -> None:
        builder = ResponseBuilder()
        verdict = type("V", (), {"value": "deny"})()
        ctx = {
            "policy_verdict": type(
                "Pol",
                (),
                {"verdict": verdict, "denial_reason": "Not authorized"},
            )()
        }
        msg = await builder.build(ctx)
        assert msg.success is False
        assert msg.error == "Not authorized"

    @pytest.mark.asyncio
    async def test_policy_denied_content(self) -> None:
        builder = ResponseBuilder()
        verdict = type("V", (), {"value": "deny"})()
        ctx = {
            "policy_verdict": type(
                "Pol",
                (),
                {"verdict": verdict, "denial_reason": "Blocked"},
            )()
        }
        msg = await builder.build(ctx)
        assert "Error: Blocked" in msg.content

    @pytest.mark.asyncio
    async def test_policy_verdict_in_metadata(self) -> None:
        builder = ResponseBuilder()
        verdict = type("V", (), {"value": "confirm"})()
        ctx = {"policy_verdict": type("Pol", (), {"verdict": verdict})()}
        msg = await builder.build(ctx)
        assert msg.metadata["policy_verdict"] == "confirm"

    @pytest.mark.asyncio
    async def test_policy_denied_metadata_flag(self) -> None:
        builder = ResponseBuilder()
        verdict = type("V", (), {"value": "deny"})()
        ctx = {"policy_verdict": type("Pol", (), {"verdict": verdict})()}
        msg = await builder.build(ctx)
        assert msg.metadata.get("policy_denied") is True


# ──────────────────────────────────────────────
# build — with reasoning
# ──────────────────────────────────────────────


class TestBuildReasoning:
    """Reasoning trace content and metadata."""

    @pytest.mark.asyncio
    async def test_reasoning_conclusion_in_content(self) -> None:
        builder = ResponseBuilder()
        ctx = {
            "reasoning_trace": type(
                "RT",
                (),
                {"conclusion": "The answer is 42", "steps": (), "token_cost": 0},
            )()
        }
        msg = await builder.build(ctx)
        assert "Conclusion: The answer is 42" in msg.content

    @pytest.mark.asyncio
    async def test_reasoning_steps_count(self) -> None:
        builder = ResponseBuilder()
        ctx = {
            "reasoning_trace": type(
                "RT",
                (),
                {"conclusion": "x", "steps": (1, 2, 3), "token_cost": 50},
            )()
        }
        msg = await builder.build(ctx)
        assert msg.metadata["reasoning_steps"] == 3

    @pytest.mark.asyncio
    async def test_reasoning_token_cost(self) -> None:
        builder = ResponseBuilder()
        ctx = {
            "reasoning_trace": type(
                "RT",
                (),
                {"conclusion": "x", "steps": (), "token_cost": 150},
            )()
        }
        msg = await builder.build(ctx)
        assert msg.metadata["reasoning_token_cost"] == 150

    @pytest.mark.asyncio
    async def test_reasoning_strategy(self) -> None:
        builder = ResponseBuilder()
        strategy = type("S", (), {"value": "chain_of_thought"})()
        ctx = {
            "reasoning_trace": type(
                "RT",
                (),
                {
                    "conclusion": "x",
                    "steps": (),
                    "token_cost": 0,
                    "strategy_used": strategy,
                },
            )()
        }
        msg = await builder.build(ctx)
        assert msg.metadata["reasoning_strategy"] == "chain_of_thought"

    @pytest.mark.asyncio
    async def test_reasoning_no_conclusion(self) -> None:
        builder = ResponseBuilder()
        ctx = {
            "reasoning_trace": type(
                "RT",
                (),
                {"conclusion": None, "steps": (), "token_cost": 0},
            )()
        }
        msg = await builder.build(ctx)
        assert msg.metadata.get("reasoning_steps") == 0


# ──────────────────────────────────────────────
# build — with goal & intent
# ──────────────────────────────────────────────


class TestBuildGoalIntent:
    """Goal hierarchy and intent metadata."""

    @pytest.mark.asyncio
    async def test_goal_type_in_metadata(self) -> None:
        builder = ResponseBuilder()
        gt = type("GT", (), {"value": "question"})()
        root = type("Root", (), {"goal_type": gt})()
        ctx = {"goal_hierarchy": type("GH", (), {"root": root})()}
        msg = await builder.build(ctx)
        assert msg.metadata["goal_type"] == "question"

    @pytest.mark.asyncio
    async def test_intent_type_in_metadata(self) -> None:
        builder = ResponseBuilder()
        primary = type("P", (), {"value": "ask_question"})()
        ctx = {"intent": type("I", (), {"primary": primary})()}
        msg = await builder.build(ctx)
        assert msg.metadata["intent_type"] == "ask_question"

    @pytest.mark.asyncio
    async def test_intent_confidence_in_metadata(self) -> None:
        builder = ResponseBuilder()
        conf = type("C", (), {"primary": 0.95})()
        ctx = {"intent": type("I", (), {"primary": None, "confidence": conf})()}
        msg = await builder.build(ctx)
        assert msg.metadata["intent_confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_goal_missing_root(self) -> None:
        builder = ResponseBuilder()
        ctx = {"goal_hierarchy": type("GH", (), {"root": None})()}
        msg = await builder.build(ctx)
        assert "goal_type" not in msg.metadata

    @pytest.mark.asyncio
    async def test_intent_missing_primary(self) -> None:
        builder = ResponseBuilder()
        ctx = {"intent": type("I", (), {"primary": None})()}
        msg = await builder.build(ctx)
        assert "intent_type" not in msg.metadata


# ──────────────────────────────────────────────
# build — with reflection
# ──────────────────────────────────────────────


class TestBuildReflection:
    """Reflection report metadata."""

    @pytest.mark.asyncio
    async def test_reflection_score(self) -> None:
        builder = ResponseBuilder()
        ctx = {
            "reflection": type(
                "R", (), {"overall_score": 0.92, "verdict": "pass", "errors": ()}
            )()
        }
        msg = await builder.build(ctx)
        assert msg.metadata["reflection_score"] == 0.92

    @pytest.mark.asyncio
    async def test_reflection_verdict(self) -> None:
        builder = ResponseBuilder()
        ctx = {
            "reflection": type(
                "R", (), {"overall_score": 0.5, "verdict": "needs_review", "errors": ()}
            )()
        }
        msg = await builder.build(ctx)
        assert msg.metadata["reflection_verdict"] == "needs_review"

    @pytest.mark.asyncio
    async def test_reflection_error_count(self) -> None:
        builder = ResponseBuilder()
        ctx = {
            "reflection": type(
                "R", (), {"overall_score": 1.0, "verdict": "pass", "errors": (1, 2)}
            )()
        }
        msg = await builder.build(ctx)
        assert msg.metadata["reflection_error_count"] == 2


# ──────────────────────────────────────────────
# build — with experience
# ──────────────────────────────────────────────


class TestBuildExperience:
    """Experience summary metadata."""

    @pytest.mark.asyncio
    async def test_experience_count_from_dict(self) -> None:
        builder = ResponseBuilder()
        ctx = {"experience": {"total_experiences": 7}}
        msg = await builder.build(ctx)
        assert msg.metadata["experience_count"] == 7

    @pytest.mark.asyncio
    async def test_experience_count_empty_dict(self) -> None:
        builder = ResponseBuilder()
        ctx: dict[str, object] = {"experience": {}}
        msg = await builder.build(ctx)
        assert msg.metadata["experience_count"] == 0

    @pytest.mark.asyncio
    async def test_experience_none(self) -> None:
        builder = ResponseBuilder()
        ctx = {"experience": None}
        msg = await builder.build(ctx)
        assert msg.metadata.get("experience_count") is None


# ──────────────────────────────────────────────
# build — with percept
# ──────────────────────────────────────────────


class TestBuildPercept:
    """Percept user input preview."""

    @pytest.mark.asyncio
    async def test_percept_preview_in_metadata(self) -> None:
        builder = ResponseBuilder()
        ctx = {"percept": type("P", (), {"normalized_content": "Hello world"})()}
        msg = await builder.build(ctx)
        assert msg.metadata["user_input_preview"] == "Hello world"

    @pytest.mark.asyncio
    async def test_percept_preview_truncated(self) -> None:
        builder = ResponseBuilder()
        long_text = "x" * 200
        ctx = {"percept": type("P", (), {"normalized_content": long_text})()}
        msg = await builder.build(ctx)
        assert len(msg.metadata["user_input_preview"]) == 100
        assert msg.metadata["user_input_preview"].endswith("x" * 100)


# ──────────────────────────────────────────────
# build — with plan
# ──────────────────────────────────────────────


class TestBuildPlan:
    """Plan metadata."""

    @pytest.mark.asyncio
    async def test_plan_id_in_metadata(self) -> None:
        builder = ResponseBuilder()
        ctx = {"plan": type("P", (), {"id": "plan_001"})()}
        msg = await builder.build(ctx)
        assert msg.metadata["plan_id"] == "plan_001"

    @pytest.mark.asyncio
    async def test_plan_no_id(self) -> None:
        builder = ResponseBuilder()
        ctx = {"plan": type("P", (), {"id": ""})()}
        msg = await builder.build(ctx)
        assert "plan_id" not in msg.metadata


# ──────────────────────────────────────────────
# build — format rendering
# ──────────────────────────────────────────────


class TestBuildFormat:
    """Output formatting (text vs markdown)."""

    @pytest.mark.asyncio
    async def test_text_format(self) -> None:
        builder = ResponseBuilder()
        ctx = {
            "response_format": "text",
            "decision": type(
                "D",
                (),
                {
                    "selected_action": type(
                        "A",
                        (),
                        {"action_type": "reply", "description": "", "confidence": 0.5},
                    )()
                },
            )(),
        }
        msg = await builder.build(ctx)
        assert msg.format == "text"

    @pytest.mark.asyncio
    async def test_markdown_format(self) -> None:
        builder = ResponseBuilder()
        ctx = {
            "response_format": "markdown",
            "decision": type(
                "D",
                (),
                {
                    "selected_action": type(
                        "A",
                        (),
                        {"action_type": "reply", "description": "", "confidence": 0.5},
                    )()
                },
            )(),
        }
        msg = await builder.build(ctx)
        assert msg.format == "markdown"

    @pytest.mark.asyncio
    async def test_markdown_bold_action(self) -> None:
        builder = ResponseBuilder()
        ctx = {
            "response_format": "markdown",
            "decision": type(
                "D",
                (),
                {
                    "selected_action": type(
                        "A",
                        (),
                        {"action_type": "reply", "description": "", "confidence": 0.5},
                    )()
                },
            )(),
        }
        msg = await builder.build(ctx)
        assert "**Action: reply**" in msg.content

    @pytest.mark.asyncio
    async def test_markdown_conclusion_header(self) -> None:
        builder = ResponseBuilder()
        ctx = {
            "response_format": "markdown",
            "reasoning_trace": type(
                "RT",
                (),
                {"conclusion": "Done", "steps": (), "token_cost": 0},
            )(),
        }
        msg = await builder.build(ctx)
        assert "## Conclusion: Done" in msg.content

    @pytest.mark.asyncio
    async def test_markdown_description_quote(self) -> None:
        builder = ResponseBuilder()
        ctx = {
            "response_format": "markdown",
            "decision": type(
                "D",
                (),
                {
                    "selected_action": type(
                        "A",
                        (),
                        {
                            "action_type": "run",
                            "description": "Execute step",
                            "confidence": 0.5,
                        },
                    )()
                },
            )(),
        }
        msg = await builder.build(ctx)
        assert "> Description: Execute step" in msg.content


# ──────────────────────────────────────────────
# build — combined context
# ──────────────────────────────────────────────


class TestBuildCombined:
    """Full context with multiple stage outputs."""

    @pytest.mark.asyncio
    async def test_all_stages_present(self) -> None:
        builder = ResponseBuilder()
        ver = type("V", (), {"value": "allow"})()
        strat = type("S", (), {"value": "chain_of_thought"})()
        gt = type("GT", (), {"value": "question"})()
        prim = type("P", (), {"value": "ask_question"})()
        conf = type("C", (), {"primary": 0.95})()
        ctx = {
            "session_id": "sess_1",
            "decision": type(
                "D",
                (),
                {
                    "selected_action": type(
                        "A",
                        (),
                        {
                            "action_type": "respond",
                            "description": "Answer",
                            "confidence": 0.9,
                        },
                    )()
                },
            )(),
            "policy_verdict": type("Pol", (), {"verdict": ver})(),
            "reasoning_trace": type(
                "RT",
                (),
                {
                    "conclusion": "Final answer",
                    "steps": (1, 2, 3),
                    "token_cost": 100,
                    "strategy_used": strat,
                },
            )(),
            "goal_hierarchy": type(
                "GH", (), {"root": type("R", (), {"goal_type": gt})()}
            )(),
            "intent": type("I", (), {"primary": prim, "confidence": conf})(),
            "reflection": type(
                "Rf", (), {"overall_score": 0.95, "verdict": "pass", "errors": ()}
            )(),
            "experience": {"total_experiences": 5},
        }
        msg = await builder.build(ctx)
        assert msg.session_id == "sess_1"
        assert msg.success is True
        assert "Action: respond" in msg.content
        assert "Conclusion: Final answer" in msg.content
        assert msg.metadata["decision_confidence"] == 0.9
        assert msg.metadata["reasoning_steps"] == 3
        assert msg.metadata["goal_type"] == "question"
        assert msg.metadata["intent_type"] == "ask_question"
        assert msg.metadata["intent_confidence"] == 0.95
        assert msg.metadata["reflection_score"] == 0.95
        assert msg.metadata["experience_count"] == 5


# ──────────────────────────────────────────────
# build — deterministic
# ──────────────────────────────────────────────


class TestBuildDeterministic:
    """Same inputs produce same outputs."""

    @pytest.mark.asyncio
    async def test_deterministic_full(self) -> None:
        builder = ResponseBuilder()
        ctx = {
            "session_id": "s1",
            "decision": type(
                "D",
                (),
                {
                    "selected_action": type(
                        "A",
                        (),
                        {"action_type": "x", "description": "y", "confidence": 0.8},
                    )()
                },
            )(),
            "reasoning_trace": type(
                "RT",
                (),
                {"conclusion": "z", "steps": (1,), "token_cost": 10},
            )(),
        }
        m1 = await builder.build(ctx)
        m2 = await builder.build(ctx)
        assert m1.content == m2.content
        assert m1.metadata == m2.metadata
        assert m1.success == m2.success
        assert m1.session_id == m2.session_id


# ──────────────────────────────────────────────
# build — metadata
# ──────────────────────────────────────────────


class TestBuildMetadata:
    """Metadata structure."""

    @pytest.mark.asyncio
    async def test_metadata_sorted(self) -> None:
        builder = ResponseBuilder()
        ctx = {
            "decision": type(
                "D",
                (),
                {
                    "selected_action": type(
                        "A",
                        (),
                        {"action_type": "x", "description": "", "confidence": 0.5},
                    )()
                },
            )(),
            "reflection": type(
                "R", (), {"overall_score": 0.9, "verdict": "pass", "errors": ()}
            )(),
        }
        msg = await builder.build(ctx)
        keys = list(msg.metadata.keys())
        assert keys == sorted(keys)


# ──────────────────────────────────────────────
# build_chunk
# ──────────────────────────────────────────────


class TestBuildChunk:
    """Streaming chunk building."""

    @pytest.mark.asyncio
    async def test_chunk_from_string(self) -> None:
        builder = ResponseBuilder()
        chunk = await builder.build_chunk("hello")
        assert isinstance(chunk, StreamChunk)
        assert chunk.content == "hello"
        assert chunk.chunk_type == "text"
        assert chunk.finished is False

    @pytest.mark.asyncio
    async def test_chunk_from_dict(self) -> None:
        builder = ResponseBuilder()
        chunk = await builder.build_chunk(
            {
                "content": "world",
                "type": "tool_call",
                "metadata": {"key": "val"},
                "finished": True,
            }
        )
        assert chunk.content == "world"
        assert chunk.chunk_type == "tool_call"
        assert chunk.metadata == {"key": "val"}
        assert chunk.finished is True

    @pytest.mark.asyncio
    async def test_chunk_defaults(self) -> None:
        builder = ResponseBuilder()
        chunk = await builder.build_chunk({})
        assert chunk.content == ""
        assert chunk.chunk_type == "text"
        assert chunk.finished is False

    @pytest.mark.asyncio
    async def test_chunk_none(self) -> None:
        builder = ResponseBuilder()
        chunk = await builder.build_chunk(None)
        assert chunk.content == ""

    @pytest.mark.asyncio
    async def test_chunk_deterministic(self) -> None:
        builder = ResponseBuilder()
        c1 = await builder.build_chunk("data")
        c2 = await builder.build_chunk("data")
        assert c1.content == c2.content
        assert c1.chunk_type == c2.chunk_type
        assert c1.finished == c2.finished


# ──────────────────────────────────────────────
# Model serialization
# ──────────────────────────────────────────────


class TestSerialization:
    """Pydantic model serialization."""

    def test_output_message_dump(self) -> None:
        msg = OutputMessage(content="hello", session_id="s1")
        data = msg.model_dump()
        assert data["content"] == "hello"
        assert data["session_id"] == "s1"

    def test_output_message_json(self) -> None:
        msg = OutputMessage(content="hi")
        json_str = msg.model_dump_json()
        assert "hi" in json_str

    def test_output_message_deserialize(self) -> None:
        data = {"content": "test", "success": False, "format": "markdown"}
        msg = OutputMessage.model_validate(data)
        assert msg.content == "test"
        assert msg.success is False
        assert msg.format == "markdown"

    def test_stream_chunk_dump(self) -> None:
        chunk = StreamChunk(content="abc", finished=True)
        data = chunk.model_dump()
        assert data["content"] == "abc"
        assert data["finished"] is True

    def test_stream_chunk_deserialize(self) -> None:
        data = {"content": "x", "chunk_type": "error"}
        chunk = StreamChunk.model_validate(data)
        assert chunk.content == "x"
        assert chunk.chunk_type == "error"


# ──────────────────────────────────────────────
# Frozen models
# ──────────────────────────────────────────────


class TestFrozenModels:
    """Response models are immutable."""

    def test_output_message_frozen(self) -> None:
        msg = OutputMessage()
        with pytest.raises(Exception):
            msg.content = "changed"  # type: ignore[misc]

    def test_stream_chunk_frozen(self) -> None:
        chunk = StreamChunk()
        with pytest.raises(Exception):
            chunk.content = "changed"  # type: ignore[misc]


# ──────────────────────────────────────────────
# Equality
# ──────────────────────────────────────────────


class TestEquality:
    """Model equality semantics."""

    def test_output_messages_equal(self) -> None:
        a = OutputMessage(content="x", success=True)
        b = OutputMessage(content="x", success=True)
        assert a == b

    def test_output_messages_not_equal(self) -> None:
        a = OutputMessage(content="x")
        b = OutputMessage(content="y")
        assert a != b

    def test_stream_chunks_equal(self) -> None:
        a = StreamChunk(content="x", finished=True)
        b = StreamChunk(content="x", finished=True)
        assert a == b


# ──────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────


class TestEdgeCases:
    """Boundary conditions."""

    def test_output_message_defaults(self) -> None:
        msg = OutputMessage()
        assert msg.content == ""
        assert msg.format == "text"
        assert msg.metadata == {}
        assert msg.session_id == ""
        assert msg.success is True
        assert msg.error is None

    def test_stream_chunk_defaults(self) -> None:
        chunk = StreamChunk()
        assert chunk.content == ""
        assert chunk.chunk_type == "text"
        assert chunk.metadata == {}
        assert chunk.finished is False

    @pytest.mark.asyncio
    async def test_error_message_content(self) -> None:
        builder = ResponseBuilder()
        ver = type("V", (), {"value": "deny"})()
        ctx = {
            "policy_verdict": type(
                "P",
                (),
                {"verdict": ver, "denial_reason": "Safety check failed"},
            )()
        }
        msg = await builder.build(ctx)
        assert "Error: Safety check failed" in msg.content
        assert msg.success is False
        assert msg.error == "Safety check failed"

    @pytest.mark.asyncio
    async def test_error_no_reason(self) -> None:
        builder = ResponseBuilder()
        ver = type("V", (), {"value": "deny"})()
        ctx = {
            "policy_verdict": type("P", (), {"verdict": ver, "denial_reason": None})()
        }
        msg = await builder.build(ctx)
        assert msg.error == "Action denied by policy"

    @pytest.mark.asyncio
    async def test_content_empty_when_denied_no_decision(self) -> None:
        builder = ResponseBuilder()
        ver = type("V", (), {"value": "deny"})()
        ctx = {
            "policy_verdict": type("P", (), {"verdict": ver, "denial_reason": "No"})()
        }
        msg = await builder.build(ctx)
        assert "Error: No" in msg.content

    @pytest.mark.asyncio
    async def test_unknown_context_type(self) -> None:
        builder = ResponseBuilder()
        msg = await builder.build(42)
        assert isinstance(msg, OutputMessage)

    @pytest.mark.asyncio
    async def test_long_conclusion_truncated(self) -> None:
        builder = ResponseBuilder()
        long_conc = "x" * 500
        ctx = {
            "reasoning_trace": type(
                "RT",
                (),
                {"conclusion": long_conc, "steps": (), "token_cost": 0},
            )()
        }
        msg = await builder.build(ctx)
        assert long_conc in msg.content  # not truncated, just included as-is

    @pytest.mark.asyncio
    async def test_policy_allowed_with_denied_action(self) -> None:
        builder = ResponseBuilder()
        ver = type("V", (), {"value": "allow"})()
        ctx = {
            "policy_verdict": type("P", (), {"verdict": ver})(),
            "decision": type(
                "D",
                (),
                {
                    "selected_action": type(
                        "A",
                        (),
                        {"action_type": "delete", "description": "", "confidence": 0.3},
                    )()
                },
            )(),
        }
        msg = await builder.build(ctx)
        assert msg.success is True
        assert "Action: delete" in msg.content

    @pytest.mark.asyncio
    async def test_markdown_error_bold(self) -> None:
        builder = ResponseBuilder()
        ver = type("V", (), {"value": "deny"})()
        ctx = {
            "response_format": "markdown",
            "policy_verdict": type(
                "P", (), {"verdict": ver, "denial_reason": "Fail"}
            )(),
        }
        msg = await builder.build(ctx)
        assert "**Error: Fail**" in msg.content


# ──────────────────────────────────────────────
# Builder with PipelineContext-like object
# ──────────────────────────────────────────────


class TestBuildWithAttrs:
    """Context as an object with attributes rather than dict."""

    @pytest.mark.asyncio
    async def test_object_with_session_id(self) -> None:
        builder = ResponseBuilder()
        ctx = type("Ctx", (), {"session_id": "s2"})()
        msg = await builder.build(ctx)
        assert msg.session_id == "s2"

    @pytest.mark.asyncio
    async def test_object_with_decision(self) -> None:
        builder = ResponseBuilder()
        action = type(
            "A", (), {"action_type": "test", "description": "", "confidence": 1.0}
        )()
        dec = type("D", (), {"selected_action": action})()
        ctx = type("Ctx", (), {"decision": dec})()
        msg = await builder.build(ctx)
        assert "Action: test" in msg.content

    @pytest.mark.asyncio
    async def test_object_with_reasoning(self) -> None:
        builder = ResponseBuilder()
        rt = type("RT", (), {"conclusion": "done", "steps": (1, 2), "token_cost": 30})()
        ctx = type("Ctx", (), {"reasoning_trace": rt})()
        msg = await builder.build(ctx)
        assert "Conclusion: done" in msg.content
        assert msg.metadata["reasoning_steps"] == 2
