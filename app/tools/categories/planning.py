"""Plan-matching tools for the deterministic agent plans.

The cognitive planning engine produces plans whose skill names are
deterministic per goal type (``analyze_question``, ``understand_task``,
…).  These pass-through tools expose those names in the tool registry so
the agent loop can match them exactly and execute every step.  Each tool
accepts optional ``text`` input and echoes it back, or returns a short
completion notice.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.models import ToolCategory, ToolParameter

#: The deterministic skill names produced by the planning engine.
PLAN_SKILL_NAMES: tuple[str, ...] = (
    # GoalType.QUESTION
    "analyze_question",
    "retrieve_knowledge",
    "formulate_answer",
    # GoalType.TASK
    "understand_task",
    "execute_skill",
    "verify_result",
    # GoalType.PROJECT
    "analyze_requirements",
    "create_milestones",
    "assign_tasks",
    "track_progress",
    # GoalType.LEARNING
    "identify_topic",
    "find_resources",
    "present_content",
    "assess_understanding",
    # GoalType.EXPLORATION
    "define_scope",
    "gather_information",
    "summarize_findings",
    # GoalType.DEBUGGING
    "reproduce_issue",
    "analyze_logs",
    "identify_root_cause",
    "apply_fix",
    "verify_fix",
    # GoalType.SYSTEM
    "handle_meta_request",
    "respond_to_user",
    # Fallback / unknown
    "analyze_input",
    "determine_action",
)

_PLAN_DESCRIPTIONS: dict[str, str] = {
    "analyze_question": "Analyze the user's question",
    "retrieve_knowledge": "Retrieve relevant knowledge",
    "formulate_answer": "Formulate the answer",
    "understand_task": "Understand the task requirements",
    "execute_skill": "Execute the requested skill",
    "verify_result": "Verify the execution result",
    "analyze_requirements": "Analyze project requirements",
    "create_milestones": "Create project milestones",
    "assign_tasks": "Assign tasks for each milestone",
    "track_progress": "Track progress against milestones",
    "identify_topic": "Identify the learning topic",
    "find_resources": "Find learning resources",
    "present_content": "Present the learning content",
    "assess_understanding": "Assess user understanding",
    "define_scope": "Define the exploration scope",
    "gather_information": "Gather relevant information",
    "summarize_findings": "Summarize key findings",
    "reproduce_issue": "Reproduce the reported issue",
    "analyze_logs": "Analyze logs and diagnostics",
    "identify_root_cause": "Identify the root cause",
    "apply_fix": "Apply the fix",
    "verify_fix": "Verify the fix resolved the issue",
    "handle_meta_request": "Handle system or meta request",
    "respond_to_user": "Respond to the user",
    "analyze_input": "Analyze the user input",
    "determine_action": "Determine the appropriate action",
}


class PlanPassThroughTool(Tool):
    """A pass-through tool bound to a single plan skill name.

    Args:
        name: The skill/tool name to expose in the registry.
        description: Human-readable description of the step.
    """

    description = "Pass-through tool for a deterministic plan step"
    category = ToolCategory.BUILTIN
    parameters = (
        ToolParameter(
            name="text",
            description="Optional input text to pass through",
            param_type="string",
            default="",
        ),
    )

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description or self.description

    def run(
        self, arguments: Mapping[str, Any], context: ToolContext | None = None
    ) -> Any:
        return arguments.get("text") or f"Completed {self.name}"


def plan_tools() -> tuple[Tool, ...]:
    """Return one pass-through tool per deterministic plan skill name.

    Returns:
        A tuple of tools whose names match the planning engine's steps.
    """
    return tuple(
        PlanPassThroughTool(
            name=name,
            description=_PLAN_DESCRIPTIONS.get(name, f"Plan step: {name}"),
        )
        for name in PLAN_SKILL_NAMES
    )
