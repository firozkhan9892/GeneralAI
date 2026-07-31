"""Planning engine — stage 4 of the cognitive pipeline."""

from __future__ import annotations

import logging

from app.kernel.goals.models import Goal, GoalType
from app.kernel.planning.models import (
    DependencyGraph,
    Plan,
    PlanningStrategy,
    SkillStep,
)

log = logging.getLogger(__name__)


def _build_steps(goal: Goal) -> tuple[SkillStep, ...]:
    steps: list[SkillStep] = []
    goal_type = goal.goal_type

    if goal_type == GoalType.QUESTION:
        steps = [
            SkillStep(
                order=0,
                skill_name="analyze_question",
                description="Analyze the user's question",
            ),
            SkillStep(
                order=1,
                skill_name="retrieve_knowledge",
                description="Retrieve relevant knowledge",
                dependencies=(0,),
            ),
            SkillStep(
                order=2,
                skill_name="formulate_answer",
                description="Formulate the answer",
                dependencies=(1,),
            ),
        ]
    elif goal_type == GoalType.TASK:
        steps = [
            SkillStep(
                order=0,
                skill_name="understand_task",
                description="Understand the task requirements",
            ),
            SkillStep(
                order=1,
                skill_name="execute_skill",
                description="Execute the requested skill",
                dependencies=(0,),
            ),
            SkillStep(
                order=2,
                skill_name="verify_result",
                description="Verify the execution result",
                dependencies=(1,),
            ),
        ]
    elif goal_type == GoalType.PROJECT:
        steps = [
            SkillStep(
                order=0,
                skill_name="analyze_requirements",
                description="Analyze project requirements",
            ),
            SkillStep(
                order=1,
                skill_name="create_milestones",
                description="Create project milestones",
                dependencies=(0,),
            ),
            SkillStep(
                order=2,
                skill_name="assign_tasks",
                description="Assign tasks for each milestone",
                dependencies=(1,),
            ),
            SkillStep(
                order=3,
                skill_name="track_progress",
                description="Track progress against milestones",
                dependencies=(2,),
            ),
        ]
    elif goal_type == GoalType.LEARNING:
        steps = [
            SkillStep(
                order=0,
                skill_name="identify_topic",
                description="Identify the learning topic",
            ),
            SkillStep(
                order=1,
                skill_name="find_resources",
                description="Find learning resources",
                dependencies=(0,),
            ),
            SkillStep(
                order=2,
                skill_name="present_content",
                description="Present the learning content",
                dependencies=(1,),
            ),
            SkillStep(
                order=3,
                skill_name="assess_understanding",
                description="Assess user understanding",
                dependencies=(2,),
            ),
        ]
    elif goal_type == GoalType.EXPLORATION:
        steps = [
            SkillStep(
                order=0,
                skill_name="define_scope",
                description="Define the exploration scope",
            ),
            SkillStep(
                order=1,
                skill_name="gather_information",
                description="Gather relevant information",
                dependencies=(0,),
            ),
            SkillStep(
                order=2,
                skill_name="summarize_findings",
                description="Summarize key findings",
                dependencies=(1,),
            ),
        ]
    elif goal_type == GoalType.DEBUGGING:
        steps = [
            SkillStep(
                order=0,
                skill_name="reproduce_issue",
                description="Reproduce the reported issue",
            ),
            SkillStep(
                order=1,
                skill_name="analyze_logs",
                description="Analyze logs and diagnostics",
                dependencies=(0,),
            ),
            SkillStep(
                order=2,
                skill_name="identify_root_cause",
                description="Identify the root cause",
                dependencies=(1,),
            ),
            SkillStep(
                order=3,
                skill_name="apply_fix",
                description="Apply the fix",
                dependencies=(2,),
            ),
            SkillStep(
                order=4,
                skill_name="verify_fix",
                description="Verify the fix resolved the issue",
                dependencies=(3,),
            ),
        ]
    elif goal_type == GoalType.SYSTEM:
        steps = [
            SkillStep(
                order=0,
                skill_name="handle_meta_request",
                description="Handle system or meta request",
            ),
            SkillStep(
                order=1,
                skill_name="respond_to_user",
                description="Respond to the user",
                dependencies=(0,),
            ),
        ]
    else:
        steps = [
            SkillStep(
                order=0,
                skill_name="analyze_input",
                description="Analyze the user input",
            ),
            SkillStep(
                order=1,
                skill_name="determine_action",
                description="Determine the appropriate action",
                dependencies=(0,),
            ),
        ]

    return tuple(steps)


def _build_dependency_graph(steps: tuple[SkillStep, ...]) -> DependencyGraph:
    edges: list[tuple[int, int]] = []
    for step in steps:
        for dep in step.dependencies:
            edges.append((step.order, dep))
    return DependencyGraph(edges=tuple(edges))


class PlanningEngine:
    """Decomposes goals into executable plans.

    Produces an ordered sequence of skill invocations with
    dependency resolution.
    """

    async def create_plan(self, goal: Goal) -> Plan:
        """Generate a deterministic plan from a goal.

        Args:
            goal: The goal to plan for.

        Returns:
            An executable plan with ordered skill steps.
        """
        steps = _build_steps(goal)
        dependency_graph = _build_dependency_graph(steps)
        estimated_total = sum(s.estimated_tokens for s in steps)

        plan = Plan(
            goal_id=goal.id,
            strategy=PlanningStrategy.TOP_DOWN,
            steps=steps,
            dependencies=dependency_graph,
            estimated_total_tokens=estimated_total,
            revision=0,
        )

        log.info(
            "Plan created for goal %s — %d step(s), strategy=%s",
            goal.id,
            len(steps),
            plan.strategy.value,
        )

        return plan

    async def revise_plan(self, plan: Plan) -> Plan:
        """Revise a plan by incrementing its revision counter.

        Args:
            plan: The plan to revise.

        Returns:
            A new plan with incremented revision.
        """
        revised = plan.model_copy(update={"revision": plan.revision + 1})

        log.info(
            "Plan revised for goal %s — revision %d -> %d",
            revised.goal_id,
            plan.revision,
            revised.revision,
        )

        return revised

    # ── Backward-compatible aliases ─────────────────────────────────────

    async def plan(self, goal: Goal) -> Plan:
        """Alias for create_plan. Provided for backward compatibility."""
        return await self.create_plan(goal)

    async def revise(self, original_plan: Plan, failure_report: str = "") -> Plan:
        """Alias for revise_plan. Provided for backward compatibility."""
        return await self.revise_plan(original_plan)
