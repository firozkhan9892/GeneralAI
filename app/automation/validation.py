"""Workflow definition validation.

Validation is a pure function of a :class:`WorkflowDefinition` and
produces a :class:`ValidationReport`.  A definition must pass validation
before it can be published; published versions are immutable.
"""

from __future__ import annotations

from typing import Any

from app.automation.graph import WorkflowGraph
from app.automation.models import (
    WorkflowDefinition,
    WorkflowOutput,
    WorkflowStep,
    WorkflowStepType,
)

# Step kinds that require their discriminating fields to be present
_REQUIRED_FIELDS: dict[WorkflowStepType, tuple[str, ...]] = {
    WorkflowStepType.TASK: ("tool_name",),
    WorkflowStepType.AGENT: ("agent_name",),
    WorkflowStepType.LLM: ("prompt_template",),
    WorkflowStepType.SUBWORKFLOW: ("workflow_id",),
    WorkflowStepType.TRANSFORM: ("expression",),
    WorkflowStepType.CONDITIONAL: ("branches",),
    WorkflowStepType.LOOP: ("iterable",),
    WorkflowStepType.PARALLEL: ("branches",),
    WorkflowStepType.CALLBACK: ("callback_url",),
}


class Violation:
    """A single validation finding."""

    __slots__ = ("severity", "code", "message", "step_id")

    def __init__(
        self,
        severity: str,
        code: str,
        message: str,
        step_id: str | None = None,
    ) -> None:
        self.severity = severity
        self.code = code
        self.message = message
        self.step_id = step_id

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "step_id": self.step_id,
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Violation):
            return NotImplemented
        return (
            self.severity == other.severity
            and self.code == other.code
            and self.message == other.message
            and self.step_id == other.step_id
        )

    def __hash__(self) -> int:
        return hash((self.severity, self.code, self.message, self.step_id))

    def __repr__(self) -> str:
        return (
            f"Violation(severity={self.severity!r}, code={self.code!r}, "
            f"message={self.message!r}, step_id={self.step_id!r})"
        )


class ValidationReport:
    """Collection of validation violations."""

    __slots__ = ("violations",)

    def __init__(self, violations: list[Violation] | None = None) -> None:
        self.violations: list[Violation] = violations or []

    @property
    def valid(self) -> bool:
        """Return ``True`` when there are no error-severity violations."""
        return not any(v.severity == "error" for v in self.violations)

    @property
    def errors(self) -> list[Violation]:
        """Return error-severity violations."""
        return [v for v in self.violations if v.severity == "error"]

    @property
    def warnings(self) -> list[Violation]:
        """Return warning-severity violations."""
        return [v for v in self.violations if v.severity == "warning"]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe report."""
        return {
            "valid": self.valid,
            "errors": [v.to_dict() for v in self.errors],
            "warnings": [v.to_dict() for v in self.warnings],
            "violations": [v.to_dict() for v in self.violations],
        }

    def __bool__(self) -> bool:
        return self.valid

    def __repr__(self) -> str:
        return (
            f"ValidationReport(valid={self.valid}, violations={len(self.violations)})"
        )


class WorkflowValidator:
    """Validates workflow definitions.

    Stateless; a single instance may validate any number of workflows.
    """

    def validate(self, definition: WorkflowDefinition) -> ValidationReport:
        """Validate *definition* and return a report."""
        violations: list[Violation] = []

        if not definition.id.strip():
            violations.append(self._error("empty_id", "Workflow id must not be empty"))
        if not definition.version.strip():
            violations.append(
                self._error("empty_version", "Workflow version must not be empty")
            )
        elif not _is_semver(definition.version):
            violations.append(
                self._error(
                    "invalid_version",
                    f"Workflow version '{definition.version}' is not valid semver",
                )
            )
        if not definition.steps:
            violations.append(
                self._error("no_steps", "Workflow must define at least one step")
            )

        violations.extend(self._validate_steps(definition))
        violations.extend(self._validate_graph(definition))
        violations.extend(self._validate_inputs(definition))
        violations.extend(self._validate_outputs(definition))
        return ValidationReport(violations)

    # ------------------------------------------------------------------
    # Step validation
    # ------------------------------------------------------------------

    def _validate_steps(self, definition: WorkflowDefinition) -> list[Violation]:
        violations: list[Violation] = []
        seen_ids: set[str] = set()

        def walk(step: WorkflowStep) -> None:
            if step.id in seen_ids:
                violations.append(
                    self._error(
                        "duplicate_step_id",
                        f"Step id '{step.id}' is used more than once",
                        step.id,
                    )
                )
            seen_ids.add(step.id)

            if not step.id.strip():
                violations.append(
                    self._error("empty_step_id", "Step id must not be empty")
                )
            if not step.name and step.type != WorkflowStepType.TRANSFORM:
                violations.append(
                    self._warning(
                        "missing_step_name",
                        f"Step '{step.id}' has no name",
                        step.id,
                    )
                )

            required = _REQUIRED_FIELDS.get(step.type, ())
            for field in required:
                value = getattr(step, field)
                if not value:
                    violations.append(
                        self._error(
                            "missing_required_field",
                            f"Step '{step.id}' of type '{step.type.value}' requires "
                            f"'{field}'",
                            step.id,
                        )
                    )

            if step.type in (WorkflowStepType.CONDITIONAL, WorkflowStepType.PARALLEL):
                if not step.branches:
                    violations.append(
                        self._error(
                            "no_branches",
                            f"Step '{step.id}' defines no branches",
                            step.id,
                        )
                    )
                branch_names: set[str] = set()
                for branch in step.branches:
                    if branch.name in branch_names:
                        violations.append(
                            self._error(
                                "duplicate_branch_name",
                                f"Step '{step.id}' has duplicate branch "
                                f"'{branch.name}'",
                                step.id,
                            )
                        )
                    branch_names.add(branch.name)
                    for sub in branch.steps:
                        walk(sub)

            if step.type == WorkflowStepType.LOOP:
                for sub in step.loop_steps:
                    walk(sub)

        for step in definition.steps:
            walk(step)
        return violations

    # ------------------------------------------------------------------
    # Graph validation
    # ------------------------------------------------------------------

    def _validate_graph(self, definition: WorkflowDefinition) -> list[Violation]:
        violations: list[Violation] = []
        graph = WorkflowGraph(definition.steps)

        for message in graph.validate_references():
            step_id = message.split("'")[1]
            violations.append(self._error("invalid_dependency", message, step_id))

        if graph.has_cycle():
            violations.append(
                self._error(
                    "cycle",
                    "Workflow dependency graph contains a cycle",
                )
            )
        return violations

    # ------------------------------------------------------------------
    # Input/output validation
    # ------------------------------------------------------------------

    def _validate_inputs(self, definition: WorkflowDefinition) -> list[Violation]:
        violations: list[Violation] = []
        names: set[str] = set()
        for item in definition.inputs:
            if item.name in names:
                violations.append(
                    self._error(
                        "duplicate_input",
                        f"Input '{item.name}' is declared more than once",
                    )
                )
            names.add(item.name)
            if not item.name.strip():
                violations.append(
                    self._error("empty_input", "Input name must not be empty")
                )
        return violations

    def _validate_outputs(self, definition: WorkflowDefinition) -> list[Violation]:
        violations: list[Violation] = []
        names: set[str] = set()
        step_ids = {step.id for step in definition.steps}
        for item in definition.outputs:
            if item.name in names:
                violations.append(
                    self._error(
                        "duplicate_output",
                        f"Output '{item.name}' is declared more than once",
                    )
                )
            names.add(item.name)
            if not item.name.strip():
                violations.append(
                    self._error("empty_output", "Output name must not be empty")
                )
            violations.extend(self._validate_output_source(item, step_ids))
        return violations

    def _validate_output_source(
        self, output: WorkflowOutput, step_ids: set[str]
    ) -> list[Violation]:
        """Validate that an output source references a known step."""
        if not output.source:
            return [
                self._warning(
                    "empty_output_source", f"Output '{output.name}' has no source"
                )
            ]
        if not output.source.startswith("${"):
            return []
        body = output.source[2:-1]
        parts = body.split(".")
        if parts and parts[0] == "step" and len(parts) >= 2:
            step_id = parts[1]
            if step_id not in step_ids:
                return [
                    self._error(
                        "invalid_output_source",
                        f"Output '{output.name}' references unknown step '{step_id}'",
                    )
                ]
        return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _error(self, code: str, message: str, step_id: str | None = None) -> Violation:
        return Violation("error", code, message, step_id)

    def _warning(
        self, code: str, message: str, step_id: str | None = None
    ) -> Violation:
        return Violation("warning", code, message, step_id)


def _is_semver(version: str) -> bool:
    """Return ``True`` for ``MAJOR.MINOR.PATCH``-style versions."""
    parts = version.split(".")
    if len(parts) < 3:
        return False
    return all(part.isdigit() for part in parts[:3])
