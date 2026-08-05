"""Workflow dependency graph.

Builds adjacency relationships from step ``depends_on`` declarations,
performs deterministic topological sorting, detects cycles and exposes
the set of steps that are ready to run at any point in execution.
"""

from __future__ import annotations

from typing import Iterable

from app.automation.exceptions import WorkflowValidationError
from app.automation.models import WorkflowDefinition, WorkflowStep


class WorkflowGraph:
    """Directed acyclic graph derived from a workflow definition.

    Nodes are step ids; an edge ``a -> b`` means ``b`` depends on ``a``.
    """

    def __init__(self, steps: Iterable[WorkflowStep]) -> None:
        self._nodes: list[str] = [step.id for step in steps]
        self._step_by_id: dict[str, WorkflowStep] = {step.id: step for step in steps}
        self._dependencies: dict[str, list[str]] = {
            step.id: list(step.depends_on) for step in steps
        }
        self._dependents: dict[str, list[str]] = {step.id: [] for step in steps}
        for step in steps:
            for dep in step.depends_on:
                self._dependents.setdefault(dep, []).append(step.id)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_definition(cls, definition: WorkflowDefinition) -> WorkflowGraph:
        """Build a graph from a workflow definition."""
        return cls(definition.steps)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def node_ids(self) -> list[str]:
        """Return all step ids in declaration order."""
        return list(self._nodes)

    @property
    def step_count(self) -> int:
        """Return the number of steps."""
        return len(self._nodes)

    def get(self, step_id: str) -> WorkflowStep | None:
        """Return a step by id, or ``None``."""
        return self._step_by_id.get(step_id)

    def has(self, step_id: str) -> bool:
        """Return ``True`` if *step_id* is a graph node."""
        return step_id in self._step_by_id

    def dependencies(self, step_id: str) -> list[str]:
        """Return the ids of steps *step_id* depends on."""
        return list(self._dependencies.get(step_id, []))

    def dependents(self, step_id: str) -> list[str]:
        """Return the ids of steps that depend on *step_id*."""
        return list(self._dependents.get(step_id, []))

    # ------------------------------------------------------------------
    # Topological sorting
    # ------------------------------------------------------------------

    def topological_order(self) -> list[str]:
        """Return steps in deterministic topological order.

        Uses Kahn's algorithm with stable ordering (input order) so the
        result is reproducible.  Raises :class:`WorkflowValidationError`
        when the graph contains a cycle.
        """
        in_degree: dict[str, int] = {
            node: len(self._dependencies[node]) for node in self._nodes
        }
        ready = [node for node in self._nodes if in_degree[node] == 0]
        order: list[str] = []
        ready_index = 0
        while ready_index < len(ready):
            node = ready[ready_index]
            ready_index += 1
            order.append(node)
            for dependent in self._dependents.get(node, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    ready.append(dependent)
        if len(order) != len(self._nodes):
            cycle_nodes = [node for node in self._nodes if node not in order]
            raise WorkflowValidationError(
                f"Workflow contains a cycle involving steps: {', '.join(cycle_nodes)}",
                violations=[f"cycle involving {node}" for node in cycle_nodes],
            )
        return order

    def has_cycle(self) -> bool:
        """Return ``True`` when the graph contains a cycle."""
        try:
            self.topological_order()
        except WorkflowValidationError:
            return True
        return False

    def ready_steps(self, completed: set[str]) -> list[str]:
        """Return steps whose dependencies are all in *completed*.

        Steps already completed are excluded.  Results are returned in
        declaration order for determinism.
        """
        ready: list[str] = []
        for node in self._nodes:
            if node in completed:
                continue
            if all(dep in completed for dep in self._dependencies[node]):
                ready.append(node)
        return ready

    def validate_references(self) -> list[str]:
        """Return a list of dependency reference errors (empty when valid)."""
        errors: list[str] = []
        for step in self._nodes:
            for dep in self._dependencies[step]:
                if dep not in self._step_by_id:
                    errors.append(f"Step '{step}' depends on unknown step '{dep}'")
                elif dep == step:
                    errors.append(f"Step '{step}' depends on itself")
        return errors
