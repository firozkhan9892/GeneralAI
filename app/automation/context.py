"""Execution context for workflow runs and steps.

Implements mandatory isolation: every step execution receives its own
:class:`StepExecutionContext`; parallel branches never share mutable
state.  Data flows only through immutable, completed step outputs stored
in a thread-safe :class:`OutputStore` — merging happens exclusively
through explicit ``input_bindings`` / ``output_mapping`` expressions.
"""

from __future__ import annotations

import threading
from typing import Any, Mapping

from app.automation.exceptions import WorkflowOutputConflictError
from app.automation.template import ExpressionContext


def _traverse_output(value: Any, path: str) -> Any:
    """Traverse a dot-separated *path* over a step's output value.

    The ``output`` segment is a namespace marker meaning "the step's
    output value".  When the stored value is wrapped in an ``output`` key
    (the documented convention from ``test_template``), it is unwrapped
    first; otherwise the raw stored output is used directly.  This lets
    both legacy raw outputs and the documented ``{"output": ...}`` wrapper
    resolve identically during the transition period.
    """
    parts = [part for part in path.split(".") if part]
    if parts and parts[0] == "output":
        parts = parts[1:]
        if isinstance(value, Mapping) and "output" in value:
            value = value["output"]
    for part in parts:
        if isinstance(value, Mapping):
            value = value.get(part)
        elif isinstance(value, (list, tuple)):
            try:
                value = value[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return value


class OutputStore:
    """Thread-safe store of completed step outputs.

    Steps write their own output once after execution; no step can
    mutate another step's output.  Outputs are JSON-safe (plain values)
    so they can be persisted and replayed.
    """

    def __init__(self) -> None:
        self._outputs: dict[str, Any] = {}
        self._lock = threading.RLock()

    def put(self, step_id: str, output: Any) -> None:
        """Store a step's output.

        Raises:
            WorkflowOutputConflictError: If *step_id* already has an output.
        """
        with self._lock:
            if step_id in self._outputs:
                raise WorkflowOutputConflictError(step_id)
            self._outputs[step_id] = output

    def get(self, step_id: str) -> Any:
        """Return a completed step's output, or ``None``."""
        with self._lock:
            return self._outputs.get(step_id)

    def has(self, step_id: str) -> bool:
        """Return ``True`` if *step_id* produced an output."""
        with self._lock:
            return step_id in self._outputs

    def keys(self) -> tuple[str, ...]:
        """Return step ids that produced outputs."""
        with self._lock:
            return tuple(sorted(self._outputs.keys()))

    def to_dict(self) -> dict[str, Any]:
        """Return a copy of all outputs keyed by step id."""
        with self._lock:
            return dict(self._outputs)


class WorkflowRunContext(ExpressionContext):
    """Shared, read-only context for a workflow run.

    Exposes workflow inputs and completed step outputs to expression
    resolution.  Nothing here is mutable by steps; steps only read.
    """

    def __init__(self, inputs: Mapping[str, Any], outputs: OutputStore) -> None:
        self._inputs: dict[str, Any] = dict(inputs)
        self._outputs = outputs

    # ------------------------------------------------------------------
    # ExpressionContext
    # ------------------------------------------------------------------

    def resolve_input(self, name: str) -> Any:
        return self._inputs.get(name)

    def resolve_step(self, step_id: str, path: str) -> Any:
        return _traverse_output(self._outputs.get(step_id), path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def inputs(self) -> dict[str, Any]:
        """Return a copy of the workflow inputs."""
        return dict(self._inputs)

    def output_for(self, step_id: str) -> Any:
        """Return a completed step's output, or ``None``."""
        return self._outputs.get(step_id)

    def completed_steps(self) -> tuple[str, ...]:
        """Return step ids that produced outputs."""
        return self._outputs.keys()


class StepExecutionContext:
    """Isolated execution context for a single step.

    Each step instance gets its own context holding only its resolved
    inputs plus read-only access to the shared run context.  A step
    cannot modify workflow inputs or other steps' outputs.
    """

    def __init__(
        self,
        step_id: str,
        inputs: Mapping[str, Any],
        shared: WorkflowRunContext,
    ) -> None:
        self._step_id = step_id
        self._inputs: dict[str, Any] = dict(inputs)
        self._shared = shared

    @property
    def step_id(self) -> str:
        """Return the executing step's id."""
        return self._step_id

    @property
    def inputs(self) -> dict[str, Any]:
        """Return a copy of the step's resolved inputs."""
        return dict(self._inputs)

    @property
    def shared(self) -> WorkflowRunContext:
        """Return read-only access to shared workflow data."""
        return self._shared

    def resolve_input(self, name: str) -> Any:
        """Resolve a step input by name."""
        return self._inputs.get(name)

    def resolve_step(self, step_id: str, path: str) -> Any:
        """Resolve another step's output via the shared context."""
        return self._shared.resolve_step(step_id, path)


class ScopedRunContext(WorkflowRunContext):
    """A read-only context layered over a shared run context.

    Used for nested step execution (branches, loops, parallel).  Step
    outputs are resolved from the local scope first, then from the
    shared run context — parallel branches each hold their own local
    scope and never share mutable state.
    """

    def __init__(
        self,
        shared: WorkflowRunContext,
        local_outputs: OutputStore,
        local_inputs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            local_inputs if local_inputs is not None else shared.inputs, local_outputs
        )
        self._shared = shared
        self._local = local_outputs

    def resolve_step(self, step_id: str, path: str) -> Any:
        if self._local.has(step_id):
            return _traverse_output(self._local.get(step_id), path)
        return self._shared.resolve_step(step_id, path)

    def resolve_input(self, name: str) -> Any:
        local = self._inputs.get(name)
        if local is not None:
            return local
        return self._shared.resolve_input(name)
