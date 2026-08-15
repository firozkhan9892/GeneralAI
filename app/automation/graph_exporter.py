"""Workflow graph serialization.

A single-responsibility home for the
:class:`WorkflowGraphExporter`, which renders a workflow definition into a
JSON-safe graph description (nodes, edges and a deterministic topological
order).  The exporter is intentionally free of orchestration state: it reads a
definition and returns a plain ``dict``.
"""

from __future__ import annotations

from typing import Any

from app.automation.graph import WorkflowGraph
from app.automation.models import WorkflowDefinition


class WorkflowGraphExporter:
    """Export a workflow definition to a JSON-safe graph representation.

    Produces ``nodes`` (step id/type/name), ``edges`` (dependency pairs)
    and the deterministic topological ordering for rendering or analysis.
    """

    def export(self, definition: WorkflowDefinition) -> dict[str, Any]:
        """Return a JSON-safe graph description of *definition*."""
        graph = WorkflowGraph.from_definition(definition)
        nodes = [
            {
                "id": step.id,
                "type": step.type.value,
                "name": step.name,
                "description": step.description,
            }
            for step in definition.steps
        ]
        edges = [
            {"source": dep, "target": node_id}
            for node_id in graph.node_ids
            for dep in graph.dependencies(node_id)
        ]
        return {
            "workflow_id": definition.id,
            "version": definition.version,
            "status": definition.status.value,
            "nodes": nodes,
            "edges": edges,
            "topological_order": graph.topological_order(),
        }
