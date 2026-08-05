"""Tests for the workflow automation REST API (Phase 12e).

Exercises every workflow and schedule endpoint exposed by
:mod:`app.server.routers.workflows` through a ``TestClient``:
definition CRUD and versioning, publishing, execution, cancellation,
approval decisions, graph export, the SSE event stream and schedule
management, plus the HTTP error mapping.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.server.app import create_app
from app.server.config import ServerSettings


def _transform_definition(workflow_id: str = "wf") -> dict[str, Any]:
    return {
        "id": workflow_id,
        "version": "1.0.0",
        "name": "Transform demo",
        "inputs": [{"name": "message", "type": "str", "required": True}],
        "steps": [
            {
                "id": "t1",
                "type": "transform",
                "expression": "${inputs.message}",
            }
        ],
        "outputs": [{"name": "out", "source": "${step.t1}"}],
    }


def _approval_definition(workflow_id: str = "approval") -> dict[str, Any]:
    return {
        "id": workflow_id,
        "version": "1.0.0",
        "name": "Approval demo",
        "steps": [
            {
                "id": "a",
                "type": "approval",
                "name": "Review",
                "timeout_s": 60.0,
            },
            {
                "id": "b",
                "type": "transform",
                "expression": "${step.a.output.approved}",
                "depends_on": ["a"],
            },
        ],
    }


@pytest.fixture
def app() -> FastAPI:
    """A fresh app with rate limiting disabled for deterministic tests."""
    return create_app(settings=ServerSettings(rate_limit_enabled=False))


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """A test client bound to a fresh, isolated app."""
    return TestClient(app)


def _create_and_publish(
    client: TestClient, definition: dict[str, Any]
) -> dict[str, Any]:
    created = client.post("/workflows", json=definition)
    assert created.status_code == 200, created.text
    published = client.post(
        f"/workflows/{definition['id']}/publish",
        json={"version": definition["version"]},
    )
    assert published.status_code == 200, published.text
    return created.json()


# ----------------------------------------------------------------------
# Definition management
# ----------------------------------------------------------------------


class TestDefinitionCRUD:
    def test_create_workflow(self, client: TestClient) -> None:
        response = client.post("/workflows", json=_transform_definition())
        assert response.status_code == 200
        body = response.json()
        assert body["id"] == "wf"
        assert body["status"] == "draft"
        assert body["steps"][0]["type"] == "transform"

    def test_create_invalid_definition_rejected(self, client: TestClient) -> None:
        response = client.post("/workflows", json={"version": "1.0.0"})
        assert response.status_code == 422

    def test_create_empty_steps_is_draft_then_fails_publish(
        self, client: TestClient
    ) -> None:
        empty = {"id": "empty", "version": "1.0.0", "steps": []}
        created = client.post("/workflows", json=empty)
        assert created.status_code == 200
        assert created.json()["status"] == "draft"
        published = client.post("/workflows/empty/publish", json={"version": "1.0.0"})
        assert published.status_code == 422

    def test_get_workflow(self, client: TestClient) -> None:
        client.post("/workflows", json=_transform_definition())
        response = client.get("/workflows/wf")
        assert response.status_code == 200
        assert response.json()["id"] == "wf"

    def test_get_unknown_workflow(self, client: TestClient) -> None:
        response = client.get("/workflows/nope")
        assert response.status_code == 404

    def test_list_workflows(self, client: TestClient) -> None:
        client.post("/workflows", json=_transform_definition("a"))
        client.post("/workflows", json=_transform_definition("b"))
        response = client.get("/workflows")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 2
        assert {w["id"] for w in body["workflows"]} == {"a", "b"}

    def test_list_workflows_filters_by_status(self, client: TestClient) -> None:
        _create_and_publish(client, _transform_definition("a"))
        client.post("/workflows", json=_transform_definition("b"))
        drafts = client.get("/workflows", params={"status": "draft"}).json()
        assert {w["id"] for w in drafts["workflows"]} == {"b"}
        published = client.get("/workflows", params={"status": "published"}).json()
        assert {w["id"] for w in published["workflows"]} == {"a"}

    def test_list_workflows_invalid_status(self, client: TestClient) -> None:
        response = client.get("/workflows", params={"status": "bogus"})
        assert response.status_code == 400

    def test_versions_endpoint(self, client: TestClient) -> None:
        client.post("/workflows", json=_transform_definition())
        response = client.get("/workflows/wf/versions")
        assert response.status_code == 200
        assert response.json()["total"] == 1

    def test_versions_unknown_workflow(self, client: TestClient) -> None:
        response = client.get("/workflows/nope/versions")
        assert response.status_code == 404

    def test_create_version(self, client: TestClient) -> None:
        client.post("/workflows", json=_transform_definition())
        body = _transform_definition()
        body["version"] = "2.0.0"
        response = client.post(
            "/workflows/wf/versions",
            json={"version": "2.0.0", "definition": body},
        )
        assert response.status_code == 200
        assert response.json()["version"] == "2.0.0"

    def test_create_duplicate_published_version_conflicts(
        self, client: TestClient
    ) -> None:
        _create_and_publish(client, _transform_definition())
        response = client.post(
            "/workflows/wf/versions",
            json={"version": "1.0.0", "definition": _transform_definition()},
        )
        assert response.status_code == 409


class TestPublish:
    def test_publish_workflow(self, client: TestClient) -> None:
        client.post("/workflows", json=_transform_definition())
        response = client.post("/workflows/wf/publish", json={"version": "1.0.0"})
        assert response.status_code == 200
        assert response.json()["status"] == "published"

    def test_publish_unknown_workflow(self, client: TestClient) -> None:
        response = client.post("/workflows/nope/publish", json={"version": "1.0.0"})
        assert response.status_code == 404

    def test_publish_invalid_definition(self, client: TestClient) -> None:
        invalid = {
            "id": "bad",
            "version": "1.0.0",
            "steps": [],
        }
        client.post("/workflows", json=invalid)
        response = client.post("/workflows/bad/publish", json={"version": "1.0.0"})
        assert response.status_code == 422
        assert "violations" in response.json()

    def test_publish_unknown_version(self, client: TestClient) -> None:
        client.post("/workflows", json=_transform_definition())
        response = client.post("/workflows/wf/publish", json={"version": "9.9.9"})
        assert response.status_code == 404


# ----------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------


class TestRun:
    def test_run_published_workflow(self, client: TestClient) -> None:
        _create_and_publish(client, _transform_definition())
        response = client.post(
            "/workflows/wf/run", json={"inputs": {"message": "hello"}}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "succeeded"
        assert body["outputs"] == {"out": "hello"}

    def test_run_unpublished_workflow(self, client: TestClient) -> None:
        client.post("/workflows", json=_transform_definition())
        response = client.post("/workflows/wf/run", json={"inputs": {"message": "hi"}})
        assert response.status_code == 404

    def test_run_unknown_workflow(self, client: TestClient) -> None:
        response = client.post("/workflows/nope/run", json={})
        assert response.status_code == 404

    def test_run_specific_version(self, client: TestClient) -> None:
        _create_and_publish(client, _transform_definition("wf"))
        body = _transform_definition("wf")
        body["version"] = "2.0.0"
        body["steps"][0]["expression"] = "const"
        client.post(
            "/workflows/wf/versions", json={"version": "2.0.0", "definition": body}
        )
        client.post("/workflows/wf/publish", json={"version": "2.0.0"})
        response = client.post(
            "/workflows/wf/run",
            json={"inputs": {"message": "x"}, "version": "1.0.0"},
        )
        assert response.status_code == 200
        assert response.json()["workflow_version"] == "1.0.0"

    def test_run_idempotency_returns_existing_run(self, client: TestClient) -> None:
        _create_and_publish(client, _transform_definition())
        first = client.post(
            "/workflows/wf/run",
            json={"inputs": {"message": "a"}, "idempotency_key": "k1"},
        ).json()
        second = client.post(
            "/workflows/wf/run",
            json={"inputs": {"message": "b"}, "idempotency_key": "k1"},
        ).json()
        assert second["run_id"] == first["run_id"]
        assert second["inputs"] == {"message": "a"}


class TestRunsListing:
    def test_list_runs(self, client: TestClient) -> None:
        _create_and_publish(client, _transform_definition())
        client.post("/workflows/wf/run", json={"inputs": {"message": "x"}})
        response = client.get("/workflows/runs")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 1
        assert body["runs"][0]["workflow_id"] == "wf"

    def test_list_runs_filtered_by_workflow(self, client: TestClient) -> None:
        _create_and_publish(client, _transform_definition("wf"))
        client.post("/workflows/wf/run", json={"inputs": {"message": "x"}})
        response = client.get("/workflows/runs", params={"workflow_id": "wf"})
        assert response.json()["total"] == 1
        empty = client.get("/workflows/runs", params={"workflow_id": "other"})
        assert empty.json()["total"] == 0

    def test_list_runs_filtered_by_status(self, client: TestClient) -> None:
        _create_and_publish(client, _transform_definition("wf"))
        client.post("/workflows/wf/run", json={"inputs": {"message": "x"}})
        succeeded = client.get("/workflows/runs", params={"status": "succeeded"})
        assert succeeded.json()["total"] == 1
        failed = client.get("/workflows/runs", params={"status": "failed"})
        assert failed.json()["total"] == 0

    def test_get_run(self, client: TestClient) -> None:
        _create_and_publish(client, _transform_definition())
        run = client.post("/workflows/wf/run", json={"inputs": {"message": "x"}}).json()
        response = client.get(f"/workflows/runs/{run['run_id']}")
        assert response.status_code == 200
        assert response.json()["run_id"] == run["run_id"]

    def test_get_unknown_run(self, client: TestClient) -> None:
        response = client.get("/workflows/runs/ghost")
        assert response.status_code == 404


class TestCancel:
    def test_cancel_paused_run(self, client: TestClient) -> None:
        _create_and_publish(client, _approval_definition())
        run = client.post("/workflows/approval/run", json={}).json()
        assert run["status"] == "waiting_approval"
        response = client.post(f"/workflows/runs/{run['run_id']}/cancel")
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"

    def test_cancel_unknown_run(self, client: TestClient) -> None:
        response = client.post("/workflows/runs/ghost/cancel")
        assert response.status_code == 404

    def test_cancel_terminal_run_is_noop(self, client: TestClient) -> None:
        _create_and_publish(client, _transform_definition())
        run = client.post("/workflows/wf/run", json={"inputs": {"message": "x"}}).json()
        assert run["status"] == "succeeded"
        response = client.post(f"/workflows/runs/{run['run_id']}/cancel")
        assert response.status_code == 200
        assert response.json()["status"] == "succeeded"


# ----------------------------------------------------------------------
# Approvals
# ----------------------------------------------------------------------


class TestApprovals:
    def _pending_run(self, client: TestClient) -> dict[str, Any]:
        _create_and_publish(client, _approval_definition())
        run = client.post("/workflows/approval/run", json={}).json()
        assert run["status"] == "waiting_approval"
        return run

    def test_approve_step(self, client: TestClient) -> None:
        run = self._pending_run(client)
        response = client.post(
            f"/workflows/runs/{run['run_id']}/steps/a/approve",
            json={"decided_by": "bob", "decision_note": "ok"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["approval_requests"][0]["status"] == "approved"
        assert body["approval_requests"][0]["decided_by"] == "bob"

    def test_reject_step(self, client: TestClient) -> None:
        run = self._pending_run(client)
        response = client.post(
            f"/workflows/runs/{run['run_id']}/steps/a/reject",
            json={"decided_by": "alice"},
        )
        assert response.status_code == 200
        assert response.json()["approval_requests"][0]["status"] == "rejected"

    def test_approve_unknown_step(self, client: TestClient) -> None:
        run = self._pending_run(client)
        response = client.post(
            f"/workflows/runs/{run['run_id']}/steps/zzz/approve",
            json={"decided_by": "bob"},
        )
        assert response.status_code == 404

    def test_approve_unknown_run(self, client: TestClient) -> None:
        response = client.post(
            "/workflows/runs/ghost/steps/a/approve", json={"decided_by": "bob"}
        )
        assert response.status_code == 404

    def test_approve_twice_conflicts(self, client: TestClient) -> None:
        run = self._pending_run(client)
        url = f"/workflows/runs/{run['run_id']}/steps/a/approve"
        assert client.post(url, json={"decided_by": "bob"}).status_code == 200
        response = client.post(url, json={"decided_by": "bob"})
        assert response.status_code == 409


# ----------------------------------------------------------------------
# Graph export
# ----------------------------------------------------------------------


class TestGraphExport:
    def test_export_json(self, client: TestClient) -> None:
        client.post("/workflows", json=_transform_definition())
        response = client.get("/workflows/wf/graph")
        assert response.status_code == 200
        body = response.json()
        assert body["workflow_id"] == "wf"
        assert {node["id"] for node in body["nodes"]} == {"t1"}

    def test_export_mermaid(self, client: TestClient) -> None:
        client.post("/workflows", json=_transform_definition())
        response = client.get("/workflows/wf/graph", params={"format": "mermaid"})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert "flowchart" in response.text

    def test_export_dot(self, client: TestClient) -> None:
        client.post("/workflows", json=_transform_definition())
        response = client.get("/workflows/wf/graph", params={"format": "dot"})
        assert response.status_code == 200
        assert "digraph" in response.text

    def test_export_invalid_format(self, client: TestClient) -> None:
        client.post("/workflows", json=_transform_definition())
        response = client.get("/workflows/wf/graph", params={"format": "svg"})
        assert response.status_code == 422

    def test_export_unknown_workflow(self, client: TestClient) -> None:
        response = client.get("/workflows/nope/graph")
        assert response.status_code == 404


# ----------------------------------------------------------------------
# SSE event stream
# ----------------------------------------------------------------------


class TestEventStream:
    def test_stream_events_for_terminal_run(self, client: TestClient) -> None:
        _create_and_publish(client, _transform_definition())
        run = client.post("/workflows/wf/run", json={"inputs": {"message": "x"}}).json()
        response = client.get(f"/workflows/runs/{run['run_id']}/events")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "workflow.run.completed" in response.text
        assert "workflow.event" in response.text

    def test_stream_unknown_run(self, client: TestClient) -> None:
        response = client.get("/workflows/runs/ghost/events")
        assert response.status_code == 404


# ----------------------------------------------------------------------
# Schedules
# ----------------------------------------------------------------------


class TestSchedules:
    def test_create_schedule(self, client: TestClient) -> None:
        _create_and_publish(client, _transform_definition())
        response = client.post(
            "/schedules",
            json={
                "workflow_id": "wf",
                "trigger_type": "interval",
                "interval_seconds": 60.0,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["schedule_id"]
        assert body["workflow_id"] == "wf"
        assert body["enabled"] is True

    def test_create_schedule_with_version(self, client: TestClient) -> None:
        _create_and_publish(client, _transform_definition())
        response = client.post(
            "/schedules",
            json={
                "workflow_id": "wf",
                "workflow_version": "1.0.0",
                "trigger_type": "interval",
                "interval_seconds": 30.0,
                "payload": {"message": "scheduled"},
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["workflow_version"] == "1.0.0"
        assert body["payload"] == {"message": "scheduled"}

    def test_list_schedules(self, client: TestClient) -> None:
        _create_and_publish(client, _transform_definition())
        client.post(
            "/schedules",
            json={
                "workflow_id": "wf",
                "trigger_type": "interval",
                "interval_seconds": 60.0,
            },
        )
        response = client.get("/schedules")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 1
        assert body["schedules"][0]["workflow_id"] == "wf"

    def test_list_schedules_filters_by_enabled(self, client: TestClient) -> None:
        _create_and_publish(client, _transform_definition())
        spec = client.post(
            "/schedules",
            json={
                "workflow_id": "wf",
                "trigger_type": "interval",
                "interval_seconds": 60.0,
            },
        ).json()
        client.patch(f"/schedules/{spec['schedule_id']}", json={"enabled": False})
        enabled = client.get("/schedules", params={"enabled": True}).json()
        assert enabled["total"] == 0
        disabled = client.get("/schedules", params={"enabled": False}).json()
        assert disabled["total"] == 1

    def test_update_schedule(self, client: TestClient) -> None:
        _create_and_publish(client, _transform_definition())
        spec = client.post(
            "/schedules",
            json={
                "workflow_id": "wf",
                "trigger_type": "interval",
                "interval_seconds": 60.0,
            },
        ).json()
        response = client.patch(
            f"/schedules/{spec['schedule_id']}", json={"interval_seconds": 120.0}
        )
        assert response.status_code == 200
        assert response.json()["interval_seconds"] == 120.0

    def test_update_unknown_schedule(self, client: TestClient) -> None:
        response = client.patch("/schedules/ghost", json={"interval_seconds": 10.0})
        assert response.status_code == 404

    def test_delete_schedule(self, client: TestClient) -> None:
        _create_and_publish(client, _transform_definition())
        spec = client.post(
            "/schedules",
            json={
                "workflow_id": "wf",
                "trigger_type": "interval",
                "interval_seconds": 60.0,
            },
        ).json()
        response = client.delete(f"/schedules/{spec['schedule_id']}")
        assert response.status_code == 204
        assert client.get("/schedules").json()["total"] == 0

    def test_delete_unknown_schedule(self, client: TestClient) -> None:
        response = client.delete("/schedules/ghost")
        assert response.status_code == 404

    def test_run_schedule(self, client: TestClient) -> None:
        _create_and_publish(client, _transform_definition())
        spec = client.post(
            "/schedules",
            json={
                "workflow_id": "wf",
                "trigger_type": "interval",
                "interval_seconds": 60.0,
                "payload": {"message": "fired"},
            },
        ).json()
        response = client.post(f"/schedules/{spec['schedule_id']}/run")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "succeeded"
        assert body["outputs"] == {"out": "fired"}

    def test_run_unknown_schedule(self, client: TestClient) -> None:
        response = client.post("/schedules/ghost/run")
        assert response.status_code == 404
