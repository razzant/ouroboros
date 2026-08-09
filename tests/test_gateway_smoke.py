from __future__ import annotations

from starlette.applications import Starlette
from starlette.testclient import TestClient

from ouroboros.gateway.router import collect_routes
from ouroboros.gateway.state import api_health


def test_gateway_core_routes_and_health_shape(tmp_path):
    routes = collect_routes(data_dir=tmp_path)
    paths = {getattr(route, "path", "") for route in routes}
    assert "/api/health" in paths
    assert "/api/state" in paths
    assert "/api/settings" in paths
    assert "/api/ui/preferences" in paths
    assert "/ws" in paths
    assert "/api/extensions" in paths
    assert "/api/tasks/{task_id}/diff" in paths
    app = Starlette(routes=routes)
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["version"] == payload["runtime_version"]
    assert callable(api_health)


def test_task_diff_route_answers_404_for_unknown_task(tmp_path):
    """The diff endpoint's ONLY 404 is an unknown task id (mounted-route check)."""
    app = Starlette(routes=collect_routes(data_dir=tmp_path))
    app.state.drive_root = tmp_path
    app.state.repo_dir = tmp_path
    with TestClient(app) as client:
        response = client.get("/api/tasks/does-not-exist/diff")
    assert response.status_code == 404
    assert response.json()["error"] == "task not found"
