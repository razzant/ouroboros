from __future__ import annotations

from starlette.applications import Starlette
from ouroboros.gateway.router import collect_routes


def test_ui_preferences_round_trip_and_normalization(tmp_path):
    from starlette.testclient import TestClient

    app = Starlette(routes=collect_routes(data_dir=tmp_path))
    app.state.drive_root = tmp_path
    with TestClient(app) as client:
        initial = client.get("/api/ui/preferences")
        assert initial.status_code == 200
        assert initial.json() == {
            "widget_order": [],
            "nested_subagents_expanded": False,
            "sidebar_width": 0,
            "project_panel_width": 0,
            "project_last_viewed": {},
            "project_hidden": {},
        }

        # v6.59.0: project_hidden merges per-project; a false flag UNHIDES (dropped).
        hide = client.post("/api/ui/preferences", json={"project_hidden": {"p1": True}})
        assert hide.status_code == 200 and hide.json()["project_hidden"] == {"p1": True}
        hide2 = client.post("/api/ui/preferences", json={"project_hidden": {"p2": True}})
        assert hide2.json()["project_hidden"] == {"p1": True, "p2": True}
        unhide = client.post("/api/ui/preferences", json={"project_hidden": {"p1": False}})
        assert unhide.json()["project_hidden"] == {"p2": True}

        # project_last_viewed MERGES per-project (a single-project update never wipes
        # the others) — drives the sidebar unread dot (v6.33.0 WS11).
        a = client.post("/api/ui/preferences", json={"project_last_viewed": {"racer": "2026-06-15T01:00:00Z"}})
        assert a.status_code == 200
        assert a.json()["project_last_viewed"] == {"racer": "2026-06-15T01:00:00Z"}
        b = client.post("/api/ui/preferences", json={"project_last_viewed": {"site": "2026-06-15T02:00:00Z"}})
        assert b.json()["project_last_viewed"] == {"racer": "2026-06-15T01:00:00Z", "site": "2026-06-15T02:00:00Z"}
        # GET reflects the merged map.
        assert client.get("/api/ui/preferences").json()["project_last_viewed"]["racer"] == "2026-06-15T01:00:00Z"

        # Resizable side-section widths round-trip and clamp (v6.33.0).
        widths = client.post(
            "/api/ui/preferences",
            json={"sidebar_width": 99999, "project_panel_width": 10},
        )
        assert widths.status_code == 200
        assert widths.json()["sidebar_width"] == 560  # clamped to max
        assert widths.json()["project_panel_width"] == 320  # clamped to min
        zero = client.post("/api/ui/preferences", json={"sidebar_width": 0})
        assert zero.status_code == 200
        assert zero.json()["sidebar_width"] == 0

        response = client.post(
            "/api/ui/preferences",
            json={
                "widget_order": ["skill:two", "skill:one", "skill:two", ""],
                "nested_subagents_expanded": False,
            },
        )
        assert response.status_code == 200
        assert response.json()["widget_order"] == ["skill:two", "skill:one"]
        assert response.json()["nested_subagents_expanded"] is False

        persisted = client.get("/api/ui/preferences")
        assert persisted.status_code == 200
        assert persisted.json()["widget_order"] == ["skill:two", "skill:one"]
        assert persisted.json()["nested_subagents_expanded"] is False

        partial_order = client.post(
            "/api/ui/preferences",
            json={"widget_order": ["skill:three"]},
        )
        assert partial_order.status_code == 200
        assert partial_order.json()["widget_order"] == ["skill:three"]
        assert partial_order.json()["nested_subagents_expanded"] is False

        partial_nested = client.post(
            "/api/ui/preferences",
            json={"nested_subagents_expanded": True},
        )
        assert partial_nested.status_code == 200
        assert partial_nested.json()["widget_order"] == ["skill:three"]
        assert partial_nested.json()["nested_subagents_expanded"] is True

        assert client.post("/api/ui/preferences", json=[]).status_code == 400
        assert client.post("/api/ui/preferences", json={"widget_order": "bad"}).status_code == 400
        assert client.post("/api/ui/preferences", json={"unknown": True}).status_code == 400
