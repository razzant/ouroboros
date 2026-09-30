from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from starlette.applications import Starlette

from ouroboros.gateway.router import collect_routes


def test_unreadable_stored_preferences_never_become_empty_layout(tmp_path):
    from starlette.testclient import TestClient

    app = Starlette(routes=collect_routes(data_dir=tmp_path))
    app.state.drive_root = tmp_path
    path = tmp_path / "state" / "ui_preferences.json"
    path.parent.mkdir(parents=True)
    invalid = b'{"widget_layout":'
    path.write_bytes(invalid)
    with TestClient(app) as client:
        assert client.get("/api/ui/preferences").status_code == 503
        assert client.post("/api/ui/preferences", json={"widget_layout": {"visible": {"x": 0, "y": 0, "w": 4, "h": 8}}}).status_code == 503
        assert path.read_bytes() == invalid
        path.write_text(json.dumps({"widget_layout": {"hidden": {"x": 4, "y": 0, "w": 4, "h": 8}}}), encoding="utf-8")
        assert "hidden" in client.get("/api/ui/preferences").json()["widget_layout"]


def test_ui_preferences_round_trip_and_normalization(tmp_path):
    from starlette.testclient import TestClient

    from ouroboros.projects_registry import create_project, increment_project_visible_revision

    app = Starlette(routes=collect_routes(data_dir=tmp_path))
    app.state.drive_root = tmp_path
    with TestClient(app) as client:
        initial = client.get("/api/ui/preferences")
        assert initial.status_code == 200
        assert initial.json() == {
            "widget_order": [],
            "widget_start_mode": {},
            "widget_layout": {},
            "nested_subagents_expanded": False,
            "sidebar_width": 0,
            "project_panel_width": 0,
            "project_seen_revision": {},
            "welcome": {"mode": "default", "text": ""},
        }

        create_project(tmp_path, "racer", name="Racer")
        create_project(tmp_path, "site", name="Site")
        increment_project_visible_revision(tmp_path, project_id="racer")
        increment_project_visible_revision(tmp_path, project_id="racer")
        increment_project_visible_revision(tmp_path, project_id="site")

        # Paint ACKs merge monotonically. A future value is clamped to the current
        # visible revision; stale tabs cannot move a cursor backwards.
        a = client.post("/api/ui/preferences", json={"project_seen_revision": {"racer": 1}})
        assert a.status_code == 200
        assert a.json()["project_seen_revision"] == {"racer": 1}
        b = client.post("/api/ui/preferences", json={"project_seen_revision": {"site": 999}})
        assert b.json()["project_seen_revision"] == {"racer": 1, "site": 1}
        stale = client.post("/api/ui/preferences", json={"project_seen_revision": {"racer": 0}})
        assert stale.json()["project_seen_revision"]["racer"] == 1
        future = client.post("/api/ui/preferences", json={"project_seen_revision": {"racer": 999}})
        assert future.json()["project_seen_revision"]["racer"] == 2
        unknown = client.post("/api/ui/preferences", json={"project_seen_revision": {"missing": 8}})
        assert "missing" not in unknown.json()["project_seen_revision"]
        assert client.get("/api/ui/preferences").json()["project_seen_revision"]["racer"] == 2

        # ABI 7.0 (ABI-3): the one-minor deprecation window is CLOSED — the
        # retired keys answer the ordinary unknown-key 400 and never appear in
        # any response payload.
        legacy = client.post(
            "/api/ui/preferences",
            json={"project_hidden": {"racer": True}},
        )
        assert legacy.status_code == 400
        assert "project_hidden" in legacy.json()["error"]
        current = client.get("/api/ui/preferences").json()
        assert "project_hidden" not in current and "project_last_viewed" not in current
        # A STORED legacy file still loads: unknown stored keys are ignored on
        # read and dropped on the next write, never fatal.
        import json as _json
        prefs_path = tmp_path / "state" / "ui_preferences.json"
        stored_now = _json.loads(prefs_path.read_text(encoding="utf-8"))
        stored_now["project_last_viewed"] = {"racer": "2026-06-15T01:00:00Z"}
        prefs_path.write_text(_json.dumps(stored_now), encoding="utf-8")
        tolerated = client.get("/api/ui/preferences")
        assert tolerated.status_code == 200
        assert "project_last_viewed" not in tolerated.json()

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
        assert client.post("/api/ui/preferences", json={"project_seen_revision": {"racer": "bad"}}).status_code == 400
        assert client.post("/api/ui/preferences", json={"unknown": True}).status_code == 400


def test_empty_chat_welcome_preference_round_trip_and_refusals(tmp_path):
    from starlette.testclient import TestClient

    app = Starlette(routes=collect_routes(data_dir=tmp_path))
    app.state.drive_root = tmp_path
    with TestClient(app) as client:
        assert client.get("/api/ui/preferences").json()["welcome"] == {"mode": "default", "text": ""}
        custom = {"mode": "custom", "text": "Привет <b>мир</b>\nagain"}
        response = client.post("/api/ui/preferences", json={"welcome": custom})
        assert response.status_code == 200
        assert response.json()["welcome"] == custom
        assert client.get("/api/ui/preferences").json()["welcome"] == custom
        for invalid in ({"mode": "custom", "text": "  "}, {"mode": "custom", "text": "x" * 501},
                        {"mode": "other", "text": "x"}, {"mode": "custom"}, None):
            assert client.post("/api/ui/preferences", json={"welcome": invalid}).status_code == 400
            assert client.get("/api/ui/preferences").json()["welcome"] == custom
        hidden = {"mode": "hidden", "text": custom["text"]}
        assert client.post("/api/ui/preferences", json={"welcome": hidden}).json()["welcome"] == hidden
        assert client.post("/api/ui/preferences", json={"widget_order": ["skill:x"]}).json()["welcome"] == hidden
        default = {"mode": "default", "text": custom["text"]}
        assert client.post("/api/ui/preferences", json={"welcome": default}).json()["welcome"] == default


def test_hand_edited_welcome_is_read_without_disturbing_other_keys(tmp_path):
    """`welcome` has no Settings control: the owner edits the file (docs/DESIGN.md)."""
    from starlette.testclient import TestClient

    app = Starlette(routes=collect_routes(data_dir=tmp_path))
    app.state.drive_root = tmp_path
    path = tmp_path / "state" / "ui_preferences.json"
    path.parent.mkdir(parents=True)
    with TestClient(app) as client:
        for welcome in ({"mode": "hidden", "text": ""}, {"mode": "custom", "text": "Доброе утро"},
                        {"mode": "default", "text": ""}):
            path.write_text(json.dumps({"sidebar_width": 300, "welcome": welcome}), encoding="utf-8")
            prefs = client.get("/api/ui/preferences").json()
            assert prefs["welcome"] == welcome and prefs["sidebar_width"] == 300
        # A value the POST contract refuses reads as the default instead of resetting every
        # other key, and cannot block their writes; the next write stores the default.
        for invalid in ({"mode": "Hidden", "text": ""}, {"mode": "custom", "text": " "},
                        {"mode": "custom", "text": "x" * 501}, {"mode": "custom"},
                        {"mode": "hidden", "text": "", "extra": 1}, "hidden", None):
            path.write_text(json.dumps({"sidebar_width": 300, "welcome": invalid}), encoding="utf-8")
            prefs = client.get("/api/ui/preferences").json()
            assert prefs["welcome"] == {"mode": "default", "text": ""} and prefs["sidebar_width"] == 300
        response = client.post("/api/ui/preferences", json={"nested_subagents_expanded": True})
        assert response.status_code == 200 and response.json()["sidebar_width"] == 300
        stored = json.loads(path.read_text(encoding="utf-8"))
        assert stored["welcome"] == {"mode": "default", "text": ""} and stored["nested_subagents_expanded"] is True


def test_welcome_text_with_a_lone_surrogate_is_refused_and_read_as_default(tmp_path):
    """JSON's "\\ud800" escape parses to a str UTF-8 cannot encode; it never reaches a response or the file."""
    from starlette.testclient import TestClient

    app = Starlette(routes=collect_routes(data_dir=tmp_path))
    app.state.drive_root = tmp_path
    path = tmp_path / "state" / "ui_preferences.json"
    path.parent.mkdir(parents=True)
    lone = {"mode": "custom", "text": "Hello \ud800 there"}
    # json.dumps escapes non-ASCII, so the file carries the astral emoji as a "🌅" pair.
    normal = {"mode": "custom", "text": "Доброе утро 🌅 中文"}
    with TestClient(app) as client:
        path.write_text(json.dumps({"sidebar_width": 300, "welcome": normal}), encoding="utf-8")
        assert client.get("/api/ui/preferences").json()["welcome"] == normal
        response = client.post("/api/ui/preferences", json={"welcome": normal})
        assert response.status_code == 200 and response.json()["welcome"] == normal
        stored = path.read_bytes()
        response = client.post("/api/ui/preferences", content=json.dumps({"welcome": lone}),
                               headers={"Content-Type": "application/json"})
        assert response.status_code == 400
        assert response.json()["error"] == "welcome text must be valid Unicode"
        assert path.read_bytes() == stored
        # A hand edit carrying one reads as the default without costing the other keys,
        # and an unrelated write succeeds and stores the default.
        path.write_text(json.dumps({"sidebar_width": 300, "welcome": lone}), encoding="utf-8")
        prefs = client.get("/api/ui/preferences").json()
        assert prefs["welcome"] == {"mode": "default", "text": ""} and prefs["sidebar_width"] == 300
        response = client.post("/api/ui/preferences", json={"nested_subagents_expanded": True})
        assert response.status_code == 200, response.text
        assert response.json()["sidebar_width"] == 300
        stored = json.loads(path.read_text(encoding="utf-8"))
        assert stored["welcome"] == {"mode": "default", "text": ""}
        assert stored["sidebar_width"] == 300 and stored["nested_subagents_expanded"] is True


def test_ui_preferences_concurrent_paint_acks_are_monotonic(tmp_path):
    from ouroboros.gateway.ui_preferences import api_ui_preferences_post
    from ouroboros.projects_registry import create_project, increment_project_visible_revision

    create_project(tmp_path, "race", name="Race")
    for _ in range(5):
        increment_project_visible_revision(tmp_path, project_id="race")
    barrier = threading.Barrier(2)

    def _post(revision: int) -> int:
        async def _json():
            barrier.wait(timeout=5)
            return {"project_seen_revision": {"race": revision}}

        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(drive_root=tmp_path)),
            json=_json,
        )
        return asyncio.run(api_ui_preferences_post(request)).status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(_post, (2, 5)))
    assert statuses == [200, 200]
    stored = json.loads((tmp_path / "state" / "ui_preferences.json").read_text(encoding="utf-8"))
    assert stored["project_seen_revision"]["race"] == 5


def test_ui_preferences_widget_start_mode_override(tmp_path):
    """Owner per-card launch-policy override: bounds only, whole-map replace, stale keys kept."""
    from starlette.testclient import TestClient

    from ouroboros.extension_ui_validation import WIDGET_START_MODES

    app = Starlette(routes=collect_routes(data_dir=tmp_path))
    app.state.drive_root = tmp_path
    with TestClient(app) as client:
        assert client.get("/api/ui/preferences").json()["widget_start_mode"] == {}

        # Round trip. Keys are NOT checked against live widgets: a key of a disabled or
        # removed skill is kept on purpose so a temporary disable never loses the choice.
        stored = {"game:main": "retain", "gone_skill:old": "manual", "gauge:live": "auto"}
        saved = client.post("/api/ui/preferences", json={"widget_start_mode": stored})
        assert saved.status_code == 200
        assert saved.json()["widget_start_mode"] == stored
        assert client.get("/api/ui/preferences").json()["widget_start_mode"] == stored

        # POST replaces the whole map (widget_order semantics); other keys are untouched.
        replaced = client.post("/api/ui/preferences", json={"widget_start_mode": {"game:main": "manual"}})
        assert replaced.status_code == 200
        assert replaced.json()["widget_start_mode"] == {"game:main": "manual"}
        other = client.post("/api/ui/preferences", json={"widget_order": ["game:main"]})
        assert other.json()["widget_start_mode"] == {"game:main": "manual"}
        assert other.json()["widget_order"] == ["game:main"]

        # Values come from the validator's enum; any other shape is a 400 and stores nothing.
        for bad in (
            {"widget_start_mode": {"game:main": "always"}},
            {"widget_start_mode": {"game:main": 1}},
            {"widget_start_mode": {"game:main": None}},
            {"widget_start_mode": ["game:main"]},
            {"widget_start_mode": "retain"},
            {"widget_start_modes": {}},  # unknown key stays 400
        ):
            assert client.post("/api/ui/preferences", json=bad).status_code == 400, bad
        assert client.get("/api/ui/preferences").json()["widget_start_mode"] == {"game:main": "manual"}

        # Blank or oversized keys are dropped; whitespace around keys and values is trimmed
        # (the validator trims render.start the same way); null clears the map.
        trimmed = client.post(
            "/api/ui/preferences",
            json={"widget_start_mode": {"": "auto", "x" * 201: "auto", " game:main ": "retain", "gauge:live": " auto "}},
        )
        assert trimmed.json()["widget_start_mode"] == {"game:main": "retain", "gauge:live": "auto"}
        assert client.post("/api/ui/preferences", json={"widget_start_mode": None}).json()["widget_start_mode"] == {}

        # Bounded like widget_order: at most 200 entries are kept.
        many = {f"skill:{i}": WIDGET_START_MODES[i % len(WIDGET_START_MODES)] for i in range(250)}
        bounded = client.post("/api/ui/preferences", json={"widget_start_mode": many})
        assert bounded.status_code == 200
        kept = bounded.json()["widget_start_mode"]
        assert len(kept) == 200
        assert all(many[key] == mode for key, mode in kept.items())


def test_ui_preferences_widget_layout_cells(tmp_path):
    """Owner Widgets grid cells: persisted per card key, whole-map replace, clamped
    into the grid, stale keys kept (a disabled skill's card returns to its place),
    bounded like the other per-card maps."""
    from starlette.testclient import TestClient

    app = Starlette(routes=collect_routes(data_dir=tmp_path))
    app.state.drive_root = tmp_path
    with TestClient(app) as client:
        assert client.get("/api/ui/preferences").json()["widget_layout"] == {}

        layout = {
            "game:main": {"x": 0, "y": 0, "w": 8, "h": 11},
            "gauge:live": {"x": 8, "y": 0, "w": 4, "h": 8},
            "gone_skill:old": {"x": 0, "y": 11, "w": 12, "h": 4},
        }
        saved = client.post(
            "/api/ui/preferences",
            json={"widget_layout": layout, "widget_order": ["game:main", "gauge:live"]},
        )
        assert saved.status_code == 200
        assert saved.json()["widget_layout"] == layout
        assert saved.json()["widget_order"] == ["game:main", "gauge:live"]
        # A revisit (and a restart: the file is the store) reads the same cells back.
        assert client.get("/api/ui/preferences").json()["widget_layout"] == layout
        stored = json.loads((tmp_path / "state" / "ui_preferences.json").read_text(encoding="utf-8"))
        assert stored["widget_layout"] == layout

        # Other keys leave the layout alone; a layout write replaces the whole map.
        other = client.post("/api/ui/preferences", json={"widget_start_mode": {"game:main": "retain"}})
        assert other.json()["widget_layout"] == layout
        replaced = client.post("/api/ui/preferences", json={"widget_layout": {"game:main": {"x": 2, "y": 3, "w": 6, "h": 6}}})
        assert replaced.json()["widget_layout"] == {"game:main": {"x": 2, "y": 3, "w": 6, "h": 6}}
        assert replaced.json()["widget_start_mode"] == {"game:main": "retain"}

        # Out-of-range cells clamp into the grid (x follows the clamped width);
        # keys are trimmed, blank / oversized keys and extra fields are dropped.
        clamped = client.post(
            "/api/ui/preferences",
            json={"widget_layout": {
                " wide:card ": {"x": 11, "y": -5, "w": 99, "h": 0, "z": 7},
                "tiny:card": {"x": 11, "y": 20000, "w": 1, "h": 500},
                "": {"x": 0, "y": 0, "w": 4, "h": 4},
                "x" * 201: {"x": 0, "y": 0, "w": 4, "h": 4},
            }},
        )
        assert clamped.status_code == 200
        assert clamped.json()["widget_layout"] == {
            "wide:card": {"x": 0, "y": 0, "w": 12, "h": 4},
            "tiny:card": {"x": 9, "y": 10000, "w": 3, "h": 48},
        }

        # Any other shape is a 400 and stores nothing.
        for bad in (
            {"widget_layout": ["game:main"]},
            {"widget_layout": "grid"},
            {"widget_layout": {"game:main": [0, 0, 4, 4]}},
            {"widget_layout": {"game:main": {"x": 0, "y": 0, "w": 4}}},
            {"widget_layout": {"game:main": {"x": "0", "y": 0, "w": 4, "h": 4}}},
            {"widget_layout": {"game:main": {"x": 0.5, "y": 0, "w": 4, "h": 4}}},
            {"widget_layout": {"game:main": {"x": True, "y": 0, "w": 4, "h": 4}}},
            {"widget_layouts": {}},
        ):
            assert client.post("/api/ui/preferences", json=bad).status_code == 400, bad
        assert set(client.get("/api/ui/preferences").json()["widget_layout"]) == {"wide:card", "tiny:card"}

        assert client.post("/api/ui/preferences", json={"widget_layout": None}).json()["widget_layout"] == {}
        many = {f"skill:{i}": {"x": 0, "y": i * 4, "w": 4, "h": 4} for i in range(250)}
        bounded = client.post("/api/ui/preferences", json={"widget_layout": many})
        assert bounded.status_code == 200
        assert list(bounded.json()["widget_layout"]) == list(many)[:200]


def test_widget_grid_bounds_have_one_value_in_python_and_the_browser():
    """The server clamps with the same bounds the browser grid plans with."""
    import re
    from pathlib import Path

    from ouroboros.gateway import ui_preferences as prefs

    source = (Path(__file__).resolve().parent.parent / "web" / "modules" / "widget_grid.js").read_text(encoding="utf-8")

    def js(name: str) -> int:
        match = re.search(rf"export const {name} = (\d+);", source)
        assert match, name
        return int(match.group(1))

    assert js("WIDGET_GRID_COLUMNS") == prefs.WIDGET_GRID_COLUMNS
    assert js("WIDGET_GRID_MIN_W") == prefs.WIDGET_GRID_MIN_W
    assert js("WIDGET_GRID_MIN_H") == prefs.WIDGET_GRID_MIN_H
    assert js("WIDGET_GRID_MAX_H") == prefs.WIDGET_GRID_MAX_H
    assert js("WIDGET_GRID_MAX_Y") == prefs.WIDGET_GRID_MAX_Y
    assert js("WIDGET_LAYOUT_MAX_ITEMS") == prefs._MAX_WIDGET_LAYOUT_ITEMS
