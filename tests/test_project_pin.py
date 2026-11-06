"""Persistent Project pins use the existing partial update and sidebar projection."""
from concurrent.futures import ThreadPoolExecutor
import json
import pathlib
import re
import typing

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from ouroboros.gateway.client_contracts import ProjectUpdateRequest
from ouroboros.gateway.contracts import ProjectEntry
from ouroboros.gateway.router import collect_routes
from ouroboros.gateway.schema import validate_ingress
from ouroboros import projects_registry as registry
from tests.test_gateway_parity import _js_typedef_fields, _js_typedef_properties

WEB_MODULES = pathlib.Path(__file__).resolve().parents[1] / "web" / "modules"


@pytest.fixture
def pin_client(tmp_path):
    app = Starlette(routes=collect_routes(data_dir=tmp_path))
    app.state.drive_root = tmp_path
    with TestClient(app) as client:
        yield client, tmp_path


def test_pin_partial_update_persists_without_changing_room_authority(pin_client):
    client, root = pin_client
    before = registry.create_project(root, "alpha", name="Alpha", working_dir="/example")
    saved = client.post("/api/projects/alpha/update", json={"pinned": True})
    assert saved.status_code == 200
    pinned = saved.json()["project"]
    for key in ("name", "working_dir", "chat_id", "routing_generation", "routing_incarnation", "last_active_at", "visible_revision"):
        assert pinned[key] == before[key]
    assert pinned["pinned"] is True
    assert client.get("/api/projects").json()["projects"][0]["pinned"] is True
    assert client.post("/api/projects/alpha/update", json={"name": "Renamed"}).json()["project"]["pinned"] is True
    assert client.post("/api/projects/alpha/update", json={"pinned": False}).json()["project"]["name"] == "Renamed"
    assert registry.get_project(root, "alpha")["pinned"] is False


@pytest.mark.parametrize("value", ["true", 0, {}])
def test_pin_is_strict_boolean_and_refuses_atomically(pin_client, value):
    client, root = pin_client
    registry.create_project(root, "alpha", name="Original")
    path = root / "state/projects.json"
    before = path.read_bytes()
    response = client.post("/api/projects/alpha/update", json={"name": "Should not save", "pinned": value})
    assert response.status_code == 400
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="boolean"):
        registry.update_project(root, "alpha", pinned=value)
    assert path.read_bytes() == before


def test_pin_missing_deleting_and_racing_delete_are_not_success(pin_client, monkeypatch):
    client, root = pin_client
    assert client.post("/api/projects/missing/update", json={"pinned": True}).status_code == 404
    registry.create_project(root, "alpha")
    original = registry.update_project

    def delete_first(*args, **kwargs):
        registry.begin_project_deletion(root, "alpha")
        return original(*args, **kwargs)

    monkeypatch.setattr(registry, "update_project", delete_first)
    assert client.post("/api/projects/alpha/update", json={"pinned": True}).status_code == 409
    assert registry.get_reserved_project(root, "alpha")["pinned"] is False
    assert client.post("/api/projects/alpha/update", json={"pinned": True}).status_code == 404


def test_empty_or_invalid_updates_preserve_pin(pin_client):
    client, root = pin_client
    registry.create_project(root, "alpha")
    registry.update_project(root, "alpha", pinned=True)
    for body in ({}, {"other": True}, {"name": ""}, {"name": "x" * 81}):
        assert client.post("/api/projects/alpha/update", json=body).status_code == 400
        assert registry.get_project(root, "alpha")["pinned"] is True


def test_pins_do_not_exhaust_recent_slice_or_disappear_at_cap(pin_client):
    client, root = pin_client
    # One registry write seeds scale without making the test about creation cost.
    registry.create_project(root, "seed", origin="owner_ui")
    path = root / "state/projects.json"
    data = json.loads(path.read_text())
    template = data["projects"][0]
    data["projects"] = [{**template, "id": f"pin-{i}", "chat_id": 10000 + i,
                         "pinned": True, "last_active_at": "2026-01-01T00:00:00Z"}
                        for i in range(55)]
    data["projects"] += [{**template, "id": f"recent-{i}", "chat_id": 20000 + i,
                          "last_active_at": f"2026-10-01T00:{i:02}:00Z"} for i in range(55)]
    path.write_text(json.dumps(data))
    listed = registry.projects_summary(root, include_pinned=True)
    assert len(listed) == 105
    assert all(row["pinned"] for row in listed[:55])
    assert [row["id"] for row in listed[55:]] == [f"recent-{i}" for i in reversed(range(5, 55))]
    api_rows = client.get("/api/projects").json()["projects"]
    assert len(api_rows) == 110 and sum(row["pinned"] for row in api_rows) == 55
    assert api_rows[0]["id"] == "recent-54"  # ordinary enumeration keeps recency
    assert all(not row["pinned"] for row in registry.projects_summary(root))
    addressed = registry.projects_summary(root, chat_id=10000)
    assert addressed[0]["id"] == "pin-0" and addressed[0]["pinned"] is True


def test_legacy_and_concurrent_partial_updates_keep_neighbour_fields(pin_client):
    client, root = pin_client
    for pid in ("alpha", "beta"):
        registry.create_project(root, pid, name=pid)
    path = root / "state/projects.json"
    raw = json.loads(path.read_text())
    for row in raw["projects"]:
        row.pop("pinned")
    path.write_text(json.dumps(raw))
    before = path.read_bytes()
    assert not any(row["pinned"] for row in registry.projects_summary(root))
    assert path.read_bytes() == before  # legacy reads never migrate the file
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda pid: registry.update_project(root, pid, pinned=True), ("alpha", "beta")))
    assert all(registry.get_project(root, pid)["pinned"] for pid in ("alpha", "beta"))
    assert [registry.get_project(root, pid)["name"] for pid in ("alpha", "beta")] == ["alpha", "beta"]


def test_pin_contract_is_optional_partial_and_mirrored_in_jsdoc(pin_client):
    """The served ABI, its ingress validator and both JSDoc twins agree: pinned is an optional
    strict boolean on the row, and an update names name and/or pinned (the handler requires one)."""
    client, _ = pin_client
    document = client.get("/api/contract").json()
    schemas = document["schemas"]
    # ProjectEntry is published inline wherever a row is answered: the list and the update.
    listed = schemas["ProjectListResponse"]["properties"]["projects"]["items"]
    answered = [s for s in schemas["ProjectUpdateResponse"]["properties"]["project"]["anyOf"] if s["type"] == "object"]
    for entry in (listed, *answered):
        assert entry["properties"]["pinned"] == {"type": "boolean"} and "pinned" not in entry.get("required", [])
    assert len(answered) == 1
    update = schemas["ProjectUpdateRequest"]
    assert update["properties"] == {"name": {"type": "string"}, "pinned": {"type": "boolean"}, "closed": {"type": "boolean"}}
    assert not update.get("required")
    route = {f"{e['method']} {e['path']}": e for e in document["http"]}["POST /api/projects/{project_id}/update"]
    assert route["request"]["schemas"] == ["ProjectUpdateRequest"]
    assert set(route["responses"]) == {"200", "400", "404", "409", "500"}
    for body in ({"pinned": True}, {"pinned": False}, {"name": "Renamed"}, {"name": "Renamed", "pinned": True}):
        assert validate_ingress(body, ProjectUpdateRequest) == [], body
    for value in ("true", 1, 0, None, [], {}):
        assert validate_ingress({"pinned": value}, ProjectUpdateRequest) == ["pinned must be a JSON boolean"], value
    rows = (WEB_MODULES / "api_types.js").read_text(encoding="utf-8")
    assert _js_typedef_fields(rows, "ProjectEntry") == set(typing.get_type_hints(ProjectEntry))
    assert re.search(r"@typedef \{Object\} ProjectEntry\b[^/]*?@property \{boolean=\} pinned\b", rows, re.S)
    twin = (WEB_MODULES / "client_contract_types.js").read_text(encoding="utf-8")
    assert _js_typedef_properties(twin, "ProjectUpdateRequest") == {("string=", "name"), ("boolean=", "pinned"), ("boolean=", "closed")}
    assert set(typing.get_type_hints(ProjectUpdateRequest)) == {"name", "pinned", "closed"}
    assert not ProjectUpdateRequest.__required_keys__


def _seed_rooms(root, count=500):
    """One registry write of ``count`` (at least 301) neutral rooms (newest last), a few pinned or closed, one deleting
    and one tombstoned; returns the ids by role. Only identity, recency and lifecycle vary."""
    registry.create_project(root, "seed", origin="owner_ui")
    path = root / "state/projects.json"
    data = json.loads(path.read_text())
    template = data["projects"][0]
    rows = []
    for i in range(count):
        row = {**template, "id": f"room-{i:03}", "chat_id": 30000 + i, "name": f"Neutral room {i:03}",
               "last_active_at": f"2026-01-{1 + i // 60:02}T00:{i % 60:02}:00Z"}
        if i in (7, 300):
            row["pinned"] = True
        if i in (5, 250):
            row["closed"] = {"at": "2026-02-01T00:00:00Z", "by": "owner", "note": "Closed by owner."}
        rows.append(row)
    rows[3]["lifecycle"] = "deleting"
    rows[4]["lifecycle"] = "tombstoned"
    rows[0]["name"] = "Ünïcode ROOM — Плановые Работы"
    data["projects"] = rows
    path.write_text(json.dumps(data))
    return {"deep": "room-002", "deleting": "room-003", "tombstoned": "room-004", "closed": "room-005",
            "unicode": "room-000"}


def test_complete_list_and_exact_id_reach_every_room_beyond_both_caps(pin_client):
    """The sidebar feed keeps pins + 50, the plain list keeps 200; `all=1` and `id=` reach every other room
    (closed ones too) with its own chat id, a deleting room says so, a tombstone and an unknown id answer none."""
    client, root = pin_client
    ids = _seed_rooms(root)
    feed = registry.projects_summary(root, include_pinned=True)
    assert len(feed) == 2 + 50 and ids["deep"] not in {row["id"] for row in feed}
    capped = client.get("/api/projects?start=200&limit=20").json()["projects"]
    assert len(capped) == 200 and capped == client.get("/api/projects").json()["projects"]  # paging is not a parameter
    assert client.get("/api/projects?all=0").json()["projects"] == capped  # only the explicit form lifts the cap
    complete = client.get("/api/projects?all=1").json()["projects"]
    listed = {row["id"]: row for row in complete}
    assert len(complete) == 499 and ids["tombstoned"] not in listed
    assert [row["id"] for row in complete] == [row["id"] for row in registry.projects_summary(root, limit=None)]
    assert [(row["last_active_at"], row["id"]) for row in complete] == sorted(
        ((row["last_active_at"], row["id"]) for row in complete), reverse=True)
    assert listed[ids["deleting"]]["lifecycle"] == "deleting" and "closed" in listed[ids["closed"]]
    assert {row["id"] for row in capped} < set(listed) and set(listed) >= {row["id"] for row in feed}
    for role in ("deep", "closed", "unicode"):
        exact = client.get(f"/api/projects?id={ids[role]}").json()["projects"]
        assert [(row["id"], row["chat_id"], row["lifecycle"]) for row in exact] == [
            (ids[role], listed[ids[role]]["chat_id"], "active")]
        assert exact[0] == listed[ids[role]]  # the same compact row the list carries
    assert [row["lifecycle"] for row in client.get(f"/api/projects?id={ids['deleting']}").json()["projects"]] == ["deleting"]
    for absent in (ids["tombstoned"], "never-created", "", "../state"):
        assert client.get("/api/projects", params={"id": absent}).json() == {"projects": []}, absent
    # The addressed room by chat id keeps its meaning: active rooms only.
    assert client.get("/api/projects?chat_id=30003").json() == {"projects": []}
    assert client.get("/api/projects?chat_id=30002").json()["projects"][0]["id"] == ids["deep"]


def test_unreadable_registry_is_an_error_for_the_complete_list_and_the_exact_id(pin_client):
    client, root = pin_client
    registry.create_project(root, "room-002", name="Neutral room 002", origin="owner_ui")
    assert len(client.get("/api/projects?all=1").json()["projects"]) == 1
    (root / "state/projects.json").write_text("{not json")
    for query in ("all=1", "id=room-002"):
        response = client.get(f"/api/projects?{query}")
        assert response.status_code >= 500 and "projects" not in response.json(), query
