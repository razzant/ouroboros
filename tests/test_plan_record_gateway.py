"""The task detail carries the host's objection fold (scope R1, Opus C2 / Astra F3): the
browser renders ``plan_objections`` and never re-derives which objections stand, so an
upgraded task's imported finding is shown open, not "not asked yet"; no row claims a hold."""
from __future__ import annotations

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import plan_record
from ouroboros.gateway.tasks import api_task_get
from tests._store_shared import seed_result
from tests.test_plan_gate import _v2


def _detail(root, task_id):
    app = Starlette(routes=[Route("/api/tasks/{task_id}", api_task_get)])
    app.state.drive_root = root
    with TestClient(app) as client:
        response = client.get(f"/api/tasks/{task_id}")
    assert response.status_code == 200
    return response.json()


def _revision(root, task_id):
    fields = plan_record.compose_revision(None, goal="ship", plan="do it", done_when=[], affected_paths=[])
    plan_record.append_revision(root, task_id, fields, body=False)


def test_the_detail_projects_the_host_fold_with_imported_objections_and_their_raisers_withdrawal(tmp_path):
    seed_result(tmp_path, "mixed", {"task_id": "mixed", "status": "running", "plan_review_state": _v2(closed=False, blocking=("s1",))})
    _revision(tmp_path, "mixed")
    view = _detail(tmp_path, "mixed")["plan_objections"]
    assert view["legacy_engaged"] is True and view["withdrawn"] == [] and "legacy_unreadable" not in view
    [row] = view["open_objections"]["s1"]
    assert row["id"] == "s1:f1" and "holds" not in row and row["legacy"] is True and row["ask_id"] == "legacy"
    plan_record.append_ask(tmp_path, "mixed", {"ask_id": "a1", "ts": "t", "revision": 1, "message": "", "participants": [
        {"seat_id": "s2", "model": "m", "route": "api_chat", "owner_marked": True, "dispatched": True, "status": "answered",
         "reply": {"prose": "", "objections": [], "withdrawn": [{"id": "s1:f1", "why": "not mine"}]}}]})
    view = _detail(tmp_path, "mixed")["plan_objections"]
    assert [r["id"] for r in view["open_objections"]["s1"]] == ["s1:f1"], "a foreign withdrawal changes nothing"
    plan_record.append_ask(tmp_path, "mixed", {"ask_id": "a2", "ts": "t", "revision": 1, "message": "", "participants": [
        {"seat_id": "s1", "model": "m", "route": "api_chat", "owner_marked": True, "dispatched": True, "status": "answered",
         "reply": {"prose": "", "objections": [], "withdrawn": [{"id": "s1:f1", "why": "fixed"}]}}]})
    view = _detail(tmp_path, "mixed")["plan_objections"]
    assert view["open_objections"] == {} and view["withdrawn"] == [{"seat_id": "s1", "id": "s1:f1", "withdrawn_in": "a2", "why": "fixed"}]


def test_a_new_only_record_a_task_without_a_plan_and_an_unreadable_earlier_state(tmp_path):
    seed_result(tmp_path, "new", {"task_id": "new", "status": "running"})
    _revision(tmp_path, "new")
    plan_record.append_ask(tmp_path, "new", {"ask_id": "a1", "ts": "t", "revision": 1, "message": "", "participants": [
        {"seat_id": "s1", "model": "m", "route": "api_chat", "owner_marked": True, "dispatched": True, "status": "answered",
         "reply": {"prose": "", "objections": [{"id": "o1", "text": "too thin"}], "withdrawn": []}},
        {"seat_id": "critic", "model": "m", "route": "api_chat", "owner_marked": False, "dispatched": True, "status": "answered",
         "reply": {"prose": "", "objections": [{"id": "k1", "text": "no"}], "withdrawn": []}}]})
    view = _detail(tmp_path, "new")["plan_objections"]
    assert view["legacy_engaged"] is False
    assert {seat: [r["id"] for r in rows] for seat, rows in view["open_objections"].items()} == {"s1": ["o1"], "critic": ["k1"]}
    assert not any("holds" in r for rows in view["open_objections"].values() for r in rows), "an open objection holds nothing"
    seed_result(tmp_path, "none", {"task_id": "none", "status": "running"})
    assert "plan_objections" not in _detail(tmp_path, "none")
    seed_result(tmp_path, "bad", {"task_id": "bad", "status": "running", "plan_review_state": {"schema_version": 99}})
    _revision(tmp_path, "bad")
    view = _detail(tmp_path, "bad")["plan_objections"]
    assert view["legacy_unreadable"]["source"] == "plan_review_state" and view["open_objections"] == {}
