"""The owner's schedule controls seen from outside the store: what a lifecycle
action says about a run already admitted, how a worker answers when it cannot
see the live queue, and the gateway that carries the same seam as the agent's
tool. Companion of ``test_consciousness_schedule_controls`` (store, suppression,
transaction and projection), split so each file stays inside one reading window.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from ouroboros import utils as ouro_utils
from supervisor import queue_schedules, schedule_lifecycle
from tests.test_consciousness_schedule_controls import (
    _Queue, _bind, _events, _once_row, _ready, _rows, _skill, _skill_row, _write,
)
from tests._shared import write_queue_document


# --- a lifecycle action governs dispatch, not a run already admitted ---------------


def test_disabling_a_schedule_reports_the_run_it_does_not_stop(tmp_path, monkeypatch):
    _bind(monkeypatch, tmp_path)
    _write(tmp_path, [{"id": "cron-1", "name": "cron", "enabled": True,
                       "trigger": {"type": "cron", "expr": "0 * * * *"},
                       "task": {"type": "task", "text": "run"}}])
    _Queue.RUNNING = {"w1": {"task": {"metadata": {"schedule_id": "cron-1"}}}}
    outcome = schedule_lifecycle.mutate_scheduled_task("disable", "cron-1", reason="stop future runs",
                                                    actor="agent", drive_root=tmp_path)
    assert outcome["ok"] and outcome["running_or_queued"] is True
    assert _rows(tmp_path)[0]["enabled"] is False


def test_a_worker_reports_unknown_rather_than_claiming_no_run_is_in_flight(tmp_path, monkeypatch):
    """PENDING/RUNNING belong to the SUPERVISOR process, not to every caller.

    ``manage_schedules`` runs in a worker, where those dicts are this process's
    own empty copies; answering False from there would claim nothing this
    schedule started is alive. The worker reads the durable queue snapshot — the
    file the supervisor rewrites on exactly those transitions — and says unknown
    when it is not there.
    """
    _bind(monkeypatch, tmp_path)
    _write(tmp_path, [{"id": "cron-1", "name": "cron", "enabled": True,
                       "trigger": {"type": "cron", "expr": "0 * * * *"},
                       "task": {"type": "task", "text": "run"}}])
    monkeypatch.setattr(queue_schedules, "in_worker_process", lambda: True)

    # No snapshot to read: unknown, not "nothing is running".
    unknown = schedule_lifecycle.mutate_scheduled_task("disable", "cron-1", reason="pause",
                                                    actor="agent", drive_root=tmp_path)
    assert unknown["ok"] is True and unknown["running_or_queued"] is None

    snapshot = tmp_path / "state" / "queue_snapshot.json"
    write_queue_document(snapshot, {
        "ts": ouro_utils.utc_now_iso(), "pending": [], "running": [
            {"id": "task-1", "task": {"id": "task-1", "metadata": {"schedule_id": "cron-1"}}}],
    })
    seen = schedule_lifecycle.mutate_scheduled_task("restore", "cron-1", reason="back",
                                                 actor="agent", drive_root=tmp_path)
    assert seen["running_or_queued"] is True

    write_queue_document(snapshot, {
        "ts": ouro_utils.utc_now_iso(), "pending": [
            {"id": "task-2", "task": {"id": "task-2", "metadata": {"schedule_id": "other"}}}],
        "running": [],
    })
    clear = schedule_lifecycle.mutate_scheduled_task("disable", "cron-1", reason="pause again",
                                                  actor="agent", drive_root=tmp_path)
    assert clear["running_or_queued"] is False

    # In the supervisor process the live queue is the authority, as before.
    monkeypatch.setattr(queue_schedules, "in_worker_process", lambda: False)
    _Queue.PENDING = [{"id": "task-3", "metadata": {"schedule_id": "cron-1"}}]
    live = schedule_lifecycle.mutate_scheduled_task("restore", "cron-1", reason="back again",
                                                 actor="agent", drive_root=tmp_path)
    assert live["running_or_queued"] is True


@pytest.mark.parametrize("snapshot", [
    {},
    {"ts": "2000-01-01T00:00:00+00:00", "pending": [], "running": []},
])
def test_worker_does_not_treat_malformed_or_stale_snapshot_as_proof_of_no_run(
        tmp_path, monkeypatch, snapshot):
    """Only a fresh, structurally complete snapshot can answer ``False``."""
    _bind(monkeypatch, tmp_path)
    _write(tmp_path, [{"id": "cron-1", "name": "cron", "enabled": True,
                       "trigger": {"type": "cron", "expr": "0 * * * *"},
                       "task": {"type": "task", "text": "run"}}])
    monkeypatch.setattr(queue_schedules, "in_worker_process", lambda: True)
    (tmp_path / "state" / "queue_snapshot.json").write_text(
        json.dumps(snapshot), encoding="utf-8")
    outcome = schedule_lifecycle.mutate_scheduled_task(
        "disable", "cron-1", reason="pause", actor="agent", drive_root=tmp_path)
    assert outcome["ok"] is True
    assert outcome["running_or_queued"] is None


def test_the_tick_waits_out_an_unknown_in_flight_run_instead_of_re_dispatching(tmp_path, monkeypatch):
    """Unknown is not permission to fire a schedule whose last run may be alive."""
    _bind(monkeypatch, tmp_path)
    _write(tmp_path, [{"id": "cron-1", "name": "cron", "enabled": True,
                       "trigger": {"type": "cron", "expr": "* * * * *"},
                       "next_run_at": "2020-01-01T00:00:00+00:00",
                       "task": {"type": "task", "text": "run"}}])
    monkeypatch.setattr(queue_schedules, "_schedule_running_or_queued", lambda *_a, **_k: None)
    enqueued: list = []
    _Queue.enqueue_task = staticmethod(lambda task: enqueued.append(task) or task)
    _Queue.load_state = staticmethod(lambda: {})
    _Queue.persist_queue_snapshot = staticmethod(lambda reason="": True)
    try:
        queue_schedules.check_scheduled_tasks()
    finally:
        for name in ("enqueue_task", "load_state", "persist_queue_snapshot"):
            delattr(_Queue, name)
    assert enqueued == []
    assert _rows(tmp_path)[0]["next_run_at"] == "2020-01-01T00:00:00+00:00"


@pytest.mark.parametrize(("action", "schedule_id", "reason", "status"), [
    ("explode", "cron-1", "why", "invalid_action"),
    ("disable", "", "why", "missing_schedule_id"),
    ("disable", "cron-1", "   ", "reason_required"),
    ("disable", "absent", "why", "not_found"),
])
def test_a_malformed_or_impossible_action_is_typed_and_writes_nothing(
        tmp_path, monkeypatch, action, schedule_id, reason, status):
    _bind(monkeypatch, tmp_path)
    _write(tmp_path, [{"id": "cron-1", "enabled": True, "trigger": {"type": "cron", "expr": "0 * * * *"}}])
    outcome = schedule_lifecycle.mutate_scheduled_task(action, schedule_id, reason=reason,
                                                    actor="agent", drive_root=tmp_path)
    assert outcome["ok"] is False and outcome["status"] == status
    assert outcome["audit"] == "not_written" and _events(tmp_path) == []
    assert _rows(tmp_path)[0]["enabled"] is True


# --- the owner's buttons and the agent's tool are the same seam --------------------


def _gateway(tmp_path):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway.schedules import (
        api_schedules_action, api_schedules_delete, api_schedules_list, api_schedules_upsert,
    )

    app = Starlette(routes=[
        Route("/api/schedules", endpoint=api_schedules_list, methods=["GET"]),
        Route("/api/schedules", endpoint=api_schedules_upsert, methods=["POST"]),
        Route("/api/schedules/{schedule_id}/action", endpoint=api_schedules_action, methods=["POST"]),
        Route("/api/schedules/{schedule_id}", endpoint=api_schedules_delete, methods=["DELETE"]),
    ])
    app.state.drive_root = tmp_path
    return TestClient(app)


def test_the_gateway_action_and_the_agent_tool_mean_the_same_thing(tmp_path, monkeypatch):
    _bind(monkeypatch, tmp_path)
    _ready(monkeypatch)
    monkeypatch.setattr("ouroboros.config.get_skills_repo_path", lambda: "")
    monkeypatch.setattr("ouroboros.skill_loader.discover_skills", lambda *_a, **_k: [_skill()])
    client = _gateway(tmp_path)

    _write(tmp_path, [_skill_row()])
    by_owner = client.post("/api/schedules/skill-demo-daily/action",
                           json={"action": "disable", "reason": "owner paused it"}).json()
    owner_row = dict(_rows(tmp_path)[0])
    _write(tmp_path, [_skill_row()])
    by_agent = schedule_lifecycle.mutate_scheduled_task(
        "disable", "skill-demo-daily", reason="agent paused it", actor="agent", drive_root=tmp_path)
    agent_row = dict(_rows(tmp_path)[0])

    assert by_owner["status"] == by_agent["status"] == "updated"
    assert by_owner["ok"] is by_agent["ok"] is True
    assert owner_row["writer_kind"] == owner_row["writer_origin"]["work_kind"] == "owner"
    assert agent_row["writer_kind"] == agent_row["writer_origin"]["work_kind"] == "consciousness"
    ignore = {"updated_at", "writer_kind", "writer_origin"}
    assert {k: v for k, v in owner_row.items() if k not in ignore} == \
           {k: v for k, v in agent_row.items() if k not in ignore}
    # Both wrote the same audit shape; only the actor and the reason differ.
    owner_event, agent_event = _events(tmp_path)[1], _events(tmp_path)[3]
    assert owner_event["actor"] == "owner:gateway" and agent_event["actor"] == "agent"
    assert owner_event.keys() == agent_event.keys()


def test_the_gateway_action_requires_a_named_action_and_a_reason(tmp_path, monkeypatch):
    _bind(monkeypatch, tmp_path)
    _write(tmp_path, [_skill_row()])
    client = _gateway(tmp_path)
    assert client.post("/api/schedules/skill-demo-daily/action", json={"reason": "x"}).status_code == 400
    assert client.post("/api/schedules/skill-demo-daily/action",
                       json={"action": "explode", "reason": "x"}).status_code == 400
    assert client.post("/api/schedules/skill-demo-daily/action",
                       json={"action": "disable"}).status_code == 400
    assert _rows(tmp_path)[0]["enabled"] is True
    # A reason is a fact about this decision, not a command: it cannot select one.
    from ouroboros.gateway import schedules as gateway_schedules

    source = pathlib.Path(gateway_schedules.__file__).read_text(encoding="utf-8")
    assert "activity_toggle" not in source and "activity_restore" not in source


def test_the_delete_endpoint_reports_the_withdrawal_it_applied(tmp_path, monkeypatch):
    _bind(monkeypatch, tmp_path)
    _write(tmp_path, [_skill_row()])
    body = _gateway(tmp_path).delete("/api/schedules/skill-demo-daily").json()
    assert body["ok"] is True and body["status"] == "withdrawn"
    assert _rows(tmp_path)[0]["delete_requested_at"]


def test_the_gateway_list_carries_the_lifecycle_words_the_ui_renders(tmp_path, monkeypatch):
    _bind(monkeypatch, tmp_path)
    _write(tmp_path, [_skill_row("c", enabled=False, delete_requested_at="2026-10-21T00:00:00+00:00"), _once_row()])
    rows = _gateway(tmp_path).get("/api/schedules").json()["tasks"]
    assert {r["status"] for r in rows} == {"delete_pending", "consumed"}
    assert all(r["retained"] for r in rows)


def test_an_unreadable_table_refuses_the_gateway_too(tmp_path, monkeypatch):
    _bind(monkeypatch, tmp_path)
    (tmp_path / "state" / "scheduled_tasks.json").write_text("{ broken", encoding="utf-8")
    client = _gateway(tmp_path)
    upsert = client.post("/api/schedules", json={
        "id": "new", "trigger": {"type": "cron", "expr": "0 * * * *"},
        "task": {"type": "task", "text": "x"}})
    assert upsert.status_code == 409 and "readable" in upsert.json()["error"]
    action = client.post("/api/schedules/new/action", json={"action": "disable", "reason": "x"})
    assert action.json()["status"] == "store_unreadable"
    # "No schedules" would be a claim; an unreadable table reads as UNKNOWN, which
    # is what the Activity section's own failed-read state is there to say.
    listed = client.get("/api/schedules")
    assert listed.status_code == 503 and "readable" in listed.json()["error"]
    assert (tmp_path / "state" / "scheduled_tasks.json").read_text(encoding="utf-8") == "{ broken"


def test_the_legacy_delete_contract_name_survives_beside_the_action_one():
    """Additive only: a published contract name is not removed by a richer reply."""
    from ouroboros.gateway import contracts
    from ouroboros.gateway.schedule_contracts import ScheduleDeleteResponse

    assert contracts.ScheduleDeleteResponse is ScheduleDeleteResponse
    assert set(ScheduleDeleteResponse.__annotations__) == {"ok"}
    for name in ("ScheduleDeleteResponse", "ScheduleActionResponse",
                 "ScheduleUpsertResponse", "ScheduledTasksResponse"):
        assert name in contracts.__all__, name
    mirror = pathlib.Path("web/modules/api_types.js").read_text(encoding="utf-8")
    assert "@typedef {Object} ScheduleDeleteResponse" in mirror


def test_the_action_outcome_contract_and_its_js_mirror_name_the_same_fields(tmp_path, monkeypatch):
    """Item 17 parity: the Python TypedDict and the JS typedef stay one contract."""
    from ouroboros.gateway.contracts import ScheduleActionResponse

    _bind(monkeypatch, tmp_path)
    _write(tmp_path, [{"id": "cron-1", "enabled": True, "name": "cron",
                       "trigger": {"type": "cron", "expr": "0 * * * *"},
                       "task": {"type": "task", "text": "run"}}])
    outcome = _gateway(tmp_path).post("/api/schedules/cron-1/action",
                                      json={"action": "disable", "reason": "why"}).json()
    declared = set(ScheduleActionResponse.__annotations__)
    assert set(outcome) <= declared, sorted(set(outcome) - declared)
    assert {"ok", "changed", "status", "schedule_id", "audit"} <= set(outcome)
    mirror = pathlib.Path("web/modules/api_types.js").read_text(encoding="utf-8")
    block = mirror.split("@typedef {Object} ScheduleActionResponse", 1)[1].split("*/", 1)[0]
    mirrored = {line.split("}", 1)[1].split()[0].rstrip("=")
                for line in block.splitlines() if "@property" in line}
    assert mirrored == declared, sorted(mirrored ^ declared)


def test_a_missed_schedule_lock_is_a_typed_refusal_not_a_bare_timeout(tmp_path, monkeypatch):
    """The lock bound is a refusal like an unreadable table: nothing changed, the
    state is unknown. The lifecycle action answers it as a typed outcome; the
    upsert raises the typed class, which the owner surfaces already catch."""
    _bind(monkeypatch, tmp_path)
    _write(tmp_path, [{"id": "cron-1", "name": "cron", "enabled": True,
                       "trigger": {"type": "cron", "expr": "0 * * * *"},
                       "task": {"type": "task", "text": "run"}}])
    monkeypatch.setattr(queue_schedules, "acquire_exclusive_file_lock", lambda *_a, **_k: None)
    outcome = schedule_lifecycle.mutate_scheduled_task("disable", "cron-1", reason="pause",
                                                    actor="agent", drive_root=tmp_path)
    assert outcome["ok"] is False and outcome["changed"] is False
    assert outcome["status"] == "lock_timeout" and outcome["audit"] == "not_written"
    assert _rows(tmp_path)[0]["enabled"] is True and _events(tmp_path) == []
    with pytest.raises(queue_schedules.ScheduleStoreUnreadable) as refusal:
        schedule_lifecycle.upsert_scheduled_task({"id": "new", "trigger": {"type": "cron", "expr": "0 * * * *"}},
                                                 drive_root=tmp_path)
    assert isinstance(refusal.value, TimeoutError)
    # The gateway turns it into the same 409 an unreadable table gets, not a 500.
    client = _gateway(tmp_path)
    response = client.post("/api/schedules", json={"id": "new", "name": "n", "trigger": {"type": "cron", "expr": "0 * * * *"},
                                                   "task": {"type": "task", "text": "x"}})
    assert response.status_code == 409 and "schedule lock" in response.json()["error"]
