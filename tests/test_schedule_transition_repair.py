"""Authored edits before admission, and custody retries after occurrence retirement."""
from __future__ import annotations

import asyncio
import copy
import datetime
import json
import queue as stdqueue
from types import SimpleNamespace

import pytest
from starlette.requests import Request

from ouroboros.task_results import load_task_result, write_task_result
from supervisor import queue_schedules
from supervisor import schedule_occurrence as occurrence
from tests.test_schedule_occurrence import _row, _rows, q  # noqa: F401

pytestmark = pytest.mark.serial


def _api(q, payload, *, action=False):  # noqa: F811
    from ouroboros.gateway.schedules import api_schedules_action, api_schedules_upsert

    async def receive():
        return {"type": "http.request", "body": json.dumps(payload).encode()}

    request = Request({"type": "http", "app": SimpleNamespace(state=SimpleNamespace(drive_root=q.root)),
                       "path_params": {"schedule_id": "s1"}}, receive)
    result = asyncio.run((api_schedules_action if action else api_schedules_upsert)(request))
    assert result.status_code == 200, result.body
    return json.loads(result.body)


def _clock(monkeypatch):
    class Clock(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls.instant.astimezone(tz) if tz else cls.instant.replace(tzinfo=None)

    Clock.instant = Clock(2026, 9, 27, 12, tzinfo=datetime.timezone.utc)
    from supervisor import schedule_time

    clock_module = SimpleNamespace(datetime=Clock, timezone=datetime.timezone, timedelta=datetime.timedelta)
    for module in (queue_schedules, occurrence, schedule_time):
        monkeypatch.setattr(module, "datetime", clock_module)
    from ouroboros import retention

    age_cutoff = retention.age_cutoff
    # Schedule consumption and its GC cutoff must use the same advancing clock.
    monkeypatch.setattr(retention, "age_cutoff", lambda days, now=None: age_cutoff(
        days, Clock.instant.timestamp() if now is None else now))
    return Clock


def _refused(q, monkeypatch, *, cron=False):  # noqa: F811
    from ouroboros import consciousness_allowance

    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_MAX_TASKS", "0")
    monkeypatch.setattr(consciousness_allowance, "allowance_window", lambda _root: {
        "status": "available", "limit_usd": 10.0, "accounted_usd": 0.0,
        "unknown_unmetered": 0, "resets_at": ""})
    _row(q, cron=cron, intent={"kind": "system_repo"}, metadata={"initiator": "consciousness"})
    q.queue.check_scheduled_tasks()
    row = _rows(q)["s1"]
    assert row["occurrence"]["phase"] == "claimed" and not q.pending
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_MAX_TASKS", "1")
    return row


@pytest.mark.parametrize("cron", [False, True])
@pytest.mark.parametrize("legacy_claim", [False, True])
def test_api_moves_refused_occurrence_to_new_due_between_retry_ticks(q, monkeypatch, cron, legacy_claim):  # noqa: F811
    clock = _clock(monkeypatch)
    row = _refused(q, monkeypatch, cron=cron)
    if legacy_claim:
        with queue_schedules.schedule_transaction(q.root):
            data = queue_schedules.load_schedule_store(q.root)
            data["tasks"][0]["occurrence"].pop("fingerprint")
            queue_schedules._write_scheduled_tasks(data)
    row["timezone"] = "UTC"
    prior_id = row["occurrence"]["task_id"]
    tomorrow = clock.instant + datetime.timedelta(days=1)
    row["trigger"] = {"type": "cron", "expr": "0 12 * * *"} if cron else {
        "type": "once", "run_at": tomorrow.isoformat()}
    _api(q, row)
    for _ in range(2):
        q.queue.check_scheduled_tasks()
    assert not q.pending, "an owner edit must be due before a refused claim can admit"
    assert not load_task_result(q.root, prior_id)
    clock.instant = tomorrow
    for _ in range(2):
        q.queue.check_scheduled_tasks()
    assert len(q.pending) == 1
    admitted = load_task_result(q.root, q.pending[0]["id"])
    assert admitted["schedule_admission"]["dispatch"] == "none"
    if not cron:
        assert _rows(q)["s1"]["completed_at"]


def test_control_disable_and_restore_recheck_refused_work(q, monkeypatch):  # noqa: F811
    clock = _clock(monkeypatch)
    row = _refused(q, monkeypatch)
    _api(q, {"action": "disable", "reason": "owner paused"}, action=True)
    q.queue.check_scheduled_tasks()
    assert not q.pending
    row["enabled"] = False
    row["trigger"]["run_at"] = (clock.instant + datetime.timedelta(days=1)).isoformat()
    _api(q, row)
    _api(q, {"action": "restore", "reason": "owner resumed"}, action=True)
    q.queue.check_scheduled_tasks()
    assert not q.pending
    clock.instant += datetime.timedelta(days=1)
    q.queue.check_scheduled_tasks()
    assert len(q.pending) == 1


def test_skill_resync_reconsiders_legacy_refused_claim_under_new_cron(q, monkeypatch):  # noqa: F811
    from tests.test_consciousness_schedule_controls import _ready, _skill

    clock = _clock(monkeypatch)
    _ready(monkeypatch)
    queue_schedules.sync_skill_schedules([_skill()], drive_root=q.root)
    data = queue_schedules.load_schedule_store(q.root)
    data["tasks"][0]["next_run_at"] = "2000-01-01T00:00:00+00:00"
    queue_schedules._write_scheduled_tasks(data)
    original = q.queue.enqueue_task
    monkeypatch.setattr(q.queue, "enqueue_task", lambda task, **_kw: {
        **task, "_admission_blocked": "worker_pool_disabled"})
    q.queue.check_scheduled_tasks()
    data = queue_schedules.load_schedule_store(q.root)
    assert data["tasks"][0]["occurrence"]["phase"] == "claimed"
    data["tasks"][0]["occurrence"].pop("fingerprint")
    queue_schedules._write_scheduled_tasks(data)
    monkeypatch.setattr(q.queue, "enqueue_task", original)
    queue_schedules.sync_skill_schedules([_skill(tasks=(("daily", "0 12 * * *"),))], drive_root=q.root)
    q.queue.check_scheduled_tasks()
    assert not q.pending
    clock.instant += datetime.timedelta(days=1)
    q.queue.check_scheduled_tasks()
    assert len(q.pending) == 1


@pytest.mark.parametrize("evidence", ["accepted", "unstarted", "dispatch", "foreign", "unreadable"])
@pytest.mark.parametrize("cron", [False, True])
@pytest.mark.parametrize("edit_timing", [False, True])
def test_api_edit_preserves_accepted_or_unprovable_occurrence(q, monkeypatch, evidence, cron, edit_timing):  # noqa: F811
    clock = _clock(monkeypatch)
    row = _refused(q, monkeypatch, cron=cron)
    held = row["occurrence"]
    if evidence in {"accepted", "dispatch"}:
        task = {"id": held["task_id"], "type": "task", "text": "frozen original", "chat_id": 42,
                "metadata": {"schedule_occurrence": {"token": held["token"], "schedule_id": "s1"}}}
        write_task_result(q.root, task["id"], "scheduled", schedule_admission={
            "token": held["token"], "schedule_id": "s1", "dispatch": "possible" if evidence == "dispatch" else "none",
            "task": task})
    elif evidence == "foreign":
        write_task_result(q.root, held["task_id"], "scheduled", schedule_admission={"token": "foreign"})
    elif evidence == "unreadable":
        (q.root / "task_results").mkdir(exist_ok=True)
        (q.root / "task_results" / f"{held['task_id']}.json").write_text("{broken")
    else:
        with queue_schedules.schedule_transaction(q.root):
            data = queue_schedules.load_schedule_store(q.root)
            data["tasks"][0]["occurrence"].pop("admission", None)
            queue_schedules._write_scheduled_tasks(data)
    row["task"]["text"] = "edited future text"
    if edit_timing:
        row["timezone"] = "UTC"
        row["trigger"] = {"type": "cron", "expr": "0 12 * * *"} if cron else {
            "type": "once", "run_at": (clock.instant + datetime.timedelta(days=1)).isoformat()}
    _api(q, row)
    q.queue.check_scheduled_tasks()
    if evidence == "accepted":
        assert [t["text"] for t in q.pending] == ["frozen original"]
    elif evidence == "unstarted":
        if edit_timing:
            assert not q.pending
            assert "occurrence" not in _rows(q)["s1"]
        else:
            assert len(q.pending) == 1 and "edited future text" in q.pending[0]["text"]
            assert _rows(q)["s1"]["occurrence"]["token"] != held["token"]
    else:
        assert not q.pending
        if evidence != "dispatch":
            assert _rows(q)["s1"]["occurrence"]["token"] == held["token"]
    if evidence in {"accepted", "dispatch"}:
        current = _rows(q)["s1"]
        if cron:
            expected = clock.instant + (datetime.timedelta(days=1) if edit_timing else datetime.timedelta(minutes=1))
            assert datetime.datetime.fromisoformat(current["next_run_at"]) == expected
        else:
            assert bool(current.get("completed_at")) is not edit_timing


@pytest.mark.parametrize("cron,successor", [(False, False), (True, False), (True, True), (False, True)])
def test_worker_crash_retry_survives_settlement_and_successor(q, monkeypatch, cron, successor):  # noqa: F811
    from supervisor import state, worker_health, workers
    from tests.test_worker_crash_retry import _make_worker

    monkeypatch.setattr(workers, "DRIVE_ROOT", q.root)
    monkeypatch.setattr(workers, "REPO_DIR", q.root.parent / "repo")
    monkeypatch.setattr(workers, "PENDING", q.pending)
    monkeypatch.setattr(workers, "RUNNING", q.queue.RUNNING)
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(workers, "CRASH_TS", [])
    monkeypatch.setattr(workers, "_LAST_SPAWN_TIME", 0)
    monkeypatch.setattr(workers, "QUEUE_MAX_RETRIES", 1)
    monkeypatch.setattr(workers, "load_state", lambda: {"owner_chat_id": 0})
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 10.0)
    monkeypatch.setattr(workers, "send_with_budget", lambda *_a, **_k: None)
    monkeypatch.setattr(workers, "_reconcile_confirmed_dead_review_owner", lambda *_a: None)
    monkeypatch.setattr(workers, "get_event_q", stdqueue.Queue)
    monkeypatch.setattr(workers, "reconstruct_task_cost", lambda *_a, **_k: {})
    jobs = stdqueue.Queue()
    monkeypatch.setattr(q.queue, "_reap_queue", jobs)
    monkeypatch.setattr(q.queue, "_ensure_reaper_started", lambda: None)
    monkeypatch.setattr(workers, "respawn_worker", lambda *_a: None)
    _row(q, cron=cron, intent={"kind": "system_repo"})
    q.queue.check_scheduled_tasks()
    [original] = copy.deepcopy(q.pending)
    worker = _make_worker(alive=True, busy_task_id=None, exitcode=1)
    workers.WORKERS[0] = worker
    workers.assign_tasks()
    assert worker.in_q.put.call_count == 1
    q.queue.check_scheduled_tasks()
    assert "occurrence" not in _rows(q)["s1"]
    if successor:
        with queue_schedules.schedule_transaction(q.root):
            data = queue_schedules.load_schedule_store(q.root)
            occurrence.claim(data["tasks"][0], "2099-01-01T00:00:00+00:00")
            queue_schedules._write_scheduled_tasks(data)
    retained = copy.deepcopy(queue_schedules.load_schedule_store(q.root))
    worker.proc.is_alive.return_value = False
    workers.ensure_workers_healthy()
    job = jobs.get_nowait()
    assert job["kind"] == "confirmed_dead_worker"
    worker_health.recover_confirmed_dead_worker(job)
    [retry] = q.pending
    assert retry["id"] == original["id"] and retry["_attempt"] == 2
    assert not occurrence.restore_allowed(copy.deepcopy(original)), "unknown snapshot is never retry authority"
    replacement = _make_worker(alive=True, busy_task_id=None, exitcode=1)
    workers.WORKERS[0] = replacement
    workers.assign_tasks()
    assert replacement.in_q.put.call_count == 1, "custody retry cannot depend on a retired row token"
    assert replacement.in_q.put.call_args.args[0]["id"] == original["id"]
    assert queue_schedules.load_schedule_store(q.root) == retained


@pytest.mark.parametrize("case", ["terminal", "foreign", "foreign_schedule", "owner_hold", "unknown"])
def test_retired_row_does_not_authorize_invalid_redispatch(q, case):  # noqa: F811
    _row(q, intent={"kind": "system_repo"})
    q.queue.check_scheduled_tasks()
    task = q.pending.pop()
    assert occurrence.record_dispatch_possible(task)
    q.queue.check_scheduled_tasks()
    if case == "unknown":
        (q.root / "task_results" / f"{task['id']}.json").write_text("{broken")
    else:
        result = load_task_result(q.root, task["id"])
        admission = result["schedule_admission"]
        if case == "foreign":
            admission["token"] = "foreign"
        if case == "foreign_schedule":
            admission["schedule_id"] = "another"
        write_task_result(q.root, task["id"], "completed" if case == "terminal" else "interrupted",
                          schedule_admission=admission,
                          **({"_owner_hold": {"source": "owner"}} if case == "owner_hold" else {}))
    assert not occurrence.record_dispatch_possible(task)


@pytest.mark.parametrize("control", ["disabled", "deleted", "future", "edited", "unchanged"])
@pytest.mark.parametrize("legacy_refusal", [False, True])
def test_legacy_unstarted_claim_obeys_current_controls(q, monkeypatch, control, legacy_refusal):  # noqa: F811
    clock = _clock(monkeypatch)
    row = _refused(q, monkeypatch)
    claimed = copy.deepcopy(row["occurrence"])
    with queue_schedules.schedule_transaction(q.root):
        data = queue_schedules.load_schedule_store(q.root)
        current = data["tasks"][0]
        if legacy_refusal:
            current["occurrence"]["admission"] = "refused"
        if control == "deleted":
            # Older versions deferred this delete because they had forgotten
            # the process-local claim witness. It must not remain a ghost.
            current.update(enabled=False, delete_requested_at=clock.instant.isoformat())
        queue_schedules._write_scheduled_tasks(data)
    if control == "disabled":
        _api(q, {"action": "disable", "reason": "owner"}, action=True)
    elif control in {"future", "edited"}:
        row["task"]["text"] = "current authored text"
        if control == "future":
            row["trigger"]["run_at"] = (clock.instant + datetime.timedelta(days=1)).isoformat()
        _api(q, row)
    q.queue.check_scheduled_tasks()
    q.queue.check_scheduled_tasks()
    if control in {"disabled", "deleted", "future"}:
        assert not q.pending and not load_task_result(q.root, claimed["task_id"])
        if control == "deleted":
            assert "s1" not in _rows(q)
        else:
            assert "occurrence" not in _rows(q)["s1"]
    else:
        assert len(q.pending) == 1
        task = q.pending[0]
        if control == "unchanged":
            assert task["id"] == claimed["task_id"]
            assert task["metadata"]["schedule_occurrence"]["token"] == claimed["token"]
        else:
            assert task["id"] != claimed["task_id"] and "current authored text" in task["text"]


def test_deferred_unstarted_delete_is_finished_during_admission_recheck(q, monkeypatch):  # noqa: F811
    _clock(monkeypatch)
    row = _refused(q, monkeypatch)
    prepared = occurrence.prepare(occurrence.view(row))
    with queue_schedules.schedule_transaction(q.root):
        data = queue_schedules.load_schedule_store(q.root)
        data["tasks"][0].update(enabled=False, delete_requested_at="2026-09-27T12:00:00+00:00")
        queue_schedules._write_scheduled_tasks(data)
    occurrence.admit([prepared])
    assert not q.pending and "s1" not in _rows(q)
