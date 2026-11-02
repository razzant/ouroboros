"""Persistent claim recovery, receipt/dispatch barriers and frozen accepted work."""
from __future__ import annotations

import copy

import pytest

from ouroboros.task_results import load_task_result, write_task_result
from supervisor import queue_schedules
from supervisor import schedule_occurrence as occurrence
from tests import test_schedule_occurrence as fixtures
from tests._store_shared import delete_result, unreadable_result

_row, _rows, q = fixtures._row, fixtures._rows, fixtures.q

pytestmark = pytest.mark.serial


def _capacity_refusal(q, monkeypatch):
    from ouroboros import consciousness_allowance

    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_MAX_TASKS", "0")
    monkeypatch.setattr(consciousness_allowance, "allowance_window", lambda _root: {
        "status": "available", "limit_usd": 10.0, "accounted_usd": 0.0, "unknown_unmetered": 0, "resets_at": ""})
    _row(q, intent={"kind": "system_repo"}, metadata={"initiator": "consciousness"})
    q.queue.check_scheduled_tasks()
    row = _rows(q)["s1"]
    assert row["occurrence"]["phase"] == "claimed" and not q.pending
    assert row["hold"]["reason"] == "consciousness_task_limit"
    assert not load_task_result(q.root, row["occurrence"]["task_id"])
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_MAX_TASKS", "1")
    return copy.deepcopy(row["occurrence"])


@pytest.mark.parametrize("failure", ["false", "raise", "no_write"])
def test_failed_receipt_cannot_dispatch_and_recovers_same_claim(q, monkeypatch, failure):  # noqa: F811
    import ouroboros.task_results as results

    held = _capacity_refusal(q, monkeypatch)
    original = results.write_task_result

    def fail_receipt(*args, **kwargs):
        if failure == "raise":
            raise OSError("isolated receipt failure")
        return False if failure == "false" else None

    monkeypatch.setattr(results, "write_task_result", fail_receipt)
    q.queue.check_scheduled_tasks()
    assert not q.pending and _rows(q)["s1"]["occurrence"] == held
    assert _rows(q)["s1"]["hold"]["reason"] == "receipt_failed"
    monkeypatch.setattr(results, "write_task_result", original)
    for _ in range(2):
        q.queue.check_scheduled_tasks()
    assert [task["id"] for task in q.pending] == [held["task_id"]]
    assert occurrence.record_dispatch_possible(q.pending[0])


def test_crash_after_enqueue_without_receipt_recovers_same_unstarted_claim(q, monkeypatch):  # noqa: F811
    held = _capacity_refusal(q, monkeypatch)
    original = occurrence._write_receipt

    def crash(*_args):
        assert [task["id"] for task in q.pending] == [held["task_id"]]
        assert occurrence.record_dispatch_possible(q.pending[0]) is False
        assert occurrence.restore_allowed(q.pending[0]) is False
        raise RuntimeError("process lost before receipt")

    monkeypatch.setattr(occurrence, "_write_receipt", crash)
    with pytest.raises(RuntimeError, match="process lost"):
        q.queue.check_scheduled_tasks()
    q.pending.clear()
    monkeypatch.setattr(occurrence, "_write_receipt", original)
    q.queue.check_scheduled_tasks()
    assert [task["id"] for task in q.pending] == [held["task_id"]]
    assert _rows(q)["s1"]["occurrence"]["token"] == held["token"]
    q.queue.check_scheduled_tasks()
    assert len(q.pending) == 1


@pytest.mark.parametrize("case", ["foreign", "unreadable", "dispatch_row", "accepted", "owner_hold"])
def test_refusal_never_overrides_current_admission_or_unknown_source(q, monkeypatch, case):  # noqa: F811
    held = _capacity_refusal(q, monkeypatch)
    task_id, token = held["task_id"], held["token"]
    if case == "foreign":
        write_task_result(q.root, task_id, "scheduled", schedule_admission={"token": "foreign"})
    elif case == "unreadable":
        unreadable_result(monkeypatch, task_id)
    elif case == "dispatch_row":
        with queue_schedules.schedule_transaction(q.root):
            data = queue_schedules.load_schedule_store(q.root)
            data["tasks"][0]["occurrence"]["dispatch"] = "possible"
            queue_schedules._write_scheduled_tasks(data)
    else:
        frozen = {"id": task_id, "type": "task", "text": "ACCEPTED ORIGINAL", "chat_id": 42,
                  "metadata": {"schedule_occurrence": {"schedule_id": "s1", "token": token}}}
        write_task_result(q.root, task_id, "scheduled", schedule_admission={
            "schedule_id": "s1", "token": token, "dispatch": "none", "task": frozen},
            **({"_owner_hold": {"source": "owner", "revision": "latest"}} if case == "owner_hold" else {}))
    for _ in range(2):
        q.queue.check_scheduled_tasks()
    if case in {"accepted", "owner_hold"}:
        [task] = q.pending
        assert task["id"] == task_id and task["text"] == "ACCEPTED ORIGINAL"
        if case == "owner_hold":
            assert task["_owner_hold"]["revision"] == "latest"
            assert occurrence.record_dispatch_possible(task) is False
    else:
        assert not q.pending
        if case != "dispatch_row":
            assert _rows(q)["s1"]["hold"]["reason"] == (
                "occurrence_task_conflict" if case == "foreign" else "occurrence_result_unreadable")


def test_receipt_loss_after_dispatch_cannot_replay_old_refusal(q, monkeypatch):  # noqa: F811
    held = _capacity_refusal(q, monkeypatch)
    q.queue.check_scheduled_tasks()
    [task] = q.pending
    assert occurrence.record_dispatch_possible(task)
    q.pending.clear()
    delete_result(q.root, held["task_id"])
    for _ in range(2):
        q.queue.check_scheduled_tasks()
    assert not q.pending and "occurrence" not in _rows(q)["s1"]
    assert _rows(q)["s1"]["last_task_id"] == held["task_id"]


def test_accepted_republish_refusal_does_not_claim_never_admitted(q, monkeypatch):  # noqa: F811
    held = _capacity_refusal(q, monkeypatch)
    q.queue.check_scheduled_tasks()
    q.pending.clear()
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_MAX_TASKS", "0")
    q.queue.check_scheduled_tasks()
    current = _rows(q)["s1"]["occurrence"]
    assert current["phase"] == "admitted" and "admission" not in current
    delete_result(q.root, held["task_id"])
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_MAX_TASKS", "1")
    q.queue.check_scheduled_tasks()
    assert not q.pending and _rows(q)["s1"]["hold"]["reason"] == "occurrence_evidence_missing"


def test_stale_preparation_cannot_duplicate_or_replace_accepted_source(q, monkeypatch):  # noqa: F811
    held = _capacity_refusal(q, monkeypatch)
    stale = occurrence.prepare(occurrence.view(_rows(q)["s1"]))
    q.queue.check_scheduled_tasks()
    occurrence.admit([stale])
    assert [task["id"] for task in q.pending] == [held["task_id"]]
    q.pending.clear()
    stale["task"]["text"] = "STALE PREPARATION"
    occurrence.admit([stale])
    assert not q.pending
    q.queue.check_scheduled_tasks()
    [task] = q.pending
    assert task["id"] == held["task_id"] and task["text"] != "STALE PREPARATION"


@pytest.mark.parametrize("source", ["followup", "owner_gateway"])
@pytest.mark.parametrize("crash_point", ["prepare_hold", "before_receipt"])
def test_fresh_process_recovers_and_physically_assigns_once(tmp_path, source, crash_point):
    """A real interpreter exit loses every local object; only the protocol survives."""
    import json
    import os
    import pathlib
    import subprocess
    import sys

    program = r'''
import json, pathlib, sys
from types import SimpleNamespace
from supervisor import queue, queue_schedules, schedule_occurrence as occurrence, state, workers
from ouroboros import config
from ouroboros.projects_registry import create_project
from tests.test_schedule_occurrence import _rows
from ouroboros.tools.followup import _handle_schedule_followup
from ouroboros.tools.registry import ToolContext
from tests.test_schedule_transition_repair import _api
from tests.test_worker_crash_retry import _make_worker
from tests.worker_pool_support import affirm_ready
base, stage, source, crash_point = pathlib.Path(sys.argv[1]), *sys.argv[2:]
root, repo, folder = base / "data", base / "repo", base / "folder"
root.mkdir(exist_ok=True); repo.mkdir(exist_ok=True)
state.init(root)
workers.init(repo, root, 1)
config.get_bg_wakeup_min_sec = lambda: 0
q = SimpleNamespace(queue=queue, pending=workers.PENDING, root=root)
assert not hasattr(queue, "REPO_DIR")
if stage == "claim":
    state.save_state({"owner_chat_id": 1})
    folder.mkdir()
    create_project(root, "proj", name="Room", working_dir=str(folder))
    if source == "followup":
        context = ToolContext(repo_dir=repo, drive_root=root, task_id="origin", project_id="proj")
        context.is_direct_chat = True
        context.chat_id = 42
        answer = _handle_schedule_followup(context, run_at="2000-01-01T00:00:00+00:00", objective="continue",
                                           relation="independent")
        assert answer.startswith("FOLLOWUP_SCHEDULED"), answer
    else:
        _api(q, {"id": "s1", "name": "owner schedule", "trigger": {"type": "once", "run_at": "2000-01-01T00:00:00+00:00"},
                 "task": {"type": "task", "text": "continue", "project_id": "proj", "chat_id": 42}})
    if crash_point == "prepare_hold":
        folder.rmdir()
        folder.write_text("a file where the folder was")  # a missing folder starts as its address (#1799)
        queue.check_scheduled_tasks()
        assert next(iter(_rows(q).values()))["hold"]["reason"] == "workspace_unusable"
    else:
        def crash(*_a):
            assert len(workers.PENDING) == 1
            assert occurrence.record_dispatch_possible(workers.PENDING[0]) is False
            assert occurrence.restore_allowed(workers.PENDING[0]) is False
            raise RuntimeError("lost process before receipt")
        occurrence._write_receipt = crash
        try:
            queue.check_scheduled_tasks()
        except RuntimeError as exc:
            assert str(exc) == "lost process before receipt"
        else:
            raise AssertionError("crash boundary not reached")
    row = next(iter(_rows(q).values()))
    print(json.dumps({"pid": __import__("os").getpid(), "occurrence": row["occurrence"]}))
else:
    if folder.is_file():
        folder.unlink()
    folder.mkdir(exist_ok=True)
    queue.check_scheduled_tasks()
    queue.check_scheduled_tasks()
    assert len(workers.PENDING) == 1
    task = workers.PENDING[0]
    assert task["workspace_root"] == str(folder.resolve())
    worker = affirm_ready(_make_worker(alive=True, busy_task_id=None, exitcode=None))
    workers.WORKERS[0] = worker
    state.budget_remaining = lambda *_a, **_k: 10.0
    workers.assign_tasks()
    assert worker.in_q.put.call_count == 1
    sent = worker.in_q.put.call_args.args[0]
    queue.check_scheduled_tasks()
    workers.RUNNING.clear()
    worker.busy_task_id = None
    queue.check_scheduled_tasks()
    workers.assign_tasks()
    assert worker.in_q.put.call_count == 1
    print(json.dumps({"pid": __import__("os").getpid(), "task_id": sent["id"],
                      "occurrence": sent["metadata"]["schedule_occurrence"], "worker_sends": worker.in_q.put.call_count}))
'''
    results = []
    for stage in ("claim", "restart"):
        env = {**os.environ, "OUROBOROS_DATA_DIR": str(tmp_path / "data"),
               "OUROBOROS_REPO_DIR": str(tmp_path / "repo")}
        completed = subprocess.run(
            [sys.executable, "-c", program, str(tmp_path), stage, source, crash_point],
            cwd=pathlib.Path(__file__).resolve().parents[1], env=env,
            capture_output=True, text=True, encoding="utf-8", timeout=45,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        results.append(json.loads(completed.stdout.strip().splitlines()[-1]))
    first, second = results
    assert first["pid"] != second["pid"]
    assert first["occurrence"]["task_id"] == second["task_id"]
    for key in ("token", "due_at", "claimed_at"):
        assert first["occurrence"][key] == second["occurrence"][key]
    assert second["worker_sends"] == 1


@pytest.mark.parametrize("state_kind", ["accepted", "missing", "unreadable", "foreign", "admitted_missing"])
def test_worker_delete_reads_receipt_from_its_canonical_table_root(q, tmp_path, monkeypatch, state_kind):  # noqa: F811
    import json
    from types import SimpleNamespace
    from ouroboros.tools.followup import _manage_schedules

    canonical, child = tmp_path / "canonical", tmp_path / "child"
    row = {"id": "s1", "enabled": True, "source": "task_followup", "task": {},
           "trigger": {"type": "once", "run_at": "2000-01-01T00:00:00+00:00"},
           "occurrence": {"phase": "admitted" if state_kind == "admitted_missing" else "claimed",
                          "token": "same-token", "task_id": "same-task"}}
    queue_schedules._write_scheduled_tasks({"tasks": [row]}, canonical)
    if state_kind in {"accepted", "foreign"}:
        write_task_result(canonical, "same-task", "scheduled", schedule_admission={
            "schedule_id": "s1", "token": "same-token" if state_kind == "accepted" else "other",
            "dispatch": "none", "task": {"id": "same-task", "text": "frozen"}})
    elif state_kind == "unreadable":
        unreadable_result(monkeypatch, "same-task", root=canonical)
    # This other queue root has a receipt too. Neither its presence nor its
    # absence can determine custody of the selected table's occurrence.
    if state_kind == "missing":
        write_task_result(q.root, "same-task", "scheduled", schedule_admission={
            "token": "other-root", "dispatch": "none", "task": {"id": "same-task"}})
    monkeypatch.setenv("OUROBOROS_IN_WORKER", "1")
    ctx = SimpleNamespace(task_metadata={}, task_id="root", drive_root=child, budget_drive_root=canonical)
    outcome = json.loads(_manage_schedules(ctx, action="delete", schedule_id="s1", reason="owner"))
    assert outcome["ok"] and outcome["running_or_queued"] is None  # no snapshot required for custody
    rows = queue_schedules.load_schedule_store(canonical)["tasks"]
    assert outcome["status"] == "withdrawn" and outcome["schedule"] is not None
    assert len(rows) == 1 and rows[0]["delete_requested_at"] and not rows[0]["enabled"]
    assert rows[0]["occurrence"] == row["occurrence"] or rows[0]["occurrence"] == {
        "fingerprint": occurrence.fingerprint(row), **row["occurrence"]}



@pytest.mark.parametrize("clocks", ["both", "receipt_only", "row_only", "unknown"])
def test_legacy_accepted_republish_preserves_authored_task_and_known_clocks(q, clocks):  # noqa: F811
    from ouroboros.context_runtime_facts import task_schedule_fact

    _row(q, intent={"kind": "system_repo"})
    q.queue.check_scheduled_tasks()
    row = _rows(q)["s1"]
    task = copy.deepcopy(q.pending[0])
    task_id = task["id"]
    timing = task["metadata"]["schedule_occurrence"]
    timing.pop("due_at", None)
    timing.pop("claimed_at", None)
    task.update(text="Accepted authored instructions, unchanged.", context="frozen context",
                workspace_root="/frozen/resource", workspace_mode="external", memory_mode="forked")
    task["metadata"]["resource_intent"] = {"kind": "explicit_resource", "root": "/frozen/resource"}
    receipt = copy.deepcopy(load_task_result(q.root, task_id)["schedule_admission"])
    receipt["task"] = task
    due, claimed_at = receipt["due_at"], row["occurrence"]["claimed_at"]
    if clocks in {"row_only", "unknown"}:
        receipt.pop("due_at")
    with queue_schedules.schedule_transaction(q.root):
        table = queue_schedules.load_schedule_store(q.root)
        if clocks in {"receipt_only", "unknown"}:
            table["tasks"][0]["occurrence"].pop("due_at")
            table["tasks"][0]["occurrence"].pop("claimed_at")
        table["tasks"][0]["task"]["text"] = "Later edit must not replace accepted text"
        queue_schedules._write_scheduled_tasks(table)
    write_task_result(q.root, task_id, "scheduled", schedule_admission=receipt)
    q.pending.clear()
    row = _rows(q)["s1"]
    verdict, frozen = occurrence.reconcile(row)
    assert verdict == "republish"
    restored = occurrence.prepare(occurrence.view(row, frozen))["task"]
    expected_due = due if clocks != "unknown" else None
    expected_claimed = claimed_at if clocks in {"both", "row_only"} else None
    facts = task_schedule_fact(restored)["schedule_occurrence"]
    assert facts["due_at"] == expected_due and facts["claimed_at"] == expected_claimed
    preserved = copy.deepcopy(restored)
    for key in ("due_at", "claimed_at"):
        preserved["metadata"]["schedule_occurrence"].pop(key, None)
    assert preserved == task  # Only the retained host clocks enriched the frozen task.
    q.queue.check_scheduled_tasks()
    [queued] = q.pending
    assert queued["text"] == task["text"] and queued["workspace_root"] == task["workspace_root"]
    stored = load_task_result(q.root, task_id)["schedule_admission"]
    assert stored["task"]["text"] == task["text"] and stored.get("due_at") == expected_due


@pytest.mark.parametrize("mismatch", ["token", "schedule_id"])
def test_clock_enrichment_does_not_cross_frozen_occurrence_identity(q, monkeypatch, mismatch):  # noqa: F811
    from ouroboros.context_runtime_facts import task_schedule_fact

    row = {"id": "schedule", "occurrence": {
        "task_id": "task", "token": "our-token", "phase": "admitted",
        "due_at": "2000-01-01T00:00:00+00:00", "claimed_at": "2000-01-01T00:00:01+00:00"}}
    timing = {"schedule_id": "schedule", "token": "our-token"}
    timing[mismatch] = "foreign"
    task = {"id": "task", "text": "frozen", "metadata": {"schedule_occurrence": timing}}
    result = {"status": "scheduled", "schedule_admission": {
        "schedule_id": "schedule", "token": "our-token", "dispatch": "none", "task": task,
        "due_at": "2000-01-01T00:00:00+00:00"}}
    monkeypatch.setattr(occurrence, "_read_back", lambda *_a, **_kw: copy.deepcopy(result))
    verdict, projected = occurrence.reconcile(row)
    assert verdict == "republish" and projected == task
    assert task_schedule_fact(projected)["schedule_occurrence"]["due_at"] is None
