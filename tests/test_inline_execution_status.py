"""Execution ownership, independent of conversation origin and reader process."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.task_status import (
    execution_owner_record, load_effective_task_result, reconcile_orphaned_running_tasks,
    task_execution_observation,
)
from ouroboros.utils import append_jsonl
from tests._shared import write_queue_document


def _stamp(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def _seed(root, tid, kind=None, **fields):
    root.mkdir(parents=True, exist_ok=True)
    (root / "state").mkdir(exist_ok=True)
    now = time.time()
    task = {"id": tid, "_attempt": 1}
    if kind:
        fields["execution_owner"] = execution_owner_record(root, task, kind)
    row = write_task_result(root, tid, "running", task_attempt=1, ts=_stamp(now - 200), **fields)
    write_queue_document(root / "state" / "queue_snapshot.json", {"ts": _stamp(now), "pending": [], "running": []})
    append_jsonl(root / "logs" / "events.jsonl", {"ts": _stamp(now - 100), "type": "worker_boot"})
    return row


def _fresh_reader(root, tid, *, reconcile=False):
    code = """
import json,sys
from pathlib import Path
from ouroboros.task_status import load_effective_task_result,reconcile_orphaned_running_tasks
root=Path(sys.argv[1]); tid=sys.argv[2]
healed=reconcile_orphaned_running_tasks(root) if sys.argv[3]=='yes' else None
row=load_effective_task_result(root,tid,materialize_artifacts=False)
print(json.dumps({'healed':healed,'status':row['status'],'observation':row['execution_observation']}))
"""
    env = {**os.environ, "OUROBOROS_DATA_DIR": str(root), "OUROBOROS_SETTINGS_PATH": str(root / "settings.json")}
    result = subprocess.run([sys.executable, "-c", code, str(root), tid, "yes" if reconcile else "no"],
                            cwd=Path(__file__).resolve().parents[1], env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.serial
@pytest.mark.parametrize("kind", ["direct", "presence"])
def test_fresh_reader_never_orphans_inline_owner(tmp_path, kind):
    fields = {"_is_direct_chat": True}
    if kind == "presence":
        fields["metadata"] = {"source": "presence", "presence": {"binding_id": "binding"}}
    _seed(tmp_path, "inline", kind, **fields)
    observed = _fresh_reader(tmp_path, "inline", reconcile=True)
    assert observed["status"] == "running" and observed["healed"] == 0
    assert observed["observation"]["kind"] == kind
    assert observed["observation"]["state"] == "unknown"
    write_task_result(tmp_path, "inline", "completed", result="Real answer")
    assert load_task_result(tmp_path, "inline")["status"] == "completed"


@pytest.mark.serial
@pytest.mark.parametrize("fields", [{}, {"metadata": {"source": "presence_promote", "presence": {"binding_id": "b"}}},
                                    {"_is_direct_chat": True, "budget_pause": {"state": "resumed"}}])
def test_genuinely_orphaned_pool_still_settles_across_processes(tmp_path, fields):
    _seed(tmp_path, "pooled", "pooled", **fields)
    observed = _fresh_reader(tmp_path, "pooled", reconcile=True)
    assert observed["status"] == "failed" and observed["healed"] == 1
    assert load_task_result(tmp_path, "pooled")["reason_code"] == "orphaned_running_after_worker_restart"


@pytest.mark.parametrize("evidence", ["missing", "stale", "incomplete", "unreadable", "fresh"])
def test_direct_fragment_is_only_positive_evidence(tmp_path, evidence):
    _seed(tmp_path, "direct", "direct", _is_direct_chat=True)
    path = tmp_path / "state" / "direct_roots.json"
    if evidence == "unreadable":
        path.write_text("{", encoding="utf-8")
    elif evidence != "missing":
        write_queue_document(path, {"ts": _stamp(time.time() - (20 if evidence == "stale" else 0)),
                                  "incomplete": evidence == "incomplete",
                                  "roots": [] if evidence == "incomplete" else [{"task_id": "direct"}]})
    row = load_effective_task_result(tmp_path, "direct", materialize_artifacts=False)
    assert row["status"] == "running"
    assert row["execution_observation"]["state"] == ("active" if evidence == "fresh" else "unknown")
    assert reconcile_orphaned_running_tasks(tmp_path) == 0
    write_task_result(tmp_path, "direct", "completed", result="done")
    assert load_effective_task_result(tmp_path, "direct")["execution_observation"]["state"] == "terminal"


@pytest.mark.parametrize("kind", ["direct", "presence", "pooled"])
def test_foreign_root_and_attempt_never_prove_ownership(tmp_path, kind):
    own, foreign = tmp_path / "own", tmp_path / "foreign"
    row = _seed(own, "same-id", kind)
    _seed(foreign, "same-id")
    write_task_result(foreign, "same-id", "running", execution_owner=row["execution_owner"])
    observed = load_effective_task_result(foreign, "same-id")
    assert observed["status"] == "running"
    assert observed["execution_observation"]["kind"] == "unknown"
    assert reconcile_orphaned_running_tasks(foreign) == 0
    changed = {**row, "task_attempt": 2}
    assert task_execution_observation(own, changed)["kind"] == "unknown"


def test_local_presence_witness_is_root_scoped(tmp_path, monkeypatch):
    from ouroboros import presence_gate
    own, foreign = tmp_path / "own", tmp_path / "foreign"
    one = _seed(own, "same-id", "presence")
    two = _seed(foreign, "same-id", "presence")
    monkeypatch.setattr(presence_gate, "_LIVE_PRESENCE_TASKS", {(str(own.resolve()), "same-id")})
    assert task_execution_observation(own, one)["state"] == "active"
    assert task_execution_observation(foreign, two)["state"] == "unknown"


def test_local_direct_witness_is_root_scoped(tmp_path, monkeypatch):
    from supervisor import active_activity
    registry = active_activity.DirectActivityRegistry()
    monkeypatch.setattr(active_activity, "_DIRECT_ACTIVITY_REGISTRY", registry)
    own, foreign = tmp_path / "own", tmp_path / "foreign"
    one = _seed(own, "same-id", "direct")
    two = _seed(foreign, "same-id", "direct")
    registry.register("same-id", 1, actor=SimpleNamespace(env=SimpleNamespace(drive_root=own)))
    assert task_execution_observation(own, one)["state"] == "active"
    assert task_execution_observation(foreign, two)["state"] == "unknown"


def test_actual_pool_assignment_rebinds_direct_and_native_split_start(tmp_path, monkeypatch):
    from ouroboros.agent import OuroborosAgent
    from supervisor import worker_assignment, workers
    root, child = tmp_path / "canonical", tmp_path / "child"
    root.mkdir(); child.mkdir()
    task = {"id": "resumed", "_attempt": 2, "_is_direct_chat": True,
            "budget_drive_root": str(root), "drive_root": str(child)}
    write_task_result(root, "resumed", "running", task_attempt=2,
                      execution_owner=execution_owner_record(root, task, "direct"))
    monkeypatch.setattr(workers, "DRIVE_ROOT", root)
    worker_assignment._mirror_assigned_running_status(task)
    assert load_task_result(root, "resumed")["execution_owner"]["kind"] == "pooled"
    actor = object.__new__(OuroborosAgent)
    actor.env = SimpleNamespace(drive_root=child, budget_drive_root=root)
    actor._task_started_ts = time.time()
    actor._persist_running_record(task)
    assert load_task_result(child, "resumed")["execution_owner"] == load_task_result(root, "resumed")["execution_owner"]
    assert load_task_result(root, "resumed")["execution_owner"]["task_attempt"] == 2


def test_nofork_resume_publishes_pool_owner_at_actual_native_start(tmp_path, monkeypatch):
    from ouroboros.agent import OuroborosAgent
    from supervisor import worker_assignment, workers

    task = {"id": "resumed", "_attempt": 1, "_is_direct_chat": True, "budget_drive_root": str(tmp_path)}
    previous = execution_owner_record(tmp_path, task, "direct")
    write_task_result(tmp_path, "resumed", "scheduled", task_attempt=1, execution_owner=previous)
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    worker_assignment._mirror_assigned_running_status(task)
    assigned = load_task_result(tmp_path, "resumed")
    assert assigned["status"] == "scheduled" and assigned["execution_owner"] == previous
    assert task["_execution_owner"]["kind"] == "pooled"
    native = object.__new__(OuroborosAgent)
    native.env = SimpleNamespace(drive_root=tmp_path)
    native._persist_running_record(task)
    running = load_task_result(tmp_path, "resumed")
    assert running["status"] == "running" and running["execution_owner"]["kind"] == "pooled"


@pytest.mark.parametrize("flags,kind", [({}, "pooled"), ({"_is_direct_chat": True}, "direct"),
                                      ({"_is_direct_chat": True, "_presence_turn": True}, "presence")])
def test_native_start_stamps_actual_execution_kind(tmp_path, flags, kind):
    from ouroboros.agent import OuroborosAgent

    actor = object.__new__(OuroborosAgent)
    actor.env = SimpleNamespace(drive_root=tmp_path)
    actor._task_started_ts = time.time()
    actor._persist_running_record({"id": "started", **flags})
    row = load_task_result(tmp_path, "started")
    assert row["execution_owner"] == execution_owner_record(tmp_path, {"id": "started"}, kind)


def test_canonical_assignment_beats_a_pre_resume_child_owner(tmp_path):
    root, child = tmp_path / "canonical", tmp_path / "child"
    _seed(root, "resumed", "pooled", _is_direct_chat=True, child_drive_root=str(child))
    _seed(child, "resumed", _is_direct_chat=True,
          execution_owner=execution_owner_record(root, {"id": "resumed", "_attempt": 1}, "direct"))
    observed = load_effective_task_result(root, "resumed")
    assert observed["execution_owner"]["kind"] == "pooled"
    assert observed["status"] == "failed"


@pytest.mark.parametrize("child_terminal", [False, True])
def test_canonical_new_attempt_stays_bound_beside_old_child(tmp_path, monkeypatch, child_terminal):
    from supervisor import worker_assignment, workers

    root, child = tmp_path / "canonical", tmp_path / "child"
    _seed(root, "retry", "pooled", child_drive_root=str(child))
    _seed(child, "retry", _attempt=1,
          execution_owner=execution_owner_record(root, {"id": "retry", "_attempt": 1}, "pooled"))
    monkeypatch.setattr(workers, "DRIVE_ROOT", root)
    worker_assignment._mirror_assigned_running_status(
        {"id": "retry", "_attempt": 2, "budget_drive_root": str(root), "drive_root": str(child)})
    write_task_result(root, "retry", "running", ts=_stamp(time.time() - 200))
    if child_terminal:
        write_task_result(child, "retry", "completed", result="Actual child answer")
    observed = load_effective_task_result(root, "retry")
    assert observed["task_attempt"] == observed["execution_owner"]["task_attempt"] == 2
    assert "_attempt" not in observed  # an old secondary carrier cannot be borrowed either
    if child_terminal:
        assert observed["status"] == "completed" and observed["result"] == "Actual child answer"
    else:
        assert observed["status"] == "failed" and observed["execution_observation"]["kind"] == "pooled"
        assert reconcile_orphaned_running_tasks(root) == 1


@pytest.mark.parametrize("outcome_failure", [False, True])
def test_projection_does_not_claim_a_terminal_was_already_recorded(tmp_path, outcome_failure):
    from ouroboros.outcomes import infra_failed_axes

    fields = {"outcome_axes": infra_failed_axes("provider_unavailable")} if outcome_failure else {}
    _seed(tmp_path, "orphan", "pooled", **fields)
    observation = load_effective_task_result(tmp_path, "orphan")["execution_observation"]
    assert observation["state"] == "terminal" and observation["reason"] == "projected_terminal"
    assert observation["basis"] == ("recorded_outcome" if outcome_failure else "pooled_worker_restart")
    assert load_task_result(tmp_path, "orphan")["status"] == "running"
    assert reconcile_orphaned_running_tasks(tmp_path) == 1
    assert load_task_result(tmp_path, "orphan")["status_reconciled_from"] == "running"
    assert load_effective_task_result(tmp_path, "orphan")["execution_observation"]["reason"] == "recorded_terminal"


def test_effective_observation_reuses_its_queue_read(tmp_path, monkeypatch):
    from ouroboros import task_status

    _seed(tmp_path, "inline", "direct", _is_direct_chat=True)
    actual, reads = task_status._load_queue_snapshot, []

    def counted(root):
        reads.append(root)
        return actual(root)

    monkeypatch.setattr(task_status, "_load_queue_snapshot", counted)
    row = load_effective_task_result(tmp_path, "inline", materialize_artifacts=False)
    assert row["status"] == "running" and row["execution_observation"]["state"] == "unknown"
    assert len(reads) == 1


def test_unconsumed_resume_keeps_its_pause_custody(tmp_path):
    _seed(tmp_path, "resuming", "pooled", _is_direct_chat=True,
          budget_pause={"state": "resume_granted", "source_ref": "checkpoint",
                        "grant": {"grant_id": "unconsumed"}})
    assert load_effective_task_result(tmp_path, "resuming")["status"] == "running"
    assert reconcile_orphaned_running_tasks(tmp_path) == 0


@pytest.mark.parametrize("kind", ["direct", "presence", "unknown"])
def test_orphan_publication_rechecks_execution_kind(tmp_path, kind):
    from ouroboros.task_custody import attempt_basis
    from ouroboros.task_status import _still_orphan_at_write

    row = _seed(tmp_path, "raced", "pooled")
    basis = (attempt_basis(row), row.get("updated_at"))
    changed = {**row, "execution_owner": execution_owner_record(tmp_path, {"id": "raced", "_attempt": 1}, kind)}
    applied = []
    fields = {"reason_code": "orphaned_running_after_worker_restart"}
    assert _still_orphan_at_write(tmp_path, "raced", basis, applied, changed, fields) is None
    assert applied == []
    assert _still_orphan_at_write(tmp_path, "raced", basis, applied, row, fields) == fields
    assert applied == [True]


@pytest.mark.serial
def test_real_presence_completion_survives_foreign_orphan_sweep(tmp_path):
    from ouroboros.presence_runner import run_presence_turn
    from tests.test_presence_runner import _admission, _event
    captured = {}

    class Agent:
        def handle_task(self, task):
            tid = task["id"]
            owner = load_task_result(tmp_path, tid)["execution_owner"]
            assert owner["kind"] == "presence"
            # Age the existing start; the independent process must not treat silence
            # and a later foreign worker boot as the death of this real live callback.
            now = time.time()
            write_task_result(tmp_path, tid, "running", ts=_stamp(now - 200))
            write_queue_document(tmp_path / "state" / "queue_snapshot.json", {"ts": _stamp(now), "running": [], "pending": []})
            append_jsonl(tmp_path / "logs" / "events.jsonl", {"ts": _stamp(now - 100), "type": "worker_boot"})
            captured.update(_fresh_reader(tmp_path, tid, reconcile=True))
            write_task_result(tmp_path, tid, "completed", metadata=task["metadata"], result="Actual reply")
            return [{"type": "presence_result", "outcome": "message", "text": "Actual reply", "work_ref": ""}]

    result = run_presence_turn(admission=_admission(), event=_event(), repo_dir=tmp_path,
                               drive_root=tmp_path, agent_factory=lambda **kwargs: Agent())
    assert captured["status"] == "running" and captured["healed"] == 0
    assert result.text == "Actual reply"
    assert load_task_result(tmp_path, result.task_id)["status"] == "completed"


@pytest.mark.parametrize("fields,kind", [
    ({"_is_direct_chat": True}, "direct"),
    ({"_is_direct_chat": True, "metadata": {"source": "presence", "presence": {"binding_id": "b"}}}, "presence"),
    ({"_is_direct_chat": True, "budget_pause": {"state": "resumed", "grant": {"consumed_at": 1}}}, "pooled"),
])
def test_legacy_origin_keeps_inline_unknown_but_consumed_resume_pooled(tmp_path, fields, kind):
    row = _seed(tmp_path, "legacy", **fields)
    assert task_execution_observation(tmp_path, row)["kind"] == kind
    assert load_effective_task_result(tmp_path, "legacy")["status"] == ("failed" if kind == "pooled" else "running")
