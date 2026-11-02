"""Housekeeping stays live across a server generation. Age-only retirement runs in the child
on producer/deadline/startup facts, with task sources behind the recovery gate. Working copies
stay on the custody pass; the memory-journal observation stays once per generation."""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from ouroboros import server_maintenance as sm
from ouroboros.startup_migrations import prepare_startup_state
from ouroboros.task_results import write_task_result

_OLD = time.time() - 30 * 86400  # past the default 7-day GC retention


def _aged(path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    os.utime(path, (_OLD, _OLD))
    return path


@pytest.fixture
def root(tmp_path, monkeypatch):
    data = tmp_path / "data"
    prepare_startup_state(data)
    monkeypatch.setattr(sm, "DATA_DIR", data)
    monkeypatch.setattr(sm, "_STARTUP_RECOVERY_GAPS", [False])
    return data


def _reconcile_pass(monkeypatch):
    """One real reconcile pass, its unrelated steps idle."""
    monkeypatch.setattr(sm, "_periodic_zombie_reconcile", lambda **_kw: None)
    monkeypatch.setattr(sm, "_run_drive_custody_pass", lambda *_a: None)
    monkeypatch.setattr(sm, "_tick_maintenance_child", lambda: None)
    monkeypatch.setattr("supervisor.workspace_preparation.recover_interrupted_preparations", lambda *a, **k: None)
    monkeypatch.setattr("ouroboros.review_operation.retire_dead_review_pointers_softly", lambda *a, **k: None)
    monkeypatch.setattr("ouroboros.late_review_marks.replay_owed_late_reviews", lambda *a, **k: None)
    sm._run_periodic_reconcile_sweep([0.0], latch=SimpleNamespace(release=lambda: None))
    from ouroboros.maintenance import run_job
    run_job(sm.DATA_DIR, "runtime_retirement")


def _call_queue(root, running=()):
    from ouroboros.utils import atomic_write_json
    stamp = {'generation': 'call-test', 'ts': datetime.fromtimestamp(time.time(), timezone.utc).isoformat()}
    atomic_write_json(root / 'state/queue_snapshot.json', {**stamp, 'running': [{'id': task} for task in running]})
    atomic_write_json(root / 'state/queue.alive', stamp)


@pytest.mark.parametrize('days', [1, 7, 30])
def test_call_wire_ages_under_shared_retention_only_after_task_is_proven_over(root, monkeypatch, days):
    from ouroboros import maintenance, maintenance_jobs, maintenance_retirement, task_status

    monkeypatch.setenv('OUROBOROS_GC_RETENTION_DAYS', str(days))
    now = time.time()
    journals = {}
    for task_id, status, age in [('old', 'completed', days + 1), ('young', 'completed', days - .5),
                                 ('running', 'running', days + 1), ('held', 'completed', days + 1),
                                 ('unknown', None, days + 1)]:
        path = root / 'task_results/artifacts' / task_id / 'conversation-wire.jsonl'
        _aged(path, 'all original wire bytes\n')
        os.utime(path, (now - age * 86400,) * 2)
        journals[task_id] = path
        if status:
            write_task_result(root, task_id, status)
    adjacent = _aged(journals['old'].parent / 'caption.jsonl', 'canonical evidence')
    # Use the real queue-snapshot liveness predicate, including its unknown case.
    _call_queue(root, ['held'])
    assert task_status.task_has_live_queue_ownership(root, 'held')
    assert not task_status.task_has_live_queue_ownership(root, 'old')
    assert maintenance_jobs.job(root, 'runtime_retirement')['next_due'] <= now
    assert maintenance.run_job(root, 'runtime_retirement')['state'] == 'done'
    assert not journals['old'].exists()
    assert all(path.exists() for name, path in journals.items() if name != 'old')
    assert adjacent.read_text() == 'canonical evidence'
    # Once live ownership ends the already-owed pass removes only the aged journal.
    _call_queue(root)
    assert maintenance.run_job(root, 'runtime_retirement')['state'] == 'done'
    assert not journals['held'].exists() and journals['running'].exists()
    deadline = journals['young'].stat().st_mtime + days * 86400
    assert maintenance_jobs.job(root, 'runtime_retirement')['next_due'] == deadline
    with monkeypatch.context() as clock:
        clock.setattr(maintenance_retirement.time, 'time', lambda: deadline + 1)
        _call_queue(root)
        assert maintenance.run_job(root, 'runtime_retirement')['state'] == 'done'
        assert not journals['young'].exists()
    assert journals['running'].read_text() == journals['unknown'].read_text() == 'all original wire bytes\n'


def test_call_wire_sweep_keeps_unknown_ownership_and_symlinks(root, monkeypatch):
    from ouroboros import maintenance_retirement, task_status

    journal = _aged(root / 'task_results/artifacts/done/conversation-wire.jsonl', 'whole wire')
    write_task_result(root, 'done', 'completed')
    # An absent snapshot cannot prove that the terminal task has no live owner.
    (root / 'state/queue_snapshot.json').unlink(missing_ok=True)
    assert task_status.task_has_live_queue_ownership(root, 'done')
    maintenance_retirement.prune_call_wire_journals(root)
    assert journal.read_text() == 'whole wire'
    _call_queue(root)
    outside = _aged(root / 'elsewhere/conversation-wire.jsonl', 'outside')
    for task_id in ('linked-dir', 'linked-file'):
        write_task_result(root, task_id, 'completed')
    (journal.parent.parent / 'linked-dir').symlink_to(outside.parent, target_is_directory=True)
    linked_file = journal.parent.parent / 'linked-file/conversation-wire.jsonl'
    linked_file.parent.mkdir()
    linked_file.symlink_to(outside)
    # Even a terminal row cannot authorize deletion with a live lineage process.
    monkeypatch.setattr('ouroboros.body_candidate.live_lineage_processes', lambda *_: [123])
    maintenance_retirement.prune_call_wire_journals(root)
    assert journal.exists()
    monkeypatch.setattr('ouroboros.body_candidate.live_lineage_processes', lambda *_: [])
    maintenance_retirement.prune_call_wire_journals(root)
    assert not journal.exists()
    assert outside.read_text() == 'outside' and linked_file.is_symlink()


def test_the_age_only_retirers_run_again_on_a_later_pass_and_trees_wait_out_a_recovery_gap(root, monkeypatch):
    from ouroboros import maintenance_jobs, maintenance_retirement, subagent_worktrees
    from ouroboros.retention import get_gc_retention_days

    def observations():
        rows = (root / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
        return sum(json.loads(row).get("type") == "memory_journal_observation" for row in rows)

    for name in ("prune_orphans", "prune_execution_snapshots"):
        monkeypatch.setattr(subagent_worktrees, name, lambda *a, **k: pytest.fail("working copies on a reconcile pass"))
    write_task_result(root, "settled", "completed", result="done", ts="2000-01-01T00:00:00+00:00")
    tree = _aged(root / "task_trees" / "settled" / "blackboard.jsonl").parent
    service = _aged(root / "services" / "settled" / "service.log")
    first = [_aged(root / "uploads" / "screenshots" / "first.png"),
             _aged(root / "state" / "code_intel" / "stale-root" / "inventory.json", "{}"),
             _aged(root / "state" / "extension_reconcile" / "failed" / "skill.json", "{}"),
             _aged(root / "state" / "delegate_supervision" / "settled.json", "{}")]

    sm._startup_prune_sweeps(preserve_task_sources=True, recovery_report={"errors": ["ownership unknown"]})
    assert observations() == 1 and all(path.exists() for path in first), "boot observes and prunes nothing"
    _reconcile_pass(monkeypatch)
    assert not any(path.exists() for path in first)
    assert tree.exists() and service.exists(), "a startup recovery gap keeps task trees and service logs"

    later = [_aged(root / "uploads" / "screenshots" / "later.png"),
             _aged(root / "state" / "delegate_supervision" / "settled.json", "{}")]
    stamp = time.time()
    for path in later:
        os.utime(path, (stamp, stamp))
        maintenance_retirement.artifact_changed(root, timestamp=stamp)
    deadline = stamp + get_gc_retention_days() * 86400
    assert maintenance_jobs.job(root, "runtime_retirement")["next_due"] == deadline
    _reconcile_pass(monkeypatch)
    assert all(path.exists() for path in later), "publication schedules age eligibility, not immediate deletion"
    with monkeypatch.context() as clock:
        clock.setattr("ouroboros.retention.time.time", lambda: deadline - 1)
        _reconcile_pass(monkeypatch)
        assert all(path.exists() for path in later)
        clock.setattr("ouroboros.retention.time.time", lambda: deadline + 1)
        _reconcile_pass(monkeypatch)
    assert not any(path.exists() for path in later), "the recorded deadline wakes retirement in the same process"
    assert tree.exists() and service.exists()
    assert observations() == 1, "the journal observation is boot's, never a pass's"

    sm._startup_prune_sweeps()  # the next generation, its recovery complete
    _reconcile_pass(monkeypatch)
    assert not tree.exists() and not service.exists()
    assert observations() == 2


def test_a_generation_closed_between_two_reconcile_prunes_starts_no_further_one(root, monkeypatch):
    from ouroboros import code_intelligence, delegate_state_sweep, skill_uninstall_state

    closed, calls = threading.Event(), []

    def first(_root):
        calls.append("delegate")
        closed.set()
        return {}

    monkeypatch.setattr(delegate_state_sweep, "sweep_settled_delegate_state", first)
    monkeypatch.setattr(skill_uninstall_state, "sweep_uninstalled_skill_state", lambda _root: calls.append("skills") or {})
    monkeypatch.setattr(code_intelligence, "prune_stale_code_intel_roots", lambda _root: calls.append("caches") or {})
    from ouroboros.maintenance import run_job
    run_job(root, "runtime_retirement", should_stop=closed.is_set)
    assert calls == ["delegate"], "the generation closed: no further retirer starts"
    run_job(root, "runtime_retirement")
    assert calls == ["delegate", "delegate", "skills", "caches"], "an open generation runs every step"


@pytest.fixture
def custody_pass(root, monkeypatch):
    """The 600 s custody pass with its reaping and delegated-run steps idle."""
    from ouroboros import claudexor_daemon, process_custody, process_scope, terminal_projection

    monkeypatch.setattr(terminal_projection, "reconcile_terminal_projections", lambda *_a: None)
    monkeypatch.setattr(claudexor_daemon, "get_owned_daemon", lambda: SimpleNamespace(
        clear_start_failure_latch=lambda *, cleared_by: False))
    monkeypatch.setattr(process_custody, "reap_orphaned_processes", lambda *a, **k: [])
    monkeypatch.setattr(process_scope, "settle_task_scopes", lambda *a, **k: None)
    monkeypatch.setattr(sm, "_installed_skill_names", lambda: None)
    monkeypatch.setattr(sm, "_reconcile_delegated_runs", lambda *a, **k: None)
    monkeypatch.setattr(sm, "_cursor_refresh_settled_terminals", lambda *a, **k: None)
    monkeypatch.setattr(sm, "_prune_review_checkouts", lambda stop_event=None: None)

    def run(stop_event=None):
        latch = threading.Lock()
        latch.acquire()
        sm._run_periodic_custody_sweep(stop_event, latch)
        assert not latch.locked()
    return run


def test_every_custody_pass_prunes_the_working_copies_with_the_supervisor_probe(custody_pass, monkeypatch):
    from ouroboros import delegate_custody, subagent_worktrees
    import supervisor.queue as queue

    def probe(_task_id):
        return False

    calls = []
    monkeypatch.setattr(queue, "task_settlement_liveness", probe)
    monkeypatch.setattr(subagent_worktrees, "prune_orphans",
                        lambda **kw: calls.append(("worktrees", kw)) or {"removed": 0, "kept": 1})
    monkeypatch.setattr(subagent_worktrees, "prune_execution_snapshots",
                        lambda open_ids: calls.append(("snapshots", open_ids)) or {"removed": [], "kept": []})
    monkeypatch.setattr(delegate_custody, "custody_log_unreadable", lambda _root: False)
    monkeypatch.setattr(delegate_custody, "open_snapshot_ids", lambda _root: {"snapshot-open"})

    custody_pass()
    custody_pass()
    assert calls == [("worktrees", {"live": probe}), ("snapshots", {"snapshot-open"})] * 2

    calls.clear()
    closed = threading.Event()
    closed.set()
    custody_pass(closed)
    assert calls == [], "a closed generation retires nothing"
