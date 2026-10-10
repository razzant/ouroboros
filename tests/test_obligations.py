"""Publication ordering of the addressed sets, including interrupted transitions."""
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from ouroboros import obligations as o
from ouroboros import task_results as results


def read_set(root, name):
    return json.loads(o.path(root, name).read_text(encoding="utf-8"))


def test_generic_set_and_concurrent_writers(tmp_path):
    with ThreadPoolExecutor() as pool:
        list(pool.map(lambda i: o.add(tmp_path, "pause_notices", str(i), {"chat_id": i}), range(12)))
    assert len(o.members(tmp_path, "pause_notices")) == 12
    o.remove(tmp_path, "pause_notices", "1")
    assert "1" not in o.members(tmp_path, "pause_notices")


def test_name_tier_publishes_and_preserves_alternating_writers(tmp_path, monkeypatch):
    monkeypatch.setattr(o.platform, "kernel_file_locks_enforced", lambda _path: False)
    acquire, calls = o.platform.acquire_exclusive_file_lock, []
    def named(path, **kwargs):
        calls.append(kwargs)
        return acquire(path, **kwargs)
    monkeypatch.setattr(o.platform, "acquire_exclusive_file_lock", named)
    monkeypatch.setattr(o.platform, "file_lock_exclusive_nb", lambda _fd: pytest.fail("name tier used a kernel lock"))
    # Alternate the two owners, then contend them, retaining every receipt.
    for tid in ("a", "b", "a", "b"):
        results.write_task_result(tmp_path, tid, "running", progress_note=tid)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda tid: o.add(tmp_path, "nonterminal", tid, {"task_id": tid}), ("c", "d")))
    assert set(o.members(tmp_path, "nonterminal")) == {"a", "b", "c", "d"}
    assert all(call == {"timeout_sec": o.LOCK_TIMEOUT_SEC, "stale_sec": 90.0,
                        "owner_aware_stale": True} for call in calls)
    assert not (tmp_path / "state/obligations.lock").exists()


def test_enforced_tier_is_bounded_and_recovers_after_release(tmp_path, monkeypatch):
    monkeypatch.setattr(o.platform, "kernel_file_locks_enforced", lambda _path: True)
    monkeypatch.setattr(o.platform, "acquire_exclusive_file_lock", lambda *a, **kw: pytest.fail("enforced tier used a name lock"))
    monkeypatch.setattr(o, "LOCK_TIMEOUT_SEC", 0.01)
    lock = tmp_path / "state/obligations.lock"
    lock.parent.mkdir()
    with lock.open("a+b") as holder:
        o.platform.file_lock_exclusive_nb(holder.fileno())
        try:
            with pytest.raises(TimeoutError, match="obligations lock unavailable"):
                with o.locked(tmp_path):
                    pytest.fail("a second descriptor entered the held lock")
        finally:
            o.platform.file_unlock(holder.fileno())
    o.add(tmp_path, "nonterminal", "after-release", {"task_id": "after-release"})
    assert set(o.members(tmp_path, "nonterminal")) == {"after-release"}
    assert lock.exists(), "descriptor ownership keeps the same inode"


def test_missing_and_corrupt_are_not_empty(tmp_path):
    with pytest.raises(o.ObligationsUnavailable):
        o.members(tmp_path, "nonterminal")
    o.add(tmp_path, "nonterminal", "a")
    o.path(tmp_path, "nonterminal").write_text("broken", encoding="utf-8")
    with pytest.raises(o.ObligationsUnavailable):
        o.add(tmp_path, "nonterminal", "b")


@pytest.mark.parametrize("status,fields,expected", [
    ("running", {}, {"nonterminal", "pending_drives"}),
    ("running", {"root_phase_checkpoint": {"post_task_synthesis": "paused"}}, {"synthesis"}),
    ("running", {"delegated_runs_unreconciled": ["r"]}, {"delegated_runs"}),
    ("running", {"child_ref_promotion": {"schema_version": 1, "status": "incomplete", "pending_refs": [{"path": "x"}]}}, {"promotions"}),
    ("completed", {}, {"terminal_projection", "synthesis"}),
])
def test_membership_precedes_result_replace(tmp_path, monkeypatch, status, fields, expected):
    from ouroboros import utils
    original = utils.atomic_write_json

    def crash(path, value, **kwargs):
        if path.parent.name == "task_results":
            for name in expected:
                assert "task" in read_set(tmp_path, name)
            raise OSError("crash at result replace")
        return original(path, value, **kwargs)

    monkeypatch.setattr(utils, "atomic_write_json", crash)
    with pytest.raises(OSError, match="crash"):
        results.write_task_result(tmp_path, "task", status, **fields)
    for name in expected:
        assert "task" in o.members(tmp_path, name)
    assert not (tmp_path / "task_results/task.json").exists()


def test_retirement_follows_durable_result(tmp_path, monkeypatch):
    """A retirement that fails after the result landed never fails the transition: the member stays an
    extra candidate, a rebuild is owed, and the next start's lifecycle import settles it."""
    from ouroboros.startup_migrations import prepare_startup_state

    prepare_startup_state(tmp_path)  # the sets exist and their import is stamped
    results.write_task_result(tmp_path, "task", "running")
    original = o.after_result

    def crash(root, row, **kwargs):
        assert json.loads((tmp_path / "task_results/task.json").read_text(encoding="utf-8"))["status"] == "completed"
        raise OSError("disk full before retirement")

    monkeypatch.setattr(o, "after_result", crash)
    results.write_task_result(tmp_path, "task", "completed")  # bookkeeping never fails the transition
    assert json.loads((tmp_path / "task_results/task.json").read_text(encoding="utf-8"))["status"] == "completed"
    assert "task" in o.members(tmp_path, "nonterminal")  # an extra candidate, never a lost one
    assert (tmp_path / "state/obligations" / o.REBUILD_MARK).exists()
    monkeypatch.setattr(o, "after_result", original)
    assert prepare_startup_state(tmp_path)["imported"] is True  # the owed rebuild runs at the next start
    assert "task" not in o.members(tmp_path, "nonterminal")
    assert not (tmp_path / "state/obligations" / o.REBUILD_MARK).exists()
    assert prepare_startup_state(tmp_path)["imported"] is False  # nothing owed: no enumeration


def test_an_unreadable_set_never_blocks_a_transition_and_the_next_start_rebuilds_it(tmp_path):
    """A torn set file (power loss without a flush, a sync client) is bookkeeping in doubt: task
    writes proceed, readers report it unavailable, and the next start re-imports it from results."""
    from ouroboros.startup_migrations import prepare_startup_state

    prepare_startup_state(tmp_path)
    o.path(tmp_path, "nonterminal").write_bytes(b"")  # torn
    results.write_task_result(tmp_path, "t1", "running")  # a start proceeds
    results.write_task_result(tmp_path, "t0", "completed")
    with pytest.raises(o.ObligationsUnavailable):
        o.members(tmp_path, "nonterminal")
    assert prepare_startup_state(tmp_path)["imported"] is True
    assert set(o.members(tmp_path, "nonterminal")) == {"t1"}


def test_a_set_torn_while_the_server_was_down_is_rebuilt_at_start(tmp_path):
    """No writer noticed (the power was lost after the last write): the start itself finds the
    unreadable set and re-imports it."""
    from ouroboros.startup_migrations import prepare_startup_state

    prepare_startup_state(tmp_path)
    results.write_task_result(tmp_path, "t1", "running")
    o.path(tmp_path, "nonterminal").write_bytes(b'{"t1": {"task_')  # torn mid-write
    assert not (tmp_path / "state/obligations" / o.REBUILD_MARK).exists()
    assert prepare_startup_state(tmp_path)["imported"] is True
    assert set(o.members(tmp_path, "nonterminal")) == {"t1"}


def test_a_torn_watermark_file_is_rewritten_and_the_import_runs(tmp_path):
    from ouroboros.startup_migrations import prepare_startup_state, watermarks

    prepare_startup_state(tmp_path)
    results.write_task_result(tmp_path, "t1", "running")
    (tmp_path / "state" / "migrations.json").write_bytes(b"")  # torn
    assert prepare_startup_state(tmp_path)["imported"] is True
    assert watermarks(tmp_path)["obligations_available"] is True
    assert set(o.members(tmp_path, "nonterminal")) == {"t1"}


def test_custody_open_before_append_closed_after(tmp_path, monkeypatch):
    from ouroboros import delegate_custody as c
    from ouroboros.delegate_custody_current import current_reads

    def append(_path, row):
        current = read_set(tmp_path, "custody_open")
        assert "run:run" in current
        assert not current["run:run"]["custody"]["settled"]
        return True

    monkeypatch.setattr(c, "append_jsonl", append)
    assert c.emit(tmp_path, c.STARTED, {"run_id": "run", "task_id": "task"})
    with current_reads(tmp_path):
        assert len(c.open_runs(tmp_path)) == 1
        assert c.emit(tmp_path, c.SETTLED, {"run_id": "run", "task_id": "task", "state": "succeeded"})
        assert c.open_runs(tmp_path) == []
    assert o.members(tmp_path, "custody_open") == {}
    assert "run" in o.members(tmp_path, "delegated_runs")["task"]["closed_runs"]


def test_failed_close_keeps_open_member(tmp_path, monkeypatch):
    from ouroboros import delegate_custody as c
    assert c.emit(tmp_path, c.STARTED, {"run_id": "run", "task_id": "task"})
    monkeypatch.setattr(c, "append_jsonl", lambda *a, **kw: False)
    assert not c.emit(tmp_path, c.SETTLED, {"run_id": "run", "task_id": "task"})
    assert not o.members(tmp_path, "custody_open")["run:run"]["custody"]["settled"]


def test_a_custody_row_lands_when_its_set_cannot_be_updated(tmp_path, monkeypatch):
    """Delegated custody follows the bookkeeping rule: the event row lands and emit reports its own
    fate when the set is torn or its lock is contended; the next start merges the chain."""
    from contextlib import contextmanager

    from ouroboros import delegate_custody as c
    from ouroboros.startup_migrations import prepare_startup_state

    prepare_startup_state(tmp_path)
    mark = tmp_path / "state" / "obligations" / o.REBUILD_MARK
    o.path(tmp_path, "custody_open").write_bytes(b"")  # torn while running
    assert c.emit(tmp_path, c.STARTED, {"run_id": "run", "task_id": "task"})
    assert mark.exists()
    assert prepare_startup_state(tmp_path)["imported"] is True and not mark.exists()
    assert "run:run" in o.members(tmp_path, "custody_open")

    @contextmanager
    def contended(_root):
        raise TimeoutError("obligations lock unavailable")
        yield

    monkeypatch.setattr(o, "locked", contended)
    assert c.emit(tmp_path, c.SETTLED, {"run_id": "run", "task_id": "task", "state": "succeeded"})
    monkeypatch.undo()
    assert mark.exists()
    rows = c.event_log_path(tmp_path).read_text(encoding="utf-8").splitlines()
    assert json.loads(rows[-1])["type"] == c.SETTLED
    # The missed discharge stays open (the safe direction): the merge keeps the set's member and
    # the reconcile sweep settles the run again; the mark is consumed.
    assert prepare_startup_state(tmp_path)["imported"] is True and not mark.exists()
    assert not o.members(tmp_path, "custody_open")["run:run"]["custody"]["settled"]


def test_an_owed_rebuild_skips_the_snapshot_prune_until_the_set_is_whole(tmp_path, monkeypatch):
    """A custody row can land while its set could not be updated, so the run is missing from the set
    until the next start merges it: the snapshot prune skips loudly meanwhile, then prunes again."""
    from ouroboros import server_maintenance, subagent_worktrees

    monkeypatch.setattr(server_maintenance, "DATA_DIR", tmp_path)
    pruned = []
    monkeypatch.setattr(subagent_worktrees, "prune_execution_snapshots", lambda keep: pruned.append(keep) or {})
    mark = tmp_path / "state" / "obligations" / o.REBUILD_MARK
    mark.parent.mkdir(parents=True)
    mark.write_text("custody: TimeoutError\n", encoding="utf-8")
    server_maintenance._prune_delegated_snapshots()
    assert pruned == []
    rows = (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(rows[-1]) | {"ts": ""} == {"ts": "", "type": "delegated_snapshot_prune_skipped",
                                                "reason": "obligations_rebuild_owed"}
    mark.unlink()
    server_maintenance._prune_delegated_snapshots()
    assert len(pruned) == 1


def test_first_import_classifies_once_and_stamps(tmp_path, monkeypatch):
    from ouroboros import startup_migrations as m
    directory = tmp_path / "task_results"
    directory.mkdir()
    for tid, status, extra in [
        ("done", "completed", {}),
        ("running", "running", {}),
        ("paused", "completed", {"root_phase_checkpoint": {"post_task_synthesis": "paused"}}),
        ("owed", "completed", {"canonical_terminal_projection_origin": "terminal_transition"}),
    ]:
        (directory / f"{tid}.json").write_text(json.dumps({"task_id": tid, "status": status,
            "_schema_version": 1, **extra}), encoding="utf-8")
    (directory / "broken.json").write_text("broken", encoding="utf-8")
    report = m.prepare_startup_state(tmp_path)
    assert report["imported"]
    assert set(o.members(tmp_path, "nonterminal")) == {"running"}
    assert set(o.members(tmp_path, "synthesis")) == {"paused", "owed"}
    assert set(o.members(tmp_path, "terminal_projection")) == {"owed"}
    assert "task:broken" in o.members(tmp_path, "unknowns")
    assert m.watermarks(tmp_path)["cancel_latches_generation"] == 7
    monkeypatch.setattr(m, "_result_records", lambda *a: pytest.fail("second history parse"))
    assert m.prepare_startup_state(tmp_path) == {"imported": False}


def test_import_failure_does_not_stamp_complete(tmp_path, monkeypatch):
    from ouroboros import startup_migrations as m
    from ouroboros import delegate_custody_current as current
    monkeypatch.setattr(current, "rebuild", lambda *a, **kw: (_ for _ in ()).throw(OSError("interrupted")))
    with pytest.raises(OSError):
        m.prepare_startup_state(tmp_path)
    assert "obligations_generation" not in m.watermarks(tmp_path)


def test_cancel_migration_stamps_only_success_and_per_generation(tmp_path, monkeypatch):
    from ouroboros import startup_migrations as m, cancel_intents as c
    calls = []
    monkeypatch.setattr(c, "migrate_legacy_cancel_latches", lambda *a, **kw: calls.append(kw) or [])
    m.migrate_cancel_latches(tmp_path, generation=6)
    m.migrate_cancel_latches(tmp_path, generation=6)
    m.migrate_cancel_latches(tmp_path, generation=7)
    m.migrate_cancel_latches(tmp_path, generation=7)
    assert len(calls) == 2
    monkeypatch.setattr(c, "migrate_legacy_cancel_latches", lambda *a, **kw: (_ for _ in ()).throw(OSError()))
    with pytest.raises(OSError):
        m.migrate_cancel_latches(tmp_path, generation=8)
    assert m.watermarks(tmp_path)["cancel_latches_generation"] == 7


def test_recovery_reads_open_members_among_2000_completed(tmp_path, monkeypatch):
    from ouroboros import startup_migrations as m, task_status as status, agent_task_pipeline as pipeline
    from ouroboros import terminal_projection as projection
    directory = tmp_path / "task_results"
    directory.mkdir()
    for i in range(2000):
        (directory / f"done{i}.json").write_text(json.dumps({"_schema_version": 1,
            "task_id": f"done{i}", "status": "completed"}), encoding="utf-8")
    m.prepare_startup_state(tmp_path)
    results.write_task_result(tmp_path, "active", "running")
    results.write_task_result(tmp_path, "paused", "completed", root_phase_checkpoint={"post_task_synthesis": "paused"})
    results.write_task_result(tmp_path, "owed", "completed")
    reads = []
    original = results.read_text_across_replace
    def counted(path):
        if path.parent == directory:
            reads.append(path.stem)
        return original(path)
    monkeypatch.setattr(results, "read_text_across_replace", counted)
    monkeypatch.setattr(results, "list_task_results", lambda *a, **kw: pytest.fail("history enumeration"))
    monkeypatch.setattr(status, "effective_task_result", lambda root, row, **kw: row)
    from ouroboros import post_task_synthesis as synthesis
    from supervisor import owner_pause_control
    monkeypatch.setattr(synthesis, "revoke_late_phase_grant", lambda *a, **kw: None)
    monkeypatch.setattr(owner_pause_control, "retain_late_phase_latch", lambda *a: None)
    monkeypatch.setattr(pipeline, "_settle_terminal_projection", lambda *a, **kw: None)
    monkeypatch.setattr(projection, "settle_terminal_projection", lambda *a, **kw: "settled")
    status.reconcile_orphaned_running_tasks(tmp_path)
    assert reads == ["active"]
    reads.clear()
    pipeline.recover_pending_root_post_task_synthesis(tmp_path)
    assert sorted(reads) == ["owed", "paused"]
    reads.clear()
    projection.reconcile_terminal_projections(tmp_path)
    assert reads == ["owed"]


def test_boot_custody_never_folds_chain(tmp_path, monkeypatch):
    from ouroboros import startup_migrations as m, delegate_custody as c, server_maintenance as sm
    m.prepare_startup_state(tmp_path)
    monkeypatch.setattr(sm, "DATA_DIR", tmp_path)
    monkeypatch.setattr(sm, "_installed_skill_names", lambda: set())
    from ouroboros import process_custody
    monkeypatch.setattr(process_custody, "reap_orphaned_processes", lambda *a, **kw: [])
    monkeypatch.setattr(c, "custody_rows", lambda *a, **kw: pytest.fail("custody history fold"))
    monkeypatch.setattr(c, "_iter_rows", lambda *a, **kw: pytest.fail("custody history fold"))
    sm._startup_custody_sweep()


def test_addressed_custody_backfill_keeps_terminal_receipt_without_history(tmp_path, monkeypatch):
    from ouroboros import startup_migrations as m, delegate_custody as c, delegate_terminal as terminal
    from ouroboros.delegate_custody_current import current_reads
    m.prepare_startup_state(tmp_path)
    c.emit(tmp_path, c.STARTED, {"run_id": "run", "task_id": "task"})
    results.write_task_result(tmp_path, "task", "completed", delegated_runs_unreconciled=["run"])
    c.emit(tmp_path, c.SETTLED, {"run_id": "run", "task_id": "task", "state": "succeeded"})
    monkeypatch.setattr(c, "custody_rows", lambda *a, **kw: pytest.fail("history fold"))
    monkeypatch.setattr(c, "task_execution_evidence", lambda *a, **kw: pytest.fail("historical evidence fold"))
    with current_reads(tmp_path):
        assert terminal.backfill_terminal_reconciliations(tmp_path) == ["task"]
    row = results.load_task_result(tmp_path, "task")
    assert row["delegated_runs_unreconciled"] == []
    assert row["delegate_terminal_reconciliation"]["terminal_runs"][0]["state"] == "succeeded"
    assert "task" not in o.members(tmp_path, "delegated_runs")


def test_downgrade_body_head_reimports_new_legacy_latch(tmp_path, monkeypatch):
    from ouroboros import startup_migrations as m
    from ouroboros.utils import atomic_write_json
    m.prepare_startup_state(tmp_path)
    m.stamp(tmp_path, observed_state_sha="current")
    atomic_write_json(tmp_path / "state/state.json", {"current_sha": "old"})
    atomic_write_json(tmp_path / "task_results/old-task.json", {"task_id": "old-task", "status": "cancel_requested"})
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: pytest.fail("version subprocess"))
    assert m.prepare_startup_state(tmp_path, repo_dir=tmp_path)["imported"]
    assert results.load_task_result(tmp_path, "old-task", strict=True)["_schema_version"] == 1
    monkeypatch.setattr(m, "_result_records", lambda *a: pytest.fail("repeat generation"))
    assert not m.prepare_startup_state(tmp_path, repo_dir=tmp_path)["imported"]
