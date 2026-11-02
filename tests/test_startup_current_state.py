"""Boot gaps, append ordering, notice receipts and the housekeeping that waits on recovery."""
import inspect
import json
from contextlib import contextmanager

import pytest

from ouroboros import obligations as o, startup_migrations as migrations
from ouroboros import server_maintenance as maintenance, task_results as results
from tests._store_shared import seed_result, unavailable_store, unreadable_result


def test_bad_cancel_row_is_a_gap_while_good_row_migrates(tmp_path, monkeypatch):
    from ouroboros import cancel_intents
    from ouroboros.utils import atomic_write_json
    for task in ("good", "bad"):
        atomic_write_json(tmp_path / f"task_results/{task}.json", {"task_id": task, "status": "cancel_requested"})
    request = cancel_intents.request_cancel
    def fail_one(root, task, **kwargs):
        if task == "bad":
            raise OSError("one row cannot migrate")
        return request(root, task, **kwargs)
    monkeypatch.setattr(cancel_intents, "request_cancel", fail_one)
    assert migrations.prepare_startup_state(tmp_path, strict=False)["imported"]
    assert cancel_intents.active_intent(tmp_path, "good")
    assert results.load_task_result(tmp_path, "good", strict=True)["_schema_version"] == 1
    assert o.members(tmp_path, "unknowns")["task:bad"]["reason"] == "cancel_latch_migration_failed"
    assert migrations.watermarks(tmp_path)["cancel_latches_generation"] == migrations.SCHEMA_GENERATION
    monkeypatch.setattr(cancel_intents, "request_cancel", request)
    migrations.prepare_startup_state(tmp_path, rebuild=True)
    assert cancel_intents.active_intent(tmp_path, "bad")
    assert not o.members(tmp_path, "unknowns")


def test_unreadable_import_leaves_boot_available_and_explicit_repair_fails(tmp_path, monkeypatch):
    """The importer cannot read the result files: boot continues with an unavailable store
    (every door answers typed), no store generation is stamped, and the explicit repair fails."""
    from ouroboros import store, store_switch, store_import
    def unreadable(root):
        raise PermissionError("directory unreadable")
    monkeypatch.setattr(store_import, "_read_results", unreadable)
    assert migrations.prepare_startup_state(tmp_path, strict=False)["status"] == "unavailable"
    assert "strict=False" in inspect.getsource(store_switch._run)  # the boot door's import owner
    with store.importing(tmp_path), pytest.raises(o.ObligationsUnavailable):  # the boot door keeps its window open
        o.members(tmp_path, "nonterminal")
    assert not store.generation_stamped(tmp_path) and not store.store_path(tmp_path).exists()
    rows = [json.loads(line) for line in (tmp_path / "logs/supervisor.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["type"] == "startup_migration" and rows[-1]["phase"] == "failed"
    with pytest.raises(PermissionError):
        migrations.prepare_startup_state(tmp_path, rebuild=True)
    monkeypatch.undo()
    assert migrations.prepare_startup_state(tmp_path)["imported"]
    assert o.members(tmp_path, "nonterminal") == {}


def test_migration_diagnostic_failure_does_not_close_boot(tmp_path, monkeypatch):
    monkeypatch.setattr(migrations, "_prepare_startup_state", lambda *a, **kw:
        (_ for _ in ()).throw(PermissionError("import unavailable")))
    monkeypatch.setattr(migrations, "append_jsonl", lambda *a, **kw:
        (_ for _ in ()).throw(OSError("logs unavailable")))
    assert migrations.prepare_startup_state(tmp_path, strict=False)["status"] == "unavailable"
    with pytest.raises(PermissionError, match="import unavailable"):
        migrations.prepare_startup_state(tmp_path, rebuild=True)


@pytest.mark.parametrize("kind", ["delegate_run_started", "delegate_run_settled"])
def test_custody_append_can_write_results_and_rows(tmp_path, monkeypatch, kind):
    """The chain append of a custody event holds no store transaction: a result write and an
    owner-held row published from inside it land, and the event's own rows follow."""
    from ouroboros import delegate_custody as custody
    from ouroboros import delegate_custody_current as current
    migrations.prepare_startup_state(tmp_path)
    assert custody.emit(tmp_path, custody.STARTED, {"run_id": "run", "task_id": "task"})
    def append(_path, _row, **_kwargs):
        assert "run" in current.runs(tmp_path, "state=?", (current.HELD,)) or kind == custody.SETTLED
        results.write_task_result(tmp_path, "other", "running")
        o.add(tmp_path, "custody_open", "invocation:other", {"request": {"invocation_id": "other"}})
        return True
    monkeypatch.setattr(current, "append_jsonl", append)
    assert custody.emit(tmp_path, kind, {"run_id": "run", "task_id": "task", "state": "succeeded"})
    assert "other" in o.members(tmp_path, "nonterminal")
    assert "invocation:other" in o.members(tmp_path, "custody_open")
    assert bool(custody.open_runs(tmp_path)) == (kind == custody.STARTED)
    if kind == custody.SETTLED:
        assert "run" in o.members(tmp_path, "delegated_runs")["task"]["closed_runs"]


def test_explicit_rebuild_replaces_custody_rows_from_the_chain(tmp_path):
    """The explicit repair refolds the custody rows from the retained event chain: a run row
    and a pending invocation no event supports are replaced, a derived set is untouched."""
    from ouroboros import delegate_custody as custody, store
    migrations.prepare_startup_state(tmp_path)
    results.write_task_result(tmp_path, "task", "running")
    o.add(tmp_path, "custody_open", "invocation:ghost", {"request": {"invocation_id": "ghost"}})
    with store.open_store(tmp_path, write=True) as txn:
        txn.put_custody("ghost", {"run_id": "ghost", "task_id": "task"}, state="held")
    migrations.prepare_startup_state(tmp_path, rebuild=True)
    assert o.members(tmp_path, "custody_open") == {}
    assert custody.replay(tmp_path) == {}
    assert set(o.members(tmp_path, "nonterminal")) == {"task"}


@pytest.mark.parametrize("settled", [False, True])
def test_custody_nested_receipts_survive_fresh_current_read(tmp_path, settled):
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_custody_current import snapshot
    migrations.prepare_startup_state(tmp_path)
    source = {"path": "source.txt"}
    request = {"source": source, "complete_sha256": "a" * 64, "complete_chars": 2}
    assert custody.emit(tmp_path, custody.STARTED, {
        "run_id": "run", "task_id": "task", "work_order_coverage": "partial",
        "work_order_fingerprint": "a" * 64, "work_order_source_request": request,
    })
    if settled:
        assert custody.emit(tmp_path, custody.SETTLED, {"run_id": "run", "state": "succeeded"})
    for interaction in ("first", "second"):
        assert custody.emit(tmp_path, custody.SOURCE_RANGE_DELIVERY_CONFIRMED, {
            "run_id": "run", "interaction_id": interaction, "source": source,
            "complete_sha256": "a" * 64, "start_char": 0, "end_char": 2,
            "text_sha256": "b" * 64, "text_chars": 2,
        })
    actual = snapshot(tmp_path)["state"]["run"]._source_delivery_confirmations
    assert actual == custody.replay(tmp_path)["run"]._source_delivery_confirmations
    assert [receipt["interaction_id"] for receipt in actual] == ["first", "second"]


@pytest.mark.parametrize("debt", ["cost-owed", "cost-paid", "later-receipt"])
@pytest.mark.parametrize("writer_first", [True, False])
def test_custody_ack_serializes_with_result_membership(tmp_path, monkeypatch, writer_first, debt):
    """The acknowledgement reads the result row and retires the closing receipts it observed in ONE
    store transaction: a debt the result shows (``delegated_runs_unreconciled``), a receipt closed
    after the observation and an unpaid ``cost_revision`` each survive it. Only the money owner
    retires a revision, the one it observed: a revision raised while it publishes stays owed."""
    from ouroboros import delegate_custody as custody, delegate_terminal, store
    from ouroboros.delegate_custody_current import snapshot
    from supervisor import events_task_done as done, queue

    def owed():
        with store.open_store(tmp_path, write=False) as txn:
            return txn.get_row("delegated_runs", "task")

    monkeypatch.setattr(queue, "task_has_live_ownership", lambda *a, **kw: False)
    migrations.prepare_startup_state(tmp_path)
    results.write_task_result(tmp_path, "task", "completed")
    custody.emit(tmp_path, custody.STARTED, {"run_id": "old", "task_id": "task"})
    custody.emit(tmp_path, custody.SETTLED, {"run_id": "old", "task_id": "task", "state": "succeeded"})
    if debt == "cost-paid":
        delegate_terminal._refresh_custody_cost(tmp_path, "task")
    assert set(owed()["closed_runs"]) == {"old"} and ("cost_revision" in owed()) is (debt != "cost-paid")
    observed = snapshot(tmp_path)
    if debt == "later-receipt":  # every custody change owes money; this receipt closed after the observation
        custody.emit(tmp_path, custody.STARTED, {"run_id": "late", "task_id": "task"})
        custody.emit(tmp_path, custody.SETTLED, {"run_id": "late", "task_id": "task", "state": "succeeded"})
    before = owed()
    if writer_first:
        results.write_task_result(tmp_path, "task", "completed", delegated_runs_unreconciled=["new"])
    opened = []
    real_open = store.open_store
    @contextmanager
    def tracked(root, **kwargs):
        with real_open(root, **kwargs) as txn:
            opened.append(kwargs.get("write"))
            yield txn
    with monkeypatch.context() as patch:
        patch.setattr(store, "open_store", tracked)
        delegate_terminal._ack_current_receipts(tmp_path, "task", observed)
    assert opened == [True], "one write transaction covers the row read and the retirement"
    unobserved = {rid: raw for rid, raw in before["closed_runs"].items() if rid != "old"}
    if debt == "cost-paid" and not writer_first:
        assert owed() is None, "nothing left to owe: the row retires"
    else:
        assert owed() == {**before, "closed_runs": unobserved}, "the exact revision and later receipts survive"
    if not writer_first:
        assert ("task" in o.members(tmp_path, "delegated_runs")) is (debt != "cost-paid")
        results.write_task_result(tmp_path, "task", "completed", delegated_runs_unreconciled=["new"])
    assert "task" in o.members(tmp_path, "delegated_runs")
    if debt != "cost-paid":
        refresh, raced = done._refresh_terminal_task_cost, []

        def racing(*args, **kwargs):
            published = refresh(*args, **kwargs)
            if not raced:
                raced.append(custody.emit(tmp_path, custody.STARTED, {"run_id": "racer", "task_id": "task"}))
            return published

        monkeypatch.setattr(done, "_refresh_terminal_task_cost", racing)
        delegate_terminal._refresh_custody_cost(tmp_path, "task")
        assert owed()["cost_revision"] > before["cost_revision"], "the revision raised meanwhile stays owed"
        delegate_terminal._refresh_custody_cost(tmp_path, "task")
        assert "cost_revision" not in (owed() or {})
        assert set((owed() or {}).get("closed_runs", {})) == set(unobserved)


def test_unavailable_recovery_sets_do_not_escape_a_boot_door(tmp_path, monkeypatch):
    migrations.prepare_startup_state(tmp_path)
    unavailable_store(monkeypatch)
    monkeypatch.setattr(maintenance, "_startup_live_task_ids", lambda *a, **kw: set())
    report = maintenance._run_startup_task_recovery(tmp_path, tmp_path, skip_live_data=False, prior_worker_pids=set())
    assert "*" in report["unresolved"] and report["errors"]


def test_notice_receipts_import_once_and_publication_never_reads_chat(tmp_path, monkeypatch):
    from ouroboros import notice_receipts, utils
    from supervisor import message_bus
    row = {"chat_id": 7, "direction": "system", "type": "reviewer_default_notice"}
    utils.append_jsonl(tmp_path / "logs/chat.jsonl", row)
    migrations.prepare_startup_state(tmp_path)
    assert notice_receipts.recorded(tmp_path, 7, row["type"])
    assert not notice_receipts.recorded(tmp_path, 8, row["type"])
    monkeypatch.setattr(utils, "jsonl_chain_handles", lambda *a, **kw: pytest.fail("chat history read"))
    monkeypatch.setattr(utils, "iter_jsonl_chain_objects", lambda *a, **kw: pytest.fail("chat history read"))
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    message_bus.log_chat("system", 8, 0, "one notice", record_type="reviewer_default_notice", require_write=True)
    assert notice_receipts.recorded(tmp_path, 8, row["type"])
    assert not migrations.prepare_startup_state(tmp_path)["imported"]


def test_prunes_wait_off_ready_path_and_trees_wait_for_readable_recovery_links(tmp_path, monkeypatch):
    """Startup prunes nothing; a due child needs readable recovery links before retiring trees.
    Repair re-arms a resting pass. The worktree family remains the custody pass's."""
    from ouroboros import headless, maintenance_jobs, subagent_worktrees
    migrations.prepare_startup_state(tmp_path)
    monkeypatch.setattr(maintenance, "DATA_DIR", tmp_path)
    monkeypatch.setattr(maintenance, "_STARTUP_RECOVERY_GAPS", [False])
    calls = []
    monkeypatch.setattr(headless, "prune_task_trees", lambda *a, **kw: calls.append("trees"))
    monkeypatch.setattr(subagent_worktrees, "prune_orphans", lambda **kw: pytest.fail("worktrees on a reconcile pass"))
    o.add(tmp_path, "pending_drives", "task")
    maintenance._startup_prune_sweeps(preserve_task_sources=True, recovery_report={"unresolved": ["task"]})
    assert calls == []
    _retire(tmp_path)
    assert calls == [], "a recovery link that cannot be read keeps every tree for this pass"
    o.remove(tmp_path, "pending_drives", "task")
    assert _retire(tmp_path) is None and calls == [], "a recovery-set change alone does not walk history"
    maintenance_jobs.record(tmp_path, "runtime_retirement", state="pending")  # explicit repair after the gap clears
    _retire(tmp_path)
    assert _retire(tmp_path) is None
    assert calls == ["trees"], "an unchanged generation rests"


@pytest.mark.parametrize("gap", [None, "unreadable", "coarse"])
def test_reconcile_prunes_keep_live_tree_and_wait_out_recovery_gaps(tmp_path, monkeypatch, gap):
    import os
    migrations.prepare_startup_state(tmp_path)
    monkeypatch.setattr(maintenance, "DATA_DIR", tmp_path)
    monkeypatch.setattr(maintenance, "_STARTUP_RECOVERY_GAPS", [False])
    running = results.write_task_result(tmp_path, "running", "running")
    results.write_task_result(tmp_path, "settled", "completed", ts="2000-01-01T00:00:00+00:00")
    for task in ("running", "settled"):
        (tmp_path / "task_trees" / task).mkdir(parents=True)
    service = tmp_path / "services/settled/service.log"
    service.parent.mkdir(parents=True)
    service.write_text("retained service output", encoding="utf-8")
    os.utime(service, (1, 1))
    unreadable = unreadable_result(monkeypatch, "running") if gap == "unreadable" else {"on": False}
    maintenance._startup_prune_sweeps(preserve_task_sources=bool(gap),
        recovery_report={"errors": ["ownership unknown"]} if gap == "coarse" else {"unresolved": ["running"]})
    _retire(tmp_path)
    if gap:
        assert (tmp_path / "task_trees/settled").exists() and service.exists()
        _retire(tmp_path)
        assert (tmp_path / "task_trees/settled").exists() and service.exists(), "a later pass keeps them too"
        unreadable["on"] = False
        assert results.load_task_result(tmp_path, "running", strict=True) == running
        maintenance._startup_prune_sweeps()  # the next generation's boot, with recovery complete
        _retire(tmp_path)
    assert not (tmp_path / "task_trees/settled").exists() and not service.exists()
    assert (tmp_path / "task_trees/running").exists()


def test_a_failed_prune_step_spares_the_others_and_the_next_pass_runs_it_again(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ouroboros import headless, owner_mailbox, review_operation
    from ouroboros.tools import services
    migrations.prepare_startup_state(tmp_path)
    monkeypatch.setattr(maintenance, "DATA_DIR", tmp_path)
    monkeypatch.setattr(maintenance, "_STARTUP_RECOVERY_GAPS", [False])
    monkeypatch.setattr(maintenance, "_periodic_zombie_reconcile", lambda **kw: None)
    monkeypatch.setattr(maintenance, "_run_drive_custody_pass", lambda *a: None)
    monkeypatch.setattr(maintenance, "_tick_maintenance_child", lambda: None)
    monkeypatch.setattr(review_operation, "retire_dead_review_pointers_softly", lambda *a, **kw: None)
    monkeypatch.setattr(owner_mailbox, "sweep_settled_owner_mailboxes",
                        lambda *a, **kw: pytest.fail("the drive-custody pass owns the mailbox sweep"))
    calls = []
    def unavailable(*_a, **_kw):
        calls.append("trees-unavailable")
        raise OSError("task trees unavailable")
    monkeypatch.setattr(headless, "prune_task_trees", unavailable)
    monkeypatch.setattr("ouroboros.delegate_state_sweep.sweep_settled_delegate_state",
                        lambda *a: calls.append("delegate") or {})
    latch = SimpleNamespace(release=lambda: None)
    maintenance._startup_prune_sweeps()
    maintenance._run_periodic_reconcile_sweep([0], latch=latch)
    _retire(tmp_path)
    assert calls == ["trees-unavailable", "delegate"]
    monkeypatch.setattr(headless, "prune_task_trees", lambda *a, **kw: calls.append("trees"))
    monkeypatch.setattr(services, "prune_service_logs", lambda *a: calls.append("services") or {})
    from ouroboros.maintenance_jobs import record
    record(tmp_path, "runtime_retirement", state="pending")  # explicit repair wakes the failed input
    maintenance._run_periodic_reconcile_sweep([0], latch=latch)
    _retire(tmp_path)
    maintenance._run_periodic_reconcile_sweep([0], latch=latch)
    _retire(tmp_path)
    assert calls == ["trees-unavailable", "delegate"] + ["trees", "services", "delegate"]


def test_tree_exclusions_follow_only_addressed_links_and_own_child_results(tmp_path, monkeypatch):
    from ouroboros.startup_task_files import startup_tree_exclusions
    migrations.prepare_startup_state(tmp_path)
    fields = ("parent_task_id", "root_task_id", "retry_task_id", "superseded_by", "original_task_id", "timeout_retry_from")
    results.write_task_result(tmp_path, "running", "running", metadata={field: field for field in fields})
    for tid in (*fields, "child", "unrelated"):
        seed_result(tmp_path, tid, {"_schema_version": 1, "task_id": tid, "status": "completed"})
    child = tmp_path / "task_drives/running"
    seed_result(child, "child", {"_schema_version": 1, "task_id": "child", "status": "completed"})
    read, listing, seen = results.load_task_result, results.list_task_results, []
    def addressed(root, task_id, **kwargs):
        if root == tmp_path:
            seen.append(task_id)
        return read(root, task_id, **kwargs)
    def own_results(root, **kwargs):
        assert root != tmp_path, "canonical history enumeration"
        return listing(root, **kwargs)
    monkeypatch.setattr(results, "load_task_result", addressed)
    monkeypatch.setattr(results, "list_task_results", own_results)
    expected = {"running", "child", *fields}
    assert startup_tree_exclusions(tmp_path) == set(seen) == expected


def _retire(root):
    from ouroboros.maintenance import run_job
    maintenance._run_reconcile_prunes()
    return run_job(root, "runtime_retirement")
