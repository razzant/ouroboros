"""Boot gaps, append ordering, notice receipts and deferred housekeeping."""
import inspect
import json
from contextlib import contextmanager

import pytest

from ouroboros import obligations as o, startup_migrations as migrations
from ouroboros import server_maintenance as maintenance, task_results as results


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
    assert migrations.watermarks(tmp_path)["obligations_generation"] == migrations.OBLIGATIONS_GENERATION
    monkeypatch.setattr(cancel_intents, "request_cancel", request)
    migrations.prepare_startup_state(tmp_path, rebuild=True)
    assert cancel_intents.active_intent(tmp_path, "bad")
    assert not o.members(tmp_path, "unknowns")


@pytest.mark.parametrize("prior_sets", [False, True])
def test_unreadable_import_leaves_boot_available_and_explicit_repair_fails(tmp_path, monkeypatch, prior_sets):
    import server
    from pathlib import Path
    if prior_sets:
        migrations.prepare_startup_state(tmp_path)
        migrations.stamp(tmp_path, obligations_generation=None)
    original = Path.iterdir
    def unreadable(path):
        if path == tmp_path / "task_results":
            raise PermissionError("directory unreadable")
        return original(path)
    monkeypatch.setattr(Path, "iterdir", unreadable)
    assert migrations.prepare_startup_state(tmp_path, strict=False)["status"] == "unavailable"
    assert "strict=False" in inspect.getsource(server.lifespan)
    with pytest.raises(o.ObligationsUnavailable):
        o.members(tmp_path, "nonterminal")
    assert migrations.watermarks(tmp_path).get("obligations_generation") is None
    rows = [json.loads(line) for line in (tmp_path / "logs/supervisor.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["type"] == "startup_migration" and rows[-1]["phase"] == "failed"
    with pytest.raises(PermissionError):
        migrations.prepare_startup_state(tmp_path, rebuild=True)


def test_migration_diagnostic_failure_does_not_close_boot(tmp_path, monkeypatch):
    monkeypatch.setattr(migrations, "_prepare_startup_state", lambda *a, **kw:
        (_ for _ in ()).throw(PermissionError("import unavailable")))
    monkeypatch.setattr(migrations, "append_jsonl", lambda *a, **kw:
        (_ for _ in ()).throw(OSError("logs unavailable")))
    assert migrations.prepare_startup_state(tmp_path, strict=False)["status"] == "unavailable"
    with pytest.raises(PermissionError, match="import unavailable"):
        migrations.prepare_startup_state(tmp_path, rebuild=True)


@pytest.mark.parametrize("kind", ["delegate_run_started", "delegate_run_settled"])
def test_custody_append_can_write_results_without_lock_nesting(tmp_path, monkeypatch, kind):
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_custody_current import current_reads
    migrations.prepare_startup_state(tmp_path)
    assert custody.emit(tmp_path, custody.STARTED, {"run_id": "run", "task_id": "task"})
    locked, held = o.locked, []
    @contextmanager
    def checked(root):
        assert not held, "obligations lock nested across append/result write"
        with locked(root):
            held.append(True)
            try:
                yield
            finally:
                held.pop()
    monkeypatch.setattr(o, "locked", checked)
    def append(_path, _row):
        assert "run:run" in o.members(tmp_path, "custody_open")
        results.write_task_result(tmp_path, "other", "running")
        o.add(tmp_path, "custody_open", "invocation:other", {"request": {"invocation_id": "other"}})
        return True
    monkeypatch.setattr(custody, "append_jsonl", append)
    assert custody.emit(tmp_path, kind, {"run_id": "run", "task_id": "task", "state": "succeeded"})
    assert "other" in o.members(tmp_path, "nonterminal")
    assert "invocation:other" in o.members(tmp_path, "custody_open")
    with current_reads(tmp_path):
        assert bool(custody.open_runs(tmp_path)) == (kind == custody.STARTED)
    if kind == custody.SETTLED:
        assert "run" in o.members(tmp_path, "delegated_runs")["task"]["closed_runs"]


def test_explicit_rebuild_replaces_corrupt_sets(tmp_path):
    migrations.prepare_startup_state(tmp_path)
    for name in ("nonterminal", "custody_open"):
        o.path(tmp_path, name).write_text("broken", encoding="utf-8")
    migrations.prepare_startup_state(tmp_path, rebuild=True)
    assert o.members(tmp_path, "nonterminal") == o.members(tmp_path, "custody_open") == {}


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


@pytest.mark.parametrize("writer_first", [True, False])
def test_custody_ack_serializes_with_result_membership(tmp_path, monkeypatch, writer_first):
    from ouroboros import delegate_custody as custody, delegate_terminal, platform_layer
    from ouroboros.delegate_custody_current import snapshot
    migrations.prepare_startup_state(tmp_path)
    results.write_task_result(tmp_path, "task", "completed")
    custody.emit(tmp_path, custody.STARTED, {"run_id": "old", "task_id": "task"})
    custody.emit(tmp_path, custody.SETTLED, {"run_id": "old", "task_id": "task", "state": "succeeded"})
    observed = snapshot(tmp_path)
    if writer_first:
        results.write_task_result(tmp_path, "task", "completed", delegated_runs_unreconciled=["new"])
    acquire = platform_layer.acquire_exclusive_file_lock
    acquisitions = []
    def track(path, **kwargs):
        acquisitions.append(path.name)
        return acquire(path, **kwargs)
    monkeypatch.setattr(platform_layer, "acquire_exclusive_file_lock", track)
    delegate_terminal._ack_current_receipts(tmp_path, "task", observed)
    assert acquisitions[0] == "task.json.lock", "result lock must precede obligations retirement"
    if not writer_first:
        assert "task" not in o.members(tmp_path, "delegated_runs")
        results.write_task_result(tmp_path, "task", "completed", delegated_runs_unreconciled=["new"])
    assert "task" in o.members(tmp_path, "delegated_runs")


@pytest.mark.parametrize("set_name", ["pending_drives", "unknowns"])
def test_unavailable_recovery_set_does_not_escape_a_boot_door(tmp_path, monkeypatch, set_name):
    migrations.prepare_startup_state(tmp_path)
    o.path(tmp_path, set_name).write_text("broken", encoding="utf-8")
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


def test_prunes_wait_off_ready_path_and_tree_debt_clears_after_recovery(tmp_path, monkeypatch):
    from ouroboros import headless
    migrations.prepare_startup_state(tmp_path)
    monkeypatch.setattr(maintenance, "DATA_DIR", tmp_path)
    for name in ("_STARTUP_PRUNES_OWED", "_STARTUP_SOURCE_PRUNES_OWED", "_STARTUP_TREES_OWED",
                 "_STARTUP_RECOVERY_GAPS", "_STARTUP_TEMP_SWEEP_OWED"):
        monkeypatch.setattr(maintenance, name, [False])
    calls = []
    monkeypatch.setattr(headless, "prune_task_trees", lambda *a, **kw: calls.append("trees"))
    monkeypatch.setattr(maintenance, "_startup_worktree_prune", lambda: calls.append("worktrees"))
    monkeypatch.setattr(maintenance, "_prune_delegated_snapshots", lambda: None)
    o.add(tmp_path, "pending_drives", "task")
    maintenance._startup_prune_sweeps(preserve_task_sources=True, recovery_report={"unresolved": ["task"]})
    assert calls == []
    maintenance._run_deferred_startup_prunes()
    assert calls == ["worktrees"] and maintenance._STARTUP_TREES_OWED == [True]
    o.remove(tmp_path, "pending_drives", "task")
    maintenance._run_deferred_startup_prunes()
    assert calls == ["worktrees", "trees"] and maintenance._STARTUP_TREES_OWED == [False]


@pytest.mark.parametrize("gap", [None, "unreadable", "coarse"])
def test_deferred_prunes_keep_live_tree_and_retry_recovery_gaps(tmp_path, monkeypatch, gap):
    import os
    from ouroboros import owner_mailbox, utils
    migrations.prepare_startup_state(tmp_path)
    monkeypatch.setattr(maintenance, "DATA_DIR", tmp_path)
    for name in ("_STARTUP_PRUNES_OWED", "_STARTUP_SOURCE_PRUNES_OWED", "_STARTUP_TREES_OWED",
                 "_STARTUP_RECOVERY_GAPS", "_STARTUP_TEMP_SWEEP_OWED"):
        monkeypatch.setattr(maintenance, name, [False])
    monkeypatch.setattr(maintenance, "_startup_worktree_prune", lambda: None)
    monkeypatch.setattr(maintenance, "_prune_delegated_snapshots", lambda: None)
    running = results.write_task_result(tmp_path, "running", "running")
    results.write_task_result(tmp_path, "settled", "completed", ts="2000-01-01T00:00:00+00:00")
    for task in ("running", "settled"):
        (tmp_path / "task_trees" / task).mkdir(parents=True)
        owner_mailbox.write_owner_message(tmp_path, "retained message", task)
    service = tmp_path / "services/settled/service.log"
    service.parent.mkdir(parents=True)
    service.write_text("retained service output", encoding="utf-8")
    os.utime(service, (1, 1))
    if gap == "unreadable":
        (tmp_path / "task_results/running.json").write_text("broken", encoding="utf-8")
    maintenance._startup_prune_sweeps(preserve_task_sources=bool(gap),
        recovery_report={"errors": ["ownership unknown"]} if gap == "coarse" else {"unresolved": ["running"]})
    maintenance._run_deferred_startup_prunes()
    if gap:
        assert (tmp_path / "task_trees/settled").exists() and service.exists()
        assert owner_mailbox._mailbox_path(tmp_path, "settled").exists()
        assert maintenance._STARTUP_TREES_OWED == maintenance._STARTUP_SOURCE_PRUNES_OWED == [True]
        assert maintenance._STARTUP_PRUNES_OWED == [False]
        utils.atomic_write_json(tmp_path / "task_results/running.json", running)
        maintenance._STARTUP_RECOVERY_GAPS[0] = False
        maintenance._run_deferred_startup_prunes()
    assert not (tmp_path / "task_trees/settled").exists() and not service.exists()
    assert not owner_mailbox._mailbox_path(tmp_path, "settled").exists()
    assert (tmp_path / "task_trees/running").exists()
    assert owner_mailbox._mailbox_path(tmp_path, "running").exists()
    assert maintenance._STARTUP_TREES_OWED == maintenance._STARTUP_SOURCE_PRUNES_OWED == [False]


def test_periodic_sweep_retries_source_only_debt_until_success(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ouroboros import headless, owner_mailbox, review_operation
    from ouroboros.tools import services
    migrations.prepare_startup_state(tmp_path)
    monkeypatch.setattr(maintenance, "DATA_DIR", tmp_path)
    for name in ("_STARTUP_PRUNES_OWED", "_STARTUP_TREES_OWED", "_STARTUP_SOURCE_PRUNES_OWED",
                 "_STARTUP_RECOVERY_GAPS", "_STARTUP_TEMP_SWEEP_OWED"):
        monkeypatch.setattr(maintenance, name, [name == "_STARTUP_SOURCE_PRUNES_OWED"])
    monkeypatch.setattr(maintenance, "_periodic_zombie_reconcile", lambda **kw: None)
    monkeypatch.setattr(maintenance, "_run_drive_custody_pass", lambda *a: None)
    monkeypatch.setattr(review_operation, "collect_orphaned_operations_softly", lambda *a, **kw: None)
    monkeypatch.setattr(headless, "prune_task_trees", lambda *a, **kw: pytest.fail("tree debt already discharged"))
    calls = []
    def unavailable(_root):
        calls.append("mail-unavailable")
        raise OSError("mailbox cleanup unavailable")
    monkeypatch.setattr(owner_mailbox, "sweep_settled_owner_mailboxes", unavailable)
    monkeypatch.setattr(services, "prune_service_logs", lambda *a: calls.append("services") or {})
    latch = SimpleNamespace(release=lambda: None)
    maintenance._run_periodic_reconcile_sweep([0], latch=latch)
    assert calls == ["mail-unavailable"] and maintenance._STARTUP_SOURCE_PRUNES_OWED == [True]
    monkeypatch.setattr(owner_mailbox, "sweep_settled_owner_mailboxes", lambda *a: calls.append("mail") or {})
    maintenance._run_periodic_reconcile_sweep([0], latch=latch)
    assert calls == ["mail-unavailable", "mail", "services"]
    assert maintenance._STARTUP_SOURCE_PRUNES_OWED == [False]
    maintenance._run_periodic_reconcile_sweep([0], latch=latch)
    assert calls == ["mail-unavailable", "mail", "services"]


def test_tree_exclusions_follow_only_addressed_links_and_own_child_results(tmp_path, monkeypatch):
    from ouroboros.startup_task_files import startup_tree_exclusions
    from ouroboros.utils import atomic_write_json
    migrations.prepare_startup_state(tmp_path)
    fields = ("parent_task_id", "root_task_id", "retry_task_id", "superseded_by", "original_task_id", "timeout_retry_from")
    results.write_task_result(tmp_path, "running", "running", metadata={field: field for field in fields})
    for tid in (*fields, "child", "unrelated"):
        atomic_write_json(tmp_path / f"task_results/{tid}.json", {"_schema_version": 1, "task_id": tid, "status": "completed"})
    child = tmp_path / "task_drives/running"
    atomic_write_json(child / "task_results/child.json", {"_schema_version": 1, "task_id": "child", "status": "completed"})
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


def test_unchanged_result_memberships_take_no_obligations_lock(tmp_path, monkeypatch):
    migrations.prepare_startup_state(tmp_path)
    results.write_task_result(tmp_path, "task", "running")
    locked, calls = o.locked, []
    @contextmanager
    def counted(root):
        calls.append(root)
        with locked(root):
            yield
    monkeypatch.setattr(o, "locked", counted)
    results.write_task_result(tmp_path, "task", "running", progress_note="next step")
    assert calls == []
    results.write_task_result(tmp_path, "task", "completed", result="saved answer")
    assert len(calls) == 2  # one pre-publication acquisition, one post-publication acquisition


def test_foreign_body_sha_replaces_result_sets_and_matching_sha_reads_no_history(tmp_path, monkeypatch):
    from ouroboros.utils import atomic_write_json
    from ouroboros import delegate_custody as custody
    migrations.prepare_startup_state(tmp_path)
    results.write_task_result(tmp_path, "became-done", "running")
    results.write_task_result(tmp_path, "became-running", "completed")
    custody.emit(tmp_path, custody.STARTED, {"run_id": "run", "task_id": "receipt-only"})
    custody.emit(tmp_path, custody.SETTLED, {"run_id": "run", "task_id": "receipt-only"})
    o.add(tmp_path, "upgrade_notices", "7:legacy_memory_notice", {"recorded": True})
    migrations.stamp(tmp_path, observed_state_sha="aware-checkout")
    atomic_write_json(tmp_path / "state/state.json", {"current_sha": "forward-checkout-by-7.5-body"})
    for tid, status in (("became-done", "completed"), ("became-running", "running")):
        atomic_write_json(tmp_path / f"task_results/{tid}.json", {
            "_schema_version": 1, "task_id": tid, "status": status,
            **({"canonical_terminal_projection_origin": "terminal_transition"} if status == "completed" else {}),
        })
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: pytest.fail("version subprocess"))
    assert migrations.prepare_startup_state(tmp_path)["imported"]
    assert set(o.members(tmp_path, "nonterminal")) == set(o.members(tmp_path, "pending_drives")) == {"became-running"}
    assert set(o.members(tmp_path, "terminal_projection")) == set(o.members(tmp_path, "synthesis")) == {"became-done"}
    assert "run" in o.members(tmp_path, "delegated_runs")["receipt-only"]["closed_runs"]
    assert o.members(tmp_path, "upgrade_notices")["7:legacy_memory_notice"]["recorded"]
    monkeypatch.setattr(migrations, "_result_records", lambda *a: pytest.fail("matching SHA enumerated history"))
    assert not migrations.prepare_startup_state(tmp_path)["imported"]
