"""The addressed sets as rows of the current-facts store: derived memberships are atomic with
the result write, owner-held rows publish before their dependent effect, custody rows open
before the event append and close after it, and boot reads only addressed members."""
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from ouroboros import obligations as o
from ouroboros import store
from ouroboros import task_results as results
from ouroboros.store import GENERATION, GENERATION_KEY


def test_generic_set_and_concurrent_writers(tmp_path):
    with ThreadPoolExecutor() as pool:
        list(pool.map(lambda i: o.add(tmp_path, "upgrade_notices", str(i), {"chat_id": i}), range(12)))
    assert len(o.members(tmp_path, "upgrade_notices")) == 12
    o.remove(tmp_path, "upgrade_notices", "1")
    assert "1" not in o.members(tmp_path, "upgrade_notices")


def test_a_lost_store_is_unavailable_never_empty(tmp_path):
    from ouroboros.startup_migrations import stamp

    stamp(tmp_path, **{GENERATION_KEY: GENERATION})  # a completed stamp beside a missing store
    with pytest.raises(o.ObligationsUnavailable):
        o.members(tmp_path, "nonterminal")
    with pytest.raises(o.ObligationsUnavailable):
        o.add(tmp_path, "unknowns", "task:x", {"task_id": "x"})


@pytest.mark.parametrize("status,fields,expected", [
    ("running", {}, {"nonterminal", "pending_drives"}),
    ("running", {"root_phase_checkpoint": {"post_task_synthesis": "paused"}}, {"synthesis"}),
    ("running", {"delegated_runs_unreconciled": ["r"]}, {"delegated_runs"}),
    ("running", {"child_ref_promotion": {"schema_version": 1, "status": "incomplete", "pending_refs": [{"path": "x"}]}}, {"promotions"}),
    ("completed", {}, {"terminal_projection", "synthesis"}),
])
def test_membership_is_atomic_with_the_result_write(tmp_path, monkeypatch, status, fields, expected):
    """The flags are columns of the row the same transaction writes: a write that fails inside the
    transaction leaves neither the result nor a member; a write that commits has both."""
    original = store.Txn.store_task

    def crash(self, document, **kwargs):
        original(self, document, **kwargs)
        raise OSError("crash inside the transaction")

    monkeypatch.setattr(store.Txn, "store_task", crash)
    with pytest.raises(OSError, match="crash"):
        results.write_task_result(tmp_path, "task", status, **fields)
    monkeypatch.undo()
    assert results.load_task_result(tmp_path, "task") is None
    for name in expected:
        assert "task" not in o.members(tmp_path, name)
    results.write_task_result(tmp_path, "task", status, **fields)
    for name in expected:
        assert "task" in o.members(tmp_path, name)
    assert not (tmp_path / "task_results" / "task.json").exists()  # the store is the authority, not a file


def test_a_transition_retires_its_memberships_in_the_same_transaction(tmp_path):
    results.write_task_result(tmp_path, "task", "running")
    assert "task" in o.members(tmp_path, "nonterminal")
    results.write_task_result(tmp_path, "task", "completed")
    assert "task" not in o.members(tmp_path, "nonterminal")
    assert "task" in o.members(tmp_path, "terminal_projection")
    assert not (tmp_path / "state" / "obligations").exists()  # no set file, no rebuild mark


def test_pending_drive_follows_the_writer_rule(tmp_path):
    """Nonterminal owes the drive; a settled result owes it until its terminal files are ready;
    ``drive_started`` publishes the pool handoff before the child can write; ``drive_finished``
    retires both halves."""
    o.drive_started(tmp_path, {"id": "t1", "drive_root": "/d/t1"})
    assert o.members(tmp_path, "pending_drives")["t1"] == {"task_id": "t1", "drive_root": "/d/t1"}
    results.write_task_result(tmp_path, "t1", "running")
    assert "t1" in o.members(tmp_path, "pending_drives")
    results.write_task_result(tmp_path, "t1", "cancel_requested")  # neither nonterminal nor settled: sticky
    assert "t1" in o.members(tmp_path, "pending_drives")
    results.write_task_result(tmp_path, "t1", "completed", result="done")
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.row("t1", "pending_drive") == 0  # settled, files ready: the writer cleared the flag
    assert "t1" in o.members(tmp_path, "pending_drives")  # the handoff row stays until drive_finished
    o.drive_finished(tmp_path, {"id": "t1"})
    assert "t1" not in o.members(tmp_path, "pending_drives")


def test_a_torn_watermark_file_is_rewritten_and_the_import_runs(tmp_path):
    from ouroboros.startup_migrations import prepare_startup_state, watermarks

    prepare_startup_state(tmp_path)
    results.write_task_result(tmp_path, "t1", "running")
    (tmp_path / "state" / "migrations.json").write_bytes(b"")  # torn
    report = prepare_startup_state(tmp_path)
    assert report["status"] in {"unchanged", "reconciled"}
    assert watermarks(tmp_path)[GENERATION_KEY] == GENERATION
    assert set(o.members(tmp_path, "nonterminal")) == {"t1"}


def test_custody_row_commits_before_the_chain_append_and_the_outbox_empties_after(tmp_path, monkeypatch):
    """The reducing transaction (run row + outbox row) commits FIRST; the chain append sees the
    committed row; the outbox row is deleted only once the append confirmed."""
    from ouroboros import delegate_custody as c
    from ouroboros import delegate_custody_current as current

    seen = []

    def append(path, row, **kwargs):
        held = current.runs(tmp_path, "state=?", (current.HELD,))
        seen.append((row["type"], "run" in held, held["run"].settled if "run" in held else None,
                     current.outbox_depth(tmp_path)))
        return current_append(path, row, **kwargs)

    current_append = current.append_jsonl
    monkeypatch.setattr(current, "append_jsonl", append)
    assert c.emit(tmp_path, c.STARTED, {"run_id": "run", "task_id": "task"})
    assert len(c.open_runs(tmp_path)) == 1
    assert c.emit(tmp_path, c.SETTLED, {"run_id": "run", "task_id": "task", "state": "succeeded"})
    assert c.open_runs(tmp_path) == []
    assert seen == [(c.STARTED, True, False, 1), (c.SETTLED, False, None, 1)]
    assert current.outbox_depth(tmp_path) == 0
    assert c.replay(tmp_path)["run"].settled
    assert "run" in o.members(tmp_path, "delegated_runs")["task"]["closed_runs"]
    rows = [json.loads(line) for line in c.event_log_path(tmp_path).read_text(encoding="utf-8").splitlines()]
    assert [row["type"] for row in rows] == [c.STARTED, c.SETTLED] and all(row["event_id"] for row in rows)


def test_a_failed_chain_append_parks_the_event_in_the_outbox_until_the_flush(tmp_path, monkeypatch):
    """The store is the authority: a refused append leaves the reduced row in place and the
    event in the outbox; the next publication appends it (once, by ``event_id``)."""
    from ouroboros import delegate_custody as c
    from ouroboros import delegate_custody_current as current

    assert c.emit(tmp_path, c.STARTED, {"run_id": "run", "task_id": "task"})
    with monkeypatch.context() as patched:
        patched.setattr(current, "append_jsonl", lambda *a, **kw: False)
        assert c.emit(tmp_path, c.SETTLED, {"run_id": "run", "task_id": "task", "state": "succeeded"})
    assert c.replay(tmp_path)["run"].settled, "the row is the authority and it landed"
    assert current.outbox_depth(tmp_path) == 1
    assert [json.loads(line)["type"] for line in c.event_log_path(tmp_path).read_text(encoding="utf-8").splitlines()] == [c.STARTED]
    assert c.emit(tmp_path, c.PATCH_CAPTURED, {"run_id": "run", "task_id": "task"})
    assert current.outbox_depth(tmp_path) == 0
    rows = [json.loads(line) for line in c.event_log_path(tmp_path).read_text(encoding="utf-8").splitlines()]
    assert [row["type"] for row in rows] == [c.STARTED, c.SETTLED, c.PATCH_CAPTURED]
    assert len({row["event_id"] for row in rows}) == 3
    assert vars(c.replay(tmp_path)["run"]) == vars(c.replay(tmp_path, rows=rows)["run"]), "the chain folds to the rows"


def test_a_busy_store_refuses_the_custody_event_and_the_repair_folds_a_legacy_chain(tmp_path, monkeypatch):
    """``emit`` reports its own fate: a store that cannot take the reducing transaction lands
    nothing (no row, no chain append — the chain never runs ahead of the authority). The
    explicit repair folds a retained chain an old body wrote into the rows."""
    from contextlib import contextmanager

    from ouroboros import delegate_custody as c
    from ouroboros import delegate_custody_current as current
    from ouroboros.startup_migrations import prepare_startup_state
    from ouroboros.utils import append_jsonl

    @contextmanager
    def busy(root, *, write, **kwargs):
        raise store.StoreBusy("current store busy")
        yield

    prepare_startup_state(tmp_path)
    with monkeypatch.context() as patched:
        patched.setattr(current.store, "open_store", busy)
        assert not c.emit(tmp_path, c.STARTED, {"run_id": "run", "task_id": "task"})
    assert not c.event_log_path(tmp_path).exists()
    assert "run" not in c.replay(tmp_path)
    append_jsonl(c.event_log_path(tmp_path), {"ts": "2026-01-01T00:00:00+00:00", "type": c.STARTED,
                                              "run_id": "legacy", "task_id": "task"})
    assert current.rebuild(tmp_path) == 1
    assert not c.replay(tmp_path)["legacy"].settled and c.open_runs(tmp_path)[0].run_id == "legacy"


def test_first_import_classifies_once_and_stamps(tmp_path, monkeypatch):
    from ouroboros import startup_migrations as m, store_import
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
    assert report["imported"] and report["quarantined"] == 1
    assert set(o.members(tmp_path, "nonterminal")) == {"running"}
    assert set(o.members(tmp_path, "synthesis")) == {"paused", "owed"}
    assert set(o.members(tmp_path, "terminal_projection")) == {"owed"}
    assert (directory / "quarantine" / "broken.json").exists()
    assert m.watermarks(tmp_path)["cancel_latches_generation"] == 7
    assert m.watermarks(tmp_path)[GENERATION_KEY] == GENERATION
    monkeypatch.setattr(store_import, "_admit", lambda *a: pytest.fail("second history parse"))
    assert m.prepare_startup_state(tmp_path)["status"] == "unchanged"


def test_import_failure_does_not_stamp_complete(tmp_path, monkeypatch):
    from ouroboros import startup_migrations as m, store_import
    monkeypatch.setattr(store_import, "_build", lambda *a, **kw: (_ for _ in ()).throw(OSError("interrupted")))
    with pytest.raises(OSError):
        m.prepare_startup_state(tmp_path)
    assert GENERATION_KEY not in m.watermarks(tmp_path)
    assert not store.store_path(tmp_path).exists()


def test_cancel_migration_stamps_only_success_and_per_generation(tmp_path, monkeypatch):
    from ouroboros import startup_migrations as m, cancel_intents as c
    calls = []
    monkeypatch.setattr(c, "migrate_legacy_cancel_latches", lambda *a, **kw: calls.append(kw) or [])
    m.migrate_cancel_latches(tmp_path, records=[], generation=6)
    m.migrate_cancel_latches(tmp_path, records=[], generation=6)
    m.migrate_cancel_latches(tmp_path, records=[], generation=7)
    m.migrate_cancel_latches(tmp_path, records=[], generation=7)
    assert len(calls) == 2
    monkeypatch.setattr(c, "migrate_legacy_cancel_latches", lambda *a, **kw: (_ for _ in ()).throw(OSError()))
    with pytest.raises(OSError):
        m.migrate_cancel_latches(tmp_path, records=[], generation=8)
    assert m.watermarks(tmp_path)["cancel_latches_generation"] == 7


def test_recovery_reads_open_members_among_2000_completed(tmp_path, monkeypatch):
    from ouroboros import startup_migrations as m, task_status as status, agent_task_pipeline as pipeline
    from ouroboros import task_store
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
    original = task_store.read_document
    def counted(root, task_id):
        reads.append(task_id)
        return original(root, task_id)
    monkeypatch.setattr(task_store, "read_document", counted)
    monkeypatch.setattr(results, "read_document", counted)
    monkeypatch.setattr(results, "list_task_results", lambda *a, **kw: pytest.fail("history enumeration"))
    monkeypatch.setattr(status, "effective_task_result", lambda root, row, **kw: row)
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
    m.prepare_startup_state(tmp_path)
    c.emit(tmp_path, c.STARTED, {"run_id": "run", "task_id": "task"})
    results.write_task_result(tmp_path, "task", "completed", delegated_runs_unreconciled=["run"])
    c.emit(tmp_path, c.SETTLED, {"run_id": "run", "task_id": "task", "state": "succeeded"})
    monkeypatch.setattr(c, "custody_rows", lambda *a, **kw: pytest.fail("history fold"))
    monkeypatch.setattr(c, "_iter_rows", lambda *a, **kw: pytest.fail("history fold"))
    assert terminal.backfill_terminal_reconciliations(tmp_path) == ["task"]
    row = results.load_task_result(tmp_path, "task")
    assert row["delegated_runs_unreconciled"] == []
    assert row["delegate_terminal_reconciliation"]["terminal_runs"][0]["state"] == "succeeded"
    assert "task" not in o.members(tmp_path, "delegated_runs")


def test_an_old_body_writing_a_legacy_latch_file_is_reconciled_into_an_intent(tmp_path):
    """P6': a file an unaware body wrote while the store existed is a new row at the next
    import; the legacy cancel latch carve-out stamps it and it becomes an intent."""
    from ouroboros import startup_migrations as m, store_import
    from ouroboros.cancel_intents import cancel_pending
    from ouroboros.utils import atomic_write_json
    m.prepare_startup_state(tmp_path)
    atomic_write_json(tmp_path / "task_results/old-task.json", {"task_id": "old-task", "status": "cancel_requested"})
    assert m.prepare_startup_state(tmp_path)["status"] == "unchanged"
    assert not cancel_pending(tmp_path, "old-task", strict=True)
    assert store_import.run_legacy_reconcile(tmp_path)["root_report"]["status"] == "reconciled"
    assert results.load_task_result(tmp_path, "old-task", strict=True)["_schema_version"] == 1
    assert cancel_pending(tmp_path, "old-task", strict=True)


def test_a_retired_sets_rows_and_unsent_lines_retire_once_without_delivery(tmp_path, monkeypatch):
    """The lost-Max episode left with its producer (29.10): the first boot of this generation removes the row an
    older body kept and stamps it, a stamped boot reads no row, the explicit rebuild repeats it, and a line that
    producer queued stays preserved as exhausted, never delivered. Other producers are unaffected."""
    from ouroboros import startup_migrations as m
    from ouroboros import terminal_outbox
    from supervisor import terminal_delivery as outbox

    m.prepare_startup_state(tmp_path)
    marks = m.watermarks(tmp_path)
    assert marks.pop("retired_sets_generation") == m.RETIRED_SETS_GENERATION  # a fresh install stamps it at once
    (tmp_path / "state" / "migrations.json").write_text(json.dumps(marks), encoding="utf-8")  # an older body's file
    o.add(tmp_path, "memory_max", "episode", {"id": "e1", "notices": {"open": {"id": "memory_max:e1:open"}}})
    o.add(tmp_path, "unknowns", "kept", {"reason": "another set"})
    assert outbox.register_pending_delivery(tmp_path, {
        "type": "send_message", "chat_id": 1, "task_id": "", "text": "Max lost: an older body's line.",
        "role": "system", "system_type": "memory_max_notice", "delivery_id": "memory_max:e1:open"})
    for did, kind in (("memory_max:legacy:open", ""), ("legacy-type-only", "memory_max_notice"),
                      ("ordinary", "task_checkpoint"), ("low_start:task:1", "low_start_notice")):
        assert outbox.register_pending_delivery(tmp_path, {"type": "send_message", "chat_id": 1,
            "text": "Retained exact payload", "role": "system", "system_type": kind, "delivery_id": did})
    before = outbox._read_registry(tmp_path)
    assert m.prepare_startup_state(tmp_path)["retired_rows"] == 1
    assert o.members(tmp_path, "memory_max") == {} and set(o.members(tmp_path, "unknowns")) == {"kept"}
    assert m.watermarks(tmp_path)["retired_sets_generation"] == m.RETIRED_SETS_GENERATION
    o.add(tmp_path, "memory_max", "episode", {"id": "e2"})  # an older body's run after the stamp
    with monkeypatch.context() as patch:
        patch.setattr(o, "members", lambda *_a: pytest.fail("a stamped generation reads no row"))
        patch.setattr(terminal_outbox, "update_json_locked", lambda *_a, **_kw: pytest.fail("a stamped boot reads no outbox"))
        assert "retired_rows" not in m.prepare_startup_state(tmp_path)
    assert m.prepare_startup_state(tmp_path, rebuild=True)["retired_rows"] == 1  # the explicit rebuild repeats it
    after = outbox._read_registry(tmp_path)
    assert set(after["pending"]) == {"ordinary", "low_start:task:1"}
    assert after["pending"] == {did: before["pending"][did] for did in after["pending"]}
    assert set(after["exhausted"]) == {"memory_max:e1:open", "memory_max:legacy:open", "legacy-type-only"}
    for did, row in after["exhausted"].items():
        assert row["reason"] == "producer_retired"
        assert {k: row[k] for k in before["pending"][did]} == before["pending"][did]
        assert not outbox.already_delivered(tmp_path, did) and did not in after.get("receipts", {})


def test_retired_delivery_migration_retries_failed_writes_and_preserves_actual_send_facts(tmp_path, monkeypatch):
    from ouroboros import startup_migrations as m, terminal_outbox
    from supervisor import terminal_delivery as outbox

    m.prepare_startup_state(tmp_path)
    m.stamp(tmp_path, retired_sets_generation=1)
    unsent, sent = "memory_max:unsent:open", "memory_max:sent:open"
    for did in (unsent, sent):
        assert outbox.register_pending_delivery(tmp_path, {"type": "send_message", "delivery_id": did,
            "chat_id": 1, "text": "Saved payload", **({"emitted": {"text": "Saved payload"}} if did == sent else {})})
    before = outbox._registry_path(tmp_path).read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(terminal_outbox, "update_json_locked", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("busy")))
        with pytest.raises(OSError):
            m.retire_sets(tmp_path)
    assert m.watermarks(tmp_path)["retired_sets_generation"] == 1
    assert outbox._registry_path(tmp_path).read_bytes() == before
    m.retire_sets(tmp_path)
    state = outbox._read_registry(tmp_path)
    assert unsent in state["exhausted"] and sent in state["pending"] and not state["delivered"]
    assert m.watermarks(tmp_path)["retired_sets_generation"] == m.RETIRED_SETS_GENERATION
