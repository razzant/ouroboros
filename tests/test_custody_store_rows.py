"""Delegated custody on the store (DELTA §3): the reducer's rows and the outbox around every
death point, the pending invocation's atomic transition, the open set, the export seam's
outbox verification and the importer's one fold of a legacy chain."""
from __future__ import annotations

import errno
import json
from contextlib import contextmanager

import pytest

from ouroboros import delegate_custody as c
from ouroboros import delegate_custody_current as current
from ouroboros import obligations as o
from ouroboros import store
from ouroboros.utils import append_jsonl


def _chain(root):
    path = c.event_log_path(root)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _writes(monkeypatch):
    """Count the store's write transactions (the reducing transaction is exactly one)."""
    opened = []
    real = store.open_store

    @contextmanager
    def tracked(root, **kwargs):
        with real(root, **kwargs) as txn:
            opened.append(bool(kwargs.get("write")))
            yield txn

    monkeypatch.setattr(store, "open_store", tracked)
    return opened


def test_a_death_between_the_append_and_the_outbox_delete_replays_the_event_once(tmp_path, monkeypatch):
    """P8 death point: the append landed, the process died before the outbox row was deleted.
    The next flush appends the same ``event_id`` again; the forensic fold and the importer
    dedup it, so the chain folds to exactly the rows."""
    assert c.emit(tmp_path, c.STARTED, {"run_id": "run", "task_id": "task"})

    @contextmanager
    def dies_after_append(root, **kwargs):
        if kwargs.get("write"):
            raise store.StoreBusy("process died before the outbox delete")
        with real(root, **kwargs) as txn:
            yield txn

    real = store.open_store
    with monkeypatch.context() as patched:
        settled = {"ts": "2026-01-01T00:00:00+00:00", "type": c.SETTLED, "event_id": "e" * 32,
                   "run_id": "run", "task_id": "task", "state": "succeeded"}
        with real(tmp_path, write=True) as txn:  # the reducing transaction committed...
            current._reduce(txn, settled, current._loaded(txn, settled), {})
            txn.outbox_put(settled)
        patched.setattr(store, "open_store", dies_after_append)
        assert current.flush_outbox(tmp_path) == 0  # ...the append landed, the delete did not
    assert [row["type"] for row in _chain(tmp_path)] == [c.STARTED, c.SETTLED]
    assert current.outbox_depth(tmp_path) == 1
    assert current.flush_outbox(tmp_path) == 1  # the next flush replays the parked event
    assert current.outbox_depth(tmp_path) == 0
    duplicated = _chain(tmp_path)
    assert [row["event_id"] for row in duplicated][1:] == ["e" * 32, "e" * 32]
    forensic = c.custody_rows(tmp_path)
    assert [row["type"] for row in forensic] == [c.STARTED, c.SETTLED], "one row per event id"
    assert vars(c.replay(tmp_path, rows=list(forensic))["run"]) == vars(c.replay(tmp_path)["run"])
    assert current.fold(duplicated)["runs"]["run"][0]["settled"] is True


def test_the_outbox_delete_follows_a_locked_fsynced_append_on_a_restored_line_boundary(tmp_path, monkeypatch):
    """B3: the chain append the outbox delete waits for is durable — under the append lock,
    the line boundary restored when the file ends mid-line, the file fsynced (the directory
    on create) — and the delete comes only after; an append that is not durable keeps the
    outbox row for the next flush, whose replay the fold dedups by ``event_id``."""
    import os

    path = c.event_log_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"ts": "2026-01-01T00:00:00+00:00", "type": "torn"}), encoding="utf-8")  # no newline: a torn tail
    order = []
    real_fsync, real_delete = os.fsync, store.Txn.outbox_delete

    def fsync(fd):
        real_fsync(fd)
        order.append(("fsync", os.fstat(fd).st_ino))

    def delete(self, event_id):
        order.append("delete")
        return real_delete(self, event_id)

    with monkeypatch.context() as patched:
        patched.setattr(current, "append_jsonl", lambda *a, **kw: (order.append(("append", kw)), False)[1])
        assert c.emit(tmp_path, c.STARTED, {"run_id": "run", "task_id": "task"})  # committed; the append did not confirm
    assert order == [("append", {"ensure_record_boundary": True, "require_lock": True})]  # locked, boundary restored
    assert current.outbox_depth(tmp_path) == 1 and path.read_text(encoding="utf-8").count("\n") == 0
    order.clear()
    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(store.Txn, "outbox_delete", delete)
    assert current.flush_outbox(tmp_path) == 1
    chain = ("fsync", path.stat().st_ino)
    assert chain in order and order.index(chain) < order.index("delete")  # durable, THEN deleted
    lines = path.read_text(encoding="utf-8").split("\n")
    assert json.loads(lines[0])["type"] == "torn" and json.loads(lines[1])["type"] == c.STARTED and lines[2] == ""
    assert current.outbox_depth(tmp_path) == 0
    # the fsync fails: the line is in the file, the row stays, the next flush replays it and the fold dedups
    with monkeypatch.context() as patched:
        patched.setattr(os, "fsync", lambda fd: (_ for _ in ()).throw(OSError(5, "Input/output error")))
        assert c.emit(tmp_path, c.SETTLED, {"run_id": "run", "task_id": "task", "state": "succeeded"})
    assert current.outbox_depth(tmp_path) == 1 and [row["type"] for row in _chain(tmp_path)][1:] == [c.STARTED, c.SETTLED]
    assert current.flush_outbox(tmp_path) == 1 and current.outbox_depth(tmp_path) == 0
    assert [row["type"] for row in c.custody_rows(tmp_path)] == [c.STARTED, c.SETTLED]  # one row per event id


def test_a_repeated_started_keeps_the_settlement_and_the_containment_fault(tmp_path):
    """M6: an idempotent re-start after SETTLED keeps the run's cost, spend disclosure,
    engine model/profile and terminal state; one after CONTAINMENT_FAULT keeps the fault —
    on the live reducer and on the importer's fold alike."""
    settled = {"run_id": "run", "task_id": "task", "state": "failed", "outcome_reason": "timeout", "model": "engine-m",
               "access_profile": "profile-a", "cost_usd": 0.75, "spend_disclosed": True, "spend_estimated": True}
    assert c.emit(tmp_path, c.STARTED, {"run_id": "run", "task_id": "task", "model": "requested-m"})
    assert c.emit(tmp_path, c.SETTLED, settled)
    assert c.emit(tmp_path, c.STARTED, {"run_id": "run", "task_id": "task", "model": "requested-m"})
    assert c.emit(tmp_path, c.STARTED, {"run_id": "faulted", "task_id": "task"})
    assert c.emit(tmp_path, c.CONTAINMENT_FAULT, {"run_id": "faulted", "task_id": "task", "reason": "cancel_unverified"})
    assert c.emit(tmp_path, c.STARTED, {"run_id": "faulted", "task_id": "task"})

    def facts(run):
        return (run.settled, run.terminal_state, run.terminal_reason, run.engine_model, run.access_profile,
                run.cost_usd, run.spend_disclosed, run.spend_estimated, run.containment_fault.get("reason"))

    expected = {"run": (True, "failed", "timeout", "engine-m", "profile-a", 0.75, True, True, None),
                "faulted": (False, "", "", "", "", None, False, False, "cancel_unverified")}
    live = c.replay(tmp_path)
    assert {rid: facts(run) for rid, run in live.items()} == expected
    folded = c.replay(tmp_path, rows=list(c.custody_rows(tmp_path)))
    assert {rid: facts(run) for rid, run in folded.items()} == expected
    assert sorted(run.run_id for run in c.open_runs(tmp_path)) == ["faulted"]  # the settled run stays closed


def test_the_first_custody_append_fsyncs_the_logs_directory(tmp_path, monkeypatch):
    synced, logs = [], c.event_log_path(tmp_path).parent
    monkeypatch.setattr(store, "fsync_directory", lambda directory: synced.append(directory))
    assert c.emit(tmp_path, c.STARTED, {"run_id": "run", "task_id": "task"})
    assert synced.count(logs) == 1
    assert c.emit(tmp_path, c.SETTLED, {"run_id": "run", "task_id": "task", "state": "succeeded"})
    assert synced.count(logs) == 1  # on create only


def test_a_pending_invocation_becomes_its_run_or_its_refusal_in_one_transaction(tmp_path, monkeypatch):
    request = {"invocation_id": "inv", "task_id": "task", "route": "codex", "surface": "delegate",
               "request": {"prompt": "work"}, "billing_group": {"billing_group_id": "group-1"}}
    assert c.emit(tmp_path, c.START_REQUESTED, request)
    assert [row["invocation_id"] for row in c.pending_invocations(tmp_path)] == ["inv"]
    assert c.invocation_record(tmp_path, "inv")["state"] == "pending"

    writes, seen = _writes(monkeypatch), []

    def append(path, row, **kwargs):  # the first observation after the reducing transaction committed
        with store.open_store(tmp_path, write=False) as txn:
            seen.append((txn.get_row("custody_open", "invocation:inv"),
                         txn.custody_rows("invocation_id=?", ("inv",))["run"]["request_event"]["request"]))
        return real_append(path, row, **kwargs)

    real_append = current.append_jsonl
    monkeypatch.setattr(current, "append_jsonl", append)
    assert c.emit(tmp_path, c.STARTED, {"run_id": "run", "task_id": "task", "invocation_id": "inv"})
    assert writes[0] is True and seen == [(None, {"prompt": "work"})], \
        "the pending row leaves and the run row lands in the ONE reducing transaction"
    record = c.invocation_record(tmp_path, "inv")
    assert record["state"] == "started" and record["run_id"] == "run" and record["request"] == {"prompt": "work"}
    assert c.replay(tmp_path)["run"].billing_group == {"billing_group_id": "group-1"}
    assert current.tree_billing_group(tmp_path, "task") is None  # the run names no tree root

    assert c.emit(tmp_path, c.START_REQUESTED, {**request, "invocation_id": "inv-2"})
    assert c.emit(tmp_path, c.START_FAILED, {"invocation_id": "inv-2", "task_id": "task", "error": "transient"})
    assert c.invocation_record(tmp_path, "inv-2")["state"] == "pending", "a transient failure keeps the request"
    assert c.emit(tmp_path, c.START_FAILED, {"invocation_id": "inv-2", "task_id": "task", "definite": True,
                                             "error": "refused"})
    assert c.pending_invocations(tmp_path) == []
    refused = c.invocation_record(tmp_path, "inv-2")
    assert refused["state"] == "failed_definite" and refused["request"] == {"prompt": "work"}
    assert "inv-2" in o.members(tmp_path, "custody_refused")


def test_the_open_set_is_the_held_runs_and_the_pending_invocations(tmp_path):
    assert c.emit(tmp_path, c.STARTED, {"run_id": "live", "task_id": "a"})
    assert c.emit(tmp_path, c.STARTED, {"run_id": "owner", "task_id": "b", "project_id": "p", "project_owned": True})
    assert c.emit(tmp_path, c.SETTLED, {"run_id": "owner", "task_id": "b", "state": "succeeded"})
    assert c.emit(tmp_path, c.STARTED, {"run_id": "done", "task_id": "c"})
    assert c.emit(tmp_path, c.SETTLED, {"run_id": "done", "task_id": "c", "state": "succeeded"})
    assert c.emit(tmp_path, c.START_REQUESTED, {"invocation_id": "inv", "task_id": "d", "request": {"prompt": "x"}})

    view = current.snapshot(tmp_path, closed=False)
    assert set(view["state"]) == {"live", "owner"}, "unsettled, or settled with an owned registration"
    assert [row["invocation_id"] for row in view["pending"]] == ["inv"]
    assert set(c.replay(tmp_path)) == {"live", "owner", "done"}
    with store.open_store(tmp_path, write=False) as txn:
        assert {rid: facts["settled"] for rid, facts in txn.custody_rows("state=?", (current.CLOSED,)).items()} == {"done": True}
    assert current.task_snapshot(tmp_path, "d") == {"runs": {}, "pending": view["pending"]}
    assert set(current.task_snapshot(tmp_path, "c")["runs"]) == {"done"}


def test_the_export_preflight_flushes_the_outbox_and_refuses_a_stuck_one(tmp_path, monkeypatch):
    from ouroboros import store_switch
    from ouroboros.startup_migrations import prepare_startup_state

    prepare_startup_state(tmp_path)
    needs = {"facts": True, "money": False, "handles": False}
    with monkeypatch.context() as patched:
        patched.setattr(current, "append_jsonl", lambda *a, **kw: False)
        assert c.emit(tmp_path, c.STARTED, {"run_id": "run", "task_id": "task"})
        assert current.outbox_depth(tmp_path) == 1 and _chain(tmp_path) == []
        refusal = store_switch._preflight(tmp_path, needs, {"steps": {}}, [tmp_path])
        assert "custody outbox" in refusal and "1 unappended" in refusal
        assert current.outbox_depth(tmp_path) == 1, "a refused preflight moves nothing"
    assert store_switch._preflight(tmp_path, needs, {"steps": {}}, [tmp_path]) == ""
    assert current.outbox_depth(tmp_path) == 0
    assert [row["type"] for row in _chain(tmp_path)] == [c.STARTED]
    # a money-only export does not touch the chain
    with monkeypatch.context() as patched:
        patched.setattr(current, "flush_outbox", lambda root: pytest.fail("flushed on a chainless export"))
        assert store_switch._preflight(tmp_path, {"facts": False, "money": False, "handles": False}, {"steps": {}}, []) == ""


def test_an_unreadable_chain_aborts_the_import_typed_and_a_missing_one_imports_empty(tmp_path, monkeypatch):
    """EMFILE/EIO while folding the custody chain: the import aborts typed, builds and stamps
    nothing and the next boot retries — never a published store with no custody row. A chain
    that does not exist is the positively empty state and imports."""
    from ouroboros.store_import import ensure_current_store

    path = c.event_log_path(tmp_path)
    append_jsonl(path, {"ts": "2026-01-01T00:00:00+00:00", "type": c.STARTED, "run_id": "run-a", "task_id": "t1"})
    real_iter_rows = c._iter_rows

    def exhausted(chain, **kwargs):
        raise OSError(errno.EMFILE, "Too many open files")

    monkeypatch.setattr(c, "_iter_rows", exhausted)
    with pytest.raises(store.StoreBusy, match="custody chain not readable \\(EMFILE\\)"):
        ensure_current_store(tmp_path)
    assert store.probe(tmp_path)[0] == "absent" and not store.generation_stamped(tmp_path)
    monkeypatch.setattr(c, "_iter_rows", real_iter_rows)
    rotated = path.with_name("events.jsonl.part")
    path.rename(rotated)
    path.mkdir()  # exists, cannot be opened: the chain probe's own unreadable answer
    with pytest.raises(store.StoreError, match="custody chain not readable \\(EIO\\)"):
        ensure_current_store(tmp_path)
    assert store.probe(tmp_path)[0] == "absent" and not store.generation_stamped(tmp_path)
    path.rmdir()
    rotated.rename(path)
    assert ensure_current_store(tmp_path)["status"] == "completed"  # the next boot retries and folds the row
    assert [run.run_id for run in c.open_runs(tmp_path)] == ["run-a"]
    other = tmp_path / "fresh"
    other.mkdir()
    assert ensure_current_store(other)["status"] == "completed" and c.open_runs(other) == []


def test_the_strict_chain_reader_raises_a_read_error_instead_of_amputating_the_chain(tmp_path, monkeypatch):
    from ouroboros import utils

    path = c.event_log_path(tmp_path)
    append_jsonl(path, {"ts": "2026-01-01T00:00:00+00:00", "type": c.STARTED, "run_id": "run-a", "task_id": "t1"})

    class Failing:
        def __init__(self, handle):
            self.handle = handle

        def __iter__(self):
            raise OSError(errno.EIO, "Input/output error")

    real = utils.jsonl_chain_handles

    @contextmanager
    def failing_handles(chain, **kwargs):
        with real(chain, **kwargs) as handles:
            yield [(p, Failing(h)) for p, h in handles]

    monkeypatch.setattr(utils, "jsonl_chain_handles", failing_handles)
    assert list(c._iter_rows(path)) == []  # the fail-soft readers keep their lenient answer
    with pytest.raises(OSError, match="Input/output error"):
        list(c._iter_rows(path, strict=True))


def test_the_import_folds_a_legacy_chain_into_the_custody_tables_once(tmp_path, monkeypatch):
    """A tree an old body wrote: no store, a retained chain. The first touch folds the chain
    once into ``custody_runs``, the pending invocations, the refusals and the task marks."""
    from ouroboros.store_import import ensure_current_store

    path = c.event_log_path(tmp_path)
    for row in (
        {"type": c.START_REQUESTED, "invocation_id": "inv-a", "task_id": "t1", "request": {"prompt": "a"}},
        {"type": c.STARTED, "run_id": "run-a", "task_id": "t1", "invocation_id": "inv-a", "root_task_id": "root"},
        {"type": c.SETTLED, "run_id": "run-a", "task_id": "t1", "state": "succeeded", "spendUsd": 0.5},
        {"type": c.START_REQUESTED, "invocation_id": "inv-b", "task_id": "t2", "request": {"prompt": "b"}},
        {"type": c.START_REQUESTED, "invocation_id": "inv-c", "task_id": "t3", "request": {"prompt": "c"}},
        {"type": c.START_FAILED, "invocation_id": "inv-c", "task_id": "t3", "definite": True},
        {"type": c.STARTED, "run_id": "run-d", "task_id": "t4", "root_task_id": "root"},
        {"type": c.CONTAINMENT_FAULT, "run_id": "run-x", "task_id": "t5", "reason": "cancel_unverified"},
        {"type": "delegate_run_patch_verdict", "task_id": "t4", "run_id": "run-d", "verdict": "accepted"},
    ):
        append_jsonl(path, {"ts": "2026-01-01T00:00:00+00:00", **row})
    folds = []
    real_fold = current.fold
    monkeypatch.setattr(current, "fold", lambda rows: folds.append(len(rows)) or real_fold(rows))
    ensure_current_store(tmp_path)
    assert folds == [9], "the chain is folded exactly once, by the importer"

    runs = c.replay(tmp_path)
    assert set(runs) == {"run-a", "run-d", "run-x"}
    assert runs["run-a"].settled and runs["run-a"].request_event["request"] == {"prompt": "a"}
    assert not runs["run-d"].settled and runs["run-x"].containment_fault["reason"] == "cancel_unverified"
    assert sorted(run.run_id for run in c.open_runs(tmp_path)) == ["run-d", "run-x"]
    assert [row["invocation_id"] for row in c.pending_invocations(tmp_path)] == ["inv-b"]
    assert c.invocation_record(tmp_path, "inv-c")["state"] == "failed_definite"
    assert current.marks(tmp_path, "t4")["patch_verdicts"][0]["verdict"] == "accepted"
    assert sorted(current.tree_task_ids(tmp_path, "root")) == ["t1", "t4"]
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.meta("import")["custody_runs"] == 3
    # the chain is left as it was: the forensic read and the rows agree
    assert {row["run_id"] for row in c.custody_rows(tmp_path) if row.get("run_id")} == {"run-a", "run-d", "run-x"}
    assert vars(c.replay(tmp_path, rows=list(c.custody_rows(tmp_path)))["run-a"]) == vars(runs["run-a"])


@pytest.mark.parametrize("journal", ["PERSIST", "WAL"])
def test_task_snapshot_materializes_only_own_runs_and_pending_requests(tmp_path, monkeypatch, journal):
    from ouroboros import delegate_pending, delegate_terminal
    from ouroboros.task_finalization import prepare_terminal_send_event
    from tests._store_journal_shared import use_journal

    use_journal(monkeypatch, journal)
    with store.open_store(tmp_path, write=True) as txn:
        for index in range(1000):
            tid = f"other-{index}"
            txn.put_custody(tid, {"run_id": tid, "task_id": tid, "model": "x" * 1000}, state=current.HELD)
            txn.put_row("custody_open", f"invocation:{tid}", {"request": {
                "type": c.START_REQUESTED, "task_id": tid, "invocation_id": tid,
                "request_ref": {"sha256": tid},
            }})
        for rid, fields in (("own-open", {}), ("own-closed", {"settled": True, "terminal_state": "failed"}),
                            ("own-review", {"source": "review_substrate"})):
            txn.put_custody(rid, {"run_id": rid, "task_id": "own", **fields},
                            state=current.CLOSED if fields.get("settled") else current.HELD)
        for iid, source in (("own-pending", "delegated_subagent"), ("review-pending", "review_substrate")):
            txn.put_row("custody_open", f"invocation:{iid}", {"request": {
                "type": c.START_REQUESTED, "task_id": "own", "root_task_id": "tree-root",
                "invocation_id": iid, "source": source, "request_ref": {"sha256": iid},
            }})

    decoded, hydrated, writes = [], [], []
    decode, body, opened = current._decode, delegate_pending.request_body, store.open_store

    def owned_decode(raw):
        decoded.append(raw["run_id"])
        assert raw["task_id"] == "own"
        return decode(raw)

    def owned_request(root, row):
        hydrated.append(row["invocation_id"])
        assert row["task_id"] == "own"
        return body(root, row)  # unreadable own source is still a pending identity

    @contextmanager
    def observed_open(root, **kwargs):
        if kwargs.get("write"):
            writes.append(root)
        with opened(root, **kwargs) as txn:
            yield txn

    monkeypatch.setattr(current, "_decode", owned_decode)
    monkeypatch.setattr(delegate_pending, "request_body", owned_request)
    monkeypatch.setattr(store, "open_store", observed_open)
    monkeypatch.setattr(store.Txn, "set_rows", lambda *a: pytest.fail("delivery read an entire obligation set"))
    usage = {}
    event = prepare_terminal_send_event(tmp_path, {"id": "own"}, "Answer", usage,
                                        {"text": "Answer"}, presence=False)
    audit = usage["delegate_terminal_reconciliation"]
    assert audit["audit_status"] == "ok" and not writes
    assert sorted(decoded) == ["own-closed", "own-open", "own-review"]
    assert hydrated == [], "typed pending identities need no historical request body"
    assert audit["open_run_ids"] == ["own-open"]
    assert audit["pending_invocation_ids"] == ["own-pending"]
    assert [row["run_id"] for row in audit["terminal_runs"]] == ["own-closed"]
    assert "own-open" in event["terminal_custody_notice"] and "own-pending" in event["terminal_custody_notice"]
    assert "review" not in event["terminal_custody_notice"] and "other-" not in event["terminal_custody_notice"]
    assert delegate_terminal.terminal_custody_disclosure(tmp_path, "absent")["unreconciled"] == []


def test_task_snapshot_keeps_a_legacy_numeric_request_task_identity(tmp_path):
    with store.open_store(tmp_path, write=True) as txn:
        txn.put_row("custody_open", "invocation:legacy", {"request": {
            "type": c.START_REQUESTED, "task_id": 17, "invocation_id": "legacy",
            "request": {"prompt": "work"},
        }})
    assert [row["invocation_id"] for row in current.task_snapshot(tmp_path, "17")["pending"]] == ["legacy"]
    assert current.task_snapshot(tmp_path, "18")["pending"] == []


def test_a_flush_records_the_chain_length_exactly_and_lists_the_archive_only_after_it_changed(tmp_path, monkeypatch):
    """Every flush records the chain's length right after its own append (the fold's P6' boundary,
    ``CHAIN_BOUNDARY``) exactly: before a rotation, after one, after another writer's rows, after a
    segment is removed. It reads that length from the previous flush's measurement (``CHAIN_LIVE``:
    the live segment's identity and size, the archive directory's stamp): while the closed segments
    are unchanged a flush lists no archive and opens no segment (one stat of the directory), and a
    rotation or a removed segment is measured once by the walk, never on every emit and never inside
    a store transaction (the chain is measured between the append and the row's delete)."""
    from supervisor.state import rotate_jsonl_log_if_needed
    from tests._cost_meter import CostMeter, tagged

    path = c.event_log_path(tmp_path)
    archive = tmp_path / "archive"
    open_transactions, measured_inside = [0], []
    real_open, real_measure = store.open_store, c.chain_measure

    @contextmanager
    def counted(root, **kwargs):
        with real_open(root, **kwargs) as txn:
            open_transactions[0] += 1
            try:
                yield txn
            finally:
                open_transactions[0] -= 1

    def measure(chain):
        measured_inside.append(open_transactions[0] > 0)
        return real_measure(chain)

    monkeypatch.setattr(store, "open_store", counted)
    monkeypatch.setattr(c, "chain_measure", measure)

    def emit(number):
        """Commit the event with its append parked, then meter the flush alone."""
        tag = f"flush-{number}"
        with monkeypatch.context() as parked:
            parked.setattr(current, "_append_durable", lambda *args: None)
            assert c.emit(tmp_path, c.STARTED, {"run_id": f"run-{number}", "task_id": "task"})
        with CostMeter(tmp_path) as meter:
            assert tagged(tag, lambda: current.flush_outbox(tmp_path)) == 1
        assert current.outbox_depth(tmp_path) == 0
        with store.open_store(tmp_path, write=False) as txn:
            boundary, live = txn.meta(current.CHAIN_BOUNDARY), txn.meta(current.CHAIN_LIVE)
        assert live["offset"] == c.chain_length(path), f"flush {number} measured the chain wrongly"
        counts = meter.snapshot(tag)["counts"]
        return boundary, {key: counts.get(key, 0) for key in ("scandir:archive", "open:archive", "stat:archive")}

    rested = {"scandir:archive": 0, "open:archive": 0, "stat:archive": 1}  # the directory's stamp, nothing listed or opened
    boundary, counts = emit(1)  # the first flush measures
    assert boundary == c.chain_length(path) and counts["scandir:archive"] == 1
    assert emit(2) == (c.chain_length(path), rested)  # the next advances by its own growth
    for number in range(3):  # another writer's rows grow the live file between flushes
        assert append_jsonl(path, {"ts": "2026-01-01T00:00:00+00:00", "type": "task_received", "n": number})
    assert emit(3) == (c.chain_length(path), rested)
    rotate_jsonl_log_if_needed(tmp_path, "events.jsonl", "events", max_bytes=1)
    assert len(list(archive.glob("events_*.jsonl"))) == 1 and path.stat().st_size == 0
    boundary, counts = emit(4)  # a rotation: measured once by the walk (the segment opened, the live file empty)
    assert boundary == c.chain_length(path) and counts["scandir:archive"] == 1
    assert emit(5) == (c.chain_length(path), rested)
    rotate_jsonl_log_if_needed(tmp_path, "events.jsonl", "events", max_bytes=1)
    boundary, counts = emit(6)
    assert boundary == c.chain_length(path) and counts["scandir:archive"] == 1
    assert emit(7) == (c.chain_length(path), rested)
    oldest = sorted(archive.glob("events_*.jsonl"))[0]
    oldest.unlink()  # a removed segment: the directory's stamp moves, the next flush measures again
    _boundary, counts = emit(8)
    assert counts["scandir:archive"] == 1
    assert emit(9)[1] == rested
    rows = [row["run_id"] for row in c.custody_rows(tmp_path)]  # every flush landed; the removed segment took 1-3
    assert rows == [f"run-{number}" for number in range(4, 10)], rows
    # The first flush, two rotations and the removal each measured the chain; none did so under a store transaction.
    assert len(measured_inside) >= 4 and not any(measured_inside), measured_inside
