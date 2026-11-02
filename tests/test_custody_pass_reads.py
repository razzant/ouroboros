"""The 600 s custody pass reads what changed (FINAL_PLAN E2; tests/test_cost_live_set.py's custody rows).

Its audits (the snapshot GC's keep-set, the registration duty, the receipt refresh's audit snapshot) and the
delegated body's tick all ask for the held set; the held set is mostly settled runs that keep a patch or a
registration for days. These laws hold the readers to the rows written since this process last read them:

- ``delegate_custody_current.held_runs`` (every held row, or those a predicate selects) parses a row's facts
  once per stamp (``write_seq``, ``updated_at``), shared by every reader of the held set; a row that leaves it
  is gone from the next answer;
- the receipt refresh (``delegate_terminal.refresh_recently_settled_terminals``) rests a task whose refresh found
  nothing to write until one of its inputs moves; a receipt still owing money never rests.
"""
from __future__ import annotations

import pytest

from ouroboros import delegate_custody as custody
from ouroboros import delegate_custody_current as current
from ouroboros import delegate_terminal, store
from ouroboros.task_results import load_task_result, write_task_result


def _held(run_id: str, task_id: str = "", **facts) -> dict:
    """A settled run custody still holds for its undisposed patch (the live majority of the held set)."""
    return {"run_id": run_id, "task_id": task_id or f"{run_id}-task", "root_task_id": task_id or f"{run_id}-task",
            "settled": True, "terminal_state": "completed", "snapshot_id": f"snap-{run_id}", "mode": "agent", **facts}


def _put(root, *rows: dict, state: str = current.HELD) -> None:
    with store.open_store(root, write=True) as txn:
        for facts in rows:
            txn.put_custody(facts["run_id"], facts, state=state)


def _parses(monkeypatch) -> list:
    """The custody facts this module parses (its ``json_loads`` alias), one entry per row text."""
    parsed, real = [], current.json_loads
    monkeypatch.setattr(current, "json_loads", lambda text: parsed.append(1) or real(text))
    return parsed


# ---- the held set, parsed once per stamp ---------------------------------------------------------------------------

def test_an_unchanged_held_row_is_parsed_once_whoever_asks(tmp_path, monkeypatch):
    _put(tmp_path, *(_held(f"run{index}") for index in range(5)), _held("open", settled=False))
    parsed = _parses(monkeypatch)
    assert set(current.held_runs(tmp_path)) == {"run0", "run1", "run2", "run3", "run4", "open"}
    assert len(parsed) == 6
    parsed.clear()
    assert set(current.held_runs(tmp_path)) == {"run0", "run1", "run2", "run3", "run4", "open"}
    assert set(current.open_held_runs(tmp_path)) == {"open"}, "a predicate's selection reads its rows from the same memory"
    assert sorted(entry.snapshot_id for entry in current.snapshot(tmp_path)["state"].values()) == sorted(
        f"snap-{name}" for name in ("run0", "run1", "run2", "run3", "run4", "open"))
    assert parsed == [], "no row moved: no facts are fetched or parsed"
    _put(tmp_path, _held("run3", patch_disposed="rejected"))  # its owner rejected the patch: one write
    entries = current.held_runs(tmp_path)
    assert len(parsed) == 1 and entries["run3"].patch_disposed == "rejected"


def test_a_row_that_left_the_held_set_leaves_the_answer(tmp_path):
    _put(tmp_path, _held("kept"), _held("closing"))
    assert set(current.held_runs(tmp_path)) == {"kept", "closing"}
    _put(tmp_path, _held("closing", patch_disposed="applied"), state=current.CLOSED)
    assert set(current.held_runs(tmp_path)) == {"kept"}
    assert set(current._HELD_SEEN[str(tmp_path)]) == {"kept"}, "the memory keeps the held set only"


def test_each_reader_receives_its_own_entries(tmp_path):
    _put(tmp_path, _held("run"))
    first = current.held_runs(tmp_path)["run"]
    first.snapshot_id = "mutated by its reader"
    first.verified_source_ranges.append((1, 2))
    again = current.held_runs(tmp_path)["run"]
    assert again.snapshot_id == "snap-run" and again.verified_source_ranges == []


@pytest.mark.parametrize("held", [1, 50])
def test_the_snapshot_keep_set_parses_nothing_over_an_unchanged_held_set(tmp_path, monkeypatch, held):
    """Equal at 1 and 50 settled held runs: the second keep-set read parses no row (the live-set law at unit size)."""
    _put(tmp_path, *(_held(f"run{index:03d}") for index in range(held)))
    assert len(custody.open_snapshot_ids(tmp_path)) == held
    parsed = _parses(monkeypatch)
    assert len(custody.open_snapshot_ids(tmp_path)) == held
    assert parsed == []


# ---- the receipt refresh rests until its inputs move ----------------------------------------------------------------

def _owed_receipt(root, task_id: str = "owner", run_id: str = "run", **debt) -> None:
    """A terminal task whose closing receipt no audit can retire yet: its run is still held (an undisposed patch)
    and changed after the receipt was parked, so the receipt and the held row differ."""
    write_task_result(root, task_id, "completed", result="done")
    _put(root, _held(run_id, task_id))
    with store.open_store(root, write=True) as txn:
        txn.put_row("delegated_runs", task_id, {"task_id": task_id, "closed_runs": {
            run_id: _held(run_id, task_id, terminal_state="failed")}, **debt})


def _body_reads(root, call) -> int:
    """Task documents ``call`` reads: whole ones (``load_task_result``) and inside a write (``Txn.load``)."""
    from ouroboros import task_results, task_store

    reads, single, inside = [], task_store.read_document, store.Txn.load
    patch = pytest.MonkeyPatch()
    patch.setattr(task_results, "read_document", lambda r, tid, **kw: reads.append(tid) or single(r, tid, **kw))
    patch.setattr(store.Txn, "load", lambda txn, tid, *a, **kw: reads.append(tid) or inside(txn, tid, *a, **kw))
    try:
        call()
    finally:
        patch.undo()
    return len(reads)


def _settle(root) -> None:
    """Passes until one writes nothing more (a first refresh may record the disclosure, the next one rests)."""
    for _attempt in range(3):
        delegate_terminal.refresh_recently_settled_terminals(root, live_task_ids=set())


def test_a_receipt_no_audit_can_retire_rests_until_its_inputs_move(tmp_path):
    _owed_receipt(tmp_path)
    _settle(tmp_path)
    assert _body_reads(tmp_path, lambda: delegate_terminal.refresh_recently_settled_terminals(
        tmp_path, live_task_ids=set())) == 0, "nothing it reads moved"
    assert "run" in (store_row := _debt(tmp_path))["closed_runs"], "the receipt stays owed: a rest is no discharge"
    _put(tmp_path, _held("run", "owner", patch_disposed="applied"), state=current.CLOSED)  # its run closed
    assert _body_reads(tmp_path, lambda: delegate_terminal.refresh_recently_settled_terminals(
        tmp_path, live_task_ids=set())) > 0, "the run's write puts the task back once"
    assert store_row is not None


def _debt(root) -> dict:
    with store.open_store(root, write=False) as txn:
        return txn.get_row("delegated_runs", "owner") or {}


def test_a_write_of_the_task_puts_it_back(tmp_path):
    _owed_receipt(tmp_path)
    _settle(tmp_path)
    write_task_result(tmp_path, "owner", "completed", result="done, and amended")
    assert _body_reads(tmp_path, lambda: delegate_terminal.refresh_recently_settled_terminals(
        tmp_path, live_task_ids=set())) > 0
    assert load_task_result(tmp_path, "owner")["result"] == "done, and amended"


def test_a_receipt_owing_money_never_rests(tmp_path, monkeypatch):
    from supervisor import queue

    _owed_receipt(tmp_path, cost_revision=3)
    monkeypatch.setattr(queue, "task_has_live_ownership", lambda *a, **k: True)  # its owner still bills
    _settle(tmp_path)
    assert _body_reads(tmp_path, lambda: delegate_terminal.refresh_recently_settled_terminals(
        tmp_path, live_task_ids=set())) > 0, "only its owner's end releases it, and no row records that"
    assert _debt(tmp_path)["cost_revision"] == 3


def test_a_live_owner_is_deferred_and_never_rested(tmp_path):
    _owed_receipt(tmp_path)
    delegate_terminal.refresh_recently_settled_terminals(tmp_path, live_task_ids={"owner"})
    assert str(tmp_path) not in delegate_terminal._RESTS or "owner" not in delegate_terminal._RESTS[str(tmp_path)]


def test_a_receipt_sqlite_cannot_read_breaks_no_pass_and_never_rests(tmp_path):
    """``machine_json`` writes NaN, which Python reads and ``json_valid`` rejects: the pass still runs, and such a
    receipt is examined every pass (it may owe money no SQL could see)."""
    _owed_receipt(tmp_path, spend_usd=float("nan"))
    _settle(tmp_path)
    assert _body_reads(tmp_path, lambda: delegate_terminal.refresh_recently_settled_terminals(
        tmp_path, live_task_ids=set())) > 0
    assert "owner" not in delegate_terminal._RESTS.get(str(tmp_path), {})


def test_the_snapshot_gc_reads_no_document_for_an_owner_with_no_record(tmp_path):
    """``body_candidate._recorded_terminal`` (the snapshot, copy and checkout retirers' owner check): a missing
    record is not terminal, known from the typed row; a recorded terminal still reads as terminal."""
    from ouroboros import body_candidate

    write_task_result(tmp_path, "finished", "completed", result="done")
    assert _body_reads(tmp_path, lambda: body_candidate._recorded_terminal("never-recorded", tmp_path)) == 0
    assert body_candidate._recorded_terminal("never-recorded", tmp_path) is False
    assert body_candidate._recorded_terminal("finished", tmp_path) is True
