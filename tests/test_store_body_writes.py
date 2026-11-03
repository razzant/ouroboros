"""The task body's write path: a rewrite updates the row in place (a same-size body journals only its
changed pages on the rollback journal), an unchanged body is not written while the row, its handoffs
and the terminal counter still move, and the mutator never gets the pristine document."""
from __future__ import annotations

import json

import pytest

from ouroboros import store, task_store
from ouroboros.task_results import load_task_result, write_task_result
from tests._store_journal_shared import use_journal


@pytest.fixture(autouse=True)
def release_connections():
    store.close_connections()
    yield
    store.close_connections()


@pytest.mark.parametrize("mode", ["PERSIST", "WAL"])
def test_a_same_size_body_rewrite_journals_only_its_changed_pages(tmp_path, monkeypatch, mode):
    use_journal(monkeypatch, mode)
    write_task_result(tmp_path, "big", "running", payload="y" * 4_000_000)
    path = store.store_path(tmp_path)
    journal = path.with_name(path.name + "-journal")
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.conn.execute("PRAGMA journal_size_limit").fetchone()[0] == store.JOURNAL_SIZE_LIMIT == 64 << 20
    before = journal.stat().st_size if mode == "PERSIST" else 0
    task_store.patch_document(tmp_path, "big", lambda document: {**document, "payload": "x" + document["payload"][1:]})
    assert load_task_result(tmp_path, "big")["payload"].startswith("xy")
    if mode == "WAL":
        return  # the same statement, no rollback journal to measure: the content above is the semantics
    assert journal.stat().st_size - before < 200_000, "the in-place rewrite journaled the whole body"
    # The control: the same change through the retired REPLACE journals the whole old chain.
    with store.open_store(tmp_path, write=True) as txn:
        document = txn.load("big")
        document["payload"] = "z" + document["payload"][1:]
        txn.conn.execute("INSERT OR REPLACE INTO task_bodies (task_id, body) VALUES (?, ?)",
                         ("big", store.encode_document(document)))
    assert journal.stat().st_size - before > 3_000_000


def _watch_body_writes(root):
    """TEMP triggers on this thread's cached connection: every INSERT or UPDATE of ``task_bodies`` leaves a row."""
    with store.open_store(root, write=True) as txn:
        txn.conn.execute("CREATE TEMP TABLE body_writes (kind TEXT)")
        for kind in ("INSERT", "UPDATE"):
            txn.conn.execute(f"CREATE TEMP TRIGGER body_{kind} AFTER {kind} ON task_bodies "
                             f"BEGIN INSERT INTO body_writes VALUES ('{kind}'); END")


def _body_writes(root):
    with store.open_store(root, write=False) as txn:
        return [row[0] for row in txn.conn.execute("SELECT kind FROM body_writes")]


def test_an_unchanged_body_is_not_rewritten_while_the_row_and_the_counters_still_move(tmp_path):
    write_task_result(tmp_path, "t", "completed", result="r", launch_handoffs={"op": {"state": "claimed"}})
    _watch_body_writes(tmp_path)
    with store.open_store(tmp_path, write=False) as txn:
        seq, terminal, body = txn.row("t", "write_seq"), txn.meta(store.TERMINAL_WRITES), txn.body_text("t")
        assert json.loads(body)["launch_handoffs"] == {}  # the body keeps the key's place; the rows hold its content
    # A row-backed field alone: the body bytes are the same, the handoff row, the signature and the terminal counter move.
    task_store.patch_document(tmp_path, "t", lambda document: {**document, "launch_handoffs": {"op": {"state": "returned"}}})
    store.update_task(tmp_path, "t", lambda document: document, pending_drive=True)  # a typed column alone
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.body_text("t") == body and txn.handoff("t", "op") == {"state": "returned"}
        assert txn.row("t", "write_seq", "pending_drive") == (seq + 2, 1)
        assert txn.meta(store.TERMINAL_WRITES) == terminal + 2
    assert _body_writes(tmp_path) == []
    task_store.patch_document(tmp_path, "t", lambda document: {**document, "result": "changed"})
    assert _body_writes(tmp_path) == ["UPDATE"]
    write_task_result(tmp_path, "fresh", "running")
    assert _body_writes(tmp_path) == ["UPDATE", "INSERT"]


def test_update_task_decodes_twice_and_encodes_once_so_the_mutator_never_holds_the_pristine_document(tmp_path, monkeypatch):
    write_task_result(tmp_path, "t", "running", nested={"items": [1]}, launch_handoffs={"op": {"state": "claimed"}})
    with store.open_store(tmp_path, write=False) as txn:
        stored = txn.body_text("t")
    seen, decodes = {}, []
    real_loads, real_dumps = json.loads, json.dumps

    def counted_loads(text, *args, **kwargs):
        if text == stored:
            decodes.append(text)
        return real_loads(text, *args, **kwargs)

    def guarded_dumps(value, *args, **kwargs):  # the deep copy went through here: the pristine document is never encoded
        assert not (isinstance(value, dict) and value.get("nested") == {"items": [1]}), "the pristine document was encoded under the lock"
        return real_dumps(value, *args, **kwargs)

    def rule(_txn, current, updated):
        seen["current"], seen["updated"] = current, updated
        return True

    def mutate(document):
        document["nested"]["items"].append(2)
        document["launch_handoffs"]["op"]["state"] = "returned"
        return document

    monkeypatch.setattr(json, "loads", counted_loads)
    monkeypatch.setattr(json, "dumps", guarded_dumps)
    store.update_task(tmp_path, "t", mutate, pending_drive=rule)
    assert len(decodes) == 2, "one pristine and one working decode of the stored bytes (a deep copy decodes it once)"
    assert seen["current"]["nested"] == {"items": [1]} and seen["current"]["launch_handoffs"]["op"]["state"] == "claimed"
    assert seen["updated"]["nested"] == {"items": [1, 2]}
    document = load_task_result(tmp_path, "t", strict=True)
    assert document["nested"]["items"] == [1, 2] and document["launch_handoffs"]["op"]["state"] == "returned"


def test_a_terminal_row_keeps_the_count_it_was_written_at_and_the_rows_written_since_are_one_indexed_read(tmp_path):
    """``terminal_seq`` is the terminal-write counter's value at the row's last terminal write: a running row
    carries none, a settled row the count, a settled row written again the new count. The rows written since a
    count are read through ``tasks_terminal_seq`` (``task_store.terminal_writes_since``), never the others, so a
    pass that kept the count looks at those rows alone instead of at every row it holds a verdict on."""
    write_task_result(tmp_path, "live", "running")
    write_task_result(tmp_path, "done", "completed", result="x")
    with store.open_store(tmp_path, write=False) as txn:
        count = txn.meta(store.TERMINAL_WRITES)
        assert txn.row("live", "terminal_seq") is None and txn.row("done", "terminal_seq") == count
    assert task_store.terminal_writes_since(tmp_path, count - 1) == ["done"]
    assert task_store.terminal_writes_since(tmp_path, count) == []
    write_task_result(tmp_path, "done", "completed", note="written again")
    write_task_result(tmp_path, "later", "failed", result="y")
    task_store.patch_handoff(tmp_path, "live", "launch_handoffs", "op", lambda _row: {"state": "claimed"})  # a live row's write
    assert task_store.terminal_writes_since(tmp_path, count) == ["done", "later"]
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.row("done", "terminal_seq") == count + 1 and txn.row("live", "terminal_seq") is None
        plan = " ".join(str(row[-1]) for row in txn.conn.execute(
            "EXPLAIN QUERY PLAN SELECT task_id FROM tasks INDEXED BY tasks_terminal_seq WHERE terminal_seq>0"))
        assert "tasks_terminal_seq" in plan, plan
