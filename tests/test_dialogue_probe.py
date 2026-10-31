"""Probe selection preserves addressed reads while fetching only the winning bodies."""
from contextlib import contextmanager
import json
import sqlite3
from types import SimpleNamespace

import pytest

from ouroboros import dialogue_store as ds, store
from tests._dialogue import seed


class CountedConnection:
    """Count the rows reaching Python, separately from SQLite VM instructions or JSON decodes: returned SQL rows,
    and the rows a one-step page (``dialogue_store._page``, a ``json_group_array``) carries in its one row."""

    def __init__(self, conn, *, legacy=False):
        self.conn, self.legacy = conn, legacy
        self.statements = self.body_rows = self.seq_rows = self.max_bindings = 0

    def __getattr__(self, name):
        if name == "getlimit" and self.legacy:
            raise AttributeError(name)  # Python 3.10 has no limit API
        return getattr(self.conn, name)

    def execute(self, sql, params=()):
        self.max_bindings = max(self.max_bindings, len(params))
        if sql.startswith("SELECT"):
            self.statements += 1

        def counted(cursor, values):
            row = sqlite3.Row(cursor, values)
            if "body" in row.keys():
                self.body_rows += 1
            elif "seq" in row.keys():
                self.seq_rows += 1
            elif len(values) == 1 and str(row.keys()[0]).startswith("json_group_array"):
                for item in json.loads(values[0] or "[]"):
                    self.body_rows += isinstance(item, dict) and "body" in item
                    self.seq_rows += type(item) is int
            return row

        cursor = self.conn.cursor()
        cursor.row_factory = counted
        return cursor.execute(sql, params)


@pytest.fixture
def records():
    rows = []
    for stream in ("chat", "progress"):
        for n in range(40):
            row = dict(chat_id=None if n % 4 == 0 else n % 3, task_id=f"task-{n % 7}",
                       parent_task_id=f"task-{n % 3}", root_task_id="root", subagent_task_id=f"sub-{n % 5}",
                       kind="quiz" if n % 3 == 0 else None, origin=f"1:owner-{n % 3}",
                       client_message_id=f"cm-'?-{n % 8}", text=f"  {stream} {n}: Слова\n\tцеликом  ")
            rows.append((stream, 17 + 101 * n, row))
    return rows


@pytest.fixture
def connection(records):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(ds._SCHEMA)
    for stream, seq, row in records:
        conn.execute("INSERT INTO dialogue(stream,seq,length,segment,ordinal,chat_id,task_id,parent_task_id,"
                     "root_task_id,subagent_task_id,kind,origin,client_message_id,body,dispatch) "
                     "VALUES(?,?,73,1,0,?,?,?,?,?,?,?,?,?,?)",
                     (stream, seq, *(row[key] for key in ("chat_id", "task_id", "parent_task_id", "root_task_id",
                      "subagent_task_id", "kind", "origin", "client_message_id")), json.dumps(row, ensure_ascii=False),
                      "pending" if row["chat_id"] == 1 else None))
    conn.commit()
    yield conn
    conn.close()


def probe(conn, keys, **kwargs):
    return ds._probed(*ds._probe(conn, kwargs.pop("stream", "chat"), keys, **{  # decoded as probe_rows decodes
        "before": None, "lower": 0, "limit": 200, "newest_first": True, **kwargs}))


@pytest.mark.parametrize("stream", ["chat", "progress"])
@pytest.mark.parametrize("newest_first", [False, True])
@pytest.mark.parametrize("limit", [-7, -1, 0, 1, 9, 10_000])
def test_selection_matches_the_addressed_union(connection, records, stream, newest_first, limit):
    key_sets = [[], [("chat_id", None)], [("task_id", None), ("kind", None)],
                [("chat_id", 1), ("task_id", "task-1"), ("chat_id", 1), ("origin", "1:owner-1")],
                [("parent_task_id", "task-2"), ("root_task_id", "root"), ("subagent_task_id", "sub-3")],
                [("client_message_id", "cm-'?-1"), ("kind", "quiz"), ("chat_id", 2)]]
    for keys in key_sets:
        for lower, before in ((0, None), (118, 1936), (118, 118), (1000, 118)):
            expected = [ds.Row(s, seq, row, "pending" if row["chat_id"] == 1 else None, 73)
                        for s, seq, row in records if s == stream and seq >= lower
                        and (before is None or seq < before)
                        and any(row[key] == value and (value is not None or key == "chat_id") for key, value in keys)]
            expected.sort(key=lambda row: row.seq, reverse=newest_first)
            expected = expected[:limit] if limit > 0 else []
            actual = probe(connection, iter(keys), stream=stream, lower=lower, before=before,
                           limit=limit, newest_first=newest_first)
            assert actual == expected, (keys, lower, before)
            assert [r.history_id for r in actual] == [f"{stream}:{r.seq}" for r in expected]
            assert [r.end for r in actual] == [r.seq + 73 for r in expected]


@pytest.mark.parametrize("limit", [-1, 0, 2])
def test_invalid_columns_are_rejected_even_when_no_rows_are_requested(connection, limit):
    for column in ("body", "seq", "task_id OR 1=1", "chat_id; DROP TABLE dialogue"):
        with pytest.raises(ValueError, match="not a dialogue probe column"):
            probe(connection, [("task_id", None), (column, None)], limit=limit)


def _many_keys(conn, count):
    for n in range(count):
        conn.execute("INSERT INTO dialogue(stream,seq,length,segment,ordinal,chat_id,task_id,parent_task_id,body) "
                     "VALUES('chat',?,40,1,?,1,?,?,?)",
                     (n * 100, n, f"task-{n}", f"task-{n // 3}", json.dumps({"text": f"row {n}"})))
    conn.commit()
    return [("chat_id", 1)] + [("task_id", f"task-{n}") for n in range(count)] + [
        ("parent_task_id", f"task-{n}") for n in range(count // 3)]


@pytest.mark.parametrize("newest_first", [False, True])
def test_overlapping_keys_fetch_only_winning_bodies(newest_first, monkeypatch):
    with sqlite3.connect(":memory:") as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(ds._SCHEMA)
        keys = _many_keys(conn, 1205)  # exceeds both the 500-arm and 999-variable SQLite floor
        counted = CountedConnection(conn)
        decoded, decode = [], ds.decode

        def counted_decode(body):
            decoded.append(body)
            return decode(body)

        monkeypatch.setattr(ds, "decode", counted_decode)
        rows = probe(counted, [*keys, *reversed(keys)], limit=80, newest_first=newest_first)
        expected = list(range(1205))[::-1] if newest_first else list(range(1205))
        assert [row.seq for row in rows] == [n * 100 for n in expected[:80]]
        assert counted.body_rows == len(decoded) == 80, "discarded key candidates must never fetch bodies"
        assert 80 < counted.seq_rows <= 3 * 80, "discarded candidates exist, one statement per column bounds them"


@pytest.mark.parametrize("newest_first", [False, True])
def test_coordinates_reaching_python_do_not_grow_with_the_room(newest_first):
    """A room's lens names every bound task three times over; the probe still hands Python at most
    columns x limit coordinates and runs one statement per column, for 30 keys as for 3,000."""
    measured = []
    for count in (10, 1000):
        with sqlite3.connect(":memory:") as conn:
            conn.row_factory = sqlite3.Row
            conn.executescript(ds._SCHEMA)
            keys = _many_keys(conn, count)
            counted = CountedConnection(conn)
            rows = probe(counted, keys, limit=7, newest_first=newest_first)
            expected = list(range(count))[::-1] if newest_first else list(range(count))
            assert [row.seq for row in rows] == [n * 100 for n in expected[:7]]
            measured.append((counted.statements, counted.seq_rows, counted.body_rows))
    assert measured[0] == measured[1] == (4, 21, 7), measured


@pytest.mark.parametrize("variables,compound", [(5, 1), (9, 3), (31, 7), (999, 500)])
@pytest.mark.parametrize("newest_first", [False, True])
def test_runtime_limits_bound_selection_and_hydration(variables, compound, newest_first):
    if not hasattr(sqlite3.Connection, "setlimit"):
        pytest.skip("Python 3.10 has no runtime SQLite limit API; conservative fallback covered separately")
    with sqlite3.connect(":memory:") as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(ds._SCHEMA)
        keys = _many_keys(conn, 80)
        conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, variables)
        conn.setlimit(sqlite3.SQLITE_LIMIT_COMPOUND_SELECT, compound)
        counted = CountedConnection(conn)
        rows = probe(counted, keys, limit=60, newest_first=newest_first)
        expected = list(range(80))[::-1] if newest_first else list(range(80))
        assert [row.seq for row in rows] == [n * 100 for n in expected[:60]]
        assert counted.max_bindings <= variables and counted.body_rows == 60


def test_python310_limit_fallback_and_host_10000_rows(tmp_path, monkeypatch):
    from ouroboros.gateway.host_service import _chat_rows

    rows = [{"chat_id": 1, "task_id": "task", "direction": "out", "text": f"row {n}"} for n in range(10_023)]
    seed(tmp_path, live={"chat": rows + [{"chat_id": 2, "text": "foreign"}]})
    ds.ensure(tmp_path)
    transaction, connections = ds.transaction, []

    @contextmanager
    def limited(*args, **kwargs):
        with transaction(*args, **kwargs) as conn:
            # Hide the API even on newer Python, while enforcing the old engine limits there.
            limits = []
            if hasattr(conn, "setlimit"):
                for category, value in ((sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999),
                                        (sqlite3.SQLITE_LIMIT_COMPOUND_SELECT, 500)):
                    limits.append((category, conn.setlimit(category, value)))
            counted = CountedConnection(conn, legacy=True)
            connections.append(counted)
            try:
                yield counted
            finally:
                for category, value in limits:
                    conn.setlimit(category, value)

    monkeypatch.setattr(ds, "transaction", limited)
    found = ds.probe_rows(tmp_path, "chat", [("chat_id", 1), ("task_id", "task")], limit=10_000)
    assert [row.row for row in reversed(found)] == rows[-10_000:]
    assert connections[-1].body_rows == 10_000 and connections[-1].max_bindings <= 999
    assert _chat_rows(SimpleNamespace(data_dir=tmp_path), 1) == rows[-10_000:]
    assert connections[-1].body_rows == 10_000 and connections[-1].max_bindings <= 999


def test_history_source_pages_an_overlapping_lens_without_fetching_discarded_bodies(tmp_path, monkeypatch):
    from ouroboros.gateway.history_paging import HistorySource, RoomLens, _PROBE_ROWS

    rows = [{"chat_id": 100, "direction": "out", "text": f"row {n}", "task_id": f"task-{n % 40}",
             "root_task_id": f"task-{n % 10}"} for n in range(240)]
    seed(tmp_path, live={"chat": [*rows[:235], "{broken\n", *rows[235:]]})
    ds.ensure(tmp_path)
    keys = [("chat_id", 100)] + [(column, f"task-{n}") for n in range(40)
                                for column in ("task_id", "root_task_id")]
    source = HistorySource(tmp_path, "chat", RoomLens(tuple(keys)))
    transaction, connections = ds.transaction, []

    @contextmanager
    def counted(*args, **kwargs):
        with transaction(*args, **kwargs) as conn:
            measured = CountedConnection(conn)
            connections.append(measured)
            yield measured

    monkeypatch.setattr(ds, "transaction", counted)
    entries, before, gaps = source.older(source.upper, 9, lambda _: True)
    assert [row["text"] for row in entries] == [f"row {n}" for n in range(231, 240)]
    assert before == entries[0]["history_position"]["offset"] and gaps == {"malformed_jsonl"}
    assert sum(conn.body_rows for conn in connections) == _PROBE_ROWS
    replay, replay_before, replay_gaps = source.replay(before, source.upper)
    assert (replay, replay_before, replay_gaps) == (entries, before, gaps)


def test_selection_and_body_hydration_share_the_read_snapshot(tmp_path, monkeypatch):
    # WAL allows the second connection to commit while the reader keeps its snapshot.
    monkeypatch.setattr(store, "_journal_mode", lambda *_: "WAL")
    receipt = ds.append(tmp_path, "chat", {"chat_id": 1, "task_id": "task", "text": "before"})
    transaction = ds.transaction
    changed, transactions = [], []

    class WriterAfterSelect:
        def __init__(self, conn):
            self.conn = conn

        def __getattr__(self, name):
            return getattr(self.conn, name)

        def execute(self, sql, params=()):
            cursor = self.conn.execute(sql, params)
            if sql.startswith("SELECT"):
                assert self.conn.in_transaction
                transactions.append(True)
                if not changed:
                    with sqlite3.connect(ds.path(tmp_path)) as writer:
                        writer.execute("UPDATE dialogue SET body=? WHERE stream='chat' AND seq=?",
                                       (json.dumps({"chat_id": 1, "task_id": "task", "text": "after"}), receipt.seq))
                    changed.append(True)
            return cursor

    @contextmanager
    def intercepted(*args, **kwargs):
        with transaction(*args, **kwargs) as conn:
            yield WriterAfterSelect(conn)

    monkeypatch.setattr(ds, "transaction", intercepted)
    found = ds.probe_rows(tmp_path, "chat", [("chat_id", 1), ("task_id", "task")])
    assert changed and len(transactions) >= 2
    assert [row.row["text"] for row in found] == ["before"]
    assert ds.row_at(tmp_path, "chat", receipt.seq).row["text"] == "after"


def test_public_readers_decode_after_releasing_the_snapshot(tmp_path, monkeypatch):
    seed(tmp_path, live={"chat": [{"chat_id": 1, "task_id": "task", "text": f"row {n}"}
                                for n in range(12)]})
    ds.ensure(tmp_path)
    transaction, decode = ds.transaction, ds.decode
    held, decoded = [], []
    @contextmanager
    def tracked(*args, **kwargs):
        with transaction(*args, **kwargs) as conn:
            held.append(True)
            try:
                yield conn
            finally:
                held.pop()
    def outside(body):
        assert not held, "materialization must not retain a SQLite reader lock"
        decoded.append(body)
        return decode(body)
    monkeypatch.setattr(ds, "transaction", tracked)
    monkeypatch.setattr(ds, "decode", outside)
    keys = [("chat_id", 1), ("task_id", "task")]
    readers = [lambda: ds.probe_rows(tmp_path, "chat", keys),
               lambda: ds.main_candidates(tmp_path, "chat", exclude_chats=[]),
               lambda: ds.task_rows(tmp_path, ["task"], stream="chat"),
               lambda: list(ds.iter_probes(tmp_path, "chat", keys, batch=5)),
               lambda: ds.newest(tmp_path, "chat", limit=12)]
    for read in readers:
        decoded.clear()
        rows = read()
        assert len(rows) == len(decoded) == 12
        assert sorted(row.row["text"] for row in rows) == sorted(f"row {n}" for n in range(12))


def test_a_value_compares_with_the_columns_affinity_as_a_bound_parameter_did():
    """``task_id = 123`` matched the stored ``'123'`` (TEXT affinity on the bound int) and ``chat_id = '5'``
    matched 5: a JSON list element carries no affinity of its own (``+value``), so batching keeps both."""
    with sqlite3.connect(":memory:") as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(ds._SCHEMA)
        for seq, chat, task, client in ((10, 5, "123", "42"), (20, 6, "abc", "x"), (30, None, "124", "43")):
            conn.execute("INSERT INTO dialogue(stream,seq,length,segment,ordinal,chat_id,task_id,client_message_id,body) "
                         "VALUES('chat',?,10,1,?,?,?,?,'{}')", (seq, seq, chat, task, client))
        for keys, expected in (([("task_id", 123)], [10]), ([("task_id", "123"), ("task_id", 124)], [30, 10]),
                               ([("client_message_id", 42)], [10]), ([("chat_id", "5")], [10]),
                               ([("chat_id", 6.0), ("root_task_id", 1)], [20]), ([("chat_id", None)], [30])):
            assert [row.seq for row in probe(conn, keys)] == expected, keys
