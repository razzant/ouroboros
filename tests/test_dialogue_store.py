"""The dialogue store (``ouroboros/dialogue_store.py``, ``ouroboros/dialogue_import.py``): the one
writer seam and its atomicity, addressed reads whose cost ignores other rooms, the cutover from the
chain files, the resumable archive import with visible coverage, legacy addresses, the rollback
export read back by today's chain reader, and concurrent writer processes."""
from __future__ import annotations

import json
import math
import multiprocessing
import os
import pathlib
import sqlite3

import pytest

from ouroboros import chat_chain, dialogue_import as di, dialogue_positions as dp, dialogue_store as ds, store
from ouroboros.utils import iter_jsonl_objects, jsonl_chain_handles, jsonl_generation_signature
from tests._dialogue import chat_rows, lines, progress_rows, seed


def chat(cmid="", text="hi", *, chat_id=1, direction="in", ts="2026-10-17T00:00:00+00:00", **extra):
    return {"ts": ts, "direction": direction, "chat_id": chat_id, "user_id": 1, "text": text,
            "client_message_id": cmid, **extra}


def progress(task_id, text="step", *, chat_id=1, ts="2026-10-17T00:00:00+00:00", **extra):
    return {"ts": ts, "type": "send_message", "task_id": task_id, "is_progress": True, "direction": "out",
            "chat_id": chat_id, "text": text, **extra}


def chain_offsets(root, stream):
    """The coordinates today's chain reader gives every row: cumulative bytes over the sorted archive + live."""
    offsets, position = {}, 0
    with jsonl_chain_handles(root / "logs" / f"{stream}.jsonl", strict=True) as handles:
        for _path, handle in handles:
            for raw in handle:
                try:
                    offsets[position] = json.loads(raw)
                except ValueError:
                    pass  # a blank or malformed line takes bytes, not a row
                position += len(raw)
    return offsets, position


# ---- a page of rows in one step ---------------------------------------------------------------

def _plain_rows(conn, where, params=()):
    """The readers' rows before one-step pages: a plain SELECT, one ``sqlite3_step`` (one GIL release) per row."""
    return [ds._row(record) for record in conn.execute(f"{ds._SELECT} {where}", params).fetchall()]


def _plain_probe(conn, keys, *, before, lower, limit, newest_first):
    """``_probe`` as it read before: each column's coordinates row by row, then the winners' rows."""
    found, order = set(), "DESC" if newest_first else "ASC"
    for column, value in keys:
        index = ds._PROBE_INDEX[column]
        test, args = ("IS NULL", ()) if value is None else ("=?", (value,))
        found.update(seq for (seq,) in conn.execute(
            f"SELECT seq FROM dialogue INDEXED BY {index} WHERE stream='chat' AND {column} {test} AND seq<? AND seq>=? "
            f"ORDER BY seq {order} LIMIT ?", (*args, before, lower, limit)))
    chosen = sorted(found, reverse=newest_first)[:limit]
    rows = {row.seq: row for row in _plain_rows(conn, f"WHERE stream='chat' AND seq IN ({','.join('?' * len(chosen))})",
                                                ("chat", *chosen)[1:])} if chosen else {}
    return [rows[seq] for seq in chosen]


def test_a_page_read_in_one_step_holds_the_rows_and_order_a_plain_select_returns(tmp_path):
    texts = ['plain', 'a "quoted" word', 'back\\slash \\n and a real\nnewline\tand a tab', 'кириллица и 🐍', '{"json": [1]}']
    rows = [chat(f"m{i}" if i % 3 else "", texts[i % len(texts)] + f" {i}", chat_id=1 + i % 4, direction="in" if i % 2 else "out",
                 ts=f"2026-10-17T{i // 3600:02d}:{i // 60 % 60:02d}:{i % 60:02d}+00:00",
                 **({"task_id": f"task{i % 7}"} if i % 5 == 0 else {})) for i in range(1_203)]
    seed(tmp_path, live={"chat": rows})
    from tests._dialogue import import_all
    import_all(tmp_path)  # one import, not a write transaction a row
    ds.append(tmp_path, "chat", chat("named", "owed", dispatch_marker=1), dispatch={"process": "p1"})
    ds.append(tmp_path, "chat", {"ts": "2026-10-18T00:00:00+00:00", "direction": "in", "text": "no chat id"})
    arrival = ds.append(tmp_path, "chat", chat("", "a guest", chat_id=4242, type="place_arrival"))
    with sqlite3.connect(ds.path(tmp_path)) as db:  # a place arrival's source pointer lands after its append
        db.execute("UPDATE dialogue SET dispatch_input=? WHERE stream='chat' AND seq=?",
                   (json.dumps({"place_context_source_ref": {"kind": "place_context", "id": "c1"}}), arrival.seq))
    with ds.transaction(tmp_path, write=False) as conn:
        every = _plain_rows(conn, "WHERE stream=? ORDER BY seq", ("chat",))
        probes = {(keys, newest): _plain_probe(conn, keys, before=every[-1].end, lower=every[4].seq, limit=60,
                                               newest_first=newest)
                  for keys in ((("chat_id", 2), ("task_id", "task3")), (("chat_id", None), ("chat_id", 4242)))
                  for newest in (True, False)}
    seqs = [row.seq for row in every]
    assert len(every) == 1_206 and every[-3].dispatch == "pending" and type(every[-1].row).__name__ == "SourceProjection"
    assert list(ds.iter_rows(tmp_path, "chat")) == every  # three pages of 500
    assert list(ds.iter_rows(tmp_path, "chat", lower=seqs[10], upper=seqs[700], batch=97)) == every[10:700]
    assert ds.rows_between(tmp_path, "chat", seqs[3], seqs[900]) == every[3:900]
    assert ds.rows_after(tmp_path, "chat", seqs[5], limit=600) == every[6:606]
    assert ds.rows_at(tmp_path, "chat", seqs[::2] + [seqs[-1] + 10**6]) == {row.seq: row for row in every[::2]}
    for (keys, newest), expected in probes.items():
        assert ds.probe_rows(tmp_path, "chat", keys, before=every[-1].end, lower=seqs[4], limit=60,
                             newest_first=newest) == expected and expected
    with ds.transaction(tmp_path, write=False) as conn:  # the legacy addresses and the counts, read row by row
        hints = {}
        for record in conn.execute("SELECT seq, gen, line FROM legacy_addresses ORDER BY seq, gen, line"):
            hints.setdefault(record["seq"], (record["gen"], record["line"]))
        pairs = sorted(set(hints.values()))[::3] + [("0" * 64, 1)]
        candidates = {pair: _plain_rows(conn, "WHERE stream='chat' AND seq IN (SELECT seq FROM legacy_addresses "
                                              "WHERE gen=? AND line=?) ORDER BY seq", pair) for pair in pairs}
        ordinals = {record["seq"]: dp.Ordinals(int(record["ordinal"]), record["pos"])
                    for record in conn.execute(f"{dp._ORDINALS} WHERE stream='chat'")}
    assert len(hints) == 1_203 and ds.legacy_addresses(tmp_path, seqs + [seqs[-1] + 10**6]) == hints
    assert ds.legacy_candidates_of(tmp_path, pairs) == candidates and candidates[pairs[0]] and not candidates[pairs[-1]]
    assert dp.ordinals_of(tmp_path, "chat", seqs + [seqs[-1] + 10**6]) == ordinals and len(ordinals) == 1_206


class _Stepped:
    """A read connection that counts the result rows its statements hand to Python: one ``sqlite3_step`` each."""

    def __init__(self, conn, stepped):
        self._conn, self._stepped = conn, stepped

    def execute(self, *args):
        rows = self._conn.execute(*args).fetchall()
        self._stepped.append(len(rows))
        return _Rows(rows)

    def __getattr__(self, name):
        return getattr(self._conn, name)


class _Rows(list):
    def fetchone(self):
        return self[0] if self else None

    def fetchall(self):
        return list(self)


def test_a_page_of_rows_reaches_python_in_one_step_whatever_its_length(tmp_path, monkeypatch):
    import contextlib

    from tests._dialogue import import_all
    seed(tmp_path, live={"chat": [chat(f"m{i}", f"row {i}", chat_id=1 + i % 3, task_id=f"t{i % 4}") for i in range(1_203)]})
    import_all(tmp_path)
    stepped, real = [], ds.transaction

    @contextlib.contextmanager
    def stepping(*args, **kwargs):
        with real(*args, **kwargs) as conn:
            yield _Stepped(conn, stepped) if not kwargs.get("write") else conn

    monkeypatch.setattr(ds, "transaction", stepping)
    every = list(ds.iter_rows(tmp_path, "chat"))
    assert len(every) == 1_203 and sum(stepped) == 3  # three pages of 500, one step each (1,203 steps per row)
    stepped.clear()
    assert len(ds.rows_at(tmp_path, "chat", [row.seq for row in every])) == 1_203 and sum(stepped) == 3
    stepped.clear()
    assert len(ds.probe_rows(tmp_path, "chat", [("chat_id", 2), ("task_id", "t1")], limit=200)) == 200
    assert sum(stepped) == 3  # each column's coordinates, then the winners' page
    stepped.clear()
    hints = ds.legacy_addresses(tmp_path, [row.seq for row in every])
    assert len(hints) == 1_203 and sum(stepped) == 3  # a step per 500 rows' addresses, not one per row
    stepped.clear()
    found = ds.legacy_candidates_of(tmp_path, hints.values())
    assert sum(map(len, found.values())) == 1_203 and sum(stepped) == 5 + 3  # 250 addresses a step, then 500 rows
    stepped.clear()
    assert len(dp.ordinals_of(tmp_path, "chat", [row.seq for row in every])) == 1_203 and sum(stepped) == 3


# ---- the writer seam ------------------------------------------------------------------------

def test_append_stores_the_exact_line_at_its_chain_coordinate(tmp_path):
    first = ds.append(tmp_path, "chat", chat("a", "первая"))
    second = ds.append(tmp_path, "chat", chat("", "ответ", direction="out"))
    assert (first.seq, first.duplicate, first.dispatch) == (0, False, "none")
    assert second.seq == len(json.dumps(chat("a", "первая"), ensure_ascii=False).encode("utf-8")) + 1
    assert second.dispatch is None, "an outbound row is not a named acceptance"
    assert lines(tmp_path) == [json.dumps(chat("a", "первая"), ensure_ascii=False), json.dumps(chat("", "ответ", direction="out"), ensure_ascii=False)]
    assert ds.stream_end(tmp_path, "chat") == second.seq + len(lines(tmp_path)[1].encode("utf-8")) + 1
    assert ds.row_at(tmp_path, "chat", first.seq).history_id == "chat:0"
    assert ds.header_tier(tmp_path) == store.TIER_ENFORCED
    assert store.header_tier(ds.path(tmp_path), store._APPLICATION_ID) is None, "its own lock protocol, not current.sqlite's"
    with pytest.raises(ValueError):
        ds.append(tmp_path, "events", {"x": 1})


def test_a_duplicate_client_message_returns_the_existing_receipt_and_distinct_ids_two_rows(tmp_path):
    first = ds.append(tmp_path, "chat", chat("cm-1", "слова"), dispatch={"text": "слова"})
    again = ds.append(tmp_path, "chat", chat("cm-1", "другие слова"), dispatch={"text": "другие слова"})
    other = ds.append(tmp_path, "chat", chat("cm-2", "слова"), dispatch={"text": "слова"})
    assert again.duplicate and (again.seq, again.row, again.dispatch) == (first.seq, first.row, "pending")
    assert not other.duplicate and other.seq != first.seq
    assert [row["client_message_id"] for row in chat_rows(tmp_path)] == ["cm-1", "cm-2"], "one row per id"
    assert ds.append(tmp_path, "chat", chat("cm-1", "x", chat_id=2)).seq not in (first.seq, other.seq), "another chat's id is its own"


def test_a_fault_between_the_statements_leaves_neither_row_nor_room_nor_obligation(tmp_path, monkeypatch):
    ds.append(tmp_path, "chat", chat("", "seed", direction="out"))
    before = (ds.stream_end(tmp_path, "chat"), ds.rooms(tmp_path), ds.counts(tmp_path))
    real = ds.touch_room

    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("disk I/O error between statements")

    monkeypatch.setattr(ds, "touch_room", fail)
    with pytest.raises(store.StoreError):
        ds.append(tmp_path, "chat", chat("cm-9", "lost"), dispatch={"text": "lost"}, feed=True)
    monkeypatch.setattr(ds, "touch_room", real)
    assert (ds.stream_end(tmp_path, "chat"), ds.rooms(tmp_path), ds.counts(tmp_path)) == before
    assert ds.accepted_row(tmp_path, 1, "cm-9") is None and ds.pending_dispatches(tmp_path) == []
    assert ds.dispatch_state(tmp_path, 1, "cm-9") is None
    receipt = ds.append(tmp_path, "chat", chat("cm-9", "lost"), dispatch={"text": "lost"})  # the retry lands whole
    assert not receipt.duplicate and ds.dispatch_state(tmp_path, 1, "cm-9") == "pending"


def test_the_unique_index_settles_the_race_the_lookup_missed(tmp_path, monkeypatch):
    first = ds.append(tmp_path, "chat", chat("race", "a"), dispatch={"text": "a"})
    real = ds._existing_named
    calls = []

    def blind_once(conn, chat_id, cmid):
        calls.append(cmid)
        return None if len(calls) == 1 else real(conn, chat_id, cmid)

    monkeypatch.setattr(ds, "_existing_named", blind_once)
    again = ds.append(tmp_path, "chat", chat("race", "a"), dispatch={"text": "a"})
    assert again.duplicate and again.seq == first.seq and len(chat_rows(tmp_path)) == 1


def test_rooms_follow_the_rows_in_the_same_transaction(tmp_path):
    ds.append(tmp_path, "chat", chat("", "a", direction="out", chat_id=7, ts="2026-10-17T00:00:01+00:00"), feed=True)
    second = ds.append(tmp_path, "chat", chat("", "b", direction="out", chat_id=7, ts="2026-10-17T00:00:02+00:00"), feed=False)
    ds.append(tmp_path, "progress", progress("t1", chat_id=7, ts="2026-10-17T00:00:03+00:00"))
    room = ds.room(tmp_path, 7)
    assert (room["revision"], room["last_seq"], room["activity_at"]) == (1, second.seq, "2026-10-17T00:00:03+00:00")
    assert ds.room(tmp_path, 8) is None and [row["chat_id"] for row in ds.rooms(tmp_path)] == [7]
    assert ds.mark_room_read(tmp_path, 7, 1)["read_revision"] == 1
    assert ds.mark_room_read(tmp_path, 7, 0)["read_revision"] == 1, "never backwards"
    assert ds.append(tmp_path, "chat", chat("", "no chat", direction="out", chat_id=None)).row["chat_id"] is None


def test_dispatch_obligation_moves_exactly_once_and_recovery_lists_only_the_pending(tmp_path):
    ds.append(tmp_path, "chat", chat("p", "owed"), dispatch={"text": "owed", "message": {"chat_id": 1}})
    ds.append(tmp_path, "chat", chat("n", "own dispatcher"))
    ds.append(tmp_path, "chat", chat("q", "also owed", chat_id=2), dispatch={"text": "also owed"})
    assert [(row.row["client_message_id"], payload["text"]) for row, payload in ds.pending_dispatches(tmp_path)] == [("p", "owed"), ("q", "also owed")]
    assert ds.claim_dispatch(tmp_path, 1, "p", process="host-1") and not ds.claim_dispatch(tmp_path, 1, "p"), "one claim"
    assert ds.dispatch_facts(tmp_path, 1, "p") == {"state": "dispatched", "process": "host-1", "dispatcher": ""}
    assert not ds.complete_dispatch(tmp_path, 1, "n") and ds.note_dispatched(tmp_path, 1, "n", process="host-1")
    assert ds.complete_dispatch(tmp_path, 1, "p") and ds.dispatch_state(tmp_path, 1, "p") == "completed"
    with ds.transaction(tmp_path, write=False) as conn:
        assert json.loads(conn.execute("SELECT dispatch_input FROM dialogue WHERE client_message_id='p'").fetchone()[0]) == {"dispatcher": None, "process": "host-1"}, "the input shrinks once handed over"
    assert ds.dispatch_facts(tmp_path, 1, "absent") is None and ds.dispatch_facts(tmp_path, 1, "") is None
    assert not ds.claim_dispatch(tmp_path, 1, "n"), "a row that owed no bus dispatch is never handed over"
    assert ds.abandon_dispatch(tmp_path, 2, "q", "inputs gone") and ds.dispatch_state(tmp_path, 2, "q") == "abandoned"
    assert ds.pending_dispatches(tmp_path) == []
    assert ds.dispatch_state(tmp_path, 1, "absent") is None


# ---- addressed reads -------------------------------------------------------------------------

def _room_fixture(root, foreign_rooms, rows_per_room=40):
    """Room 100 (20 own rows, 2 bound tasks) then ``foreign_rooms`` rooms' newer rows."""
    for n in range(20):
        ds.append(root, "chat", chat("", f"own {n}", direction="out", chat_id=100, ts=f"2026-10-17T00:00:{n:02d}+00:00"))
    ds.append(root, "chat", chat("", "task row", direction="out", chat_id=1, task_id="t-bound"))
    ds.append(root, "chat", chat("", "child row", direction="out", chat_id=1, task_id="t-child", parent_task_id="t-bound", root_task_id="t-bound"))
    ds.append(root, "chat", chat("origin-1", "owner", chat_id=1))
    ds.append(root, "chat", chat("", "started", direction="out", chat_id=1, origin_message_ref={"chat_id": 1, "client_message_id": "origin-1"}))
    for room in range(foreign_rooms):
        for n in range(rows_per_room):
            ds.append(root, "chat", chat("", f"foreign {room}-{n}", direction="out", chat_id=1000 + room))
            ds.append(root, "progress", progress(f"ft-{room}", chat_id=1000 + room))


def test_room_candidates_are_the_rooms_own_rows_its_tasks_and_its_origins(tmp_path):
    _room_fixture(tmp_path, 3)
    page = ds.room_candidates(tmp_path, "chat", 100, tasks=["t-bound"], origins=["1:origin-1"], limit=10)
    texts = [row.row["text"] for row in page]
    assert texts[:3] == ["started", "child row", "task row"] and len(page) == 10 and texts[3] == "own 19"
    older = ds.room_candidates(tmp_path, "chat", 100, before=page[-1].seq, limit=10)
    assert [row.row["text"] for row in older] == [f"own {n}" for n in range(12, 2, -1)]
    assert not any("foreign" in row.row["text"] for row in page + older)
    assert [row.row["text"] for row in ds.task_rows(tmp_path, ["t-bound"], stream="chat")] == ["task row", "child row"]
    main = ds.main_candidates(tmp_path, "chat", exclude_chats={100, 1000, 1001, 1002}, limit=5)
    assert [row.row["text"] for row in main] == ["started", "owner", "child row", "task row"]
    replay = ds.rows_between(tmp_path, "chat", page[-1].seq, page[0].seq + 1)
    assert [row.seq for row in replay] == sorted(row.seq for row in replay) and {row.seq for row in page} < {row.seq for row in replay}
    assert [row.row["text"] for row in ds.rows_after(tmp_path, "chat", page[0].seq, limit=2)] == ["foreign 0-0", "foreign 0-1"]
    assert ds.newest(tmp_path, "chat", limit=1)[0].row["text"] == "foreign 2-39"
    assert [row.seq for row in ds.iter_rows(tmp_path, "chat", batch=7)] == [row.seq for row in ds.newest(tmp_path, "chat", limit=10_000)]
    assert [row.row["text"] for row in ds.iter_rows(tmp_path, "chat", lower=page[-1].seq, upper=page[0].seq + 1, batch=3)] == [row.row["text"] for row in replay]
    assert ds.last_inbound(tmp_path, 1).row["text"] == "owner" and ds.last_inbound(tmp_path, 100) is None
    assert ds.last_inbound(tmp_path, 1, before=ds.last_inbound(tmp_path, 1).seq) is None
    with pytest.raises(ValueError):
        ds.probe_rows(tmp_path, "chat", [("body", "x")])


def _metered(monkeypatch):
    """Count statements and VM steps on every connection the store opens from now on."""
    counts = {"statements": 0, "steps": 0}
    real = store.connect

    def connect(*args, **kwargs):
        conn = real(*args, **kwargs)
        conn.set_trace_callback(lambda _statement: counts.__setitem__("statements", counts["statements"] + 1))
        conn.set_progress_handler(lambda: counts.__setitem__("steps", counts["steps"] + 1) or 0, 1)
        return conn

    store.close_connections()
    monkeypatch.setattr(store, "connect", connect)
    return counts


@pytest.mark.parametrize("read", [
    lambda root: ds.room_candidates(root, "chat", 100, tasks=["t-bound"], origins=["1:origin-1"], limit=20),
    lambda root: ds.task_rows(root, ["t-bound"], stream="chat"),
    lambda root: ds.accepted_row(root, 1, "origin-1"),
    lambda root: ds.main_candidates(root, "chat", exclude_chats=range(1000, 1030), limit=20),
])
def test_a_room_page_costs_the_same_with_one_and_thirty_other_rooms(tmp_path, monkeypatch, read):
    results = {}
    for k in (1, 30):
        root = tmp_path / f"k{k}"
        _room_fixture(root, k)
        store.close_connections()
        counts = _metered(monkeypatch)
        warm = read(root)
        counts.update(statements=0, steps=0)
        assert read(root) == warm
        results[k] = dict(counts)
        monkeypatch.undo()
    assert results[1]["statements"] == results[30]["statements"] > 0
    assert results[30]["steps"] <= results[1]["steps"] + 128 * math.ceil(math.log2(30)), results


def test_an_absent_store_answers_empty_for_a_root_without_dialogue_and_cuts_over_one_with_chain_files(tmp_path):
    assert ds.rooms(tmp_path) == [] and ds.newest(tmp_path, "chat", limit=5) == [] and ds.stream_end(tmp_path, "chat") == 0
    assert not ds.present(tmp_path), "a root without dialogue gets no store from a read"
    seed(tmp_path, live={"chat": [chat("", "old body wrote", direction="out")]})
    assert di.coverage(tmp_path, "chat")["imported"] is False, "files nobody cut over are not 'no rows'"
    assert [row.row["text"] for row in ds.newest(tmp_path, "chat", limit=5)] == ["old body wrote"], "a read cuts the root over: its own files"
    assert ds.present(tmp_path) and (tmp_path / "logs" / "chat.jsonl").stat().st_size == 0
    other = tmp_path / "child"
    seed(other, live={"progress": [progress("t1", "child step")]})
    assert ds.append(other, "chat", chat("", "first", direction="out")).seq == 0, "a write cuts the root over first"
    assert [row["text"] for row in progress_rows(other)] == ["child step"] and [row["text"] for row in chat_rows(other)] == ["first"]
    assert ds.header_tier(tmp_path / "nowhere") is None and ds.room(tmp_path / "nowhere", 1) is None


# ---- cutover, import, coverage -----------------------------------------------------------------

def _legacy_root(root, *, chat_segments=3, progress_segments=2, rows_per_segment=5):
    archives = {}
    for index in range(chat_segments):
        archives[f"chat_2026100{index + 1}T000000.jsonl"] = [
            chat(f"c{index}-{n}", f"сообщение {index}-{n}", chat_id=1 + (n % 2), ts=f"2026-10-0{index + 1}T00:00:{n:02d}+00:00")
            if n % 2 else chat("", f"ответ {index}-{n}", direction="out", ts=f"2026-10-0{index + 1}T00:00:{n:02d}+00:00")
            for n in range(rows_per_segment)]
    archives["chat_20261001T000000.jsonl"].insert(2, "{not json\n")
    archives["chat_20261001T000000.jsonl"].append("\n")
    for index in range(progress_segments):
        archives[f"progress_2026100{index + 1}T000000.jsonl"] = [progress(f"t{index}", f"шаг {n}") for n in range(rows_per_segment)]
    live = {"chat": [chat("live-1", "живая"), chat("", "живой ответ", direction="out")],
            "progress": [progress("t9", "живой шаг")]}
    seed(root, archives=archives, live=live)
    return archives, live


def test_cutover_rotates_the_live_files_imports_them_and_keeps_every_chain_coordinate(tmp_path):
    _legacy_root(tmp_path)
    expected = {stream: chain_offsets(tmp_path, stream) for stream in ds.STREAMS}
    report = di.cutover(tmp_path)
    assert report["created"] and report["first"]
    for stream in ds.STREAMS:
        assert (tmp_path / "logs" / f"{stream}.jsonl").stat().st_size == 0, "rotated into the archive one last time"
        assert report[stream]["rotated"].startswith(f"{stream}_") and report[stream]["imported"]["status"] == "done"
        offsets, total = expected[stream]
        assert ds.stream_end(tmp_path, stream) == total, "the next row continues the chain"
        imported = {row.seq: row.row for row in ds.newest(tmp_path, stream, limit=1000)}
        for seq, row in imported.items():
            assert offsets[seq] == row, "an imported row sits at its chain offset"
    assert [row["text"] for row in chat_rows(tmp_path)] == ["живая", "живой ответ"], "only the final live segment so far"
    assert ds.accepted_row(tmp_path, 1, "live-1").row["text"] == "живая", "recent ids dedup before the first message"
    assert di.coverage(tmp_path, "chat")["pending"] == 3 and not di.coverage(tmp_path, "chat")["imported"]
    end = ds.stream_end(tmp_path, "chat")
    assert di.coverage(tmp_path, "chat", end)["imported"] is False and di.coverage(tmp_path, "chat", 0)["imported"]
    again = di.cutover(tmp_path)  # a steady boot: nothing another body wrote, nothing planned again
    assert not again["created"] and not again["first"] and again["replanned"] is False and "chat" not in again
    assert ds.meta(tmp_path, "cutover")["chat"]["cutover_seq"] == end


def test_a_cutover_that_failed_after_create_is_finished_before_any_row_takes_a_coordinate(tmp_path, monkeypatch):
    """The boot's first cutover can fail after ``create`` published the store (a held chain append lock,
    Windows refusing to rename an open file) while the dialogue still opens: readers then say importing,
    and the first stored row finishes the cutover, so it never takes a coordinate inside the archive's
    space nor stores the live file's message a second time."""
    seed(tmp_path, archives={"chat_20261001T000000.jsonl": [
        chat(f"a{n}", f"archive {n}", ts=f"2026-10-01T00:00:{n:02d}+00:00") for n in range(3)]},
        live={"chat": [chat("live-1", "live before the upgrade", ts="2026-10-03T00:00:00+00:00")]})
    rotate, calls = di.rotate_final, []

    def busy_once(root, stream):
        calls.append(stream)
        if len(calls) == 1:
            raise ds.StoreBusy("the append lock of logs/chat.jsonl is held", reason="contention")
        return rotate(root, stream)

    monkeypatch.setattr(di, "rotate_final", busy_once)
    with pytest.raises(ds.StoreBusy):
        di.cutover(tmp_path)
    assert ds.present(tmp_path) and ds.meta(tmp_path, "cutover") is None
    assert di.coverage(tmp_path, "chat") == {"imported": False, "pending": 1, "mismatch": 0, "gaps": 0, "unplanned": True}
    assert ds.range_coverage(tmp_path, "chat")["imported"] is False
    with pytest.raises(ds.StoreMigrating):
        ds.require_imported(tmp_path, "chat")
    new = ds.append(tmp_path, "chat", chat("new-1", "accepted after the failed cutover", ts="2026-10-17T10:00:00+00:00"),
                    dispatch={"text": "x"})
    assert ds.meta(tmp_path, "cutover") is not None, "the first stored row finished the cutover"
    redelivered = ds.append(tmp_path, "chat", chat("live-1", "live before the upgrade", ts="2026-10-03T00:00:00+00:00"),
                            dispatch={"text": "x"})
    assert redelivered.duplicate and redelivered.seq < new.seq, "the live file's message was stored by the cutover"
    di.run_dialogue_import(tmp_path)
    assert [row.row["text"] for row in ds.newest(tmp_path, "chat", limit=100)] == [
        "archive 0", "archive 1", "archive 2", "live before the upgrade", "accepted after the failed cutover"]
    assert di.coverage(tmp_path, "chat")["imported"] and "unplanned" not in ds.range_coverage(tmp_path, "chat")


def test_the_archive_import_resumes_after_an_interruption_and_imports_every_row_once(tmp_path, monkeypatch):
    archives, live = _legacy_root(tmp_path)
    expected = {stream: chain_offsets(tmp_path, stream) for stream in ds.STREAMS}
    di.cutover(tmp_path)
    seen = []
    stops = iter([False, False, False, True])  # the pre-run check, then one per segment
    first = di.run_dialogue_import(tmp_path, progress=seen.append, should_stop=lambda: next(stops, False))
    assert first["state"] == "pending" and first["done"] == 2 and first["cursor"].startswith("chat:")
    oldest = tmp_path / "archive" / "chat_20261001T000000.jsonl"
    assert ds.legacy_generation(tmp_path, jsonl_generation_signature(oldest)["first_line_sha256"]) is None, "not imported yet: unknown, not gone"
    assert [row["name"] for row in di.pending_segments(tmp_path)][:1] == ["chat_20261001T000000.jsonl"], "chat first, newest first"
    reads = []
    real = pathlib.Path.read_bytes
    monkeypatch.setattr(pathlib.Path, "read_bytes", lambda self: reads.append(self.name) or real(self))
    second = di.run_dialogue_import(tmp_path)
    assert second["state"] == "done" and second["done"] == 5 == second["total"]
    assert sorted(reads) == sorted(["chat_20261001T000000.jsonl", "progress_20261001T000000.jsonl", "progress_20261002T000000.jsonl"]), "done segments are never read again"
    reads.clear()
    assert di.run_dialogue_import(tmp_path)["state"] == "done" and reads == [], "the rest law"
    for stream in ds.STREAMS:
        offsets, total = expected[stream]
        assert {row.seq: row.row for row in ds.newest(tmp_path, stream, limit=1000)} == offsets, "every row once, at its coordinate"
        assert di.coverage(tmp_path, stream)["imported"] and di.import_status(tmp_path)[stream]["pending"] == 0
    assert di.import_status(tmp_path)["chat"]["rows"] == len(expected["chat"][0])
    with ds.transaction(tmp_path, write=False) as conn:
        bad = conn.execute("SELECT bad FROM import_ledger WHERE name='chat_20261001T000000.jsonl'").fetchone()["bad"]
    assert bad == 1, "the malformed line is counted, the blank one is not"
    assert ds.room(tmp_path, 2)["last_seq"] == max(seq for seq, row in expected["chat"][0].items() if row.get("chat_id") == 2)
    assert ds.room(tmp_path, 2)["revision"] == 0, "imported rows do not count as unread"


def test_a_miss_during_the_import_is_checked_against_the_pending_segments(tmp_path, monkeypatch):
    _legacy_root(tmp_path)
    di.cutover(tmp_path)
    assert ds.accepted_row(tmp_path, 2, "c0-1").row["text"] == "сообщение 0-1", "found in a segment not yet imported"
    assert ds.accepted_row(tmp_path, 1, "c0-1") is None and ds.accepted_row(tmp_path, 2, "absent") is None
    di.run_dialogue_import(tmp_path)
    monkeypatch.setattr(di, "scan_pending", lambda *a, **k: pytest.fail("no scan once the import is complete"))
    assert ds.accepted_row(tmp_path, 2, "c0-1").row["text"] == "сообщение 0-1"
    assert ds.accepted_row(tmp_path, 1, "absent") is None


def test_legacy_gen_line_addresses_resolve_to_the_row_the_chain_reader_returned(tmp_path):
    _legacy_root(tmp_path)
    di.cutover(tmp_path)
    di.run_dialogue_import(tmp_path)
    for path in sorted((tmp_path / "archive").glob("chat_*.jsonl")):
        gen = jsonl_generation_signature(path)["first_line_sha256"]
        for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                row = json.loads(raw)
            except ValueError:
                continue
            found, resolution = chat_chain.resolve_row(tmp_path, chat_chain.row_address(row, gen=gen, line=line_number))
            assert resolution["status"] == "ok" and (resolution["gen"], resolution["line"]) == (gen, line_number)
            stored = ds.resolve_legacy(tmp_path, gen, line_number)
            assert stored.row == found == row
            assert ds.legacy_address(tmp_path, stored.seq) == (gen, line_number)
            assert row in [entry.row for entry in ds.rows_by_identity(tmp_path, row["chat_id"], row["ts"])]
        generation = ds.legacy_generation(tmp_path, gen)
        assert (generation["name"], generation["size"]) == (path.name, path.stat().st_size)
        offset = len(path.read_text(encoding="utf-8").splitlines(keepends=True)[0].encode("utf-8"))
        assert ds.row_at(tmp_path, "chat", generation["base"] + offset).row == json.loads(path.read_text(encoding="utf-8").splitlines()[1]), "a legacy (gen, byte offset) cursor is base + offset"
    assert ds.resolve_legacy(tmp_path, "nope", 1) is None and ds.resolve_legacy(tmp_path, gen, 0) is None
    assert ds.legacy_generation(tmp_path, "nope") is None and ds.legacy_generation(tmp_path, "") is None


# ---- the rollback export -----------------------------------------------------------------------

def test_the_export_appends_the_post_cutover_rows_byte_identical_and_a_re_cutover_takes_them_back(tmp_path):
    _legacy_root(tmp_path)
    di.cutover(tmp_path)
    cut = ds.stream_end(tmp_path, "chat")
    new_rows = [chat("after-1", "после"), chat("", "ответ после", direction="out"), chat("after-2", "ещё", chat_id=2)]
    for row in new_rows:
        ds.append(tmp_path, "chat", row, dispatch={"text": row["text"]} if row["client_message_id"] else None)
    ds.append(tmp_path, "progress", progress("t10", "после"))
    report = di.export_for_older_body(tmp_path)
    assert report[str(tmp_path)]["chat"] == {"rows": 3, "from": cut, "to": ds.stream_end(tmp_path, "chat"), "bytes": sum(len(ds.encode(r).encode("utf-8")) + 1 for r in new_rows)}
    live = tmp_path / "logs" / "chat.jsonl"
    assert live.read_bytes() == b"".join(ds.encode(row).encode("utf-8") + b"\n" for row in new_rows)
    assert list(iter_jsonl_objects(live)) == new_rows, "today's reader returns the rows unchanged"
    offsets, total = chain_offsets(tmp_path, "chat")
    assert total == ds.stream_end(tmp_path, "chat") and all(ds.row_at(tmp_path, "chat", seq).row == row for seq, row in offsets.items() if seq >= cut)
    assert di.export_for_older_body(tmp_path)[str(tmp_path)]["chat"]["rows"] == 0, "nothing exported twice"
    # The older body ran and appended; the newer body boots again: its rows are imported, none duplicated.
    seed(tmp_path, live={"chat": [chat("old-body", "от старого тела")]})
    report = di.cutover(tmp_path)
    assert report["chat"]["rotated"] and report["chat"]["imported"]["rows"] == 4
    assert [row["text"] for row in chat_rows(tmp_path)][-4:] == ["после", "ответ после", "ещё", "от старого тела"]
    assert ds.accepted_row(tmp_path, 1, "old-body") is not None and ds.dispatch_state(tmp_path, 1, "after-1") == "pending"
    assert ds.stream_end(tmp_path, "chat") == chain_offsets(tmp_path, "chat")[1]
    assert di.export_for_older_body(tmp_path)[str(tmp_path)]["chat"]["rows"] == 0, "the files hold everything again"


def test_the_export_appends_through_a_binary_descriptor(tmp_path, monkeypatch):
    """Windows opens a descriptor in text mode unless ``O_BINARY`` is set, and ``os.write`` then
    turns each LF into CRLF: the exported lines would no longer be the stored rows at their
    coordinates. The export's open carries ``O_BINARY`` wherever the platform has it; a sentinel
    stands in for it here, so the flag is checked on every platform."""
    sentinel, real_binary, real_open, flags = 1 << 40, getattr(os, "O_BINARY", 0), os.open, []
    di.cutover(tmp_path)
    ds.append(tmp_path, "chat", chat("after-1", "после"))
    live = tmp_path / "logs" / "chat.jsonl"

    def recording_open(path, mode, *args, **kwargs):
        if pathlib.Path(path) == live:
            flags.append(mode)
            mode = (mode & ~sentinel) | (real_binary if mode & sentinel else 0)
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(os, "O_BINARY", sentinel, raising=False)
    monkeypatch.setattr(os, "open", recording_open)
    assert di.export_for_older_body(tmp_path)[str(tmp_path)]["chat"]["rows"] == 1
    assert flags and all(mode & sentinel for mode in flags), f"the export opened {live.name} without O_BINARY"


@pytest.mark.parametrize("written", ["whole", "part", "none", "absent", "foreign"])
def test_an_export_interrupted_before_its_coordinate_moved_finishes_the_batch_once(tmp_path, monkeypatch, written):
    """A death after the batch reached the file and before ``files_seq`` moved: the retry finishes
    the recorded batch (appends only what is missing) instead of appending it a second time; bytes
    in its place that are not the batch's refuse the export and append nothing."""
    di.cutover(tmp_path)
    new_rows = [chat("after-1", "после"), chat("", "ответ после", direction="out")]
    for row in new_rows:
        ds.append(tmp_path, "chat", row)
    batch = b"".join(ds.encode(row).encode("utf-8") + b"\n" for row in new_rows)
    live = tmp_path / "logs" / "chat.jsonl"
    real_now = di.utc_now_iso

    def died(*_args):
        raise SystemExit("the process died before the export's coordinate moved")

    monkeypatch.setattr(di, "utc_now_iso", died)
    with pytest.raises(SystemExit):
        di.export_for_older_body(tmp_path)
    monkeypatch.setattr(di, "utc_now_iso", real_now)
    assert live.read_bytes() == batch and ds.meta(tmp_path, "export_batch:chat")["to"] == ds.stream_end(tmp_path, "chat")
    if written == "part":
        live.write_bytes(batch[:len(batch) // 2])  # the write itself was cut short
    elif written == "none":
        live.write_bytes(b"")
    elif written == "absent":  # the death came after the record and before the file existed
        live.unlink()
    elif written == "foreign":
        live.write_bytes(b'{"text": "not the batch"}\n')
        with pytest.raises(ds.StoreError, match="changed under an interrupted export"):
            di.export_for_older_body(tmp_path)
        assert live.read_bytes() == b'{"text": "not the batch"}\n', "nothing appended over foreign bytes"
        return
    ds.append(tmp_path, "chat", chat("after-2", "ещё"))  # a row after the batch: exported after it, once
    report = di.export_for_older_body(tmp_path)[str(tmp_path)]["chat"]
    assert report["rows"] == 3 and report["to"] == ds.stream_end(tmp_path, "chat")
    assert live.read_bytes() == batch + ds.encode(chat("after-2", "ещё")).encode("utf-8") + b"\n"
    assert ds.meta(tmp_path, "export_batch:chat") is None
    assert di.export_for_older_body(tmp_path)[str(tmp_path)]["chat"]["rows"] == 0, "nothing exported twice"


def test_a_torn_export_that_a_boot_cutover_took_in_never_refuses_later_exports(tmp_path, monkeypatch):
    """An export cut inside a row, then a boot whose cutover took the live file in (its whole rows
    imported, the cut line a skipped interval) before the seam retried: the batch record is stale,
    and later exports start afresh from the files' coordinate instead of refusing every time. The
    row the write cut stays a skipped interval in the files (it is in the store)."""
    di.cutover(tmp_path)
    rows = [chat("after-1", "после"), chat("", "ответ после", direction="out"), chat("after-2", "ещё")]
    for row in rows:
        ds.append(tmp_path, "chat", row)
    encoded = [ds.encode(row).encode("utf-8") + b"\n" for row in rows]
    live = tmp_path / "logs" / "chat.jsonl"
    real_now = di.utc_now_iso
    monkeypatch.setattr(di, "utc_now_iso", lambda *_a: (_ for _ in ()).throw(SystemExit("died")))
    with pytest.raises(SystemExit):
        di.export_for_older_body(tmp_path)
    monkeypatch.setattr(di, "utc_now_iso", real_now)
    live.write_bytes(b"".join(encoded)[:len(encoded[0]) + 7])  # the kill came inside the second row
    report = di.cutover(tmp_path)  # the next boot plans the non-empty live file again
    assert report["chat"]["rotated"] and ds.range_coverage(tmp_path, "chat")["gaps"] == 1
    assert ds.meta(tmp_path, "export_batch:chat") is None, "the rotation settled the batch it held"
    exported = di.export_for_older_body(tmp_path)[str(tmp_path)]["chat"]
    assert exported["rows"] == 1 and live.read_bytes() == encoded[2], "the rows after the files' coordinate, once"
    assert ds.meta(tmp_path, "export_batch:chat") is None
    assert di.export_for_older_body(tmp_path)[str(tmp_path)]["chat"]["rows"] == 0


@pytest.mark.parametrize("interruption", ["none", "after_the_rename", "final_transaction"])
def test_a_cutover_that_rotates_an_exported_prefix_settles_the_unwritten_batch(tmp_path, monkeypatch, interruption):
    """A batch recorded but never written (the append itself failed), then a boot whose cutover rotated
    the live file holding only rows exported before: the rotation settles the record, and the retry
    exports the batch's rows once into the new live file instead of refusing every time. A cutover
    stopped after its rename leaves the record to the next run, which rotates nothing and still settles
    it (the live file is shorter than the batch's offset)."""
    import os

    di.cutover(tmp_path)
    ds.append(tmp_path, "chat", chat("after-1", "после"))
    assert di.export_for_older_body(tmp_path)[str(tmp_path)]["chat"]["rows"] == 1
    ds.append(tmp_path, "chat", chat("after-2", "ещё"))
    live = tmp_path / "logs" / "chat.jsonl"
    real_open = os.open

    def refused(path, flags, *args):
        if str(path) == str(live) and flags & os.O_APPEND:
            raise OSError(28, "No space left on device")
        return real_open(path, flags, *args)

    monkeypatch.setattr(os, "open", refused)
    with pytest.raises(OSError):
        di.export_for_older_body(tmp_path)
    monkeypatch.setattr(os, "open", real_open)
    assert ds.meta(tmp_path, "export_batch:chat") is not None
    exported_before = live.read_bytes()
    if interruption != "none":
        def stopped(*_args):
            raise OSError(5, f"the cutover stopped: {interruption}")

        with monkeypatch.context() as patch:
            patch.setattr(di, "_plan" if interruption == "after_the_rename" else "_archive_tails", stopped)
            with pytest.raises(OSError):
                di.cutover(tmp_path)
        assert live.read_bytes() == b"" and ds.meta(tmp_path, "export_batch:chat") is not None
        store.close_connections()
    report = di.cutover(tmp_path)  # the next boot rotates the live file of exported rows
    assert bool(report["chat"]["rotated"]) == (interruption == "none")
    assert ds.meta(tmp_path, "export_batch:chat") is None
    assert di.export_for_older_body(tmp_path)[str(tmp_path)]["chat"]["rows"] == 1
    assert live.read_bytes() == ds.encode(chat("after-2", "ещё")).encode("utf-8") + b"\n"
    archived = b"".join(path.read_bytes() for path in sorted((tmp_path / "archive").glob("chat_*.jsonl")))
    assert archived.endswith(exported_before) and di.export_for_older_body(tmp_path)[str(tmp_path)]["chat"]["rows"] == 0


# ---- concurrent writer processes ---------------------------------------------------------------

def _spawned_writer(root, index, count, done):
    try:
        from ouroboros import dialogue_store as module

        for n in range(count):
            module.append(pathlib.Path(root), "progress", progress(f"w{index}", f"writer {index} row {n}", ts=f"2026-10-17T00:{index:02d}:{n % 60:02d}+00:00"))
        done.put((index, None))
    except BaseException as exc:  # pragma: no cover - surfaced in the parent
        done.put((index, repr(exc)))


@pytest.mark.serial
def test_four_writer_processes_lose_and_duplicate_nothing(tmp_path):
    ds.append(tmp_path, "progress", progress("seed", "seed"))
    ctx = multiprocessing.get_context("spawn")
    done = ctx.Queue()
    writers = [ctx.Process(target=_spawned_writer, args=(str(tmp_path), index, 30, done)) for index in range(4)]
    for process in writers:
        process.start()
    outcomes = []
    try:
        for _ in writers:
            outcomes.append(done.get(timeout=120))
    finally:
        for process in writers:
            process.join(30)
            if process.is_alive():
                process.kill()
    assert [failure for _index, failure in outcomes] == [None] * 4, outcomes
    stored = ds.newest(tmp_path, "progress", limit=10_000)
    texts = sorted(row.row["text"] for row in stored if row.row["task_id"] != "seed")
    assert texts == sorted(f"writer {index} row {n}" for index in range(4) for n in range(30)), "no lost or doubled row"
    seqs = [row.seq for row in stored]
    assert len(set(seqs)) == len(seqs) == 121
    for previous, following in zip(stored, stored[1:]):
        assert following.seq == previous.seq + len(ds.encode(previous.row).encode("utf-8")) + 1, "the chain coordinates stay contiguous"
    assert ds.stream_end(tmp_path, "progress") == stored[-1].seq + len(ds.encode(stored[-1].row).encode("utf-8")) + 1
