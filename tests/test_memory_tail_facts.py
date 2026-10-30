"""The chat rows after the legacy frontier keep their facts beside the chronicle (``memory_inventory._tail``).

Every reader decoded, hashed and projected every row after the frontier on every call (about 22,000 rows
for each Main view on the owner's install). The chronicle index keeps row addresses, metadata and
positions (``tail_rows``), invalidating them on a text revision; a reader otherwise reads only new rows and stores
them, and the maintenance job fills the history once, so no turn pays it. Membership and seals stay
applied at read. Each test holds the stored answer equal to the rows read live.
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sqlite3

from ouroboros import chat_chain as cc, dialogue_store as ds, maintenance, memory_inventory as mi, store
from ouroboros.chronicle_store import ChronicleStore
from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
from tests import _memory_inventory_shared as shared

ROW_META = mi.row_meta
LIGHT = {"kind": "helper", "route": "configured-light"}


def _live(root):
    """The reference: every row after the frontier read and projected now, as every call did before."""
    lower = cc.position_boundary(root, mi._active_frontier(ChronicleStore(root))["pos"])
    return [(address, ROW_META(row), pos) for address, row, pos in cc.iter_rows(root, lower=lower)]


def _projected(monkeypatch):
    """Rows projected (decoded into their facts) from now on."""
    counted = []
    monkeypatch.setattr(mi, "row_meta", lambda row: counted.append(1) or ROW_META(row))
    return counted


def _fill(root):
    return mi.fill_tail(root, lambda: False)


def _stored(root):
    """The stored state and rows, read straight from the index file."""
    with sqlite3.connect(ChronicleStore(root).index_path) as db:
        return ChronicleStore(root).tail_state(), db.execute("SELECT * FROM tail_rows ORDER BY pos").fetchall()


def test_rows_are_projected_once_and_a_reader_projects_only_the_rows_appended_since(tmp_path, monkeypatch):
    shared.world(tmp_path)
    reference = _live(tmp_path)
    projected = _projected(monkeypatch)
    assert mi.rows_after_frontier(tmp_path) == reference and len(projected) == 10  # nothing stored yet: read live
    assert _stored(tmp_path) == (None, []) and mi.tail_owed(tmp_path)  # a reader never fills the history
    assert _fill(tmp_path) == {"state": "done", "items": 10} and not mi.tail_owed(tmp_path)
    projected.clear()
    assert mi.rows_after_frontier(tmp_path) == reference and projected == []
    ds.append(tmp_path, "chat", shared.msg("2026-09-04T00:00:00+00:00", "new", client_message_id="n1"))
    assert mi.rows_after_frontier(tmp_path) == _live(tmp_path) and len(projected) == 1  # the row appended since
    projected.clear()
    assert mi.rows_after_frontier(tmp_path) == _live(tmp_path) and projected == []  # stored by that reader


def test_a_fill_stopped_midway_resumes_at_its_stored_prefix_and_readers_leave_the_rest_to_it(tmp_path, monkeypatch):
    shared.world(tmp_path)
    monkeypatch.setattr(mi, "_FILL_ROWS", 4)
    assert mi.fill_tail(tmp_path, lambda: True) == {"state": "pending", "items": 4}  # stopped after one transaction
    state, rows = _stored(tmp_path)
    assert not state["complete"] and [row[0] for row in rows] == [10, 11, 12, 13] and mi.tail_owed(tmp_path)
    projected = _projected(monkeypatch)
    assert mi.rows_after_frontier(tmp_path) == _live(tmp_path) and len(projected) == 6  # the prefix, then live
    assert _stored(tmp_path) == (state, rows)  # an unfinished fill is the job's: a reader stores nothing
    assert _fill(tmp_path) == {"state": "done", "items": 6} and not mi.tail_owed(tmp_path)
    projected.clear()
    assert mi.rows_after_frontier(tmp_path) == _live(tmp_path) and projected == []


def test_rows_appended_while_the_lock_is_held_appear_and_the_next_reader_stores_them(tmp_path, monkeypatch):
    shared.world(tmp_path)
    _fill(tmp_path)
    ds.append(tmp_path, "chat", shared.msg("2026-09-04T00:00:00+00:00", "new", client_message_id="n1"))
    lock, projected = ChronicleStore(tmp_path).lock_path, _projected(monkeypatch)
    fd = acquire_exclusive_file_lock(lock, timeout_sec=0)  # a publisher holds the publication lock
    try:
        assert mi.rows_after_frontier(tmp_path) == _live(tmp_path) and len(projected) == 1  # one zero-wait try
    finally:
        release_exclusive_file_lock(lock, fd)
    assert mi.rows_after_frontier(tmp_path) == _live(tmp_path) and len(projected) == 2
    assert mi.rows_after_frontier(tmp_path) == _live(tmp_path) and len(projected) == 2


def test_a_replaced_dialogue_store_or_a_moved_frontier_is_never_answered_from_the_stored_facts(tmp_path, monkeypatch):
    shared.world(tmp_path)
    _fill(tmp_path)
    path = ds.path(tmp_path)
    store.close_connections_under(tmp_path)
    shutil.copy2(path, tmp_path / "copy.sqlite")
    os.replace(tmp_path / "copy.sqlite", path)  # a restore: the same rows in a new file
    projected = _projected(monkeypatch)
    assert mi.rows_after_frontier(tmp_path) == _live(tmp_path) and len(projected) == 10 and mi.tail_owed(tmp_path)
    assert _fill(tmp_path)["items"] == 10
    frontier = mi._active_frontier(ChronicleStore(tmp_path))
    assert ChronicleStore(tmp_path).publish([], scan_state={"legacy_frontier": {**frontier, "pos": 15}}).ok
    projected.clear()
    assert [pos for _a, _m, pos in mi.rows_after_frontier(tmp_path)] == list(range(15, 20)) and len(projected) == 5
    assert mi.tail_owed(tmp_path) and _fill(tmp_path)["items"] == 5
    projected.clear()
    assert mi.rows_after_frontier(tmp_path) == _live(tmp_path) and projected == []


def test_a_binding_made_after_the_facts_were_stored_moves_their_rows_and_writes_none(tmp_path, monkeypatch):
    from ouroboros.projects_registry import bind_task_to_project

    beta = str(shared.world(tmp_path)["beta"])
    _fill(tmp_path)
    before, projected = _stored(tmp_path), _projected(monkeypatch)
    rooms = mi.open_rows_by_room(tmp_path)
    assert [pos for _a, _m, pos in rooms[beta]] == [14] and {12, 16} <= {pos for _a, _m, pos in rooms["1"]}
    bind_task_to_project(tmp_path, "t2", "beta", origin={"absent": "post_hoc_unresolved"})
    rooms = mi.open_rows_by_room(tmp_path)
    assert [pos for _a, _m, pos in rooms[beta]] == [12, 14, 16] and not {12, 16} & {pos for _a, _m, pos in rooms["1"]}
    assert projected == [] and _stored(tmp_path) == before


def test_outside_a_capture_the_terminal_links_read_only_the_stored_terminal_rows(tmp_path, monkeypatch):
    from ouroboros.chronicle_store import capture

    shared.world(tmp_path)
    _fill(tmp_path)
    with capture(tmp_path):
        whole = mi.terminal_rows(tmp_path)  # a capture's whole tail, read once for every reader
    built, real = [], mi._entry
    monkeypatch.setattr(mi, "_entry", lambda fact: built.append(fact[0]) or real(fact))
    assert mi.terminal_rows(tmp_path) == whole and [pos for _a, _m, pos, _s in whole] == [16]
    assert built == [16]  # the one stored terminal row, not the ten rows after the frontier


def test_a_chunk_of_tail_facts_is_stored_in_one_statement(tmp_path, monkeypatch):
    chronicle = ChronicleStore(tmp_path)
    assert chronicle.publish([], scan_state={"probe": 1}).ok
    rows = [[n, 100 + n, f"{n:064x}", ["gen", n] if n % 2 else None,
             {"chat_id": 1, "text_chars": n, "source_keys": [[1, "", "t", "h"]]}, "" if n % 3 == 0 else None]
            for n in range(500)]
    traced, real = [], sqlite3.connect

    def connect(*args, **kwargs):
        db = real(*args, **kwargs)
        db.set_trace_callback(traced.append)
        return db

    monkeypatch.setattr(sqlite3, "connect", connect)
    state = {"source": {"store": [1, 2], "frontier": 0}, "upper": 700, "anchor": [599, rows[-1][2]], "complete": True}
    assert chronicle.extend_tail(None, rows, state, replace=True)
    assert len([sql for sql in traced if "INTO tail_rows" in sql]) == 1  # 500 rows, one execution (one step)
    monkeypatch.undo()
    assert chronicle.tail_facts(state["source"]) == (state, rows)


def test_a_tail_writer_holding_a_stale_state_stores_nothing(tmp_path):
    chronicle = ChronicleStore(tmp_path)
    assert chronicle.publish([], scan_state={"probe": 1}).ok
    source = {"store": [1, 2], "frontier": 0}
    first, rows = {"source": source, "upper": 10, "anchor": [5, "a"], "complete": True}, [[0, 5, "a", None, {"chat_id": 1}, None]]
    assert chronicle.extend_tail(None, rows, first, replace=True)
    assert not chronicle.extend_tail(None, rows, first, replace=True)  # another writer stored first
    second, more = {**first, "upper": 20, "anchor": [15, "b"]}, [[1, 15, "b", ["g", 3], {"chat_id": 2}, ""]]
    assert chronicle.extend_tail(first, more, second)
    assert not chronicle.extend_tail(first, more, second)  # the same gap twice: stored once
    assert chronicle.tail_facts(source) == (second, rows + more) and chronicle.tail_facts({**source, "frontier": 1}) is None


def test_seals_rejections_and_a_row_in_two_rooms_change_the_open_rows_and_never_the_stored_facts(tmp_path):
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.projects_registry import bind_task_to_project

    beta = str(shared.world(tmp_path)["beta"])
    words, ts = "Start beta's second piece", "2026-09-04T00:00:00+00:00"
    ds.append(tmp_path, "chat", shared.msg(ts, words, client_message_id="origin-b"))  # the owner's words, in Main
    ref = build_owner_message_ref(chat_id=1, client_message_id="origin-b", ts=ts, text=words)
    bind_task_to_project(tmp_path, "tb2", "beta", origin={"ref": ref, "text": words})  # ...that started work in beta
    _fill(tmp_path)
    before, chronicle = _stored(tmp_path), ChronicleStore(tmp_path)
    origin = next(entry for entry in mi.open_rows_by_room(tmp_path)[beta] if entry[1].get("client_message_id") == "origin-b")

    def open_in(room):
        return {entry[0]["row_sha256"] for entry in mi.open_rows_by_room(tmp_path).get(room, ())}

    sha = origin[0]["row_sha256"]
    assert sha in open_in(beta) and sha in open_in("1")  # one row, two rooms
    draft = chronicle.publish_page(room_id=beta, text="sealed in beta", author=LIGHT,
                                   covers={"rows": [sha], "stream_span": [origin[2], origin[2]]}).record["id"]
    assert sha not in open_in(beta) and sha in open_in("1")  # a seal acts in its own room only
    assert chronicle.decide(draft, False, shared.MIND, "not this").ok
    assert sha in open_in(beta)  # a rejected draft stops acting
    candidate = chronicle.publish_page(room_id=beta, text="for the mind", author=LIGHT, metadata={"candidate": True},
                                       covers={"rows": [sha], "stream_span": [origin[2], origin[2]]}).record["id"]
    assert sha in open_in(beta) and chronicle.decide(candidate, True, shared.MIND, "adopted").ok and sha not in open_in(beta)
    assert _stored(tmp_path) == before


def test_a_stored_prefix_whose_last_row_no_longer_answers_is_read_live(tmp_path, monkeypatch):
    shared.world(tmp_path)
    _fill(tmp_path)
    anchor = ChronicleStore(tmp_path).tail_state()["anchor"]
    with sqlite3.connect(ds.path(tmp_path)) as db:  # the file keeps its identity, its last stored row does not
        body = json.loads(db.execute("SELECT body FROM dialogue WHERE stream='chat' AND seq=?", (anchor[0],)).fetchone()[0])
        db.execute("UPDATE dialogue SET body=? WHERE stream='chat' AND seq=?",
                   (json.dumps({**body, "text": "another history"}), anchor[0]))
    projected = _projected(monkeypatch)
    assert mi.rows_after_frontier(tmp_path) == _live(tmp_path) and len(projected) == 10 and mi.tail_owed(tmp_path)


def test_after_the_index_is_lost_a_view_reads_live_until_the_job_refills_it(tmp_path, monkeypatch):
    from ouroboros import memory_view as mv

    shared.world(tmp_path)
    task = {"id": "turn0001", "chat_id": 1, "_is_direct_chat": True}
    spec = mv.view_spec_for_task(task, tmp_path)
    assert maintenance.run_job(tmp_path, "memory_tail")["state"] == "done"
    stored = mv.capture_memory_view(tmp_path, task, spec)
    ChronicleStore(tmp_path).index_path.unlink()
    projected = _projected(monkeypatch)
    live = mv.capture_memory_view(tmp_path, task, spec)
    assert live.store_status == {"state": "active", "tail_facts": "filling"} and len(projected) == 10
    assert dataclasses.replace(live, store_status=stored.store_status) == stored and _stored(tmp_path)[1] == []
    assert "memory_tail" in maintenance.due(tmp_path) and maintenance.run_job(tmp_path, "memory_tail")["state"] == "done"
    projected.clear()
    assert mv.capture_memory_view(tmp_path, task, spec) == stored and projected == []
    assert "memory_tail" not in maintenance.due(tmp_path)


def test_each_focus_holds_in_its_one_capture_the_stored_rows_its_view_shows(tmp_path, monkeypatch):
    """A view with live rooms holds every stored row (their open rows); one without them, the terminal rows its
    pages link (presence, a child, a nanny's own room): the view it would read holding every row, with the
    chronicle entered twice (activation and the capture), never again for a row the capture lacks."""
    from ouroboros import chronicle_store, memory_view as mv

    rooms = shared.world(tmp_path)
    assert _fill(tmp_path)["state"] == "done"
    writer = {"kind": "mind", "task_id": "t2", "focus": {"role": "root", "task_id": "t2"}}
    sealed = next(address for address, _meta, pos in mi.open_room_rows(tmp_path, "1") if pos == 11)
    page = ChronicleStore(tmp_path).publish_page(room_id="1", text="t2's page", author=writer,
                                                 covers={"rows": [sealed["row_sha256"]], "stream_span": [11, 11]})
    assert page.ok, page  # its writer's later terminal row (16) stands under it
    held, init = [], chronicle_store._Snapshot.__init__
    monkeypatch.setattr(chronicle_store._Snapshot, "__init__", lambda self, texts, scope: init(self, texts, scope)
                        or held.append((scope, [row[0] for row in self.tail_rows])))
    entered, index = [], ChronicleStore._index
    monkeypatch.setattr(ChronicleStore, "_index", lambda self: entered.append(1) or index(self))
    task, scoped = {"id": "turn0001", "chat_id": 1}, mv.capture
    for role, room, scope in (("integrator", "1", "all"), ("presence", "1", "terminal"),
                              ("child", str(rooms["alpha"]), "terminal"), ("nanny", "1", "terminal")):
        spec = dataclasses.replace(mv.ROLE_DEFAULTS[role], room_id=room)
        monkeypatch.setattr(mv, "capture", scoped)
        held.clear()
        entered.clear()
        view = mv.capture_memory_view(tmp_path, task, spec)
        assert held == [(scope, list(range(10, 20)) if scope == "all" else [16])] and len(entered) == 2, role
        monkeypatch.setattr(mv, "capture", lambda root, tail: chronicle_store.capture(root, tail="all"))
        assert mv.capture_memory_view(tmp_path, task, spec) == view, role
        assert any(entry.get("terminal_links") for entry in [*view.story, *(view.room or {}).get("under_parts", ())]
                   ) or not spec.story, role


def test_a_place_arrival_whose_source_pointer_lands_after_it_was_stored_reads_as_live(tmp_path):
    from ouroboros.place_arrivals import SourceProjection

    shared.world(tmp_path)
    arrival = ds.append(tmp_path, "chat", shared.msg("2026-09-04T00:00:00+00:00", "a guest wrote", chat_id=4242,
                                                     type="place_arrival", place_id="p1"))
    _fill(tmp_path)
    with sqlite3.connect(ds.path(tmp_path)) as db:  # the intake points at the place's context after the append
        db.execute("UPDATE dialogue SET dispatch_input=? WHERE stream='chat' AND seq=?",
                   (json.dumps({"place_context_source_ref": {"kind": "place_context", "id": "c1"}}), arrival.seq))
    assert isinstance(ds.row_at(tmp_path, "chat", arrival.seq).row, SourceProjection)
    assert mi.rows_after_frontier(tmp_path) == _live(tmp_path)


def test_an_unreadable_tail_fact_never_holds_back_the_other_maintenance_jobs(tmp_path, monkeypatch):
    shared.world(tmp_path)

    def unreadable(_root):
        raise ValueError("chronicle authority shortened")

    monkeypatch.setattr(mi, "tail_owed", unreadable)
    assert "memory_tail" not in maintenance.due(tmp_path)  # deferred to the next look; the other jobs still answer
    monkeypatch.undo()
    assert "memory_tail" in maintenance.due(tmp_path)
