"""One root snapshot per owner message: typed columns, no result body, a failed read disclosed.

Routing runs on the supervisor loop thread. It used to walk every RUNNING and PENDING root
three or four times per message, decoding each one's result body (and its child drive's) to
read one status, and two of those walks were unguarded: a busy store dropped the message.
"""

from __future__ import annotations

import pytest

from ouroboros import server_owner_routing, server_routing_context as routing, store, task_store
from ouroboros.projects_registry import bind_task_to_project, create_project
from ouroboros.task_results import write_task_result
from tests.test_project_routing_v664 import _ctx


class _Now:
    def __init__(self, target, args=(), kwargs=None, daemon=False):
        self.target, self.args, self.kwargs = target, args, kwargs or {}

    def start(self):
        self.target(*self.args, **self.kwargs)


def _root(task_id, **fields):
    return {"id": task_id, "chat_id": 1, "root_task_id": task_id, "delegation_role": "root",
            "objective": f"work {task_id}", **fields}


@pytest.fixture
def main_host(tmp_path, monkeypatch):
    """Main with a running root, a split root running in its child drive, a pending root and a root
    whose answer settled while paid post-work keeps it RUNNING."""
    child = tmp_path / "child-drive"
    child.mkdir()
    write_task_result(tmp_path, "run-a", "running")
    write_task_result(child, "run-split", "running")
    write_task_result(tmp_path, "pend-c", "scheduled")
    write_task_result(tmp_path, "done-d", "completed", result="answered")
    turns = []
    ctx = _ctx(tmp_path, running={
        "run-a": {"task": _root("run-a"), "started_at": 1.0},
        "run-split": {"task": _root("run-split", drive_root=str(child)), "started_at": 2.0},
        "done-d": {"task": _root("done-d"), "started_at": 3.0},
    }, pending=[_root("pend-c")], direct=lambda *_a, **kwargs: turns.append(kwargs["task_metadata"]))
    monkeypatch.setattr(server_owner_routing.threading, "Thread", _Now)
    monkeypatch.setattr(routing, "_recent_root_results", lambda *_a: ([], {}))  # the result window's own reads
    return ctx, turns


def _route(ctx, client_message_id="m-1", chat_id=1, **task_metadata):
    server_owner_routing._route_owner_message(object(), ctx, {
        "chat_id": chat_id, "text": "how is it going?", "client_message_id": client_message_id,
        "log_text": "how is it going?", "origin_message_ref": {"chat_id": chat_id, "client_message_id": client_message_id},
        **({"task_metadata": task_metadata} if task_metadata else {})})


def _count(monkeypatch, owner, name, calls, *, key):
    real = getattr(owner, name)
    monkeypatch.setattr(owner, name, lambda *args, **kwargs: calls.append(key) or real(*args, **kwargs))


def test_a_main_message_reads_its_roots_once_and_decodes_no_result_body(main_host, monkeypatch):
    ctx, turns = main_host
    decoded, snapshots = [], []
    _count(monkeypatch, store, "materialize", decoded, key="body")
    _count(monkeypatch, task_store, "_decode", decoded, key="bodies")
    _count(monkeypatch, server_owner_routing, "addressable_roots", snapshots, key="route")
    _count(monkeypatch, routing, "addressable_roots", snapshots, key="rebuild")

    _route(ctx)

    assert decoded == [] and snapshots == ["route"]
    [metadata] = turns
    listed = [row["task_id"] for row in metadata["current_chat"]["addressable_root_tasks"]]
    assert listed == ["run-a", "run-split", "pend-c"]  # the settled post-work root is not steerable
    assert [row["task_id"] for row in metadata["main_routing_manifest"]["root_tasks"]] == listed
    assert "live_roots_unavailable" not in metadata["routing_contract"]


def test_a_project_message_reads_its_roots_once_for_delivery_and_the_room_manifest(main_host, monkeypatch):
    ctx, turns = main_host
    room = create_project(ctx.DRIVE_ROOT, "snapshot-room")
    ctx.PENDING.append(_root("room-queued", chat_id=room["chat_id"], project_id=room["id"]))
    write_task_result(ctx.DRIVE_ROOT, "room-queued", "scheduled")
    monkeypatch.setattr(routing, "_latest_project_task_result", lambda *_a: None)  # the pointer's own read
    decoded, snapshots = [], []
    _count(monkeypatch, store, "materialize", decoded, key="body")
    _count(monkeypatch, task_store, "_decode", decoded, key="bodies")
    _count(monkeypatch, server_owner_routing, "addressable_roots", snapshots, key="route")
    _count(monkeypatch, routing, "addressable_roots", snapshots, key="rebuild")

    _route(ctx, "m-room", chat_id=room["chat_id"])  # a queued root reads no room input: a decision turn

    assert decoded == [] and snapshots == ["route"]
    [metadata] = turns
    assert [row["task_id"] for row in metadata["project_routing_manifest"]["active_roots"]] == ["room-queued"]


def test_a_busy_store_is_a_disclosed_gap_and_the_message_still_gets_its_turn(main_host, monkeypatch):
    ctx, turns = main_host

    def busy(*_args, **_kwargs):
        raise store.StoreBusy("index busy", reason="contention")

    create_project(ctx.DRIVE_ROOT, "listed-room")  # Main's manifest stays: it lists the room
    monkeypatch.setattr(task_store, "task_drive_status", busy)
    _route(ctx)

    [metadata] = turns
    assert metadata["routing_contract"]["live_roots_unavailable"] == "StoreBusy"
    assert "current_chat" not in metadata
    assert metadata["main_routing_manifest"]["omissions"]["root_tasks"] is None  # unknown, not zero


def test_a_room_row_reaches_the_other_live_roots_as_context_once_whatever_the_routing_outcome(tmp_path, monkeypatch):
    """Two live readers: the row is addressed to none and a labelled context entry lands in both, no corpus and no
    seal generation moves; a sole reader gets the addressed mail and the other root the context entry naming it;
    a sealed reader's refusal still writes the context entry once, under the head-routed marker; a late answer is
    my human's words too (routed to its asker's head, carried to the other root with its card); a press on a card is
    a typed act, not words: it follows its author's thread and reaches no other root."""
    from ouroboros import dialogue_store, owner_pause
    from ouroboros.chat_chain import format_address, row_address
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.owner_relay import ROUTED_BY_HEAD
    from supervisor import queue as q

    room = create_project(tmp_path, "carry-room")
    chat = int(room["chat_id"])
    turns = []
    running = {tid: {"task": _root(tid, chat_id=chat, project_id=room["id"], drive_root=str(tmp_path)), "started_at": 1.0}
               for tid in ("root-a", "root-b")}
    for tid in running:
        write_task_result(tmp_path, tid, "running")
    ctx = _ctx(tmp_path, running=running, direct=lambda *_a, **kwargs: turns.append(kwargs["task_metadata"]))
    monkeypatch.setattr(server_owner_routing.threading, "Thread", _Now)
    monkeypatch.setattr(routing, "_recent_root_results", lambda *_a: ([], {}))
    monkeypatch.setattr(q, "DRIVE_ROOT", tmp_path)
    fence = {"token": "f" * 32, "root_task_id": "root-a", "task_id": "root-a", "status": "active", "owner_message_generation": 0}
    monkeypatch.setattr(q, "ACCEPTANCE_FENCES", {"root-a": fence})

    def arrives(client_message_id, text, **typed):
        row = {"chat_id": chat, "direction": "in", "ts": f"2026-10-01T00:0{client_message_id[-1]}:00Z", "text": text,
               "client_message_id": client_message_id, **typed}
        dialogue_store.append(tmp_path, "chat", row)
        _route(ctx, client_message_id, chat_id=chat, **typed)
        return format_address(row_address(row))

    def mail(tid):
        return {entry["msg_id"]: entry for entry in drain_owner_entries(tmp_path, tid)}

    first = arrives("m-1", "how is it going?")
    assert len(turns) == 1 and fence["owner_message_generation"] == 0  # a head decides; nothing was delivered
    for tid in ("root-a", "root-b"):
        [entry] = mail(tid).values()
        assert entry["msg_id"] == f"relay:carry-context:{first}:{tid}" and entry["source_task_id"] == tid
        assert entry["host_carried"] == {"went_to": ROUTED_BY_HEAD} and entry["relay"]["source"]["label"] == "carry-room"
        assert entry["owner_rows"][0]["text"] == "how is it going?" and entry["text"] == ""
    # The sole reader (the other root paused) takes the row as addressed mail; the other hears where it went.
    owner_pause.install_fence(tmp_path, "root-b", request_id="press-1")
    second = arrives("m-2", "and the charts?")
    assert len(turns) == 1 and fence["owner_message_generation"] == 1
    assert set(mail("root-a")) == {f"relay:carry-context:{first}:root-a", "m-2:root-a"}
    assert mail("root-b")[f"relay:carry-context:{second}:root-b"]["host_carried"] == {"went_to": "root-a"}
    assert "m-2:root-b" not in mail("root-b")
    # A sealed reader refuses the row: no addressed mail, and the context entry is still written once.
    fence["status"] = "sealed"
    third = arrives("m-3", "one more thing")
    assert len(turns) == 2 and fence["owner_message_generation"] == 1
    assert "m-3:root-a" not in mail("root-a")
    assert mail("root-a")[f"relay:carry-context:{third}:root-a"]["host_carried"] == {"went_to": ROUTED_BY_HEAD}
    assert mail("root-b")[f"relay:carry-context:{third}:root-b"]["host_carried"] == {"went_to": ROUTED_BY_HEAD}
    assert len(mail("root-a")) == 3 and len(mail("root-b")) == 3
    # A late answer: its asker's head (root-a, unsealed) takes it as addressed mail; the other root hears, as
    # context, where it went, with the card the accepted row kept; the head gets no second, context copy.
    fence["status"] = "active"
    running["root-a"]["task"]["metadata"] = {"late_answer": {"task_id": "asker", "quiz_id": "q0"}}
    card = {"quiz_id": "q1", "state": "answered", "answered_after_terminal": True, "question": "Which db?",
            "options": ["sqlite", "postgres"], "answered_index": 1}
    fourth = arrives("m-4", "2. postgres", late_answer={"task_id": "asker", "quiz_id": "q1",
                                                        "answer_snapshot": {"task_id": "asker", "quiz": card}})
    assert len(turns) == 2 and fence["owner_message_generation"] == 2 and "m-4:root-a" in mail("root-a")
    [context] = [entry for entry in mail("root-b").values() if fourth in entry["msg_id"]]
    assert context["msg_id"] == f"relay:carry-context:{fourth}:root-b" and context["host_carried"] == {"went_to": "root-a"}
    assert "Question was: Which db?" in context["relay"]["quotes"][0]["question_frame"]
    assert context["owner_rows"][0]["text"] == "2. postgres" and f"relay:carry-context:{fourth}:root-a" not in mail("root-a")
    # A press on a card: the same thread takes it; no root of the room hears it as context.
    fifth = arrives("m-5", "Book it", card_press={"task_id": "asker", "card_id": "card_" + "0" * 16, "request_id": "r1"})
    assert "m-5:root-a" in mail("root-a") and fence["owner_message_generation"] == 3
    assert not any(fifth in msg_id for tid in ("root-a", "root-b") for msg_id in mail(tid))
    assert len(mail("root-a")) == 5 and len(mail("root-b")) == 4


def test_settlement_is_read_on_the_actors_own_drive(tmp_path):
    child = tmp_path / "child"
    child.mkdir()
    write_task_result(tmp_path, "running-in-child", "completed")  # settled canonically, its actor still runs
    write_task_result(child, "running-in-child", "running")
    write_task_result(tmp_path, "settled-in-child", "running")  # settled where it runs; copyback lags
    write_task_result(child, "settled-in-child", "completed")
    ctx = _ctx(tmp_path, running={
        "running-in-child": {"task": _root("running-in-child", drive_root=str(child))},
        "settled-in-child": {"task": _root("settled-in-child", child_drive_root=str(child))},
        "unknown-yet": {"task": _root("unknown-yet")},  # no row anywhere: unsettled, as an absent body was
    })

    assert [row["task_id"] for row in routing.addressable_roots(ctx).rows()] == ["running-in-child", "unknown-yet"]


def test_rows_of_a_chat_keep_the_membership_rule(tmp_path):
    room = create_project(tmp_path, "rows-room")
    bind_task_to_project(tmp_path, "bound", room["id"], origin={"absent": "system"})
    ctx = _ctx(tmp_path, running={tid: {"task": _root(tid, **fields)} for tid, fields in (
        ("bound", {}), ("main-only", {}), ("malformed", {"chat_id": "not-a-chat"}), ("chatless", {"chat_id": None}),
    )})
    snapshot = routing.addressable_roots(ctx)

    def ids(chat_id):
        return [row["task_id"] for row in snapshot.rows(chat_id)]

    assert ids(None) == ["bound", "main-only", "malformed", "chatless"]
    assert ids(1) == ["bound", "main-only"]  # its own chat, whatever room it is bound to
    assert ids(room["chat_id"]) == ["bound"]  # the binding's room
    assert ids(0) == ["main-only", "malformed", "chatless"]  # unbound: room 0, as the binding answers
    assert ids(5) == []
    rows = snapshot.rows(1)
    rows[0]["task_id"] = "edited"
    assert ids(1) == ["bound", "main-only"]  # callers get copies


def test_a_reading_direct_turn_replaces_its_own_queue_row(tmp_path, monkeypatch):
    queue_row = {"task": _root("same")}
    ctx = _ctx(tmp_path, running={"same": queue_row, "other": {"task": _root("other")}})
    direct = {"task_id": "same", "status": "running", "chat_id": 1, "direct_chat": True, "room_reader": True}
    monkeypatch.setattr(routing, "_active_direct_roots", lambda _ctx: [direct])

    assert [(row["task_id"], bool(row.get("direct_chat"))) for row in routing.addressable_roots(ctx).rows(1)] == [
        ("other", False), ("same", True)]
    monkeypatch.setattr(routing, "_active_direct_roots", lambda _ctx: [{**direct, "room_reader": False}])
    assert [(row["task_id"], bool(row.get("direct_chat"))) for row in routing.addressable_roots(ctx).rows(1)] == [
        ("same", False), ("other", False)]


def test_one_field_readers_take_the_typed_column_not_the_body(tmp_path, monkeypatch):
    from ouroboros.owner_mailbox import mailbox_drain_ended
    from supervisor import queue as q

    child = tmp_path / "child"
    write_task_result(tmp_path, "split", "running", child_drive_root=str(child))
    write_task_result(tmp_path, "done", "completed")
    monkeypatch.setattr(q, "DRIVE_ROOT", tmp_path)
    decoded = []
    _count(monkeypatch, store, "materialize", decoded, key="body")
    _count(monkeypatch, task_store, "_decode", decoded, key="bodies")

    assert q._task_drive_for_task({"id": "split"}, "split") == child
    assert q._task_drive_for_task({"id": "absent"}, "absent") == tmp_path
    assert mailbox_drain_ended(tmp_path, "done") is True and mailbox_drain_ended(tmp_path, "split") is False
    assert mailbox_drain_ended(tmp_path, "absent") is False
    assert decoded == []


@pytest.fixture
def running_roots(tmp_path):
    running = {f"run-{index}": {"task": _root(f"run-{index}"), "started_at": float(index)} for index in range(6)}
    for tid in running:
        write_task_result(tmp_path, tid, "running")
    return _ctx(tmp_path, running=running)


def test_the_running_roots_pause_fences_cost_one_store_read(running_roots, monkeypatch):
    """Whether a running root still reads its room depends on its owner-Pause fence: one read of the store
    answers every root (one statement over ``json_each``), however many run; a paused root reads no input."""
    from ouroboros import owner_pause, store_hot_facts

    owner_pause.install_fence(running_roots.DRIVE_ROOT, "run-3", request_id="press-1")
    reads = []
    _count(monkeypatch, store_hot_facts, "_read_rows", reads, key="fences")

    readers = {row["task_id"]: row["room_reader"] for row in routing.addressable_roots(running_roots).rows()}

    assert readers == {**{f"run-{index}": True for index in range(6)}, "run-3": False}
    assert len(reads) == 1


def test_a_fence_store_that_cannot_answer_is_a_disclosed_gap(running_roots, monkeypatch):
    from ouroboros import store_hot_facts

    def busy(*_args, **_kwargs):
        raise store.StoreBusy("index busy", reason="contention")

    monkeypatch.setattr(store_hot_facts, "_read_rows", busy)
    snapshot = routing.addressable_roots(running_roots)
    assert snapshot.gap == "StoreBusy" and snapshot.rows() == []
