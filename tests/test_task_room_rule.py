"""One room rule (owner decision A4): the bus write seam, replay membership and cognition
agree on the room a task-addressed row belongs to, and a warm frame costs no store read.

Two-sided: every case fails on d375d9fd0 (the bus never re-addressed a row that reached
it without lineage, and replay membership read only the explicit-roots map) and asserts
the still-working case beside it (a Main direct task with a project_id stays Main, a
pinned or origin-addressed row keeps its producer's address).
"""
from __future__ import annotations

from tests import _dialogue as dialogue_test

import json
from types import SimpleNamespace

import pytest

from ouroboros import projects_registry as pr
from ouroboros.project_dialogue import ROOM_FIXED_ROW_TYPES, room_membership
from ouroboros.task_results import write_task_result

pytestmark = pytest.mark.serial


def _bus(monkeypatch, root):
    from supervisor import message_bus

    sent = []
    monkeypatch.setattr(message_bus, "DATA_DIR", root)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 1, "owner_chat_id": 1})
    monkeypatch.setattr(message_bus, "get_bridge", lambda: SimpleNamespace(send_message=lambda *a, **k: sent.append((a, k))))
    return message_bus, sent


def _progress_rows(root):
    return dialogue_test.log_rows(root / "logs/progress.jsonl")


def _world(root):
    """A converted root with a cancelled child, a root bound to A with a child bound to B, a
    room-born root, and a Main direct task that carries a project_id."""
    a, b, native = (pr.create_project(root, pid, name=pid) for pid in ("room-a", "room-b", "room-native"))
    write_task_result(root, "root-1", "running", root_task_id="root-1", chat_id=1)
    pr.bind_task_to_project(root, "root-1", a["id"], origin={"absent": "system"})
    write_task_result(root, "kid-1", "cancelled", delegation_role="subagent", parent_task_id="root-1",
                      root_task_id="root-1", chat_id=1)
    write_task_result(root, "mid-1", "running", delegation_role="subagent", parent_task_id="root-1",
                      root_task_id="root-1", chat_id=1)
    pr.bind_task_to_project(root, "mid-1", b["id"], origin={"absent": "system"})
    write_task_result(root, "grand-1", "running", delegation_role="subagent", parent_task_id="mid-1",
                      root_task_id="root-1", chat_id=1)
    write_task_result(root, "born-1", "running", root_task_id="born-1", chat_id=native["chat_id"],
                      project_id=native["id"])
    write_task_result(root, "direct-1", "running", root_task_id="direct-1", chat_id=1, _is_direct_chat=True,
                      project_id=native["id"])
    return {"a": a["chat_id"], "b": b["chat_id"], "native": native["chat_id"]}


def _membership(root, chat):
    projects = pr.reserved_project_chat_ids(root)
    return room_membership(chat if chat in projects else 1, projects, [], pr.RoomResolver(root))


@pytest.mark.parametrize("task_id, room", [
    ("kid-1", "a"),        # a child of a converted root that carries only its id
    ("grand-1", "b"),      # root bound to A, its parent bound to B: the nearest conversion wins
    ("born-1", "native"),  # born in a Project, bound to nothing
])
def test_the_bus_and_replay_put_a_project_tasks_bare_row_in_its_room(tmp_path, monkeypatch, task_id, room):
    rooms = _world(tmp_path)
    bus, sent = _bus(monkeypatch, tmp_path)
    bus.send_with_budget(1, "Stopped by Ouroboros", is_progress=True, task_id=task_id, role="system",
                         system_type="subagent_terminal_notice")
    row = _progress_rows(tmp_path)[-1]
    assert row["chat_id"] == rooms[room] == sent[-1][0][0], "the durable row and the live frame share the room"
    if task_id != "born-1":
        assert (row["delegation_role"], row["parent_task_id"], row["root_task_id"]) == (
            "subagent", {"kid-1": "root-1", "grand-1": "mid-1"}[task_id], "root-1"), "lineage rides the row"
    # Replay: the row as it was recorded before the seam (chat 1, task id only) lives in the same room.
    legacy = {"type": "send_message", "task_id": task_id, "chat_id": 1}
    assert _membership(tmp_path, rooms[room])(1, legacy) is True
    assert _membership(tmp_path, 1)(1, legacy) is False, "Main does not keep another room's task"
    assert all(_membership(tmp_path, rooms[other])(1, legacy) is False for other in rooms if other != room)


def test_live_routing_and_replay_agree_for_a_deep_chain_with_two_conversions(tmp_path):
    rooms = _world(tmp_path)
    assert pr.project_chat_for_task_tree(tmp_path, "grand-1") == rooms["b"]
    assert pr.project_chat_for_task_tree(tmp_path, "grand-1", "mid-1", "root-1") == rooms["b"], "a supplied root never shortcuts the parent chain"
    assert pr.project_chat_for_task_tree(tmp_path, "kid-1", "root-1", "root-1") == rooms["a"]
    row = {"task_id": "grand-1", "parent_task_id": "mid-1", "root_task_id": "root-1"}
    assert _membership(tmp_path, rooms["b"])(1, row) and not _membership(tmp_path, rooms["a"])(1, row)


def test_a_main_direct_task_with_a_project_id_stays_main(tmp_path, monkeypatch):
    rooms = _world(tmp_path)
    bus, sent = _bus(monkeypatch, tmp_path)
    for meta in ({"_is_direct_chat": True}, {}):  # the flagged turn and its unflagged finalization frame
        bus.send_with_budget(1, "Thinking", is_progress=True, task_id="direct-1", role="system",
                             system_type="host_progress", progress_meta=meta)
        assert _progress_rows(tmp_path)[-1]["chat_id"] == 1 == sent[-1][0][0]
    assert pr.project_chat_for_task_tree(tmp_path, "direct-1") == 0, "project_id is a copy, never the room"
    assert _membership(tmp_path, 1)(1, {"task_id": "direct-1"}) is True
    assert _membership(tmp_path, rooms["native"])(1, {"task_id": "direct-1"}) is False


@pytest.mark.parametrize("system_type", sorted(ROOM_FIXED_ROW_TYPES))
def test_a_fixed_row_type_keeps_its_producers_address_and_borrows_no_lineage(tmp_path, monkeypatch, system_type):
    rooms = _world(tmp_path)
    bus, sent = _bus(monkeypatch, tmp_path)
    bus.send_with_budget(rooms["native"], "Not delivered", task_id="kid-1", role="system", system_type=system_type)
    stored = dialogue_test.log_rows(tmp_path / "logs/chat.jsonl")[-1]
    assert stored["chat_id"] == rooms["native"] == sent[-1][0][0], "the producer's address stands"
    assert not {"delegation_role", "parent_task_id", "root_task_id"} & set(stored), "the target is not the speaker"


def _count_store_work(monkeypatch):
    from ouroboros import store, task_status, task_store

    counts = {"open_store": 0, "load_effective_task_result": 0}
    real_open, real_load = store.open_store, task_status.load_effective_task_result

    def _open(*a, **k):
        counts["open_store"] += 1
        return real_open(*a, **k)

    def _load(*a, **k):
        counts["load_effective_task_result"] += 1
        return real_load(*a, **k)

    monkeypatch.setattr(store, "open_store", _open)
    monkeypatch.setattr(task_store, "open_store", _open)
    monkeypatch.setattr(pr, "open_store", _open)
    monkeypatch.setattr(task_status, "load_effective_task_result", _load)
    return counts


@pytest.mark.parametrize("through_handler", [False, True], ids=["send_with_budget", "event_queue_handler"])
def test_a_warm_frame_opens_no_store_and_a_cold_task_id_opens_at_most_once(tmp_path, monkeypatch, through_handler):
    from supervisor.events_chat_delivery import _handle_send_message

    rooms = _world(tmp_path)
    bus, sent = _bus(monkeypatch, tmp_path)
    counts = _count_store_work(monkeypatch)
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, send_with_budget=bus.send_with_budget)

    def frame(task_id, meta, n=10):
        for i in range(n):
            if through_handler:
                _handle_send_message({"type": "send_message", "chat_id": 1, "text": f"frame {i}", "is_progress": True,
                                      "role": "system", "system_type": "host_progress", "task_id": task_id,
                                      "progress_meta": dict(meta)}, ctx)
            else:
                bus.send_with_budget(1, f"frame {i}", is_progress=True, task_id=task_id, progress_meta=dict(meta),
                                     role="system", system_type="host_progress")

    lineage = {"delegation_role": "subagent", "parent_task_id": "root-1", "root_task_id": "root-1"}
    frame("kid-1", lineage)
    assert counts == {"open_store": 0, "load_effective_task_result": 0}, "a frame carrying its lineage opens nothing"
    frame("grand-1", {})
    opened = counts["open_store"]
    assert 0 < opened <= 2, "one indexed read per unseen id on the chain (grand-1, mid-1 is bound)"
    frame("grand-1", {})
    assert counts["open_store"] == opened, "warm frames open nothing"
    assert counts["load_effective_task_result"] == 0, "the seam never parses a result body"
    assert {a[0][0] for a in sent} == {rooms["a"], rooms["b"]}


def test_a_conversion_invalidates_room_answers_and_a_store_commit_does_not(tmp_path):
    rooms = _world(tmp_path)
    assert pr.project_chat_for_task_tree(tmp_path, "kid-1") == rooms["a"]
    write_task_result(tmp_path, "kid-1", "cancelled", progress_note="a later commit")
    assert pr.RoomResolver(tmp_path).answers.get("kid-1") == rooms["a"], "a commit keeps the warm answer"
    c = pr.create_project(tmp_path, "room-c", name="room-c")
    pr.bind_task_to_project(tmp_path, "kid-1", c["id"], origin={"absent": "system"})
    assert pr.project_chat_for_task_tree(tmp_path, "kid-1") == c["chat_id"], "the nearest conversion wins at once"


def test_an_unadmitted_id_is_answered_but_a_miss_is_never_remembered(tmp_path):
    rooms = _world(tmp_path)
    assert pr.project_chat_for_task_tree(tmp_path, "later-1") == 0 and pr.RoomResolver(tmp_path).lineage("later-1") == {}
    assert pr.project_chat_for_task_tree(tmp_path, "later-2", "root-1", "root-1") == rooms["a"], "supplied lineage routes an unseen id"
    for task_id in ("later-1", "later-2"):
        write_task_result(tmp_path, task_id, "scheduled", delegation_role="subagent", parent_task_id="root-1",
                          root_task_id="root-1", chat_id=1)
        assert pr.project_chat_for_task_tree(tmp_path, task_id) == rooms["a"], "admission creates the facts"


def _busy_store(monkeypatch, busy):
    """``open_store`` raises StoreBusy while ``busy["on"]``: the store under contention."""
    from ouroboros import store

    real_open = store.open_store

    def _open(*a, **k):
        if busy["on"]:
            raise store.StoreBusy("database is locked")
        return real_open(*a, **k)

    monkeypatch.setattr(pr, "open_store", _open)
    return _open


def test_an_unavailable_ancestor_read_is_never_memoized(tmp_path, monkeypatch):
    """Astra's counterexample: leaf -> middle -> mid (bound to B) -> root (bound to A); the leaf's
    facts are warm, the middle's read hits a busy store. The fallback answer of that call may
    be wrong, but it is not remembered: once the store answers again the leaf resolves to B."""
    a, b = (pr.create_project(tmp_path, pid, name=pid) for pid in ("room-a", "room-b"))
    write_task_result(tmp_path, "root", "running", root_task_id="root", chat_id=1)
    pr.bind_task_to_project(tmp_path, "root", a["id"], origin={"absent": "system"})
    write_task_result(tmp_path, "mid", "running", delegation_role="subagent", parent_task_id="root", root_task_id="root", chat_id=1)
    pr.bind_task_to_project(tmp_path, "mid", b["id"], origin={"absent": "system"})
    write_task_result(tmp_path, "middle", "running", delegation_role="subagent", parent_task_id="mid", root_task_id="root", chat_id=1)
    write_task_result(tmp_path, "leaf", "running", delegation_role="subagent", parent_task_id="middle", root_task_id="root", chat_id=1)
    assert pr.RoomResolver(tmp_path).lineage("leaf")["parent_task_id"] == "middle"  # warm leaf facts, cold middle
    busy = {"on": True}
    _busy_store(monkeypatch, busy)
    contended = pr.project_chat_for_task_tree(tmp_path, "leaf")
    assert contended in (0, a["chat_id"]), "a best-effort answer while the middle is unreadable"
    assert "leaf" not in pr.RoomResolver(tmp_path).answers, "an incomplete ancestry read is not memoized"
    busy["on"] = False
    assert pr.project_chat_for_task_tree(tmp_path, "leaf") == b["chat_id"], "the retried read finds the nearer conversion"
    assert pr.RoomResolver(tmp_path).answers["leaf"] == b["chat_id"], "a complete read is memoized"


def test_a_healthy_warm_frame_still_opens_no_store(tmp_path, monkeypatch):
    rooms = _world(tmp_path)
    assert pr.project_chat_for_task_tree(tmp_path, "grand-1") == rooms["b"]
    counts = _count_store_work(monkeypatch)
    for _ in range(10):
        assert pr.project_chat_for_task_tree(tmp_path, "grand-1") == rooms["b"]
    assert counts["open_store"] == 0


def test_an_absent_id_is_read_once_per_resolver(tmp_path, monkeypatch):
    _world(tmp_path)
    counts = _count_store_work(monkeypatch)
    rooms = pr.RoomResolver(tmp_path)
    for _ in range(5):
        assert rooms("ghost-1") == 0 and rooms.lineage("ghost-1") == {}
    assert counts["open_store"] == 1, "one read per resolver for an id the store holds no row for"
    rooms.warm(["ghost-1", "ghost-2"])
    assert counts["open_store"] == 2 and rooms("ghost-2") == 0 and counts["open_store"] == 2
    assert pr.RoomResolver(tmp_path)("ghost-1") == 0 and counts["open_store"] == 3, "a fresh resolver (a new frame) asks again"


def test_a_warm_follows_the_ancestors_of_remembered_tasks_and_says_when_it_could_not(tmp_path, monkeypatch):
    """Children whose lineage is remembered still send the rule to their ancestors: the warm reads those
    too, so the rule then walks them with no read per id, and it answers False while the store cannot
    answer them (a memory count is then not reused) instead of leaving one failing read per task."""
    rooms = _world(tmp_path)
    leaves = [f"leaf-{i}" for i in range(6)]
    for leaf in leaves:
        write_task_result(tmp_path, leaf, "running", delegation_role="subagent", parent_task_id="grand-1",
                          root_task_id="root-1", chat_id=1)
    for key in [key for key in pr._TASK_LINEAGE if key[0] == str(tmp_path)]:
        del pr._TASK_LINEAGE[key]
    assert set(pr._read_lineage(tmp_path, leaves)) == set(leaves)  # the leaves are remembered, their ancestors not
    attempts, real = [], pr._read_lineage
    monkeypatch.setattr(pr, "_read_lineage", lambda *args, **_kwargs: attempts.append(sorted(args[1])))  # None: busy
    assert pr.RoomResolver(tmp_path).warm(leaves) is False and attempts == [["grand-1"]]  # a root is only asked its binding
    monkeypatch.setattr(pr, "_read_lineage", real)
    counts = _count_store_work(monkeypatch)
    resolver = pr.RoomResolver(tmp_path)
    assert resolver.warm(leaves) is True and counts["open_store"] == 2  # two generations: grand-1, then its parent and root
    assert [resolver(leaf) for leaf in leaves] == [rooms["b"]] * 6 and counts["open_store"] == 2


def test_a_warm_stops_where_the_rule_stops_at_a_remembered_tasks_binding(tmp_path, monkeypatch):
    """A remembered task whose walk ends at a bound parent needs no further lineage: the warm reads none
    (not the bound parent, not the root), even while the store is busy, and the rule still answers."""
    rooms = _world(tmp_path)
    for key in [key for key in pr._TASK_LINEAGE if key[0] == str(tmp_path)]:
        del pr._TASK_LINEAGE[key]
    assert set(pr._read_lineage(tmp_path, ["grand-1"])) == {"grand-1"}  # remembered; its parent mid-1 is bound to B
    attempts, real = [], pr._read_lineage
    monkeypatch.setattr(pr, "_read_lineage", lambda *args, **_kwargs: attempts.append(sorted(args[1])))  # None: busy
    assert pr.RoomResolver(tmp_path).warm(["grand-1"]) is True and attempts == []
    monkeypatch.setattr(pr, "_read_lineage", real)
    counts = _count_store_work(monkeypatch)
    resolver = pr.RoomResolver(tmp_path)
    assert resolver.warm(["grand-1"]) is True and resolver("grand-1") == rooms["b"] and counts["open_store"] == 0


def _request(root, body):
    async def _json():
        return body

    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(drive_root=root)), json=_json)


def _history(root, chat):
    import asyncio

    from ouroboros.gateway.history import make_chat_history_endpoint

    response = asyncio.run(make_chat_history_endpoint(root)(SimpleNamespace(query_params={"chat_id": str(chat)})))
    assert response.status_code == 200, response.body
    return [row["task_id"] for row in json.loads(response.body)["messages"] if row.get("task_id")]


def test_converting_a_middle_task_moves_its_subtrees_earlier_rows_out_of_the_old_room(tmp_path, monkeypatch):
    """Astra's counterexample through the real conversion endpoint: root R bound to A, its child C
    emits a row into A, then the owner turns C into project B. Live delivery already goes to B;
    history, chat_chain and the memory inventory must show C's earlier row in B only."""
    import asyncio

    from ouroboros import chat_chain, memory_inventory
    from ouroboros.gateway.projects import api_project_from_task

    monkeypatch.setattr("ouroboros.gateway.state._chat_activities_snapshot_safe", lambda *_a, **_k: [])
    a = pr.create_project(tmp_path, "room-a", name="Room A")
    write_task_result(tmp_path, "root-1", "running", root_task_id="root-1", chat_id=1)
    pr.bind_task_to_project(tmp_path, "root-1", a["id"], origin={"absent": "system"})
    write_task_result(tmp_path, "kid-1", "running", delegation_role="subagent", parent_task_id="root-1",
                      root_task_id="root-1", chat_id=1)
    bus, sent = _bus(monkeypatch, tmp_path)
    bus.send_with_budget(1, "Root planning", is_progress=True, task_id="root-1", role="system", system_type="host_progress")
    bus.send_with_budget(1, "Child working", is_progress=True, task_id="kid-1", role="system", system_type="host_progress")
    assert [row["chat_id"] for row in _progress_rows(tmp_path)] == [a["chat_id"]] * 2, "before: both rows live in A"
    assert _history(tmp_path, a["chat_id"]) == ["root-1", "kid-1"]

    response = asyncio.run(api_project_from_task(_request(tmp_path, {"task_id": "kid-1", "id": "task-kid-1"})))
    payload = json.loads(response.body)
    b = int(payload["project"]["chat_id"])
    assert response.status_code == 200 and payload["binding"]["project_id"] == payload["project"]["id"] and b != a["chat_id"]

    bus.send_with_budget(1, "Child still working", is_progress=True, task_id="kid-1", role="system", system_type="host_progress")
    assert sent[-1][0][0] == b, "live delivery follows the conversion"
    assert _history(tmp_path, a["chat_id"]) == ["root-1"], "A no longer shows the converted subtree's earlier row"
    assert _history(tmp_path, b) == ["kid-1", "kid-1"], "B shows the earlier row beside the new one"
    assert _history(tmp_path, 1) == [], "Main shows neither"
    assert [row.get("task_id") for _addr, row, _pos in chat_chain.iter_room_rows(tmp_path, a["chat_id"])] == []
    assert [row.get("task_id") for _addr, row, _pos in chat_chain.iter_room_rows(tmp_path, b)] == []
    facts = memory_inventory.membership_facts(tmp_path)
    early = _progress_rows(tmp_path)[1]
    assert memory_inventory.rooms_of_row(memory_inventory.row_meta(early), facts) == {str(b)}
    # Controls: the root's own row stays in its room, a Project-born row in its own room, Main work in Main.
    assert memory_inventory.rooms_of_row(memory_inventory.row_meta(_progress_rows(tmp_path)[0]), facts) == {str(a["chat_id"])}
    native = pr.create_project(tmp_path, "room-native", name="Native")
    write_task_result(tmp_path, "born-1", "running", root_task_id="born-1", chat_id=native["chat_id"])
    facts = memory_inventory.membership_facts(tmp_path)  # the registry facts of the moment, as a capture takes them
    assert memory_inventory.rooms_of_row({"chat_id": native["chat_id"], "task_id": "born-1"}, facts) == {str(native["chat_id"])}
    assert memory_inventory.rooms_of_row({"chat_id": 1, "task_id": "direct-9"}, facts) == {"1"}


def test_new_quiz_uses_the_same_nearest_ancestor_room_then_keeps_its_publication(tmp_path):
    from supervisor.log_addressing import resolve_quiz_publication

    rooms = _world(tmp_path)
    write_task_result(tmp_path, 'great-1', 'running', parent_task_id='grand-1',
                      root_task_id='root-1', chat_id=1, delegation_role='subagent')
    assert resolve_quiz_publication(tmp_path, 'great-1', {}, 1, 'grand-1', 'root-1') == rooms['b']
    assert resolve_quiz_publication(tmp_path, 'born-1', {}, 1) == rooms['native']
    assert resolve_quiz_publication(tmp_path, 'direct-1', {}, 1) == 1
    pinned = {'publication_chat_id': 1, 'publication_evidence': 'pre_exposure'}
    assert resolve_quiz_publication(tmp_path, 'great-1', pinned, rooms['b']) == 1
    # An outage may not silently select Main for a new publication; an already-bound one needs no resolver.
    from ouroboros.projects_registry import _bindings_path
    _bindings_path(tmp_path).write_text('{')
    with pytest.raises(ValueError):
        resolve_quiz_publication(tmp_path, 'great-1', {}, 1)
    assert resolve_quiz_publication(tmp_path, 'great-1', pinned, 0) == 1


@pytest.mark.parametrize('kind', ['quiz', 'quiz_answer', 'quiz_closed'])
def test_quiz_display_follows_current_room_without_rewriting_publication(tmp_path, kind):
    rooms = _world(tmp_path)
    row = {'type': kind, 'task_id': 'grand-1', 'chat_id': rooms['native'],
           'quiz': {'quiz_id': 'accepted-before-conversion'}}
    assert not _membership(tmp_path, rooms['native'])(rooms['native'], row)
    assert _membership(tmp_path, rooms['b'])(rooms['native'], row)
    assert row['chat_id'] == rooms['native'], 'membership never rewrites the publication address'
    assert not _membership(tmp_path, 1)(rooms['native'], row)
    # Ordinary work still obeys the public room rule.
    assert _membership(tmp_path, rooms['b'])(rooms['native'], {**row, 'type': 'send_message'})
