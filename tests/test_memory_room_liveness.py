"""Human words, mind notes, running work and gaps keep rooms live; everything else keeps an address."""
import dataclasses

import pytest

from ouroboros import chat_chain, memory_floor, memory_inventory as mi, memory_view as mv
from ouroboros.chronicle_store import ChronicleStore
from ouroboros.tools.chronicle import _memory_read, page_covers
from ouroboros.tools.registry import ToolContext
from tests import _memory_inventory_shared as shared
from tests._store_shared import seed_result
from tests.test_memory_view_room import MAIN_TASK, _addr, _install, _ts, _view


def seal(root, room, row):
    coverage = page_covers(root, room, from_addr=_addr(row), to_addr=_addr(row))["covers"]
    assert ChronicleStore(root).publish_page(room_id=room, text="This arc is closed.", covers=coverage,
                                             author=shared.MIND).ok


@pytest.mark.parametrize("mode", ["lines", "lines_with_words", "directory"])
def test_own_words_and_host_facts_share_one_addressed_line_in_every_live_mode(tmp_path, mode):
    rooms, rows = _install(tmp_path)
    for room, row in ((str(rooms["beta"]), rows[13]), ("777", rows[18])):
        seal(tmp_path, room, row)
    snapshot, text = _view(tmp_path, MAIN_TASK, live_rooms=mode)
    assert {r["room_id"] for r in snapshot.live_rooms} == {str(rooms["alpha"])}
    assert {r["room_id"] for r in snapshot.other_rooms} == {str(rooms["beta"]), "777"}
    assert "2 open rooms;" in text and "memory_read(room_id='*', start=0)" in text
    assert f"### Project Beta [chat_id={rooms['beta']}]" not in text
    assert not [e for e in memory_floor.floor_elements(snapshot) if e[0] == "F1b"]
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="reader01", current_chat_id=1)
    directory = _memory_read(ctx, room_id="*", start=0)
    assert f"chat_id={rooms['beta']}]" in directory and "777" in directory
    assert "beta reply" in _memory_read(ctx, room_id=str(rooms["beta"]), rows=True)
    assert mv.snapshot_from_json(mv.snapshot_json(snapshot)) == snapshot
    assert "## Live rooms" not in mv.render_room(dataclasses.replace(snapshot, spec=dataclasses.replace(snapshot.spec, live_rooms="none")))


@pytest.mark.parametrize("change,quiet", [({}, True), ({"wake_silent": False}, False),
    ({"owner_message_sent": True}, False), ({"terminal_host_notice": "host notice"}, False),
    ({"direction": "system", "type": "host_notice"}, False),
    ({"delegation_role": "subagent", "subagent_task_id": "kid1"}, False),
    ({"direction": "system", "type": "task_summary", "summary_kind": "terminal_root_projection",
      "outcome_authority": "canonical_task_result_after_finalization", "quiet_effective": True}, True),
    ({"direction": "system", "type": "task_summary", "summary_kind": "terminal_root_projection",
      "outcome_authority": "canonical_task_result_after_finalization", "quiet_effective": False}, False)])
def test_only_a_typed_quiet_finish_of_the_mind_considers_earlier_words_without_sealing(tmp_path, change, quiet):
    rooms, rows = _install(tmp_path)
    beta = str(rooms["beta"])
    finish = shared.msg(_ts(30), "silent is only text", chat_id=int(beta), direction="out", task_id="wake1",
                        root_task_id="wake1", terminal_origin="model_final", wake_silent=True)
    finish.update(change)
    shared.append(tmp_path / "logs" / "chat.jsonl", finish)
    snapshot, _text = _view(tmp_path, MAIN_TASK)
    assert (beta in {r["room_id"] for r in snapshot.live_rooms}) is not quiet
    assert _addr(rows[13]) in {chat_chain.format_address(a) for a, _row, _p in mi.open_room_rows(tmp_path, beta)}
    if quiet:
        seal(tmp_path, beta, finish)  # the quiet row being sealed never resurrects considered words
        assert beta in {r["room_id"] for r in _view(tmp_path, MAIN_TASK)[0].other_rooms}
        shared.append(tmp_path / "logs" / "chat.jsonl", shared.msg(_ts(31), "new words", chat_id=int(beta)))
        assert beta in {r["room_id"] for r in _view(tmp_path, MAIN_TASK)[0].live_rooms}


def test_running_work_is_read_from_typed_facts_and_completion_stops_holding_the_room(tmp_path):
    rooms, rows = _install(tmp_path)
    beta = str(rooms["beta"])
    seal(tmp_path, beta, rows[13])
    seed_result(tmp_path, "tb1", {"status": "running"})
    snapshot, _ = _view(tmp_path, MAIN_TASK)
    assert next(r for r in snapshot.live_rooms if r["room_id"] == beta)["facts"]["running_tasks"] == ["tb1"]
    seed_result(tmp_path, "tb1", {"status": "completed"})
    assert beta in {r["room_id"] for r in _view(tmp_path, MAIN_TASK)[0].other_rooms}
    store = ChronicleStore(tmp_path)
    store.write_note(room_id=beta, task_id="tb1", text="A thought to carry.", author=shared.MIND)
    assert beta in {r["room_id"] for r in _view(tmp_path, MAIN_TASK)[0].live_rooms}


def test_bound_running_work_keeps_an_empty_room_live_without_reading_history(tmp_path, monkeypatch):
    from ouroboros.projects_registry import bind_task_to_project
    from ouroboros import task_store

    rooms, rows = _install(tmp_path)
    beta = str(rooms["beta"])
    bind_task_to_project(tmp_path, "tb1", "beta", origin={"absent": "post_hoc_unresolved"})
    for row in rows[13:15]:
        seal(tmp_path, beta, row)
    seed_result(tmp_path, "tb1", {"status": "running"})
    _view(tmp_path, MAIN_TASK)  # load the fixture's legacy spans before guarding against new history walks
    original, reads = task_store.task_facts, []

    def facts(root, **query):
        result = original(root, **query)
        reads.append((query, set(result)))
        return result

    monkeypatch.setattr(task_store, "task_facts", facts)
    monkeypatch.setattr(task_store, "display_documents", lambda *a, **kw: pytest.fail("running needs no task body"))
    monkeypatch.setattr(chat_chain, "iter_rows", lambda *a, **kw: pytest.fail("no history walk"))
    snapshot, text = _view(tmp_path, MAIN_TASK)
    assert reads == [({"where": "status=?", "params": ("running",)}, {"tb1"})]
    room = next(r for r in snapshot.live_rooms if r["room_id"] == beta)
    assert room["facts"]["rows"] == 0 and "get_task_result(task_id='tb1')" in text
    seed_result(tmp_path, "tb1", {"status": "completed"})
    assert beta not in {r["room_id"] for r in _view(tmp_path, MAIN_TASK)[0].live_rooms}


def test_explicit_words_stay_live_after_a_quiet_finish_even_when_the_finish_is_sealed(tmp_path):
    rooms, _rows = _install(tmp_path)
    beta = str(rooms["beta"])
    finish = shared.msg(_ts(30), "No answer needed", chat_id=int(beta), direction="out", task_id="wake1",
                        root_task_id="wake1", terminal_origin="model_final", wake_silent=True)
    shared.append(tmp_path / "logs" / "chat.jsonl", finish)
    seal(tmp_path, beta, finish)
    assert beta in {room["room_id"] for room in _view(tmp_path, MAIN_TASK)[0].other_rooms}
    snapshot, text = _view(tmp_path, MAIN_TASK, live_rooms="lines_with_words")
    room = next(room for room in snapshot.live_rooms if room["room_id"] == beta)
    assert room["facts"]["unconsidered_people"] == 0 and room["words"]
    assert "beta words" in text


def test_finished_bindings_add_no_task_bodies_to_lines_or_an_inactive_directory_room(tmp_path, monkeypatch):
    from ouroboros import projects_registry, task_store

    rooms, rows = _install(tmp_path)
    beta = str(rooms["beta"])
    seal(tmp_path, beta, rows[13])
    _view(tmp_path, MAIN_TASK)  # finish legacy import before measuring view reads
    read, seen = task_store.display_documents, []

    def documents(root, task_ids):
        seen.append(set(task_ids) - {""})
        return read(root, task_ids)

    monkeypatch.setattr(task_store, "display_documents", documents)
    measured = []
    for count in (0, 40):
        for index in range(count):
            task_id = f"finished-{index}"
            projects_registry.bind_task_to_project(tmp_path, task_id, "beta", origin={"absent": "system"})
            seed_result(tmp_path, task_id, {"status": "completed", "project_id": "beta"})
        for mode in ("lines", "lines_with_words"):
            seen.clear()
            _view(tmp_path, MAIN_TASK, live_rooms=mode)
            assert not seen
        seen.clear()
        snapshot, _ = _view(tmp_path, MAIN_TASK, live_rooms="directory")
        assert beta in {room["room_id"] for room in snapshot.other_rooms}
        measured.append(seen[:])
    assert measured[0] == measured[1] and measured[0]


def test_a_gap_keeps_a_room_live_without_words_or_notes(tmp_path):
    _install(tmp_path)
    spec = mv.ViewSpec("integrator", room_id="1")
    label = lambda room, _sample=None: str(room)
    rooms = mv._live_rooms(ChronicleStore(tmp_path), tmp_path, spec, label, {}, {}, {}, {"555": {"reply_target_unreadable"}})
    assert rooms[0]["room_id"] == "555" and rooms[0]["live"]
    assert not mv._live_rooms(ChronicleStore(tmp_path), tmp_path, spec, label, {}, {}, {}, {})
