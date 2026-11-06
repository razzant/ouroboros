"""What holds a room in my other conversations and what considers its words: open rows, notes, running work and
gaps hold it; an act of mine there -- my own words, a quiet finish, a note -- considers what came before, a finish
that wrote its closing sentence retells it; never a time, never another room's act (CONSENSUS_v3 3.4)."""
import dataclasses

import pytest

from ouroboros import chat_chain, memory_floor, memory_view as mv, memory_view_conversations as conversations
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


def _room(snapshot, room):
    return next((item for group in snapshot.conversations for item in group["rooms"] if item["room_id"] == room), None)


def _rank(snapshot, room, row):
    return next(word["rank"] for word in _room(snapshot, room)["words"] if word["address"] == _addr(row))


def test_every_room_with_open_rows_stays_in_the_block_and_the_directory_reads_each_by_conversation(tmp_path):
    rooms, rows = _install(tmp_path)
    for room, row in ((str(rooms["beta"]), rows[13]), ("777", rows[18])):
        seal(tmp_path, room, row)
    snapshot, text = _view(tmp_path, MAIN_TASK)
    assert {room["room_id"] for group in snapshot.conversations for room in group["rooms"]} == {
        str(rooms["alpha"]), str(rooms["beta"]), "777"}  # a sealed word leaves; the room's other open rows hold it
    assert "beta words" not in text and "beta reply" in text and f"### Project Beta [chat_id={rooms['beta']}]" in text
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="reader01", current_chat_id=1)
    directory = _memory_read(ctx, room_id="*", start=0)
    assert directory.startswith("Open rooms: 4 in 4 conversations")  # Main too: the reader excludes no room
    assert f"chat_id={rooms['beta']}]" in directory and "memory_read(room_id='777', rows=true)" in directory
    assert "beta reply" in _memory_read(ctx, room_id=str(rooms["beta"]), rows=True)
    assert mv.snapshot_from_json(mv.snapshot_json(snapshot)) == snapshot
    alone = dataclasses.replace(snapshot, spec=dataclasses.replace(snapshot.spec, other_conversations=False))
    assert "## Other conversations" not in mv.render_room(alone) and "## Other conversations" in text


@pytest.mark.parametrize("change,quiet,mine", [({}, True, False), ({"wake_silent": False}, False, True),
    ({"owner_message_sent": True}, False, True), ({"terminal_host_notice": "host notice"}, False, True),
    ({"direction": "system", "type": "host_notice"}, False, False),
    ({"delegation_role": "subagent", "subagent_task_id": "kid1"}, False, False),
    ({"direction": "system", "type": "task_summary", "summary_kind": "terminal_root_projection",
      "outcome_authority": "canonical_task_result_after_finalization", "quiet_effective": True}, True, False),
    ({"direction": "system", "type": "task_summary", "summary_kind": "terminal_root_projection",
      "outcome_authority": "canonical_task_result_after_finalization", "quiet_effective": False}, False, False)])
def test_a_quiet_finish_or_my_own_word_considers_earlier_words_and_a_host_or_child_row_does_not(tmp_path, change, quiet,
                                                                                                 mine):
    """A typed quiet finish of the mind considers what came before it, as my own words there do (a finish that
    spoke: its row is my reply); a host notice, a child's report or a canonical row without quiet does not."""
    rooms, _rows = _install(tmp_path)
    beta = str(rooms["beta"])
    late = shared.msg(_ts(29), "late beta words", chat_id=int(beta))
    finish = shared.msg(_ts(30), "silent is only text", chat_id=int(beta), direction="out", task_id="wake1",
                        root_task_id="wake1", terminal_origin="model_final", wake_silent=True)
    finish.update(change)
    shared.append(tmp_path / "logs" / "chat.jsonl", late, finish)
    snapshot, text = _view(tmp_path, MAIN_TASK)
    assert _rank(snapshot, beta, late) == (1 if quiet or mine else 2) and "late beta words" in text  # whole either way
    if quiet:
        seal(tmp_path, beta, finish)  # the quiet row sealed still considers the words before it
        assert _rank(_view(tmp_path, MAIN_TASK)[0], beta, late) == 1
        newer = shared.msg(_ts(31), "new words", chat_id=int(beta))
        shared.append(tmp_path / "logs" / "chat.jsonl", newer)
        assert _rank(_view(tmp_path, MAIN_TASK)[0], beta, newer) == 2  # nothing considered what came after


def test_my_answer_a_note_and_a_closing_sentence_each_move_the_point_and_time_or_another_room_never_do(tmp_path):
    rooms, rows = _install(tmp_path)
    alpha, beta = str(rooms["alpha"]), str(rooms["beta"])
    late = shared.msg(_ts(29), "late beta words", chat_id=int(beta))
    shared.append(tmp_path / "logs" / "chat.jsonl", late)
    assert _rank(_view(tmp_path, MAIN_TASK)[0], beta, late) == 2  # after my answer there: nobody considered it
    assert _rank(_view(tmp_path, MAIN_TASK)[0], beta, rows[13]) == 1  # before it: my answer considered it
    other = shared.msg(_ts(30), "", chat_id=int(alpha), direction="out", task_id="w2", root_task_id="w2",
                       terminal_origin="model_final", wake_silent=True)
    shared.append(tmp_path / "logs" / "chat.jsonl", other)  # another room's quiet finish, and time passing
    assert _rank(_view(tmp_path, MAIN_TASK)[0], beta, late) == 2
    ChronicleStore(tmp_path).write_note(room_id=beta, task_id="tb1", text="Beta asked late.", author=shared.MIND)
    assert _rank(_view(tmp_path, MAIN_TASK)[0], beta, late) == 1  # a note I wrote there after it
    retold = shared.msg(_ts(32), "Done.", chat_id=int(beta), direction="system", type="task_summary", task_id="tb2",
                        root_task_id="tb2", summary_kind="terminal_root_projection", status="completed",
                        wake_summary="Answered Beta's late question.")
    shared.append(tmp_path / "logs" / "chat.jsonl", retold)
    snapshot, text = _view(tmp_path, MAIN_TASK)
    assert _rank(snapshot, beta, late) == 0 and _rank(snapshot, beta, rows[13]) == 0  # its sentence retells them
    assert "completed; «Answered Beta's late question.»; rows 1; get_task_result(task_id='tb2')" in text


def test_running_work_names_itself_in_its_conversation_and_its_completion_takes_the_line_away(tmp_path):
    from tests.test_memory_view_room import _section

    rooms, rows = _install(tmp_path)
    beta = str(rooms["beta"])
    seal(tmp_path, beta, rows[13])
    seed_result(tmp_path, "tb1", {"status": "running"})
    text = _view(tmp_path, MAIN_TASK)[1]
    assert "running get_task_result(task_id='tb1')" in _section(text, f"### Project Beta [chat_id={beta}]")
    seed_result(tmp_path, "tb1", {"status": "completed"})
    snapshot, text = _view(tmp_path, MAIN_TASK)
    assert _room(snapshot, beta) is not None and "(task tb1)" in text and "running get_task_result" not in text
    ChronicleStore(tmp_path).write_note(room_id=beta, task_id="tb1", text="A thought to carry.", author=shared.MIND)
    assert _room(_view(tmp_path, MAIN_TASK)[0], beta)["notes"]


def test_bound_running_work_holds_an_empty_room_without_reading_history_or_finished_bodies(tmp_path, monkeypatch):
    from ouroboros.projects_registry import bind_task_to_project
    from ouroboros import task_store

    rooms, rows = _install(tmp_path)
    beta = str(rooms["beta"])
    bind_task_to_project(tmp_path, "tb1", "beta", origin={"absent": "post_hoc_unresolved"})
    for index in range(20):  # finished work bound there adds no task body to read
        bind_task_to_project(tmp_path, f"finished-{index}", "beta", origin={"absent": "system"})
        seed_result(tmp_path, f"finished-{index}", {"status": "completed", "project_id": "beta"})
    for row in rows[13:15]:
        seal(tmp_path, beta, row)
    seed_result(tmp_path, "tb1", {"status": "running"})
    _view(tmp_path, MAIN_TASK)  # load the fixture's legacy spans before guarding against new history walks
    original, reads, bodies = task_store.task_facts, [], []

    def facts(root, **query):
        result = original(root, **query)
        reads.append((query, set(result)))
        return result

    read = task_store.display_documents
    monkeypatch.setattr(task_store, "task_facts", facts)
    monkeypatch.setattr(task_store, "display_documents", lambda root, ids, *a: bodies.append(set(ids)) or read(root, ids))
    monkeypatch.setattr(chat_chain, "iter_rows", lambda *a, **kw: pytest.fail("no history walk"))
    snapshot, text = _view(tmp_path, MAIN_TASK)
    assert reads == [({"where": "status=?", "params": ("running",)}, {"tb1"})]
    assert bodies == [{"tb1"}]  # the running task's focus alone
    assert _room(snapshot, beta)["facts"]["rows"] == 0 and "get_task_result(task_id='tb1')" in text
    seed_result(tmp_path, "tb1", {"status": "completed"})
    assert _room(_view(tmp_path, MAIN_TASK)[0], beta) is None


def test_a_room_where_neither_my_human_nor_i_spoke_is_one_line_and_a_room_where_i_spoke_once_keeps_its_words(tmp_path):
    from ouroboros.presence_bindings import conversation_key
    from ouroboros.presence_runner import presence_chat_id

    _install(tmp_path)
    base = {"provider": "github", "account_id": "me", "conversation_id": "repo.1"}
    rooms = {thread: presence_chat_id(conversation_key("github", "me", "repo.1", thread)) for thread in ("i1", "i2")}

    def said(thread, minute, text, **extra):
        return shared.msg(_ts(minute), text, chat_id=rooms[thread], source="presence:github",
                          transport={**base, "thread_id": thread, "actor": {"platform_actor_id": "u7"}}, **extra)

    answer = shared.msg(_ts(42), "Looking.", chat_id=rooms["i2"], direction="out", type="presence_delivery", task_id="p2",
                        source="presence:github", transport={**base, "thread_id": "i2", "delivery": {"state": "delivered"}})
    shared.append(tmp_path / "logs" / "chat.jsonl", said("i1", 40, "a dependabot bump"),
                  said("i2", 41, "can you look?"), answer, said("i2", 43, "thanks"))
    seal(tmp_path, str(rooms["i2"]), answer)
    snapshot, text = _view(tmp_path, MAIN_TASK)
    group = next(g for g in snapshot.conversations if g["key"] == "github:me:repo.1")
    silent, spoke = (_room(snapshot, str(rooms[t])) for t in ("i1", "i2"))
    assert not silent["engaged"] and not silent["words"] and "a dependabot bump" not in text
    assert silent["silent"].endswith(f"1 words nobody considered; memory_read(room_id='{rooms['i1']}', rows=true)")
    assert spoke["engaged"] and "thanks" in text  # my sealed answer there still makes it a room where I spoke
    assert [room["room_id"] for room in group["rooms"]] == [str(rooms["i1"]), str(rooms["i2"])]


def test_a_gap_holds_a_room_in_the_block_without_words_or_notes(tmp_path):
    _install(tmp_path)
    spec = mv.ViewSpec("integrator", room_id="1")
    label = lambda room, _sample=None: f"Room {room}"
    store, kw = ChronicleStore(tmp_path), {"quiet": {}, "sealed": {}, "projects": []}
    groups = conversations.capture(store, tmp_path, spec, label, {}, {}, {}, gaps={"555": {"reply_target_unreadable"}}, **kw)
    assert [room["room_id"] for group in groups for room in group["rooms"]] == ["555"]
    assert not conversations.capture(store, tmp_path, spec, label, {}, {}, {}, gaps={}, **kw)
    assert memory_floor.floor_elements(mv.MemoryViewSnapshot(spec=spec, store_status={"state": "active"}, frontier={},
                                                             conversations=tuple(groups)))
