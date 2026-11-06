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


def _thread():
    """A GitHub thread: its room, a person's (u7) words there and my delivery there (a room where I spoke)."""
    from ouroboros.presence_bindings import conversation_key
    from ouroboros.presence_runner import presence_chat_id

    room = presence_chat_id(conversation_key("github", "me", "repo.1", "i1"))
    base = {"provider": "github", "account_id": "me", "conversation_id": "repo.1", "thread_id": "i1"}

    def said(minute, text):
        return shared.msg(_ts(minute), text, chat_id=room, source="presence:github",
                          transport={**base, "actor": {"platform_actor_id": "u7"}})

    def delivered(minute):
        return shared.msg(_ts(minute), "Looking.", chat_id=room, direction="out", type="presence_delivery",
                          task_id="p1", source="presence:github", transport={**base, "delivery": {"state": "delivered"}})
    return str(room), said, delivered


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
    _install(tmp_path)
    room, said, delivered = _thread()
    late = said(29, "late thread words")
    finish = shared.msg(_ts(30), "silent is only text", chat_id=int(room), direction="out", task_id="wake1",
                        root_task_id="wake1", terminal_origin="model_final", wake_silent=True)
    finish.update(change)
    shared.append(tmp_path / "logs" / "chat.jsonl", delivered(28), late, finish)
    snapshot, text = _view(tmp_path, MAIN_TASK)
    assert _rank(snapshot, room, late) == (1 if quiet or mine else 2) and "late thread words" in text  # whole either way
    if quiet:
        seal(tmp_path, room, finish)  # the quiet row sealed still considers the words before it
        assert _rank(_view(tmp_path, MAIN_TASK)[0], room, late) == 1
        newer = said(31, "new words")
        shared.append(tmp_path / "logs" / "chat.jsonl", newer)
        assert _rank(_view(tmp_path, MAIN_TASK)[0], room, newer) == 2  # nothing considered what came after


def test_my_answer_a_note_and_a_closing_sentence_each_move_the_point_and_time_or_another_room_never_do(tmp_path):
    rooms, _rows = _install(tmp_path)
    alpha, beta = str(rooms["alpha"]), str(rooms["beta"])
    room, said, delivered = _thread()
    early, late, human = said(27, "early thread words"), said(29, "late thread words"), shared.msg(
        _ts(29), "late beta words", chat_id=int(beta))
    shared.append(tmp_path / "logs" / "chat.jsonl", early, delivered(28), late, human)
    snapshot = _view(tmp_path, MAIN_TASK)[0]
    assert _rank(snapshot, room, late) == 2  # after my delivery there: nobody considered it
    assert _rank(snapshot, room, early) == 1  # before it: my delivery considered it
    assert _rank(snapshot, beta, human) == 1  # my human's words wait on F7, never as words nobody considered
    other = shared.msg(_ts(30), "", chat_id=int(alpha), direction="out", task_id="w2", root_task_id="w2",
                       terminal_origin="model_final", wake_silent=True)
    shared.append(tmp_path / "logs" / "chat.jsonl", other)  # another room's quiet finish, and time passing
    assert _rank(_view(tmp_path, MAIN_TASK)[0], room, late) == 2
    ChronicleStore(tmp_path).write_note(room_id=room, task_id="tb1", text="u7 asked late.", author=shared.MIND)
    assert _rank(_view(tmp_path, MAIN_TASK)[0], room, late) == 1  # a note I wrote there after it
    retold = shared.msg(_ts(32), "Done.", chat_id=int(room), direction="system", type="task_summary", task_id="tb2",
                        root_task_id="tb2", summary_kind="terminal_root_projection", status="completed",
                        wake_summary="Answered the late question.")
    shared.append(tmp_path / "logs" / "chat.jsonl", retold)
    snapshot, text = _view(tmp_path, MAIN_TASK)
    assert _rank(snapshot, room, late) == 0 and _rank(snapshot, room, early) == 0  # its sentence retells them
    assert "completed; «Answered the late question.»; rows 1; get_task_result(task_id='tb2')" in text


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


@pytest.mark.parametrize("fold", [True, False])
def test_a_finish_that_folds_what_it_read_stands_for_it_elsewhere_and_leaves_it_whole_here(tmp_path, fold):
    """S5: in my other foci a ``fold_read`` finish's sentence and their address stand for the room's words up to it
    until a page seals them; my human's words there stay whole, nothing is sealed, the room's own view keeps every
    row, and later words stay whole. Without the flag the words stay whole too."""
    rooms, rows = _install(tmp_path)
    beta = str(rooms["beta"])
    finish = shared.msg(_ts(30), "Done.", chat_id=int(beta), direction="system", type="task_summary", task_id="tb9",
                        root_task_id="tb9", summary_kind="terminal_root_projection", status="completed",
                        wake_summary="Read Beta's thread; nothing owed.", **({"fold_read": True} if fold else {}))
    later = shared.msg(_ts(31), "one more beta question", chat_id=int(beta))
    shared.append(tmp_path / "logs" / "chat.jsonl", finish, later)
    snapshot, text = _view(tmp_path, MAIN_TASK)
    item = _room(snapshot, beta)
    beta_section = text[text.index(f"### Project Beta [chat_id={beta}]"):]
    assert "beta words" in beta_section and "one more beta question" in beta_section  # my human's, and later words
    assert ("beta reply" in beta_section) is not fold  # my reply there: folded into the sentence, or whole
    if fold:
        assert item["folded"]["count"] == 1 and _addr(rows[14]) in item["folded"]["to"]
        assert ("«Read Beta's thread; nothing owed.» (2026-09-05): 1 words up to it read and folded into that "
                f"sentence; memory_read(room_id='{beta}', rows=true, from='") in beta_section
    else:
        assert not item["folded"] and "read and folded" not in text
    assert ChronicleStore(tmp_path).sealed_row_refs(beta) == set()  # a fold seals nothing
    own, own_text = _view(tmp_path, {"id": "turn-beta", "type": "task", "chat_id": int(beta)})
    assert "beta reply" in own_text and "read and folded" not in own_text  # the room itself reads every row


def test_my_accounts_arrival_considers_nothing_and_my_delivery_there_does(tmp_path):
    """§3.4 (i): the considered point is my own outward row. An arrival from the account I send as (an operator session
    or a `gh` call publishing as me) leaves a person's earlier word nobody considered, though it keeps the room one
    where I spoke; my delivery there considers it."""
    from ouroboros.presence_bindings import conversation_key
    from ouroboros.presence_runner import presence_chat_id

    _install(tmp_path)
    room = presence_chat_id(conversation_key("github", "me", "repo.1", "i1"))
    base = {"provider": "github", "account_id": "me", "conversation_id": "repo.1", "thread_id": "i1"}
    asked = shared.msg(_ts(40), "can you look?", chat_id=room, source="presence:github",
                       transport={**base, "actor": {"platform_actor_id": "u7"}})
    mine = shared.msg(_ts(41), "A PR description my account published.", chat_id=room, source="presence:github",
                      task_id="presence-woken", transport={**base, "actor": {"platform_actor_id": "ouro"},
                                                           "identity": {"sends_as": {"id": "ouro", "kind": "bot"}}})
    shared.append(tmp_path / "logs" / "chat.jsonl", asked, mine)
    snapshot, _text = _view(tmp_path, MAIN_TASK)
    item = _room(snapshot, str(room))
    assert item["engaged"] and _rank(snapshot, str(room), asked) == 2  # my account's event is no answer of mine
    assert any(word["head"].startswith(f"[{mine['ts']}; the account I send as: ") for word in item["words"])
    answer = shared.msg(_ts(42), "Looking.", chat_id=room, direction="out", type="presence_delivery", task_id="p1",
                        source="presence:github", transport={**base, "delivery": {"state": "delivered"}})
    shared.append(tmp_path / "logs" / "chat.jsonl", answer)
    assert _rank(_view(tmp_path, MAIN_TASK)[0], str(room), asked) == 1  # my delivery there considers it


def test_a_later_delivery_of_the_same_words_is_one_line_naming_the_earlier_copy(tmp_path, monkeypatch):
    """The first remedy for a crowded view (BIBLE P1): a room's later delivery of words it already holds (the same
    text, whitespace aside) is one line, its own head, «delivered again», the earlier copy's address and reader, and
    its text is never read. A row with no recorded speaker repeats the room's earliest copy; a recorded speaker's row
    repeats only that speaker's own earlier copy, never an unrecorded one. Edited words, the same words in another
    room or from two different recorded speakers, wordless rows and my human's words stay as they are."""
    from ouroboros import chat_chain, memory_view as mv
    from ouroboros.presence_bindings import conversation_key
    from ouroboros.presence_runner import presence_chat_id

    rooms, _rows = _install(tmp_path)
    beta = str(rooms["beta"])
    room = {thread: presence_chat_id(conversation_key("github", "me", "repo.1", thread))
            for thread in ("i1", "i2", "i3", "i4")}
    base = {"provider": "github", "account_id": "me", "conversation_id": "repo.1"}

    def said(thread, minute, text, actor="u7"):
        who = {"platform_actor_id": actor} if actor else {"platform": "github", "attribution": "unknown"}
        return shared.msg(_ts(minute), text, chat_id=room[thread], source="presence:github",
                          transport={**base, "thread_id": thread, "actor": who})

    def reply(thread, minute):  # my delivery there: both threads are rooms where I spoke
        return shared.msg(_ts(minute), "Looking.", chat_id=room[thread], direction="out", type="presence_delivery",
                          task_id=f"p-{thread}", source="presence:github",
                          transport={**base, "thread_id": thread, "delivery": {"state": "delivered"}})

    body = "## Summary\nKeep the old parser; the migration costs too much."
    first, again = said("i1", 40, body), said("i1", 41, " ".join(body.split()), actor="")  # GitHub re-sends it
    edited, other = said("i1", 42, body + "\n\nMerged."), said("i1", 43, body, actor="u8")
    elsewhere, empty, empty_again = said("i2", 44, body), said("i1", 45, ""), said("i1", 46, "")
    ship, ship_again = (shared.msg(_ts(minute), "ship it", chat_id=int(beta)) for minute in (47, 48))  # my human's
    unknown, alice, bob = (said("i3", minute, "Approved.", who) for minute, who in ((51, ""), (52, "u10"), (53, "u11")))
    alice4, bob4, bob4_again = (said("i4", minute, "Approved.", who) for minute, who in ((54, "u10"), (55, "u11"),
                                                                                        (56, "u11")))
    shared.append(tmp_path / "logs" / "chat.jsonl", first, again, edited, other, elsewhere, empty, empty_again,
                  reply("i1", 49), reply("i2", 50), ship, ship_again, unknown, alice, bob, alice4, bob4, bob4_again,
                  reply("i3", 57), reply("i4", 58))
    asked, read = [], mv._row_texts
    monkeypatch.setattr(mv, "_row_texts", lambda root, entries: (
        asked.extend(entry[0]["row_sha256"] for entry in entries), read(root, entries))[1])
    snapshot, _text = _view(tmp_path, MAIN_TASK)
    words = {word["address"]: word["line"] for group in snapshot.conversations for item in group["rooms"]
             for word in item["words"]}
    earlier = _addr(first)
    assert words[_addr(again)].startswith(f"[{again['ts']}; author not recorded [provider=github")
    assert words[_addr(again)].endswith(f"] delivered again: the same words as {earlier}; memory_read(room_id="
                                        f"'{room['i1']}', rows=true, from='{earlier}', to='{earlier}')")
    assert chat_chain.row_address(again)["row_sha256"] not in asked  # its text is never read
    for row in (first, edited, other, elsewhere, empty, empty_again, ship, ship_again, unknown, alice, bob, alice4,
                bob4):  # each as it is: unknown → Alice → Bob keeps both people's words whole
        assert "delivered again" not in words[_addr(row)] and chat_chain.row_address(row)["row_sha256"] in asked
    assert "Keep the old parser" in words[_addr(first)] and "ship it" in words[_addr(ship_again)]
    bob_first = _addr(bob4)  # Alice → Bob → Bob: Bob's second copy is his own first copy's, not Alice's
    assert words[_addr(bob4_again)].endswith(f"] delivered again: the same words as {bob_first}; memory_read(room_id="
                                             f"'{room['i4']}', rows=true, from='{bob_first}', to='{bob_first}')")


def test_a_row_with_no_recorded_speaker_is_nobody_and_only_a_persons_word_waits_unconsidered(tmp_path):
    """A connector event without an actor (GitHub's later revision of a description) shows its text whole but is no
    person: not in the room's people count, not in a silent room's «words nobody considered», and it ranks with the
    considered words, as a reaction and my account's arrival do; a person's word after my last act is the word
    nobody considered, counted and ranked last."""
    from ouroboros.presence_bindings import conversation_key
    from ouroboros.presence_runner import presence_chat_id

    _install(tmp_path)
    room = {thread: presence_chat_id(conversation_key("github", "me", "repo.1", thread)) for thread in ("i1", "i2")}
    base = {"provider": "github", "account_id": "me", "conversation_id": "repo.1"}

    def said(thread, minute, text, actor="u7", **transport):
        who = {"platform_actor_id": actor} if actor else {"platform": "github", "attribution": "unknown"}
        return shared.msg(_ts(minute), text, chat_id=room[thread], source="presence:github",
                          transport={**base, "thread_id": thread, "actor": who, **transport})

    reply = shared.msg(_ts(40), "Looking.", chat_id=room["i1"], direction="out", type="presence_delivery",
                       task_id="p1", source="presence:github",
                       transport={**base, "thread_id": "i1", "delivery": {"state": "delivered"}})
    revision = said("i1", 41, "## Summary\nThe description, revised.", actor="")
    reaction = said("i1", 42, "", message={"reaction": {"schema_version": 1, "action": "added",
                                                        "values": [{"name": "eyes"}]}})
    mine = said("i1", 43, "A comment my account published.", actor="ouro",
                identity={"sends_as": {"id": "ouro", "kind": "bot"}})
    asked = said("i1", 44, "can you look?")
    shared.append(tmp_path / "logs" / "chat.jsonl", reply, revision, reaction, mine, asked,
                  said("i2", 45, "## Summary\nAnother revision.", actor=""), said("i2", 46, "a question", actor="u9"))
    snapshot, text = _view(tmp_path, MAIN_TASK)
    spoke, silent = _room(snapshot, str(room["i1"])), _room(snapshot, str(room["i2"]))
    assert [_rank(snapshot, str(room["i1"]), row) for row in (revision, reaction, mine, asked)] == [1, 1, 1, 2]
    assert "The description, revised." in text  # its text stays whole where it is shown
    assert (spoke["facts"]["people"], spoke["facts"]["unattributed"], spoke["facts"]["reactions"]) == (1, 1, 1)
    assert "people 1, mine 2, reactions 1, author not recorded 1, " in text
    assert not silent["engaged"] and "; 1 words nobody considered; " in silent["silent"]  # the person alone

