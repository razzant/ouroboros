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
from tests._memory_view_synthetic import conversations_text
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


def test_a_task_line_in_another_conversation_reads_the_cards_word_else_its_status(tmp_path):
    """Another conversation's line for a task reads the word its card stored, as the task's own room does: a stopped
    root reads Stopped, a finished one Done; a row that stored no word keeps its lifecycle status."""
    _install(tmp_path)
    room, _said, delivered = _thread()
    ended = [shared.msg(_ts(32 + index), text, chat_id=int(room), direction="system", type="task_summary",
                        task_id=task, root_task_id=task, summary_kind="terminal_root_projection", status="completed",
                        **stored)
             for index, (task, text, stored) in enumerate((("tstop", "Stopped.", {"outcome": "Stopped"}),
                                                           ("tdone", "Done.", {"outcome": "Done"}),
                                                           ("tbare", "Done.", {})))]
    shared.append(tmp_path / "logs" / "chat.jsonl", delivered(28), *ended)
    text = _view(tmp_path, MAIN_TASK)[1]
    for task, word in (("tstop", "Stopped"), ("tdone", "Done"), ("tbare", "completed")):
        assert f"; host; task {task}] {word}; rows 1; get_task_result(task_id='{task}')" in text, text


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


def test_a_bot_event_is_one_line_and_a_room_where_i_spoke_once_keeps_its_words(tmp_path):
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
    bot = said("i1", 40, "a dependabot bump\nDETAIL NOT IN THE EVENT LINE")
    bot["transport"]["actor"]["is_bot"] = True
    shared.append(tmp_path / "logs" / "chat.jsonl", bot,
                  said("i2", 41, "can you look?"), answer, said("i2", 43, "thanks"))
    seal(tmp_path, str(rooms["i2"]), answer)
    snapshot, text = _view(tmp_path, MAIN_TASK)
    group = next(g for g in snapshot.conversations if g["key"] == "github:me:repo.1")
    silent, spoke = (_room(snapshot, str(rooms[t])) for t in ("i1", "i2"))
    assert not silent["engaged"] and not silent["words"] and "a dependabot bump" in text
    assert "DETAIL NOT IN THE EVENT LINE" not in text
    assert silent["events"]["count"] == 1 and f"memory_read(room_id='{rooms['i1']}', rows=true)" in text
    assert spoke["engaged"] and "thanks" in text  # my sealed answer there still makes it a room where I spoke
    assert [room["room_id"] for room in group["rooms"]] == [str(rooms["i1"]), str(rooms["i2"])]


def test_a_forward_from_my_humans_account_engages_the_room_and_still_waits_as_anothers_words(tmp_path):
    """A forward keeps its sender: a room where my human forwarded is one where they acted, so it keeps its words (a
    stranger's forward also records a person); the forwarded words are not my human's, so they rank as unconsidered."""
    from ouroboros import places
    from ouroboros.presence_bindings import conversation_key
    from ouroboros.presence_runner import presence_chat_id

    _install(tmp_path)
    place = places.connect_place(tmp_path, connector="bridge", provider="github", account_id="me", label="Repo")
    places.edit_place(tmp_path, place.place_id, "confirm_owner_account", author="owner", platform_actor_id="anton",
                      account_id="me")
    base = {"provider": "github", "account_id": "me", "conversation_id": "repo.1"}
    rooms = {thread: presence_chat_id(conversation_key("github", "me", "repo.1", thread)) for thread in ("f1", "f2")}

    def forwarded(thread, minute, actor):
        return shared.msg(_ts(minute), f"words {actor} forwarded", chat_id=rooms[thread], source="presence:github",
                          presence_provenance={"place_id": place.place_id},
                          transport={**base, "thread_id": thread, "actor": {"platform_actor_id": actor},
                                     "message": {"forward_origin": {"sender_user_name": "Other"}}})

    mine, theirs = forwarded("f1", 40, "anton"), forwarded("f2", 41, "u7")
    shared.append(tmp_path / "logs" / "chat.jsonl", mine, theirs)
    snapshot, text = _view(tmp_path, MAIN_TASK)
    engaged, stranger = _room(snapshot, str(rooms["f1"])), _room(snapshot, str(rooms["f2"]))
    assert engaged["engaged"] and "words anton forwarded" in text, "my human acted in this room"
    assert [word["owner_side"] for word in engaged["words"]] == [False], "the words are another author's"
    assert _rank(snapshot, str(rooms["f1"]), mine) == 2, "they wait unconsidered, as a person's words do"
    assert stranger["engaged"] and "words u7 forwarded" in text
    assert _rank(snapshot, str(rooms["f2"]), theirs) == 2 and not stranger["words"][0]["owner_side"]


@pytest.mark.parametrize("owner", [False, True])
@pytest.mark.parametrize("origins,names", [(("First Writer", "Second Writer"), ("first.zip", "second.zip")),
    (("Same Writer", "Same Writer"), ("report final.zip", "report  final.zip"))])
def test_repeated_words_keep_each_distinct_connectors_facts(tmp_path, owner, origins, names):
    """The same words with different origins or failed files are different observations; an actual repeat
    still points to the matching observation, without reading its body again."""
    from ouroboros import places
    from ouroboros.presence_bindings import conversation_key
    from ouroboros.presence_runner import presence_chat_id

    _install(tmp_path)
    place = places.connect_place(tmp_path, connector="bridge", provider="github", account_id="me", label="Repo")
    if owner:
        places.edit_place(tmp_path, place.place_id, "confirm_owner_account", author="owner",
                          platform_actor_id="u7", account_id="me")
    room = presence_chat_id(conversation_key("github", "me", "repo.1", "i1"))
    flow = {"provider": "github", "account_id": "me", "conversation_id": "repo.1", "thread_id": "i1"}
    def sent(minute, origin, file_name):
        return shared.msg(_ts(minute), "Please inspect this archive", chat_id=room, source="presence:github",
                          presence_provenance={"place_id": place.place_id},
                          transport={**flow, "actor": {"platform_actor_id": "u7"}, "message": {
                              "forward_origin": {"sender_user_name": origin},
                              "attachments": [{"file_name": file_name, "status": "too_large", "file_size": 40000000}]}})
    first = sent(40, origins[0], names[0])
    second = sent(41, origins[1], names[1])
    third = sent(42, origins[1], names[1])
    answer = shared.msg(_ts(39), "Looking.", chat_id=room, direction="out", type="presence_delivery",
                        task_id="p1", source="presence:github", transport={**flow, "delivery": {"state": "delivered"}})
    shared.append(tmp_path / "logs" / "chat.jsonl", answer, first, second, third)
    snapshot, _text = _view(tmp_path, MAIN_TASK)
    words = {word["address"]: word["line"] for word in _room(snapshot, str(room))["words"]}
    assert f'"{origins[0]}"' in words[_addr(first)] and f'"{names[0]}"' in words[_addr(first)]
    assert f'"{origins[1]}"' in words[_addr(second)] and f'"{names[1]}"' in words[_addr(second)]
    assert "delivered again" not in words[_addr(second)]
    assert "delivered again" in words[_addr(third)] and _addr(second) in words[_addr(third)]
    assert _addr(first) not in words[_addr(third)]


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
        assert f"[{_addr(rows[14])}] «Read Beta's thread; nothing owed.»: 1 words read and folded" in beta_section
        assert ("Read row addresses with memory_read(room_id=<room from address>, rows=true, "
                "from=<first>, to=<last>)") in text  # one reader for this block, the source coordinates in the row
    else:
        assert not item["folded"] and "read and folded" not in text
    assert ChronicleStore(tmp_path).sealed_row_refs(beta) == set()  # a fold seals nothing
    own, own_text = _view(tmp_path, {"id": "turn-beta", "type": "task", "chat_id": int(beta)})
    assert "beta reply" in own_text and "read and folded" not in own_text  # the room itself reads every row


@pytest.mark.parametrize("quoted", [True, False], ids=["owner-quoted", "note-only"])
def test_a_fold_read_finish_leaves_a_relay_row_whole_elsewhere_and_still_folds_my_reply(tmp_path, quoted):
    """rev_carbon REV3C: a ``fold_read`` finish folded a relay row that quotes my human's words into its sentence, so my
    other foci lost those words at a full view. A relay row stays out of the fold, like my human's own words; my
    reply in the same room still folds into the finish's sentence."""
    from ouroboros.owner_relay import owner_quote, relay_envelope
    rooms, rows = _install(tmp_path)
    beta = str(rooms["beta"])
    said = shared.msg(_ts(26), "Keep the March batch frozen until Friday.", chat_id=1, client_message_id="said-limit")
    relay = shared.msg(_ts(27), "A limit from my human for this work.", chat_id=int(beta), direction="out",
                       type="relay_entry", task_id="tb9", client_message_id="relay:limit:tb9",
                       relay={**relay_envelope("limit", [owner_quote(tmp_path, said)] if quoted else [], chat_id=1,
                                               summary="Passed on a limit."), "sender": {"task_id": "teller"}})
    finish = shared.msg(_ts(30), "Done.", chat_id=int(beta), direction="system", type="task_summary", task_id="tb9",
                        root_task_id="tb9", summary_kind="terminal_root_projection", status="completed",
                        wake_summary="Read Beta's thread; nothing owed.", fold_read=True)
    shared.append(tmp_path / "logs" / "chat.jsonl", said, relay, finish)
    snapshot, text = _view(tmp_path, MAIN_TASK)
    item = _room(snapshot, beta)
    beta_section = text[text.index(f"### Project Beta [chat_id={beta}]"):]
    assert any(word["address"] == _addr(relay) for word in item["words"])  # the relay row stands in the block
    assert ("Keep the March batch frozen until Friday." in beta_section) is quoted
    assert "beta reply" not in beta_section and item["folded"]["count"] == 1  # my reply there still folds


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
    assert silent["engaged"] and "Another revision." in text and "a question" in text
    assert [word["rank"] for word in silent["words"]] == [1, 2]  # only the person's words are unread



def test_one_row_several_rooms_hold_is_whole_once_and_my_humans_copy_keeps_its_f6_protection(tmp_path, monkeypatch):
    """My human's message in Main that a Project's binding names as its origin is one canonical row in both rooms
    (``memory_inventory.rooms_of_row``): whole once, in the room it was written in; the Project keeps its entry as one
    line naming that room and the row's reader, its text never read. The whole copy survives its group's line (F6g)
    and the count (F6a); only F7 addresses it, and it is the copy F7 and an owner-note capsule
    (``memory_floor.owner_rows``) see; the repeated-delivery fact is not a word to fold (R12).
    Equal words of another row stay whole."""
    from ouroboros import memory_inventory
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.projects_registry import bind_task_to_project, list_projects

    rooms, _rows_ = _install(tmp_path)
    ids = {int(project["chat_id"]): project["id"] for project in list_projects(tmp_path)}
    words = "Take the parser migration to Alpha; keep the old one until it lands."
    for task, room, cmid, minute in (("alpha2", rooms["alpha"], "o2", 40), ("betaroot", rooms["beta"], "b0", 39)):
        ref = build_owner_message_ref(chat_id=1, client_message_id=cmid, ts=_ts(minute), text=words)
        bind_task_to_project(tmp_path, task, ids[room], origin={"ref": ref, "text": words})
    origin = shared.msg(_ts(40), words, client_message_id="o2")
    again = shared.msg(_ts(42), words, client_message_id="o3")  # the same words, another row: not a copy
    shared.append(tmp_path / "logs" / "chat.jsonl", origin, again)
    alpha, address = str(rooms["alpha"]), _addr(origin)
    assert memory_inventory.rooms_of_row(memory_inventory.row_meta(origin), memory_inventory.membership_facts(
        tmp_path)) == {"1", alpha}
    asked, read = [], mv._row_texts
    monkeypatch.setattr(mv, "_row_texts", lambda root, entries: (
        asked.extend((entry[0]["row_sha256"], entry[2]) for entry in entries), read(root, entries))[1])
    snapshot, _text = _view(tmp_path, {"id": "betaroot", "type": "task", "chat_id": rooms["beta"]})
    assert snapshot.spec.room_id == str(rooms["beta"])  # a third room: Main and Alpha are both other conversations
    copies = {item["room_id"]: word for group in snapshot.conversations for item in group["rooms"]
              for word in item["words"] if word["address"] == address}
    assert set(copies) == {"1", alpha}  # both memberships stay
    assert words in copies["1"]["line"] and not copies["1"].get("again")
    assert copies[alpha]["line"] == (f"{copies[alpha]['head']} delivered again: the same row as in room 1: {address}; "
                                     f"memory_read(room_id='1', rows=true, from='{address}', to='{address}')")
    assert copies[alpha]["owner_side"] and copies[alpha]["again"] == address  # the author stays my human
    assert sum(sha == chat_chain.row_address(origin)["row_sha256"] for sha, _pos in asked) == 1  # read once
    second = [word for group in snapshot.conversations for item in group["rooms"] for word in item["words"]
              if word["address"] == _addr(again)]
    assert second and all(words in word["line"] and not word.get("again") for word in second)
    steps = {step: [ident for ident, *_ in elements] for step, elements in conversations.floor_steps(
        snapshot.conversations).items()}
    assert f"1:{address}" in steps["F7"] and f"{alpha}:{address}" not in steps["F7"]
    assert f"{alpha}:{address}" not in steps["F6w"] and f"1:{address}" not in steps["F6w"]
    assert copies[alpha]["line"] in conversations_text(snapshot.conversations, {"F6w": list(steps["F6w"])})
    assert set(memory_floor.owner_rows(snapshot)) & {f"1:{address}", f"{alpha}:{address}"} == {f"1:{address}"}
    keys = [group["key"] for group in snapshot.conversations]
    for gone in ({"F6g": ["main"]}, {"F6g": keys}, {"F6g": keys, "F6a": ["conversations"]},
                 {"F6w": list(steps["F6w"]), "F6g": keys}):  # the group holding it lined, every group, the count
        shown = conversations_text(snapshot.conversations, gone)
        assert shown.count(f"{words}") == 1 + len(second), gone  # once, besides the other row's own copies
    folded = conversations_text(snapshot.conversations, {"F7": [f"1:{address}"]})
    assert folded.count(words) == len(second) and address in folded  # F7 alone turns it into its address
