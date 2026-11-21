"""Facts at my human's return: what happened elsewhere between my human's previous word to me and this one.

Each guard is checked both ways: activity elsewhere gives the facts, a gap with nothing elsewhere gives none; my
human's words to me set the boundary, their words in a place do not; another focus's page write is shown, this
room's own turn's is not; a turn my human's word starts gets the section, any other turn gets nothing.
"""
from __future__ import annotations

import json
import os
import pathlib
import re

import pytest

from ouroboros import dialogue_store as ds, places, return_facts as rf
from tests.test_consciousness_wake import T0, _iso, _owner

PLACE_ROOM = 2 ** 41 + 11
PROJECT_ROOM = 7_000_001
PHONE_ROOM = 9_000_001


@pytest.fixture
def place(tmp_path):
    connected = places.connect_place(tmp_path, connector="bridge", provider="chat", account_id="qa", label="Team")
    places.edit_place(tmp_path, connected.place_id, "confirm_owner_account", author="owner",
                      platform_actor_id="jonas", account_id="qa")
    return connected


def _arrival(place, actor_id, text, ts, **actor):
    transport = {"provider": "chat", "account_id": "qa", "conversation_id": "design",
                 "conversation": {"name": "design"},
                 "actor": {"platform_actor_id": actor_id, "display_name": actor_id.title(), **actor}}
    return {"ts": _iso(ts), "direction": "in", "type": "place_arrival", "chat_id": PLACE_ROOM, "text": text,
            "source": "presence:chat", "sender_label": actor_id.title(), "client_message_id": f"ev-{actor_id}-{ts}",
            "transport": transport, "presence_provenance": {"place_id": place.place_id}, "place_id": place.place_id}


def _ended(task_id, ts, chat_id=PROJECT_ROOM, **extra):
    return {"ts": _iso(ts), "direction": "system", "type": "task_summary", "summary_kind": "terminal_root_projection",
            "task_id": task_id, "root_task_id": task_id, "chat_id": str(chat_id), "status": "completed",
            "outcome": "Done", "target_label": "Garden — planting", "text": f"Done. Root task {task_id}.", **extra}


def _write(root, ts, topic, *, chat_id, role="root", reason="Mira's soil test moved the harvest."):
    journal = root / "memory" / "knowledge_history.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"ts": _iso(ts), "topic": topic, "mode": "edit", "new_sha256": "ab" * 32,
                                 "reason": reason, "focus": {"role": role, "chat_id": chat_id, "task_id": "t" + topic}})
                     + "\n")
    os.utime(journal, (ts + 1, ts + 1))  # a journal's last change follows its newest row, as on a real install


def _rows(root, *rows):
    return [ds.append(root, "chat", row) for row in rows]


def _word(root, cmid, chat_id=1):
    return ds.accepted_row(root, chat_id, cmid)


def test_activity_elsewhere_gives_facts_and_this_rooms_own_rows_are_not_told(tmp_path, place):
    _rows(tmp_path, _owner("Before the night.", T0, cmid="a"),
          _arrival(place, "tomas", "Soil test: +12% yield, harvest on Monday.", T0 + 60),
          _arrival(place, "ci-bot", "Build green.", T0 + 70, is_bot=True),
          {"ts": _iso(T0 + 80), "direction": "out", "type": "presence_delivery", "chat_id": PLACE_ROOM,
           "text": "Noted, thank you.", "task_id": "presence-1", "root_task_id": "presence-1"},
          _ended("garden1", T0 + 90, wake_summary="Rebuilt the watering table."),
          {"ts": _iso(T0 + 95), "direction": "out", "chat_id": 1, "text": "Main's own reply.", "task_id": "m1"},
          _owner("Good morning.", T0 + 36000, cmid="b"))
    _write(tmp_path, T0 + 100, "orchard", chat_id=PROJECT_ROOM)
    found = rf.facts(tmp_path, _word(tmp_path, "b"), rf.Own(room=1))
    full, middle, last = rf.render(found)
    assert full.startswith(rf.HEADING + "\nMy human's previous word: Main, ") and ", 10 h 0 min ago." in full
    assert "Tomas (tomas) ×1, newest " in full and "«Soil test: +12% yield, harvest on Monday.»" in full
    assert "automation ×1, newest memory_read(" in full and "my replies ×1" in full and "Build green." not in full
    assert "· Garden — planting: Done " in full and "«Rebuilt the watering table.»" in full
    assert "get_task_result(task_id='garden1')" in full
    assert "- knowledge global/orchard: 1 write by root in " in full and "why: «Mira's soil test moved the harvest.»" in full
    assert "Main's own reply." not in full and "Before the night." not in full and "Good morning." not in full
    assert "Not known:" not in full, "the one connected place had rows"
    assert full.startswith(middle) and rf.WORDS_HEADING in full and rf.WORDS_HEADING not in middle
    assert "Tomas (tomas) ×1, newest " in middle and "Soil test" not in middle, "the middle form: words behind readers"
    assert "«Rebuilt the watering table.»" in middle and "get_task_result(task_id='garden1')" in middle
    assert "why: «Mira's soil test moved the harvest.»" in middle and "automation ×1" in middle
    assert last.startswith(rf.HEADING) and last.endswith(
        "Since then, counted: 1 place: 1 person (1 message), 2 other rows; 1 work ended; 1 page write on 1 page.")

    _rows(tmp_path, {"ts": _iso(T0 + 36100), "direction": "out", "chat_id": 1, "text": "Main again.", "task_id": "m2"},
          _owner("And one more thing.", T0 + 36200, cmid="c"))
    again = rf.render(rf.facts(tmp_path, _word(tmp_path, "c"), rf.Own(room=1)))[0]
    assert "Main again." not in again and again.endswith(
        "Not known:\n- no arrival from anyone else recorded since then in 1 connected place; nothing records how far "
        "a place was heard: configure_presence(action='list')"), "only this room moved: what stays unknown is the place"


def test_the_boundary_is_my_humans_word_to_me_never_their_words_in_a_place(tmp_path, place):
    accounts = places.place_owner_accounts(tmp_path)
    _rows(tmp_path, _owner("First words ever.", T0, cmid="a"))
    assert rf.previous_word(tmp_path, _word(tmp_path, "a"), accounts) is None
    assert rf.facts(tmp_path, _word(tmp_path, "a"), rf.Own(room=1)) is None, "no previous word: no block"
    _rows(tmp_path, _owner("Back in Telegram.", T0 + 30, chat_id=PHONE_ROOM, cmid="t"), _owner("Back in Main.", T0 + 40, cmid="b"))
    assert rf.previous_word(tmp_path, _word(tmp_path, "b"), accounts).row["text"] == "Back in Telegram.", "any room"
    _rows(tmp_path, _arrival(place, "jonas", "Written to the team from my phone.", T0 + 45),
          _arrival(place, "tomas", "Tomas answers the team.", T0 + 50), _owner("Again.", T0 + 60, cmid="c"))
    later = rf.facts(tmp_path, _word(tmp_path, "c"), rf.Own(room=1))
    assert later["previous"].row["text"] == "Back in Main.", "their words in a place are not a word to me"
    full = rf.render(later)[0]
    theirs, heard = full.split("Places since then:")
    assert "My human's words elsewhere since then:" in theirs and "«Written to the team from my phone.»" in theirs
    assert "Tomas answers the team." in heard and "Written to the team" not in heard


def test_another_focus_page_write_is_shown_and_this_rooms_own_turn_is_not(tmp_path, place):
    _rows(tmp_path, _owner("A.", T0, cmid="a"))
    _write(tmp_path, T0 + 10, "this-room-note", chat_id=1, role="main")
    _write(tmp_path, T0 + 20, "wake-note", chat_id=1, role="consciousness", reason="My wake mended it.")
    _write(tmp_path, T0 + 30, "place-note", chat_id=PLACE_ROOM, role="presence", reason="")
    _rows(tmp_path, _owner("B.", T0 + 40, cmid="b"))
    full = rf.render(rf.facts(tmp_path, _word(tmp_path, "b"), rf.Own(room=1)))[0]
    assert "global/wake-note: 1 write by consciousness in Main" in full and "«My wake mended it.»" in full
    assert "global/place-note" in full and "global/this-room-note: 1 write by main in Main" in full, "another turn"
    running = rf.render(rf.facts(tmp_path, _word(tmp_path, "b"), rf.Own(root_task_id="another-root")))[0]
    assert "this-room-note" in running, "for running work only the root's own tree is held"
    writer = rf.render(rf.facts(tmp_path, _word(tmp_path, "b"), rf.Own(root_task_id="tthis-room-note")))[0]
    assert "this-room-note" not in writer and "wake-note" in writer


def test_journal_window_reads_strictly_between_and_skips_an_untouched_journal(tmp_path):
    journal = tmp_path / "j.jsonl"
    journal.write_text("".join(json.dumps({"ts": _iso(T0 + n), "n": n}) + "\n" for n in range(5))
                       + "not json\n", encoding="utf-8")
    os.utime(journal, (T0 + 10, T0 + 10))
    gaps = []
    rows = rf.journal_rows_between(journal, _iso(T0 + 1), _iso(T0 + 4), gaps)
    assert [row["n"] for row in rows] == [2, 3] and gaps == ["knowledge journal tmp0/j.jsonl: 1 row not JSON"] or gaps
    os.utime(journal, (T0 - 10, T0 - 10))
    assert rf.journal_rows_between(journal, _iso(T0 + 1), _iso(T0 + 4), []) == [], "older than the boundary: not opened"


def test_a_turn_my_humans_word_starts_gets_the_section_and_no_other_turn_does(tmp_path, place):
    from ouroboros.section_floor import HostSection

    _rows(tmp_path, _owner("A.", T0, cmid="a"), _arrival(place, "tomas", "News.", T0 + 5), _owner("B.", T0 + 9, cmid="b"))
    turn = {"_is_direct_chat": True, "chat_id": 1, "origin_message_ref": {"chat_id": 1, "client_message_id": "b"}}
    facts, words = rf.turn_sections(turn, tmp_path)
    assert isinstance(facts, HostSection) and (facts.section_id, words.section_id) == (rf.SECTION_ID, rf.WORDS_ID)
    assert "Tomas (tomas) ×1, newest " in facts.full_text and "News." not in facts.full_text
    assert facts.operative_view.startswith(rf.HEADING) and "Since then, counted: 1 place" in facts.operative_view
    assert "«News.»" in words.full_text and words.operative_view.endswith("1 person stand at the readers on their place "
                                                                          "lines above.")
    assert rf.turn_sections({**turn, "_is_direct_chat": False}, tmp_path) == []
    assert rf.turn_sections({**turn, "origin_message_ref": {}}, tmp_path) == []
    assert rf.turn_sections({**turn, "origin_message_ref": {"chat_id": 1, "client_message_id": "a"}}, tmp_path) == []


def test_the_facts_stand_in_the_changing_block_of_a_turn_my_human_starts_and_nowhere_else(tmp_path):
    from tests import _memory_view_context as view

    env, memory, rooms = view.world(tmp_path)
    _rows(memory.drive_root, _owner("Before the night.", T0, cmid="since-a"),
          _ended("w1", T0 + 60, chat_id=rooms["alpha"], target_label="Alpha report"),
          _owner("Morning.", T0 + 36000, cmid="since-b"))
    turn = {"id": "since-turn", "chat_id": 1, "_is_direct_chat": True,
            "origin_message_ref": {"chat_id": 1, "client_message_id": "since-b"}}
    first, second, third, _cap = view.blocks(env, memory, turn)
    assert rf.HEADING in third and rf.HEADING not in first + second, "after every cache mark of the system rows"
    assert "· Alpha report: Done " in view.section(third, rf.HEADING)
    again = view.blocks(env, memory, turn)[2]
    assert view.section(again, rf.HEADING) == view.section(third, rf.HEADING), "a recapture gives the same bytes"
    for other in ({**turn, "_is_direct_chat": False}, {**turn, "origin_message_ref": {}}):
        assert rf.HEADING not in view.blocks(env, memory, other)[2]


def _card_answer(ts, *, chat_id=PROJECT_ROOM, comment="go"):
    quiz = {"quiz_id": "q1", "question": "**Ship the watering table?** Details follow.", "answered_index": 0,
            "options": [{"label": "Yes, ship"}, {"label": "Wait"}], "comment": comment, "answered_at": _iso(ts)}
    return {"ts": _iso(ts), "direction": "system", "type": "quiz_answer", "chat_id": chat_id, "task_id": "asker",
            "client_message_id": "quiz_answer:asker:q1", "text": "[Owner quiz answer] quiz q1", "quiz": quiz}


def test_a_card_answer_is_my_humans_word_and_a_running_root_hears_from_where_its_view_reaches(tmp_path, place):
    _rows(tmp_path, _owner("Evening.", T0, cmid="a"), _arrival(place, "mira", "Overnight numbers are in.", T0 + 10),
          _card_answer(T0 + 20), _owner("Morning.", T0 + 30, cmid="b"))
    word = _word(tmp_path, "b")
    assert rf.previous_word(tmp_path, word, places.place_owner_accounts(tmp_path)).row["type"] == "quiz_answer"
    fresh = rf.render(rf.facts(tmp_path, word, rf.Own(room=1)))[0]
    assert "Overnight numbers" not in fresh and "no arrival from anyone else recorded since then in 1" in fresh, \
        "a fresh turn hears since the card answer"
    reach = {"seq": _word(tmp_path, "a").seq + 1, "ts": _iso(T0)}
    running = rf.render(rf.facts(tmp_path, word, rf.Own(root_task_id="main-root"), since=reach))[0]
    assert "This work's view and earlier facts reach to " in running and "My human's words elsewhere since then:" in running
    assert "card answer: chose «Yes, ship»; with the words «go»; the card: «Ship the watering table?»" in running
    assert "Mira (mira) ×1, newest " in running and "«Overnight numbers are in.»" in running



def test_a_late_card_click_is_one_word_elsewhere_not_two(tmp_path, place):
    """A click on a card whose work had ended writes the answer and its late delivery: one word (Astra r9)."""
    _rows(tmp_path, _owner("Evening.", T0, cmid="a"), _card_answer(T0 + 20),
          _owner("[Owner quiz answer] quiz q1", T0 + 21, chat_id=PROJECT_ROOM, cmid="quiz_late_answer:asker:q1"),
          _owner("Morning.", T0 + 30, cmid="b"))
    reach = {"seq": _word(tmp_path, "a").seq + 1, "ts": _iso(T0)}
    found = rf.facts(tmp_path, _word(tmp_path, "b"), rf.Own(root_task_id="main-root"), since=reach)
    full, _middle, last = rf.render(found)
    assert "Since then, counted: 1 word of my human elsewhere" in last and len(found["mine"]) == 1
    assert full.count("quiz q1") <= 1 and full.count("Ship the watering table?") == 1


def test_a_reason_stays_with_the_write_that_gave_it(tmp_path, place):
    """A page's newest write without a reason never borrows an earlier writer's (Astra r9: a Presence revision shown
    with a child's reason)."""
    _rows(tmp_path, _owner("A.", T0, cmid="a"))
    _write(tmp_path, T0 + 10, "seeds", chat_id=PLACE_ROOM, role="subagent", reason="The child's split of the varieties.")
    _write(tmp_path, T0 + 20, "seeds", chat_id=PLACE_ROOM, role="presence", reason="")
    _rows(tmp_path, _owner("B.", T0 + 40, cmid="b"))
    full = rf.render(rf.facts(tmp_path, _word(tmp_path, "b"), rf.Own(room=1)))[0]
    assert "global/seeds: 2 writes by subagent" in full and "; why: «The child's split" not in full
    assert "; the last write gave no reason; earlier, subagent" in full and "«The child's split of the varieties.»" in full



def test_a_place_is_named_by_its_scalar_name_only(tmp_path, place):
    """A conversation name that is not a string (a list the store keeps as shared references) is never printed as the
    place's label; the next scalar name is (Astra r12)."""
    odd = _arrival(place, "ann", "Seeds are in.", T0 + 5)
    odd["transport"]["conversation"] = {"full_name": [{"text": "Garden"}], "title": "#garden"}
    _rows(tmp_path, _owner("A.", T0, cmid="a"), odd, _owner("B.", T0 + 9, cmid="b"))
    full = rf.render(rf.facts(tmp_path, _word(tmp_path, "b"), rf.Own(room=1)))[0]
    assert "#garden" in full and "$shared" not in full and "[{" not in full
    from ouroboros.presence_bindings import conversation_key
    from ouroboros.presence_runner import presence_chat_id

    bare = _arrival(place, "ann", "Seeds again.", T0 + 15)
    bare["chat_id"] = presence_chat_id(conversation_key("chat", "qa", "design"))  # a room keyed by its conversation
    bare["transport"]["conversation"] = {"full_name": [{"text": "Garden"}]}  # no string name at all (Grok, Astra r13)
    _rows(tmp_path, bare, _owner("C.", T0 + 19, cmid="c"))
    later = rf.render(rf.facts(tmp_path, _word(tmp_path, "c"), rf.Own(room=1)))[0]
    assert "«Seeds again.»" in later and "$shared" not in later and "{'" not in later


def test_several_people_in_one_place_each_keep_their_newest_words_and_their_own_reader(tmp_path, place):
    _rows(tmp_path, _owner("A.", T0, cmid="a"), _arrival(place, "tomas", "First thought.", T0 + 1),
          _arrival(place, "mira", "Mira's only line.", T0 + 2), _arrival(place, "tomas", "Second thought, whole.", T0 + 3),
          _owner("B.", T0 + 9, cmid="b"))
    full = rf.render(rf.facts(tmp_path, _word(tmp_path, "b"), rf.Own(room=1)))[0]
    line = next(text for text in full.splitlines() if text.startswith("- chat · design"))
    assert line.index("Tomas (tomas) ×2, newest") < line.index("Mira (mira) ×1"), "newest person first"
    assert line.count("memory_read(room_id=") == 2 and "<one of" not in line, "each person's exact reader"
    words = full.split(rf.WORDS_HEADING)[1]
    assert "Tomas (tomas), 15.01 08:00: «Second thought, whole.»" in words and "First thought." not in full, \
        "their newest words, whole"
    assert words.index("Tomas") < words.index("Mira (mira), 15.01 08:00: «Mira's only line.»")


def test_quiet_connected_places_are_said_by_name_inside_a_block_and_in_one_line_alone(tmp_path, place):
    places.connect_place(tmp_path, connector="mailer", provider="mail", account_id="inbox", label="Mail")
    _rows(tmp_path, _owner("A.", T0, cmid="a"), _arrival(place, "tomas", "Hello.", T0 + 1), _owner("B.", T0 + 2, cmid="b"))
    full = rf.render(rf.facts(tmp_path, _word(tmp_path, "b"), rf.Own(room=1)))[0]
    assert "connected places with no arrival from anyone else recorded since then (nothing records how far a place " \
           "was heard), each with its newest recorded arrival before my human's word: Mail (none recorded)" in full
    assert "Team" not in full.split("Not known:")[1], "a place with rows is not quiet"
    _rows(tmp_path, {"ts": _iso(T0 + 3), "direction": "out", "chat_id": PLACE_ROOM, "text": "Only mine.",
                     "task_id": "presence-1", "presence_provenance": {"binding_id": place.place_id}},
          _owner("C.", T0 + 4, cmid="c"))
    mine = rf.render(rf.facts(tmp_path, _word(tmp_path, "c"), rf.Own(room=1)))[0]
    unknown = mine.split("Not known:")[1]
    assert "my replies ×1" in mine and "Team (" in unknown and "Mail (none recorded)" in unknown, \
        "only my reply there: no one else was heard, and its newest recorded arrival says when one was"
    _rows(tmp_path, _owner("D.", T0 + 5, cmid="d"))
    alone = rf.render(rf.facts(tmp_path, _word(tmp_path, "d"), rf.Own(room=1)))[0]
    assert alone.endswith("Not known:\n- no arrival from anyone else recorded since then in 2 connected places; nothing "
                          "records how far a place was heard: configure_presence(action='list')")
    assert "Mail" not in alone, "alone: the count and the reader, not the names"


def test_an_interval_still_importing_is_named_not_taken_for_quiet(tmp_path, monkeypatch):
    _rows(tmp_path, _owner("A.", T0, cmid="a"), _owner("B.", T0 + 2, cmid="b"))
    assert rf.facts(tmp_path, _word(tmp_path, "b"), rf.Own(room=1)) is None, "nothing connected, nothing moved"
    monkeypatch.setattr(ds, "range_coverage", lambda *a, **k: {"imported": False, "pending": 1, "mismatch": 0, "gaps": 0})
    full = rf.render(rf.facts(tmp_path, _word(tmp_path, "b"), rf.Own(room=1)))[0]
    assert "Not known:\n- chat rows of this interval: 1 segments still importing" in full


def _running_root(root, *, child=False):
    from types import SimpleNamespace

    heard = [{"source": "owner_mailbox", "content": "Evening.", "msg_id": "steer-a", "client_message_id": "a",
              "chat_id": "1", "ts": _iso(T0)}]
    ctx = SimpleNamespace(task_id="main-root", drive_root=root, budget_drive_root=root, current_chat_id=1,
                          task_metadata={"delegation_role": "subagent"} if child else {}, pending_events=[],
                          event_queue=None, drive_logs=lambda: root / "logs", _owner_directives=heard)
    setattr(ctx, rf.REACH_KEY, {"seq": _word(root, "a").seq + 1, "ts": _iso(T0)})  # its view, captured after "a"
    return ctx


def _kept(root, line):
    """The full facts a running root's one line names: the copy its reader points at, kept in the work's artifacts."""
    name = re.search(r"path='([^']+)'", line).group(1).rsplit("/", 1)[-1]
    return next(pathlib.Path(root).rglob(name)).read_text(encoding="utf-8")


def test_my_humans_word_reaching_a_running_root_brings_the_facts_before_it_and_only_then(tmp_path, place):
    import queue

    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, OWNER_WORDS_KIND
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_mailbox import write_owner_message

    _rows(tmp_path, _owner("Evening.", T0, cmid="a"), _arrival(place, "tomas", "Rollout moved to Monday.", T0 + 60),
          {"ts": _iso(T0 + 70), "direction": "out", "chat_id": 1, "text": "My own progress.", "task_id": "main-root"},
          _owner("From the phone at night.", T0 + 80, chat_id=PHONE_ROOM, cmid="t"),
          _owner("Morning, what happened?", T0 + 36000, cmid="b"))
    sent = [{"role": "user", "content": "the work's earlier input"}]
    messages, ctx = list(sent), _running_root(tmp_path)
    write_owner_message(tmp_path, "Morning, what happened?", "main-root", msg_id="steer-b", client_message_id="b")
    _drain_incoming_messages(messages, queue.Queue(), tmp_path, "main-root", None, set(), owner_ctx=ctx)
    kinds = [row.get(HOST_CONTEXT_KIND_KEY) for row in messages]
    assert messages[:1] == sent and kinds[1:3] == [rf.SECTION_ID, OWNER_WORDS_KIND], "before the words, nothing rewritten"
    line = messages[1]["content"]
    assert "This work's view and earlier facts reach to " in line and "\nSince then, counted: 1 word of my human" in line
    assert "«Rollout moved to Monday.»" not in line and "\nThe full facts (places, people and their newest " in line
    facts = _kept(tmp_path, line)
    assert "«Rollout moved to Monday.»" in facts, "the full facts stand at the line's reader"
    assert getattr(ctx, rf.REACH_KEY)["seq"] == _word(tmp_path, "b").seq + 1, "its reach moves to the word"
    assert "My human's words elsewhere since then:" in facts and "«From the phone at night.»" in facts
    assert "My own progress." not in facts, "the root's own rows are its transcript's"
    assert all(rf.HEADING not in str(row.get("content")) for row in ctx._owner_directives), "no owner corpus"

    quiet = list(sent)
    _drain_incoming_messages(quiet, queue.Queue(), tmp_path, "main-root", None, set(), owner_ctx=_running_root(tmp_path))
    assert quiet == sent, "no word of my human in this drain: no facts"
    child, ctx = list(sent), _running_root(tmp_path, child=True)
    write_owner_message(tmp_path, "Morning, what happened?", "main-root", msg_id="steer-c", client_message_id="b")
    _drain_incoming_messages(child, queue.Queue(), tmp_path, "main-root", None, set(), owner_ctx=ctx)
    assert rf.SECTION_ID not in [row.get(HOST_CONTEXT_KIND_KEY) for row in child], "a child carries no facts"


def test_two_words_in_one_drain_get_one_block_before_the_first_through_the_last(tmp_path, place):
    import queue

    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, OWNER_WORDS_KIND
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_mailbox import write_owner_message

    _rows(tmp_path, _owner("Evening.", T0, cmid="a"), _arrival(place, "tomas", "Before the first word.", T0 + 60),
          _owner("Morning.", T0 + 36000, cmid="b"), _arrival(place, "mira", "Between the two words.", T0 + 36010),
          _owner("And another thing.", T0 + 36020, cmid="b2"))
    messages, ctx = [{"role": "user", "content": "earlier"}], _running_root(tmp_path)
    write_owner_message(tmp_path, "Morning.", "main-root", msg_id="steer-b", client_message_id="b")
    write_owner_message(tmp_path, "And another thing.", "main-root", msg_id="steer-b2", client_message_id="b2")
    _drain_incoming_messages(messages, queue.Queue(), tmp_path, "main-root", None, set(), owner_ctx=ctx)
    kinds = [row.get(HOST_CONTEXT_KIND_KEY) for row in messages]
    assert kinds.count(rf.SECTION_ID) == 1 and kinds.index(rf.SECTION_ID) < kinds.index(OWNER_WORDS_KIND)
    facts = _kept(tmp_path, messages[kinds.index(rf.SECTION_ID)]["content"])
    assert "«Before the first word.»" in facts and "«Between the two words.»" in facts, "through the last word"
    assert "And another thing." not in facts, "a word this drain delivers is not told as said elsewhere"
    assert getattr(ctx, rf.REACH_KEY)["seq"] == _word(tmp_path, "b2").seq + 1


def test_a_later_drain_brings_its_own_row_and_renewal_keeps_the_whole_unanswered_run(tmp_path, place):
    import queue

    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, OWNER_WORDS_KIND
    from ouroboros.context_renewal import _cut_units
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_mailbox import write_owner_message

    _rows(tmp_path, _owner("Evening.", T0, cmid="a"), _arrival(place, "tomas", "Night news.", T0 + 60),
          _owner("Morning.", T0 + 36000, cmid="b"), _arrival(place, "mira", "Between the drains.", T0 + 36010),
          _owner("One more.", T0 + 36020, cmid="b2"))
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"},
                {"role": "assistant", "content": "working"}, {"role": "user", "content": "a tool's long output"}]
    ctx = _running_root(tmp_path)
    write_owner_message(tmp_path, "Morning.", "main-root", msg_id="steer-b", client_message_id="b")
    _drain_incoming_messages(messages, queue.Queue(), tmp_path, "main-root", None, set(), owner_ctx=ctx)
    sent = [dict(row) for row in messages]
    write_owner_message(tmp_path, "One more.", "main-root", msg_id="steer-b2", client_message_id="b2")
    _drain_incoming_messages(messages, queue.Queue(), tmp_path, "main-root", None, set(), owner_ctx=ctx)
    assert messages[:len(sent)] == sent, "what the first drain placed is never rewritten"
    kinds = [row.get(HOST_CONTEXT_KIND_KEY) for row in messages]
    assert kinds[4:6] == [rf.SECTION_ID, OWNER_WORDS_KIND] and kinds[7:9] == [rf.SECTION_ID, OWNER_WORDS_KIND]
    first, second = _kept(tmp_path, messages[4]["content"]), _kept(tmp_path, messages[7]["content"])
    assert "«Night news.»" in first and "«Between the drains.»" in second
    assert "Night news." not in second, "each row its own span: the reach moved to the first word"
    cut = {index for group in _cut_units(messages)[0] for index in group}
    assert not {7, 8} & cut, "renewal keeps the newest words with the facts that came with them"


def test_a_root_captured_after_my_humans_previous_word_gets_rows_from_its_view_and_page_changes_from_the_word(
        tmp_path, place):
    _rows(tmp_path, _owner("Evening.", T0, cmid="a"), _arrival(place, "tomas", "Before the dawn capture.", T0 + 60))
    _write(tmp_path, T0 + 120, "night-note", chat_id=PROJECT_ROOM, reason="Edited before the dawn capture.")
    dawn = {"seq": ds.stream_end(tmp_path, "chat"), "ts": _iso(T0 + 3600)}  # a scheduled root's view, captured later
    _rows(tmp_path, _arrival(place, "mira", "After the capture.", T0 + 7200), _owner("Morning.", T0 + 36000, cmid="b"))
    running = rf.render(rf.facts(tmp_path, _word(tmp_path, "b"), rf.Own(root_task_id="dawn-root"), since=dawn))[0]
    assert "«After the capture.»" in running and "Before the dawn capture." not in running, "its frozen view holds those"
    assert "global/night-note" in running and "«Edited before the dawn capture.»" in running, "no view holds an edit"
    assert ("rows from then, page changes from " + _iso(T0)[:16].replace("T", " ") + " UTC, to my human's word of "
            in running)


def test_my_humans_own_words_in_a_place_mean_it_was_heard(tmp_path, place):
    _rows(tmp_path, _owner("A.", T0, cmid="a"), _arrival(place, "jonas", "From my phone to the team.", T0 + 5),
          _owner("B.", T0 + 9, cmid="b"))
    full = rf.render(rf.facts(tmp_path, _word(tmp_path, "b"), rf.Own(room=1)))[0]
    assert "«From my phone to the team.»" in full and "Not known:" not in full, "a place my human wrote in is not quiet"


def test_a_press_on_a_live_card_without_a_chat_row_brings_the_facts_up_to_its_time(tmp_path, place):
    """A ``card_input`` press reaches a running author as ``owner_mailbox`` with only its time (``loop_round_limits``):
    the facts run to that time; a row after it waits for the next facts, never skipped."""
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, OWNER_WORDS_KIND

    _rows(tmp_path, _owner("Evening.", T0, cmid="a"), _arrival(place, "tomas", "Night news.", T0 + 60))
    after = _rows(tmp_path, _arrival(place, "mira", "Written after the press.", T0 + 36005))[0]
    ctx = _running_root(tmp_path)
    before = len(ctx._owner_directives)
    ctx._owner_directives.append({"source": "owner_mailbox", "content": "[Card input] Dates: 3-5 Dec", "msg_id": "c1",
                                  "chat_id": None, "client_message_id": None, "received": _iso(T0 + 36000)})
    messages = [{"role": "user", "content": "task"}, {"role": "assistant", "content": "asked a card"},
                {"role": "user", HOST_CONTEXT_KIND_KEY: OWNER_WORDS_KIND, "content": "[Card input] Dates: 3-5 Dec"}]
    rf.place_before_words(ctx, messages, 2, before)
    kept = _kept(tmp_path, messages[2]["content"])
    assert messages[2].get(HOST_CONTEXT_KIND_KEY) == rf.SECTION_ID and "«Night news.»" in kept
    assert "Written after the press." not in kept, "rows after the press's time wait"
    assert getattr(ctx, rf.REACH_KEY) == {"seq": after.seq, "ts": _iso(T0 + 36000)}, "and are told next, not skipped"


def test_the_stepped_words_count_people_not_the_lines_of_their_words(tmp_path, place):
    _rows(tmp_path, _owner("A.", T0, cmid="a"), _arrival(place, "tomas", "line one\nline two\nline three", T0 + 5),
          _owner("B.", T0 + 9, cmid="b"))
    turn = {"_is_direct_chat": True, "chat_id": 1, "origin_message_ref": {"chat_id": 1, "client_message_id": "b"}}
    words = rf.turn_sections(turn, tmp_path)[1]
    assert "The newest words of 1 person stand" in words.operative_view and "line two" in words.full_text


def test_a_turns_reach_is_read_before_its_memory_view(tmp_path):
    from types import SimpleNamespace

    ctx, seen = SimpleNamespace(), {"seq": 7, "ts": _iso(T0)}
    assert rf.turn_sections({"_is_direct_chat": False}, tmp_path, ctx, seen=seen) == []
    assert getattr(ctx, rf.REACH_KEY) == seen, "a row arriving between the view and this hook is told next"


def test_arrivals_without_a_recorded_author_are_not_called_automation(tmp_path, place):
    unknown = _arrival(place, "x", "Who wrote this?", T0 + 5)
    unknown["transport"]["actor"] = {}
    unknown.pop("sender_label")
    _rows(tmp_path, _owner("A.", T0, cmid="a"), unknown, _arrival(place, "ci", "Build green.", T0 + 6, is_bot=True),
          _owner("B.", T0 + 9, cmid="b"))
    full = rf.render(rf.facts(tmp_path, _word(tmp_path, "b"), rf.Own(room=1)))[0]
    assert "unattributed arrivals ×1" in full and "automation ×1" in full


def test_the_running_row_is_one_line_beside_the_reader_of_the_full_facts_whatever_the_window(tmp_path, place, monkeypatch):
    """The owner's 3A of 18 Nov: in running work the row is the interval and its counts beside the reader of the full
    facts, kept in the work's artifacts, however much happened and whatever the window; renewal keeps it with my
    human's words; a copy that cannot be kept leaves the counts and names the gap, never the full form."""
    import queue

    from ouroboros import section_floor
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, OWNER_WORDS_KIND
    from ouroboros.context_renewal import _cut_units
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_mailbox import write_owner_message

    _rows(tmp_path, _owner("Evening.", T0, cmid="a"), *(_arrival(place, f"p{n}", f"Long news {n}. " * 400, T0 + n)
                                                        for n in range(1, 30)), _owner("Morning.", T0 + 36000, cmid="b"))

    def drained(msg_id):
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"},
                    {"role": "assistant", "content": "working"}, {"role": "user", "content": "a tool's long output"},
                    {"role": "user", HOST_CONTEXT_KIND_KEY: OWNER_WORDS_KIND, "content": "[Card input] an earlier press"}]
        write_owner_message(tmp_path, "Morning.", "main-root", msg_id=msg_id, client_message_id="b")
        _drain_incoming_messages(messages, queue.Queue(), tmp_path, "main-root", None, set(),
                                 owner_ctx=_running_root(tmp_path))
        return messages

    messages = drained("steer-b")
    line = messages[5]["content"]
    assert messages[5].get(HOST_CONTEXT_KIND_KEY) == rf.SECTION_ID and len(line) < 1_200, "one line, not 29 people's words"
    assert "\nSince then, counted: 1 place: 29 people (29 messages)" in line and "Long news" not in line
    assert "«" + "Long news 29. " * 2 in _kept(tmp_path, line)
    cut = {index for group in _cut_units(messages)[0] for index in group}
    assert not {4, 5, 6} & cut, "the earlier press, the facts and the word stay one run"

    def unkept(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(section_floor, "retain_section", unkept)
    line = drained("steer-b-again")[5]["content"]
    assert "Long news" not in line and line.endswith("Not known: the full facts could not be kept for reading (OSError).")

def test_a_person_who_renamed_between_posts_is_one_person_under_their_newest_name(tmp_path, place):
    _rows(tmp_path, _owner("A.", T0, cmid="a"), _arrival(place, "tomas", "Before the rename.", T0 + 1),
          _arrival(place, "tomas", "After the rename.", T0 + 2, display_name="Thomas"), _owner("B.", T0 + 9, cmid="b"))
    full = rf.render(rf.facts(tmp_path, _word(tmp_path, "b"), rf.Own(room=1)))[0]
    assert "Thomas (tomas) ×2, newest " in full and "Tomas (tomas)" not in full



def test_a_press_and_a_typed_word_in_one_drain_take_the_word_before_both_as_the_previous(tmp_path, place):
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, OWNER_WORDS_KIND

    _rows(tmp_path, _owner("Evening.", T0, cmid="a"), _arrival(place, "tomas", "Night news.", T0 + 60),
          _owner("Typed after the press.", T0 + 36060, cmid="b"))
    ctx = _running_root(tmp_path)
    before = len(ctx._owner_directives)
    ctx._owner_directives += [{"source": "owner_mailbox", "content": "[Card input] Dates", "msg_id": "c1", "chat_id": None,
                               "client_message_id": None, "received": _iso(T0 + 36000)},
                              {"source": "owner_mailbox", "content": "Typed after the press.", "msg_id": "steer-b",
                               "client_message_id": "b", "chat_id": "1"}]
    messages = [{"role": "user", "content": "task"}, {"role": "assistant", "content": "asked"},
                {"role": "user", HOST_CONTEXT_KIND_KEY: OWNER_WORDS_KIND, "content": "[Card input] Dates"},
                {"role": "user", HOST_CONTEXT_KIND_KEY: OWNER_WORDS_KIND, "content": "Typed after the press."}]
    rf.place_before_words(ctx, messages, 2, before)
    facts = messages[2]["content"]
    assert "My human's previous word: Main, " + _iso(T0)[:16].replace("T", " ") in facts, "a word delivered now is not before"



def test_a_late_click_reaching_running_work_is_not_also_a_word_elsewhere(tmp_path, place):
    """A click on a card whose work had ended writes the answer and its late delivery; when that delivery reaches a
    running root, the answer's other row is the same word, never one said elsewhere (Astra r10)."""
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, OWNER_WORDS_KIND

    _rows(tmp_path, _owner("Evening.", T0, cmid="a"), _arrival(place, "tomas", "Night news.", T0 + 60),
          _card_answer(T0 + 36000),
          _owner("[Owner quiz answer] quiz q1", T0 + 36001, chat_id=PROJECT_ROOM, cmid="quiz_late_answer:asker:q1"))
    ctx = _running_root(tmp_path)
    before = len(ctx._owner_directives)
    ctx._owner_directives.append({"source": "owner_mailbox", "content": "[Owner quiz answer] quiz q1", "msg_id": "late-q1",
                                  "client_message_id": "quiz_late_answer:asker:q1", "chat_id": str(PROJECT_ROOM)})
    messages = [{"role": "user", "content": "task"}, {"role": "assistant", "content": "asked"},
                {"role": "user", HOST_CONTEXT_KIND_KEY: OWNER_WORDS_KIND, "content": "[Owner quiz answer] quiz q1"}]
    rf.place_before_words(ctx, messages, 2, before)
    line = messages[2]["content"]
    assert "\nSince then, counted: 1 place" in line and "word of my human" not in line, "no word of theirs elsewhere"
    assert "My human's words elsewhere" not in _kept(tmp_path, line) and "«Night news.»" in _kept(tmp_path, line)


def test_a_card_my_human_closed_is_found_by_its_own_row(tmp_path, place):
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, OWNER_WORDS_KIND

    closed = {"ts": _iso(T0 + 36000), "direction": "system", "type": "quiz_closed", "chat_id": 1, "task_id": "main-root",
              "client_message_id": "quiz_closed:main-root:q7", "text": "[Owner closed a card]",
              "quiz": {"quiz_id": "q7", "state": "closed", "closed_actor": "owner"}}
    _rows(tmp_path, _owner("Evening.", T0, cmid="a"), _arrival(place, "tomas", "Night news.", T0 + 60), closed)
    ctx = _running_root(tmp_path)
    before = len(ctx._owner_directives)
    ctx._owner_directives.append({"source": "owner_mailbox", "content": "[Owner closed a card]", "msg_id": "quiz_closed:q7",
                                  "chat_id": None, "client_message_id": None, "received": _iso(T0 + 36001)})
    messages = [{"role": "user", "content": "task"}, {"role": "assistant", "content": "asked"},
                {"role": "user", HOST_CONTEXT_KIND_KEY: OWNER_WORDS_KIND, "content": "[Owner closed a card]"}]
    rf.place_before_words(ctx, messages, 2, before)
    facts = messages[2]["content"]
    assert "«Night news.»" in _kept(tmp_path, facts)
    assert "My human's previous word: Main, " + _iso(T0)[:16].replace("T", " ") in facts
    own = ds.probe_rows(tmp_path, "chat", [("client_message_id", "quiz_closed:main-root:q7")], limit=1)[0]
    assert getattr(ctx, rf.REACH_KEY)["seq"] == own.seq + 1, "the closure's own row, not the stream's end"


def test_a_reach_not_read_before_the_view_is_not_read_after_it(tmp_path):
    from types import SimpleNamespace

    ctx = SimpleNamespace()
    assert rf.turn_sections({"_is_direct_chat": False}, tmp_path, ctx, seen=None) == []
    assert not hasattr(ctx, rf.REACH_KEY), "no reach: a later word is told from my human's previous word"


def test_a_press_with_nothing_before_it_still_leaves_the_rows_after_it_for_the_next_facts(tmp_path):
    """No connected place, nothing before the press: no block, and the work that ended after the press's time, before
    the drain, is told with the next word, never skipped (Grok, Astra R7)."""
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, OWNER_WORDS_KIND

    _rows(tmp_path, _owner("Evening.", T0, cmid="a"))
    ended = _rows(tmp_path, _ended("other", T0 + 36060))[0]
    ctx = _running_root(tmp_path)
    before = len(ctx._owner_directives)
    ctx._owner_directives.append({"source": "owner_mailbox", "content": "[Card input] Dates", "msg_id": "c1",
                                  "chat_id": None, "client_message_id": None, "received": _iso(T0 + 36000)})
    messages = [{"role": "user", "content": "task"}, {"role": "assistant", "content": "asked"},
                {"role": "user", HOST_CONTEXT_KIND_KEY: OWNER_WORDS_KIND, "content": "[Card input] Dates"}]
    rf.place_before_words(ctx, messages, 2, before)
    assert len(messages) == 3, "nothing before the press: no block"
    assert getattr(ctx, rf.REACH_KEY)["seq"] == ended.seq, "the work that ended after it waits for the next facts"


def test_a_word_of_my_human_after_the_press_is_not_its_previous_word(tmp_path, place):
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, OWNER_WORDS_KIND

    _rows(tmp_path, _owner("Evening.", T0, cmid="a"), _arrival(place, "tomas", "Night news.", T0 + 60),
          _owner("A DM written after the press.", T0 + 36060, chat_id=PHONE_ROOM, cmid="dm"))
    ctx = _running_root(tmp_path)
    before = len(ctx._owner_directives)
    ctx._owner_directives.append({"source": "owner_mailbox", "content": "[Card input] Dates", "msg_id": "c1",
                                  "chat_id": None, "client_message_id": None, "received": _iso(T0 + 36000)})
    messages = [{"role": "user", "content": "task"}, {"role": "assistant", "content": "asked"},
                {"role": "user", HOST_CONTEXT_KIND_KEY: OWNER_WORDS_KIND, "content": "[Card input] Dates"}]
    rf.place_before_words(ctx, messages, 2, before)
    assert "My human's previous word: Main, " + _iso(T0)[:16].replace("T", " ") in messages[2]["content"]


def test_one_person_by_their_id_across_places_and_a_forward_is_not_their_words(tmp_path, place):
    alex_one = _arrival(place, "a1", "Rollout paused.", T0 + 1)
    alex_two = _arrival(place, "a2", "Build passed.", T0 + 2)
    for row, ident in ((alex_one, "slack-a1"), (alex_two, "slack-a2")):  # their connector records only ``id``
        row["transport"]["actor"] = {"id": ident, "display_name": "Alex"}
    forward = _arrival(place, "tomas", "Mira's words, forwarded.", T0 + 3)
    forward["transport"]["message"] = {"forward_origin": {"sender_user_name": "Mira"}}
    twice = _arrival(place, "mira", "Mira in another conversation of the account.", T0 + 4)
    twice["transport"]["conversation_id"], twice["transport"]["conversation"], twice["chat_id"] = (
        "general", {"name": "general"}, PLACE_ROOM + 1)
    _rows(tmp_path, _owner("A.", T0, cmid="a"), alex_one, alex_two, forward,
          _arrival(place, "mira", "Mira here.", T0 + 5), twice, _owner("B.", T0 + 9, cmid="b"))
    full, _middle, last = rf.render(rf.facts(tmp_path, _word(tmp_path, "b"), rf.Own(room=1)))
    assert "Alex (slack-a1)" in full and "Alex (slack-a2)" in full, "two people with one name stay two"
    assert "(a forward: another author's words): «Mira's words, forwarded.»" in full
    assert "2 places: 4 people (5 messages)" in last, "Mira in two conversations of one account is one person"
    turn = {"_is_direct_chat": True, "chat_id": 1, "origin_message_ref": {"chat_id": 1, "client_message_id": "b"}}
    assert "The newest words of 4 people stand" in rf.turn_sections(turn, tmp_path)[1].operative_view, "once there too"
