"""Facts at my human's return: what happened elsewhere between my human's previous word to me and this one.

Each guard is checked both ways: activity elsewhere gives the facts, a gap with nothing elsewhere gives none; my
human's words to me set the boundary, their words in a place do not; another focus's page write is shown, this
room's own turn's is not; a turn my human's word starts gets the section, any other turn gets nothing.
"""
from __future__ import annotations

import json
import os

import pytest

from ouroboros import dialogue_store as ds, places, return_facts as rf
from tests.test_consciousness_wake import T0, _iso, _owner

PLACE_ROOM = 2 ** 41 + 11
PROJECT_ROOM = 7_000_001


@pytest.fixture
def place(tmp_path):
    connected = places.connect_place(tmp_path, connector="bridge", provider="chat", account_id="qa", label="Team")
    places.edit_place(tmp_path, connected.place_id, "confirm_owner_account", author="owner",
                      platform_actor_id="anton", account_id="qa")
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
            "outcome": "Done", "target_label": "Nova — economics", "text": f"Done. Root task {task_id}.", **extra}


def _write(root, ts, topic, *, chat_id, role="root", reason="Nina's price test moved the rollout."):
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
          _arrival(place, "sasha", "Price test: +12% ARPU, rollout on Monday.", T0 + 60),
          _arrival(place, "ci-bot", "Build green.", T0 + 70, is_bot=True),
          {"ts": _iso(T0 + 80), "direction": "out", "type": "presence_delivery", "chat_id": PLACE_ROOM,
           "text": "Noted, thank you.", "task_id": "presence-1", "root_task_id": "presence-1"},
          _ended("nova1", T0 + 90, wake_summary="Rebuilt the economy table."),
          {"ts": _iso(T0 + 95), "direction": "out", "chat_id": 1, "text": "Main's own reply.", "task_id": "m1"},
          _owner("Good morning.", T0 + 36000, cmid="b"))
    _write(tmp_path, T0 + 100, "joi", chat_id=PROJECT_ROOM)
    found = rf.facts(tmp_path, _word(tmp_path, "b"), rf.Own(room=1))
    full, middle, last = rf.render(found)
    assert full.startswith(rf.HEADING + "\nMy human's previous word: Main, ") and ", 10 h 0 min ago." in full
    assert "Sasha (sasha) ×1, newest " in full and "«Price test: +12% ARPU, rollout on Monday.»" in full
    assert "automation ×1, newest memory_read(" in full and "my replies ×1" in full and "Build green." not in full
    assert "· Nova — economics: Done " in full and "«Rebuilt the economy table.»" in full
    assert "get_task_result(task_id='nova1')" in full
    assert "- knowledge global/joi: 1 write by root in " in full and "why: «Nina's price test moved the rollout.»" in full
    assert "Main's own reply." not in full and "Before the night." not in full and "Good morning." not in full
    assert "Not known:" not in full, "the one connected place had rows"
    assert full.startswith(middle) and rf.WORDS_HEADING in full and rf.WORDS_HEADING not in middle
    assert "Sasha (sasha) ×1, newest " in middle and "Price test" not in middle, "the middle form: words behind readers"
    assert "«Rebuilt the economy table.»" in middle and "get_task_result(task_id='nova1')" in middle
    assert "why: «Nina's price test moved the rollout.»" in middle and "automation ×1" in middle
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
    _rows(tmp_path, _owner("Back in Telegram.", T0 + 30, chat_id=144346271, cmid="t"), _owner("Back in Main.", T0 + 40, cmid="b"))
    assert rf.previous_word(tmp_path, _word(tmp_path, "b"), accounts).row["text"] == "Back in Telegram.", "any room"
    _rows(tmp_path, _arrival(place, "anton", "Written to the team from my phone.", T0 + 45),
          _arrival(place, "sasha", "Sasha answers the team.", T0 + 50), _owner("Again.", T0 + 60, cmid="c"))
    later = rf.facts(tmp_path, _word(tmp_path, "c"), rf.Own(room=1))
    assert later["previous"].row["text"] == "Back in Main.", "their words in a place are not a word to me"
    full = rf.render(later)[0]
    theirs, heard = full.split("Places since then:")
    assert "My human's words elsewhere since then:" in theirs and "«Written to the team from my phone.»" in theirs
    assert "Sasha answers the team." in heard and "Written to the team" not in heard


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

    _rows(tmp_path, _owner("A.", T0, cmid="a"), _arrival(place, "sasha", "News.", T0 + 5), _owner("B.", T0 + 9, cmid="b"))
    turn = {"_is_direct_chat": True, "chat_id": 1, "origin_message_ref": {"chat_id": 1, "client_message_id": "b"}}
    facts, words = rf.turn_sections(turn, tmp_path)
    assert isinstance(facts, HostSection) and (facts.section_id, words.section_id) == (rf.SECTION_ID, rf.WORDS_ID)
    assert "Sasha (sasha) ×1, newest " in facts.full_text and "News." not in facts.full_text
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
    quiz = {"quiz_id": "q1", "question": "**Ship the economy table?** Details follow.", "answered_index": 0,
            "options": [{"label": "Yes, ship"}, {"label": "Wait"}], "comment": comment, "answered_at": _iso(ts)}
    return {"ts": _iso(ts), "direction": "system", "type": "quiz_answer", "chat_id": chat_id, "task_id": "asker",
            "client_message_id": "quiz_answer:asker:q1", "text": "[Owner quiz answer] quiz q1", "quiz": quiz}


def test_a_card_answer_is_my_humans_word_and_a_running_root_hears_from_where_its_view_reaches(tmp_path, place):
    _rows(tmp_path, _owner("Evening.", T0, cmid="a"), _arrival(place, "nina", "Overnight numbers are in.", T0 + 10),
          _card_answer(T0 + 20), _owner("Morning.", T0 + 30, cmid="b"))
    word = _word(tmp_path, "b")
    assert rf.previous_word(tmp_path, word, places.place_owner_accounts(tmp_path)).row["type"] == "quiz_answer"
    fresh = rf.render(rf.facts(tmp_path, word, rf.Own(room=1)))[0]
    assert "Overnight numbers" not in fresh and "no arrival from anyone else recorded since then in 1" in fresh, \
        "a fresh turn hears since the card answer"
    reach = {"seq": _word(tmp_path, "a").seq + 1, "ts": _iso(T0)}
    running = rf.render(rf.facts(tmp_path, word, rf.Own(root_task_id="main-root"), since=reach))[0]
    assert "This work's view and earlier facts reach to " in running and "My human's words elsewhere since then:" in running
    assert "card answer: chose «Yes, ship»; with the words «go»; the card: «Ship the economy table?»" in running
    assert "Nina (nina) ×1, newest " in running and "«Overnight numbers are in.»" in running



def test_a_late_card_click_is_one_word_elsewhere_not_two(tmp_path, place):
    """A click on a card whose work had ended writes the answer and its late delivery: one word (Astra r9)."""
    _rows(tmp_path, _owner("Evening.", T0, cmid="a"), _card_answer(T0 + 20),
          _owner("[Owner quiz answer] quiz q1", T0 + 21, chat_id=PROJECT_ROOM, cmid="quiz_late_answer:asker:q1"),
          _owner("Morning.", T0 + 30, cmid="b"))
    reach = {"seq": _word(tmp_path, "a").seq + 1, "ts": _iso(T0)}
    found = rf.facts(tmp_path, _word(tmp_path, "b"), rf.Own(root_task_id="main-root"), since=reach)
    full, _middle, last = rf.render(found)
    assert "Since then, counted: 1 word of my human elsewhere" in last and len(found["mine"]) == 1
    assert full.count("quiz q1") <= 1 and full.count("Ship the economy table?") == 1


def test_a_reason_stays_with_the_write_that_gave_it(tmp_path, place):
    """A page's newest write without a reason never borrows an earlier writer's (Astra r9: a Presence revision shown
    with a child's reason)."""
    _rows(tmp_path, _owner("A.", T0, cmid="a"))
    _write(tmp_path, T0 + 10, "gifts", chat_id=PLACE_ROOM, role="subagent", reason="The child's split of the types.")
    _write(tmp_path, T0 + 20, "gifts", chat_id=PLACE_ROOM, role="presence", reason="")
    _rows(tmp_path, _owner("B.", T0 + 40, cmid="b"))
    full = rf.render(rf.facts(tmp_path, _word(tmp_path, "b"), rf.Own(room=1)))[0]
    assert "global/gifts: 2 writes by subagent" in full and "; why: «The child's split" not in full
    assert "; the last write gave no reason; earlier, subagent" in full and "«The child's split of the types.»" in full


def test_several_people_in_one_place_each_keep_their_newest_words_and_their_own_reader(tmp_path, place):
    _rows(tmp_path, _owner("A.", T0, cmid="a"), _arrival(place, "sasha", "First thought.", T0 + 1),
          _arrival(place, "nina", "Nina's only line.", T0 + 2), _arrival(place, "sasha", "Second thought, whole.", T0 + 3),
          _owner("B.", T0 + 9, cmid="b"))
    full = rf.render(rf.facts(tmp_path, _word(tmp_path, "b"), rf.Own(room=1)))[0]
    line = next(text for text in full.splitlines() if text.startswith("- chat · design"))
    assert line.index("Sasha (sasha) ×2, newest") < line.index("Nina (nina) ×1"), "newest person first"
    assert line.count("memory_read(room_id=") == 2 and "<one of" not in line, "each person's exact reader"
    words = full.split(rf.WORDS_HEADING)[1]
    assert "Sasha (sasha), 15.01 08:00: «Second thought, whole.»" in words and "First thought." not in full, \
        "their newest words, whole"
    assert words.index("Sasha") < words.index("Nina (nina), 15.01 08:00: «Nina's only line.»")


def test_quiet_connected_places_are_said_by_name_inside_a_block_and_in_one_line_alone(tmp_path, place):
    places.connect_place(tmp_path, connector="mailer", provider="mail", account_id="inbox", label="Mail")
    _rows(tmp_path, _owner("A.", T0, cmid="a"), _arrival(place, "sasha", "Hello.", T0 + 1), _owner("B.", T0 + 2, cmid="b"))
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


def test_a_root_captured_after_my_humans_previous_word_gets_rows_from_its_view_and_page_changes_from_the_word(
        tmp_path, place):
    _rows(tmp_path, _owner("Evening.", T0, cmid="a"), _arrival(place, "sasha", "Before the dawn capture.", T0 + 60))
    _write(tmp_path, T0 + 120, "night-note", chat_id=PROJECT_ROOM, reason="Edited before the dawn capture.")
    dawn = {"seq": ds.stream_end(tmp_path, "chat"), "ts": _iso(T0 + 3600)}  # a scheduled root's view, captured later
    _rows(tmp_path, _arrival(place, "nina", "After the capture.", T0 + 7200), _owner("Morning.", T0 + 36000, cmid="b"))
    running = rf.render(rf.facts(tmp_path, _word(tmp_path, "b"), rf.Own(root_task_id="dawn-root"), since=dawn))[0]
    assert "«After the capture.»" in running and "Before the dawn capture." not in running, "its frozen view holds those"
    assert "global/night-note" in running and "«Edited before the dawn capture.»" in running, "no view holds an edit"
    assert ("rows from then, page changes from " + _iso(T0)[:16].replace("T", " ") + " UTC, to my human's word of "
            in running)


def test_my_humans_own_words_in_a_place_mean_it_was_heard(tmp_path, place):
    _rows(tmp_path, _owner("A.", T0, cmid="a"), _arrival(place, "anton", "From my phone to the team.", T0 + 5),
          _owner("B.", T0 + 9, cmid="b"))
    full = rf.render(rf.facts(tmp_path, _word(tmp_path, "b"), rf.Own(room=1)))[0]
    assert "«From my phone to the team.»" in full and "Not known:" not in full, "a place my human wrote in is not quiet"


def test_arrivals_without_a_recorded_author_are_not_called_automation(tmp_path, place):
    unknown = _arrival(place, "x", "Who wrote this?", T0 + 5)
    unknown["transport"]["actor"] = {}
    unknown.pop("sender_label")
    _rows(tmp_path, _owner("A.", T0, cmid="a"), unknown, _arrival(place, "ci", "Build green.", T0 + 6, is_bot=True),
          _owner("B.", T0 + 9, cmid="b"))
    full = rf.render(rf.facts(tmp_path, _word(tmp_path, "b"), rf.Own(room=1)))[0]
    assert "unattributed arrivals ×1" in full and "automation ×1" in full


def test_the_stepped_words_count_people_not_the_lines_of_their_words(tmp_path, place):
    _rows(tmp_path, _owner("A.", T0, cmid="a"), _arrival(place, "sasha", "line one\nline two\nline three", T0 + 5),
          _owner("B.", T0 + 9, cmid="b"))
    turn = {"_is_direct_chat": True, "chat_id": 1, "origin_message_ref": {"chat_id": 1, "client_message_id": "b"}}
    words = rf.turn_sections(turn, tmp_path)[1]
    assert "The newest words of 1 person stand" in words.operative_view and "line two" in words.full_text


def test_a_person_who_renamed_between_posts_is_one_person_under_their_newest_name(tmp_path, place):
    _rows(tmp_path, _owner("A.", T0, cmid="a"), _arrival(place, "sasha", "Before the rename.", T0 + 1),
          _arrival(place, "sasha", "After the rename.", T0 + 2, display_name="Alexander"), _owner("B.", T0 + 9, cmid="b"))
    full = rf.render(rf.facts(tmp_path, _word(tmp_path, "b"), rf.Own(room=1)))[0]
    assert "Alexander (sasha) ×2, newest " in full and "Sasha (sasha)" not in full


def test_one_person_by_their_id_across_places_and_a_forward_is_not_their_words(tmp_path, place):
    alex_one = _arrival(place, "a1", "Rollout paused.", T0 + 1)
    alex_two = _arrival(place, "a2", "Build passed.", T0 + 2)
    for row, ident in ((alex_one, "slack-a1"), (alex_two, "slack-a2")):  # their connector records only ``id``
        row["transport"]["actor"] = {"id": ident, "display_name": "Alex"}
    forward = _arrival(place, "sasha", "Nina's words, forwarded.", T0 + 3)
    forward["transport"]["message"] = {"forward_origin": {"sender_user_name": "Nina"}}
    twice = _arrival(place, "nina", "Nina in another conversation of the account.", T0 + 4)
    twice["transport"]["conversation_id"], twice["transport"]["conversation"], twice["chat_id"] = (
        "general", {"name": "general"}, PLACE_ROOM + 1)
    _rows(tmp_path, _owner("A.", T0, cmid="a"), alex_one, alex_two, forward,
          _arrival(place, "nina", "Nina here.", T0 + 5), twice, _owner("B.", T0 + 9, cmid="b"))
    full, _middle, last = rf.render(rf.facts(tmp_path, _word(tmp_path, "b"), rf.Own(room=1)))
    assert "Alex (slack-a1)" in full and "Alex (slack-a2)" in full, "two people with one name stay two"
    assert "(a forward: another author's words): «Nina's words, forwarded.»" in full
    assert "2 places: 4 people (5 messages)" in last, "Nina in two conversations of one account is one person"
    turn = {"_is_direct_chat": True, "chat_id": 1, "origin_message_ref": {"chat_id": 1, "client_message_id": "b"}}
    assert "The newest words of 4 people stand" in rf.turn_sections(turn, tmp_path)[1].operative_view, "once there too"
