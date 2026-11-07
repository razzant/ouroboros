"""A wake records people's words whole in its stored observation, every new row under its host-known author and
exact address. The text it reads lists every new place message whole (a room my view shows as one line holds no
words; an engaged room's new rows may stand in both) and counts my human's messages, which my view always carries
(``## This room`` and ``## Other conversations``, folded last; owner 06.11, Q2-A). The fallback form keeps my human's
card answers whole beside the exact pointer, and a task's terminal line says the card's outcome and an explicit stop's
recorded reason.

Every guard is checked both ways (recorded whole vs the earlier 240-character cut; my human vs a participant, a
shared account or a bot; counted vs listed; coverage only beside a stored source; a plain finish vs an author's stop).
"""
from __future__ import annotations

import json

import pytest

from ouroboros import consciousness_wake as wake, places
from ouroboros.chat_chain import format_address, row_address
from ouroboros.dialogue_provenance import RoomLabelResolver
from tests.test_consciousness_wake import T0, _chat, _iso, _observe, _owner, _result, _summary

ROOM = 2 ** 41 + 11
TAIL = " — and the last line changes everything: do not merge it."
LONG = "Here is the whole plan for the release, step by step. " * 8 + TAIL


@pytest.fixture
def place(tmp_path):
    connected = places.connect_place(tmp_path, connector="bridge", provider="chat", account_id="qa", label="Team")
    places.edit_place(tmp_path, connected.place_id, "confirm_owner_account", author="owner",
                      platform_actor_id="anton", account_id="qa")
    return connected


def _place_row(place, actor_id, text, ts, *, sends_as="", sends_kind="user", **actor):
    transport = {"provider": "chat", "account_id": "qa", "conversation_id": "design",
                 "actor": {"platform_actor_id": actor_id, "display_name": actor_id.title(), **actor}}
    if sends_as:
        transport["identity"] = {"sends_as": {"id": sends_as, "label": "Shared", "kind": sends_kind}}
    return {"ts": _iso(ts), "direction": "in", "chat_id": ROOM, "text": text, "source": "presence:chat",
            "sender_label": actor_id.title(), "client_message_id": f"ev-{actor_id}-{ts}", "transport": transport,
            "presence_provenance": {"place_id": place.place_id}, "place_id": place.place_id}


def _stored(tmp_path):
    from ouroboros import dialogue_store as ds

    return {entry.row["text"]: format_address(row_address(entry.row)) for entry in ds.iter_rows(tmp_path, "chat")}


def test_people_words_reach_the_wake_whole_with_author_and_address(tmp_path, place):
    since = T0 - 3600
    _chat(tmp_path, _owner(LONG, since + 1, cmid="a"),
          _place_row(place, "sasha", "Sasha's long answer: " + LONG, since + 2),
          _place_row(place, "anton", "From my phone: " + LONG, since + 3, sends_as="anton"),  # I write from it too
          _place_row(place, "ouro-bot", "Echoed: " + LONG, since + 4, sends_as="ouro-bot", sends_kind="bot"),
          _place_row(place, "ci-bot", "Bot report: " + LONG, since + 5, kind="bot", is_bot=True))
    observation = _observe(tmp_path)
    kinds = [kind for kind, _offset, _line in observation.events]
    assert kinds == ["owner_message", "correspondent_message", "owner_message", "correspondent_message",
                     "correspondent_message"]
    lines = [line for _kind, _offset, line in observation.events]
    addresses = _stored(tmp_path)
    for line, text in zip(lines, [LONG, "Sasha's long answer: " + LONG, "From my phone: " + LONG,
                                  "Echoed: " + LONG, "Bot report: " + LONG]):
        assert line.endswith(text) and addresses[text] in line, "whole words under their exact address"
    assert "my human: Anton" in lines[2] and lines[2].startswith("- owner message in ")
    assert "; the account I send as: " in lines[3] and lines[3].startswith("- message in a place in ")  # my account
    assert "Sasha [" in lines[1] and "my human" not in lines[1] and "a bot" not in lines[1]
    assert "a bot: Ci-Bot" in lines[4], "a connector-marked bot comes whole with its author fact (owner Q2=A)"
    text = observation.full_text()
    assert all(lines[n] in text for n in (1, 3, 4)) and "…[" not in text, "a place message: listed whole"
    assert lines[0] not in text and lines[2] not in text, "my human's: counted, my view holds their words"
    assert "2 owner messages (counted, not listed: my view holds the words), 3 messages in places; " in text
    source = observation.source_bytes().decode("utf-8")
    assert source.count(TAIL.replace('"', '\\"')) == 5, "the stored source holds the same whole lines"


def test_the_fallback_form_points_at_the_exact_source_and_lists_no_words_my_view_holds(tmp_path, place):
    from ouroboros.artifacts import read_actor_source_bytes

    since = T0 - 3600
    _chat(tmp_path, _owner(LONG, since + 1, cmid="a"), _place_row(place, "sasha", "Sasha: " + LONG, since + 2))
    observation = _observe(tmp_path)
    task = {"id": "wake-1", "metadata": {}}
    wake.bind_wake_observation(tmp_path, task, observation, lambda events: "[Wake-up] " + events)
    facts = task["metadata"][wake.WAKE_OBSERVATION_KEY]
    projection = facts["projection_text"]
    assert LONG not in projection and "Sasha" not in projection, "my human's and Sasha's words: in my view"
    assert json.dumps(facts["source"]["read"], sort_keys=True, ensure_ascii=False) in projection
    assert "pointer_text" not in facts, "nothing inline: the projection is the bare pointer"
    assert not {"whole_rows", "inline_rows_in_fallback", "owner_projection_text"} & set(facts)
    stored = read_actor_source_bytes(tmp_path, "wake-1", facts["source"]).decode("utf-8")
    assert stored.count(TAIL.replace('"', '\\"')) == 2, "the exact source keeps both rows whole"


def test_a_bare_pointer_is_bound_only_when_the_fallback_form_holds_more(tmp_path, place):
    since = T0 - 3600
    _result(tmp_path, "t1", ts=since + 5)
    _chat(tmp_path, _summary("t1", since + 6), _place_row(place, "sasha", "Sasha: " + LONG, since + 7))
    plain = {"id": "wake-4", "metadata": {}}
    wake.bind_wake_observation(tmp_path, plain, _observe(tmp_path), lambda events: events)
    assert "pointer_text" not in plain["metadata"][wake.WAKE_OBSERVATION_KEY], "no card answer: the projection is bare"
    block = {"quiz_id": "q1", "state": "answered", "asked_at": _iso(since + 60), "answered_at": _iso(since + 70),
             "options": ["A", "B"], "comment": "Take A.", "question": "Which?"}
    _chat(tmp_path, {"ts": _iso(since + 70), "direction": "system", "type": "quiz_answer", "task_id": "asker",
                     "quiz": block})
    answered = {"id": "wake-5", "metadata": {}}
    wake.bind_wake_observation(tmp_path, answered, _observe(tmp_path), lambda events: events)
    facts = answered["metadata"][wake.WAKE_OBSERVATION_KEY]
    assert facts["pointer_text"] != facts["projection_text"] and "Take A." in facts["projection_text"]


def test_my_humans_card_answer_stays_whole_in_the_fallback_form(tmp_path):
    since = T0 - 3600
    comment = "Keep the old parser.\n\nReason: " + "the migration cost is too high; " * 40 + "END-OF-COMMENT"
    block = {"quiz_id": "qown", "state": "answered", "asked_at": _iso(since + 60), "answered_at": _iso(since + 2400),
             "options": ["A", "B"], "comment": comment, "question": "Which parser?"}
    _chat(tmp_path, {"ts": _iso(since + 2400), "direction": "system", "type": "quiz_answer", "task_id": "asker",
                     "quiz": block})
    task = {"id": "wake-6", "metadata": {}}
    wake.bind_wake_observation(tmp_path, task, _observe(tmp_path), lambda events: events)
    facts = task["metadata"][wake.WAKE_OBSERVATION_KEY]
    assert comment in facts["projection_text"], "my human's answer, whole"
    assert "END-OF-COMMENT" not in facts["pointer_text"], "the bare pointer carries no words"


def test_coverage_is_claimed_only_beside_a_stored_source(tmp_path, place, monkeypatch):
    _chat(tmp_path, _owner(LONG, T0 - 10, cmid="a"))
    observation = _observe(tmp_path)

    def unstorable(*_args, **_kwargs):
        raise OSError("store unavailable")

    monkeypatch.setattr("ouroboros.chat_chain.retain_memory_source", unstorable)
    task = {"id": "wake-2", "metadata": {}}
    wake.bind_wake_observation(tmp_path, task, observation, lambda events: events)
    facts = task["metadata"][wake.WAKE_OBSERVATION_KEY]
    assert "source_error" in facts
    assert not {"whole_rows", "inline_rows_in_fallback", "pointer_text", "projection_text"} & set(facts)


def _stop_result(tmp_path, task_id, ts, rationale):
    from tests._store_shared import seed_result

    seed_result(tmp_path, task_id, {
        "task_id": task_id, "status": "completed", "ts": _iso(ts), "updated_at": _iso(ts), "_schema_version": 1,
        "accounted_upper_bound_usd": 0.5, "description": f"do {task_id}", "metadata": {},
        "outcome_axes": {"lifecycle": {"status": "completed"},  # as ``outcomes`` records finish_task(action=stop)
                         "execution": {"status": "ok", "task_completion": {"action": "stop", "rationale": rationale}},
                         "objective": {"status": "fail", "source": "task_completion", "reason": "author_stop"}}})


def test_two_completed_tasks_read_their_card_outcomes_not_a_bare_lifecycle(tmp_path):
    since = T0 - 3600
    _result(tmp_path, "done1", ts=since + 1)
    _stop_result(tmp_path, "stop1", since + 2, "The API stayed down, so nothing could be verified.")
    _chat(tmp_path, _summary("done1", since + 3), _summary("stop1", since + 4))
    observation = _observe(tmp_path)  # chat-announced terminals: the ``_transition_line`` path
    done, stopped = [line for kind, _offset, line in observation.events if kind == "task_terminal"]
    assert done.startswith("- task done1 Done") and " completed" not in done
    assert stopped.startswith("- task stop1 Stopped") and "Ouroboros stopped with unfinished work" in stopped
    assert "The API stayed down, so nothing could be verified." in stopped and done != stopped
    from ouroboros.task_store import read_document

    results = {tid: read_document(tmp_path, tid) for tid in ("done1", "stop1")}
    direct = [wake._event_line("task_terminal", {"task_id": tid, "ts": _iso(since)}, results, RoomLabelResolver(projects=[]),
                               {}, T0) for tid in ("done1", "stop1")]
    assert direct[0].startswith("- task done1 Done") and "The API stayed down" in direct[1]
    assert stopped.endswith(" · Continue open (author_stop)") and direct[1].endswith(" · Continue open (author_stop)")
    assert "Continue open" not in done and "Continue open" not in direct[0], "a finished root offers no Continue"


@pytest.mark.parametrize("synthesis", ["pending_once", "running", "completed"])
def test_a_finished_root_reads_its_outcome_while_its_synthesis_is_still_open(tmp_path, synthesis):
    """Every root's terminal write leaves its post-task synthesis open for a while, and the card reads Working until it
    settles. The wake consumes the terminal transition at its capture, so its line must already say the outcome: a later
    wake never says it again."""
    from tests._store_shared import seed_result
    from ouroboros.task_store import read_document

    seed_result(tmp_path, "open1", {
        "task_id": "open1", "status": "completed", "ts": _iso(T0 - 60), "updated_at": _iso(T0 - 60), "_schema_version": 1,
        "accounted_upper_bound_usd": 0.5, "description": "do open1", "metadata": {},
        "root_phase_checkpoint": {"post_task_synthesis": synthesis},
        "outcome_axes": {"lifecycle": {"status": "completed"}, "execution": {"status": "ok"}}})
    line = wake._event_line("task_terminal", {"task_id": "open1", "ts": _iso(T0 - 60)},
                            {"open1": read_document(tmp_path, "open1")}, RoomLabelResolver(projects=[]), {}, T0)
    assert line.startswith("- task open1 Done") and "Working" not in line
    # Its recorded cause reads the same settled record: the line equals the one a settled synthesis prints.
    for status, reason, objective in (("fail", "author_finish", "best_effort"), ("degraded", "review_degraded", "pass")):
        record = {"status": "completed", "reason_code": "final_message", "description": "do it", "metadata": {}, "outcome_axes": {
            "lifecycle": {"status": "completed"}, "execution": {"status": "ok"}, "objective": {"status": objective},
            "review": {"status": status, "acceptance_decision": {"status": "finalized_unaccepted", "reason": reason}}}}
        lines = [wake._event_line("task_terminal", {"task_id": "r1", "ts": _iso(T0 - 60)}, {"r1": {
            **record, "root_phase_checkpoint": {"post_task_synthesis": phase}}}, RoomLabelResolver(projects=[]), {}, T0)
            for phase in (synthesis, "completed")]
        assert lines[0] == lines[1] and lines[1].endswith(")"), lines


def test_an_event_copy_of_a_wake_task_keeps_the_forms_sizes_not_their_bodies(tmp_path, place):
    """``task_received`` goes to events.jsonl and the live log frame: the forms grow with the window (one long message
    of my human or thousands of short ones), while the task document and the stored source keep them whole."""
    from ouroboros.utils import sanitize_task_for_event

    since = T0 - 3600
    _chat(tmp_path, _owner(LONG * 40, since + 1, cmid="a"), _place_row(place, "sasha", "Sasha: " + LONG, since + 2))
    task = {"id": "wake-1", "text": "[Wake-up]", "metadata": {}}
    wake.bind_wake_observation(tmp_path, task, _observe(tmp_path), lambda events: "[Wake-up] " + events)
    facts = task["metadata"][wake.WAKE_OBSERVATION_KEY]
    event = sanitize_task_for_event(task, tmp_path / "logs")
    copied = event["metadata"][wake.WAKE_OBSERVATION_KEY]
    forms = [key for key in facts if key.endswith("_text")]
    assert forms and all(key not in copied and copied[f"{key}_len"] == len(facts[key]) for key in forms)
    assert copied["source"] == facts["source"] and copied["composition"] == facts["composition"]
    assert len(json.dumps(event)) < len(LONG) * 10, "the event carries no copy of my human's 40 × LONG words"
    assert task["metadata"][wake.WAKE_OBSERVATION_KEY] is facts and LONG not in facts["projection_text"]


def test_an_overflow_delivers_the_bare_pointer_while_a_measured_shortfall_keeps_my_humans_words(tmp_path,
                                                                                                monkeypatch):
    from ouroboros.loop_model_call import _project_wake_input
    from tests.test_wake_observation_fit import FULL, WAKE, _ctx, _record_events

    _record_events(monkeypatch)
    inline = "Wake facts: the complete observation is stored at <source>. My human's words, whole: " + LONG
    pointer = "Wake facts: the complete observation is stored at <source>."
    ctx, _ = _ctx(tmp_path, window=0, status="failed", wake={**WAKE, "projection_text": inline, "pointer_text": pointer})
    assert _project_wake_input(ctx) is False and ctx.messages[1]["content"] == FULL
    assert _project_wake_input(ctx, overflowed=True) is True and ctx.messages[1]["content"] == pointer
