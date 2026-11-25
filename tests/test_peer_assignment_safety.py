"""A letter that assigns work to my human's linked Ouroboros is read once, as its prepared operation.

``safety.check_peer_assignment`` is the seam its sender calls right before a possible send (a first send and a retry
of a recorded letter alike); the tool's raw arguments are left unread. The Supervisor here is the hermetic one
(``tests/_safety_fixtures``): these pin the host's policy around a reading, never a verdict's content, on neutral
words and ids that name no real place.
"""
from __future__ import annotations

import json

import pytest

from tests._safety_fixtures import supervisor as supervisor_fixture

supervisor = supervisor_fixture
TOOL, PLACE, LINK, LABEL = "delegate_to_ouroboros", "place-1", "link-0001", "Linked"
WORDS = "Count the sample rows."
LETTER = {"letter_id": "inv-1", "text": WORDS, "title": "Count", "cost_ceiling_usd": 2.0, "protocol": 1,
          "chain": [{"install_id": "install-a", "task_id": "work"}]}
GATE_HEAD = "These words to this destination were read DANGEROUS in this work:"


def _assign(h, **letter):
    return h.module.check_peer_assignment(h.ctx, place_id=PLACE, link_id=LINK, letter={**LETTER, **letter})


def _labelled(monkeypatch):
    """The link's place as production resolves it for words addressed by ``place_id`` (``place_guard``)."""
    from ouroboros import place_guard

    monkeypatch.setattr(place_guard, "call_place_facts", lambda ctx, name, args, **kw: {
        "effect": "message", "declared": True, "targets": {}, "attributable": True,
        "places": [{"place_id": PLACE, "place_label": LABEL, "place_state": "active", "conversation_id": LINK,
                    "conversation_key": "key", "conversation_keys": ["key"]}]})


def _card(h, quiz, *lines):
    """My human's answer to a card holding ``lines``; each answer is its own row (its time tells two apart)."""
    h.ctx._owner_directives = [{"source": "owner_quiz_answer", "content": "1. Yes", "quiz_id": quiz, "ts": quiz,
                                "reply_to": {"address": f"quiz {quiz} of task work",
                                             "text": "\n".join(("May I?", *lines, "1. Yes"))}}]


def _operation(**letter):
    """The rest of the call as an assignment's card must hold it: every prepared field but its words."""
    rest = {key: value for key, value in {**LETTER, **letter, "place_id": PLACE, "link_id": LINK}.items()
            if key not in ("letter_id", "text")}
    return f"{TOOL} {json.dumps(rest, ensure_ascii=False, sort_keys=True)}"


def _bundle(prompt):
    return json.loads(prompt[prompt.index("{"):prompt.index("\nProposed tool call:")])


def _arguments(prompt):
    return json.loads(prompt[prompt.index("\nArguments:\n") + len("\nArguments:\n"):prompt.index("\nIs this safe?")])


def test_the_raw_call_is_left_unread_and_every_other_builtin_keeps_its_own_check(supervisor):
    """Only this built-in's raw arguments are deferred: they are not its effect (a retry names no words, a first send
    no resolved ceiling). Its prepared letter is read; a message to the same place is read as before."""
    from types import SimpleNamespace

    from ouroboros import peer_letters

    h = supervisor
    for raw in ({"place_id": PLACE, "text": WORDS, "cost_ceiling_usd": 2.0}, {"place_id": PLACE, "retry_of": "inv-1"}):
        assert h.module.check_safety(TOOL, raw, ctx=h.ctx, declared_policy="check") == (True, "")
    # What the registry resolves for a built-in is never the prepared letter.
    assert h.module.check_safety(TOOL, {"place_id": PLACE, "text": WORDS}, ctx=h.ctx,
                                 resolved_binding=SimpleNamespace(root="user_files")) == (True, "")
    assert h.calls == [] and h.events == [], "no reading and no skip was recorded for a deferred call"
    assert h.module.check_safety("message_ouroboros", {"place_id": PLACE, "text": WORDS}, ctx=h.ctx) == (True, "")
    assert len(h.calls) == 1, "an ordinary message to that place keeps its own reading"
    assert _assign(h) == (True, "") and len(h.calls) == 2, "the prepared letter is what is read"
    assert h.module.check_peer_assignment is peer_letters.check_peer_assignment, "the seam is bound in safety"


def test_the_whole_prepared_operation_reaches_the_supervisor_and_the_letter_id_only_as_a_fact(supervisor):
    h = supervisor
    assert _assign(h) == (True, "")
    prompt = h.calls[-1]["user_prompt"]
    assert "\nTool: delegate_to_ouroboros\n" in prompt
    sent = _arguments(prompt)
    assert sent == {key: value for key, value in {**LETTER, "place_id": PLACE, "link_id": LINK}.items()
                    if key != "letter_id"}, "its place, its link and every field of the letter but its id"
    target = _bundle(prompt)["subject"]["physical_target"]["peer_letter"]
    assert target["letter_id"] == "inv-1" and "starts ordinary work there" in target["operation"]
    assert "cost_ceiling_usd" in target["operation"] and "that installation's budget" in target["operation"]


def test_a_new_letter_id_is_the_same_decision_and_a_changed_ceiling_is_another_operation(supervisor):
    """A first send mints a new letter id each time, so the same intent issued again in a later step is the identical
    call for the mind's decision; a changed effective ceiling is another operation and is read again. A retry sends
    its recorded letter, so its subject is that of the send it repeats."""
    h = supervisor
    h.replies.append({"status": "SUSPICIOUS", "reason": "look again"})
    ok, message = _assign(h)
    assert not ok and message.startswith("⚠️ SAFETY_RETURNED: not run; nothing was sent or changed.")
    assert "issuing the identical call in a later step" in message and len(h.calls) == 1
    assert not _assign(h, letter_id="inv-2")[0] and len(h.calls) == 1, "the step that got the return decides nothing"
    h.later()
    ok, message = _assign(h, letter_id="inv-2")
    assert ok and message.startswith("⚠️ SAFETY_NOTE: let through on your later identical call") and len(h.calls) == 1
    assert _assign(h, cost_ceiling_usd=1.5) == (True, "") and len(h.calls) == 2, "a changed ceiling is read again"
    recorded = json.loads(json.dumps({**LETTER, "cost_ceiling_usd": 1.5}))  # a retry's letter, as its record holds it
    digest = h.module._subject_digest
    assert digest(TOOL, {**{k: v for k, v in recorded.items() if k != "letter_id"}, "place_id": PLACE,
                         "link_id": LINK}) == digest(TOOL, _arguments(h.calls[-1]["user_prompt"]))


def test_a_raw_call_between_two_readings_keeps_the_returned_reading_in_view(supervisor):
    """The mind's re-issue passes the registry's deferred raw check first: that is no call let through, so a changed
    letter is still read with the reading its predecessor was returned with."""
    h = supervisor
    h.replies.append({"status": "SUSPICIOUS", "reason": "look again"})
    assert not _assign(h)[0]
    assert h.module.check_safety(TOOL, {"place_id": PLACE, "text": "Count the other rows."}, ctx=h.ctx) == (True, "")
    h.later()
    assert _assign(h, text="Count the other rows.") == (True, "")
    pending = _bundle(h.calls[-1]["user_prompt"])["subject"]["returned_and_pending"]
    assert (pending["status"], pending["reason"]) == ("SUSPICIOUS", "look again")


def test_a_card_for_words_alone_releases_a_message_and_never_an_assignment(supervisor, monkeypatch):
    """My human's answered card holding words and their destination lets an unread message with those words through
    (the existing exit). The same card is no consent to create and fund work there: the assignment is read."""
    h = supervisor
    _labelled(monkeypatch)
    _card(h, "q1", WORDS, f"To {LABEL}")
    ok, message = h.module.check_safety("message_ouroboros", {"place_id": PLACE, "text": WORDS}, ctx=h.ctx)
    assert ok and message.startswith("⚠️ SAFETY_NOTE: let through on your human's answer (q1)") and h.calls == []
    assert _assign(h) == (True, "") and len(h.calls) == 1, "the assignment met a reading, not that card"
    assert [event["type"] for event in h.events].count("safety_sent_without_assessment") == 1


def test_a_flagged_assignment_waits_for_a_card_that_holds_its_whole_operation(supervisor, monkeypatch):
    """A DANGEROUS reading of an assignment waits for my human. A card with its words and destination alone does not
    release it, nor does a card for another ceiling (each later answer of theirs earns one new reading, as for any
    call, and nothing more); the card that also holds the rest of this operation does, so a separately authorized
    assignment stays possible, also for a later send of the same letter under a new id."""
    h = supervisor
    _labelled(monkeypatch)
    danger = {"status": "DANGEROUS", "reason": "the note keeps this apart", "quote": WORDS}
    h.replies.append(danger)
    ok, message = _assign(h)
    assert not ok and "waits for your human" in message and len(h.calls) == 1
    assert (f"a card holding these exact words on their own lines and the destination {LABEL}, {_operation()}."
            in message), message
    h.later()
    _card(h, "q2", WORDS, f"To {LABEL}")
    h.replies.append(danger)
    ok, message = _assign(h)
    assert not ok and "waits for your human" in message and len(h.calls) == 2, "words and a place are not the operation"
    h.later()
    _card(h, "q3", WORDS, f"To {LABEL}", _operation())
    h.replies.append(danger)
    ok, message = _assign(h, cost_ceiling_usd=50.0)
    assert not ok and len(h.calls) == 3 and _operation(cost_ceiling_usd=50.0) in message, "that card names another ceiling"
    assert not any(event["type"] == "safety_sent_despite_check" for event in h.events), "no card released anything yet"
    h.later()
    ok, message = _assign(h, letter_id="inv-later")
    assert ok and message.startswith("⚠️ SAFETY_NOTE: let through on your human's answer (q3)") and len(h.calls) == 3
    assert "a card holding this exact text and destination, not what they answered" in message
    assert (h.events[-1]["type"], h.events[-1]["quiz_id"]) == ("safety_sent_despite_check", "q3")


@pytest.mark.parametrize("first", ["message_ouroboros", TOOL])
def test_dangerous_words_keep_waiting_through_the_other_door_in_both_directions(supervisor, monkeypatch, first):
    """Words to this place read DANGEROUS wait for my human whichever of the two tools carries them next: neither a
    message after an assignment nor an assignment after a message buys a fresh reading."""
    h = supervisor
    _labelled(monkeypatch)

    def send(name):
        return (_assign(h) if name == TOOL
                else h.module.check_safety(name, {"place_id": PLACE, "text": WORDS}, ctx=h.ctx))
    h.replies.append({"status": "DANGEROUS", "reason": "the note keeps this apart", "quote": WORDS})
    assert not send(first)[0] and len(h.calls) == 1
    h.later()
    ok, message = send("message_ouroboros" if first == TOOL else TOOL)
    assert not ok and GATE_HEAD in message and "waits for your human" in message and len(h.calls) == 1


@pytest.mark.parametrize("mode", ["advanced", "cyber_pro"])
def test_an_assignment_no_reading_reached_is_held_like_words_in_every_mode(supervisor, monkeypatch, mode):
    h = supervisor
    monkeypatch.setattr(h.module, "get_runtime_mode", lambda: mode)
    h.replies.append(RuntimeError("the checker is down"))
    ok, message = _assign(h)
    assert not ok and message.startswith("⚠️ SAFETY_UNAVAILABLE: not run."), message
    assert h.events[-1]["type"] == "safety_hold"
