"""A press is an act (core-interpretation repair, Ф2): my human's press of an offered option without words is their
act on a label the asking task wrote, never their words; words typed beside a press stay theirs, verbatim.

Each test asserts both directions: the press reads as an act (the frame, the card, the corpus, the wake line,
Continue, a carried quote), and the words beside a press read as their words. A press stays an owner input exactly
as before: counted by ``owner_input_revision``, owed a reply, delivered to the asker at its next step, and read by
Safety's card check (``safety._owner_yes``, which reads ``OWNER_SOURCES``).
"""
from __future__ import annotations

import datetime
import queue
from types import SimpleNamespace

import pytest

ASKED = "2026-11-13T19:40:00+00:00"
CARD = {"quiz_id": "q7", "question": "Publish exactly this reply?", "state": "answered",
        "options": ["Publish it", "Not now"], "option_details": ["Sends it now", ""], "asked_at": ASKED,
        "answered_at": "2026-11-13T19:48:59+00:00"}
PRESS = {**CARD, "answered_index": 0}
COMMENT = "  but only after lunch  "
WORDED = {**PRESS, "comment": COMMENT}


def test_one_builder_names_whose_text_the_label_is_and_never_reads_the_silence():
    from ouroboros.answer_facts import option_answer

    mine = option_answer(PRESS, 0, "", asker="asker", reader="asker")
    assert mine == ("The owner pressed option 1 and wrote no words. The label «Publish it» is your text, as is its "
                    f"detail «Sends it now», from when you asked ({ASKED}).")
    other = option_answer(PRESS, 0, "", asker="asker", reader="child")
    assert other == ("The owner pressed option 1 and wrote no words. The label «Publish it» is the text of task asker, "
                     f"as is its detail «Sends it now», from when it asked ({ASKED}).")
    assert ", as is its detail" not in option_answer({**CARD, "answered_index": 1}, 1, "", asker="asker", reader="asker")
    worded = option_answer(WORDED, 0, COMMENT, asker="asker", reader="asker")
    assert worded == f"The owner pressed option 1 («Publish it», your text) and wrote: {COMMENT}"  # verbatim, edges kept
    assert option_answer(WORDED, 0, COMMENT, asker="asker").startswith("The owner pressed option 1 («Publish it», the "
                                                                       "text of task asker) and wrote: ")
    for text in (mine, other, worded):  # a fact of who wrote what, never a reading of the silence
        assert "reason" not in text and "not recorded" not in text


def test_the_quiz_frame_states_the_press_for_its_reader_and_keeps_the_free_answer():
    from ouroboros.owner_quiz import quiz_answer_frame, recorded_answer_frame

    letter = recorded_answer_frame(PRESS, asker="asker", reader="asker")
    room = recorded_answer_frame(PRESS, asker="asker")  # the room's row, a relayed quote, a late answer's head
    assert "wrote no words. The label «Publish it» is your text" in letter
    assert "wrote no words. The label «Publish it» is the text of task asker" in room and "your text" not in room
    assert "chose option" not in letter + room
    worded = recorded_answer_frame(WORDED, asker="asker", reader="asker")
    assert f"(«Publish it», your text) and wrote: {COMMENT}" in worded and "wrote no words" not in worded
    free = quiz_answer_frame({**CARD, "comment": "neither"}, None, "neither", asker="asker", reader="asker")
    assert "without choosing an offered option. Verbatim: neither" in free and "pressed" not in free


def test_a_card_press_without_values_says_whose_text_its_label_is():
    from ouroboros.gateway.card_decision import card_input_frame

    record = {"card_id": "c1", "title": "Flights"}
    assert card_input_frame(record, "Book it", None) == (
        "[Card input] The owner pressed «Book it» on your card «Flights» (card c1). The label is your text; no words "
        "of their own.")
    assert card_input_frame(record, "Book it", None, author="t1").endswith(
        "that task t1 showed (card c1). The label is the text of task t1; no words of their own.")
    valued = card_input_frame(record, "Dates", {"from": "May 3"})  # typed values are their words: no such sentence
    assert "no words of their own" not in valued and '"from": "May 3"' in valued


def _drain_answer(tmp_path, block, ctx):
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_mailbox import KIND_QUIZ_ANSWER, write_owner_message
    from ouroboros.owner_quiz import recorded_answer_frame

    letter = recorded_answer_frame(block, asker="asker", reader="asker")
    assert write_owner_message(tmp_path, letter, "asker", msg_id="quiz_answer:asker:q7", kind=KIND_QUIZ_ANSWER,
                               quiz_delivery={"task_id": "asker", "quiz_id": "q7", "answer_snapshot": block})
    messages = [{"role": "user", "content": "Draft the reply"}]
    from ouroboros.loop_messages import _initialize_owner_directives

    _initialize_owner_directives(ctx, messages)
    _drain_incoming_messages(messages, queue.Queue(), tmp_path, "asker", None, set(), owner_ctx=ctx)
    return letter, messages


@pytest.mark.parametrize("block", [PRESS, WORDED], ids=["press", "press_with_words"])
def test_a_live_answer_stays_an_owner_input_and_only_words_are_their_words(tmp_path, block):
    from ouroboros import owner_words as ow
    from ouroboros.loop_messages import owner_input_revision, owner_reply_owed

    ctx = SimpleNamespace(task_id="asker")
    letter, messages = _drain_answer(tmp_path, block, ctx)
    [row] = [item for item in ctx._owner_directives if item.get("quiz_id") == "q7"]
    # Unchanged: counted, owed a reply, delivered to the asker at its next step as a typed row of my human's.
    assert owner_input_revision(ctx) == 1 and owner_reply_owed(ctx)
    assert messages[-1]["content"] == letter and messages[-1]["_host_context_kind"] == "owner_words"
    rendered = ow.render_owner_words(ow.directive_owner_rows(ctx), audience="child")
    if block is WORDED:
        assert (row["source"], row["content"]) == ("owner_quiz_answer", COMMENT)
        assert f"\n{COMMENT}\n" in rendered and ow.owner_words_contain(ctx, "but only after lunch")
        assert ow.cited_owner_basis(ctx, "but only after lunch")[0]["content"] == COMMENT
    else:
        assert (row["source"], row["content"]) == ("owner_press", "1. Publish it")
        assert row["source"] in ow.OWNER_SOURCES and row["source"] not in ow.OWNER_WORD_SOURCES
        assert (" · owner · owner_press of task asker · msg quiz_answer:asker:q7]\n(host record: my human pressed "
                "«1. Publish it» and wrote no words; the label is the text of task asker)\n") in rendered
        assert not ow.owner_words_contain(ctx, "Publish it")  # the label is never cited or attested as their words
        basis, error = ow.cited_owner_basis(ctx, "Publish it")
        assert not error and basis["source"] == "owner_press" and basis["basis"] == "cited"
        assert basis["cited"] == "Publish it" and basis["msg_id"] == row["msg_id"] and basis["reply_to"] == row["reply_to"]
        for other in ("Publish", "Not now", "Sends it now", "1. Publish it"):
            assert ow.cited_owner_basis(ctx, other)[0] is None
        assert not ow.owner_words_contain(SimpleNamespace(_owner_directives=[basis]), "Publish it")


def test_safety_still_reads_a_press_as_my_humans_answer_to_the_card():
    """Capability kept: the card check (protected ``safety._owner_yes``) releases a call whose words and destination the
    answered card holds; a press without words is still such an answer, and a non-owner row carrying the same card
    is not."""
    from ouroboros import safety
    from ouroboros.safety_evidence import with_author
    from ouroboros.owner_words import answered_card_reply

    card = {"quiz_id": "q9", "question": "Send to @alice:\nThanks, taking it up.", "state": "answered",
            "options": ["Send it", "Hold"], "answered_index": 0}
    row = {"source": "owner_press", "content": "1. Send it", "quiz_id": "q9", "reply_to": answered_card_reply(card, "a")}
    subject = lambda: ("Thanks, taking it up.", ["@alice"], False, ["@alice"])  # noqa: E731 - the host's resolved subject
    assert safety._owner_yes(SimpleNamespace(_owner_directives=[row]), subject) == "q9"
    peer = {**row, "source": "principal_task_message"}
    assert safety._owner_yes(SimpleNamespace(_owner_directives=[peer]), subject) == ""
    evidence = with_author(row, {})["author"]
    assert "answer" in evidence and "press" in evidence and "no words of their own" in evidence
    assert "press" not in with_author({**row, "source": "owner_quiz_answer"}, {})["author"]


def test_the_wake_line_reads_a_press_as_an_act_and_words_as_words():
    from ouroboros import consciousness_wake as wake

    now = datetime.datetime.fromisoformat("2026-11-13T20:00:00+00:00").timestamp()
    pressed = wake._answered_card_line("asker", "q7", PRESS, now=now)
    assert ("The owner pressed option 1 and wrote no words. The label «Publish it» is the text of task asker, as is "
            f"its detail «Sends it now», from when it asked ({ASKED}).") in pressed
    worded = wake._answered_card_line("asker", "q7", WORDED, now=now)
    assert f"(«Publish it», the text of task asker) and wrote: {COMMENT}; question: " in worded
    assert "wrote no words" not in worded and "chose option" not in pressed + worded


def test_continue_reads_a_press_as_an_act_with_its_card_and_without_one():
    from ouroboros.owner_continue import _card_lines, work_order_text

    with_card = _card_lines({"source": "owner_press", "content": "1. Publish it", "source_task_id": "asker",
                             "quiz_id": "q7", "card": PRESS})
    assert with_card[2].startswith("  The owner pressed option 1 and wrote no words. The label «Publish it» is the text "
                                   "of task asker")
    worded = _card_lines({"source": "owner_quiz_answer", "content": COMMENT, "source_task_id": "asker", "card": WORDED})
    assert f"(«Publish it», the text of task asker) and wrote: {COMMENT} (answered " in worded[2]
    text = work_order_text("pred", "round_limit", {"later": [
        {"source": "owner_press", "content": "1. Publish it", "ts": "T1"},
        {"source": "owner_mailbox", "content": "Ship it on Friday", "ts": "T2"}]})
    assert ("- T1 (host record: the owner pressed «1. Publish it» and wrote no words; the label is the asking task's "
            "text)") in text
    assert "- T2 Ship it on Friday" in text and "- T1 1. Publish it" not in text


@pytest.mark.parametrize("block", [PRESS, WORDED], ids=["press", "press_with_words"])
def test_a_carried_answer_reaches_a_member_as_an_act_or_as_their_words(block):
    from ouroboros.owner_relay import _quote_body

    row = {"type": "quiz_answer", "task_id": "asker", "quiz": block, "direction": "system", "chat_id": 1,
           "client_message_id": "quiz_answer:asker:q7", "ts": block["answered_at"]}
    body = _quote_body(row, "[Owner words quoted verbatim; handle h]")
    words = body.split("\n[Host question frame]\n")[0].split("\n", 1)[1]
    if block is WORDED:
        assert words == COMMENT
    else:
        assert words == ("(host record: my human pressed «1. Publish it» and wrote no words; the label is the text of "
                         "task asker)")
    assert "The label «Publish it» is the text of task asker" in body or "the text of task asker) and wrote" in body


@pytest.mark.parametrize("door, reference", [
    ("late quiz answer", {"late_answer": {"task_id": "elm-desk", "quiz_id": "elm-map"}}),
    ("living submission", {"card_press": {"task_id": "elm-desk", "card_id": "card_0123456789abcdef", "request_id": "elm-press"}}),
])
def test_first_turn_guide_names_the_source_of_each_unavailable_input(tmp_path, door, reference):
    from ouroboros.loop_messages import _initialize_owner_directives

    origin = {"origin_message_ref": {"chat_id": 3, "client_message_id": "elm-opening"}}
    ctx = SimpleNamespace(drive_root=tmp_path, task_metadata={**origin, **reference})
    _initialize_owner_directives(ctx, [{"role": "user", "content": "Elm map"}])
    [unavailable] = ctx._owner_directives
    assert unavailable["source"] == "owner_card_input"
    typed = SimpleNamespace(drive_root=tmp_path, task_metadata=origin)
    _initialize_owner_directives(typed, [{"role": "user", "content": unavailable["content"]}])
    assert typed._owner_directives[0]["source"] == "initial_user"
    sentences = " ".join(_initialize_owner_directives.__doc__.split()).lower().split(". ")
    assert any(door in sentence and f"``{unavailable['source']}``" in sentence for sentence in sentences)
