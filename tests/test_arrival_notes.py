"""At my human's arrival the host names the working-view notes in view (core-interpretation repair, Ф4).

When their words or a press reach a work whose view holds notes the mind authored through ``compact_context`` folds,
the delivered row closes with one host line naming each note (unit id, write time or "not recorded", first line).
Both directions in each test: the line comes with my human's row and a note in view; it does not come without a
note, for a row that is not theirs, or on an earlier row; every earlier message keeps its bytes.
"""
from __future__ import annotations

import copy
import json
import queue
from types import SimpleNamespace

from tests._working_notes import _authored, _large_unit, _note, _unattributed, _undated

WORDS = "Use the March numbers"


def _written_note(tmp_path, text, call_id, written_at):
    """A note the mind wrote at ``written_at`` (its ``compact_context`` call time), through the real fold."""
    from ouroboros import context_compaction as cc

    source = _large_unit(call_id)
    folded, receipt = cc.compact_tool_history_llm(
        source, request=_authored(source, note=text), observed_messages=source, tool_schemas=[],
        fit_candidate=lambda *_: {"accepted": True}, drive_root=tmp_path, task_id="compaction", written_at=written_at)
    assert receipt.status == "applied" and len(folded) == 1
    return folded[0]


def _drain(tmp_path, messages, ctx=None, *texts):
    from ouroboros.loop_round_limits import _drain_incoming_messages

    incoming = queue.Queue()
    for index, text in enumerate(texts or (WORDS,)):
        incoming.put({"text": text, "client_message_id": f"c{index}", "ts": f"2026-11-17T10:0{index}:00+00:00"})
    _drain_incoming_messages(messages, incoming, tmp_path, "work", None, set(), owner_ctx=ctx or SimpleNamespace())
    return messages


def _unit_id(messages, index):
    from ouroboros import context_compaction as cc

    return next(unit.unit_id for unit in cc.context_units(messages, scope="dialogue") if unit.start == index)


def test_an_arrival_names_the_notes_in_view_and_changes_no_earlier_message(tmp_path):
    goal = _undated(_note(tmp_path, "Goal: keep the reply short.\nThe owner wants one paragraph.", "goal"))
    dated = _written_note(tmp_path, "Decided: send on Friday.", "dated", "2026-11-13T22:40:12+03:00")
    view = [{"role": "user", "content": "Draft the reply"}, goal, dated, {"role": "assistant", "content": "Drafted."}]
    before = json.dumps(view, sort_keys=True)
    _drain(tmp_path, view)
    assert json.dumps(view[:4], sort_keys=True) == before  # append only: earlier messages keep their bytes
    assert view[-1]["content"] == (
        f"[Message from my human]: {WORDS}\n[Host fact: your working-view notes in view now: {_unit_id(view, 1)}, "
        f"written not recorded, «Goal: keep the reply short.»; {_unit_id(view, 2)}, written 2026-11-13 19:40 UTC, "
        "«Decided: send on Friday.»]")


def test_no_note_in_view_and_a_row_not_my_humans_add_no_line(tmp_path):
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_mailbox import write_task_message
    from ouroboros.tools.compact_context import view_notes

    plain = _drain(tmp_path, [{"role": "user", "content": "Draft the reply"}])
    assert plain[-1]["content"] == f"[Message from my human]: {WORDS}"
    note = _note(tmp_path, "Goal: keep the reply short.", "goal")
    view = [{"role": "user", "content": "Brief"}, note]
    assert write_task_message(tmp_path, "Use the Q3 numbers only", "work", source_task_id="root-1", msg_id="pm-1")
    _drain_incoming_messages(view, queue.Queue(), tmp_path, "work", None, set(), owner_ctx=SimpleNamespace())
    assert view[-1]["content"].endswith("Use the Q3 numbers only") and "working-view notes" not in json.dumps(view)
    host = copy.deepcopy(note)  # a host source view is the host's projection, not a note of the mind's
    host["content"][0]["_context_capsule"].update(authorship="host", retention="source_view")
    assert view_notes([host]) == [] and len(view_notes([note], own=True)) == 1
    legacy = _unattributed(note)  # a note in view (a fold receipt lists it) whose author is not recorded: not "yours"
    assert len(view_notes([legacy])) == 1 and view_notes([legacy], own=True) == []
    assert _drain(tmp_path, [{"role": "user", "content": "Brief"}, legacy])[-1]["content"] == (
        f"[Message from my human]: {WORDS}")


def test_two_rows_of_my_human_in_one_drain_close_once_on_the_last(tmp_path):
    view = [{"role": "user", "content": "Brief"}, _note(tmp_path, "Goal: keep the reply short.", "goal")]
    _drain(tmp_path, view, None, "First", "Second")
    assert view[-2]["content"] == "[Message from my human]: First"
    assert view[-1]["content"].startswith("[Message from my human]: Second\n[Host fact: your working-view notes")


def test_a_notes_list_already_named_in_view_is_named_again_only_once_it_changes(tmp_path):
    """Both directions: the first arrival names the notes in view; a later arrival does not repeat the same list while
    the row that named it is among the messages in hand; without that row the list is named again, and so is a list
    that changed. What was named is read from the rows themselves, never stored."""
    view = [{"role": "user", "content": "Brief"}, _note(tmp_path, "Goal: keep the reply short.", "goal")]
    _drain(tmp_path, view, None, "First")
    named = view[-1]["content"].rsplit("\n", 1)[-1]
    assert named.startswith("[Host fact: your working-view notes in view now: ")
    _drain(tmp_path, view, None, "Second")
    assert view[-1]["content"] == "[Message from my human]: Second"  # the same list, already named in view
    without = [*view[:2], *view[3:]]  # the messages in hand no longer hold the row that named it
    assert _drain(tmp_path, without, None, "Third")[-1]["content"].endswith("\n" + named)
    view.append(_written_note(tmp_path, "Decided: send on Friday.", "friday", "2026-11-17T09:00:00+00:00"))
    _drain(tmp_path, view, None, "Fourth")
    changed = view[-1]["content"].rsplit("\n", 1)[-1]
    assert changed.startswith("[Host fact: your working-view notes in view now: ") and changed != named
    assert "«Goal: keep the reply short.»" in changed and "«Decided: send on Friday.»" in changed


def test_every_press_names_their_last_own_words_while_an_unchanged_notes_list_is_not_repeated(tmp_path):
    """Both directions: each press without words closes with their last own words, the second too; only the notes
    list, unchanged and already named in view, is left out of the second."""
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_mailbox import KIND_QUIZ_ANSWER, write_owner_message

    said = {"source": "direct_incoming", "content": "Keep it short", "client_message_id": "c0", "chat_id": "1",
            "ts": "2026-11-13T17:13:00+00:00"}
    ctx = SimpleNamespace(task_id="pressing", _owner_directives=[dict(said)])
    view = [{"role": "user", "content": "Brief"}, _note(tmp_path, "Goal: keep the reply short.", "goal")]
    for quiz in ("q1", "q2"):
        card = {"quiz_id": quiz, "question": "Publish?", "state": "answered", "options": ["Publish it"], "answered_index": 0}
        assert write_owner_message(tmp_path, f"[Owner quiz answer] quiz {quiz}", "pressing",
                                   msg_id=f"quiz_answer:asker:{quiz}", kind=KIND_QUIZ_ANSWER,
                                   quiz_delivery={"task_id": "asker", "quiz_id": quiz, "answer_snapshot": card})
        _drain_incoming_messages(view, queue.Queue(), tmp_path, "pressing", None, set(), owner_ctx=ctx)
    assert [row["content"].split("\n")[0] for row in view[2:]] == ["[Owner quiz answer] quiz q1",
                                                                   "[Owner quiz answer] quiz q2"]
    first, second = view[2]["content"], view[3]["content"]
    words = "[Host fact: their last own words in this work: 2026-11-13T17:13:00+00:00 (chat 1 / c0)]"
    assert words in first and second.endswith("\n" + words)  # per press, never left out
    assert "working-view notes in view now" in first and "working-view notes in view now" not in second


def test_a_line_the_host_cannot_compute_never_costs_the_delivery(tmp_path, monkeypatch, caplog):
    from ouroboros import arrival_notes

    def unreadable(_messages):
        raise RuntimeError("unreadable capsule")

    view = [{"role": "user", "content": "Brief"}, _note(tmp_path, "Goal: keep the reply short.", "goal")]
    with monkeypatch.context() as patched:
        patched.setattr(arrival_notes, "notes_line", unreadable)
        failed = _drain(tmp_path, copy.deepcopy(view))
    assert failed[-1]["content"] == f"[Message from my human]: {WORDS}"  # delivered, without the line
    assert "Arrival lines not computed" in caplog.text
    assert "working-view notes in view now" in _drain(tmp_path, copy.deepcopy(view))[-1]["content"]  # computed: shown


def _press(tmp_path, ctx, block):
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_mailbox import KIND_QUIZ_ANSWER, write_owner_message

    assert write_owner_message(tmp_path, "[Owner quiz answer] quiz q7", ctx.task_id, msg_id="quiz_answer:asker:q7",
                               kind=KIND_QUIZ_ANSWER,
                               quiz_delivery={"task_id": "asker", "quiz_id": "q7", "answer_snapshot": block})
    messages = [{"role": "user", "content": "Brief"}]
    _drain_incoming_messages(messages, queue.Queue(), tmp_path, ctx.task_id, None, set(), owner_ctx=ctx)
    return messages[-1]["content"]


def test_a_press_without_words_names_their_last_own_words_when_the_corpus_dates_them(tmp_path):
    card = {"quiz_id": "q7", "question": "Publish?", "state": "answered", "options": ["Publish it"], "answered_index": 0}
    said = {"source": "direct_incoming", "content": "Keep it short", "client_message_id": "c0", "chat_id": "1",
            "ts": "2026-11-13T17:13:00+00:00"}
    dated = SimpleNamespace(task_id="dated", _owner_directives=[dict(said)])
    assert _press(tmp_path, dated, card).endswith(
        "\n[Host fact: their last own words in this work: 2026-11-13T17:13:00+00:00 (chat 1 / c0)]")
    worded = SimpleNamespace(task_id="worded", _owner_directives=[dict(said)])  # words beside a press: theirs, no line
    assert "last own words" not in _press(tmp_path, worded, {**card, "comment": "but after lunch"})
    silent = SimpleNamespace(task_id="silent", _owner_directives=[])  # no words of theirs on record: nothing guessed
    assert "last own words" not in _press(tmp_path, silent, card)
