"""The loop's drain stamps the owner message it actually DELIVERED (#896).

`steer_task` asks its context one question: has a LATER owner message reached this
turn, or is it still acting on the message that started it? The answer used to be a
counter both mailbox writers increment at WRITE time, so a follow-up still sitting
in the mailbox already ended the exact-bytes window and took the steer's receipt
with it. The typed fact is stamped at the seam that actually delivers the text, and
only for owner DIALOGUE: a task message, a quiz answer or a typed control is not
the owner steering this turn.
"""

from __future__ import annotations

import queue
from types import SimpleNamespace

from ouroboros.loop_round_limits import _drain_incoming_messages


def _drain(tmp_path, *, task_id="root", incoming=None, ctx=None):
    ctx = SimpleNamespace() if ctx is None else ctx
    _drain_incoming_messages(
        [], incoming if incoming is not None else queue.Queue(),
        tmp_path, task_id, None, set(), owner_ctx=ctx,
    )
    return ctx


def test_a_mailbox_owner_message_stamps_the_id_it_relays(tmp_path):
    from ouroboros.owner_mailbox import write_owner_message

    write_owner_message(
        tmp_path, "you may ask Anton questions", task_id="root",
        msg_id="msg-2:root:tok", client_message_id="msg-2",
    )

    ctx = _drain(tmp_path)

    assert ctx.last_owner_delivery["msg_id"] == "msg-2:root:tok"
    assert ctx.last_owner_delivery["client_message_id"] == "msg-2"
    assert ctx.last_owner_delivery["text"] == "you may ask Anton questions"
    assert ctx.last_owner_delivery["ts"]


def test_a_legacy_entry_without_the_field_still_ends_the_window(tmp_path):
    """An entry written before this field existed (or by a producer that knows no
    owner-message id) still DELIVERED owner text: the window closes, and the steer
    falls back to its own receipt id rather than borrowing the origin's."""
    from ouroboros.owner_mailbox import write_owner_message

    write_owner_message(tmp_path, "legacy follow-up", task_id="root", msg_id="legacy-1")

    ctx = _drain(tmp_path)

    assert ctx.last_owner_delivery == {
        "msg_id": "legacy-1", "client_message_id": "",
        "text": "legacy follow-up", "ts": ctx.last_owner_delivery["ts"],
    }


def test_a_task_message_and_a_quiz_answer_are_not_owner_steering(tmp_path):
    """Neither is the owner's own steering text, so neither ends the exact-bytes
    window: a turn that received a sibling's relayed note still relays the message
    it was started with."""
    from ouroboros.owner_mailbox import KIND_QUIZ_ANSWER, write_owner_message, write_task_message

    write_task_message(tmp_path, "ancestor says hello", "root", source_task_id="parent")
    write_owner_message(
        tmp_path, "[Answer] option B", task_id="root", msg_id="quiz-1",
        kind=KIND_QUIZ_ANSWER, client_message_id="msg-quiz",
    )

    ctx = _drain(tmp_path)

    assert getattr(ctx, "last_owner_delivery", None) is None


def test_the_in_process_queue_path_stamps_its_client_message_id(tmp_path):
    """A direct turn receives owner text through the in-process queue, not a file."""
    incoming: queue.Queue = queue.Queue()
    incoming.put({"text": "direct follow-up", "client_message_id": "msg-direct"})

    ctx = _drain(tmp_path, incoming=incoming)

    assert ctx.last_owner_delivery["client_message_id"] == "msg-direct"
    assert ctx.last_owner_delivery["text"] == "direct follow-up"


def test_a_plain_string_payload_stamps_a_delivery_with_no_id(tmp_path):
    incoming: queue.Queue = queue.Queue()
    incoming.put("bare owner text")

    ctx = _drain(tmp_path, incoming=incoming)

    assert ctx.last_owner_delivery == {
        "msg_id": "", "client_message_id": "", "text": "bare owner text", "ts": "",
    }


def test_two_owner_entries_in_one_drain_leave_the_last_one(tmp_path):
    """Latest wins: the steer relays the message the turn most recently received."""
    from ouroboros.owner_mailbox import write_owner_message

    write_owner_message(
        tmp_path, "first", task_id="root", msg_id="m1", client_message_id="msg-1",
    )
    write_owner_message(
        tmp_path, "second", task_id="root", msg_id="m2", client_message_id="msg-2",
    )

    ctx = _drain(tmp_path)

    assert ctx.last_owner_delivery["client_message_id"] == "msg-2"
    assert ctx.last_owner_delivery["text"] == "second"


def test_a_drain_with_no_owner_context_is_a_no_op(tmp_path):
    """Internal drains pass no context; stamping must not invent one."""
    from ouroboros.owner_mailbox import write_owner_message

    write_owner_message(
        tmp_path, "follow-up", task_id="root", msg_id="m1", client_message_id="msg-1",
    )

    _drain_incoming_messages([], queue.Queue(), tmp_path, "root", None, set(), owner_ctx=None)


def test_the_delivery_fact_lives_for_one_drain(tmp_path):
    """A task that relayed one owner message is a task again on its next round:
    the next drain clears the fact before reading the mailbox, so the routing
    issuer does not keep reading the task's own later words as the owner's."""
    from ouroboros.owner_mailbox import write_owner_message
    from ouroboros.tools.control_routing import ISSUER_OWNER_TURN, ISSUER_TASK, _routing_issuer

    write_owner_message(tmp_path, "please also check the tests", task_id="root", msg_id="m-1", client_message_id="cm-1")
    ctx = _drain(tmp_path, ctx=SimpleNamespace(task_id="root", task_metadata={}))
    assert ctx.last_owner_delivery["client_message_id"] == "cm-1"
    assert _routing_issuer(ctx)["kind"] == ISSUER_OWNER_TURN

    _drain(tmp_path, ctx=ctx)  # nothing new this round
    assert ctx.last_owner_delivery is None
    assert _routing_issuer(ctx)["kind"] == ISSUER_TASK

