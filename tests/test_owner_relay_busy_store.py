"""A store too busy for bookkeeping never fails the owner's turn; its held sources are carried later."""
import logging
import queue
from types import SimpleNamespace

import pytest

from ouroboros import owner_relay, task_store, working_checkpoint
from ouroboros.loop_round_limits import _drain_incoming_messages
from ouroboros.owner_relay import flush_held_state, held_owner_messages, persist_held_state, quote_rows, write_relay
from ouroboros.store import StoreBusy, StoreLost
from tests.test_owner_relay import context, source


def _busy(confirmed=True):
    exc = StoreBusy("current store busy: database is locked", reason="contention")
    exc.rollback_confirmed = confirmed
    return exc


class _Store:
    """``task_store.patch_document`` as a double: scripted refusals, then the mutator's own output."""

    def __init__(self, refusals=()):
        self.refusals, self.calls, self.written, self.current = list(refusals), 0, [], {"task_id": "x"}
        self.busy = False  # every call refused while set, as under a reader convoy

    def __call__(self, root, task_id, mutator):
        self.calls += 1
        if self.busy:
            raise _busy()
        if self.refusals:
            raise self.refusals.pop(0)
        updated = mutator(self.current)
        if updated is not None:
            self.current = updated
            self.written.append(updated)
        return self.current


@pytest.fixture
def store(monkeypatch):
    double = _Store()
    monkeypatch.setattr(task_store, "patch_document", double)
    return double


def test_the_mailbox_drain_survives_a_busy_store_and_the_note_reaches_the_model(tmp_path, store, caplog):
    row, handle = source(tmp_path)
    quotes, error = quote_rows(context(tmp_path, row), [handle])
    assert not error
    assert write_relay(tmp_path, "receiver", relay_id="r1", note="MODEL NOTE", quotes=quotes, chat_id=1,
                       teller_task_id="teller")[1]
    receiver = context(tmp_path)
    receiver.task_id = "receiver"
    baseline = len(store.written)  # the teller's own quote lookup persisted above
    store.busy = True  # the whole drain meets a store that will not take the write
    caplog.set_level(logging.WARNING, logger=owner_relay.__name__)
    messages = []
    _drain_incoming_messages(messages, queue.Queue(), tmp_path, "receiver", None, set(), owner_ctx=receiver)
    assert "MODEL NOTE" in str(messages), "the owner's note was delivered to the model"
    assert handle in held_owner_messages(receiver) and receiver._held_state_dirty is True
    assert store.calls >= baseline + 2 and len(store.written) == baseline
    assert sum("not persisted" in rec.message for rec in caplog.records) == 1, "the gap is logged once"
    store.busy = False
    flush_held_state(receiver)
    assert receiver._held_state_dirty is False
    assert store.written and handle in store.written[-1]["held_owner_messages"]


def test_the_reply_path_delivers_exactly_once_and_the_tool_reports_ok(tmp_path, store):
    from ouroboros.tools.control_runtime import _send_user_message
    row, handle = source(tmp_path)
    recipient = context(tmp_path)
    recipient.current_chat_id = 42
    recipient.task_metadata = {"origin_suppressed": True}
    recipient._open_relays = {"r": {"relay_id": "r", "source": {"chat_id": 1, "address": handle}}}
    store.busy = True
    assert _send_user_message(recipient, "answer", reply_to_relay="r").startswith("OK:")
    assert len(recipient.pending_events) == 1 and recipient.pending_events[0]["system_type"] == "relay_reply"
    assert "r" not in recipient._open_relays and recipient._held_state_dirty is True


def test_the_first_held_messages_read_of_a_turn_survives_a_busy_store(tmp_path, store):
    row, handle = source(tmp_path)
    ctx = context(tmp_path, row)
    store.refusals = [_busy()]
    assert list(held_owner_messages(ctx)) == [handle], "the context build keeps its quote handles"
    assert ctx._held_state_dirty is True


def test_maps_this_context_already_persisted_open_no_write_transaction(tmp_path, store):
    row, handle = source(tmp_path)
    ctx = context(tmp_path, row)
    held_owner_messages(ctx)
    assert len(store.written) == 1
    persist_held_state(ctx)
    persist_held_state(ctx)
    assert len(store.written) == 1 and store.calls == 1, "unchanged maps never reach the write gate"
    ctx._open_relays = {"r": {"relay_id": "r", "source": {"chat_id": 1, "address": handle}}}
    persist_held_state(ctx)
    assert len(store.written) == 2 and store.calls == 2, "a changed map is persisted again"
    store.refusals = [_busy()]
    ctx._open_relays = {}
    persist_held_state(ctx)
    assert ctx._held_state_dirty is True and store.calls == 3
    persist_held_state(ctx)
    assert store.calls == 4 and ctx._held_state_dirty is False, "a refused persist is retried even for the same maps"
    assert store.written[-1]["open_relays"] == {}


@pytest.mark.parametrize("error", [StoreLost("current store unreadable"), _busy(confirmed=False)])
def test_other_store_errors_still_fail_the_write(tmp_path, store, error):
    row, handle = source(tmp_path)
    ctx = context(tmp_path, row)
    store.refusals = [error]
    with pytest.raises(type(error)):
        held_owner_messages(ctx)


def test_the_terminal_save_flushes_a_dirty_held_state_even_without_pending_acks(tmp_path, store, caplog):
    row, handle = source(tmp_path)
    ctx = context(tmp_path, row)
    store.refusals = [_busy()]
    held_owner_messages(ctx)
    assert ctx._held_state_dirty is True and ctx.__dict__.get("_pending_content_acks") is None
    limit = SimpleNamespace(tools=SimpleNamespace(_ctx=ctx))
    working_checkpoint.save_terminal(limit)
    assert ctx._held_state_dirty is False and handle in store.written[-1]["held_owner_messages"]
    # A store still busy at the terminal save keeps the warning and raises nothing.
    ctx._held_state_dirty = True
    store.refusals = [_busy()]
    caplog.set_level(logging.WARNING, logger=owner_relay.__name__)
    working_checkpoint.save_terminal(limit)
    assert ctx._held_state_dirty is True and any("not persisted" in rec.message for rec in caplog.records)
