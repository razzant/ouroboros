"""Retire the first-act substitution window: quote possession is the authority."""
from types import SimpleNamespace
import queue

import pytest

from ouroboros.chat_chain import format_address, row_address
from ouroboros.owner_relay import held_owner_messages, quote_rows, retain_owner_message
from ouroboros.owner_mailbox import write_owner_message
from ouroboros.loop_round_limits import _drain_incoming_messages
from ouroboros import dialogue_store
from ouroboros.tools.control_relay import _tell_task
from tests.test_owner_relay import source, context


def test_two_notes_keep_their_own_words_and_distinct_relay_ids(tmp_path, monkeypatch):
    row, handle = source(tmp_path)
    ctx = context(tmp_path, row)
    events = []
    monkeypatch.setattr("ouroboros.tools.control_routing._emit_and_wait_for_routing",
                        lambda ctx, event: (events.append(event) or "live", {"status": "delivered"}))
    _tell_task(ctx, "receiver", "first")
    _tell_task(ctx, "receiver", "second")
    assert [event["message"] for event in events] == ["first", "second"]
    assert events[0]["relay_id"] != events[1]["relay_id"]
    assert all(event["quotes"] == [] for event in events)
    assert all(event["client_message_id"].startswith("agent-steer:") for event in events)


def test_written_owner_message_is_not_held_until_drained(tmp_path):
    row, origin = source(tmp_path)
    ctx = context(tmp_path, row)
    later = {**row, "text": "later words", "client_message_id": "later", "ts": "2026-10-01T01:00:00Z"}
    dialogue_store.append(tmp_path, "chat", later)
    handle = format_address(row_address(later))
    write_owner_message(tmp_path, later["text"], "teller", msg_id="mail", client_message_id="later")
    assert quote_rows(ctx, [handle])[1].startswith("QUOTE_NOT_HELD")
    messages = []
    _drain_incoming_messages(messages, queue.Queue(), tmp_path, "teller", None, set(), owner_ctx=ctx)
    assert set(held_owner_messages(ctx)) == {origin, handle}
    assert quote_rows(ctx, [handle])[0][0]["text"] == later["text"]
    assert handle in str(messages)


def test_retained_quotes_survive_compaction_and_exact_pause_restore(tmp_path, monkeypatch):
    from ouroboros.owner_wait import continuation_state, restore_continuation_state
    row, handle = source(tmp_path)
    ctx = context(tmp_path, row)
    held_owner_messages(ctx)
    ctx._owner_directives = []
    ctx._open_relays = {"r": {"relay_id": "r", "source": {"chat_id": 1}}}
    state = continuation_state(ctx, [], {}, {}, 0, [], set())
    def replayed(*_args, **_kwargs):
        raise AssertionError("possession replayed dialogue history")

    monkeypatch.setattr(dialogue_store, "transaction", replayed)  # possession no longer depends on history
    restored = context(tmp_path)
    restore_continuation_state(SimpleNamespace(_ctx=restored), state, [], {}, {}, set())
    assert quote_rows(restored, [handle])[0][0]["text"] == row["text"]
    assert restored._open_relays == ctx._open_relays


def test_continue_carries_only_rows_the_predecessor_held(tmp_path):
    from ouroboros.task_results import write_task_result, load_task_result
    from ouroboros.owner_continue import owner_sources
    row, handle = source(tmp_path)
    ctx = context(tmp_path, row)
    write_task_result(tmp_path, "teller", status="running", origin_message_text=row["text"],
                      origin_message_ref=ctx.task_metadata["origin_message_ref"])
    retain_owner_message(ctx, row)
    sources = owner_sources(tmp_path, load_task_result(tmp_path, "teller"), "teller")
    successor = context(tmp_path)
    successor.task_metadata = {"continuation": {"owner_sources": sources}}
    assert list(held_owner_messages(successor)) == [handle]
    assert quote_rows(successor, [handle])[0][0]["text"] == row["text"]


def test_attachment_manifest_and_each_quote_source_remain_exact(tmp_path):
    row, _ = source(tmp_path)
    row["attachments"] = [{"upload": "opaque-upload", "name": "plan.txt", "sha256": "a" * 64,
                           "kind": "file", "size": 5, "mime": "text/plain"}]
    ctx = context(tmp_path)
    retain_owner_message(ctx, row)
    handle = format_address(row_address(row))
    quotes, error = quote_rows(ctx, [handle, handle])
    assert not error and len(quotes) == 1
    assert quotes[0]["attachments"] == row["attachments"]
    assert quotes[0]["source"]["address"] == handle
    assert quotes[0]["source"]["client_message_id"] == row["client_message_id"]


@pytest.mark.parametrize("handles", ["row:made-up", ["row:made-up"], [42]])
def test_unknown_or_malformed_quote_refusal_names_the_held_handles(tmp_path, handles):
    row, handle = source(tmp_path)
    quotes, refusal = quote_rows(context(tmp_path, row), handles)
    assert not quotes and "QUOTE_NOT_HELD" in refusal and handle in refusal


def test_continue_retains_an_unanswered_return_address(tmp_path):
    from ouroboros.task_results import write_task_result, load_task_result
    from ouroboros.owner_continue import owner_sources
    from ouroboros.owner_relay import inject_relay, relay_envelope
    row, handle = source(tmp_path)
    ctx = context(tmp_path, row)
    write_task_result(tmp_path, "teller", "running", origin_message_text=row["text"],
                      origin_message_ref=ctx.task_metadata["origin_message_ref"])
    quotes, _ = quote_rows(ctx, [handle])
    relay = relay_envelope("return", quotes, chat_id=1)
    inject_relay(ctx, {"relay": relay, "owner_rows": [row], "text": "context"}, [])
    sources = owner_sources(tmp_path, load_task_result(tmp_path, "teller"), "teller")
    successor = context(tmp_path)
    successor.task_metadata = {"continuation": {"owner_sources": sources}}
    held_owner_messages(successor)
    assert successor._open_relays["return"] == relay


# Shared host doubles used by the registry-corruption tests too.
def _supervisor_ctx(tmp_path, notices):
    return SimpleNamespace(DRIVE_ROOT=tmp_path,
        RUNNING={"t-target": {"task": {"id": "t-target", "chat_id": 1}}}, PENDING=[],
        send_with_budget=lambda _chat, text, **kw: notices.append(text))


def _live_tool_ctx(tmp_path, supervisor_ctx, emitted, *, task_id="turn-1", metadata=None):
    from ouroboros.project_dialogue import build_owner_message_ref
    from supervisor.steering import _handle_steer_task
    ctx = context(tmp_path)
    ctx.is_direct_chat = True
    ctx.task_id = task_id
    ctx.task_metadata = dict(metadata or {})
    if ctx.task_metadata.get("client_message_id"):
        ctx.task_metadata["origin_message_ref"] = build_owner_message_ref(chat_id=1,
            client_message_id=ctx.task_metadata["client_message_id"], ts="2026-10-01T00:00:00Z", text=ctx.task_metadata.get("origin_message_text", ""))
    def dispatch(event):
        emitted.append(event)
        _handle_steer_task(event, supervisor_ctx)
    ctx.event_queue = SimpleNamespace(put_nowait=dispatch)
    return ctx


def held_quote(tmp_path, text, chat_id=1):
    from supervisor.message_bus import log_chat
    row = log_chat("in", chat_id, 1, text, client_message_id="quoted-owner", drive_root=tmp_path)
    return quote_rows(context(tmp_path, row), [format_address(row_address(row))])[0]
