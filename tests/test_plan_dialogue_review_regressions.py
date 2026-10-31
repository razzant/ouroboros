"""Real native preparation of a plan ask, found by review: a native seat's first request
leaves room for a real inspection, an owner's model choice is resolved before the dialogue
is fitted, the room snapshot is an observed (never a required) source, and the reply object
of a plan ask's native participant is read whole, with no extraction send."""
from __future__ import annotations

from dataclasses import replace

import pytest

from tests._review_substrate_shared import user_text as _user_text
from tests.test_plan_task_tool import CLEAN, _write, harness  # noqa: F401 — `harness` is the fixture
from ouroboros.utils import append_jsonl


@pytest.mark.parametrize('large', [False, True])
def test_native_first_request_has_room_for_a_real_inspection(harness, monkeypatch, large):  # noqa: F811
    from ouroboros import reviewer_window
    from ouroboros.review_native_episode import NativeToolRoundReviewExecutor
    from ouroboros.review_execution import ReviewAssignment
    from tests.test_native_tool_round_executor import _ScriptedLLM, _tool_call

    ctx = harness.make_ctx()
    ctx.current_chat_id = 1
    harness.state['slots'] = [replace(harness.state['slots'][0], subagent_id='native-reader',
                                      native_retrieval_override=True)]
    monkeypatch.setattr(reviewer_window, 'reviewer_context_window', lambda *a, **kw: 200_000)
    append_jsonl(harness.drive / 'logs/chat.jsonl', {'direction': 'out', 'chat_id': 1,
                 'text': 'Discussion and tradeoffs. ' * (50000 if large else 10)})
    substrate = harness.install({'s1': CLEAN})
    _write(ctx, ask='council')
    request, slot = substrate.calls[0]['request'], substrate.calls[0]['slots'][0]
    llm = _ScriptedLLM([{'tool_calls': [_tool_call('read_file', {'path': 'notes.md'})]}, {'content': CLEAN}])
    result = NativeToolRoundReviewExecutor(ReviewAssignment(request=request, slot=slot,
                        call_id='native-fit-regression'), llm=llm).execute()
    assert len(llm.calls) == 2
    assert result.usage['native_tool_receipts'][0]['path'] == 'notes.md'
    if large:
        assert 'exact omitted prefix' in request.slot_session_tasks[slot.slot_id]
    else:
        assert 'exact omitted prefix' not in request.slot_session_tasks[slot.slot_id]


def test_owner_model_override_is_resolved_before_dialogue_fit(harness, monkeypatch):  # noqa: F811
    from types import SimpleNamespace

    from ouroboros import model_wait, reviewer_window
    from ouroboros.review_execution import _messages_char_count

    ctx = harness.make_ctx()
    ctx.current_chat_id = 1
    harness.state['slots'] = [harness.state['slots'][0]]
    choices = {'reviewer:s1': {'model': 'm/small', 'model_account_override': '', 'use_local': False}}
    monkeypatch.setattr(model_wait, 'current_model_wait', lambda: SimpleNamespace(overrides=choices))
    monkeypatch.setattr(reviewer_window, 'reviewer_context_window',
                        lambda model, **kw: 100_000 if model == 'm/small' else 1_000_000)
    append_jsonl(harness.drive / 'logs/chat.jsonl', {'direction': 'out', 'chat_id': 1,
                 'text': 'Earlier explanations. ' * 30000})
    substrate = harness.install({'s1': CLEAN})
    _write(ctx, ask='council')
    assert len(substrate.calls) == 1 and substrate.calls[0]['slots'][0].model == 'm/small'
    assert _messages_char_count(substrate.calls[0]['request'].slot_messages['s1']) < 200000


def test_native_first_send_carries_no_required_source_manifest_for_the_room(harness):  # noqa: F811
    """The room snapshot is an OBSERVED source for sessions only: a native episode's first
    send names no required-source manifest, so it is never told to read the multi-MB
    snapshot and no capability delta is filed for not doing so."""
    from ouroboros.review_execution import ReviewAssignment
    from ouroboros.review_native_episode import NativeToolRoundReviewExecutor

    ctx = harness.make_ctx()
    ctx.current_chat_id = 1
    harness.state['slots'] = [replace(harness.state['slots'][0], subagent_id='native-reader',
                                      native_retrieval_override=True)]
    append_jsonl(harness.drive / 'logs/chat.jsonl', {'direction': 'out', 'chat_id': 1, 'text': 'Discussion and tradeoffs.'})
    substrate = harness.install({'s1': CLEAN})
    _write(ctx, ask='council')
    request, slot = substrate.calls[0]['request'], substrate.calls[0]['slots'][0]
    assert request.policy['observed_sources'][0]['root'] == 'artifact_store' and 'native_required_sources' not in request.policy
    executor = NativeToolRoundReviewExecutor(ReviewAssignment(request=request, slot=slot, call_id='native-observed'))
    _registry, _schemas, messages, _size = executor._open_episode(str(harness.workspace), harness.drive)
    text = _user_text(messages[-1]['content'])
    assert 'Required source manifest' not in text
    assert 'Discussion and tradeoffs.' in text


def test_a_native_participants_reply_object_is_read_whole_without_extraction(harness):  # noqa: F811
    """The real organ request on the real native executor: the participant's reply object
    reaches the record whole and no third (extraction) send is made — the scripted third
    entry is what an obedient light model would return for a findings-array contract."""
    from ouroboros.review_execution import ReviewAssignment
    from ouroboros.review_native_episode import NativeToolRoundReviewExecutor
    from ouroboros.tools import plan_ask
    from tests.test_native_tool_round_executor import _ScriptedLLM, _tool_call

    ctx = harness.make_ctx()
    ctx.current_chat_id = 1
    harness.state['slots'] = [replace(harness.state['slots'][0], subagent_id='native-reader',
                                      native_retrieval_override=True)]
    substrate = harness.install({'s1': CLEAN})
    _write(ctx, ask='council')
    request, slot = substrate.calls[0]['request'], substrate.calls[0]['slots'][0]
    llm = _ScriptedLLM([{'tool_calls': [_tool_call('read_file', {'path': 'notes.md'})]}, {'content': CLEAN},
                        {'content': '[]'}])
    result = NativeToolRoundReviewExecutor(ReviewAssignment(request=request, slot=slot,
                        call_id='native-reply-shape'), llm=llm).execute()
    facts = plan_ask.reply_facts_from_row(plan_ask._actor_row(
        {'raw_text': result.raw_text, 'status': 'ok', 'usage': result.usage}, slot))
    assert len(llm.calls) == 2
    assert facts['status'] == 'answered' and facts['reply']['prose'] == 'Looks sound to me.'


def test_an_ask_delivers_the_full_dialogue_to_a_packet_participant(harness):  # noqa: F811
    """The complete planning dialogue (no display cap) reaches a packet participant as the room
    wrote it and a relayed peer named by its provenance (the acceptance packet's own ledger of
    the owner's directives left with the task-acceptance machine)."""
    from ouroboros.owner_mailbox import write_task_message

    ctx = harness.make_ctx()
    ctx.current_chat_id = 1
    ctx._owner_directives = [{"source": "initial_user", "content": "Build the deck", "msg_id": "t:1"}]
    append_jsonl(harness.drive / "logs" / "chat.jsonl", {"direction": "in", "chat_id": 1,
                 "text": "Build the deck; key sk-abcdefghijklmnopqrstuvwxyz1234"})
    append_jsonl(harness.drive / "logs" / "chat.jsonl", {"direction": "out", "chat_id": 1,
                 "text": "Option A is faster but less flexible. " + "detail " * 6000})
    write_task_message(harness.drive, "Proposal from sibling: use a different format", task_id=ctx.task_id,
                       source_task_id="parent-9", provenance="peer_via_ancestor", relayed_from_task_id="sibling-7")
    substrate = harness.install({'s1': CLEAN, 's2': CLEAN, 's3': CLEAN})
    _write(ctx, ask='council')
    packet = "".join(_user_text(m["content"]) for m in substrate.calls[0]["request"].slot_messages["s1"]
                     if m.get("role") == "user")
    assert "## OWNER REQUIREMENTS AND DECISIONS" not in packet
    assert "Option A is faster but less flexible. " + "detail " * 6000 in packet
    assert "Build the deck; key sk-abcdefghijklmnopqrstuvwxyz1234" in packet
    assert "from task sibling-7, relayed by ancestor parent-9" in packet
