"""Real native preparation and typed replay boundaries found by review."""
from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from tests.test_plan_review_engine import harness as _harness, _call, _state, CLEAN, _finding
from tests.test_plan_review_reconciliation import _collect, _install_barrier_substrate
from ouroboros.utils import append_jsonl

harness = _harness


@pytest.mark.parametrize('large', [False, True])
def test_native_first_request_has_room_for_a_real_inspection(harness, monkeypatch, large):
    from ouroboros import reviewer_window
    from ouroboros.review_native_episode import NativeToolRoundReviewExecutor
    from ouroboros.review_execution import ReviewAssignment
    from tests.test_native_tool_round_executor import _ScriptedLLM, _tool_call

    ctx = harness.make_ctx()
    ctx.current_chat_id = 1
    harness.state['slots'] = [replace(harness.state['slots'][0], subagent_id='native-reader')]
    monkeypatch.setattr(reviewer_window, 'reviewer_context_window', lambda *a, **kw: 200_000)
    append_jsonl(harness.drive / 'logs/chat.jsonl', {'direction': 'out', 'chat_id': 1,
                 'text': 'Discussion and tradeoffs. ' * (50000 if large else 10)})
    substrate = harness.install({'s1': CLEAN})
    _call(ctx)
    request, slot = substrate.calls[0]['request'], substrate.calls[0]['slots'][0]
    llm = _ScriptedLLM([{'tool_calls': [_tool_call('read_file', {'path': 'notes.md'})]}, {'content': CLEAN}])
    result = NativeToolRoundReviewExecutor(ReviewAssignment(request=request, slot=slot,
                        call_id='native-fit-regression'), llm=llm).execute()
    assert len(llm.calls) == 2
    assert result.usage['native_tool_receipts'][0]['path'] == 'notes.md'
    if large:
        assert 'exact omitted prefix' in request.slot_session_tasks[slot.slot_id]


def test_owner_model_override_is_resolved_before_dialogue_fit(harness, monkeypatch):
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
    _call(ctx)
    assert len(substrate.calls) == 1 and substrate.calls[0]['slots'][0].model == 'm/small'
    assert _messages_char_count(substrate.calls[0]['request'].slot_messages['s1']) < 200000


def test_missing_historical_policy_is_a_typed_collection_failure(harness, monkeypatch):
    from ouroboros.task_results import record_plan_review_wave
    from ouroboros.tools.plan_review_artifacts import persist_wave, read_wave, authority_wave

    calls = []
    _install_barrier_substrate(monkeypatch, calls)
    ctx = harness.make_ctx()
    _call(ctx)
    first = _state(harness)['waves'][-1]
    exact = read_wave(harness.drive, ctx.task_id, first['wave_artifact'])
    exact.pop('request_policy')
    exact.pop('slot_prompt_chars')
    hot = {**first, 'wave_artifact': persist_wave(harness.drive, ctx.task_id, exact)}
    hot.pop('request_policy', None)
    hot.pop('slot_prompt_chars', None)
    record_plan_review_wave(harness.drive, ctx.task_id, hot)
    before = _state(harness)['cycles_paid']
    text = _collect(ctx, first['request_fingerprint'])
    assert 'PLAN_REVIEW_SOURCE_UNAVAILABLE' in text
    assert len(calls) == 1 and _state(harness)['cycles_paid'] == before
    after = authority_wave(harness.drive, ctx.task_id, _state(harness)['waves'][-1])
    assert 'request_policy' not in after


def test_paid_same_author_retry_discloses_its_recorded_snapshot(harness, monkeypatch):
    from ouroboros.tools import plan_review
    from tests.test_plan_review_epoch import _patch_health

    _patch_health(monkeypatch, lambda slots: {})
    monkeypatch.setattr(plan_review, '_plan_review_slots', lambda default_effort='': [
        replace(slot, effort=default_effort or slot.effort, declared_effort=default_effort)
        for slot in harness.state['slots']])
    substrate = harness.install({'s1': json.dumps([_finding('n', 'blocking', breaks='claim_1')]),
                                  's2': CLEAN, 's3': CLEAN})
    ctx = harness.make_ctx()
    ctx.current_chat_id = 1
    append_jsonl(harness.drive / 'logs/chat.jsonl', {'direction': 'in', 'chat_id': 1, 'text': 'First words'})
    _call(ctx, reviewer_effort='low')
    first = _state(harness)['waves'][-1]
    append_jsonl(harness.drive / 'logs/chat.jsonl', {'direction': 'in', 'chat_id': 1, 'text': 'Later clarification'})
    output = _call(ctx, reviewer_effort='max')
    second = _state(harness)['waves'][-1]
    assert len(substrate.calls) == 2 and _state(harness)['cycles_paid'] == 2
    assert first['dialogue_source_ref'] == second['dialogue_source_ref']
    assert 'Later messages are not claimed reviewed' in output
    assert first['dialogue_source_ref']['sha256'] in output


def test_paid_lost_only_wave_at_cap_uses_existing_exhausted_exit(harness, monkeypatch):
    from ouroboros import review_substrate
    from ouroboros.task_results import plan_review_gate_projection
    from tests.test_plan_review_engine import DECK_SPEC

    monkeypatch.setenv('OUROBOROS_REVIEW_MAX_CYCLES', '1')
    sends = []
    def missing(request, **kwargs):
        sends.append(request.reconcile_only)
        return SimpleNamespace(actors=[])
    monkeypatch.setattr(review_substrate, 'run_review_request', missing)
    ctx = harness.make_ctx()
    _call(ctx)
    state = _state(harness)
    assert state['cycles_paid'] == 1 and state['waves'][-1]['custody_pending']
    result = _call(ctx, spec={**DECK_SPEC, 'in_scope': ['A revised plan']})
    assert 'PLAN_REVIEW_CYCLES_EXHAUSTED' in result
    state = _state(harness)
    assert state['cycles_paid'] == 1 and sends == [False]
    assert state['waves'][-1]['custody_pending'] is True
    assert {actor['operation_state'] for actor in state['waves'][-1]['actors']} == {'custody_lost'}
    gate = plan_review_gate_projection(state, 'blocking')
    assert gate['allow'] and gate['status'] == 'cycles_exhausted'


@pytest.mark.parametrize('paid,cycles,actors', [
    (False, 0, [{'operation_state': 'custody_lost'}]),
    (True, 1, []), (True, 1, [{}]), (True, 1, [None]),
    (True, 1, [{'operation_state': 'custody_lost'}, {'operation_state': 'in_flight'}]),
    (False, 0, [{'operation_state': 'pending_dispatch'}]),
])
def test_uncertain_or_running_panels_keep_committed_capacity(paid, cycles, actors):
    from ouroboros.tools.plan_review_collect import in_flight_hold

    state = {'cycles_paid': cycles, 'current_attempt': {'fingerprint': 'original'},
             'waves': [{'request_fingerprint': 'original', 'custody_pending': True, 'paid': paid, 'actors': actors}]}
    assert in_flight_hold(state, fingerprint='revised', cap=1)


def test_native_first_send_carries_no_required_source_manifest_for_the_room(harness):
    """The room snapshot is an OBSERVED source for sessions only: a native episode's first
    send names no required-source manifest, so it is never told to read the multi-MB
    snapshot and no capability delta is filed for not doing so."""
    from ouroboros.review_execution import ReviewAssignment
    from ouroboros.review_native_episode import NativeToolRoundReviewExecutor

    ctx = harness.make_ctx()
    ctx.current_chat_id = 1
    harness.state['slots'] = [replace(harness.state['slots'][0], subagent_id='native-reader')]
    append_jsonl(harness.drive / 'logs/chat.jsonl', {'direction': 'out', 'chat_id': 1, 'text': 'Discussion and tradeoffs.'})
    substrate = harness.install({'s1': CLEAN})
    _call(ctx)
    request, slot = substrate.calls[0]['request'], substrate.calls[0]['slots'][0]
    assert request.policy['observed_sources'][0]['root'] == 'artifact_store' and 'native_required_sources' not in request.policy
    executor = NativeToolRoundReviewExecutor(ReviewAssignment(request=request, slot=slot, call_id='native-observed'))
    _registry, _schemas, messages, _size = executor._open_episode(str(harness.workspace), harness.drive)
    assert 'Required source manifest' not in messages[-1]['content']
    assert 'Discussion and tradeoffs.' in messages[-1]['content']
