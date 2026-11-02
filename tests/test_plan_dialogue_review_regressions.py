"""Real native preparation of a plan ask, found by review: a native seat's first request
leaves room for a real inspection, an owner's model choice is resolved before the dialogue
is fitted, the room snapshot is an observed (never a required) source whose actual reads reach
the record and its readers, the rows a delivery carried inline reach them only as far as that
delivery went, and the reply object of a plan ask's native participant is read whole, with no
extraction send."""
from __future__ import annotations

import re
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
    """The room snapshot is an OBSERVED source: a native episode's first send names no
    required-source manifest, so it is never told to read the multi-MB snapshot and no
    capability delta is filed for not doing so."""
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


def _room_conversation(drive, ctx):
    """Two conversation rows the delivery carries inline and one progress row only the snapshot file holds."""
    from ouroboros import dialogue_store

    ctx.current_chat_id = 1
    for row in ({'direction': 'in', 'text': 'Use the agreed outline'}, {'direction': 'out', 'text': 'Five slides it is'}):
        dialogue_store.append(drive, 'chat', {'chat_id': 1, **row})
    dialogue_store.append(drive, 'progress', {'chat_id': 1, 'content': 'PROGRESS ' * 40})


def _native_answer(slot, script, *, call_id='native-room', prepare=None):
    """The seat's answer as the substrate returns it: the real executor over the organ's own request,
    reading the snapshot the ask actually declared. Every scripted round is exactly one paid send."""
    from ouroboros.review_execution import ReviewAssignment
    from ouroboros.review_native_episode import NativeToolRoundReviewExecutor
    from tests.test_native_tool_round_executor import _ScriptedLLM

    def answer(request):
        room = request.policy['observed_sources'][0]
        if prepare:
            prepare(room)
        llm = _ScriptedLLM(script(room))
        result = NativeToolRoundReviewExecutor(ReviewAssignment(request=request, slot=slot, call_id=call_id), llm=llm).execute()
        assert len(llm.calls) == len(script(room)), 'an observed source buys no further send'
        return {'raw_text': result.raw_text, 'usage': result.usage}
    return answer


def _reads(*args_list):
    from tests.test_native_tool_round_executor import _tool_call

    return [{'tool_calls': [_tool_call('read_file', args, f'r{i}')]} for i, args in enumerate(args_list)] + [{'content': CLEAN}]


@pytest.mark.parametrize('case', ['part', 'whole', 'unread', 'another source', 'changed bytes'])
def test_a_native_seats_room_reads_reach_the_record_its_mind_and_its_detail(harness, case):  # noqa: F811
    """Actual host-observed reads of the exact declared snapshot survive the real native route, settlement and the
    plan record to the mind's read and the task detail, under `host_observed`. Unread, or read elsewhere, says 0 of
    the snapshot and still reports the rows it had inline; neither buys a send, a capability delta or a changed
    answer. Changed bytes at the same path are not the declared snapshot."""
    import pathlib

    from tests._store_shared import seed_result
    from tests.test_plan_record_gateway import _detail
    from tests.test_plan_task_tool import _call, _record

    seed_result(harness.drive, 'task-1', {'task_id': 'task-1', 'status': 'running'})
    ctx = harness.make_ctx()
    _room_conversation(harness.drive, ctx)
    slot = replace(harness.state['slots'][0], subagent_id='native-reader', native_retrieval_override=True)
    harness.state['slots'] = [slot]

    def script(room):
        target = {'root': 'artifact_store', 'path': room['path']}
        return _reads(*{'part': [{**target, 'max_lines': 1}], 'whole': [target], 'unread': [], 'another source': [{'path': 'notes.md'}],
                        'changed bytes': [target]}[case])

    def prepare(room):
        if case == 'changed bytes':
            pathlib.Path(room['file']).write_text('rewritten after the ask declared it\n', encoding='utf-8')

    harness.install({'s1': _native_answer(slot, script, prepare=prepare)})
    _write(ctx, ask='council')
    [seat] = _record(harness)['asks'][0]['participants']
    assert seat['status'] == 'answered' and seat['reply']['prose'] == 'Looks sound to me.'
    assert seat['capability_delta'] == [] and seat['host_file_read_attestation'] == 'host_observed'
    assert seat['room_delivery'] == {'conversation_rows': 2, 'conversation_inline_rows': 2, 'other_rows': 1}
    read = seat['room_read_coverage']
    assert read['provenance'] == 'host_observed'
    total = read['complete_chars']
    assert total > 0 and {'part': 0 < read['covered_chars'] < total, 'whole': read['covered_chars'] == total,
                          'unread': read['covered_chars'] == 0, 'another source': read['covered_chars'] == 0,
                          'changed bytes': read['covered_chars'] == 0}[case]
    assert read['status'] == ('complete' if case == 'whole' else 'incomplete')
    text, _result = _call(ctx)
    assert '2 of 2 conversation rows inline' in text
    assert f"file reads covered {read['covered_chars']} of {total} chars (host_observed)" in text
    [shown] = _detail(harness.drive, 'task-1')['plan_record']['asks'][0]['participants']
    assert shown['room_read_coverage'] == read and shown['room_delivery'] == seat['room_delivery']


def test_a_paged_brief_keeps_its_own_manifest_and_never_reads_as_the_room(harness, monkeypatch, tmp_path):  # noqa: F811
    """A brief too large for the seat's window becomes that seat's own required source. Reading it whole is no read
    of the room, and the room's own observation survives apart; a sibling sharing the wave's one request (a native
    seat that did not page, a session folding its journal) keeps its own manifest and its own room fact."""
    from ouroboros import review_native_episode
    from ouroboros.review_execution import ReviewAssignment
    from ouroboros.review_native_episode import NativeToolRoundReviewExecutor
    from ouroboros.review_session_reads import session_read_facts
    from ouroboros.tools import plan_ask
    from tests.test_native_tool_round_executor import _ScriptedLLM, _tool_call
    from tests.test_review_session_reads import _claude_event, _journal

    ctx = harness.make_ctx()
    _room_conversation(harness.drive, ctx)
    harness.state['slots'] = [replace(harness.state['slots'][0], subagent_id='native-reader', native_retrieval_override=True)]
    substrate = harness.install({'s1': CLEAN})
    _write(ctx, ask='council')
    request, slot = substrate.calls[0]['request'], substrate.calls[0]['slots'][0]
    declared = dict(request.policy)
    room = request.policy['observed_sources'][0]

    class ReadsItsBrief(_ScriptedLLM):
        def chat(self, **kwargs):
            if not self.calls:
                brief = {'root': 'artifact_store', 'path': executor._initial_source['path']}
                self.script = [{'tool_calls': [_tool_call('read_file', brief, 'brief'),
                                               _tool_call('read_file', {'root': 'artifact_store', 'path': room['path'], 'max_lines': 1}, 'room')]},
                               {'content': CLEAN}]
            return super().chat(**kwargs)

    with monkeypatch.context() as paged:
        paged.setattr(review_native_episode, 'native_landing_at', lambda bound: 1)
        llm = ReadsItsBrief([])
        executor = NativeToolRoundReviewExecutor(ReviewAssignment(request=request, slot=slot, call_id='paged'), llm=llm)
        result = executor.execute()
    [brief] = result.usage['native_read_coverage']['sources']
    assert brief['path'] == executor._initial_source['path'] and brief['status'] == 'complete'
    assert 'native_incomplete' not in result.usage and not result.usage.get('capability_delta')
    row = plan_ask._actor_row({'raw_text': result.raw_text, 'status': 'ok', 'usage': result.usage}, slot)
    [observed] = result.usage['observed_read_coverage']['sources']
    assert row['room_read_coverage'] == {'status': 'incomplete', 'covered_chars': observed['covered_chars'],
                                         'complete_chars': room['complete_chars'], 'provenance': 'host_observed'}
    assert 0 < observed['covered_chars'] < room['complete_chars'] != brief['complete_chars']

    assert request.policy == declared and 'native_required_sources' not in request.policy
    sibling_llm = _ScriptedLLM([{'content': CLEAN}])
    sibling = NativeToolRoundReviewExecutor(ReviewAssignment(request=request, slot=slot, call_id='sibling'), llm=sibling_llm).execute()
    assert 'Required source manifest' not in _user_text(sibling_llm.calls[0]['messages'][-1]['content'])
    assert plan_ask._actor_row({'raw_text': sibling.raw_text, 'status': 'ok', 'usage': sibling.usage}, slot)['room_read_coverage']['covered_chars'] == 0
    assert not sibling.usage.get('capability_delta')
    _journal(tmp_path, _claude_event(room['file'], offset=1, limit=1), run='session')
    session = session_read_facts(str(tmp_path / 'session'), request.policy, session_root=str(harness.workspace))
    assert [source['path'] for source in session['native_read_coverage']['sources']] == [room['path']] and 'native_incomplete' not in session


def test_an_unread_brief_is_still_its_own_required_gap_and_never_the_rooms(harness, monkeypatch):  # noqa: F811
    """What a paged brief owes is unchanged (its own capability delta), and that gap never reads as the room's."""
    from ouroboros import review_native_episode
    from ouroboros.review_execution import ReviewAssignment
    from ouroboros.review_native_episode import NativeToolRoundReviewExecutor
    from ouroboros.tools import plan_ask
    from tests.test_native_tool_round_executor import _ScriptedLLM

    ctx = harness.make_ctx()
    _room_conversation(harness.drive, ctx)
    harness.state['slots'] = [replace(harness.state['slots'][0], subagent_id='native-reader', native_retrieval_override=True)]
    substrate = harness.install({'s1': CLEAN})
    _write(ctx, ask='council')
    request, slot = substrate.calls[0]['request'], substrate.calls[0]['slots'][0]
    monkeypatch.setattr(review_native_episode, 'native_landing_at', lambda bound: 1)
    result = NativeToolRoundReviewExecutor(ReviewAssignment(request=request, slot=slot, call_id='unread'),
                                           llm=_ScriptedLLM([{'content': CLEAN}])).execute()
    assert result.usage['native_incomplete'] == 'required_source_coverage_incomplete'
    assert [delta['reason'] for delta in result.usage['capability_delta']] == ['native_required_source_coverage_incomplete']
    row = plan_ask._actor_row({'raw_text': result.raw_text, 'status': 'ok', 'usage': result.usage}, slot)
    assert row['room_read_coverage']['complete_chars'] == request.policy['observed_sources'][0]['complete_chars']
    assert row['room_read_coverage']['covered_chars'] == 0


def test_a_seat_with_no_inspection_measured_nothing_of_the_room(harness, monkeypatch):  # noqa: F811
    """A failed registry reads nothing: the fold says unobserved and the plan row claims no count, where the older
    projector read the declared row as 0 of N."""
    from ouroboros.review_execution import ReviewAssignment, ReviewRouteUnavailable
    from ouroboros.review_native_episode import NativeToolRoundReviewExecutor
    from ouroboros.tools import plan_ask
    from tests.test_native_tool_round_executor import _ScriptedLLM

    def broken(self, root, drive_root):
        raise ReviewRouteUnavailable('no schemas', code='native_inspection_unavailable')

    ctx = harness.make_ctx()
    _room_conversation(harness.drive, ctx)
    harness.state['slots'] = [replace(harness.state['slots'][0], subagent_id='native-reader', native_retrieval_override=True)]
    substrate = harness.install({'s1': CLEAN})
    _write(ctx, ask='council')
    request, slot = substrate.calls[0]['request'], substrate.calls[0]['slots'][0]
    monkeypatch.setattr(NativeToolRoundReviewExecutor, '_inspection_registry', broken)
    executor = NativeToolRoundReviewExecutor(ReviewAssignment(request=request, slot=slot, call_id='no-inspection', custody_root=harness.drive),
                                             llm=_ScriptedLLM([]))
    with pytest.raises(ReviewRouteUnavailable):
        executor.execute()
    usage = executor.failure_custody()
    coverage = usage['native_read_coverage']
    assert (coverage['status'], coverage['reason']) == ('unobserved', 'native_inspection_unavailable')
    assert 'room_read_coverage' not in plan_ask._actor_row({'status': 'error', 'error': 'no schemas', 'usage': usage}, slot)


@pytest.mark.parametrize('stored', [{'status': '', 'covered_chars': 0, 'complete_chars': 1234, 'provenance': ''},
                                    {'status': 'unobserved', 'covered_chars': 0, 'complete_chars': 1234, 'provenance': 'harness_observed'},
                                    {'status': 'incomplete', 'covered_chars': 10, 'complete_chars': 900000, 'provenance': ''}],
                         ids=['blank row', 'unobserved row', 'paged brief under no provenance'])
def test_a_stored_row_that_measured_nothing_says_nothing_to_the_mind(stored):
    """Records keep what an older projector stored, and its false rows never read as a count of the room."""
    from ouroboros.tools.plan_review import _room_facts

    assert _room_facts({'room_read_coverage': stored}) == ''
    assert _room_facts({'room_read_coverage': {**stored, 'status': 'incomplete', 'provenance': 'host_observed'}}) != ''


@pytest.mark.parametrize('unresolved', [False, True])
def test_pending_inline_preparation_is_not_confirmed_delivery(unresolved):
    from ouroboros.tools.plan_review import _room_facts

    participant = {'status': 'pending', 'unresolved': unresolved,
                   'room_delivery': {'conversation_rows': 2, 'conversation_inline_rows': 2, 'other_rows': 1}}
    assert 'counted for a delivery that may not have been sent' in _room_facts(participant)


def _record_facts_lines(drive, task_id):
    """The owner's Record facts lines for the task's plan: the real task detail, read by the unmodified SPA module."""
    import json
    import pathlib
    import shutil
    import subprocess

    from tests.test_plan_record_gateway import _detail

    node = shutil.which('node')
    assert node, 'Node is required for the Record facts reader'
    script = ("import { planRecordGroup } from './modules/plan_record_presentation.js';\n"
              "let raw = ''; for await (const chunk of process.stdin) raw += chunk;\n"
              "const { owner, detail } = JSON.parse(raw);\n"
              "process.stdout.write(JSON.stringify(planRecordGroup(owner, detail.plan_record, detail).attempts.map((a) => a.detailText)));\n")
    done = subprocess.run([node, '--input-type=module', '-e', script], cwd=pathlib.Path(__file__).resolve().parents[1] / 'web',
                          input=json.dumps({'owner': task_id, 'detail': _detail(drive, task_id)}), capture_output=True, text=True,
                          encoding='utf-8', timeout=60)
    assert done.returncode == 0, done.stderr
    return [line for text in json.loads(done.stdout) for line in text.split('\n')]


def _late_custody(drive, request, slot, operation, *, refused):
    """What the real coordinator and engine leave for a released session seat: its prompt retained under the coordinator's
    own identity, the engine's start request and, for a refusal, the engine's definite refusal of that start."""
    from dataclasses import asdict

    from ouroboros import delegate_custody as custody
    from ouroboros.observability import persist_call
    from ouroboros.review_dispatch import review_operation_binding, review_reconciliation_identity

    request.reconciliation_identity = review_reconciliation_identity(request, [slot], root_task_id='task-1')
    persist_call(drive, task_id='task-1', call_id=f'{operation}_prompt', call_type='plan_review_prompt',
                 payload={'request': asdict(request), 'slot': asdict(slot)},
                 manifest={'surface': 'plan_review', 'slot_id': slot.slot_id, 'model': slot.model,
                           'review_operation_binding': review_operation_binding(request, slot, operation)})
    start = {'invocation_id': f'inv-{operation}', 'task_id': 'task-1'}
    custody.emit(drive, custody.START_REQUESTED, {**start, 'operation_id': operation, 'slot_id': slot.slot_id, 'surface': 'plan_review',
                                                  'root_task_id': 'task-1', 'request': {'prompt': 'review', 'harnesses': ['cursor']}})
    if refused:
        custody.emit(drive, custody.START_FAILED, {**start, 'definite': True, 'error': 'the engine refused this start'})


@pytest.mark.parametrize('outcome', ['answered', 'refused', 'unknown', 'refused late', 'unknown late'])
def test_a_seats_inline_rows_reach_its_readers_only_as_far_as_its_delivery_went(harness, outcome):  # noqa: F811
    """The rows a seat's delivery carries inline are counted when the ask is prepared, before any send, and the record keeps
    that count. Both readers, the mind's plan_task() and Record facts over the real task detail, join it to the seat's own
    dispatch outcome. A definite refusal delivered nothing, whether it came back at dispatch or the real cold collector
    settled it later from the engine's refusal of the recorded start. An unknown outcome (an exception after the hand-off, a
    start no run was ever bound to) names the rows as counted for a delivery that may not have been sent. An answer keeps
    them as delivered."""
    from tests._store_shared import seed_result
    from tests.test_plan_task_tool import _call, _record, _slots

    seed_result(harness.drive, 'task-1', {'task_id': 'task-1', 'status': 'running'})
    ctx = harness.make_ctx()
    _room_conversation(harness.drive, ctx)
    harness.state['slots'] = _slots(('s1', 'm/a', 'session'))

    def lost(_request):
        raise ConnectionError('the engine connection reset after the hand-off')

    late = {'status': 'error', 'raw_text': '', 'operation_state': 'pending_dispatch', 'late_result_pending': True}
    sub = harness.install({'s1': {'answered': CLEAN, 'unknown': lost, 'refused': {
        'status': 'not_dispatched', 'operation_state': 'not_dispatched', 'error': 'Owner deadline exhausted before physical review dispatch'},
    }.get(outcome, late)})
    _write(ctx, ask='council')
    prepared = {'conversation_rows': 2, 'conversation_inline_rows': 2, 'other_rows': 1}
    if outcome.endswith('late'):
        [seat] = _record(harness)['asks'][0]['participants']
        assert seat['status'] == 'pending' and seat['room_delivery'] == prepared, seat
        _late_custody(harness.drive, sub.calls[0]['request'], sub.calls[0]['slots'][0], seat['operation_id'], refused=outcome == 'refused late')
    text, _result = _call(ctx)  # the mind's free read, which collects a late seat at $0
    [seat] = _record(harness)['asks'][0]['participants']
    assert seat['room_delivery'] == prepared, 'the record keeps what preparation counted'
    assert (seat['status'], bool(seat.get('unresolved'))) == {
        'answered': ('answered', False), 'refused': ('not_dispatched', False), 'unknown': ('pending', True),
        'refused late': ('not_dispatched', False), 'unknown late': ('pending', True)}[outcome], seat
    if outcome.endswith('late'):  # the collector read the engine's custody itself, never a gap of this harness
        refused = outcome == 'refused late'
        assert ('definitively refused' if refused else 'no run was ever bound') in str(
            seat.get('not_dispatched_reason' if refused else 'error')), seat
    mind = next(line for line in text.splitlines() if 's1 (m/a)' in line)
    assert {'answered': ': answered', 'refused': ': not sent — ', 'unknown': ': pending — outcome unknown ('}[outcome.split()[0]] in mind, mind
    readers = {'mind': re.findall(r'\[room: [^\]]*\]', mind),
               'owner': [line for line in _record_facts_lines(harness.drive, 'task-1') if line.startswith('Participant s1 room: ')]}
    rows, maybe = '2 of 2 conversation rows inline', 'counted for a delivery that may not have been sent'
    assert readers == {
        'answered': {'mind': [f'[room: {rows}]'], 'owner': [f'Participant s1 room: {rows}, 1 other rows only in the snapshot file']},
        'refused': {'mind': [], 'owner': []},
        'unknown': {'mind': [f'[room: {rows}, {maybe}]'],
                    'owner': [f'Participant s1 room: {rows}, 1 other rows only in the snapshot file · {maybe}']},
    }[outcome.split()[0]], readers
    assert len(sub.calls) == 1, 'nothing is sent again'


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
