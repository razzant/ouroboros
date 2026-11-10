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
