"""Author completion is explicit after a hold, with exact bytes and observed work."""
import hashlib
import json
import queue
from types import SimpleNamespace

import pytest

import ouroboros.loop as loop
from ouroboros.loop_delivery import selected_completion_text
from ouroboros.outcomes import derive_loop_outcome
from ouroboros.tools.registry import ToolRegistry


def finish(answer=None, *, action='finish', **kw):
    arguments = {'action': action, **kw}
    if answer is not None:
        arguments['answer'] = answer
    return {'role': 'assistant', 'content': None, 'tool_calls': [{
        'id': 'complete', 'type': 'function',
        'function': {'name': 'finish_task', 'arguments': json.dumps(arguments)},
    }]}


def hold_first_delivery(monkeypatch):
    """Hold the first delivery once at the host's result-review step, then let it pass.

    ``result_review_at_delivery`` is the delivery-time host step whose ``True`` holds
    the answer for another turn; it records the content it was asked about."""
    seen = []

    def review(_ctx, content, _trace, _hold):
        seen.append(content)
        return len(seen) == 1

    monkeypatch.setattr('ouroboros.tools.review_result.result_review_at_delivery', review)
    return seen


@pytest.fixture
def turn(tmp_path, monkeypatch):
    monkeypatch.setenv('OUROBOROS_TASK_REVIEW_MODE', 'off')
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.is_direct_chat = True
    calls, notes = [], []
    def run(responses):
        iterator = iter(responses)
        def respond(_llm, messages, *_a, **_kw):
            calls.append(json.loads(json.dumps(messages)))
            response = next(iterator)
            return (response() if callable(response) else response), 0.0
        monkeypatch.setattr(loop, 'call_llm_with_retry', respond)
        return loop.run_llm_loop([{'role': 'user', 'content': 'Please help'}], registry,
            SimpleNamespace(default_model=lambda: 'test-model'), tmp_path / 'logs',
            lambda text, **_kw: notes.append(text), queue.Queue(), task_id='author-task', drive_root=tmp_path)
    return registry, calls, notes, run


def test_plain_reply_and_voluntary_selection_need_one_round(turn):
    registry, calls, notes, run = turn
    text, usage, trace = run([finish('Hi')])
    assert text == 'Hi' and len(calls) == 1
    assert trace['tool_calls'][0]['completion_control'] is True
    assert 'task_completion' not in trace


@pytest.mark.parametrize('statement', [None, 'Solo would have avoided coordination; parallel checks helped.'])
def test_accepted_completion_retains_only_an_authored_swarm_comparison(turn, tmp_path, statement):
    from ouroboros.swarm_receipt import publish
    from ouroboros.task_results import write_task_result

    registry, calls, notes, run = turn
    write_task_result(tmp_path, 'author-task', 'running')
    schema = registry.get_schema_by_name('finish_task')['function']['parameters']['properties']
    assert schema['swarm_comparison']['type'] == 'string'
    kwargs = {'swarm_comparison': statement} if statement is not None else {}
    text, usage, trace = run([finish('Done.', **kwargs)])
    assert text == 'Done.' and len(calls) == 1
    comparison = publish(tmp_path, 'author-task')['mind_comparison']
    if statement is None:
        assert comparison['state'] == 'not_recorded'
    else:
        assert comparison == {'source': 'task.swarm_comparison', 'author': 'mind', 'value': statement}
    assert registry.get_schema_by_name('finish_task')['function']['parameters']['properties'] == schema


def test_held_completion_does_not_publish_its_swarm_comparison(turn, tmp_path, monkeypatch):
    from ouroboros.task_results import load_task_result, write_task_result

    registry, calls, notes, run = turn
    write_task_result(tmp_path, 'author-task', 'running')
    hold_first_delivery(monkeypatch)
    text, usage, trace = run([finish('Held answer', swarm_comparison='An unaccepted comparison'), finish('Done.')])
    assert text == 'Done.' and len(calls) == 2
    assert 'swarm_comparison' not in load_task_result(tmp_path, 'author-task')


@pytest.mark.parametrize('statement, accepted', [('An authored comparison', True), (17, False)])
def test_swarm_comparison_argument_is_text_without_deciding_completion(turn, statement, accepted):
    from ouroboros.tools.control_runtime import _finish_task

    registry, calls, notes, run = turn
    result = _finish_task(registry._ctx, action='finish', answer='Done.', swarm_comparison=statement)
    assert ('completion_requested' in result) is accepted
    assert ('swarm_comparison must be text' in result) is not accepted


def test_retained_report_interim_then_short_complete_selection(turn, monkeypatch):
    registry, calls, notes, run = turn
    review_calls = hold_first_delivery(monkeypatch)
    report = 'The retained report is intentionally much longer than the final correction.'
    text, usage, trace = run([{'content': report}, {'content': 'Checking.'},
        {'content': 'Checking.'}, finish('No.')])
    assert text == 'No.' and review_calls == [report, 'No.']
    assert len(calls) == 4
    for previous, current in zip(calls[1:], calls[2:]):
        assert current[:len(previous)] == previous
        assert len(current) > len(previous) and current[-1]['role'] == 'user'
    assert sum(row.get('role') == 'assistant' and row.get('content') == 'Checking.' for row in calls[-1]) == 2


def test_select_latest_whole_held_response_without_regeneration(turn, monkeypatch):
    registry, calls, notes, run = turn
    seen = hold_first_delivery(monkeypatch)
    replacement = 'Correction: yes.'
    text, _, _ = run([{'content': 'Original long report.'}, {'content': replacement},
        finish(answer_sha256=hashlib.sha256(replacement.encode()).hexdigest())])
    assert text == replacement and seen == ['Original long report.', replacement]


@pytest.mark.parametrize('position', [0, 1])
@pytest.mark.parametrize('failure', [False, True])
def test_same_response_work_result_requires_actual_observation(turn, position, failure):
    registry, calls, notes, run = turn
    registry.override_handler('chat_history', lambda _ctx, **kw: 'ERROR: TOOL_ERROR: failed read' if failure else 'read result')
    response = finish('Chosen before the result')
    response['tool_calls'].insert(position, {'id': 'read', 'type': 'function', 'function': {
        'name': 'chat_history', 'arguments': '{}'}})
    text, _, trace = run([response, finish('Now informed')])
    assert text == 'Now informed' and len(calls) == 2
    assert trace['completion_refusals'][0]['reason'].startswith('unobserved_tool_results')
    assert 'read result' in str(calls[-1]) or 'failed read' in str(calls[-1])


@pytest.mark.parametrize('name', ['enable_tools', 'list_available_tools'])
@pytest.mark.parametrize('failure', [False, True])
def test_schema_bookkeeping_exempts_only_successful_results(turn, name, failure):
    registry, calls, notes, run = turn
    response = finish('Chosen before the result')
    arguments = {'tools': 'finish_task'} if name == 'enable_tools' else {}
    response['tool_calls'].insert(0, {'id': 'schema', 'type': 'function', 'function': {
        'name': name, 'arguments': '{bad json' if failure else json.dumps(arguments)}})
    text, _, trace = run([response, finish('Now informed')])
    assert trace['tool_calls'][0]['is_error'] is failure
    assert len(calls) == (2 if failure else 1)
    assert text == ('Now informed' if failure else 'Chosen before the result')
    if failure:
        assert trace['completion_refusals'][0]['reason'].startswith('unobserved_tool_results')
    else:
        assert not trace.get('completion_refusals')


@pytest.mark.parametrize('name, arguments', [('chronicle_write', {'kind': 'page', 'text': 'What I did'}),
                                             ('memory_mark', {}),
                                             ('update_scratchpad', {'content': 'Next: the docs'})])
@pytest.mark.parametrize('answer, held', [('written', False), ('⚠️ PARTIAL_WRITE: two decisions not recorded', True),
                                          ('⚠️ TOOL_ERROR: write failed', True)])
def test_a_receipt_write_rides_with_the_answer_unless_it_said_more(turn, name, arguments, answer, held):
    """The page, a note or an addressed message written in the finishing response does not cost
    the answer a round (owner decision 27.10); a warning or a failure from it is read first."""
    registry, calls, notes, run = turn
    registry.override_handler(name, lambda _ctx, **kw: answer)
    response = finish('The asked fact')
    response['tool_calls'].insert(0, {'id': 'receipt', 'type': 'function', 'function': {
        'name': name, 'arguments': json.dumps(arguments)}})
    text, _, trace = run([response, finish('Read it first')])
    assert len(calls) == (2 if held else 1)
    assert text == ('Read it first' if held else 'The asked fact')
    assert bool(trace.get('completion_refusals')) is held


def test_only_an_ok_receipt_write_skips_the_read_before_finish():
    from ouroboros.tool_capabilities import COMPLETION_RECEIPT_TOOLS, completion_observation_calls

    def call(tool, code='OK', is_error=False):
        return {'tool': tool, 'tool_result_code': code, 'is_error': is_error}

    assert completion_observation_calls([call(name) for name in sorted(COMPLETION_RECEIPT_TOOLS)]) == []
    held = [call('tell_task', 'LEGACY_WARNING'), call('forward_to_worker', 'TOOL_ERROR', True),
            call('update_identity'), call('update_world'), call('read_file'), call('verify_and_record'),
            call('a_tool_added_later'), call('chronicle_write', None)]
    assert completion_observation_calls(held) == held


def test_stop_avoids_reviews_nudges_and_question_parking(turn, monkeypatch):
    registry, calls, notes, run = turn
    monkeypatch.setattr('ouroboros.tools.review_result.result_review_at_delivery',
                        lambda *a, **kw: pytest.fail('stop purchased a review'))
    for name in ('_maybe_inject_finalization_nudges', 'wait_after_tools', '_finish_tool_round_budget'):
        monkeypatch.setattr(loop, name, lambda *a, **kw: pytest.fail('stop purchased/parked/advanced'))
    text, _, trace = run([finish('Work remains.', action='stop', rationale='The child has not answered.')])
    assert text == 'Work remains.' and len(calls) == 1
    assert trace['task_completion']['rationale'] == 'The child has not answered.'
    axes = derive_loop_outcome(text, {}, trace)['outcome_axes']
    assert axes['objective']['reason'] == 'author_stop'
    assert axes['execution']['task_completion']['action'] == 'stop'


def test_completion_conflict_preserves_sources_and_requires_reselection(turn):
    registry, calls, notes, run = turn
    response = finish('first')
    second = finish('second')['tool_calls'][0]
    second['id'] = 'second'
    response['tool_calls'].append(second)
    text, _, trace = run([response, finish('informed')])
    assert text == 'informed' and len(calls) == 2
    assert any(call['is_error'] for call in trace['tool_calls'])
    assert trace['completion_refusals'][0]['reason'].startswith('contradictory_completion_requests')


def test_unavailable_held_selector_never_reconstructs_a_preview():
    text = 'Whole original response'
    selector = hashlib.sha256(text.encode()).hexdigest()
    assert selected_completion_text({'answer_sha256': selector}, None,
        [{'role': 'assistant', 'content': 'preview'}], selector)[0] is None


def test_observation_precedes_model_response_and_siblings(turn, monkeypatch):
    registry, calls, notes, run = turn
    from ouroboros.loop_messages import owner_source_sha256
    seen = {}
    def response():
        seen.update(registry._ctx._completion_observation)
        # An owner message is observed only on the next Main request, never inferred here.
        registry._ctx._owner_directives.append({'source': 'owner', 'content': 'New requirement'})
        return finish('Answer predating the owner input')
    text, _, trace = run([response, finish('Now considered')])
    assert text == 'Now considered' and len(calls) == 2
    assert seen['owner_source_sha256'] != owner_source_sha256(registry._ctx)
    assert trace['completion_refusals'][0]['reason'].startswith('owner_input_changed')


def test_stop_with_completed_error_sibling_preserves_error_without_more_work(turn):
    registry, calls, notes, run = turn
    from ouroboros.tools.tool_result import ToolResult, _publish_tool_result
    registry.override_handler('chat_history', lambda ctx: _publish_tool_result(ctx,
        ToolResult(status='error', code='TOOL_ERROR', text='Read unavailable')))
    response = finish('Partial answer', action='stop', rationale='Read failed; no complete result.')
    response['tool_calls'].append({'id': 'bad-read', 'type': 'function', 'function': {
        'name': 'chat_history', 'arguments': '{}'}})
    text, _, trace = run([response])
    assert len(calls) == 1 and text == 'Partial answer'
    assert trace['tool_calls'][1]['is_error'] is True
    assert trace['task_completion']['action'] == 'stop'


def test_stop_does_not_require_prospective_wrapup_money(turn, monkeypatch):
    registry, calls, notes, run = turn
    import ouroboros.task_pacing as pacing
    monkeypatch.setattr(pacing, 'wrapup_reservation_fits', lambda **kw: pytest.fail('no new call to reserve'))
    text, _, trace = run([finish('Partial', action='stop', rationale='Not finished.')])
    assert text == 'Partial' and len(calls) == 1 and trace['task_completion']['action'] == 'stop'


def test_failed_completion_remains_visible_work(turn):
    registry, calls, notes, run = turn
    text, _, trace = run([finish('', action='stop'), {'content': 'Ordinary final after invalid call.'}])
    assert trace['tool_calls'][0]['is_error'] is True
    assert len(calls) == 2 and text == 'Ordinary final after invalid call.'


def test_reply_later_is_consumed_once_then_a_hold_needs_new_selection(turn, monkeypatch):
    from tests.test_presence_completion import _call
    from ouroboros.presence_authority import presence_ceiling_payload
    from tests.test_presence_runner import _admission
    registry, calls, notes, run = turn
    registry._ctx.task_contract = {'capability_ceiling': presence_ceiling_payload(_admission().capability_ceiling)}
    from tests.presence_legacy_helpers import register_legacy_completion
    register_legacy_completion(registry)
    checked = hold_first_delivery(monkeypatch)
    text, _, _ = run([_call('message', ''), {'content': 'Chosen reply-later'},
        {'content': 'Interim, unselected'}, _call('message', 'Deliberately selected')])
    assert text == 'Deliberately selected' and checked == ['Chosen reply-later', 'Deliberately selected']
    assert len(calls) == 4


def test_owner_hard_stop_keeps_selected_bytes_and_zero_spend(turn, tmp_path):
    from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, write_owner_message
    from ouroboros.tools.control_runtime import _finish_task
    from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN
    registry, calls, notes, run = turn
    def stop(ctx, **kw):
        result = _finish_task(ctx, **kw)
        write_owner_message(tmp_path, REASON_OWNER_STOPPED_DIRECT_TURN, 'author-task', kind=KIND_FINALIZE_NOW)
        return result
    registry.override_handler('finish_task', stop)
    text, usage, trace = run([finish('Selected partial', action='stop', rationale='Unfinished')])
    assert text == 'Selected partial' and len(calls) == 1
    assert usage['reason_code'] == 'owner_requested_finalization'
    assert registry._ctx._skip_post_task_synthesis is True


def test_completion_does_not_reset_skill_or_nanny_work_baselines(turn, monkeypatch):
    from ouroboros.nanny_pacing import note_nanny_delegate_activity
    from ouroboros.tools.control_runtime import _finish_task
    registry, calls, notes, run = turn
    original = {'round': 1, 'cost': 2.0}
    def completion(ctx, **kw):
        ctx._skill_finalization_injected = True
        ctx._nanny_route_dispatched = True
        ctx._nanny_delegate_baseline = dict(original)
        note_nanny_delegate_activity(ctx, 3, {'cost': 4.0}, finish('x')['tool_calls'])
        return _finish_task(ctx, **kw)
    registry.override_handler('finish_task', completion)
    text, usage, trace = run([finish('Stopped', action='stop', rationale='Owner grant is still missing')])
    assert registry._ctx._skill_finalization_injected is True
    assert registry._ctx._nanny_delegate_baseline == original
    assert registry._ctx._nanny_metered_progress == {'round': 3, 'cost': 4.0}


def test_child_stop_never_seals_the_parent_root(turn):
    registry, calls, notes, run = turn
    registry._ctx.parent_task_id = 'parent'
    registry._ctx.root_task_id = 'parent'
    registry._ctx.task_metadata = {'parent_task_id': 'parent', 'root_task_id': 'parent', 'delegation_role': 'subagent'}
    registry._ctx.begin_acceptance_fence = lambda **kw: pytest.fail('child sealed its parent')
    text, _, trace = run([finish('Partial child', action='stop', rationale='Not finished')])
    assert text == 'Partial child' and len(calls) == 1
    assert trace['task_completion']['action'] == 'stop'


@pytest.mark.parametrize('refusal', ['unseen_work', 'invalid_subject', 'missing_answer', 'new_owner'])
def test_refused_completion_keeps_the_retained_answer_unselected(tmp_path, refusal):
    from ouroboros.loop_delivery import completion_observation, consume_completion_request
    from ouroboros.loop_messages import capture_acceptance_observation
    from ouroboros.tools.control_runtime import _finish_task
    from tests.test_acceptance_preparation_loop import _delivery_ctx

    trace = {'tool_calls': [], 'reasoning_notes': []}
    registry, ctx = _delivery_ctx(tmp_path, trace)
    candidate = loop._replace_delivery_candidate(registry, ctx, trace, 'Retained work', control='candidate')
    observed = capture_acceptance_observation(registry._ctx, trace, None)
    registry._ctx._completion_observation = completion_observation(registry._ctx, trace)
    _finish_task(registry._ctx, action='finish', rationale='The retained work answers the request.',
        answer_sha256='unavailable' if refusal == 'missing_answer' else candidate.content_sha256,
        acceptance_subject={'owner_source_sha256': 'wrong' if refusal == 'invalid_subject'
                            else observed['owner_source_sha256']})
    if refusal == 'unseen_work':
        trace['tool_calls'].append({'tool': 'chat_history', 'result': 'New result', 'is_error': False})
    elif refusal == 'new_owner':
        registry._ctx._owner_directives = [{'source': 'owner', 'content': 'A new requirement'}]
    assert consume_completion_request(registry, ctx, trace) is False
    assert trace['completion_refusals']
    assert not getattr(registry._ctx, '_completion_selected', None)
    assert registry._ctx._delivery_candidate.full_text == 'Retained work'
    assert registry._ctx._delivery_candidate.acceptance_binding['authoritative'] is False
