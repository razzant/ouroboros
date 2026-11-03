"""Actual delegated preparation failures survive complete producer replay.

Ported from Ouroboros contribution 03 (task d4ce874483ff407a, tree aaa0bdbc2, staged
65463cd9…): the retired plan machine's cycle test is dropped (the organ has no cycles); the
deleted ``_row_has_physical_dispatch`` assertion reads the organ's own participant facts."""
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from ouroboros import delegate_custody, review_checking
from ouroboros import review_execution, review_operation, review_substrate, subagents
from ouroboros.observability import persist_call, read_call_payload
from tests._review_session_route_shared import (
    FakeLLM, _agent_request, _agent_slot,
    _owned_gateway_uses_each_test_transport, fake_route,  # noqa: F401
)

pytestmark = pytest.mark.serial


def fail(message):
    def raise_error(*args, **kwargs):
        raise OSError(message)
    return raise_error


@pytest.mark.parametrize('stage', ['unconfigured', 'target', 'root', 'gateway', 'health', 'checking', 'budget', 'deadline', 'stamp', 'start_false', 'start_landed_false', 'start_ambiguous', 'after_row'])
def test_actual_prestart_failures_collect_without_new_start(tmp_path, monkeypatch, fake_route, stage):  # noqa: F811
    from ouroboros import budget_pause

    stamps = []
    request = _agent_request(task_id='presend', task_attempt=1, retry_key='one-review',
                             session_root=str(tmp_path), session_task='Review the fixture.')
    slot = _agent_slot(session_access='readonly')
    if stage == 'unconfigured':
        slot = replace(slot, session_target='')
        monkeypatch.setattr(review_execution, 'review_session_route', lambda: None)
    elif stage == 'target':
        monkeypatch.setattr(subagents, 'parse_subagent_harness', lambda *_: None)
    elif stage == 'root':
        request = replace(request, session_root='')
    elif stage == 'gateway':
        monkeypatch.setattr('ouroboros.claudexor_daemon.ensure_owned_gateway', fail('gateway unavailable'))
    elif stage == 'health':
        monkeypatch.setattr(subagents, 'route_health', lambda *a, **k: ('route_unavailable', ''))
    elif stage == 'checking':
        slot = replace(slot, session_access='workspace_write')
        monkeypatch.setattr(review_checking, 'session_checking', fail('checking unavailable'))
        monkeypatch.setattr(subagents, 'route_health', lambda *a, **k: ('', ''))
    elif stage == 'budget':
        monkeypatch.setattr(budget_pause, 'dispatch_fenced', lambda *_: True)
    elif stage == 'deadline':
        # The outer coordinator admits; the real inner producer refuses before its START row.
        monkeypatch.setattr(review_execution, 'owner_deadline_exhausted', lambda **_: True)
        request = replace(request, deadline_at='2099-01-01T00:00:00+00:00')
    elif stage == 'start_false':
        monkeypatch.setattr(delegate_custody, 'record_start_requested', lambda *a, **k: False)
    elif stage == 'start_landed_false':
        original_start = delegate_custody.record_start_requested
        def landed_then_false(*args, **kwargs):
            assert original_start(*args, **kwargs)
            return False
        monkeypatch.setattr(delegate_custody, 'record_start_requested', landed_then_false)
    elif stage == 'start_ambiguous':
        monkeypatch.setattr(delegate_custody, 'record_start_requested', fail('ambiguous write'))
    elif stage == 'after_row':
        from ouroboros.usage_accounting import UsageAccountingError
        fake_route.start_error = UsageAccountingError('fixture failure after POST entry')
    ctx = SimpleNamespace(task_id='presend', root_task_id='presend', task_attempt=1,
                          task_metadata={}, drive_root=tmp_path, budget_drive_root=tmp_path)
    # Use the actual worker/producer storage. No live engine or model participates.
    def paid_stamp(stamp):
        stamps.append('paid')
        if stage == 'stamp':
            raise OSError('paid stamp publication outcome is unknown')
    monkeypatch.setattr(review_execution, 'invoke_review_paid_stamp', paid_stamp)
    result = review_substrate.run_review_request(request, slots=[slot], drive_root=tmp_path,
                                                 llm=FakeLLM(), usage_ctx=ctx)
    row = asdict(result.actors[0]) if not isinstance(result.actors[0], dict) else result.actors[0]
    manifest, payload, _ = read_call_payload(tmp_path, task_id='presend', call_id=row['operation_id'] + '_error')
    assert manifest['producer_complete'] and row['status'] == 'error'
    original, original_slot = review_operation.recorded_review_assignment(
        tmp_path, task_id='presend', surface=request.surface, row=row, retry_key='one-review')
    before = sum(len(g.start_requests) for g in fake_route.instances)
    recovered = review_operation.recover_review_producer(tmp_path, original, original_slot, row)
    assert sum(len(g.start_requests) for g in fake_route.instances) == before
    if stage in {'stamp', 'start_false', 'start_landed_false', 'start_ambiguous', 'after_row'}:
        assert payload['usage'].get('delegated_start_unrequested') is not True
        if stage != 'after_row':
            assert recovered.operation_state == 'custody_lost'
        assert stamps == ['paid']
        assert before == (1 if stage == 'after_row' else 0)
    else:
        assert payload['usage']['delegated_start_unrequested'] is True
        assert recovered.error == row['error'] and recovered.operation_state == 'not_dispatched'
        assert recovered.status == 'error' and not recovered.raw_text and before == 0
        assert stamps == []
        # The organ's reading of this seat: a positive pre-dispatch refusal, never a reached reviewer (scope R1 F1).
        from ouroboros.tools.plan_ask import reply_facts_from_row
        from ouroboros.tools.plan_review_runtime import _plan_row_from_actor
        organ = reply_facts_from_row(_plan_row_from_actor(asdict(recovered), original_slot))
        assert organ['status'] == 'not_dispatched' and organ['dispatched'] is False
        # An old source without positive provenance remains unknown, not inferred free.
        payload['usage'].pop('delegated_start_unrequested')
        persist_call(tmp_path, task_id='presend', call_id=row['operation_id'] + '_error',
                     call_type='review_error', payload=payload, manifest={
                         'producer_complete': True, 'review_operation_binding': manifest['review_operation_binding']})
        _, rewritten, _ = read_call_payload(tmp_path, task_id='presend', call_id=row['operation_id'] + '_error')
        assert 'delegated_start_unrequested' not in rewritten['usage']
        legacy = review_operation.recover_review_producer(tmp_path, original, original_slot, row)
        assert legacy.operation_state == 'custody_lost'


@pytest.mark.parametrize('contradiction', ['missing', 'false', 'text', 'success', 'no_error', 'token', 'run', 'started', 'response'])
def test_positive_prestart_receipt_rejects_contradictory_or_legacy_data(tmp_path, contradiction):
    from ouroboros.review_session_custody import validate_producer_invocation

    usage = {'delegated_start_unrequested': True}
    outcome = {'status': 'error', 'error': 'original refusal'}
    token, error_source = '', True
    if contradiction == 'missing':
        usage.clear()
    elif contradiction in {'false', 'text'}:
        usage['delegated_start_unrequested'] = False if contradiction == 'false' else 'true'
    elif contradiction == 'success':
        outcome['status'] = 'ok'
    elif contradiction == 'no_error':
        outcome['error'] = ''
    elif contradiction == 'token':
        token = 'unproven-invocation'
    elif contradiction == 'run':
        usage['delegated_run_id'] = 'run-unknown'
    elif contradiction == 'started':
        usage['delegated_run_started'] = True
    else:
        error_source = False
    with pytest.raises(ValueError, match='exact delegated invocation'):
        validate_producer_invocation(tmp_path, token, {}, {'usage': usage}, outcome, error_source=error_source)


@pytest.mark.parametrize('earlier', ['none', 'settled', 'unknown'])
def test_no_start_projection_preserves_earlier_physical_history(earlier):
    from ouroboros.review_custody import _ReviewAttemptHistory, _review_exception_projection
    history = _ReviewAttemptHistory(dispatched=earlier != 'none',
        capture_state='settled' if earlier == 'settled' else 'unresolved' if earlier == 'unknown' else '',
        unknown_outcome_seen=earlier == 'unknown')
    usage, _, _, state, _ = _review_exception_projection(
        OSError('local preparation refusal'), {'delegated_start_unrequested': True}, history, {})
    assert state == {'none': 'not_dispatched', 'settled': 'settled', 'unknown': 'custody_lost'}[earlier]
    if earlier != 'none':
        assert usage['physical_attempt_state'] == history.capture_state


@pytest.mark.parametrize('site', ['lookup', 'gateway'])
def test_recovery_failure_before_gateway_preserves_original_error(tmp_path, monkeypatch, fake_route, site):  # noqa: F811
    from ouroboros import claudexor_daemon
    from ouroboros.gateways import claudexor
    from tests._review_session_route_shared import FakeGateway
    from tests.test_delegate_prestart_cleanup import failures

    FakeGateway.project_unregistered = True
    state = {}
    invocation = review_execution.SessionInvocation(
        task_id='t-b', surface='scope_review', slot_id='scope_slot_1', timeout_sec=30, retry_state=state)

    def start():
        return review_execution.run_delegated_review_session(
            prompt='review this', root='/tmp/fake-repo', custody_drive=tmp_path, invocation=invocation)

    FakeGateway.start_error = claudexor.ClaudexorUnavailable('daemon_unreachable', 'lost response')
    with pytest.raises(claudexor.ClaudexorUnavailable):
        start()
    token, rows = state['pending_invocation_id'], len(failures(tmp_path))
    starts = sum(len(g.start_requests) for g in FakeGateway.instances)
    error = OSError('pre-gateway failure')

    def refuse(*args, **kwargs):
        raise error

    if site == 'lookup':
        monkeypatch.setattr(delegate_custody, 'invocation_record', refuse)
    else:
        monkeypatch.setattr(claudexor_daemon, 'ensure_owned_gateway', refuse)
    with pytest.raises(OSError) as caught:
        start()
    assert caught.value is error and not hasattr(error, 'delegated_start_unrequested')
    assert state['pending_invocation_id'] == token and len(failures(tmp_path)) == rows
    assert sum(len(g.start_requests) for g in FakeGateway.instances) == starts


def test_unwritable_start_row_cleans_once_without_no_send(tmp_path, monkeypatch, fake_route):  # noqa: F811
    from ouroboros.task_results import write_task_result
    from tests._review_session_route_shared import FakeGateway, _run_session_directly
    from tests.test_delegate_prestart_cleanup import failures

    write_task_result(tmp_path, 't-b', 'running', root_task_id='t-b')
    repo = tmp_path / 'repo'
    repo.mkdir()
    FakeGateway.project_unregistered = True
    state = {}
    monkeypatch.setattr(delegate_custody, 'record_start_requested', lambda *a, **k: False)
    with pytest.raises(review_execution.ReviewRouteUnavailable) as caught:
        _run_session_directly(tmp_path, root=str(repo), retry_state=state)
    assert caught.value.code == 'start_request_row_unwritable'
    assert getattr(caught.value, 'delegated_start_unrequested', None) is None
    gateway = FakeGateway.instances[-1]
    assert gateway.removals == ['proj-new'] and gateway.start_requests == []
    assert [row['definite'] for row in failures(tmp_path)] == [True]
    assert 'pending_invocation_id' not in state
