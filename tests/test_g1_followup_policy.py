"""G1 through the published tool, schedule lifecycle, queue and final handoff."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
from tests.worker_pool_support import affirm_ready

from ouroboros.cancel_intents import request_cancel, active_intent
from ouroboros.task_results import write_task_result, load_task_result
from ouroboros.tools.followup import _handle_schedule_followup
from ouroboros.usage_admission import task_billing_fields
from supervisor import queue, queue_schedules as schedules
from supervisor.followup_policy import observed_store, record_restart
from tests.test_schedule_followup import _ctx

ORIGIN = 'root-1'
BINDING = {'billing_group_id': ORIGIN, 'billing_group_limit_usd': 10.0,
           'billing_group_limit_source': 'initial_task_admission', 'billing_group_limit_revision': 'original-revision'}
DEADLINE = '2099-01-01T00:00:00+00:00'


@pytest.fixture
def world(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    ctx.current_chat_id = 0
    ctx.task_metadata["resource_intent"] = {"kind": "system_repo"}
    root = ctx.drive_root
    queue.init(root)
    pending = []
    queue.init_queue_refs(pending, {}, {'value': 0})
    monkeypatch.setenv('TOTAL_BUDGET', '1000')
    write_task_result(root, ORIGIN, 'running', root_task_id=ORIGIN,
                      billing_group=BINDING, deadline_at=DEADLINE)
    return ctx, root, pending


def register(world, relation='related', **extra):
    ctx, root, _ = world
    result = _handle_schedule_followup(ctx, relation=relation, run_at='2000-01-01T00:00:00+00:00',
                                      objective='Original work', **extra)
    assert result.startswith('FOLLOWUP_SCHEDULED'), result
    return schedules.load_schedule_store(root)['tasks'][-1]


def row(root, sid):
    return next(r for r in schedules.load_schedule_store(root)['tasks'] if r['id'] == sid)


def stop(root):
    return request_cancel(root, ORIGIN, reason='Explicit Stop', source='http_single', requested_by='owner')


def restore(root, sid, hold, **extra):
    return queue.mutate_scheduled_task('restore', sid, reason='Reviewed this exact hold', actor='owner:test',
                                       drive_root=root, expected_hold_id=hold, **extra)


def test_new_relation_required_and_unrecorded_legacy_keeps_its_own_wallet_until_resolved(world):
    ctx, root, _ = world
    # New work must decide its relationship; the published producer never recorded one.
    assert 'followup_relation_required' in _handle_schedule_followup(ctx, objective='x', run_at=DEADLINE)
    legacy = {'id': 'legacy', 'source': 'task_followup', 'task': {'metadata': {'origin_task_id': ORIGIN}}}
    garbled = {'id': 'garbled', 'source': 'task_followup', 'followup_relation': {'kind': 'garbled'},
               'task': {'metadata': {'origin_task_id': ORIGIN}}}
    schedules._write_scheduled_tasks({'tasks': [legacy, garbled]}, root)
    queue.check_scheduled_tasks()
    assert not row(root, 'legacy').get('followup_hold'), 'no blanket hold merely for being published'
    assert 'followup_relation' not in row(root, 'legacy'), 'the relationship stays honestly unrecorded'
    assert row(root, 'garbled')['followup_hold']['reason'] == 'relationship_unknown'
    # Its fired task pays from its OWN root (the previous rule), never the origin's
    # cap or a template's forged group; a recorded but unreadable relationship
    # still acquires no wallet at all.
    task = {'id': 'legacy-fired', 'metadata': {'schedule_id': 'legacy', 'billing_group': BINDING}}
    result = task_billing_fields(task, task['id'], 999.0, root, pin_initial=True, persist_initial=False)
    assert (result['billing_group_id'], result['billing_group_limit_usd']) == ('legacy-fired', 999.0)
    garbled_task = {'id': 'garbled-fired', 'metadata': {'schedule_id': 'garbled'}}
    from ouroboros.usage_ledger import BillingAuthorityUnavailable
    with pytest.raises(BillingAuthorityUnavailable) as refused:
        task_billing_fields(garbled_task, 'garbled-fired', 999.0, root, pin_initial=True, persist_initial=False)
    assert (refused.value.kind, refused.value.cause) == ('refused', 'related_followup_unbound')
    assert load_task_result(root, task['id']) is None
    # Only an actual Stop holds it; its release then needs the relationship decision.
    stop(root)
    queue.check_scheduled_tasks()
    held = row(root, 'legacy')['followup_hold']['hold_id']
    result = restore(root, 'legacy', held)
    assert result['status'] == 'relationship_required'
    result = restore(root, 'legacy', held, relation='related')
    assert result['status'] == 'hold_released', result
    resolved = row(root, 'legacy')['followup_relation']
    assert resolved['billing_group'] == BINDING and resolved['deadline_at'] == DEADLINE
    assert task_billing_fields(task, task['id'], 999.0, root, persist_initial=False)['billing_group_id'] == ORIGIN


def test_unknown_related_resolution_cannot_invent_missing_origin(world):
    _, root, _ = world
    queue.upsert_scheduled_task({'id': 'lost', 'source': 'task_followup',
                                'task': {'metadata': {'origin_task_id': 'lost-origin'}}}, drive_root=root)
    before = row(root, 'lost')
    # A collected origin records no Stop: the published row is not held, and its
    # template lineage survives normalization as host provenance.
    assert not before.get('followup_hold') and before['followup_origin']['task_id'] == 'lost-origin'
    result = restore(root, 'lost', '', relation='related')
    assert result['status'] == 'followup_origin_unavailable'
    assert row(root, 'lost') == before


def test_restart_marker_temporarily_fences_but_does_not_hold_healthy_schedules(world):
    _, root, pending = world
    related = register(world)
    from ouroboros.delegate_recovery import prepare_owner_restart, consume_owner_restart
    door = prepare_owner_restart(root, legacy=True)
    queue.check_scheduled_tasks()
    assert pending == []
    record_restart(root)
    consume_owner_restart(root, door)
    queue.check_scheduled_tasks()
    assert [task['metadata']['schedule_id'] for task in pending] == [related['id']]
    assert not row(root, related['id']).get('followup_hold')
    record_restart(root, new=True)
    assert not observed_store(root, schedules.load_schedule_store(root))['tasks'][0].get('followup_hold')


def test_refused_restriction_audit_blocks_without_false_persisted_hold(world, monkeypatch):
    _, root, pending = world
    registered = register(world)
    stop(root)
    monkeypatch.setattr(schedules, '_audit_schedule_mutation', lambda **kw: False)
    queue.check_scheduled_tasks()
    assert pending == [] and not row(root, registered['id']).get('followup_hold')
    data = observed_store(root, schedules.load_schedule_store(root))
    projected = schedules.schedule_tool_projection(data)['tasks'][0]
    assert projected['status'] == 'held' and projected['hold_persisted'] is False


def test_group_scope_includes_sibling_and_original_late_charge(world):
    from ouroboros import usage_accounting as ua
    _, root, pending = world
    registered = register(world)
    queue.check_scheduled_tasks()
    [task] = pending
    fields = task_billing_fields(task, task['id'], 999.0, root)
    for who, amount in ((ORIGIN, 5.0), ('sibling', 5.0)):  # known spend reaches the group's $10
        with ua.usage_scope(ua.UsageScope(drive_root=root, task_id=who, root_task_id=who,
                                         source='test', root_limit_usd=10.0, **BINDING)):
            reservation = ua.reserve_attempt(ua.AttemptRequest(model='openai/gpt-5.2', provider='openai', reservation_usd=amount))
            ua.mark_dispatched(reservation)
            ua.settle_attempt(reservation, {}, cost_usd=amount, cost_final=True)
    with ua.usage_scope(ua.UsageScope(drive_root=root, task_id=task['id'], root_task_id=task['id'], source='test', **fields)):
        with pytest.raises(ua.BudgetExceeded):
            ua.reserve_attempt(ua.AttemptRequest(model='openai/gpt-5.2', provider='openai', reservation_usd=3.0))
    assert row(root, registered['id'])['followup_relation']['billing_group'] == BINDING


def test_public_enqueue_refuses_stop_after_prepare_and_independent_still_starts(world):
    _, root, pending = world
    related = register(world)
    independent = register(world, 'independent')
    prepared = queue._task_from_schedule(related)
    independent_task = queue._task_from_schedule(independent)
    stop(root)
    assert queue.enqueue_task(prepared)['_admission_blocked'] == 'followup_control_wait'
    assert not queue.enqueue_task(independent_task).get('_admission_blocked')
    assert [t['id'] for t in pending] == [independent_task['id']]


def test_no_speculative_release_and_existing_first_cause_does_not_hide_stop(world):
    _, root, _ = world
    registered = register(world)
    original = request_cancel(root, ORIGIN, reason='Earlier technical cause', source='timeout')
    later = stop(root)
    assert later['source'] == original['source'] == 'timeout'
    assert later['followup_stop']['source'] == 'http_single'
    # GET observes the restriction but has not produced a durable hold.
    observed = observed_store(root, schedules.load_schedule_store(root))['tasks'][0]
    assert not observed['hold_persisted']
    result = restore(root, registered['id'], observed['followup_hold']['hold_id'])
    assert result['status'] == 'hold_not_observed'
    assert row(root, registered['id'])['followup_hold']


def test_failed_hold_persistence_and_restore_outcome_audit_are_honest(world, monkeypatch):
    _, root, pending = world
    registered = register(world)
    stop(root)
    real_write = schedules._write_scheduled_tasks
    monkeypatch.setattr(schedules, '_write_scheduled_tasks', lambda *a, **k: (_ for _ in ()).throw(OSError('disk refused')))
    queue.check_scheduled_tasks()
    assert pending == [] and not row(root, registered['id']).get('followup_hold')
    monkeypatch.setattr(schedules, '_write_scheduled_tasks', real_write)
    queue.check_scheduled_tasks()
    held = row(root, registered['id'])['followup_hold']['hold_id']
    audit = schedules._audit_schedule_mutation
    monkeypatch.setattr(schedules, '_audit_schedule_mutation', lambda **kw: False if kw['phase'] == 'outcome' else audit(**kw))
    result = restore(root, registered['id'], held)
    assert result['changed'] and result['status'] == 'changed_audit_incomplete' and not result['ok']
    assert not row(root, registered['id']).get('followup_hold')


def test_related_final_handoff_success_is_bound_and_original_late_charge_still_counts(world, monkeypatch):
    from supervisor.worker_assignment import _claim_worker_launch
    from ouroboros import usage_accounting as ua
    _, root, pending = world
    register(world)
    queue.check_scheduled_tasks()
    [task] = pending
    sent = []
    worker = SimpleNamespace(in_q=SimpleNamespace(put=sent.append))
    assert _claim_worker_launch(queue, task, worker)
    assert sent == [task] and task['admitted_dispatch'] == 'possible'
    scope = dict(drive_root=root, root_task_id=ORIGIN, task_id=ORIGIN, root_limit_usd=100.0, source='test', **BINDING)
    with ua.usage_scope(ua.UsageScope(**scope)):
        old = ua.reserve_attempt(ua.AttemptRequest(model='openai/gpt-5.2', provider='openai', reservation_usd=1.0))
        ua.mark_dispatched(old)
    fields = task_billing_fields(task, task['id'], 100.0, root)
    with ua.usage_scope(ua.UsageScope(drive_root=root, root_task_id=task['id'], task_id=task['id'], source='test', **fields)):
        sibling = ua.reserve_attempt(ua.AttemptRequest(model='openai/gpt-5.2', provider='openai', reservation_usd=3.0))
        ua.mark_dispatched(sibling)
        ua.settle_attempt(sibling, {}, cost_usd=3.0, cost_final=True)
    # The original operation's late true cost exceeds its estimate.
    ua.settle_attempt(old, {}, cost_usd=7.0, cost_final=True)
    with ua.usage_scope(ua.UsageScope(drive_root=root, root_task_id=task['id'], task_id=task['id'], source='test', **fields)):
        with pytest.raises(ua.BudgetExceeded):
            ua.reserve_attempt(ua.AttemptRequest(model='openai/gpt-5.2', provider='openai', reservation_usd=0.1))


def test_completed_schedule_gc_preserves_host_group_for_late_accounting(world):
    _, root, pending = world
    register(world)
    queue.check_scheduled_tasks()
    [task] = pending
    write_task_result(root, task['id'], 'completed', result='Finished scheduled work')
    # Existing GC can discard history after completion; it must not reset money.
    schedules._write_scheduled_tasks({'tasks': []}, root)
    assert task_billing_fields({'id': task['id']}, task['id'], 999.0, root)['billing_group_id'] == ORIGIN
    assert task_billing_fields({'id': task['id']}, task['id'], 999.0, root)['billing_group_limit_usd'] == 10.0


def test_context_and_continue_expose_held_obligations_without_auto_release(world):
    from supervisor.schedule_view import digest
    from datetime import datetime, timezone
    from ouroboros.owner_continue import work_order_text
    _, root, _ = world
    registered = register(world)
    queue.check_scheduled_tasks()
    stop(root)
    queue.check_scheduled_tasks()
    view = digest(schedules.load_schedule_store(root)['tasks'], datetime.now(timezone.utc))
    assert not view['one_shots'] and view['attention'][0]['id'] == registered['id']
    assert view['attention'][0]['status'] == 'consumed' and view['attention'][0]['hold_persisted']
    text = work_order_text(ORIGIN, 'owner_restart', {}, deadline_at=DEADLINE)
    assert 'manage_schedules' in text and 'does not release their holds' in text and DEADLINE in text


def _assignment_world(world, monkeypatch, count=1):
    from tests._budget_pause_exact_helpers import _install_queue
    ctx, root, _ = world
    _, _, workers = _install_queue(root, monkeypatch)
    sent = []
    for wid in range(count):
        workers.WORKERS[wid] = affirm_ready(SimpleNamespace(wid=wid, busy_task_id=None,
            in_q=SimpleNamespace(put=lambda task, wid=wid: sent.append((wid, copy.deepcopy(task)))
                                 if task.get("type") != "worker_quiescence" else None)))
    return (ctx, root, workers.PENDING), workers, sent


@pytest.mark.parametrize('door', ['http', 'tool'])
def test_stop_action_retry_and_later_action_across_release(world, monkeypatch, door):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient
    from ouroboros.gateway.tasks import api_task_cancel
    from ouroboros.tools.join_ledger import _cancel_task
    from supervisor import owner_stop
    ctx, root, _ = world
    registered = register(world)
    sid = registered['id']
    queue.RUNNING[ORIGIN] = {'task': {'id': ORIGIN}, 'worker_id': 0}
    begun = []
    monkeypatch.setattr(owner_stop, 'begin_graceful_stop', begun.append)
    app = Starlette(routes=[Route('/api/tasks/{task_id}/cancel', api_task_cancel, methods=['POST'])])
    with TestClient(app) as client:
        def cancel(action_id):
            if door == 'http':
                reply = client.post(f'/api/tasks/{ORIGIN}/cancel', json={
                    'stop_policy': 'finalize_then_cancel', 'stop_action_id': action_id})
                assert reply.status_code == 202, reply.text
                return reply.json()
            return _cancel_task(ctx, ORIGIN, reason='Stop this work', stop_action_id=action_id)
        cancel('one-action')  # Its response is lost to the caller.
        first = active_intent(root, ORIGIN)
        queue.check_scheduled_tasks()
        assert restore(root, sid, row(root, sid)['followup_hold']['hold_id'])['ok']
        cancel('one-action')
        replay = active_intent(root, ORIGIN)
        assert replay['request_id'] == first['request_id']
        assert replay['followup_stop'] == first['followup_stop']
        assert not observed_store(root, schedules.load_schedule_store(root))['tasks'][0].get('followup_hold')
        cancel('later-action')
        later = active_intent(root, ORIGIN)
        assert later['request_id'] == first['request_id']
        assert later['followup_stop']['control_id'] != first['followup_stop']['control_id']
        queue.check_scheduled_tasks()
        assert restore(root, sid, row(root, sid)['followup_hold']['hold_id'])['ok']
        cancel('one-action')  # Delayed older replay must not replace the later control.
        assert active_intent(root, ORIGIN)['followup_stop'] == later['followup_stop']
        assert not observed_store(root, schedules.load_schedule_store(root))['tasks'][0].get('followup_hold')
