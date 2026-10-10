"""G1 through the published tool, schedule lifecycle, queue and final handoff."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from ouroboros.cancel_intents import request_cancel, settle_intent, active_intent
from ouroboros.task_results import write_task_result, load_task_result
from ouroboros.tools.followup import _handle_schedule_followup
from ouroboros.usage_admission import task_billing_fields
from supervisor import queue, queue_schedules as schedules
from supervisor.followup_policy import observed_store, record_restart
from tests.test_schedule_followup import _ctx

pytestmark = pytest.mark.serial
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
    monkeypatch.setattr(schedules, '_last_skill_schedule_sync', float('inf'))
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


def test_same_author_related_independent_and_unknown(world):
    _, root, pending = world
    related = register(world)
    independent = register(world, 'independent')
    legacy = {'id': 'legacy', 'source': 'task_followup', 'enabled': True,
              'trigger': {'type': 'once', 'run_at': '2000-01-01T00:00:00+00:00'},
              'task': {'text': 'same words', 'metadata': {'origin_task_id': ORIGIN, 'origin_root_task_id': ORIGIN}}}
    queue.upsert_scheduled_task(legacy, drive_root=root)
    stop(root)
    queue.check_scheduled_tasks()
    assert [t['metadata']['schedule_id'] for t in pending] == [independent['id']]
    assert row(root, related['id'])['followup_hold']['reason'] == 'origin_stopped'
    assert row(root, 'legacy')['followup_hold']['reason'] == 'relationship_unknown'
    assert not row(root, related['id']).get('last_task_id')


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
    assert task_billing_fields(garbled_task, 'garbled-fired', 999.0, root, pin_initial=True,
                               persist_initial=False)['billing_group_id'].startswith('unavailable:')
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


def test_completion_wins_but_stop_survives_retirement_and_late_registration(world):
    _, root, pending = world
    intent = stop(root)
    write_task_result(root, ORIGIN, 'completed', result='Finished original', billing_group=BINDING, deadline_at=DEADLINE)
    assert settle_intent(root, ORIGIN, outcome='already_settled', expected_generation=intent['generation'],
                         request_id=intent['request_id'])
    assert active_intent(root, ORIGIN) is None
    original = load_task_result(root, ORIGIN)
    assert original['status'] == 'completed' and original['result'] == 'Finished original'
    assert original['followup_stop'] == intent['followup_stop']
    held = register(world)
    assert held['followup_hold']['controls']['stop:' + ORIGIN] == intent['followup_stop']['control_id']
    queue.check_scheduled_tasks()
    assert pending == []


def test_stop_after_prepare_blocks_final_worker_handoff_without_refire(world, monkeypatch):
    from supervisor.worker_assignment import _claim_worker_launch
    _, root, pending = world
    registered = register(world)
    queue.check_scheduled_tasks()
    [candidate] = pending
    frozen = copy.deepcopy(candidate)
    sent = []
    worker = SimpleNamespace(in_q=SimpleNamespace(put=sent.append))
    stop(root)  # after scheduler sweep and task preparation
    assert not _claim_worker_launch(queue, candidate, worker)
    assert sent == [] and candidate == frozen
    queue.check_scheduled_tasks()
    assert len(pending) == 1 and pending[0]['id'] == frozen['id']
    held = row(root, registered['id'])['followup_hold']['hold_id']
    # Preserve unrelated task-level holds and money; release is only the schedule facet.
    candidate['_budget_hold'] = {'reason': 'independent-budget-hold'}
    assert restore(root, registered['id'], held)['ok']
    assert candidate['_budget_hold'] == {'reason': 'independent-budget-hold'}
    assert row(root, registered['id'])['completed_at']
    assert row(root, registered['id'])['enabled'] is False
    assert candidate['metadata']['billing_group'] == BINDING and candidate['deadline_at'] == DEADLINE
    queue.check_scheduled_tasks()
    assert len(pending) == 1


def test_stale_restore_never_clears_new_stop_or_enables_independent_disable(world):
    _, root, _ = world
    registered = register(world)
    stop(root)
    queue.check_scheduled_tasks()
    old = row(root, registered['id'])['followup_hold']['hold_id']
    queue.mutate_scheduled_task('disable', registered['id'], reason='Separate disable', actor='owner:test', drive_root=root)
    stop(root)
    outcome = restore(root, registered['id'], old)
    assert outcome['status'] == 'stale_hold'
    new = row(root, registered['id'])['followup_hold']['hold_id']
    assert new != old
    assert restore(root, registered['id'], new)['ok']
    restored = row(root, registered['id'])
    assert restored['enabled'] is False
    assert restored['followup_relation'] == registered['followup_relation']


def test_pause_is_transient_wait_and_resume_preserves_eligibility(world):
    from ouroboros.owner_pause import install_fence, release_fence
    _, root, pending = world
    related = register(world)
    independent = register(world, 'independent')
    install_fence(root, ORIGIN, request_id='pause-request')
    queue.check_scheduled_tasks()
    current = row(root, related['id'])
    assert current['followup_wait'] == 'origin_owner_paused' and not current.get('followup_hold')
    assert [t['metadata']['schedule_id'] for t in pending] == [independent['id']]
    release_fence(root, ORIGIN, reason='Owner Resume')
    queue.check_scheduled_tasks()
    assert len(pending) == 2


def test_restart_marker_blocks_before_restriction_write_and_new_generation_invalidates_restore(world):
    _, root, pending = world
    related = register(world)
    marker = root / 'state/owner_restart_no_resume.flag'
    marker.write_text('owner_restart')  # crash before restriction persistence
    queue.check_scheduled_tasks()
    assert pending == []
    record_restart(root)
    marker.unlink()
    queue.check_scheduled_tasks()
    held = row(root, related['id'])['followup_hold']['hold_id']
    record_restart(root, new=True)
    assert restore(root, related['id'], held)['status'] == 'stale_hold'


def test_template_forgery_and_edit_cannot_replace_host_binding(world):
    _, root, pending = world
    registered = register(world, billing_group={'billing_group_id': 'forged'})
    forged = copy.deepcopy(registered)
    forged['followup_relation'] = {'kind': 'independent'}
    forged['followup_origin'] = {'task_id': 'fake', 'root_task_id': 'fake'}
    forged['task']['metadata'].update(billing_group={'billing_group_id': 'fresh', 'billing_group_limit_usd': 9000},
                                     continuation={'billing_group_id': 'fresh'}, origin_task_id='fake')
    forged['task']['task_contract'] = {'deadline_at': '2199-01-01T00:00:00+00:00'}
    queue.upsert_scheduled_task(forged, drive_root=root)
    queue.check_scheduled_tasks()
    [task] = pending
    assert task['metadata']['billing_group'] == BINDING
    assert task['metadata']['origin_task_id'] == ORIGIN
    assert task['task_contract']['deadline_at'] == DEADLINE
    assert 'continuation' not in task['metadata']


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


def test_failed_stop_transfer_keeps_active_intent_and_completed_result(world, monkeypatch):
    import ouroboros.cancel_intents as ci
    _, root, _ = world
    intent = stop(root)
    write_task_result(root, ORIGIN, 'completed', result='Answer survives')
    real = ci.update_json_locked
    def fail_result(path, *args, **kwargs):
        if path.name != 'cancel_intents.json':
            raise OSError('result persistence refused')
        return real(path, *args, **kwargs)
    monkeypatch.setattr(ci, 'update_json_locked', fail_result)
    with pytest.raises(OSError, match='persistence refused'):
        settle_intent(root, ORIGIN, outcome='already_settled')
    assert active_intent(root, ORIGIN)['followup_stop'] == intent['followup_stop']
    assert load_task_result(root, ORIGIN)['result'] == 'Answer survives'


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


def test_legacy_template_normalization_keeps_origin_but_cannot_forge_money(world):
    _, root, _ = world
    legacy = {'id': 'legacy', 'source': 'task_followup', 'enabled': False,
              'task': {'metadata': {'origin_task_id': ORIGIN, 'origin_root_task_id': ORIGIN,
                                   'billing_group': {'billing_group_id': 'fresh', 'billing_group_limit_usd': 999},
                                   'continuation': {'billing_group_id': 'fresh'}}}}
    schedules._write_scheduled_tasks({'tasks': [legacy]}, root)
    edited = copy.deepcopy(legacy)
    edited['task']['metadata']['origin_task_id'] = 'forged'
    stop(root)  # the actual control that holds an unrecorded relationship
    stored = queue.upsert_scheduled_task(edited, drive_root=root)
    assert stored['followup_origin'] == {'task_id': ORIGIN, 'root_task_id': ORIGIN}
    assert 'billing_group' not in stored['task']['metadata']
    assert 'continuation' not in stored['task']['metadata']
    assert restore(root, 'legacy', stored['followup_hold']['hold_id'], relation='related')['ok']
    assert row(root, 'legacy')['followup_relation']['billing_group'] == BINDING
    assert not row(root, 'legacy')['enabled']


def test_legacy_fired_unknown_without_receipt_keeps_id_and_waits_for_custody(world):
    _, root, pending = world
    legacy = {'id': 'legacy', 'source': 'task_followup', 'enabled': False,
              'trigger': {'type': 'once'}, 'completed_at': '2026-01-01T00:00:00+00:00',
              'last_task_id': 'same-fired-id',
              'task': {'metadata': {'origin_task_id': ORIGIN, 'origin_root_task_id': ORIGIN}}}
    schedules._write_scheduled_tasks({'tasks': [legacy]}, root)
    write_task_result(root, 'same-fired-id', 'scheduled', metadata={'schedule_id': 'legacy'})
    queue.check_scheduled_tasks()
    # The upgrade alone holds nothing, and a receipt-less fired task is never replayed.
    assert not row(root, 'legacy').get('followup_hold') and pending == []
    stop(root)
    queue.check_scheduled_tasks()
    held = row(root, 'legacy')['followup_hold']['hold_id']
    assert restore(root, 'legacy', held, relation='related')['ok']
    current = row(root, 'legacy')
    assert current['last_task_id'] == 'same-fired-id' and current['completed_at'] == legacy['completed_at']
    assert current['followup_wait'] == 'pending_binding_unavailable'
    assert pending == []


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


def test_refused_scheduled_result_never_reaches_public_enqueue(world, monkeypatch):
    import ouroboros.task_results as results
    _, root, pending = world
    registered = register(world)
    monkeypatch.setattr(results, 'write_task_result', lambda *a, **k: False)
    queue.check_scheduled_tasks()
    assert pending == []
    current = row(root, registered['id'])
    assert current['hold']['reason'] == 'receipt_failed'
    assert not current.get('completed_at') and not current.get('last_task_id')


def test_public_gateway_rejects_template_money_and_save_cannot_release_hold(world):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient
    from ouroboros.gateway.schedules import api_schedules_upsert
    _, root, _ = world
    registered = register(world)
    stop(root)
    queue.check_scheduled_tasks()
    held = row(root, registered['id'])['followup_hold']
    app = Starlette(routes=[Route('/api/schedules', api_schedules_upsert, methods=['POST'])])
    app.state.drive_root = root
    with TestClient(app) as client:
        body = {'id': registered['id'], 'trigger': registered['trigger'], 'enabled': True,
                'task': {'text': 'Keep work', 'metadata': {'billing_group': {'billing_group_id': 'fresh'}}}}
        assert client.post('/api/schedules', json=body).status_code == 400
        body['task']['metadata'] = {}
        body['followup_relation'] = {'kind': 'independent'}
        body['followup_released'] = held['controls']
        response = client.post('/api/schedules', json=body)
        assert response.status_code == 200, response.text
    current = row(root, registered['id'])
    assert current['followup_hold'] == held and current['followup_relation'] == registered['followup_relation']


def test_followup_disable_is_an_ordinary_disable_and_restore_lifts_only_it(world):
    _, root, pending = world
    registered = register(world)
    queue.mutate_scheduled_task('disable', registered['id'], reason='Independent owner disable', actor='owner:test', drive_root=root)
    current = row(root, registered['id'])
    # Only skill-manifest rows keep a resync-proof suppression marker.
    assert current['enabled'] is False and 'manual_override' not in current
    assert schedules.schedule_lifecycle_status(current) == 'disabled'
    queue.check_scheduled_tasks()
    assert pending == []
    result = restore(root, registered['id'], '')
    assert result['ok'] and result['status'] == 'updated', result
    assert schedules.schedule_lifecycle_status(row(root, registered['id'])) == 'active'
    queue.check_scheduled_tasks()
    assert len(pending) == 1


def test_context_and_continue_expose_held_obligations_without_auto_release(world):
    from ouroboros.context import _scheduled_tasks_digest
    from ouroboros.owner_continue import work_order_text
    _, root, _ = world
    registered = register(world)
    queue.check_scheduled_tasks()
    stop(root)
    queue.check_scheduled_tasks()
    digest = _scheduled_tasks_digest(SimpleNamespace(drive_path=lambda rel: root / rel))
    assert not digest['active'] and digest['held'][0]['id'] == registered['id']
    assert digest['held'][0]['status'] == 'consumed' and digest['held'][0]['hold_persisted']
    text = work_order_text(ORIGIN, 'owner_restart', {}, deadline_at=DEADLINE)
    assert 'manage_schedules' in text and 'does not release their holds' in text and DEADLINE in text


@pytest.mark.parametrize('suppression', ['disable', 'delete'])
@pytest.mark.parametrize('later', [False, True])
def test_exact_restore_replay_cannot_lift_separate_suppression(world, suppression, later):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient
    from ouroboros.gateway.schedules import api_schedules_action
    _, root, _ = world
    registered = register(world)
    stop(root)
    queue.check_scheduled_tasks()
    sid = registered['id']
    held = row(root, sid)['followup_hold']['hold_id']
    app = Starlette(routes=[Route('/api/schedules/{schedule_id}/action', api_schedules_action, methods=['POST'])])
    app.state.drive_root = root
    with TestClient(app) as client:
        def action(verb, **kw):
            response = client.post(f'/api/schedules/{sid}/action', json={'action': verb, 'reason': 'Explicit action', **kw})
            assert response.status_code == 200, response.text
            return response.json()
        if not later:
            first = action(suppression)
            assert first['changed']
            if suppression == 'delete':
                # Never fired: Delete really removes it; nothing restorable remains.
                assert first['status'] == 'deleted' and first['schedule'] is None
                assert action('restore', expected_hold_id=held)['status'] == 'not_found'
                return
        assert action('restore', expected_hold_id=held)['status'] == 'hold_released'
        if later:
            outcome = action(suppression)
            assert outcome['changed']
            if suppression == 'delete':
                assert outcome['status'] == 'deleted'
                replay = action('restore', expected_hold_id=held)
                assert replay['status'] == 'not_found' and not replay['changed']
                return
        before = row(root, sid)
        replay = action('restore', expected_hold_id=held)
        assert not replay['changed'], replay
        assert row(root, sid) == before and not before['enabled']
        # The separate, explicitly generic Restore remains available.
        assert action('restore')['ok']
        assert row(root, sid)['enabled']


def _assignment_world(world, monkeypatch, count=1):
    from tests._budget_pause_exact_helpers import _install_queue
    ctx, root, _ = world
    _, _, workers = _install_queue(root, monkeypatch)
    sent = []
    for wid in range(count):
        workers.WORKERS[wid] = SimpleNamespace(wid=wid, busy_task_id=None,
            in_q=SimpleNamespace(put=lambda task, wid=wid: sent.append((wid, copy.deepcopy(task)))))
    return (ctx, root, workers.PENDING), workers, sent


@pytest.mark.parametrize('workers_count', [1, 3])
def test_assign_tasks_skips_final_refusal_for_same_available_worker(world, monkeypatch, workers_count):
    world, workers, sent = _assignment_world(world, monkeypatch, workers_count)
    _, root, pending = world
    a = register(world)
    b = register(world, 'independent')
    queue.check_scheduled_tasks()
    frozen = copy.deepcopy(pending[0])
    stop(root)  # After preparation; actual final control check must refuse A.
    workers.assign_tasks()
    assert len(sent) == 1 and sent[0][0] == 0
    assert sent[0][1]['metadata']['schedule_id'] == b['id']
    assert pending == [frozen] and row(root, a['id'])['last_task_id'] == frozen['id']
    assert frozen['admitted_dispatch'] == 'none'
    workers.assign_tasks()
    assert len(sent) == 1 and pending == [frozen]


@pytest.mark.parametrize('state', ['pending', 'running', 'missing', 'unreadable', 'settled_live', 'settled'])
def test_consumed_row_retained_until_actual_work_settles(world, monkeypatch, state):
    world, workers, sent = _assignment_world(world, monkeypatch)
    _, root, pending = world
    registered = register(world)
    queue.check_scheduled_tasks()
    [task] = pending
    frozen = copy.deepcopy(task)
    data = schedules.load_schedule_store(root)
    data['tasks'][0]['completed_at'] = '2000-01-01T00:00:00+00:00'
    schedules._write_scheduled_tasks(data, root)
    path = root / 'task_results' / (task['id'] + '.json')
    if state != 'pending':
        pending.clear()
    if state in {'running', 'settled_live'}:
        workers.RUNNING[task['id']] = {'task': task, 'worker_id': 0}
    if state in {'settled', 'settled_live'}:
        write_task_result(root, task['id'], 'completed', result='Work settled')
    elif state == 'running':
        write_task_result(root, task['id'], 'running')
    elif state == 'missing':
        path.unlink()
    elif state == 'unreadable':
        path.write_text('{torn')
    queue.check_scheduled_tasks()
    kept = schedules.load_schedule_store(root)['tasks']
    assert bool(kept) is (state != 'settled')
    if state == 'pending':
        assert pending == [frozen]
        workers.assign_tasks()
        assert len(sent) == 1 and sent[0][1]['id'] == frozen['id']
        assert sent[0][1]['metadata']['billing_group'] == BINDING
        assert sent[0][1]['deadline_at'] == DEADLINE
        assert row(root, registered['id'])['completed_at'] == '2000-01-01T00:00:00+00:00'


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


def test_stop_action_receipt_survives_settlement_and_storage_failure(world, monkeypatch):
    _, root, _ = world
    first = request_cancel(root, ORIGIN, source='http_single', stop_action_id='first')
    write_task_result(root, ORIGIN, 'completed', result='Preserved completion')
    assert settle_intent(root, ORIGIN, outcome='already_settled', request_id=first['request_id'])
    stored = load_task_result(root, ORIGIN)
    replay = request_cancel(root, ORIGIN, source='http_single', stop_action_id='first', allow_settled_target=True)
    assert replay['stop_action_replayed'] and active_intent(root, ORIGIN) is None
    assert load_task_result(root, ORIGIN) == stored
    # A real later action creates authority despite the original completion.
    second = request_cancel(root, ORIGIN, source='http_single', stop_action_id='second', allow_settled_target=True)
    assert second['followup_stop']['control_id'] != first['followup_stop']['control_id']
    import ouroboros.cancel_intents as ci
    def fail(*args, **kwargs):
        raise OSError('injected durable write failure')
    with monkeypatch.context() as m:
        m.setattr(ci, 'update_json_locked', fail)
        with pytest.raises(OSError):
            request_cancel(root, ORIGIN, source='http_single', stop_action_id='third', allow_settled_target=True)
    assert active_intent(root, ORIGIN) == {k: v for k, v in second.items() if k not in {'already_requested', 'already_settled'}}
    assert load_task_result(root, ORIGIN) == stored


def test_action_identity_conflict_refuses_without_changing_authority(world):
    _, root, _ = world
    first = request_cancel(root, ORIGIN, source='http_single', stop_action_id='same',
                           requested_stop_policy='finalize_then_cancel')
    from supervisor.followup_policy import StopActionConflict
    with pytest.raises(StopActionConflict):
        request_cancel(root, ORIGIN, source='http_single', stop_action_id='same',
                       requested_stop_policy='immediate')
    assert active_intent(root, ORIGIN)['followup_stop'] == first['followup_stop']
