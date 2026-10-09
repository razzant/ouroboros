"""Late acceptance through the registered owner tool and existing paid engine."""
import copy
import json
import queue
import threading
from types import SimpleNamespace

import pytest

from ouroboros.agent_task_pipeline import _store_task_result
from ouroboros.headless import copy_child_task_result
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.reviewer_slot_config import REVIEWER_SLOTS_ENV
from supervisor import events_chat_delivery as chat
from supervisor.terminal_delivery import delivery_id_for, register_pending_delivery
from tests.test_acceptance_history import _fixture, _caller, _source, _request
from tests.test_review_operation_collection import _send_ctx, fresh_sends as fresh_sends
from tests.test_review_operation_lifetime import until
from tests._usage_store_testing import ledger_rows


@pytest.fixture
def late(tmp_path, monkeypatch, fresh_sends):
    # The model transport is synthetic; its budget quote must be offline too.
    # Cold live catalogue I/O otherwise races the panel's 10s test wait. This
    # synthetic tariff preserves real budget admission, reservation and settlement.
    from ouroboros import pricing
    for name in ("_cached_pricing", "_pricing_fetched_at", "_pricing_retry_after"):
        monkeypatch.setattr(pricing, name, {})
    monkeypatch.setattr(pricing, "_pricing_fetch_in_progress", set())
    monkeypatch.setattr(pricing, "_fetch_live_rows", lambda provider, model="":
                        {"openai/gpt-4.1-nano": (1.0, 1.0, 1.0, 1.0)}
                        if provider == "openrouter" and not model else {})
    monkeypatch.setenv('OUROBOROS_TASK_REVIEW_MODE', 'auto')
    monkeypatch.setenv(REVIEWER_SLOTS_ENV, json.dumps({'triad': [
        {'slot_id': str(i), 'route': {'kind': 'api_chat', 'target_id': 'openai/gpt-4.1-nano'}} for i in range(3)],
        'scope': [{'slot_id': 'unused-scope', 'route': {'kind': 'api_chat', 'target_id': 'openai/gpt-4.1-nano'}}]}))
    calls, gates = [], []
    config = SimpleNamespace(fail=False, verdict='PASS', slot_gates={}, slot_verdicts={})

    def transport(_self, **kwargs):
        from ouroboros.usage_accounting import AttemptRequest, current_usage_scope, execute_physical_attempt
        scope = current_usage_scope()
        def send():
            calls.append((scope, kwargs))
            for gate in gates:
                assert gate.wait(10)
            if scope.review_slot_id in config.slot_gates:
                assert config.slot_gates[scope.review_slot_id].wait(10)
            if config.fail:
                raise TimeoutError('unknown after physical dispatch')
            return {'content': json.dumps({'verdict': config.slot_verdicts.get(scope.review_slot_id, config.verdict),
                'findings': [], 'summary': 'Frozen answer checked.',
                'outcome_tier': 'best_effort', 'completion_coach': 'Retain the historical source gaps.',
                'criteria_used': [{'criterion': 'Original criterion', 'status': 'partial'}]})}, {'prompt_tokens': 1, 'completion_tokens': 1}
        return execute_physical_attempt(AttemptRequest(model=kwargs['model'], provider='openai', reservation_usd=0.01),
                                        send, extractor=lambda response: (response[1], 0.01, True))

    monkeypatch.setattr('ouroboros.llm.LLMClient.chat', transport)
    yield SimpleNamespace(calls=calls, gates=gates, config=config)
    for gate in [*gates, *config.slot_gates.values()]:
        gate.set()
    from ouroboros.review_operation import _LIVE
    until(lambda: not _LIVE)


def delivered(tmp_path, monkeypatch, *, retry=False, receipt='exact', cap='finite', automatic=False, chat_id=7,
              copyback=True, capture=None, deadline=''):
    f = _fixture(tmp_path, split=True, retry=retry, cap=cap)
    from supervisor import queue as task_queue
    monkeypatch.setattr(task_queue, 'INITIALIZED', True)
    monkeypatch.setattr(task_queue, 'DRIVE_ROOT', f.root)
    if deadline:
        f.task['task_contract']['deadline_at'] = deadline
        write_task_result(f.root, f.accounting, 'running', task_contract=f.task['task_contract'])
    if retry:
        write_task_result(f.root, f.accounting, 'failed', superseded_by=f.tid, retry_task_id=f.tid)
        f.task['supersedes_task_id'] = f.accounting
    text = 'Frozen exact delivered answer'
    event = {'type': 'send_message', 'task_id': f.tid, 'chat_id': chat_id, 'text': text, 'format': 'markdown',
             'delivery_id': delivery_id_for(f.tid, text)}
    evidence = {'task_inputs': {'text': 'Frozen owner requirements'}, **(capture(f) if capture else {})}
    _store_task_result(SimpleNamespace(drive_root=f.worker, repo_dir=tmp_path), f.task,
                      text, {}, f.trace, review_evidence=evidence, final_delivery=event)
    if copyback:
        copy_child_task_result(f.root, {'id': f.tid, 'drive_root': str(f.worker)})
    if retry and copyback:
        write_task_result(f.root, f.tid, 'completed', supersedes_task_id=f.accounting)
    f.event = event
    if receipt == 'wrong_id':
        event = {**event, 'delivery_id': delivery_id_for(f.tid, 'another identity')}
    elif receipt == 'wrong_chat':
        event = {**event, 'chat_id': 42}
    elif receipt == 'wrong_digest':
        event = {**event, 'text': 'another answer'}
    assert register_pending_delivery(f.root, event)
    monkeypatch.setattr(chat, '_bound_project_chat_id', lambda *_a: 42 if receipt == 'wrong_chat' else chat_id)
    with monkeypatch.context() as policy:
        if not automatic:
            policy.setenv('OUROBOROS_TASK_REVIEW_MODE', 'off')
        _deliver_fixture(f, event, receipt, monkeypatch)
    return f


def _deliver_fixture(f, event, receipt, monkeypatch):
    if receipt != 'owed':
        if receipt == 'registration_failure':
            with monkeypatch.context() as fault:
                fault.setattr('supervisor.terminal_delivery.register_delivery', lambda *_a, **_k: True)
                chat._handle_send_message(event, _send_ctx(f.root, []))
        else:
            chat._handle_send_message(event, _send_ctx(f.root, []))


@pytest.mark.parametrize('retry', [False, True])
@pytest.mark.parametrize('mode', ['auto', 'off'])
def test_explicit_owner_runs_full_panel_on_delivered_root_and_retry(late, tmp_path, monkeypatch, retry, mode):
    f = delivered(tmp_path, monkeypatch, retry=retry)
    monkeypatch.setenv('OUROBOROS_TASK_REVIEW_MODE', mode)
    ctx = _caller(f)
    ctx.event_queue = queue.Queue()
    before = copy.deepcopy(load_task_result(f.root, f.tid))
    result = _request(f, ctx, _source(ctx, text='Review this delivered historical answer.'))
    assert result['status'] in {'pending', 'announced', 'published', 'settled'}, result
    def panel():
        return next((p for p in (load_task_result(f.root, f.tid).get('review_projection') or {}).get('panels', [])
                     if p.get('late_settlement')), None)
    p = until(panel)
    assert p['late_settlement']['reviewed_revision'] == 'delivered'
    assert len(late.calls) == 3
    assert all(scope.task_id == f.tid and scope.root_task_id == f.accounting and scope.root_limit_usd == 4
               for scope, _ in late.calls)
    assert {scope.review_slot_id for scope, _ in late.calls} == {'0', '1', '2'}
    after = load_task_result(f.root, f.tid)
    for key in ('result', 'status', 'review_status', 'outcome_axes', 'acceptance_debt'):
        assert after.get(key) == before.get(key), key
    until(lambda: not __import__('ouroboros.review_operation', fromlist=['_LIVE'])._LIVE)
    notices = [row for row in list(ctx.event_queue.queue) if row.get('system_type') == 'acceptance_late_settlement']
    assert len(notices) == 1 and notices[0]['chat_id'] == 7
    again = _request(f, ctx, _source(ctx, text='Review this delivered historical answer.'))
    assert again['reason'] == 'existing_paid_operation' and len(late.calls) == 3


@pytest.mark.parametrize('first_callback', ['quorum', 'complete'])
@pytest.mark.parametrize('last_verdict', ['PASS', 'FAIL'])
def test_delayed_quorum_callback_keeps_one_final_historical_notice(
        late, tmp_path, monkeypatch, first_callback, last_verdict):
    """A quorum snapshot may reach publication after the last producer settles."""
    from ouroboros import acceptance_settlement as settlement
    from ouroboros import review_operation
    from ouroboros.artifacts import read_actor_source_bytes
    from supervisor.terminal_delivery import pending_deliveries

    f = delivered(tmp_path, monkeypatch)
    ctx = _caller(f)
    ctx.event_queue = queue.Queue()
    before = copy.deepcopy(load_task_result(f.root, f.tid))
    late.config.slot_gates = {slot: threading.Event() for slot in ('0', '1', '2')}
    late.config.slot_verdicts['2'] = last_verdict
    entered = {kind: threading.Event() for kind in ('quorum', 'complete')}
    release = {kind: threading.Event() for kind in entered}
    returned = {kind: threading.Event() for kind in entered}
    announce = settlement.announce_acceptance_settlement
    waves = {}

    def held_announcement(usage_ctx, request, wave):
        kind = 'complete' if all(wave['slots'].values()) else 'quorum'
        waves[kind] = copy.deepcopy(wave)
        entered[kind].set()
        try:
            assert release[kind].wait(10)
            return announce(usage_ctx, request, wave)
        finally:
            returned[kind].set()

    monkeypatch.setattr(settlement, 'announce_acceptance_settlement', held_announcement)
    try:
        result = _request(f, ctx, _source(ctx, text='Review this delivered historical answer.'))
        assert result['status'] == 'pending', result
        late.config.slot_gates['0'].set()
        late.config.slot_gates['1'].set()
        assert entered['quorum'].wait(10)
        late.config.slot_gates['2'].set()
        assert entered['complete'].wait(10)  # All producer facts are now collectable.
        release[first_callback].set()
        assert returned[first_callback].wait(10)
        other = 'complete' if first_callback == 'quorum' else 'quorum'
        release[other].set()
        assert returned[other].wait(10)
    finally:
        for gate in [*late.config.slot_gates.values(), *release.values()]:
            gate.set()
        until(lambda: not review_operation._LIVE)

    assert waves['quorum']['slots']['2'] == ''
    assert waves['complete']['verdicts']['2']['verdict'] == last_verdict
    after = load_task_result(f.root, f.tid)
    panel, = after['review_projection']['panels']
    source = json.loads(read_actor_source_bytes(f.root, f.tid, panel['applied_source_ref']))
    assert {actor['slot_id']: actor['parsed']['verdict'] for actor in source['actors']} == {
        '0': 'PASS', '1': 'PASS', '2': last_verdict}
    assert source['request']['subject'] == before['result']
    for key in ('result', 'status', 'review_status', 'outcome_axes', 'acceptance_debt'):
        assert after.get(key) == before.get(key), key
    assert len(late.calls) == 3
    assert all(scope.task_id == f.tid and scope.root_task_id == f.accounting and scope.root_limit_usd == 4
               for scope, _ in late.calls)
    assert all(row['state'] == 'collected' for row in after['review_operations'].values())
    assert not settlement.late_publication_owed(f.tid, source['request']['retry_key'])
    notices = [row for row in list(ctx.event_queue.queue) if row.get('system_type') == 'acceptance_late_settlement']
    owed, = pending_deliveries(f.root)
    sends = []
    sender = _send_ctx(f.root, sends)
    for notice in [*notices, owed]:  # The real consumer also sees an outbox replay.
        chat._handle_send_message(notice, sender)
    assert len(sends) == 1 and sends[0][0] == 7, sends
    # The owner row says each outcome in words (DESIGN §4), one seat per roster slot in
    # roster order, so the final notice reads slot 2's own verdict at seat 3.
    words = {'PASS': 'passed it', 'FAIL': 'rejected it'}[last_verdict]
    seats = [line.split(' — ')[0] for line in sends[0][1].split('\n')[1:]]
    assert seats == [f'- openai/gpt-4.1-nano (requested) (seat {seat}): {outcome}'
                     for seat, outcome in ((1, 'passed it'), (2, 'passed it'), (3, words))], {
        'notices': len(notices), 'sent': sends, 'retained': panel['late_settlement']['note']}
    assert len(notices) == 1, notices
    assert sends[0][1] == owed['text'] == panel['late_settlement']['note'] == source['late_settlement']['note']
    assert not pending_deliveries(f.root)
    again = _request(f, ctx, _source(ctx, text='Review this delivered historical answer again.'))
    assert again['reason'] == 'existing_paid_operation' and len(late.calls) == 3


@pytest.mark.parametrize('order', ['drain_first', 'together'])
def test_drain_and_last_slot_callback_settling_one_wave_queue_one_live_notice(late, tmp_path, monkeypatch, order):
    """The drain's own settlement and the last slot's complete callback settle one wave.

    ``drain_first``: the drain collects every slot and announces; the callback's
    mailbox duty mark then lands and its settlement republishes. ``together``:
    both reconcile the one pending run at once. The outbox owes one notice and
    exactly one live copy is queued; no reviewer is bought again.
    """
    from ouroboros import acceptance_settlement as settlement
    from ouroboros import review_dispatch, review_operation
    from supervisor.terminal_delivery import pending_deliveries

    f = delivered(tmp_path, monkeypatch)
    ctx = _caller(f)
    ctx.event_queue = queue.Queue()
    drain = threading.current_thread()
    complete, drain_entered, drain_done = threading.Event(), threading.Event(), threading.Event()
    announce, settle = settlement.announce_acceptance_settlement, settlement.settle_acceptance_operation
    collect, met, barrier = review_dispatch.collect_task_acceptance_run, [], threading.Barrier(2, timeout=10)

    def last_slot_callback(usage_ctx, request, wave):
        if all(wave['slots'].values()):
            complete.set()
            assert (drain_done if order == 'drain_first' else drain_entered).wait(10)
        return announce(usage_ctx, request, wave)

    def drain_settlement(usage_ctx, **kwargs):
        if threading.current_thread() is not drain:
            return settle(usage_ctx, **kwargs)
        assert complete.wait(10)  # every slot is already in custody
        drain_entered.set()
        try:
            return settle(usage_ctx, **kwargs)
        finally:
            drain_done.set()

    def meeting_collect(run, **kwargs):
        barrier.wait()
        met.append(threading.current_thread().name)
        return collect(run, **kwargs)

    monkeypatch.setattr(settlement, 'announce_acceptance_settlement', last_slot_callback)
    monkeypatch.setattr(settlement, 'settle_acceptance_operation', drain_settlement)
    if order == 'together':
        monkeypatch.setattr(review_dispatch, 'collect_task_acceptance_run', meeting_collect)
    try:
        result = _request(f, ctx, _source(ctx, text='Review this delivered historical answer.'))
    finally:
        for gate in (complete, drain_entered, drain_done):
            gate.set()
        until(lambda: not review_operation._LIVE)

    assert result['status'] in {'announced', 'published'}, result
    assert len(met) == (2 if order == 'together' else 0), met
    notices = [row for row in list(ctx.event_queue.queue) if row.get('system_type') == 'acceptance_late_settlement']
    owed, = pending_deliveries(f.root)
    panel, = load_task_result(f.root, f.tid)['review_projection']['panels']
    assert len(notices) == 1 and notices[0]['chat_id'] == 7, notices
    assert notices[0]['delivery_id'] == owed['delivery_id'] and notices[0]['text'] == owed['text']
    assert owed['text'] == panel['late_settlement']['note'] and len(late.calls) == 3, json.dumps([
        {key: actor.get(key) for key in ('slot_id', 'transport_status', 'parse_status', 'operation_state', 'reason')}
        for actor in panel['actors']])
    if order == 'drain_first':  # the callback's duty mark was consumed, not left to re-announce
        assert not settlement.late_publication_owed(f.tid, panel['late_settlement']['reviewed_subject']['retry_key'])


def _late_effects(f):
    from ouroboros import review_operation
    return {'live': list(review_operation._LIVE), 'ledger': ledger_rows(f.root),
            'rows': {tid: {k: v for k, v in load_task_result(f.root, tid).items()
                           if k not in ('acceptance_root_cap_amendments', 'updated_at')} for tid in (f.tid, f.accounting)},
            'sources': sorted(p.name for p in f.root.rglob('source_handles/context_checkpoints/*'))}


def _paid_facts(f, out, late):
    root = load_task_result(f.root, f.accounting).get('task_acceptance_review_accounting') or {}
    return {'status': out['status'], 'reason': out['reason'], 'dispatched': out['dispatched'], 'calls': len(late.calls),
            'operations': len(load_task_result(f.root, f.tid).get('review_operations') or {}),
            'claims': len(root.get('claims_by_binding') or {})}


@pytest.mark.parametrize('action,retry', [('amend_cap', False), ('amend_cap', True), (None, True)])
def test_cap_only_amendment_prepares_and_buys_nothing_even_with_every_gate_open(late, tmp_path, monkeypatch, action, retry):
    """Owner 2A: the absolute original-root cap is money only; a panel is requested separately."""
    from ouroboros import review_operation
    from ouroboros.acceptance_history import historical_review_controls
    from ouroboros.acceptance_late import _receipt
    f = delivered(tmp_path, monkeypatch, retry=retry)
    ctx = _caller(f)
    ctx.event_queue = queue.Queue()
    target = load_task_result(f.root, f.tid)
    # Exact delivered zero-physical debt; no Stop/Pause/fence/deadline/prohibition anywhere.
    assert _receipt(f.root, target['acceptance_debt'])
    assert historical_review_controls(f.root, target, target['acceptance_debt'], caller_task_id=ctx.task_id,
                                      automatic=False) == []
    before = _late_effects(f)
    out = _request(f, ctx, _source(ctx, text='Raise that task root cap to 9 USD; no review now.'),
                   new_original_root_cap_usd=9, **({'action': action} if action else {}))
    until(lambda: not review_operation._LIVE)
    after = _late_effects(f)
    stored = set(after.pop('sources')) - set(before.pop('sources'))
    assert (out['status'], out['dispatched'], late.calls) == ('amended', False, []), _paid_facts(f, out, late)
    assert after == before
    assert [name.split('-')[0] for name in stored] == ['historical_owner_request'], stored
    assert out['review_requested'] is False and out['execution_blocked_by'] == [], out
    assert out['effective_original_root_cap'] == {'state': 'finite', 'usd': 9, 'source': 'owner_amendment'}
    amendments = load_task_result(f.root, f.accounting)['acceptance_root_cap_amendments']
    assert [(a['new_cap_usd'], a['previous_cap']['usd']) for a in amendments] == [(9, 4)]
    assert not [e for e in list(ctx.event_queue.queue) if e.get('system_type') == 'acceptance_late_settlement']
    # A later explicit panel buys from the same original root under the amended cap.
    review = _request(f, ctx, _source(ctx, text='Now review that historical answer.'), action='review')
    assert review['status'] in {'pending', 'announced', 'published', 'settled'}, review
    until(lambda: len(late.calls) == 3)
    assert all(scope.task_id == f.tid and scope.root_task_id == f.accounting and scope.root_limit_usd == 9
               and scope.root_limit_source == 'owner_amendment' for scope, _ in late.calls)
    until(lambda: not review_operation._LIVE)
    root = load_task_result(f.root, f.accounting)
    assert root['acceptance_root_cap_amendments'] == amendments
    assert len(root['task_acceptance_review_accounting']['claims_by_binding']) == 1


def test_explicit_review_may_combine_the_absolute_cap_before_its_one_panel(late, tmp_path, monkeypatch):
    from ouroboros import review_operation
    f = delivered(tmp_path, monkeypatch, retry=True)
    ctx = _caller(f)
    out = _request(f, ctx, _source(ctx), action='review', new_original_root_cap_usd=9)
    assert out['status'] in {'pending', 'announced', 'published', 'settled'}, out
    assert out['action'] == 'review' and out['cap_amendment']['new_cap_usd'] == 9, out
    until(lambda: len(late.calls) == 3)
    assert all(scope.root_task_id == f.accounting and scope.root_limit_usd == 9 for scope, _ in late.calls)
    until(lambda: not review_operation._LIVE)


@pytest.mark.parametrize('extra,reason', [
    ({'action': None}, 'late_review_action_invalid'),
    ({'action': None, 'new_original_root_cap_usd': None}, 'late_review_action_invalid'),
    ({'action': False}, 'late_review_action_invalid'),
    ({'action': {}}, 'late_review_action_invalid'),
    ({'action': 'amend'}, 'late_review_action_invalid'),
    ({'action': 'Review'}, 'late_review_action_invalid'),
    ({'action': ''}, 'late_review_action_invalid'),
    ({'action': ['amend_cap']}, 'late_review_action_invalid'),
    ({'action': 'amend_cap', 'new_original_root_cap_usd': None}, 'absolute_root_cap_required'),
    ({'action': 'review', 'new_original_root_cap_usd': 0}, 'absolute_root_cap_must_be_positive_finite'),
])
def test_malformed_late_action_is_refused_before_any_effect(late, tmp_path, monkeypatch, extra, reason):
    from ouroboros import review_operation
    f = delivered(tmp_path, monkeypatch, retry=True)
    ctx = _caller(f)
    source = _source(ctx)
    before = _late_effects(f)
    amendments = load_task_result(f.root, f.accounting).get('acceptance_root_cap_amendments')
    out = _request(f, ctx, source, **{'new_original_root_cap_usd': 9, **extra})
    until(lambda: not review_operation._LIVE)
    assert (out['status'], out['reason'], late.calls) == ('refused', reason, []), _paid_facts(f, out, late)
    after = _late_effects(f)
    assert after == before, set(after.pop('sources')) ^ set(before.pop('sources'))
    assert load_task_result(f.root, f.accounting).get('acceptance_root_cap_amendments') == amendments


@pytest.mark.parametrize('receipt', ['wrong_id', 'wrong_chat', 'wrong_digest', 'owed', 'registration_failure'])
def test_explicit_receipt_requires_id_routed_chat_and_exact_bytes(late, tmp_path, monkeypatch, receipt):
    f = delivered(tmp_path, monkeypatch, receipt=receipt)
    ctx = _caller(f)
    result = _request(f, ctx, _source(ctx))
    assert result['reason'] == 'exact_delivery_unconfirmed', result
    assert not late.calls and not load_task_result(f.root, f.accounting).get('task_acceptance_review_accounting')


@pytest.mark.parametrize('cap', ['unlimited', 'unknown', 'prior_hold', 'prior_spend', 'spent_at_cap',
                                 'spent_over_cap', 'live_global', 'live_global_spent'])
def test_original_cap_and_live_money_fences_use_original_wallet(late, tmp_path, monkeypatch, cap):
    import time
    from ouroboros import pricing, review_operation
    from ouroboros.usage_accounting import AttemptRequest, execute_physical_attempt, reserve_attempt, usage_projection
    # The wave fence needs a priced reviewer seat: an unpriced one adds nothing and fits. Pin the
    # catalog row; a live fetch that times out, or a test's leftover 30 s retry_after, leaves none.
    monkeypatch.setitem(pricing._cached_pricing, 'openrouter', {'openai/gpt-4.1-nano': (0.1, 0.025, None, 0.4)})
    monkeypatch.setitem(pricing._pricing_fetched_at, 'openrouter', time.time())
    live = cap.startswith('live_global')
    f = delivered(tmp_path, monkeypatch, retry=True,
                  cap='unlimited' if cap == 'unlimited' or live else 'unknown' if cap == 'unknown' else 'finite')
    monkeypatch.setenv('OUROBOROS_PER_TASK_COST_USD', '0.00001')

    def paid(usd, root):
        # A settled actual price is KNOWN spend: admitted while known is below the
        # cap, and recorded whole even where it lands beyond it (#1487).
        execute_physical_attempt(AttemptRequest(model='openai/gpt-4.1-nano', provider='openai', drive_root=f.root,
            task_id=root, root_task_id=root, root_limit_usd=4, reservation_usd=usd),
            lambda: 'original paid work', extractor=lambda _response: ({}, usd, True))
    if cap == 'prior_hold':
        # An open hold is disclosed exposure, never spending: it refuses nothing.
        reserve_attempt(AttemptRequest(model='openai/gpt-4.1-nano', provider='openai', drive_root=f.root,
            task_id=f.accounting, root_task_id=f.accounting, root_limit_usd=4, reservation_usd=3.9999))
    if cap in {'prior_spend', 'spent_at_cap', 'spent_over_cap'}:
        paid({'prior_spend': 3.9999, 'spent_at_cap': 4, 'spent_over_cap': 4.5}[cap], f.accounting)
    if cap == 'live_global_spent':
        paid(0.00001, 'unrelated-root')  # another root's known spend reaches the live global cap exactly
    if live:
        monkeypatch.setattr('ouroboros.settings_setup_contract.resolve_total_budget_usd', lambda: 0.00001)
    ctx = _caller(f)
    gate = threading.Event()
    late.gates.append(gate)
    result = _request(f, ctx, _source(ctx))
    if cap in {'unlimited', 'prior_hold', 'prior_spend', 'live_global'}:
        assert result['status'] in {'pending', 'announced', 'published', 'settled'}, result
        # The gate holds every settlement: all three seats pass reserve and dispatch
        # on the same known spend, whatever the prior and sibling holds add up to.
        until(lambda: len(late.calls) == 3)
        gate.set()
        until(lambda: not review_operation._LIVE)
        assert all(scope.root_limit_usd == (4 if cap.startswith('prior') else None) and scope.root_limit_source
                   for scope, _ in late.calls)
        if cap == 'prior_spend':
            # Seats sent in parallel on known $3.9999 overshoot the $4 cap: recorded, not prevented.
            assert usage_projection(f.root, root_task_id=f.accounting)['settled_usd'] == pytest.approx(4.0299)
    else:
        assert result['status'] == 'owed', result
        assert result['reason'] == ('original_root_cap_unknown' if cap == 'unknown' else 'review_wave_budget_insufficient')
        assert not late.calls
        assert not load_task_result(f.root, f.accounting).get('task_acceptance_review_accounting')


@pytest.mark.parametrize('stopped', ['target', 'caller', 'root'])
def test_terminal_http_stop_reaches_the_live_historical_operation(late, tmp_path, monkeypatch, stopped):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient
    from ouroboros.gateway import tasks
    from ouroboros.cancel_intents import cancel_pending
    from ouroboros import review_operation
    from supervisor import queue as task_queue

    f = delivered(tmp_path, monkeypatch, retry=True)
    gate = threading.Event()
    late.gates.append(gate)
    ctx = _caller(f)
    result = _request(f, ctx, _source(ctx))
    assert result['status'] == 'pending', result
    until(lambda: len(late.calls) == 3)
    write_task_result(f.root, ctx.task_id, 'completed', result='Current author finished')
    target = {'target': f.tid, 'caller': ctx.task_id, 'root': f.accounting}[stopped]
    monkeypatch.setattr(task_queue, 'DRIVE_ROOT', f.root)
    assert task_queue.task_has_live_ownership(target)
    app = Starlette(routes=[Route('/api/tasks/{task_id}/cancel', tasks.api_task_cancel, methods=['POST'])])
    app.state.drive_root = f.root
    with TestClient(app) as client:
        response = client.post(f'/api/tasks/{target}/cancel', json={})
    assert response.status_code == 503, response.text  # still owned, never a false 404 or completed cancellation
    assert cancel_pending(f.root, ctx.task_id if stopped == 'caller' else f.tid, strict=True)
    operation = next(op for op in review_operation._LIVE.values() if op.task_id == f.tid)
    until(lambda: operation.control() == 'cancelled')
    assert load_task_result(f.root, f.tid)['result'] == f.event['text']
    gate.set()
    from ouroboros.review_operation import _LIVE
    until(lambda: not _LIVE)
    panels = load_task_result(f.root, f.tid)['review_projection']['panels']
    assert len(panels) == 1 and panels[0].get('late_settlement')


@pytest.mark.parametrize('failed_publish', ['', 'publication', 'outbox'])
def test_immediate_complete_and_failed_publication_retry_keep_one_supplement(late, tmp_path, monkeypatch, failed_publish):
    import time
    from ouroboros import review_operation, review_substrate, review_projection
    from supervisor.terminal_delivery import pending_deliveries
    f = delivered(tmp_path, monkeypatch, retry=True)
    ctx = _caller(f)
    ctx.event_queue = queue.Queue()
    original_run = review_substrate.run_review_request
    def synchronous(request, **kwargs):
        request.drain_deadline = time.monotonic() + 5
        result = original_run(request, **kwargs)
        assert all(actor['operation_state'] not in {'pending_dispatch', 'in_flight'} for actor in result.actors)
        return result
    monkeypatch.setattr(review_substrate, 'run_review_request', synchronous)
    with monkeypatch.context() as fault:
        if failed_publish == 'publication':
            fault.setattr(review_projection, 'publish_acceptance_checkpoint', lambda *_a, **_k: {'status': 'unavailable'})
        elif failed_publish == 'outbox':
            fault.setattr('supervisor.terminal_delivery.register_pending_delivery', lambda *_a, **_k: False)
        result = _request(f, ctx, _source(ctx))
        until(lambda: not review_operation._LIVE)
        if failed_publish:
            assert result['status'] == 'unpublished', result
            if failed_publish == 'publication':
                assert not [e for e in list(ctx.event_queue.queue) if e.get('system_type') == 'acceptance_late_settlement']
            assert not pending_deliveries(f.root)
    if failed_publish:
        # A cold collector has no local routing variables; current chat must
        # never retarget the retained historical delivery or close duty early.
        from ouroboros import acceptance_settlement
        before = load_task_result(f.root, f.tid)
        assert any(p['state'] == 'unpublished' for p in before['review_operations'].values())
        write_task_result(f.root, f.tid, 'completed', chat_id=99999)
        with acceptance_settlement._LATE_LOCK:
            acceptance_settlement._LATE_UNPUBLISHED.clear()
        report = review_operation.recover_orphaned_acceptance_operations(f.root)
        assert report['settled'], report
    panels = load_task_result(f.root, f.tid)['review_projection']['panels']
    assert len(panels) == 1 and panels[0]['late_settlement']['reviewed_revision'] == 'delivered'
    assert panels[0]['panel_id'] == result['panel_id']
    assert len(late.calls) == 3
    owed = pending_deliveries(f.root)
    assert len(owed) == 1 and owed[0]['delivery_id'].startswith('acceptance-late:task_acceptance:')
    assert owed[0]['chat_id'] == 7
    assert all(p['state'] == 'collected' for p in load_task_result(f.root, f.tid)['review_operations'].values())
    _request(f, ctx, _source(ctx, text='Another rationale on the same debt'))
    assert len(pending_deliveries(f.root)) == 1 and len(late.calls) == 3


@pytest.mark.parametrize('outcome', ['unknown', 'FAIL'])
@pytest.mark.parametrize('original_signal', ['PASS', 'FAIL'])
def test_unknown_or_critic_fail_never_resends_or_changes_original_decision(late, tmp_path, monkeypatch, outcome, original_signal):
    from ouroboros.review_operation import _LIVE
    f = delivered(tmp_path, monkeypatch)
    original = load_task_result(f.root, f.tid)
    decision = {'status': 'complete', 'acceptance_decision': {'signal': original_signal, 'rationale': 'Original author decision'}}
    write_task_result(f.root, f.tid, original['status'], review_status=decision)
    ctx = _caller(f)
    task_rows = set((f.root / 'task_results').glob('*.json'))
    late.config.fail = outcome == 'unknown'
    late.config.verdict = outcome
    _request(f, ctx, _source(ctx))
    until(lambda: not _LIVE)
    assert len(late.calls) == 3
    after = load_task_result(f.root, f.tid)
    assert after['review_status'] == decision and after['result'] == original['result']
    assert after['status'] == original['status'] and after['acceptance_debt'] == original['acceptance_debt']
    if outcome == 'FAIL':
        assert after['review_projection']['panels'][0]['aggregate_signal'] == 'FAIL'
    result = _request(f, ctx, _source(ctx, text='A fresh source does not grant a second paid identity'))
    assert result['reason'] == 'existing_paid_operation' and len(late.calls) == 3
    assert set((f.root / 'task_results').glob('*.json')) == task_rows


def test_current_caller_calendar_survives_a_fresh_operation_window(late, tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone
    from ouroboros import review_operation
    f = delivered(tmp_path, monkeypatch, retry=True)
    ctx = _caller(f)
    future = datetime.now(timezone.utc) + timedelta(minutes=10)
    ctx.task_metadata['deadline_at'] = future.isoformat()
    gate = threading.Event()
    late.gates.append(gate)
    result = _request(f, ctx, _source(ctx))
    assert result['status'] == 'pending', result
    until(lambda: len(late.calls) == 3)
    operation = next(op for op in review_operation._LIVE.values() if op.task_id == f.tid)
    # Ending the new caller and mutating its context cannot erase its captured
    # human calendar limit from a fresh historical operation's lifetime.
    write_task_result(f.root, ctx.task_id, 'completed')
    ctx.task_metadata['deadline_at'] = ''
    monkeypatch.setattr('ouroboros.deadline_utils.utc_now', lambda: future + timedelta(seconds=1))
    until(lambda: operation.control() == 'cancelled')
    gate.set()
    until(lambda: operation.closed)
    assert len(late.calls) == 3
    assert load_task_result(f.root, f.tid)['result'] == f.event['text']


def test_two_explicit_entries_share_one_paid_identity(late, tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    f = delivered(tmp_path, monkeypatch)
    gate = threading.Event()
    late.gates.append(gate)
    ctx = _caller(f)
    source = _source(ctx)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: _request(f, ctx, source), range(2)))
    until(lambda: len(late.calls) == 3)
    claims = load_task_result(f.root, f.accounting)['task_acceptance_review_accounting']['claims_by_binding']
    assert len(claims) == 1, results
    gate.set()
    from ouroboros.review_operation import _LIVE
    until(lambda: not _LIVE)
    panels = load_task_result(f.root, f.tid)['review_projection']['panels']
    assert len(panels) == 1 and panels[0].get('late_settlement')


@pytest.mark.parametrize('automatic,retry,interruption', [
    (False, False, ''), (True, False, ''), (False, True, ''), (True, True, ''),
    *[(False, False, reason) for reason in (
        'stop', 'panic', 'paid', 'foreign_binding', 'foreign_operation',
        'in_flight', 'unknown', 'foreign_controller', 'changed_intent', 'changed_receipt')],
])
def test_pending_publication_before_first_paid_stamp_keeps_exact_live_authority(
        late, tmp_path, monkeypatch, automatic, retry, interruption):
    """Force the real sender/tool to return before the first physical claim."""
    from ouroboros import review_dispatch, review_operation
    from ouroboros.task_results import claim_task_acceptance_review_cycle

    f = delivered(tmp_path, monkeypatch, retry=retry, receipt='owed')
    before = copy.deepcopy(load_task_result(f.root, f.tid))
    entered, release = threading.Event(), threading.Event()
    original_stamp = review_dispatch.invoke_review_paid_stamp

    def held_stamp(stamp):
        if callable(stamp) and getattr(stamp, 'fail_closed', False):
            entered.set()
            assert release.wait(10)
        return original_stamp(stamp)

    monkeypatch.setattr(review_dispatch, 'invoke_review_paid_stamp', held_stamp)
    try:
        if automatic:
            chat._handle_send_message(f.event, _send_ctx(f.root, []))
            until(lambda: (load_task_result(f.root, f.tid).get('review_projection') or {}).get('panels'))
        else:
            with monkeypatch.context() as mode:
                mode.setenv('OUROBOROS_TASK_REVIEW_MODE', 'off')
                chat._handle_send_message(f.event, _send_ctx(f.root, []))
            ctx = _caller(f)
            result = _request(f, ctx, _source(ctx))
            assert result['status'] == 'pending', result
        assert entered.wait(5)
        row = load_task_result(f.root, f.tid)
        panel = row['review_projection']['panels'][0]
        pointer = next(iter(row['review_operations'].values()))
        assert pointer['source_ref'] and pointer['state'] == 'dispatched'
        assert {a['operation_state'] for a in panel['actors']} == {'pending_dispatch'}
        assert not late.calls and not load_task_result(f.root, f.accounting).get('task_acceptance_review_accounting')
        if interruption == 'stop':
            from ouroboros.cancel_intents import request_cancel
            request_cancel(f.root, f.tid, allow_settled_target=True)
        elif interruption == 'panic':
            (f.root / 'state' / 'panic_stop.flag').write_text('panic')
        elif interruption == 'paid':
            claim = claim_task_acceptance_review_cycle(f.root, f.accounting,
                {key: panel[key] for key in ('binding_hash', 'candidate_hash', 'evidence_revision', 'fence_hash')},
                claimed_by_task_id=f.tid)
            assert claim['status'] == 'claimed'
        elif interruption in {'foreign_binding', 'foreign_operation', 'in_flight', 'unknown'}:
            projection = copy.deepcopy(row['review_projection'])
            actor = projection['panels'][0]['actors'][0]
            if interruption == 'foreign_binding':
                projection['panels'][0]['binding_hash'] = '0' * 64
            elif interruption == 'foreign_operation':
                actor['operation_id'] = 'other-physical-operation'
            else:
                actor['operation_state'] = interruption
            # Fault injection bypasses the normal publisher's conflict merge;
            # prove the conflicting canonical bytes reached the paid seam.
            write_task_result(f.root, f.tid, row['status'],
                _field_projector=lambda _row, _fields: {'review_projection': projection})
            assert load_task_result(f.root, f.tid)['review_projection'] == projection
        elif interruption in {'foreign_controller', 'changed_intent'}:
            pointers = copy.deepcopy(row['review_operations'])
            pointer = next(iter(pointers.values()))
            pointer['controller' if interruption == 'foreign_controller' else 'intent_ref'] = {}
            write_task_result(f.root, f.tid, row['status'], review_operations=pointers)
        elif interruption == 'changed_receipt':
            from supervisor import terminal_delivery
            monkeypatch.setattr(terminal_delivery, 'terminal_answer_receipts', lambda *_a: {'delivered': []})
    finally:
        release.set()
    until(lambda: not review_operation._LIVE)
    after = load_task_result(f.root, f.tid)
    claims = (load_task_result(f.root, f.accounting).get('task_acceptance_review_accounting') or {}).get('claims_by_binding') or {}
    facts = {'pointers': after.get('review_operations'), 'projection': after.get('review_projection'), 'claims': claims}
    if interruption:
        assert not late.calls, facts
        assert len(claims) == (1 if interruption == 'paid' else 0), facts
    else:
        assert len(late.calls) == 3 and len(claims) == 1, facts
        assert all(scope.root_task_id == f.accounting and scope.root_limit_usd == 4 for scope, _ in late.calls)
        assert len(after['review_projection']['panels']) == 1
        assert after['review_projection']['panels'][0]['late_settlement']
        for field in ('result', 'status', 'review_status', 'outcome_axes', 'acceptance_debt'):
            assert after.get(field) == before.get(field), field
        ctx = _caller(f)
        _request(f, ctx, _source(ctx))
        chat._handle_send_message(f.event, _send_ctx(f.root, []))
        assert len(late.calls) == 3


@pytest.mark.parametrize('boundary', ['before_pointer', 'after_pointer', 'after_claim'])
def test_handoff_failures_preserve_unknown_identity_without_resend(late, tmp_path, monkeypatch, boundary):
    from ouroboros import review_operation, review_dispatch
    f = delivered(tmp_path, monkeypatch, retry=True)
    ctx = _caller(f)
    source = _source(ctx)
    with monkeypatch.context() as fault:
        if boundary == 'after_claim':
            original = review_dispatch.invoke_review_paid_stamp
            def fail(stamp):
                original(stamp)
                if callable(stamp) and getattr(stamp, 'fail_closed', False):
                    raise review_dispatch.TaskAcceptanceDispatchUnavailable('simulated crash after durable claim')
            fault.setattr(review_dispatch, 'invoke_review_paid_stamp', fail)
        else:
            original = review_operation._write_operation_pointer
            def fail(*args, **kwargs):
                if boundary == 'after_pointer':
                    original(*args, **kwargs)
                raise OSError('simulated pointer handoff failure')
            fault.setattr(review_operation, '_write_operation_pointer', fail)
        first = _request(f, ctx, source)
        until(lambda: not review_operation._LIVE)
        assert not late.calls, first
    pointers = load_task_result(f.root, f.tid).get('review_operations') or {}
    assert pointers  # the preparation intent precedes the complete paid-request pointer
    assert any(p.get('source_ref') for p in pointers.values()) is (boundary != 'before_pointer')
    # Only a recorded refusal, never an unknown preparation, leaves the debt retryable below.
    assert boundary != 'before_pointer' or [p.get('state') for p in pointers.values()] == ['preparation_refused'], pointers
    claims = (load_task_result(f.root, f.accounting).get('task_acceptance_review_accounting') or {}).get('claims_by_binding') or {}
    assert bool(claims) is (boundary == 'after_claim')
    second = _request(f, ctx, _source(ctx, text='A separately requested retry after the handoff fault'))
    if boundary == 'before_pointer':
        until(lambda: len(late.calls) == 3)
    else:
        assert second['reason'] == 'existing_paid_operation' and not late.calls, second


@pytest.mark.parametrize('reader', ['author', 'paid_stamp'])
def test_a_read_denied_by_a_concurrent_replace_never_refuses_the_owner_panel(late, tmp_path, monkeypatch, reader):
    """Windows denies an open that meets another thread's atomic replace of the same
    task result. One such instant at a strict preclaim read (the author's before
    dispatch, or the paid stamp's on a reviewer racing the author's publication) is
    not unreadable authority: the owner's panel still buys its three reviewers once."""
    import pathlib
    from ouroboros import review_dispatch, review_operation
    f = delivered(tmp_path, monkeypatch, retry=True)
    ctx = _caller(f)
    inside, denied = set(), []
    preclaim, read_text = review_dispatch.task_acceptance_preclaim_refusal, pathlib.Path.read_text

    def observed_preclaim(admission):
        if (threading.current_thread() is not threading.main_thread()) == (reader == 'paid_stamp'):
            inside.add(threading.get_ident())
        try:
            return preclaim(admission)
        finally:
            inside.discard(threading.get_ident())

    def denied_once(path, *args, **kwargs):
        if threading.get_ident() in inside and path.parent.name == 'task_results' and not denied:
            denied.append(path.name)
            raise PermissionError(13, 'The process cannot access the file', str(path))
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(review_dispatch, 'task_acceptance_preclaim_refusal', observed_preclaim)
    monkeypatch.setattr(pathlib.Path, 'read_text', denied_once)
    result = _request(f, ctx, _source(ctx))
    assert result['status'] in {'pending', 'announced', 'published', 'settled'}, result
    until(lambda: len(late.calls) == 3)
    until(lambda: not review_operation._LIVE)
    claims = (load_task_result(f.root, f.accounting).get('task_acceptance_review_accounting') or {}).get('claims_by_binding') or {}
    assert denied and len(late.calls) == 3 and len(claims) == 1, (denied, claims)


@pytest.mark.parametrize('amount,explicit,expected', [(None, 'original_admission', None), (2.0, 'original_admission', 2.0), (None, '', 9.0)])
def test_review_substrate_preserves_bound_original_unlimited_cap(tmp_path, monkeypatch, amount, explicit, expected):
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request
    from ouroboros.usage_accounting import UsageScope, current_usage_scope, usage_scope

    monkeypatch.setenv('OUROBOROS_PER_TASK_COST_USD', '9')
    captured = []

    class Observed:
        def chat(self, **kwargs):
            captured.append(current_usage_scope())
            return {'content': json.dumps({'verdict': 'PASS', 'findings': [], 'summary': 'ok'})}, {}

    with usage_scope(UsageScope(drive_root=tmp_path, task_id='late', root_task_id='original',
                               root_limit_usd=amount, root_limit_source=explicit)):
        run_review_request(ReviewRequest(surface='task_acceptance', goal='review', task_id='late'),
                           slots=[ReviewSlot(slot_id='slot', model='test/model')], drive_root=tmp_path, llm=Observed())
    assert captured[0].root_task_id == 'original'
    assert captured[0].root_limit_usd == expected and captured[0].root_limit_source == explicit


def test_terminal_control_addresses_survive_intent_to_request_upgrade(late, tmp_path, monkeypatch):
    from ouroboros import review_operation
    f = delivered(tmp_path, monkeypatch, retry=True)
    ctx = _caller(f)
    original = review_operation._link_historical_controls
    observed = []
    def inspect_upgrade(operation, entry):
        if entry.get('source_ref'):
            # Primary now has the full request, while caller/root still address
            # the predispatch intent. Both remain exact live control addresses.
            assert review_operation.task_has_live_review_operation(f.root, ctx.task_id)
            assert review_operation.task_has_live_review_operation(f.root, f.accounting)
            observed.append(entry['intent_ref'])
        return original(operation, entry)
    monkeypatch.setattr(review_operation, '_link_historical_controls', inspect_upgrade)
    _request(f, ctx, _source(ctx))
    until(lambda: not review_operation._LIVE)
    assert observed and len(late.calls) == 3
