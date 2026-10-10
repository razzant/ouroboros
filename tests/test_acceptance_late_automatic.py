"""Actual terminal sender/outbox, preparation owner and registered explicit consumers."""
import copy
import hashlib
import json
import queue
import threading
from types import SimpleNamespace

import pytest

from tests.test_acceptance_late_consumers import late as late, delivered
from tests.test_acceptance_history import _caller, _source, _request
from tests.test_review_operation_collection import _send_ctx, fresh_sends as fresh_sends
from tests.test_review_operation_lifetime import until
from ouroboros.task_results import load_task_result
from ouroboros import review_operation
from supervisor import events_chat_delivery as chat
from tests.test_terminal_file_boundary import terminal_context as terminal_context


@pytest.mark.parametrize('retry', [False, True])
@pytest.mark.parametrize('order', ['receipt_first', 'canonical_first', 'checkpoint_first', 'file_recovery'])
@pytest.mark.parametrize('receipt,control', [('owed', ''), ('wrong_id', ''), ('wrong_chat', ''),
                                           ('wrong_digest', ''), ('registration_failure', ''),
                                           ('owed', 'hurry'), ('owed', 'finalize_now'),
                                           ('owed', 'panic'), ('owed', 'stop')])
def test_real_terminal_publication_and_receipt_converge(late, terminal_context, tmp_path, monkeypatch,
                                                       retry, order, receipt, control):
    """Exercise worker send -> file save -> task_done and its ordinary recovery."""
    from ouroboros import headless
    from supervisor import events, queue as task_queue, task_reaper, workers

    f = delivered(tmp_path, monkeypatch, retry=retry, receipt=receipt, copyback=False)
    ctx, worker, pushed = terminal_context
    ctx.DRIVE_ROOT = f.root
    task = {**f.task, '_attempt': 1, 'chat_id': 7, 'type': 'task'}
    ctx.RUNNING = {f.tid: {'task': task, 'attempt': 1, 'worker_id': 7}}
    worker.busy_task_id = f.tid
    sends, returned, jobs = [], queue.Queue(), queue.Queue()
    sender = _send_ctx(f.root, sends)
    ctx.send_with_budget = sender.send_with_budget
    ctx.event_queue = returned
    monkeypatch.setattr(task_queue, 'DRIVE_ROOT', f.root)
    monkeypatch.setattr(task_queue, 'RUNNING', ctx.RUNNING)
    monkeypatch.setattr(task_queue, '_ensure_reaper_started', lambda: None)
    monkeypatch.setattr(task_queue, '_reap_queue', jobs)
    monkeypatch.setattr(workers, 'RUNNING', ctx.RUNNING)
    monkeypatch.setattr(workers, 'WORKERS', ctx.WORKERS)
    monkeypatch.setattr(workers, 'DRIVE_ROOT', f.root)
    monkeypatch.setattr(workers, 'get_event_q', lambda: returned)
    done = {'type': 'task_done', 'task_id': f.tid, 'worker_id': 7, 'status': 'completed',
            '_files_prepared_attempt': 1}
    if order == 'checkpoint_first':
        # A terminal checkpoint can expose the debt before adopted child body
        # and retained source bytes are published. Receipt alone is insufficient.
        from ouroboros.task_results import write_task_result
        write_task_result(f.root, f.tid, 'completed', result=f.event['text'],
                          acceptance_debt=load_task_result(f.worker, f.tid)['acceptance_debt'])
        assert not headless.terminal_task_files_ready(f.root, task, load_task_result(f.root, f.tid))
    if order != 'canonical_first' and receipt == 'owed':
        events.dispatch_event(f.event, ctx)
        events.dispatch_event(f.event, ctx)  # buffered duplicate also precedes copyback
        assert sends == [(7, f.event['text'])]
        early = load_task_result(f.root, f.tid) or {}
        assert not late.calls and not early.get('review_operations') and not review_operation._LIVE
        assert bool(early.get('acceptance_debt')) is (order == 'checkpoint_first')
    if control:
        from ouroboros.owner_mailbox import write_owner_message
        if control == 'panic':
            (f.root / 'state' / 'panic_stop.flag').write_text('stop')
        elif control == 'stop':
            from ouroboros.cancel_intents import request_cancel
            request_cancel(f.root, f.tid, reason='owner_stop', allow_settled_target=True)
        else:
            write_owner_message(f.root, 'Owner control', task_id=f.tid, kind=control)
    if order == 'file_recovery':
        # Failed worker copyback is retried by the existing file owner; it must
        # never dispatch a reviewer from the serial reaper itself.
        with monkeypatch.context() as fault:
            fault.setattr(headless, 'write_task_result', lambda *_a, **_k: (_ for _ in ()).throw(OSError('disk full')))
            assert headless.prepare_terminal_task_files(f.root, task)['error']
        events.dispatch_event(done, ctx)
        assert jobs.qsize() == 1
        task_reaper._recover_terminal_files(jobs.get_nowait())
        assert not late.calls and not review_operation._LIVE
        events.dispatch_event(returned.get_nowait(), ctx)
    else:
        assert not headless.prepare_terminal_task_files(f.root, task)['error']
        events.dispatch_event(done, ctx)
    if order == 'canonical_first' and receipt == 'owed':
        assert not late.calls and not review_operation._LIVE
        events.dispatch_event(f.event, ctx)
    if receipt != 'owed' or control:
        until(lambda: not review_operation._LIVE)
        assert not late.calls and not load_task_result(f.root, f.accounting).get('task_acceptance_review_accounting')
        return
    until(lambda: panel(f))
    until(lambda: not review_operation._LIVE)
    assert worker.busy_task_id is None and f.tid not in ctx.RUNNING
    assert len(late.calls) == 3
    assert all(scope.root_task_id == f.accounting for scope, _ in late.calls)
    events.dispatch_event(done, ctx)
    events.dispatch_event(f.event, ctx)
    chat._DELIVERED_MESSAGE_IDS.clear()
    events.dispatch_event(f.event, ctx)
    assert len(late.calls) == 3 and sends == [(7, f.event['text'])]
    assert len(load_task_result(f.root, f.tid)['review_operations']) == 1
    assert len(load_task_result(f.root, f.accounting)['task_acceptance_review_accounting']['claims_by_binding']) == 1


def panel(f):
    return next((p for p in (load_task_result(f.root, f.tid).get('review_projection') or {}).get('panels', [])
                 if p.get('late_settlement')), None)


def postwork(f, monkeypatch):
    """Run the real existing post-task owner; replace its paid stage transports."""
    from ouroboros import agent_task_pipeline as pipeline
    from ouroboros.post_task_checkpoint import post_task_synthesis_in_flight
    observed = []
    for name in ('_record_task_facts', '_run_scratchpad_consolidation',
                 '_run_reflection', '_update_improvement_backlog'):
        monkeypatch.setattr(pipeline, name, lambda *_a, _name=name, **_k: observed.append(_name))
    monkeypatch.setattr('ouroboros.post_task_evolution.maybe_promote', lambda *_a, **_k: observed.append('promotion'))
    task = {**f.task, 'drive_root': str(f.root)}
    task.pop('_skip_post_task_synthesis', None)
    env = SimpleNamespace(drive_root=f.root, repo_dir=f.tmp_path)
    pipeline._run_post_task_processing_async(env, task, {}, {}, {}, f.root / 'logs', blocking=True)
    assert not post_task_synthesis_in_flight(f.root, f.tid)
    assert load_task_result(f.root, f.tid)['root_phase_checkpoint']['post_task_synthesis'] == 'completed'
    assert 'promotion' in observed


@pytest.mark.parametrize('retry', [False, True])
@pytest.mark.parametrize('replay', [False, True])
def test_receipt_after_entire_postwork_starts_once_through_sender_or_outbox(late, tmp_path, monkeypatch, retry, replay):
    from supervisor import queue as task_queue
    from supervisor import terminal_delivery
    f = delivered(tmp_path, monkeypatch, retry=retry, receipt='owed')
    monkeypatch.setattr(task_queue, 'DRIVE_ROOT', f.root)
    postwork(f, monkeypatch)
    before = copy.deepcopy(load_task_result(f.root, f.tid))
    sends, events = [], queue.Queue()
    if replay:
        monkeypatch.setattr(terminal_delivery, '_REPLAY_MIN_AGE_SEC', 0)
        assert terminal_delivery.replay_pending_deliveries(f.root, event_queue=events) == [f.event['delivery_id']]
        event = events.get_nowait()
    else:
        event = f.event
    ctx = _send_ctx(f.root, sends)
    ctx.event_queue = events
    chat._handle_send_message(event, ctx)
    until(lambda: panel(f))
    until(lambda: not review_operation._LIVE)
    assert len(late.calls) == 3 and sends == [(7, f.event['text'])]
    assert all(scope.task_id == f.tid and scope.root_task_id == f.accounting for scope, _ in late.calls)
    for field in ('result', 'status', 'review_status', 'outcome_axes', 'acceptance_debt'):
        assert load_task_result(f.root, f.tid).get(field) == before.get(field)
    chat._handle_send_message(event, ctx)  # in-memory duplicate
    chat._DELIVERED_MESSAGE_IDS.clear()
    chat._handle_send_message(event, ctx)  # cold durable duplicate
    owner = _caller(f)
    _request(f, owner, _source(owner))  # explicit concurrent identity is collection only
    assert len(late.calls) == 3 and len(load_task_result(f.root, f.tid)['review_projection']['panels']) == 1
    assert len(terminal_delivery.pending_deliveries(f.root)) == 1


@pytest.mark.parametrize('receipt', ['wrong_id', 'wrong_chat', 'wrong_digest', 'owed', 'registration_failure'])
def test_automatic_wrong_or_unproven_receipt_never_owns_or_sends(late, tmp_path, monkeypatch, receipt):
    f = delivered(tmp_path, monkeypatch, receipt=receipt, automatic=True)
    assert not late.calls and not load_task_result(f.root, f.tid).get('review_operations')


@pytest.mark.parametrize('automatic', [True, False])
def test_main_to_project_routing_is_host_proven_and_supplement_keeps_historical_room(late, tmp_path, monkeypatch, automatic):
    from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID
    from ouroboros.projects_registry import bind_task_to_project
    from ouroboros.artifacts import read_actor_source_bytes
    from supervisor.log_addressing import bound_project_chat_id
    from supervisor.terminal_delivery import pending_deliveries, terminal_answer_receipts
    f = delivered(tmp_path, monkeypatch, receipt='owed', chat_id=WEB_UI_CHAT_ID)
    pinned = copy.deepcopy(load_task_result(f.root, f.tid)['acceptance_debt'])
    binding = bind_task_to_project(f.root, f.tid, 'late-project', origin={'absent': 'post_hoc_unresolved'})
    monkeypatch.setattr(chat, '_bound_project_chat_id', bound_project_chat_id)
    sends = []
    sender = _send_ctx(f.root, sends)
    with monkeypatch.context() as mode:
        if not automatic:
            mode.setenv('OUROBOROS_TASK_REVIEW_MODE', 'off')
        chat._handle_send_message(f.event, sender)
    if not automatic:
        ctx = _caller(f)
        _request(f, ctx, _source(ctx))
    p = until(lambda: panel(f))
    until(lambda: not review_operation._LIVE)
    assert sends == [(binding['project_chat_id'], f.event['text'])]
    receipt = terminal_answer_receipts(f.root, f.tid)['delivered'][0]
    captured = json.loads(read_actor_source_bytes(f.root, f.tid, receipt['source_ref']))
    assert captured['routing'] == {'basis': 'terminal_sender_bound_project', 'intended_chat_id': WEB_UI_CHAT_ID,
                                   'routed_chat_id': binding['project_chat_id']}
    assert load_task_result(f.root, f.tid)['acceptance_debt'] == pinned
    assert p['late_settlement']['reviewed_revision'] == 'delivered' and len(late.calls) == 3
    # A later project binding cannot redirect this historical supplement.
    monkeypatch.setattr(chat, '_bound_project_chat_id', lambda *_a: 99999)
    notice = pending_deliveries(f.root)[0]
    chat._handle_send_message(notice, sender)
    assert sends[-1][0] == binding['project_chat_id']


@pytest.mark.parametrize('first', ['preparation', 'callback'])
@pytest.mark.parametrize('retry', [False, True])
def test_overlapping_historical_collections_keep_the_queued_notice_room(late, tmp_path, monkeypatch, first, retry):
    """Both real collectors can observe pending before either publishes its result."""
    from ouroboros import acceptance_late, acceptance_settlement as settlement, review_dispatch
    from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID
    from ouroboros.gateway.task_archive import serve_task_source
    from ouroboros.projects_registry import bind_task_to_project
    from supervisor.log_addressing import bound_project_chat_id
    from supervisor.terminal_delivery import pending_deliveries, terminal_answer_receipts

    monkeypatch.setattr('ouroboros.pricing._fetch_live_rows', lambda *_a, **_kw: {})
    f = delivered(tmp_path, monkeypatch, receipt='owed', chat_id=WEB_UI_CHAT_ID, retry=retry)
    before = copy.deepcopy(load_task_result(f.root, f.tid))
    binding = bind_task_to_project(f.root, f.tid, 'late-project', origin={'absent': 'post_hoc_unresolved'})
    monkeypatch.setattr(chat, '_bound_project_chat_id', bound_project_chat_id)
    transport = threading.Event()
    late.gates.append(transport)
    kinds = ('preparation', 'callback')
    entered, collect, collected, release, published = (
        {kind: threading.Event() for kind in kinds} for _ in range(5))
    real_collect = review_dispatch.collect_task_acceptance_run
    real_enqueue = settlement.enqueue_late_acceptance_settlement

    def caller():
        return 'preparation' if threading.current_thread().name.startswith('review-prepare-') else 'callback'

    def held_collection(*args, **kwargs):
        kind = caller()
        entered[kind].set()
        assert collect[kind].wait(10)
        result = real_collect(*args, **kwargs)
        collected[kind].set()
        assert release[kind].wait(10)
        return result

    def observed_enqueue(*args, **kwargs):
        result = real_enqueue(*args, **kwargs)
        published[caller()].set()
        return result

    monkeypatch.setattr(review_dispatch, 'collect_task_acceptance_run', held_collection)
    monkeypatch.setattr(settlement, 'enqueue_late_acceptance_settlement', observed_enqueue)
    sends = []
    sender = _send_ctx(f.root, sends)
    sender.event_queue = queue.Queue()
    try:
        chat._handle_send_message(f.event, sender)
        assert entered['preparation'].wait(10)
        transport.set()
        assert entered['callback'].wait(10)
        # Each real collection reads the pending snapshot and all settled actors.
        for kind in kinds:
            collect[kind].set()
            assert collected[kind].wait(10)
        release[first].set()
        assert published[first].wait(10)
        other = next(kind for kind in kinds if kind != first)
        release[other].set()
        assert published[other].wait(10)
    finally:
        transport.set()
        for gate in [*collect.values(), *release.values()]:
            gate.set()
        until(lambda: not review_operation._LIVE)

    p = panel(f)
    notice, = pending_deliveries(f.root)
    assert notice['system_type'] == 'acceptance_late_settlement'
    assert notice['progress_meta']['late_evidence']['source_ref'] != p['applied_source_ref']
    # The notice's record link still opens the exact record it was sent with (#1369).
    sent = notice['progress_meta']['late_evidence']['source_ref']
    served = serve_task_source(f.root, [], load_task_result(f.root, f.tid), f.tid, sent['path'].rsplit('/', 1)[1],
                               sent['path'])
    assert served.status_code == 200 and hashlib.sha256(served.body).hexdigest() == sent['sha256']
    assert notice['text'] == p['late_settlement']['note']
    receipt, = terminal_answer_receipts(f.root, f.tid)['delivered']
    assert p['late_settlement']['historical_delivery'] == {
        key: receipt[key] for key in ('task_id', 'delivery_id', 'chat_id', 'text_sha256', 'source_ref')}
    for key in ('result', 'status', 'review_status', 'outcome_axes', 'acceptance_debt'):
        assert load_task_result(f.root, f.tid).get(key) == before.get(key), key
    assert len(late.calls) == 3
    assert all(scope.task_id == f.tid and scope.root_task_id == f.accounting and scope.root_limit_usd == 4
               for scope, _ in late.calls)
    assert len(load_task_result(f.root, f.accounting)['task_acceptance_review_accounting']['claims_by_binding']) == 1
    # A caller-supplied stale pointer alone cannot claim the historical route.
    for key, value in (('task_id', 'another-task'), ('delivery_id', 'another-delivery'),
                       ('chat_id', 123), ('text', 'another note'), ('system_type', 'another-type')):
        assert acceptance_late.supplement_chat(f.root, {**notice, key: value}) is None, key
    altered = copy.deepcopy(notice)
    altered['progress_meta']['late_evidence']['source_ref']['sha256'] = '0' * 64
    assert acceptance_late.supplement_chat(f.root, altered) is None
    with monkeypatch.context() as absent:
        absent.setattr('supervisor.terminal_delivery.pending_deliveries', lambda _root: [])
        assert acceptance_late.supplement_chat(f.root, notice) is None
    monkeypatch.setattr(chat, '_bound_project_chat_id', lambda *_a: 99999)
    chat._handle_send_message(notice, sender)
    assert sends[-1][0] == binding['project_chat_id']
    assert sends == [(binding['project_chat_id'], f.event['text']), (binding['project_chat_id'], notice['text'])]
    chat._DELIVERED_MESSAGE_IDS.clear()
    chat._handle_send_message(notice, sender)
    for event in list(sender.event_queue.queue):
        if event.get('system_type') == 'acceptance_late_settlement':
            chat._handle_send_message(event, sender)
    chat._handle_send_message(f.event, sender)
    assert len(sends) == 2 and len(late.calls) == 3 and not pending_deliveries(f.root)


@pytest.mark.parametrize('control', ['hurry', 'finalize_now', 'panic'])
def test_preparation_owns_stop_before_sources_and_wrap_cancels_without_purchase(late, tmp_path, monkeypatch, control):
    from ouroboros import review_source_closure
    from ouroboros.owner_mailbox import write_owner_message
    from supervisor import queue as task_queue
    from supervisor.queue_transitions import task_has_live_ownership
    f = delivered(tmp_path, monkeypatch, receipt='owed')
    monkeypatch.setattr(task_queue, 'DRIVE_ROOT', f.root)
    entered, release = threading.Event(), threading.Event()
    original = review_source_closure.retain_review_request_sources
    def blocked(*a, **kw):
        entered.set()
        assert release.wait(10)
        return original(*a, **kw)
    monkeypatch.setattr(review_source_closure, 'retain_review_request_sources', blocked)
    try:
        chat._handle_send_message(f.event, _send_ctx(f.root, []))
        assert entered.wait(5)  # sender returned while the owned preparation is blocked
        row = load_task_result(f.root, f.tid)
        pointer = next(iter(row['review_operations'].values()))
        assert pointer['state'] == 'preparing' and pointer['intent_ref'] and not pointer.get('source_ref')
        assert task_has_live_ownership(f.tid) and not late.calls
        if control == 'panic':
            (f.root / 'state' / 'panic_stop.flag').write_text('stop')
        else:
            assert write_owner_message(f.root, 'Owner control', task_id=f.tid, kind=control)
        operation = next(iter(review_operation._LIVE.values()))
        until(lambda: operation.control() == 'cancelled')
    finally:
        release.set()
    until(lambda: not review_operation._LIVE)
    assert not late.calls
    chat._handle_send_message(f.event, _send_ctx(f.root, []))
    assert not review_operation._LIVE and not late.calls
    if control != 'panic':
        ctx = _caller(f)
        result = _request(f, ctx, _source(ctx))  # later explicit source may spend permitted original funds
        assert result['status'] in {'pending', 'announced', 'published', 'settled'}, result
        until(lambda: len(late.calls) == 3 or not review_operation._LIVE)
        assert len(late.calls) == 3, {'result': result, 'row': load_task_result(f.root, f.tid)}


def test_abandoned_intent_is_unknown_and_maintenance_never_purchases(late, tmp_path, monkeypatch):
    f = delivered(tmp_path, monkeypatch, receipt='owed')
    # A crash after durable identity but before worker start leaves the same
    # visible intent; the existing maintenance owner may classify, never launch.
    with monkeypatch.context() as crash:
        crash.setattr(threading.Thread, 'start', lambda _self: None)
        chat._handle_send_message(f.event, _send_ctx(f.root, []))
    operation = next(iter(review_operation._LIVE.values()))
    review_operation._LIVE.pop(operation.owner_id)
    operation.wait.close()
    monkeypatch.setattr(review_operation, 'controller_state', lambda _identity: 'dead')
    report = review_operation.recover_orphaned_acceptance_operations(f.root)
    assert report['pending'] and not late.calls
    assert load_task_result(f.root, f.tid)['review_operations'][operation.owner_id]['state'] == 'preparation_unknown'
    chat._handle_send_message(f.event, _send_ctx(f.root, []))
    ctx = _caller(f)
    result = _request(f, ctx, _source(ctx))
    assert result['status'] == 'unknown' and not late.calls


def test_auto_preparation_and_explicit_entry_share_one_retained_owner(late, tmp_path, monkeypatch):
    from ouroboros import review_source_closure
    f = delivered(tmp_path, monkeypatch, receipt='owed', retry=True)
    entered, release = threading.Event(), threading.Event()
    original = review_source_closure.retain_review_request_sources
    def blocked(*a, **kw):
        entered.set()
        assert release.wait(10)
        return original(*a, **kw)
    monkeypatch.setattr(review_source_closure, 'retain_review_request_sources', blocked)
    try:
        chat._handle_send_message(f.event, _send_ctx(f.root, []))
        assert entered.wait(5)
        ctx = _caller(f)
        result = _request(f, ctx, _source(ctx))
        assert result['status'] == 'preparing' and not late.calls
        chat._handle_send_message(f.event, _send_ctx(f.root, []))
        assert len(load_task_result(f.root, f.tid)['review_operations']) == 1
    finally:
        release.set()
    until(lambda: panel(f))
    until(lambda: not review_operation._LIVE)
    assert len(late.calls) == 3
    assert len(load_task_result(f.root, f.accounting)['task_acceptance_review_accounting']['claims_by_binding']) == 1


def test_durable_hurry_survives_postwork_mailbox_cleanup(late, tmp_path, monkeypatch):
    from ouroboros.owner_hurry import record_requested, reconcile_terminal
    from ouroboros.owner_mailbox import cleanup_task_mailbox, write_owner_message
    f = delivered(tmp_path, monkeypatch, receipt='owed')
    record_requested(f.root, f.tid, request_id='hurry-before-send', attempt=1)
    write_owner_message(f.root, 'owner_hurry', task_id=f.tid, kind='hurry')
    reconcile_terminal(f.root, f.tid)
    postwork(f, monkeypatch)
    cleanup_task_mailbox(f.root, f.tid)
    chat._handle_send_message(f.event, _send_ctx(f.root, []))
    until(lambda: not review_operation._LIVE)
    pointer = next(iter(load_task_result(f.root, f.tid)['review_operations'].values()))
    assert pointer['preparation_outcome']['reason'].startswith('owner_finalization') and not late.calls
    ctx = _caller(f)
    _request(f, ctx, _source(ctx))
    until(lambda: len(late.calls) == 3)


def test_receipt_before_durable_intent_failure_stays_owed_for_later_explicit_owner(late, tmp_path, monkeypatch):
    from supervisor.terminal_delivery import terminal_answer_receipts
    f = delivered(tmp_path, monkeypatch, receipt='owed')
    with monkeypatch.context() as fault:
        def unavailable(*_a, **_kw):
            raise OSError('intent pointer store unavailable')
        fault.setattr(review_operation, '_update_operations', unavailable)
        chat._handle_send_message(f.event, _send_ctx(f.root, []))
    assert terminal_answer_receipts(f.root, f.tid)['state'] == 'delivered'
    assert not load_task_result(f.root, f.tid).get('review_operations')
    assert not late.calls and not review_operation._LIVE
    ctx = _caller(f)
    _request(f, ctx, _source(ctx))
    until(lambda: len(late.calls) == 3)


@pytest.mark.serial
@pytest.mark.parametrize('replay', [False, True])
@pytest.mark.parametrize('recorded', ['ready', 'paid', 'unknown', 'settled'])
def test_duplicate_sender_leaves_held_collection_to_maintenance(late, tmp_path, monkeypatch, replay, recorded):
    """A real duplicate final cannot put collection ahead of the next send/Panic."""
    from ouroboros import acceptance_settlement, review_dispatch
    from supervisor import events, terminal_delivery
    from tests.test_server_control_panic_daemon import _run_panic

    f = delivered(tmp_path, monkeypatch, receipt='owed')
    outbox = queue.Queue()
    monkeypatch.setattr(terminal_delivery, '_REPLAY_MIN_AGE_SEC', 0)
    assert terminal_delivery.replay_pending_deliveries(f.root, event_queue=outbox) == [f.event['delivery_id']]
    replay_event = outbox.get_nowait()  # queued before a competing delivery wins
    sends, ctx = [], _caller(f)
    sender = _send_ctx(f.root, sends)
    with monkeypatch.context() as mode:
        mode.setenv('OUROBOROS_TASK_REVIEW_MODE', 'off')
        events.dispatch_event(f.event, sender)
    with monkeypatch.context() as fault:
        if recorded == 'ready':
            original = review_operation._write_operation_pointer
            def after_pointer(*a, **kw):
                original(*a, **kw)
                raise OSError('controller ended after ready pointer, before claim')
            fault.setattr(review_operation, '_write_operation_pointer', after_pointer)
        elif recorded == 'paid':
            original = review_dispatch.invoke_review_paid_stamp
            def after_claim(stamp):
                original(stamp)
                if callable(stamp) and getattr(stamp, 'fail_closed', False):
                    raise review_dispatch.TaskAcceptanceDispatchUnavailable('controller ended after claim')
            fault.setattr(review_dispatch, 'invoke_review_paid_stamp', after_claim)
        late.config.fail = recorded == 'unknown'
        result = _request(f, ctx, _source(ctx))
        until(lambda: not review_operation._LIVE)
    row = load_task_result(f.root, f.tid)
    pointers = copy.deepcopy(row['review_operations'])
    claims = copy.deepcopy(load_task_result(f.root, f.accounting).get('task_acceptance_review_accounting'))
    assert all(p.get('source_ref') for p in pointers.values())
    assert bool(claims) is (recorded != 'ready'), {'result': result, 'pointers': pointers, 'claims': claims}
    calls_before = len(late.calls)
    review_operation._update_operations(f.root, f.tid, lambda rows: {
        owner: {**p, 'state': 'unpublished'} for owner, p in rows.items()})
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    collector_threads, errors, reports, panic = [], [], [], []
    original_trace = acceptance_settlement.canonical_acceptance_trace
    def held_collector(*a, **kw):
        collector_threads.append(threading.current_thread().name)
        entered.set()
        assert release.wait(10)
        return original_trace(*a, **kw)
    monkeypatch.setattr(acceptance_settlement, 'canonical_acceptance_trace', held_collector)
    def maintenance():
        reports.append(review_operation.recover_orphaned_acceptance_operations(f.root))
    def drain():
        try:
            events.dispatch_event(replay_event if replay else f.event, sender)
            events.dispatch_event({'type': 'send_message', 'chat_id': 7, 'text': 'Other delivery'}, sender)
            _run_panic(monkeypatch, f.root, daemon_stop=lambda: True,
                       panic_request=lambda **kw: panic.append(kw) or [])
        except BaseException as exc:
            errors.append(exc)
        finally:
            returned.set()
    collector = threading.Thread(target=maintenance, name='ordinary-maintenance')
    delivery = threading.Thread(target=drain, name='sender-event-drain')
    collector.start()
    try:
        assert entered.wait(5)
        if replay:
            chat._DELIVERED_MESSAGE_IDS.clear()  # durable dedupe after reconnect
        delivery.start()
        assert returned.wait(2), 'duplicate final blocked the event drain on paid collection'
        assert not errors and panic == [{'request_only': True}] and not release.is_set()
        assert sends == [(7, f.event['text']), (7, 'Other delivery')]
        assert collector_threads == ['ordinary-maintenance']
        assert (f.root / 'state' / 'panic_stop.flag').read_text() == 'panic'
    finally:
        release.set()
        collector.join(10)
        if delivery.ident is not None:
            delivery.join(10)
    assert not collector.is_alive() and not delivery.is_alive() and not errors
    assert reports and not reports[0]['errors'] and not review_operation._LIVE
    assert len(late.calls) == calls_before
    assert load_task_result(f.root, f.accounting).get('task_acceptance_review_accounting') == claims
    assert set(load_task_result(f.root, f.tid)['review_operations']) == set(pointers)


@pytest.mark.parametrize('order', ['receipt_first', 'canonical_first'])
@pytest.mark.parametrize('retry', [False, True])
@pytest.mark.parametrize('source_state', ['retained', 'missing', 'foreign'])
def test_deferred_history_closes_sources_before_automatic_freeze(
        late, terminal_context, tmp_path, monkeypatch, order, retry, source_state):
    from pathlib import Path
    from ouroboros import artifacts, headless, observability, review_source_closure
    from ouroboros.review_native_episode import inspection_registry
    from supervisor import events, queue as task_queue
    from supervisor.terminal_delivery import pending_deliveries

    def capture(f):
        f.log_ref = observability.write_blob(f.worker, 'EXACT HOST LOG', kind='txt')
        f.unowned = artifacts.store_actor_source_bytes(f.worker, f.tid, category='tool_results',
            source_id='quoted-only', data=b'NOT A HOST EDGE', extension='txt')
        call = {'tool': 'service_logs', 'result': json.dumps({
            'full_log_ref': f.log_ref, 'tail': {'source_ref': f.unowned}}, separators=(',', ':'))}
        source = tmp_path / 'unbound-drive' if source_state == 'foreign' else f.worker
        f.trajectory = artifacts.persist_tool_trajectory_source(source, f.tid, [call])
        f.raw = artifacts.read_actor_source_bytes(source, f.tid, f.trajectory)
        if source_state == 'missing':
            (artifacts.task_artifact_dir_path(source, f.tid) / f.trajectory['path']).unlink()
        return {'tool_trajectory_source_ref': f.trajectory}

    f = delivered(tmp_path, monkeypatch, retry=retry, receipt='owed', copyback=False, capture=capture)
    ctx, _worker, _pushed = terminal_context
    ctx.DRIVE_ROOT, ctx.RUNNING = f.root, {}
    ctx.event_queue = queue.Queue()
    sends = []
    ctx.send_with_budget = _send_ctx(f.root, sends).send_with_budget
    monkeypatch.setattr(task_queue, 'DRIVE_ROOT', f.root)
    monkeypatch.setattr(task_queue, 'RUNNING', {})
    entered, release, frozen = threading.Event(), threading.Event(), []
    retain = review_source_closure.retain_review_request_sources
    write_pointer = review_operation._write_operation_pointer

    def held_sources(*args, **kwargs):
        entered.set()
        assert release.wait(10), 'sender blocked on source preparation'
        return retain(*args, **kwargs)

    def observe_freeze(operation, request, *args, **kwargs):
        closure = request.policy['review_source_closure']
        read_root = Path(closure['read_root'])
        named = next(row for row in closure['sources'] if row['name'] == 'evidence.tool_trajectory_source_ref')
        registry, _reader, _ = inspection_registry(str(tmp_path), read_root, f.tid)
        read = registry.execute_result('read_file', named['source_ref']['read']['arguments'])
        assert read.status == 'ok'  # log is a separately named blob
        assert artifacts.read_actor_source_bytes(read_root, f.tid, named['source_ref']) == f.raw
        assert observability.read_blob_ref(read_root, f.log_ref, expected_kind='txt') == 'EXACT HOST LOG'
        assert not (artifacts.task_artifact_dir_path(read_root, f.tid) / f.unowned['path']).exists()
        frozen.append(copy.deepcopy(request))
        return write_pointer(operation, request, *args, **kwargs)

    monkeypatch.setattr(review_source_closure, 'retain_review_request_sources', held_sources)
    monkeypatch.setattr(review_operation, '_write_operation_pointer', observe_freeze)
    try:
        if order == 'receipt_first':
            events.dispatch_event(f.event, ctx)
            assert sends == [(7, f.event['text'])] and not entered.is_set()
        outcome = headless.prepare_terminal_task_files(f.root, f.task)
        assert not outcome['error']
        row = load_task_result(f.root, f.tid)
        assert headless.terminal_task_files_ready(f.root, f.task, row)
        assert row['child_ref_promotion']['pending_refs'] and f.worker.exists()
        assert not (artifacts.task_artifact_dir_path(f.root, f.tid) / f.trajectory['path']).exists()
        events.dispatch_event({'type': 'task_done', 'task_id': f.tid, 'status': 'completed',
                               '_files_prepared_attempt': 1}, ctx)
        if order == 'canonical_first':
            assert not entered.is_set()
            events.dispatch_event(f.event, ctx)
        assert sends == [(7, f.event['text'])]
        assert entered.wait(10) and not late.calls and not frozen
    finally:
        release.set()
    until(lambda: not review_operation._LIVE)
    row = load_task_result(f.root, f.tid)
    if source_state != 'retained':
        assert not late.calls and not frozen and not panel(f)
        assert all(p['state'] == 'preparation_refused' and
                   'historical_review_unavailable' in p['preparation_outcome']['reason']
                   for p in row['review_operations'].values())
        assert not load_task_result(f.root, f.accounting).get('task_acceptance_review_accounting')
        assert not pending_deliveries(f.root)
        return
    assert frozen and len(late.calls) == 3 and panel(f)
    assert len(pending_deliveries(f.root)) == 1
    assert len(load_task_result(f.root, f.accounting)['task_acceptance_review_accounting']['claims_by_binding']) == 1
    headless.retry_child_task_refs(f.root, f.worker, f.tid)
    assert headless.remove_subagent_task_drive(f.root, f.tid, live=lambda _task: False)
    for request in frozen:
        read_root = Path(request.policy['review_source_closure']['read_root'])
        assert artifacts.read_actor_source_bytes(read_root, f.tid, f.trajectory) == f.raw
        assert observability.read_blob_ref(read_root, f.log_ref, expected_kind='txt') == 'EXACT HOST LOG'
    events.dispatch_event(f.event, ctx)
    assert len(late.calls) == 3  # a replay cannot lose the automatic handoff or buy again
