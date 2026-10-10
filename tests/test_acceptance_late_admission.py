"""Late NEW-panel admission through sender and registered owner-tool consumers."""
import copy
import json
import multiprocessing
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import review_operation
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.utils import utc_now_iso
from supervisor import events_chat_delivery as chat
from tests.test_acceptance_history import _caller, _request, _source
from tests.test_acceptance_late_consumers import delivered, late as late
from tests.test_review_operation_collection import _send_ctx, fresh_sends as fresh_sends
from tests.test_review_operation_lifetime import until
from tests._usage_store_testing import ledger_rows


def _unpaid(f, late):
    until(lambda: not review_operation._LIVE)
    row = load_task_result(f.root, f.tid)
    assert not late.calls and row['acceptance_debt'] and not row.get('review_projection', {}).get('panels')
    assert not load_task_result(f.root, f.accounting).get('task_acceptance_review_accounting')
    assert not ledger_rows(f.root)
    return row


@pytest.mark.parametrize('automatic,seconds,floor,buy', [
    (True, 1, 200, False), (True, 199, 200, False), (True, 200, 200, False),
    (True, 201, 200, True), (True, 600, 600, False), (True, 601, 600, True),
    (True, 3600, 200, True), (True, None, 200, True), (False, 199, 200, True),
])
def test_original_calendar_window_uses_review_floor_without_author_reserve(
        late, tmp_path, monkeypatch, automatic, seconds, floor, buy):
    now = datetime.now(timezone.utc)
    deadline = (now + timedelta(seconds=seconds)).isoformat() if seconds is not None else ''
    monkeypatch.setenv('OUROBOROS_ACCEPTANCE_REVIEW_EST_SEC', str(floor))
    monkeypatch.setattr('ouroboros.deadline_utils.utc_now', lambda: now)
    f = delivered(tmp_path, monkeypatch, receipt='owed', deadline=deadline)
    if automatic:
        chat._handle_send_message(f.event, _send_ctx(f.root, []))
    else:
        with monkeypatch.context() as mode:
            mode.setenv('OUROBOROS_TASK_REVIEW_MODE', 'off')
            chat._handle_send_message(f.event, _send_ctx(f.root, []))
        ctx = _caller(f)
        _request(f, ctx, _source(ctx))
    if buy:
        until(lambda: len(late.calls) == 3)
        until(lambda: not review_operation._LIVE)
    else:
        row = _unpaid(f, late)
        outcome = next(iter(row['review_operations'].values()))['preparation_outcome']
        assert outcome['reason'] == 'review_skipped_deadline_reserve'


def test_automatic_rechecks_window_after_original_writer_drains(late, tmp_path, monkeypatch):
    from ouroboros import acceptance_late, review_source_closure
    from supervisor import queue as task_queue
    now = datetime.now(timezone.utc)
    clock = [now]
    monkeypatch.setattr('ouroboros.deadline_utils.utc_now', lambda: clock[0])
    f = delivered(tmp_path, monkeypatch, receipt='owed', deadline=(now + timedelta(seconds=201)).isoformat())
    monkeypatch.setattr(task_queue, 'RUNNING', {f.tid: {'task': f.task}})
    entered, release = threading.Event(), threading.Event()
    writer_live = acceptance_late._historical_writer_live
    observations = []
    preparations = []
    retain_sources = review_source_closure.retain_review_request_sources

    def prepare_sources(*args, **kwargs):
        preparations.append(True)
        return retain_sources(*args, **kwargs)

    def drain(*args, **kwargs):
        live = writer_live(*args, **kwargs)
        observations.append(live)
        if live:
            entered.set()
            assert release.wait(5)
            live = writer_live(*args, **kwargs)  # Observe the real writer after its drain.
            observations.append(live)
        return live

    monkeypatch.setattr(acceptance_late, '_historical_writer_live', drain)
    monkeypatch.setattr(review_source_closure, 'retain_review_request_sources', prepare_sources)
    try:
        chat._handle_send_message(f.event, _send_ctx(f.root, []))
        assert entered.wait(5) and not late.calls
        clock[0] += timedelta(seconds=2)
        task_queue.RUNNING.clear()
    finally:
        task_queue.RUNNING.clear()
        release.set()
    row = _unpaid(f, late)
    assert observations == [True, False]
    assert not preparations, "expired post-drain window must refuse before preparing reviewer sources"
    pointer = next(iter(row['review_operations'].values()))
    assert pointer['preparation_outcome']['reason'] == 'review_skipped_deadline_reserve'
    assert not pointer.get('source_ref')  # no source preparation after the drain


@pytest.mark.parametrize('automatic', [False, True])
def test_mutable_result_cannot_extend_original_calendar(late, tmp_path, monkeypatch, automatic):
    now = datetime.now(timezone.utc)
    monkeypatch.setattr('ouroboros.deadline_utils.utc_now', lambda: now)
    f = delivered(tmp_path, monkeypatch, deadline=(now + timedelta(seconds=200 if automatic else -1)).isoformat())
    write_task_result(f.root, f.tid, 'completed', task_contract={})
    ctx = _caller(f)
    if automatic:
        from ouroboros.acceptance_late import run_historical_acceptance
        debt = load_task_result(f.root, f.tid)['acceptance_debt']
        run_historical_acceptance(ctx, task_id=f.tid, debt_id=debt['debt_id'], automatic=True)
        row = _unpaid(f, late)
        result = next(iter(row['review_operations'].values()))['preparation_outcome']
    else:
        result = _request(f, ctx, _source(ctx))
        _unpaid(f, late)
    assert result['reason'] == ('review_skipped_deadline_reserve' if automatic else 'owner_deadline')


@pytest.mark.parametrize('unknown', [False, True])
def test_existing_paid_or_unknown_panel_collects_after_calendar_closes(late, tmp_path, monkeypatch, unknown):
    from ouroboros.acceptance_late import run_historical_acceptance
    now = datetime.now(timezone.utc)
    f = delivered(tmp_path, monkeypatch, deadline=(now + timedelta(seconds=3600)).isoformat())
    ctx = _caller(f)
    late.config.fail = unknown
    _request(f, ctx, _source(ctx))
    until(lambda: not review_operation._LIVE)
    assert len(late.calls) == 3
    before = ledger_rows(f.root)
    monkeypatch.setattr('ouroboros.deadline_utils.utc_now', lambda: now + timedelta(seconds=3601))
    debt = load_task_result(f.root, f.tid)['acceptance_debt']
    automatic = run_historical_acceptance(ctx, task_id=f.tid, debt_id=debt['debt_id'], automatic=True)
    assert automatic['reason'] == 'existing_operation_collection_owned'
    explicit = _request(f, ctx, _source(ctx))
    assert explicit['reason'] == 'existing_paid_operation'
    assert len(late.calls) == 3 and ledger_rows(f.root) == before


@pytest.mark.parametrize('venue,evidence,buy', [
    ('pooled', 'live', False), ('pooled', 'missing', False), ('pooled', 'stale', False),
    ('pooled', 'invalid', False), ('pooled', 'gone', True),
    ('direct', 'live', False), ('direct', 'postwork', False), ('direct', 'missing', False),
    ('direct', 'stale', False), ('direct', 'incomplete', False),
    ('direct', 'invalid', False), ('direct', 'phase_unknown', False), ('direct', 'gone', True),
])
def test_worker_public_adapter_uses_venue_authority(late, tmp_path, monkeypatch, venue, evidence, buy):
    """Worker maps are empty even while the supervisor's original writer lives."""
    import queue
    from supervisor import queue as task_queue, workers
    f = delivered(tmp_path, monkeypatch)
    before = copy.deepcopy(load_task_result(f.root, f.tid)['acceptance_debt'])
    monkeypatch.setenv('OUROBOROS_IN_WORKER', '1')
    monkeypatch.setattr(task_queue, 'INITIALIZED', False)
    monkeypatch.setattr(task_queue, 'RUNNING', {})
    monkeypatch.setattr(workers, 'WORKERS', {})
    write_task_result(f.root, f.tid, 'completed', _is_direct_chat=venue == 'direct',
        root_phase_checkpoint={'post_task_synthesis': 'running' if evidence == 'postwork'
                               else 'unknown' if evidence == 'phase_unknown' else 'completed'})
    snapshot = {'ts': utc_now_iso(), 'running': [], 'pending': []}
    fragment = {'ts': utc_now_iso(), 'roots': [], 'incomplete': False}
    target = snapshot if venue == 'pooled' else fragment
    if evidence == 'live':
        target['running' if venue == 'pooled' else 'roots'] = [{'task_id': f.tid}]
    elif evidence == 'stale':
        target['ts'] = '2000-01-01T00:00:00Z'
    elif evidence == 'incomplete':
        target['incomplete'] = True
    for name, payload in [('queue_snapshot', snapshot), ('direct_roots', fragment)]:
        selected = (venue == 'pooled') == (name == 'queue_snapshot')
        if selected and evidence == 'missing':
            continue
        (f.root / 'state' / f'{name}.json').write_text(
            '{' if selected and evidence == 'invalid' else json.dumps(payload))
    ctx = _caller(f)
    ctx.event_queue = queue.Queue()  # This consumer owns a live notification sink.
    source = _source(ctx)
    # A split worker reads the canonical authority, never its private data copy.
    ctx.drive_root = f.worker
    result = _request(f, ctx, source)
    if buy:
        until(lambda: len(late.calls) == 3)
        until(lambda: not review_operation._LIVE)
        assert result['status'] in {'pending', 'announced', 'published', 'settled'}, result
        notices = [event for event in list(ctx.event_queue.queue)
                   if event.get('system_type') == 'acceptance_late_settlement']
        assert len(notices) == 1 and notices[0]['task_id'] == f.tid
    else:
        reason = 'control_authority_unavailable' if venue == 'pooled' and evidence == 'invalid' else 'historical_writer_still_live'
        assert result['reason'] == reason, result
        _unpaid(f, late)
    assert load_task_result(f.root, f.tid)['acceptance_debt'] == before


def _spawned_request(root, worker, workspace, tid, source, results):
    """Fresh process, real registered tool/operation; only the paid transport is synthetic."""
    from supervisor import queue as task_queue, workers
    os.environ['OUROBOROS_IN_WORKER'] = '1'
    f = SimpleNamespace(root=Path(root), worker=Path(worker), tmp_path=Path(workspace), tid=tid)
    with pytest.MonkeyPatch.context() as patch:
        fixture = late.__wrapped__(f.tmp_path, patch, None)
        transport = next(fixture)
        try:
            ctx = _caller(f)
            ctx.drive_root = f.worker
            result = _request(f, ctx, source)
            until(lambda: not review_operation._LIVE)
            results.put({'result': result, 'calls': len(transport.calls),
                         'initialized': task_queue.INITIALIZED,
                         'maps_empty': not task_queue.RUNNING and not workers.WORKERS})
        finally:
            next(fixture, None)


@pytest.mark.serial
@pytest.mark.parametrize('venue', ['pooled', 'direct_postwork'])
def test_spawned_worker_public_request_waits_for_real_original_owner(late, tmp_path, monkeypatch, venue):
    from ouroboros import agent_task_pipeline as pipeline
    from supervisor import queue as task_queue
    from supervisor.direct_roots import publish_direct_roots
    from supervisor.queue_snapshot import persist_queue_snapshot
    f = delivered(tmp_path, monkeypatch)
    source = _source(_caller(f))
    entered, release = threading.Event(), threading.Event()
    postwork = None
    if venue == 'pooled':
        monkeypatch.setattr(task_queue, 'QUEUE_SNAPSHOT_PATH', f.root / 'state' / 'queue_snapshot.json')
        monkeypatch.setattr(task_queue, 'RUNNING', {f.tid: {'task': f.task, 'worker_id': 4}})
        assert persist_queue_snapshot('synthetic original writer still owns its slot')
    else:
        write_task_result(f.root, f.tid, 'completed', _is_direct_chat=True)
        publish_direct_roots(f.root)  # direct turn ended; its postwork has not
        def hold(*_a, **_kw):
            entered.set()
            assert release.wait(20)
        monkeypatch.setattr(pipeline, '_record_task_facts', hold)
        for name in ('_run_scratchpad_consolidation', '_run_reflection', '_update_improvement_backlog'):
            monkeypatch.setattr(pipeline, name, lambda *_a, **_k: None)
        monkeypatch.setattr('ouroboros.post_task_evolution.maybe_promote', lambda *_a, **_k: None)
        task = {**f.task, 'drive_root': str(f.root), '_is_direct_chat': True}
        task.pop('_skip_post_task_synthesis')
        postwork = threading.Thread(target=lambda: pipeline._run_post_task_processing_async(
            SimpleNamespace(drive_root=f.root, repo_dir=tmp_path), task, {}, {}, {}, f.root / 'logs', blocking=True))
        postwork.start()
        assert entered.wait(5)
    mp = multiprocessing.get_context('spawn')
    try:
        for live in (True, False):
            results = mp.Queue()
            process = mp.Process(target=_spawned_request, args=(str(f.root), str(f.worker), str(tmp_path), f.tid, source, results))
            try:
                process.start()
                process.join(20)
                assert process.exitcode == 0
                result = results.get(timeout=2)
                assert not result['initialized'] and result['maps_empty']
                if live:
                    assert result['result']['reason'] == 'historical_writer_still_live', result
                    assert result['calls'] == 0
                else:
                    assert result['calls'] == 3, result  # own preparation cannot deadlock the worker
            finally:
                if process.is_alive():
                    process.terminate()
                    process.join(5)
                results.close()
                results.join_thread()
            if venue == 'pooled':
                task_queue.RUNNING.clear()
                assert persist_queue_snapshot('synthetic original writer drained')
            else:
                release.set()
                postwork.join(5)
                assert not postwork.is_alive()
                publish_direct_roots(f.root)
    finally:
        release.set()
        if postwork is not None:
            postwork.join(5)
