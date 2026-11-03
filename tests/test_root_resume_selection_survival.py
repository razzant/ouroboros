"""UNRUN: R2 producer/consumer regressions; no provider or real worker process.

Carrier removal is the injected fault. Grant creation and selection use their
real owners. Warm parking is a fixture input, as in test_saved_root_resume_grant.
"""
import copy
import time
import uuid
from types import SimpleNamespace

import pytest

from ouroboros import budget_pause as bp, cancel_intents, owner_pause
from ouroboros import working_checkpoint as wc
from ouroboros.task_results import write_task_result
from ouroboros.task_store import patch_document
from supervisor import budget_resume as br, continuation_admission
from supervisor import events_budget as eb
from supervisor.restart_retention import recover_saved_running
from tests._budget_pause_exact_helpers import (
    _fast_hold, _loop_ctx, _mock_pause_observation, _parked,
)
from tests.test_budget_pause_holds import _idle_worker
from tests.test_budget_pause_lineage_restore import _consume, _restart
from tests.test_saved_root_resume_grant import _saved_tree

pytestmark = pytest.mark.serial
ROOT, CHILD, OTHER = 'saved-root', 'first', 'second'


def _row(q, task_id):
    return next(row for row in q.PENDING if row['id'] == task_id)


def _save_current(root, workers):
    meta = workers.RUNNING[ROOT]
    ctx, limit = _loop_ctx(root, ROOT, attempt=meta['attempt'])
    wc.save_round(limit, 'post_batch')


def _exact_root(root, monkeypatch, q, workers):
    meta = workers.RUNNING[ROOT]
    ctx, limit = _loop_ctx(root, ROOT, attempt=meta['attempt'])
    _fast_hold(monkeypatch, bp)
    _mock_pause_observation(monkeypatch, bp, [])
    with pytest.raises(bp.BudgetPauseRequested) as paused:
        bp.request_pause(limit, rail=bp.RAIL_GLOBAL_EXHAUSTED,
                         scope='global', reason_text='test budget boundary')
    bp.end_dispatch_fence(ROOT)
    sup = SimpleNamespace(
        DRIVE_ROOT=root, RUNNING=workers.RUNNING, PENDING=workers.PENDING,
        WORKERS=workers.WORKERS, sort_pending=q.sort_pending,
        persist_queue_snapshot=q.persist_queue_snapshot, bridge=None)
    eb.install_exact_budget_pause(
        sup, ROOT, bp.exact_pause_marker(paused.value.pause)['checkpoint'])
    result = q.resume_budget_paused_task(ROOT)
    assert result['ok'] and result['exact_continuation'], result
    workers.WORKERS[0].busy_task_id = None
    workers.assign_tasks()
    handoff = workers.RUNNING[ROOT]['task']['_budget_pause_resume']
    assert handoff['grant_id'] == result['grant_id']
    _consume(monkeypatch, ctx, limit, handoff)
    stored = bp.budget_pause_row(root, ROOT)
    assert stored['state'] == bp.STATE_RESUMED and stored['grant']['consumed_at']
    assert stored['grant']['grant_id'] == result['grant_id']
    return result['grant_id']


def _warm_root(root, monkeypatch, q, workers):
    fence, _ = owner_pause.install_fence(root, ROOT, request_id=uuid.uuid4().hex)
    eb._set_root_budget_pause_locked(ROOT, {**fence, 'cause': 'owner_pause'})
    workers.RUNNING[ROOT]['owner_wait'] = {'state': 'waiting', 'reason': 'owner_pause'}
    monkeypatch.setattr(continuation_admission, 'action_writers', lambda *a, **k: [])
    result = br.resume_warm_owner_pause_root(ROOT)
    assert result['ok'], result
    assert owner_pause.read_fence(root, ROOT)['resume_grant']['grant_id'] == result['grant_id']
    assert eb.live_root_resume_grant(q, ROOT, root)['grant_id'] == result['grant_id']
    return result['grant_id']


def _hold_root(root, q, workers):
    # Input to the real saved-running recovery owner: this physical worker ended.
    # Pending children are deliberately not replaced by manually authored holds.
    _save_current(root, workers)
    captured = workers.RUNNING.pop(ROOT)
    workers.WORKERS[0].busy_task_id = None
    restored = recover_saved_running(
        [captured], root, transaction={}, cause='app_stop')
    assert len(restored) == 1
    q.PENDING.insert(0, restored[0])
    assert eb.budget_hold_fact(restored[0])['reason'] == eb.HOLD_SAVED_WORK
    return restored[0]


def _saved_again(root, q, workers):
    _hold_root(root, q, workers)
    result = q.resume_budget_paused_task(ROOT)
    assert result['ok'], result
    grant = eb.live_root_resume_grant(q, ROOT, root)['grant_id']
    workers.assign_tasks()
    assert ROOT in workers.RUNNING
    return grant


def _world(root, monkeypatch, kind):
    q, workers, _ = _saved_tree(root, monkeypatch)
    sent = []
    _idle_worker(workers, sent)
    assert q.resume_budget_paused_task(ROOT)['ok']
    workers.assign_tasks()
    assert [row['id'] for row in sent] == [ROOT]
    if kind in {'exact', 'saved_after_exact', 'warm_after_exact'}:
        _exact_root(root, monkeypatch, q, workers)
    if kind in {'warm', 'exact_after_warm', 'warm_after_exact'}:
        _warm_root(root, monkeypatch, q, workers)
    if kind == 'exact_after_warm':
        _exact_root(root, monkeypatch, q, workers)
    if kind == 'saved_after_exact':
        _saved_again(root, q, workers)
    _save_current(root, workers)
    return q, workers, sent


def _select(root, monkeypatch, q, *, exact=False, by_owner=False):
    if exact:
        q.PENDING.remove(_row(q, CHILD))
        _parked(root, monkeypatch, task_id=CHILD, root_task_id=ROOT,
                extra={'parent_task_id': ROOT, 'delegation_role': 'subagent', 'depth': 1})
    result = q.resume_budget_paused_task(CHILD, selected_by='' if by_owner else ROOT)
    assert result['ok'], result
    return _row(q, CHILD)


def _hide_root(root, monkeypatch, q, workers, shape):
    task = copy.deepcopy(workers.RUNNING.pop(ROOT)['task'])
    workers.WORKERS[0].busy_task_id = None
    task.pop('_budget_pause_resume', None)
    task.pop(eb.SELECTION_KEY, None)
    if shape == 'pending_without_carrier':
        q.PENDING.insert(0, task)
        allowed = workers.repo_writer_task_allowed
        monkeypatch.setattr(workers, 'repo_writer_task_allowed',
                            lambda candidate: candidate['id'] != ROOT and allowed(candidate))
    else:
        write_task_result(root, ROOT, 'completed', root_task_id=ROOT, result='done')
    return task


@pytest.mark.parametrize('kind', [
    'saved', 'exact', 'warm', 'saved_after_exact', 'exact_after_warm', 'warm_after_exact',
])
@pytest.mark.parametrize('shape', ['absent', 'pending_without_carrier'])
@pytest.mark.parametrize('exact_child', [False, True], ids=['held-child', 'exact-child'])
def test_existing_choice_survives_carrier_loss(
        tmp_path, monkeypatch, kind, shape, exact_child):
    q, workers, sent = _world(tmp_path, monkeypatch, kind)
    child = _select(tmp_path, monkeypatch, q, exact=exact_child)
    selected = copy.deepcopy(child)
    _hide_root(tmp_path, monkeypatch, q, workers, shape)
    assert eb.budget_resume_dispatch_allowed(q, child) is True
    assert child == selected, 'the reader cannot manufacture a replacement selection'
    workers.assign_tasks()
    assert [row['id'] for row in sent].count(CHILD) == 1
    assert OTHER not in [row['id'] for row in sent]
    assert eb.budget_hold_fact(_row(q, OTHER))
    launched = next(row for row in sent if row['id'] == CHILD)
    assert launched.get('_budget_pause_resume') or launched['_working_recovery']['source_ref']


def test_selected_saved_child_keeps_basis_through_acknowledged_restore(tmp_path, monkeypatch):
    q, workers, sent = _world(tmp_path, monkeypatch, 'saved_after_exact')
    child = _select(tmp_path, monkeypatch, q)
    before = copy.deepcopy(child[eb.SELECTION_KEY])
    _hide_root(tmp_path, monkeypatch, q, workers, 'absent')
    _restart(q, workers, monkeypatch)
    restored = _row(q, CHILD)
    assert restored[eb.SELECTION_KEY] == before
    assert eb.budget_resume_dispatch_allowed(q, restored) is True
    workers.assign_tasks()
    assert [row['id'] for row in sent].count(CHILD) == 1
    assert eb.budget_hold_fact(_row(q, OTHER))


def _erase_issue_times(root):
    def legacy(document):
        document = copy.deepcopy(document)
        for grant in (
                (document.get('budget_pause') or {}).get('grant') or {},
                (document.get('owner_pause') or {}).get('resume_grant') or {}):
            grant.pop('granted_at', None)
            grant.pop('granted_at_ts', None)
        return document
    patch_document(root, ROOT, legacy)


@pytest.mark.parametrize('old_kind,new_kind', [
    ('saved', 'exact'), ('exact', 'warm'), ('warm', 'exact'),
    ('exact', 'saved'), ('warm', 'saved'),
])
@pytest.mark.parametrize('stamps', ['equal', 'backward', 'missing'])
def test_real_newer_resume_defeats_old_choice_after_root_leaves(
        tmp_path, monkeypatch, old_kind, new_kind, stamps):
    now = [time.time()]
    monkeypatch.setattr(br, 'time', SimpleNamespace(time=lambda: now[0]))
    monkeypatch.setattr(br, 'utc_now_iso', lambda: '2026-10-16T08:00:00+00:00')
    monkeypatch.setattr(eb, 'utc_now_iso', lambda: '2026-10-16T08:00:00+00:00')
    q, workers, sent = _world(tmp_path, monkeypatch, old_kind)
    child = _select(tmp_path, monkeypatch, q)
    old = copy.deepcopy(child)
    if stamps == 'backward':
        now[0] -= 100
        monkeypatch.setattr(br, 'utc_now_iso', lambda: '2026-10-15T08:00:00+00:00')
    if new_kind == 'exact':
        _exact_root(tmp_path, monkeypatch, q, workers)
    elif new_kind == 'warm':
        _warm_root(tmp_path, monkeypatch, q, workers)
    else:
        _saved_again(tmp_path, q, workers)
    if stamps == 'missing':
        _erase_issue_times(tmp_path)
    if new_kind != 'saved':
        # An older queue projection cannot bypass the durable supersession.
        child.clear()
        child.update(old)
    else:
        assert eb.budget_hold_fact(child), 'new no-fence publication invalidates the old choice'
    _hide_root(tmp_path, monkeypatch, q, workers, 'absent')
    if new_kind != 'saved':
        assert eb.budget_resume_dispatch_allowed(q, child) is False
    workers.assign_tasks()
    assert CHILD not in [row['id'] for row in sent]
    assert eb.budget_hold_fact(_row(q, OTHER))
    # Explicit owner reselection remains possible; no historical grant is minted.
    selected = q.resume_budget_paused_task(CHILD)
    assert selected['ok'], selected
    workers.assign_tasks()
    assert [row['id'] for row in sent].count(CHILD) == 1


@pytest.mark.parametrize('fault', ['retry', 'pause', 'owner_fence'])
def test_unknown_authority_preserves_choice_until_read_recovers(tmp_path, monkeypatch, fault):
    q, workers, sent = _world(tmp_path, monkeypatch, 'saved_after_exact')
    child = _select(tmp_path, monkeypatch, q)
    _hide_root(tmp_path, monkeypatch, q, workers, 'absent')
    before = copy.deepcopy(child)
    with monkeypatch.context() as broken:
        def unreadable(*args, **kwargs):
            raise OSError('bounded authority outage')
        if fault == 'retry':
            broken.setattr(cancel_intents, '_validated_single_cancel_target', unreadable)
        elif fault == 'pause':
            broken.setattr(bp, 'budget_pause_row', unreadable)
        else:
            broken.setattr(owner_pause, 'read_fence', unreadable)
        assert eb.budget_resume_dispatch_allowed(q, child) is None
        workers.assign_tasks()
        assert child == before
        assert CHILD not in [row['id'] for row in sent]
    assert eb.budget_resume_dispatch_allowed(q, child) is True
    workers.assign_tasks()
    assert [row['id'] for row in sent].count(CHILD) == 1


def test_failed_no_fence_publication_restores_bound_child_choice(tmp_path, monkeypatch):
    q, workers, _ = _world(tmp_path, monkeypatch, 'saved_after_exact')
    child = _select(tmp_path, monkeypatch, q)
    root = _hold_root(tmp_path, q, workers)
    before_root, before_child = copy.deepcopy(root), copy.deepcopy(child)
    snapshot = q.persist_queue_snapshot
    with monkeypatch.context() as failed:
        failed.setattr(q, 'persist_queue_snapshot', lambda *a, **k:
                       False if k.get('reason') == 'budget_hold_selected' else snapshot(*a, **k))
        result = q.resume_budget_paused_task(ROOT)
    assert result['error'] == 'snapshot_not_persisted'
    assert root == before_root and child == before_child
    assert q.resume_budget_paused_task(ROOT)['ok']
    assert eb.budget_hold_fact(child)


@pytest.mark.parametrize('kind', ['saved', 'exact', 'warm'])
def test_consumed_history_cannot_issue_a_new_model_choice_after_root_leaves(
        tmp_path, monkeypatch, kind):
    q, workers, _ = _world(tmp_path, monkeypatch, kind)
    _hide_root(tmp_path, monkeypatch, q, workers, 'absent')
    assert eb.live_root_resume_grant(q, ROOT, tmp_path) == {}
    refused = q.resume_budget_paused_task(CHILD, selected_by=ROOT)
    assert not refused['ok'] and refused['error'] == 'root_resume_grant_missing'
    assert q.resume_budget_paused_task(CHILD)['ok'], 'the owner can make a new explicit choice'


@pytest.mark.parametrize('kind', ['saved', 'exact', 'warm_after_exact'])
def test_legacy_choice_is_matching_or_explicitly_reselectable(tmp_path, monkeypatch, kind):
    q, workers, sent = _world(tmp_path, monkeypatch, kind)
    child = _select(tmp_path, monkeypatch, q)
    child[eb.SELECTION_KEY].pop('root_resume_basis')
    if kind == 'warm_after_exact':
        def legacy(document):
            document = copy.deepcopy(document)
            document['budget_pause']['grant'].pop('root_resume_basis', None)
            document['owner_pause']['resume_grant'].pop('root_resume_basis', None)
            return document
        patch_document(tmp_path, ROOT, legacy)
    _hide_root(tmp_path, monkeypatch, q, workers, 'absent')
    if kind == 'warm_after_exact':
        assert eb.budget_resume_dispatch_allowed(q, child) is False
        workers.assign_tasks()
        assert eb.budget_hold_fact(child)
        assert q.resume_budget_paused_task(CHILD)['ok']
    else:
        assert eb.budget_resume_dispatch_allowed(q, child) is True
    workers.assign_tasks()
    assert [row['id'] for row in sent].count(CHILD) == 1
    assert eb.budget_hold_fact(_row(q, OTHER))


def test_owner_unbound_choice_is_rebound_and_survives_root_completion(tmp_path, monkeypatch):
    q, workers, _ = _saved_tree(tmp_path, monkeypatch)
    assert q.resume_budget_paused_task(CHILD)['ok']
    assert not _row(q, CHILD)[eb.SELECTION_KEY].get('root_grant_id')
    assert q.resume_budget_paused_task(ROOT)['ok']
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    _hide_root(tmp_path, monkeypatch, q, workers, 'absent')
    assert eb.budget_resume_dispatch_allowed(q, _row(q, CHILD)) is True
    workers.assign_tasks()
    assert [row['id'] for row in sent] == [ROOT, CHILD]
    assert eb.budget_hold_fact(_row(q, OTHER))


# Physical retry identity is produced by the existing retry fixture, not by
# assigning a root role to a fabricated member row.
from tests.test_swarm_host_admission import host as host  # pytest fixture export
from tests.test_execution_clock_resume_handover import _supervisor_events
from tests.test_execution_clock_resume_handover import clock as clock  # pytest fixture export
from tests.test_retry_pause_identity import _assign, _fresh_plan, _loop_once
from tests.test_retry_pause_identity import reaper as reaper  # pytest fixture export
from tests.test_retry_pause_generations import _park, _running_actor
from tests.test_dir02_consumed_grant_retry import _child, _select_child_as, CHILD as RETRY_CHILD


def test_selected_child_survives_current_physical_retry_root_completion(
        host, monkeypatch, clock, reaper):
    from supervisor import queue, workers
    from supervisor.owner_pause_control import request_owner_pause

    running = _running_actor(host, monkeypatch, clock, reaper, 'retry')
    actor = running['id']
    assert actor != 'held'
    plan = _fresh_plan(monkeypatch)
    _child(parent=actor)
    assert request_owner_pause(actor, request_id='r2-physical-root')['ok']
    _park(host, running, plan)
    granted = queue.resume_budget_paused_task(actor)
    assert granted['ok']
    [resumed] = _assign(host, monkeypatch)
    assert resumed['id'] == actor
    _loop_once(host, monkeypatch, resumed, plan=plan, events=_supervisor_events(host))
    assert _select_child_as(host, actor)['ok']
    selected = _row(queue, RETRY_CHILD)
    workers.RUNNING.pop(actor)
    write_task_result(host.root, actor, 'completed', root_task_id='held', result='done')
    assert cancel_intents._validated_single_cancel_target(host.root, 'held') == actor
    assert eb.budget_resume_dispatch_allowed(queue, selected) is True
    assert [row['id'] for row in _assign(host, monkeypatch)] == [RETRY_CHILD]
