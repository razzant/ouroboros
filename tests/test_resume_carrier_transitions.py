"""Regressions for saved-selection -> exact Resume carrier transitions.

Park/grant/consumption/recovery/assignment are real; physical workers and
cognition use the existing suite's doubles. The owner-wait copy below tests
retirement only, not a complete restored owner-wait execution.
"""
import copy
from types import SimpleNamespace

import pytest

from ouroboros.work_stop import retain_consumed_selection
from ouroboros import budget_pause as bp, working_checkpoint as wc
from supervisor import budget_resume as br, events_budget as eb
from tests._budget_pause_exact_helpers import (
    _fast_hold, _loop_ctx, _mock_pause_observation, _parked,
)
from tests.test_budget_pause_holds import _idle_worker
from tests.test_budget_pause_lineage_restore import _consume, _restart
from tests.test_root_resume_selection_survival import (
    _exact_root, _save_current, _world,
)
from tests.test_saved_root_resume_grant import _saved_tree

pytestmark = pytest.mark.serial
ROOT, CHILD, OTHER = 'saved-root', 'first', 'second'


def test_consumed_exact_member_retires_its_previous_saved_selection(tmp_path, monkeypatch):
    q, workers, sent = _world(tmp_path, monkeypatch, 'saved')
    assert q.resume_budget_paused_task(CHILD, selected_by=ROOT)['ok']
    workers.WORKERS[0].busy_task_id = None
    workers.assign_tasks()
    previous = copy.deepcopy(workers.RUNNING[CHILD]['task'][eb.SELECTION_KEY])
    newer = _exact_root(tmp_path, monkeypatch, q, workers)
    assert newer != previous['root_grant_id']
    _save_current(tmp_path, workers)

    ctx, limit = _loop_ctx(
        tmp_path, CHILD, attempt=workers.RUNNING[CHILD]['attempt'])
    ctx.root_task_id = ROOT
    ctx.task_metadata = {
        'root_task_id': ROOT, 'parent_task_id': ROOT,
        'delegation_role': 'subagent', 'depth': 1,
    }
    _fast_hold(monkeypatch, bp)
    _mock_pause_observation(monkeypatch, bp, [])
    with pytest.raises(bp.BudgetPauseRequested) as paused:
        bp.request_pause(
            limit, rail=bp.RAIL_GLOBAL_EXHAUSTED, scope='global',
            reason_text='later budget boundary')
    bp.end_dispatch_fence(CHILD)
    sup = SimpleNamespace(
        DRIVE_ROOT=tmp_path, RUNNING=workers.RUNNING, PENDING=workers.PENDING,
        WORKERS=workers.WORKERS, sort_pending=q.sort_pending,
        persist_queue_snapshot=q.persist_queue_snapshot, bridge=None)
    eb.install_exact_budget_pause(
        sup, CHILD, bp.exact_pause_marker(paused.value.pause)['checkpoint'])
    assert q.resume_budget_paused_task(CHILD, selected_by=ROOT)['ok']
    workers.WORKERS[0].busy_task_id = None
    workers.assign_tasks()
    live = workers.RUNNING[CHILD]['task']
    handoff = copy.deepcopy(live['_budget_pause_resume'])
    assert handoff['root_grant_id'] == newer
    assert eb.SELECTION_KEY not in live and eb.BUDGET_HOLD_KEY not in live
    _consume(monkeypatch, ctx, limit, handoff)
    exact = bp.budget_pause_row(tmp_path, CHILD)
    assert exact['state'] == bp.STATE_RESUMED and exact['grant']['consumed_at']
    wc.save_round(limit, 'post_batch')

    # The other actual caller must retire the same obsolete selected carrier.
    waiting = copy.deepcopy(live)
    waiting['_owner_wait_resume'] = {
        'wait_id': 'retirement-only', 'restart_transaction_id': 'fixture'}
    assert br.revoke_exact_budget_resume(waiting, 'restart_before_dispatch') is False
    assert '_budget_pause_resume' not in waiting
    assert eb.SELECTION_KEY not in waiting
    assert eb.BUDGET_HOLD_KEY not in waiting

    # Even a stale handoff beside a newer UNSELECTED hold cannot erase that hold.
    held = copy.deepcopy(live)
    eb.hold_budget_row(held, reason=eb.HOLD_SAVED_WORK, detail='newer hold')
    held['_budget_pause_resume'] = handoff
    before_held = copy.deepcopy(held)
    assert retain_consumed_selection(held, exact) is False
    assert held == before_held

    # No fixture cleanup: acknowledged recovery itself must retire the old hold.
    _restart(q, workers, monkeypatch)
    child = next(row for row in q.PENDING if row['id'] == CHILD)
    assert '_budget_pause_resume' not in child
    assert eb.SELECTION_KEY not in child
    assert eb.BUDGET_HOLD_KEY not in child
    assert bp.budget_pause_row(tmp_path, CHILD)['grant'] == exact['grant']
    assert eb.budget_resume_dispatch_allowed(q, child) is True
    sent.clear()
    for _ in range(3):
        workers.WORKERS[0].busy_task_id = None
        workers.assign_tasks()
    dispatched = [row['id'] for row in sent]
    assert dispatched.count(CHILD) == 1, dispatched
    assert OTHER not in dispatched
    assert eb.budget_hold_fact(next(row for row in q.PENDING if row['id'] == OTHER))


@pytest.mark.parametrize('failed_first_publish', [False, True])
def test_owner_unbound_exact_choice_survives_saved_root_resume(
        tmp_path, monkeypatch, failed_first_publish):
    q, workers, rows = _saved_tree(tmp_path, monkeypatch)
    q.PENDING.remove(rows[CHILD])
    member, _ = _parked(
        tmp_path, monkeypatch, task_id=CHILD, root_task_id=ROOT,
        extra={'parent_task_id': ROOT, 'delegation_role': 'subagent', 'depth': 1})
    granted = q.resume_budget_paused_task(CHILD)
    assert granted['ok'] and granted['exact_continuation'], granted
    original = copy.deepcopy(member['_budget_pause_resume'])
    durable = copy.deepcopy(bp.budget_pause_row(tmp_path, CHILD))
    assert original['selected_by'] == 'owner' and not original['root_grant_id']

    if failed_first_publish:
        before_root, before_member = copy.deepcopy(rows[ROOT]), copy.deepcopy(member)
        snapshot = q.persist_queue_snapshot
        with monkeypatch.context() as failed:
            failed.setattr(
                q, 'persist_queue_snapshot',
                lambda *a, **k: False if k.get('reason') == 'budget_hold_selected'
                else snapshot(*a, **k))
            refused = q.resume_budget_paused_task(ROOT)
        assert refused['error'] == 'snapshot_not_persisted', refused
        assert refused.get('ok') is False, refused
        assert rows[ROOT] == before_root and member == before_member
        assert bp.budget_pause_row(tmp_path, CHILD) == durable

    resumed = q.resume_budget_paused_task(ROOT)
    assert resumed['ok'], resumed
    current = member['_budget_pause_resume']
    root_grant = eb.live_root_resume_grant(q, ROOT, tmp_path)
    assert current['root_grant_id'] == root_grant['grant_id']
    assert current['root_resume_generation'] == root_grant['generation']
    assert {key: current[key] for key in ('grant_id', 'pause_id', 'source_ref')} == {
        key: original[key] for key in ('grant_id', 'pause_id', 'source_ref')}
    assert bp.budget_pause_row(tmp_path, CHILD) == durable, 'no exact grant reissued'
    assert eb.budget_resume_dispatch_allowed(q, member) is True

    sent = []
    _idle_worker(workers, sent)
    for _ in range(3):
        workers.WORKERS[0].busy_task_id = None
        workers.assign_tasks()
    dispatched = [row['id'] for row in sent]
    assert dispatched.count(ROOT) == dispatched.count(CHILD) == 1, dispatched
    assert OTHER not in dispatched and eb.budget_hold_fact(rows[OTHER])
    ctx, limit = _loop_ctx(tmp_path, CHILD)
    ctx.root_task_id = ROOT
    ctx.task_metadata = {
        'root_task_id': ROOT, 'parent_task_id': ROOT,
        'delegation_role': 'subagent', 'depth': 1,
    }
    _consume(monkeypatch, ctx, limit, current)
    consumed = bp.budget_pause_row(tmp_path, CHILD)
    assert consumed['grant']['grant_id'] == original['grant_id']
    assert consumed['grant']['consumed_at'] and not consumed['grant'].get('revoked_at')


def test_no_latch_root_grant_cannot_disappear_as_supersession(tmp_path, monkeypatch):
    """Grant/consume are real; only the consumed root's queue removal is injected."""
    q, _workers, rows = _saved_tree(tmp_path, monkeypatch)
    q.PENDING.remove(rows[CHILD])
    child, _ = _parked(
        tmp_path, monkeypatch, task_id=CHILD, root_task_id=ROOT,
        extra={'parent_task_id': ROOT, 'delegation_role': 'subagent'})
    assert not q.BUDGET_ROOT_FENCES
    granted = q.resume_budget_paused_task(CHILD)
    assert granted['ok'], granted
    original = copy.deepcopy(child['_budget_pause_resume'])
    assert original['selected_by'] == 'owner' and not original['root_grant_id']
    assert not original['root_fence_id']
    q.PENDING.remove(rows[ROOT])
    root, _ = _parked(tmp_path, monkeypatch, task_id=ROOT)
    resumed = q.resume_budget_paused_task(ROOT)
    assert resumed['ok'], resumed
    assert child['_budget_pause_resume'] == original
    assert eb.budget_resume_dispatch_allowed(q, child) is False
    ctx, limit = _loop_ctx(tmp_path, ROOT)
    _consume(monkeypatch, ctx, limit, root['_budget_pause_resume'])
    q.PENDING.remove(root)
    assert eb.budget_resume_dispatch_allowed(q, child) is False
