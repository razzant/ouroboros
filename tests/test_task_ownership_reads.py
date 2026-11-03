"""Durable ownership parses happen before the queue mutation interlock."""
from collections import Counter
from types import SimpleNamespace

import pytest

from ouroboros import task_results, review_operation
from supervisor import queue, queue_transitions, workers
from supervisor.task_ownership import TaskOwnershipRead


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, 'DRIVE_ROOT', tmp_path)
    monkeypatch.setattr(queue, 'RUNNING', {})
    monkeypatch.setattr(queue, 'PENDING', [])
    monkeypatch.setattr(queue, 'INITIALIZED', True)
    monkeypatch.setattr(queue, 'QUEUE_MAX_RETRIES', 3)
    monkeypatch.setattr(workers, 'WORKERS', {})
    monkeypatch.setattr(workers, 'direct_chat_turn', lambda _: None)
    monkeypatch.setattr(queue_transitions, 'post_task_synthesis_in_flight', lambda *_: False)
    task_results.write_task_result(tmp_path, 'root', 'completed', result='Retained answer')
    return tmp_path


def count_reads(monkeypatch):
    counts = Counter()
    load = task_results.load_task_result
    def read(root, tid, **kw):
        assert not queue._queue_lock._is_owned(), 'result parse under queue lock'
        counts[tid] += 1
        return load(root, tid, **kw)
    monkeypatch.setattr(task_results, 'load_task_result', read)
    return counts


def test_one_parse_feeds_all_durable_checks(env, monkeypatch):
    counts = count_reads(monkeypatch)
    assert not queue_transitions.task_has_live_ownership('root')
    assert counts == {'root': 1}


def test_reciprocal_retry_links_read_once_before_locked_resolution(env, monkeypatch):
    lineage = dict(root_task_id='root', parent_task_id='', delegation_role='root',
                   original_task_id='root', timeout_retry_from='root', supersedes_task_id='root')
    task_results.write_task_result(env, 'retry', 'completed', **lineage)
    task_results.write_task_result(env, 'root', 'completed', superseded_by='retry', retry_task_id='retry')
    queue.RUNNING['retry'] = {'task': {'id': 'retry', **lineage}}
    counts = count_reads(monkeypatch)
    reads = TaskOwnershipRead(env)
    assert queue_transitions.task_has_live_ownership('root', ownership=reads)
    with queue._queue_lock:
        assert queue_transitions._live_retry_target_locked(queue, 'root', results=reads) == ('retry', '')
        assert queue_transitions.task_has_live_ownership('root', ownership=reads)
    assert counts == {'root': 1, 'retry': 1}


def test_control_review_and_paused_fence_share_the_same_parse(env, monkeypatch):
    """The review-operation check and the Pause fence read one parse. A preparation bound to the
    owner's Pause was written only by the retired task acceptance: with its controller dead it holds
    nothing any more, while a live controller still owns the task."""
    entry = {'state': review_operation.OPERATION_PREPARING,
             'controller': {'fixture': 'dead'}, 'intent_ref': {'id': 'intent'},
             'preparation_pause': {'root_task_id': 'root', 'fence_id': 'fence'}}
    task_results.write_task_result(env, 'root', 'completed', owner_pause={'state': 'paused', 'fence_id': 'fence'},
                                   review_operations={'operation': entry})
    monkeypatch.setattr(review_operation, 'controller_state', lambda _: 'dead')
    counts = count_reads(monkeypatch)
    assert not queue_transitions.task_has_live_ownership('root')
    assert counts == {'root': 1}
    monkeypatch.setattr(review_operation, 'controller_state', lambda _: 'alive')
    assert queue_transitions.task_has_live_ownership('root')
    assert counts == {'root': 2}  # one parse per check


def test_mutation_boundary_observes_new_memory_owner(env, monkeypatch):
    counts = count_reads(monkeypatch)
    reads = TaskOwnershipRead(env)
    assert not queue_transitions.task_has_live_ownership('root', ownership=reads)
    workers.WORKERS[1] = SimpleNamespace(busy_task_id='root')
    with queue._queue_lock:
        assert queue_transitions.task_has_live_ownership('root', ownership=reads)
    assert counts == {'root': 1}


def test_changed_durable_row_retains_custody_without_second_parse(env, monkeypatch):
    reads = TaskOwnershipRead(env)
    assert not queue_transitions.task_has_live_ownership('root', ownership=reads)
    task_results.write_task_result(env, 'root', 'completed', root_phase_checkpoint={'post_task_synthesis': 'paused'})
    counts = count_reads(monkeypatch)
    with queue._queue_lock:
        assert queue_transitions.task_has_live_ownership('root', ownership=reads)
    assert not counts


def test_settlement_preparation_survives_the_callers_rlock(env, monkeypatch):
    counts = count_reads(monkeypatch)
    assert queue_transitions.task_settlement_liveness('root') is False
    with queue._queue_lock:
        assert queue_transitions.task_settlement_liveness('root') is False
    assert counts == {'root': 1}
    with queue._queue_lock:
        assert queue_transitions.task_settlement_liveness('unprepared') is None


def test_unconsumed_settlement_preparations_are_bounded_per_thread(tmp_path):
    """An unlocked probe whose locked follow-up never comes keeps its parsed result bodies only up to
    the memory bound; the newest stay (a multi-occupant settlement consumes its own right away)."""
    from supervisor import task_ownership

    for index in range(task_ownership._PREPARED_READS_MAX + 40):
        task_ownership.settlement_reads(tmp_path, f"t{index}", locked=False,
                                        prepared=task_ownership.TaskOwnershipRead(tmp_path))
    pending = task_ownership._SETTLEMENT_READS.pending
    assert len(pending) == task_ownership._PREPARED_READS_MAX
    newest = f"t{task_ownership._PREPARED_READS_MAX + 39}"
    assert task_ownership.settlement_reads(tmp_path, newest, locked=True) is not None
    pending.clear()


def test_recheck_is_tri_state_and_never_waits_under_the_queue_lock(env, monkeypatch):
    """A3: the prepared signature is re-checked as UNCHANGED / CHANGED / BUSY. Under Q the
    probe uses a zero budget, so a busy store answers BUSY (unknown) — never CHANGED and
    never UNCHANGED — while ``unchanged()`` keeps its boolean 'proven unchanged' meaning."""
    from ouroboros import store, task_store
    from supervisor.task_ownership import BUSY, CHANGED, UNCHANGED

    budgets = []
    real_budget = store.acquisition_budget

    def _spy_budget(seconds):
        budgets.append((seconds, queue._queue_lock._is_owned()))
        return real_budget(seconds)

    monkeypatch.setattr(store, 'acquisition_budget', _spy_budget)
    reads = TaskOwnershipRead(env)
    reads.load('root')
    assert budgets == [], 'the signed read needs no zero-budget probe outside Q'
    assert reads.recheck(('root',)) == UNCHANGED and reads.unchanged(('root',)) is True
    assert budgets == []
    with queue._queue_lock:
        assert reads.recheck(('root',)) == UNCHANGED
    assert budgets == [(0, True)]

    task_results.write_task_result(env, 'root', 'completed', result='Rewritten answer')
    with queue._queue_lock:
        assert reads.recheck(('root',)) == CHANGED and reads.unchanged(('root',)) is False
        with pytest.raises(ValueError, match='changed'):
            reads.load('root')

    reads = TaskOwnershipRead(env)
    reads.load('root')
    real_sig = task_store.signature

    def _busy_at_zero(root, tid, *, txn=None):
        if store._WAIT_BUDGET.get() == 0:
            raise store.StoreBusy('writer active')
        return real_sig(root, tid, txn=txn)

    monkeypatch.setattr(task_store, 'signature', _busy_at_zero)
    with queue._queue_lock:
        assert reads.recheck(('root',)) == BUSY and reads.unchanged(('root',)) is False
        with pytest.raises(ValueError, match='busy'):
            reads.load('root')
        assert reads.prepared('root')['status'] == 'completed', 'the prepared row needs no re-check'
    assert reads.recheck(('root',)) == UNCHANGED, 'outside Q the probe waits with the normal budget'
