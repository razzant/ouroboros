"""External review discriminator: one owner Resume owes both paused late paths.

All paid transport is synthetic. Source is imported read-only; data lives in the
safe_test disposable root. The retained preparation controller is joined.
"""
from types import SimpleNamespace

import pytest

from tests.test_acceptance_late_consumers import delivered
from tests.test_acceptance_late_consumers import late as late  # noqa: F401 — fixture
from tests.test_review_operation_collection import fresh_sends as fresh_sends  # noqa: F401 — fixture
from tests.test_late_phase_pause_acceptance import _preparing_process, _end_preparer, _tick
from tests._budget_pause_exact_helpers import _install_queue

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("unknown_preparation", [False, True])
def test_one_resume_restores_post_task_and_dead_acceptance_preparation(late, tmp_path, monkeypatch, unknown_preparation):
    from ouroboros import agent_task_pipeline as pipeline, review_operation
    from ouroboros.owner_pause import read_fence
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.usage_accounting import UsageScope, usage_scope, AttemptRequest, execute_physical_attempt
    from supervisor.owner_pause_control import request_owner_pause
    from tests.test_late_phase_pause_resume import _spawned, _join

    f = delivered(tmp_path, monkeypatch)
    q, _, workers = _install_queue(f.root, monkeypatch)
    monkeypatch.setattr(workers, 'REPO_DIR', tmp_path, raising=False)
    monkeypatch.delenv('OUROBOROS_IN_WORKER', raising=False)
    proc, log, preparation = _preparing_process(f)
    task = {**f.task, '_skip_post_task_synthesis': False}
    env = SimpleNamespace(drive_root=f.root, repo_dir=tmp_path, drive_path=lambda rel: f.root / rel)
    entered = []

    from ouroboros.model_wait import model_waitable
    @model_waitable
    def synthetic_chat(self, model='fixture', model_role='light', **kwargs):
        return execute_physical_attempt(
            AttemptRequest(model=model, provider='fixture', reservation_usd=.01),
            lambda: ({'content': 'synthetic stage result'}, {}),
            extractor=lambda response: (response[1], .01, True))

    def late_stage(*_a, **_kw):
        entered.append('scratchpad_consolidation')
        if len(entered) == 1:
            assert request_owner_pause(f.tid, request_id='combined-late-pause')['ok']
        # A real unsent paid boundary observes the accepted Pause. No provider I/O.
        synthetic_chat(None)

    # The first paid late stage (scratchpad consolidation since the dialogue writer is retired).
    monkeypatch.setattr(pipeline, '_run_scratchpad_consolidation', late_stage)
    monkeypatch.setattr(pipeline, '_run_reflection', lambda *_a, **_k: None)
    try:
        with usage_scope(UsageScope(drive_root=f.root, task_id=f.tid, root_task_id=f.tid,
                                    root_limit_usd=4.0, category='post_task')):
            _, initial_threads = _spawned(lambda: pipeline._run_post_task_processing_async(env, task, {}, {}, {}, f.root / 'logs', blocking=False))
        _join(initial_threads)
        before = load_task_result(f.root, f.tid)
        assert before['root_phase_checkpoint']['post_task_synthesis'] == 'paused'
        old = before['review_operations'][preparation['owner_id']]
        assert old['state'] == 'preparing' and old['preparation_pause']['fence_id'] == read_fence(f.root, f.tid)['fence_id']
    finally:
        _end_preparer(proc, log)
    assert review_operation.controller_state(old['controller']) == 'dead'
    assert pipeline.recover_pending_root_post_task_synthesis(f.root, tmp_path) == 0
    recovered = review_operation.recover_orphaned_acceptance_operations(f.root)
    assert any(x['reason'] == 'owner_paused_preparation' for x in recovered['deferred'])
    _tick(q)
    if unknown_preparation:
        row = load_task_result(f.root, f.tid)
        row['review_operations'][preparation['owner_id']]['state'] = 'preparation_unknown'
        write_task_result(f.root, f.tid, row['status'], review_operations=row['review_operations'])
    # Keep the old queue projection during Resume. The exact durable fence is
    # already reopened before its snapshot update; it owns admission authority.
    persist = q.persist_queue_snapshot
    monkeypatch.setattr(q, 'persist_queue_snapshot', lambda *, reason='', **kw:
                        False if reason == 'owner_pause_late_resumed' else persist(reason=reason, **kw))
    fence_before = read_fence(f.root, f.tid)
    resumed, threads = _spawned(lambda: q.resume_budget_paused_task(f.tid))
    _join(threads)
    if unknown_preparation:
        assert not resumed['ok'], resumed
        assert read_fence(f.root, f.tid) == fence_before
        assert load_task_result(f.root, f.tid)['root_phase_checkpoint']['post_task_synthesis'] == 'paused'
        assert entered == ['scratchpad_consolidation'] and not late.calls
        return
    assert resumed['ok'], resumed
    from tests.test_review_operation_lifetime import until
    until(lambda: not review_operation._LIVE)  # the retained panel can outlive its preparation worker
    after = load_task_result(f.root, f.tid)
    assert after['root_phase_checkpoint']['post_task_synthesis'] == 'completed', after['root_phase_checkpoint']
    assert after['status'] == before['status'] and after['result'] == before['result']
    observed = after['review_operations'][preparation['owner_id']]
    recovery_after_resume = review_operation.recover_orphaned_acceptance_operations(f.root)
    final = load_task_result(f.root, f.tid)['review_operations'][preparation['owner_id']]
    print('COMBINED_OBSERVATION', {'resume': resumed, 'fence': read_fence(f.root, f.tid),
          'post_phase': after['root_phase_checkpoint']['post_task_synthesis'],
          'review_state_after_resume': observed['state'], 'review_state_after_maintenance': final['state'],
          'same_dead_controller': observed['controller'] == old['controller'],
          'review_sends': len(late.calls), 'maintenance': recovery_after_resume})
    assert observed['controller'] != old['controller'], 'Resume released the shared fence without restoring its dead unsent acceptance controller'
    assert len(late.calls) == 3
    assert set(after['review_operations']) == {preparation['owner_id']}
    assert after['acceptance_debt'] == before['acceptance_debt']
    assert entered == ['scratchpad_consolidation', 'scratchpad_consolidation']
    assert all(scope.task_id == f.tid and scope.root_task_id == f.tid and scope.root_limit_usd == 4.0
               for scope, _ in late.calls)
    assert not q.resume_budget_paused_task(f.tid)['ok']
    review_operation.recover_orphaned_acceptance_operations(f.root)
    assert len(late.calls) == 3 and not review_operation._LIVE
