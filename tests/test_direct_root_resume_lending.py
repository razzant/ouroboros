"""Direct-registry consumer for warm Resume lending; parking/quiescence are fixture inputs."""
import pytest

from tests.test_saved_root_resume_grant import _saved_tree
from tests.test_direct_chat_turn_owner_control import _live_chat_agent

pytestmark = pytest.mark.serial


@pytest.mark.parametrize('exact_child', [False, True], ids=['saved-child', 'exact-child'])
def test_direct_warm_root_can_select_saved_child(tmp_path, monkeypatch, exact_child):
    from ouroboros import owner_pause
    from ouroboros.task_results import write_task_result
    from supervisor import budget_resume as br, continuation_admission
    from supervisor import events_budget as eb
    from supervisor.active_activity import get_direct_activity_registry
    from supervisor.workers import direct_chat_turn
    from tests._budget_pause_exact_helpers import _parked

    q, workers, rows = _saved_tree(tmp_path, monkeypatch)
    q.PENDING.remove(rows['saved-root'])
    if exact_child:
        q.PENDING.remove(rows['first'])
        child, _ = _parked(tmp_path, monkeypatch, task_id='first', root_task_id='saved-root',
                           extra={'parent_task_id': 'saved-root', 'delegation_role': 'subagent', 'depth': 1})
    else:
        child = rows['first']
    agent = _live_chat_agent(monkeypatch, task_id='saved-root')
    registry = get_direct_activity_registry()
    try:
        assert direct_chat_turn('saved-root')['id'] == 'saved-root'
        assert 'saved-root' not in workers.RUNNING
        write_task_result(tmp_path, 'saved-root', 'running', chat_id=0, root_task_id='saved-root',
                          owner_wait={'state': 'waiting', 'reason': 'owner_pause'})
        fence, _ = owner_pause.install_fence(tmp_path, 'saved-root', request_id='direct-warm')
        eb._set_root_budget_pause_locked('saved-root', {**fence, 'cause': 'owner_pause'})
        monkeypatch.setattr(continuation_admission, 'action_writers', lambda *a, **k: [])
        resumed = br.resume_warm_owner_pause_root('saved-root')
        assert resumed['ok'] and resumed['warm'], resumed
        assert direct_chat_turn('saved-root')
        assert 'saved-root' not in workers.RUNNING
        assert not any(row['id'] == 'saved-root' for row in q.PENDING)
        outcome = q.resume_budget_paused_task('first', selected_by='saved-root')
        assert outcome['ok'], outcome
        carrier = child.get('_budget_pause_resume') or child[eb.SELECTION_KEY]
        assert carrier['root_grant_id'] == resumed['grant_id']
        assert eb.budget_hold_fact(rows['second'])
        # A queue/durable record alone cannot make a completed direct actor a lender.
        agent._busy = False
        assert not direct_chat_turn('saved-root')
        assert eb.live_root_resume_grant(q, 'saved-root', tmp_path) == {}
    finally:
        registry.unregister('saved-root')


def test_direct_warm_resume_then_acknowledged_restart_can_select_child(tmp_path, monkeypatch):
    from ouroboros import owner_pause, owner_wait, working_checkpoint as wc
    from ouroboros.task_results import write_task_result
    from supervisor import budget_resume as br, continuation_admission, events_budget as eb
    from supervisor.active_activity import get_direct_activity_registry
    from supervisor.restart_retention import prepare_restart_returns
    from supervisor.workers import direct_chat_turn
    from tests._budget_pause_exact_helpers import _loop_ctx
    from tests.test_direct_chat_turn_owner_control import _live_chat_agent
    from tests.test_restart_saved_work import _ack, _stop_and_boot
    from tests.test_saved_root_resume_grant import _saved_tree
    from tests.test_budget_pause_holds import _idle_worker

    q, workers, rows = _saved_tree(tmp_path, monkeypatch)
    q.PENDING.remove(rows['saved-root'])
    _live_chat_agent(monkeypatch, task_id='saved-root')
    registry = get_direct_activity_registry()
    try:
        write_task_result(tmp_path, 'saved-root', 'running', chat_id=0, root_task_id='saved-root',
                          _is_direct_chat=True, task_attempt=1, owner_wait={'state': 'waiting', 'reason': 'owner_pause'})
        fence, _ = owner_pause.install_fence(tmp_path, 'saved-root', request_id='direct-before-restart')
        eb._set_root_budget_pause_locked('saved-root', {**fence, 'cause': 'owner_pause'})
        monkeypatch.setattr(continuation_admission, 'action_writers', lambda *a, **k: [])
        resumed = br.resume_warm_owner_pause_root('saved-root')
        assert resumed['ok'], resumed
        # The direct stack has returned to ordinary work. Its checkpoint and
        # retained control identities are real; physical parking/OS exit are doubles.
        owner_wait.set_owner_wait(tmp_path, 'saved-root', {'state': 'resumed', 'reason': 'owner_pause'})
        ctx, limit = _loop_ctx(tmp_path, 'saved-root', direct=True)
        wc.save_round(limit, 'post_batch')
        turn = direct_chat_turn('saved-root')
        assert turn and not workers.RUNNING
        returns = prepare_restart_returns(tmp_path, workers.RUNNING, q.PENDING,
                                          transaction_id='direct-warm-return', direct=[turn])
        assert 'saved-root' in returns
        _ack(tmp_path, 'direct-warm-return')
        from supervisor.direct_roots import collect_direct_roots, take_direct_roots
        from ouroboros.utils import atomic_write_json
        atomic_write_json(tmp_path / 'state' / 'direct_roots.json', collect_direct_roots())
        registry.unregister('saved-root')
        # The new queue generation takes the previous process's real collected roster.
        monkeypatch.setattr(q, 'PRIOR_DIRECT_ROOTS', take_direct_roots(tmp_path))
        _stop_and_boot(q, workers)
        restored = next(row for row in q.PENDING if row['id'] == 'saved-root')
        assert restored['_working_recovery']['cause'] == 'restart'
        assert eb.budget_hold_fact(restored) is None
        sent = []
        _idle_worker(workers, sent)
        workers.assign_tasks()
        assert [row['id'] for row in sent] == ['saved-root']
        assert not direct_chat_turn('saved-root') and 'saved-root' in workers.RUNNING
        result = q.resume_budget_paused_task('first', selected_by='saved-root')
        assert result['ok'], result
        assert eb.budget_hold_fact(next(row for row in q.PENDING if row['id'] == 'second'))
        assert owner_pause.read_fence(tmp_path, 'saved-root')['resume_grant']['grant_id'] == resumed['grant_id']
    finally:
        registry.unregister('saved-root')
