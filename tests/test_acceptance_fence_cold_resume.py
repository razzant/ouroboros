"""#1183: exact cold Resume can move an active acceptance fence to another PID.

Admission, agent Pause, durable grant, assignment, loop consumption and pooled
fence event/ack are real; processes and paid transports are synthetic.
"""
import copy
import os
import queue as stdqueue
from types import SimpleNamespace

import pytest

from ouroboros import agent, budget_pause, loop
from supervisor import queue, workers
from tests import test_resume_handover as handover
from tests.test_acceptance_fence_transport import _Supervisor, _pooled_agent
from tests.test_swarm_host_admission import host  # noqa: F401

pytestmark = pytest.mark.serial


def _transport(host, monkeypatch, ctx, pid):  # noqa: F811
    events = stdqueue.Queue()
    _Supervisor(events, host.root)
    actor = _pooled_agent(host.root, events, 'held', attempt=2)
    actor.tools._ctx = ctx

    def begin(**kwargs):
        # Only this synthetic process boundary changes PID; store/process locks keep theirs.
        with monkeypatch.context() as process:
            process.setattr(agent, 'os', SimpleNamespace(**{**vars(os), 'getpid': lambda: pid}))
            return actor._begin_acceptance_fence(**kwargs)

    ctx.begin_acceptance_fence = begin
    ctx.inspect_acceptance_fence = actor._inspect_acceptance_fence
    ctx.end_acceptance_fence = actor._end_acceptance_fence


def _paused_fence(host, tmp_path, monkeypatch, reason='owner', owner_change='', boot=False):  # noqa: F811
    original = handover._pause_through_agent
    prior = {}

    def pause(*args, **kwargs):
        workers.WORKERS[0].proc = SimpleNamespace(pid=1111, is_alive=lambda: True)
        ctx = SimpleNamespace(task_id='held', root_task_id='held', task_attempt=2, drive_root=host.root)
        _transport(host, monkeypatch, ctx, 1111)
        assert loop._begin_task_acceptance_fence(ctx, 'held')[0]
        row = queue.ACCEPTANCE_FENCES['held']
        if owner_change == 'sealed':
            row['status'] = 'sealed'
        elif owner_change == 'direct':
            row.pop('pooled_owner')
        elif owner_change == 'legacy':
            row['pooled_owner'] = {}
        elif owner_change == 'foreign_pid':
            row['pooled_owner']['pid'] = 9999
        elif owner_change == 'foreign_attempt':
            row['pooled_owner']['task_attempt'] = 3
        elif owner_change == 'foreign_task':
            row['task_id'] = 'other'
        row['owner_message_generation'] = 7
        prior.update(copy.deepcopy(row))
        result = original(*args, **kwargs)
        assert {key: value for key, value in queue.ACCEPTANCE_FENCES['held'].items()
                if key != 'paused_continuation'} == prior, 'Pause keeps its fence intact'
        if boot:
            assert handover._stop_boot(monkeypatch) == 1
        return result

    monkeypatch.setattr(handover, '_pause_through_agent', pause)
    rail = budget_pause.RAIL_OWNER_PAUSE if reason == 'owner' else budget_pause.RAIL_GLOBAL_EXHAUSTED
    saved = handover._recovered_then_paused(host, tmp_path, monkeypatch, 'main', rail=rail)
    return saved, prior


def _resume(host, monkeypatch, pid=2222, *, consume=True):  # noqa: F811
    task = handover._dispatch_one(host, monkeypatch)
    workers.WORKERS[0].proc = SimpleNamespace(pid=pid, is_alive=lambda: True)
    if consume:
        seen = handover._run_loop(host, monkeypatch, task, 'after_resume')
        ctx = seen['ctx']
        assert seen['kind'] == 'pause' and seen['round_idx'] == 29
        assert not getattr(ctx, '_task_acceptance_fence_token', None)
    else:
        ctx = SimpleNamespace(task_id='held', root_task_id='held', task_attempt=2, drive_root=host.root)
    _transport(host, monkeypatch, ctx, pid)
    return ctx


@pytest.mark.parametrize('reason', ['owner', 'budget'])
@pytest.mark.parametrize('pid', [1111, 2222], ids=['same-worker', 'different-worker'])
@pytest.mark.parametrize('boot', [False, True], ids=['park', 'park-boot'])
def test_exact_cold_resume_rebinds_and_seals(host, tmp_path, monkeypatch, reason, pid, boot):  # noqa: F811
    from ouroboros.loop_delivery import _no_tool_final_answer
    from ouroboros.task_results import load_task_result, write_task_result

    _, prior = _paused_fence(host, tmp_path, monkeypatch, reason, boot=boot)
    paid = [{'review_id': 'independent-paid-review', 'status': 'pending', 'cost_usd': .25}]
    write_task_result(host.root, 'held', 'scheduled', review_runs=paid)
    ctx = _resume(host, monkeypatch, pid)
    opened, token = loop._begin_task_acceptance_fence(ctx, 'held')
    assert opened, opened
    row = queue.ACCEPTANCE_FENCES['held']
    assert token != prior['token'] and row['pooled_owner'] == {'task_attempt': 2, 'pid': pid}
    assert row['owner_message_generation'] == prior['owner_message_generation']
    # A late old token cannot inspect/release/seal its successor's row.
    for action, outcome in [('inspect', ''), ('end', 'revision'), ('end', 'terminal')]:
        assert queue.transition_acceptance_fence(action=action, token=prior['token'], outcome=outcome)['row_absent']
    if pid != 1111:
        assert queue.transition_acceptance_fence(action='begin', token=prior['token'], task_id='held',
            root_task_id='held', pooled_owner=prior['pooled_owner'])['status'] == 'stale_attempt'
    from ouroboros.tools.registry import ToolRegistry
    tools = ToolRegistry(repo_dir=host.root, drive_root=host.root)
    tools._ctx = ctx
    # The seal runs at an author stop, right before that stop is delivered.
    ctx._completion_selected = {'action': 'stop', 'rationale': 'The cold continuation ends here; nothing else was asked.'}
    trace = {'reasoning_notes': [], 'tool_calls': []}
    limit = loop._RoundLimitContext(messages=[], llm=None, active_model='fixture', active_effort='high',
        max_retries=1, drive_logs=host.root / 'logs', task_id='held', round_idx=30, event_queue=None,
        accumulated_usage={}, task_type='task', active_use_local=False, max_rounds=None,
        drive_root=host.root, tools=tools, llm_trace=trace, tool_schemas=[])
    # Independent reviewers already have custody; an author stop buys no review or model call.
    monkeypatch.setattr('ouroboros.tools.review_result.result_review_at_delivery',
                        lambda *_a, **_k: pytest.fail('an author stop bought a result review'))
    monkeypatch.setattr(loop, '_maybe_inject_finalization_nudges', lambda *_a, **_k: False)
    delivered = _no_tool_final_answer('Completed cold continuation.', limit, trace, tools,
        stdqueue.Queue(), set(), lambda *_a, **_k: None, explicit_candidate=True)
    assert delivered is not None and delivered[0] == 'Completed cold continuation.'
    assert trace['task_completion']['action'] == 'stop'
    assert queue.ACCEPTANCE_FENCES['held']['status'] == 'sealed'
    assert load_task_result(host.root, 'held')['review_runs'] == paid


@pytest.mark.parametrize('change', ['unconsumed', 'newer_pause', 'revoked', 'grant', 'generation',
                                   'foreign_root', 'unreadable', 'missing_binding', 'new_owner_pause', 'terminal'])
def test_cold_handoff_needs_current_consumed_authority(host, tmp_path, monkeypatch, change):  # noqa: F811
    _paused_fence(host, tmp_path, monkeypatch)
    ctx = _resume(host, monkeypatch, consume=change != 'unconsumed')
    current = budget_pause.budget_pause_row(host.root, 'held')
    if change == 'unreadable':
        from tests._store_shared import unreadable_result
        unreadable_result(monkeypatch, 'held')
    elif change == 'new_owner_pause':
        from supervisor.owner_pause_control import request_owner_pause
        assert request_owner_pause('held', request_id='new-pause')['ok']
    elif change == 'terminal':
        from ouroboros.task_results import write_task_result
        write_task_result(host.root, 'held', 'completed', result='Already delivered')
    elif change == 'missing_binding':
        queue.ACCEPTANCE_FENCES['held'].pop('paused_continuation', None)
    elif change != 'unconsumed':
        if change == 'newer_pause':
            current.update(pause_id='newer-pause', state=budget_pause.STATE_PAUSED)
        elif change == 'revoked':
            current['grant']['revoked_at'] = 'revoked'
        elif change == 'grant':
            current['grant']['grant_id'] = 'another-grant'
        elif change == 'generation':
            current['grant']['generation'] += 1
        elif change == 'foreign_root':
            current['root_task_id'] = 'another-root'
        budget_pause.set_budget_pause(host.root, 'held', current)
    before = copy.deepcopy(queue.ACCEPTANCE_FENCES)
    assert not loop._begin_task_acceptance_fence(ctx, 'held')[0]
    assert queue.ACCEPTANCE_FENCES == before


@pytest.mark.parametrize('owner', ['sealed', 'direct', 'legacy', 'foreign_pid', 'foreign_attempt', 'foreign_task'])
def test_cold_pause_never_transfers_an_unbound_owner(host, tmp_path, monkeypatch, owner):  # noqa: F811
    _paused_fence(host, tmp_path, monkeypatch, owner_change=owner)
    ctx = _resume(host, monkeypatch)
    before = copy.deepcopy(queue.ACCEPTANCE_FENCES)
    assert not loop._begin_task_acceptance_fence(ctx, 'held')[0]
    assert queue.ACCEPTANCE_FENCES == before


@pytest.mark.parametrize('consume', [False, True])
def test_a_new_pid_cannot_reuse_the_old_token(host, tmp_path, monkeypatch, consume):  # noqa: F811
    _, prior = _paused_fence(host, tmp_path, monkeypatch)
    _resume(host, monkeypatch, consume=consume)
    before = copy.deepcopy(queue.ACCEPTANCE_FENCES)
    response = queue.transition_acceptance_fence(action='begin', token=prior['token'], task_id='held',
        root_task_id='held', pooled_owner={'task_attempt': 2, 'pid': 2222})
    assert not response['ok'] and queue.ACCEPTANCE_FENCES == before


def _dying_before_consumption(monkeypatch, reap_queue):
    """The resumed worker dies before consuming its grant; health hands its job to the reaper."""
    from tests.test_worker_crash_retry import _make_worker

    worker = _make_worker(busy_task_id='held', exitcode=1)
    worker.proc.pid = 2222
    workers.WORKERS[0] = worker
    monkeypatch.setattr(workers, '_LAST_SPAWN_TIME', 0)
    monkeypatch.setattr(workers, 'CRASH_TS', [])
    monkeypatch.setattr(queue, '_reap_queue', reap_queue)
    monkeypatch.setattr(queue, '_ensure_reaper_started', lambda: None)
    monkeypatch.setattr(workers, 'respawn_worker', lambda *_a, **_k: None)
    monkeypatch.setattr('ouroboros.headless.prepare_terminal_task_files',
                        lambda *_a, **_k: {'terminal_source_present': False})
    workers.ensure_workers_healthy()


def _die_before_consumption(monkeypatch):
    """...and health recovery re-parks it."""
    from supervisor import worker_health

    jobs = stdqueue.Queue()
    _dying_before_consumption(monkeypatch, jobs)
    worker_health.recover_confirmed_dead_worker(jobs.get_nowait())


def _pause_with_latch_pending(host, monkeypatch, task_id, request_id):  # noqa: F811
    """The public Pause whose latch step meets the root's busy launch lock (``latch_pending``):
    another thread holds that real lock from just after the durable fence lands to the answer."""
    import threading

    from ouroboros import owner_pause
    from supervisor.owner_pause_control import request_owner_pause

    held, release, threads = threading.Event(), threading.Event(), []
    real = owner_pause.install_fence

    def fence_landed(*args, **kwargs):
        landed = real(*args, **kwargs)

        def holder():
            with owner_pause.launch_lock(host.root, 'held'):
                held.set()
                release.wait(10)

        threads.append(threading.Thread(target=holder, daemon=True))
        threads[0].start()
        assert held.wait(5), 'the launch lock is held at the latch boundary'
        return landed

    with monkeypatch.context() as boundary:
        boundary.setattr(owner_pause, 'install_fence', fence_landed)
        try:
            return request_owner_pause(task_id, request_id=request_id)
        finally:
            release.set()
            for thread in threads:
                thread.join(10)


@pytest.mark.parametrize('reason', ['owner', 'budget'])
def test_unconsumed_worker_death_preserves_the_same_pause_binding(host, tmp_path, monkeypatch, reason):  # noqa: F811
    _, prior = _paused_fence(host, tmp_path, monkeypatch, reason)
    _resume(host, monkeypatch, consume=False)
    _die_before_consumption(monkeypatch)
    assert queue.ACCEPTANCE_FENCES['held']['token'] == prior['token']
    assert len(host.pending) == 1 and host.pending[0]['_attempt'] == 2
    binding = copy.deepcopy(queue.ACCEPTANCE_FENCES['held'].get('paused_continuation'))
    assert queue.resume_budget_paused_task('held')['ok']
    # The re-park binds an owner Pause's latch to its durable owner fence (#1730), so one owner
    # Resume dispatches it exactly like a budget pause; nothing new is admitted.
    assert queue.ACCEPTANCE_FENCES['held'].get('paused_continuation') == binding
    ctx = _resume(host, monkeypatch, pid=3333)
    assert loop._begin_task_acceptance_fence(ctx, 'held')[0]
    assert queue.ACCEPTANCE_FENCES['held']['pooled_owner']['pid'] == 3333


def _resumes_once_on(host, monkeypatch, current):  # noqa: F811
    """Nothing runs under the Pause; one ordinary Resume lifts its latch and dispatches."""
    from tests.test_project_hold_recovery import worker as free_worker

    sent = free_worker(host, monkeypatch)
    workers.assign_tasks()
    assert not sent, 'the Pause still holds the re-parked root'
    assert queue.resume_budget_paused_task('held')['ok']
    assert 'held' not in queue.BUDGET_ROOT_FENCES
    workers.assign_tasks()
    row = budget_pause.budget_pause_row(host.root, 'held')
    assert [t['id'] for t in sent] == ['held'], row
    assert row['state'] == budget_pause.STATE_RESUME_GRANTED and not row['grant'].get('revoked_at')
    assert row['grant']['owner_pause_fence_id'] == current


def test_a_crash_repark_latches_the_current_owner_fence(host, tmp_path, monkeypatch):  # noqa: F811
    """Pause F1 -> Resume G1 dispatched unconsumed -> newer Pause F2 -> death -> re-park.

    The latch names the CURRENT durable fence, which the next Resume binds and lifts.
    """
    from ouroboros import owner_pause
    from supervisor.owner_pause_control import request_owner_pause

    _paused_fence(host, tmp_path, monkeypatch, 'owner')
    first = owner_pause.read_fence(host.root, 'held')['fence_id']
    _resume(host, monkeypatch, consume=False)
    newer = request_owner_pause('held', request_id='newer-pause')
    assert newer['ok'] and not newer['duplicate'] and newer['fence_id'] != first
    current = owner_pause.read_fence(host.root, 'held')['fence_id']
    _die_before_consumption(monkeypatch)
    latch = queue.BUDGET_ROOT_FENCES.get('held') or {}
    print('crash re-park latch witness:', {'first': first, 'current': current, 'latch': latch})
    assert latch.get('fence_id') == current and latch.get('cause') == 'owner_pause', latch
    _resumes_once_on(host, monkeypatch, current)


@pytest.mark.parametrize('pause', ['same', 'newer_latched', 'newer_latch_pending'])
def test_an_unknown_current_fence_defers_the_crash_repark_to_its_retained_replay(
        host, tmp_path, monkeypatch, pause):  # noqa: F811
    """Pause F1 -> Resume G1 dispatched unconsumed -> [newer Pause F2, its latch standing or
    ``latch_pending``] -> death. The re-park's read of the current fence fails: no fence is
    projected (never F1, the one the row parked under), the reaper retains its job and every
    launch stays refused. Once the read recovers, the health tick's replay latches the current
    fence and one ordinary Resume dispatches the root.
    """
    import inspect
    from types import SimpleNamespace

    from ouroboros import owner_pause
    from supervisor import task_reaper
    from supervisor.owner_pause_control import request_owner_pause
    from tests.test_retry_pause_stop_transfer import _Drained, _ReapPump

    _paused_fence(host, tmp_path, monkeypatch, 'owner')
    first = owner_pause.read_fence(host.root, 'held')['fence_id']
    _resume(host, monkeypatch, consume=False)
    newer = {}
    if pause != 'same':
        newer = (request_owner_pause('held', request_id='newer') if pause == 'newer_latched'
                 else _pause_with_latch_pending(host, monkeypatch, 'held', 'newer'))
        assert newer['ok'] and not newer['duplicate'] and newer['fence_id'] != first, newer
        assert bool(newer.get('latch_pending')) is (pause == 'newer_latch_pending'), newer
    current = owner_pause.read_fence(host.root, 'held')['fence_id']
    standing = (queue.BUDGET_ROOT_FENCES.get('held') or {}).get('fence_id')
    pump = _ReapPump()
    monkeypatch.setattr(task_reaper, 'reap_queue', pump)
    monkeypatch.setattr(task_reaper, '_deferred_reap_jobs', [])
    real, injected = owner_pause.read_fence, []

    def unreadable(*args, **kwargs):  # only the park's own read of the current fence
        if '_owner_park_fence_id' in {frame.function for frame in inspect.stack()}:
            injected.append(True)
            raise ValueError('owner_pause_authority_unreadable')
        return real(*args, **kwargs)

    def reaper_pass():
        with pytest.raises(_Drained):
            task_reaper.reaper_loop()

    with monkeypatch.context() as fault:
        fault.setattr(owner_pause, 'read_fence', unreadable)
        _dying_before_consumption(monkeypatch, pump)
        reaper_pass()
    gate = SimpleNamespace(task_id='held', root_task_id='held', budget_drive_root=host.root)
    refused = []
    try:
        with owner_pause.launch_admission(gate, wait=False):
            pass
    except owner_pause.OwnerPauseRefused as exc:
        refused.append(str(exc))
    facts = {'pause': pause, 'latch_pending': newer.get('latch_pending'), 'injected': len(injected),
             'first': first, 'current': current, 'standing': standing,
             'latch_while_unknown': (queue.BUDGET_ROOT_FENCES.get('held') or {}).get('fence_id'),
             'retained': len(task_reaper._deferred_reap_jobs), 'pending': [t['id'] for t in host.pending],
             'running': 'held' in workers.RUNNING, 'launch_refused': refused}
    print('unknown-fence crash re-park witness:', facts)
    assert facts['injected'] and facts['latch_while_unknown'] == standing, facts  # nothing projected
    assert facts['retained'] == 1 and facts['pending'] == [] and facts['running'], facts
    assert refused, facts  # the durable fence alone holds every launch
    task_reaper.retry_terminal_file_recoveries()  # the read recovered: the health tick hands it back
    reaper_pass()
    latch = queue.BUDGET_ROOT_FENCES.get('held') or {}
    assert not task_reaper._deferred_reap_jobs and [t['id'] for t in host.pending] == ['held'], facts
    assert latch.get('fence_id') == current and latch.get('cause') == 'owner_pause', (facts, latch)
    _resumes_once_on(host, monkeypatch, current)
