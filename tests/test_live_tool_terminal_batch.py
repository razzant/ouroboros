"""A control that ends a live call mid-batch leaves the batch's history honest, then propagates untouched.

Stop, Pause and accounting controls are terminal for the call. Effects that completed before them keep their exact
results in canonical history; the call they interrupted may have had its effect, and calls never entered say they did
not run. Sequential and parallel batches differ in what the remainder is. The ordinary loop keeps raising at once.
"""
import asyncio
import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor

import pytest

from ouroboros.model_wait import ModelWaitInterrupted
from ouroboros.usage_accounting import BudgetExceeded, UsageAccountingError
from tests._extension_loader_shared import _clear_loader_state  # noqa: F401
from tests._live_world import host_world as host_world
from tests.test_live_tool_continuity import invocation, is_request, saved, unit_at


def tool_row(host, call_id):
    rows = [row for row in host.messages if row.get('role') == 'tool' and row.get('tool_call_id') == call_id]
    assert len(rows) == 1, (call_id, rows)
    return rows[0]


def effects(monkeypatch, behaviours):
    """Each named tool does what ``behaviours`` says and records that it was entered."""
    from ouroboros.tools.registry_core import ToolRegistry
    entered = []
    lock = threading.Lock()

    def execute_result(registry, name, arguments):
        with lock:
            entered.append(name)
        return behaviours[name](registry)
    monkeypatch.setattr(ToolRegistry, 'execute_result', execute_result)
    return entered


def completed(text='Exact completed effect result.', **meta):
    from ouroboros.tools.tool_result import ToolResult

    def behaviour(registry):
        return ToolResult(status='ok', code='OK', text=text, meta=meta)
    return behaviour


def raising(control):
    def behaviour(registry):
        raise control
    return behaviour


@pytest.mark.parametrize('control', [ModelWaitInterrupted('owner_stop'), BudgetExceeded('Synthetic spend limit reached'),
                                     UsageAccountingError('Synthetic accounting authority failure')],
                         ids=['stop', 'budget', 'accounting'])
def test_a_control_after_a_completed_effect_keeps_it_and_names_the_rest(host_world, monkeypatch, control):
    """Sequential: the first effect's exact result, the interrupted call's unknown outcome, the third never entered."""
    host = host_world.host
    entered = effects(monkeypatch, {'write_file': completed(), 'edit_file': raising(control), 'run_shell': completed('Never.')})
    calls = [invocation('done', 'write_file', path='synthetic', content='done'),
             invocation('cut', 'edit_file', path='synthetic'), invocation('left', 'run_shell', cmd=['true'])]

    async def run():
        with pytest.raises(type(control)) as raised:
            await host.tool_batch(calls)
        assert raised.value is control, 'the original control object, never a substitute'
        await host.finish()

    asyncio.run(run())
    assert entered == ['write_file', 'edit_file'], 'no effect reran and nothing started after the control'
    assert tool_row(host, 'done')['content'].startswith('Exact completed effect result.')
    assert not json.loads(json.dumps(tool_row(host, 'done').get('_tool_result_record', {}))).get('is_error')
    interrupted = tool_row(host, 'cut')['content']
    assert 'may' in interrupted and 'did not run' not in interrupted.lower() and 'not started' not in interrupted.lower()
    assert 'NOT STARTED' in tool_row(host, 'left')['content']
    unit = host.messages[unit_at(host.messages, 'done')]
    assert [call['id'] for call in unit['tool_calls']] == ['done', 'cut', 'left'], 'exact identities in their order'
    assert unit['tool_calls'] == calls
    ids = [row['tool_call_id'] for row in host.messages if row.get('role') == 'tool']
    assert ids == ['done', 'cut', 'left'], 'adjacent to their unit, in request order'
    assert sum(is_request(row, 'left') for row in host.messages) == 1
    assert not any('did not run: the call ended first' in str(row.get('content')) for row in host.messages), (
        'a call the batch already answered gets no second, contradicting fact')
    assert set(host.executed) == {'done', 'cut', 'left'}
    checkpoint = saved(host)
    assert {'done', 'cut', 'left'} <= {row.get('tool_call_id') for row in checkpoint['messages']}, 'durable before the raise'


def test_a_host_failure_mid_batch_leaves_the_same_whole_history_and_stays_itself(host_world, monkeypatch):
    from ouroboros import loop_tool_execution as execution
    host = host_world.host
    failure = RuntimeError('Synthetic host failure')
    original = execution._execute_with_timeout

    def execute(tools, call, *args):
        if call['id'] == 'cut':
            raise failure
        return original(tools, call, *args)
    monkeypatch.setattr(execution, '_execute_with_timeout', execute)
    entered = effects(monkeypatch, {'write_file': completed(), 'run_shell': completed('Never.')})

    async def run():
        with pytest.raises(RuntimeError) as raised:
            await host.tool_batch([invocation('done', 'write_file', path='s', content='d'), invocation('cut', 'edit_file', path='s'),
                                   invocation('left', 'run_shell', cmd=['true'])])
        assert raised.value is failure
        await host.finish()

    asyncio.run(run())
    assert entered == ['write_file']
    assert tool_row(host, 'done')['content'].startswith('Exact completed effect result.')
    assert 'may' in tool_row(host, 'cut')['content'] and 'NOT STARTED' in tool_row(host, 'left')['content']


def test_a_control_keeps_an_effects_own_rows_and_metadata(host_world, monkeypatch):
    """What a completed effect produced beside its text still follows its unit."""
    host = host_world.host
    attached = {'role': 'user', 'content': 'An attachment the completed effect staged.', '_capture_source': 'synthetic'}

    def staging(registry):
        registry._ctx.messages.append(attached)
        return completed(marker='kept-metadata')(registry)
    effects(monkeypatch, {'write_file': staging, 'edit_file': raising(ModelWaitInterrupted('owner_pause'))})

    async def run():
        with pytest.raises(ModelWaitInterrupted):
            await host.tool_batch([invocation('done', 'write_file', path='s', content='d'), invocation('cut', 'edit_file', path='s')])
        await host.finish()

    asyncio.run(run())
    trace = next(row for row in host.trace['tool_calls'] if row['tool_call_id'] == 'done')
    assert trace['is_error'] is False and trace.get('tool_result_meta', {}).get('marker') == 'kept-metadata'
    last_tool = max(i for i, row in enumerate(host.messages) if row.get('role') == 'tool')
    assert any(row.get('content') == attached['content'] for row in host.messages[last_tool + 1:])


def test_a_control_starts_no_call_tool_of_the_same_batch(host_world, monkeypatch):
    host = host_world.host
    control = ModelWaitInterrupted('owner_stop')
    effects(monkeypatch, {'write_file': raising(control)})
    ran = []

    async def handler(host, arguments):
        ran.append(arguments)
        return 'Never.'
    from ouroboros.contracts.live_conversation import CallTool
    host.call_tools['call_side'] = CallTool('call_side', 'Synthetic.', {'type': 'object', 'properties': {}}, handler)

    async def run():
        with pytest.raises(ModelWaitInterrupted) as raised:
            await host.tool_batch([invocation('cut', 'write_file', path='s', content='d'), invocation('side', 'call_side')])
        assert raised.value is control
        await host.finish()

    asyncio.run(run())
    assert not ran and 'did not run' in tool_row(host, 'side')['content']
    assert [row['tool_call_id'] for row in host.messages if row.get('role') == 'tool'] == ['cut', 'side']


def test_a_publication_failure_never_replaces_the_control(host_world, monkeypatch):
    host = host_world.host
    control = ModelWaitInterrupted('owner_stop')
    effects(monkeypatch, {'write_file': raising(control)})

    def broken(*args, **kwargs):
        raise OSError('Synthetic checkpoint failure')
    monkeypatch.setattr(host, 'checkpoint', lambda boundary: broken() if boundary == 'post_batch' else None)

    async def run():
        with pytest.raises(ModelWaitInterrupted) as raised:
            await host.tool_batch([invocation('cut', 'write_file', path='s', content='d')])
        assert raised.value is control
    asyncio.run(run())


def test_without_a_control_a_publication_failure_stays_itself(host_world, monkeypatch):
    host = host_world.host
    effects(monkeypatch, {'write_file': completed()})

    def broken(*args, **kwargs):
        raise OSError('Synthetic checkpoint failure')
    monkeypatch.setattr(host, 'checkpoint', lambda boundary: broken() if boundary == 'post_batch' else None)

    async def run():
        with pytest.raises(OSError, match='Synthetic checkpoint failure'):
            await host.tool_batch([invocation('done', 'write_file', path='s', content='d')])
    asyncio.run(run())


# --- parallel batches: what a sibling did is known only after the pool has settled ----------------------


class DeterministicPool:
    """Runs the batch's first ``started`` submissions at once; the rest stay queued until a shutdown cancels them.
    A call's own one-worker pool stays real."""

    def __init__(self, started):
        self.started, self.queued, self.shutdowns = started, [], []

    def __call__(self, max_workers):
        return self if max_workers > 1 else ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, function, *args):
        future = Future()
        if self.started:
            self.started -= 1
            try:
                future.set_result(function(*args))
            except BaseException as exc:  # a real worker hands any failure to its future
                future.set_exception(exc)
        else:
            self.queued.append(future)
        return future

    def shutdown(self, wait, cancel_futures):
        self.shutdowns.append((wait, cancel_futures))
        if cancel_futures:
            for future in self.queued:
                future.cancel()


def parallel(monkeypatch):
    """The whole batch runs as the legacy pool, under its own rule for an escaped exception."""
    from ouroboros import loop_tool_execution as execution
    monkeypatch.setattr(execution, 'legacy_pool_response', lambda names: True)
    return execution


def test_a_parallel_batch_keeps_finished_siblings_and_says_what_never_began(host_world, monkeypatch):
    host = host_world.host
    execution = parallel(monkeypatch)
    control, other = ModelWaitInterrupted('owner_stop'), BudgetExceeded('Synthetic spend limit reached')
    pool = DeterministicPool(4)
    monkeypatch.setattr(execution, 'ThreadPoolExecutor', pool)

    original = execution._execute_with_timeout

    def execute(tools, call, *args):
        if call['id'] == 'broken':  # outside the tool's own error conversion: the batch's generic path
            raise RuntimeError('Synthetic worker failure')
        return original(tools, call, *args)
    monkeypatch.setattr(execution, '_execute_with_timeout', execute)
    entered = effects(monkeypatch, {'read_file': completed('First sibling.'), 'list_available_tools': raising(control),
                                    'get_task_result': raising(other), 'search_code': completed('Never.')})
    calls = [invocation('done', 'read_file', path='s'), invocation('cut', 'list_available_tools'),
             invocation('cut-too', 'get_task_result', task_id='s'), invocation('broken', 'search_code', query='s'),
             invocation('left', 'search_code', query='t')]

    async def run():
        with pytest.raises((ModelWaitInterrupted, BudgetExceeded)) as raised:
            await host.tool_batch(calls)
        assert raised.value in (control, other), 'one original control, whichever the batch saw first'
        await host.finish()

    asyncio.run(run())
    assert sorted(entered) == ['get_task_result', 'list_available_tools', 'read_file']
    assert pool.shutdowns == [(True, True)], 'started siblings are joined and queued ones cancelled, as before'
    assert tool_row(host, 'done')['content'].startswith('First sibling.')
    for key in ('cut', 'cut-too'):
        assert 'may' in tool_row(host, key)['content'] and 'NOT STARTED' not in tool_row(host, key)['content']
    assert 'Synthetic worker failure' in tool_row(host, 'broken')['content']
    assert 'NOT STARTED' in tool_row(host, 'left')['content']
    assert [row['tool_call_id'] for row in host.messages if row.get('role') == 'tool'] == [call['id'] for call in calls]


def test_a_real_pool_lets_running_siblings_finish_and_records_them(host_world, monkeypatch):
    """A real worker that was already inside its effect finishes it; its exact result is what history shows."""
    host = host_world.host
    parallel(monkeypatch)
    raised_control = threading.Event()
    control = ModelWaitInterrupted('owner_stop')

    def interrupt(registry):
        raised_control.set()
        raise control

    def sibling(registry):
        assert raised_control.wait(10)
        return completed('Sibling finished after the control.')(registry)
    effects(monkeypatch, {'read_file': sibling, 'list_available_tools': interrupt})

    async def run():
        with pytest.raises(ModelWaitInterrupted) as raised:
            await host.tool_batch([invocation('done', 'read_file', path='s'), invocation('cut', 'list_available_tools')])
        assert raised.value is control
        await host.finish()

    asyncio.run(run())
    assert tool_row(host, 'done')['content'].startswith('Sibling finished after the control.')
    assert 'may' in tool_row(host, 'cut')['content']


# --- the ordinary loop keeps its semantics ------------------------------------------------------------


def ordinary_batch(host, calls):
    from ouroboros.loop_tool_execution import execute_tool_calls
    return execute_tool_calls(calls, host.tools.copy_for(host.ctx), host.ctx.drive_root / 'logs', host.ctx.task_id,
                              host.executor, host.trace)


def test_the_ordinary_sequential_loop_still_raises_at_once_and_starts_nothing_after(host_world, monkeypatch):
    host = host_world.host
    control = ModelWaitInterrupted('owner_stop')
    entered = effects(monkeypatch, {'write_file': completed(), 'edit_file': raising(control), 'run_shell': completed('Never.')})
    with pytest.raises(ModelWaitInterrupted) as raised:
        ordinary_batch(host, [invocation('done', 'write_file', path='s', content='d'), invocation('cut', 'edit_file', path='s'),
                              invocation('left', 'run_shell', cmd=['true'])])
    assert raised.value is control and entered == ['write_file', 'edit_file']


def test_the_ordinary_parallel_loop_still_raises_after_joining_its_siblings(host_world, monkeypatch):
    host = host_world.host
    execution = parallel(monkeypatch)
    pool = DeterministicPool(2)
    monkeypatch.setattr(execution, 'ThreadPoolExecutor', pool)
    control = ModelWaitInterrupted('owner_stop')
    effects(monkeypatch, {'read_file': completed('Sibling.'), 'list_available_tools': raising(control), 'search_code': completed('Never.')})
    with pytest.raises(ModelWaitInterrupted) as raised:
        ordinary_batch(host, [invocation('done', 'read_file', path='s'), invocation('cut', 'list_available_tools'),
                              invocation('left', 'search_code', query='s')])
    assert raised.value is control and pool.shutdowns == [(True, True)]


def test_an_ordinary_parallel_batch_without_control_returns_results_in_request_order(host_world, monkeypatch):
    host = host_world.host
    execution = parallel(monkeypatch)

    def broken(tools, call, *args):
        if call['id'] == 'broken':
            raise RuntimeError('Synthetic worker failure')
        return original(tools, call, *args)
    original = execution._execute_with_timeout
    monkeypatch.setattr(execution, '_execute_with_timeout', broken)
    effects(monkeypatch, {'read_file': completed('First.'), 'search_code': completed('Second.')})
    results = ordinary_batch(host, [invocation('first', 'read_file', path='s'), invocation('broken', 'search_code', query='s'),
                                    invocation('second', 'search_code', query='t')])
    assert [row['tool_call_id'] for row in results] == ['first', 'broken', 'second']
    assert results[0]['result'] == 'First.' and results[2]['result'] == 'Second.'
    assert results[1]['is_error'] and results[1]['tool_result'].code == 'EXECUTOR_ERROR'
    assert 'TOOL_ERROR' in results[1]['result'] and 'Synthetic worker failure' in results[1]['result']


# --- groups: consecutive calls of the together class run at once, everything else alone, in order -------------------


def batch_of(host, calls):
    from ouroboros.loop_tool_execution import execute_tool_batch
    return execute_tool_batch(calls, host.tools.copy_for(host.ctx), host.ctx.drive_root / 'logs', host.ctx.task_id,
                              host.executor, host.trace)


def test_a_group_runs_at_once_and_a_single_call_starts_only_after_everything_before_it(host_world, monkeypatch):
    host = host_world.host
    barrier, lock, log = threading.Barrier(2, timeout=10), threading.Lock(), []

    def alone(text):
        def behaviour(registry):
            with lock:
                log.append(('start', text))
            with lock:
                log.append(('end', text))
            return completed(text)(registry)
        return behaviour

    def together(text):
        def behaviour(registry):
            with lock:
                log.append(('start', text))
            barrier.wait()  # only a partner running at the same time releases it
            with lock:
                log.append(('end', text))
            return completed(text)(registry)
        return behaviour
    effects(monkeypatch, {'edit_text': alone('Edited.'), 'read_file': together('Read.'),
                          'knowledge_read': together('Note.'), 'write_file': alone('Written.')})
    calls = [invocation('edit', 'edit_text', path='s'), invocation('read', 'read_file', path='s'),
             invocation('note', 'knowledge_read', topic='t'), invocation('write', 'write_file', path='s', content='c')]
    results, control = batch_of(host, calls)
    assert control is None
    assert [row['tool_call_id'] for row in results] == ['edit', 'read', 'note', 'write'], 'request order'
    assert [row['result'] for row in results] == ['Edited.', 'Read.', 'Note.', 'Written.']
    assert log[:2] == [('start', 'Edited.'), ('end', 'Edited.')], 'the group starts after the edit returned'
    assert {entry[0] for entry in log[2:4]} == {'start'} and {entry[0] for entry in log[4:6]} == {'end'}, (
        'both members were inside their handlers at once')
    assert log[6:] == [('start', 'Written.'), ('end', 'Written.')], 'the write starts after the group ended'


@pytest.mark.parametrize('control', [ModelWaitInterrupted('owner_stop'), BudgetExceeded('Synthetic spend limit reached')],
                         ids=['stop', 'budget'])
def test_a_control_in_a_group_lets_started_members_finish_and_starts_nothing_after(host_world, monkeypatch, control):
    host = host_world.host
    raised_control = threading.Event()

    def interrupt(registry):
        raised_control.set()
        raise control

    def sibling(registry):
        assert raised_control.wait(10)
        return completed('Member finished after the control.')(registry)
    entered = effects(monkeypatch, {'read_file': sibling, 'knowledge_read': interrupt, 'write_file': completed('Never.')})
    calls = [invocation('done', 'read_file', path='s'), invocation('cut', 'knowledge_read', topic='t'),
             invocation('left', 'write_file', path='s', content='c')]
    results, raised = batch_of(host, calls)
    assert raised is control and sorted(entered) == ['knowledge_read', 'read_file'], 'the next segment never began'
    rows = {row['tool_call_id']: row['result'] for row in results}
    assert rows['done'] == 'Member finished after the control.', 'a started member keeps its exact result'
    assert 'may' in rows['cut'] and 'NOT STARTED' not in rows['cut']
    assert 'NOT STARTED' in rows['left']
    with pytest.raises(type(control)):
        ordinary_batch(host, calls)


@pytest.mark.parametrize('shape', ['mixed', 'older-reads', 'old-new', 'new-new'])
def test_an_exception_escaping_a_group_member_keeps_the_rule_the_response_had(host_world, monkeypatch, shape):
    """A response of only the older reads keeps the pool's row; any other response, a group of newer names included,
    ends where the exception escaped, its started members still settle and the segment after it never begins."""
    from ouroboros import loop_tool_execution as execution
    host = host_world.host
    failure = RuntimeError('Synthetic staging failure')
    original = execution._execute_with_timeout

    def execute(tools, call, *args):
        if call['id'] == 'broken':
            raise failure
        return original(tools, call, *args)
    monkeypatch.setattr(execution, '_execute_with_timeout', execute)
    first, member, last = {'mixed': ('read_file', 'knowledge_read', 'write_file'),
                           'older-reads': ('read_file', 'search_code', 'search_code'),
                           'old-new': ('read_file', 'knowledge_read', None), 'new-new': ('tree_read', 'knowledge_read', None)}[shape]
    entered = effects(monkeypatch, {first: completed('Read.'), member: completed('Unused.'), last: completed('Last.')})
    calls = [invocation('read', first, path='s'), invocation('broken', member, query='s'),
             *([invocation('last', last, path='s')] if last else [])]
    results, raised = batch_of(host, calls)
    rows = {row['tool_call_id']: row for row in results}
    assert rows['read']['result'] == 'Read.'
    if shape == 'older-reads':
        assert execution.call_segments([call['function']['name'] for call in calls]) == [[0, 1, 2]]
        assert raised is None and rows['last']['result'] == 'Last.'
        assert rows['broken']['tool_result'].code == 'EXECUTOR_ERROR' and 'Synthetic staging failure' in rows['broken']['result']
        return
    assert raised is failure and 'may' in rows['broken']['result'], 'the escaped exception ends the response, as before'
    if shape != 'mixed':
        assert execution.call_segments([call['function']['name'] for call in calls]) == [[0, 1]]
        return
    assert execution.call_segments([call['function']['name'] for call in calls]) == [[0, 1], [2]]
    assert 'write_file' not in entered and 'NOT STARTED' in rows['last']['result'], 'the next segment never began'


def test_of_several_endings_in_one_group_the_first_by_request_index_ends_the_response(host_world, monkeypatch):
    """The earlier call's budget refusal is what the loop above reads, as one call at a time gave it, even when a later
    member's staging failure settled first; the started members settle and the write never begins."""
    from ouroboros import loop_tool_execution as execution
    host = host_world.host
    budget, failure, failed = BudgetExceeded('Synthetic spend limit reached'), RuntimeError('Synthetic staging'), threading.Event()
    original = execution._execute_with_timeout

    def execute(tools, call, *args):
        if call['id'] == 'later':
            failed.set()
            raise failure
        return original(tools, call, *args)
    monkeypatch.setattr(execution, '_execute_with_timeout', execute)

    def refused(registry):
        assert failed.wait(10)
        raise budget
    entered = effects(monkeypatch, {'read_file': refused, 'knowledge_read': completed(), 'write_file': completed('Never.')})
    calls = [invocation('first', 'read_file', path='s'), invocation('later', 'knowledge_read', topic='t'),
             invocation('write', 'write_file', path='s', content='c')]
    results, raised = batch_of(host, calls)
    assert raised is budget, 'first by request index, not by completion'
    rows = {row['tool_call_id']: row['result'] for row in results}
    assert 'may' in rows['first'] and 'may' in rows['later'] and 'NOT STARTED' in rows['write']
    assert 'write_file' not in entered


def test_a_group_whose_thread_cannot_start_runs_the_rest_alone_and_keeps_every_result(host_world, monkeypatch):
    """The second submit fails as on a PID-limited container, after the pool queued its item: every member runs once,
    the queued item by whichever takes it first, the rest alone in order; no result is lost."""
    from ouroboros import loop_tool_execution as execution
    host = host_world.host
    submits = []

    class NoSecondThread(ThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            future = super().submit(*args, **kwargs)
            if self._max_workers > 1:
                submits.append(future)
                if len(submits) == 2:
                    raise RuntimeError("can't start new thread")
            return future
    monkeypatch.setattr(execution, 'ThreadPoolExecutor', NoSecondThread)
    entered = effects(monkeypatch, {'read_file': completed('Read.'), 'knowledge_read': completed('Note.'),
                                    'tree_read': completed('Tree.'), 'write_file': completed('Written.')})
    calls = [invocation('read', 'read_file', path='s'), invocation('note', 'knowledge_read', topic='t'),
             invocation('tree', 'tree_read'), invocation('write', 'write_file', path='s', content='c')]
    results, raised = batch_of(host, calls)
    assert raised is None and len(submits) == 2
    assert [row['result'] for row in results] == ['Read.', 'Note.', 'Tree.', 'Written.']
    assert sorted(entered) == ['knowledge_read', 'read_file', 'tree_read', 'write_file'], 'each call entered once'


def test_a_delivery_confirmed_while_a_group_runs_rides_one_row(host_world, monkeypatch):
    """A connector confirms an earlier send of this conversation while two reads run: one row carries it, the first
    member by request index, as one call at a time would have."""
    from ouroboros import presence_context
    host = host_world.host
    confirmed = threading.Event()
    monkeypatch.setattr(presence_context, 'confirmed_delivery_receipts',
                        lambda ctx: frozenset({('earlier-send', '0')}) if confirmed.is_set() else frozenset())

    def confirming(registry):
        confirmed.set()
        return completed('Mapped.')(registry)
    effects(monkeypatch, {'read_file': completed('Read.'), 'memory_map': confirming})
    results, raised = batch_of(host, [invocation('read', 'read_file', path='s'), invocation('map', 'memory_map')])
    assert raised is None
    carried = [row['tool_call_id'] for row in results if row.get('result_meta', {}).get('presence_delivery_confirmed')]
    assert carried == ['read'], carried
    assert results[0]['result_meta']['presence_delivery_confirmed'] == [('earlier-send', '0')]
