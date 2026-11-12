"""A live call stays a conversation while one of its ordinary tools runs.

The tool's request is a host fact before its effect; history (captions, mail, facts,
adopted connections) stays writable while the effect runs; the completed call/result
unit lands whole and reaches the same conversation once, natively only on the
conversation that issued it. Two layers here: the real ConversationHost with a held
effect and the real driver over its fake port. ``test_live_tool_continuity_stack.py``
holds the third: the real stack with native wires.
"""
import asyncio
import json
import threading
import time

import pytest

from ouroboros import call_protocol as cp
from ouroboros.contracts.live_conversation import ContextUpdate, ToolOutputs
from ouroboros.contracts.realtime import RealtimeSetupError
from tests._extension_loader_shared import _clear_loader_state  # noqa: F401
from tests._live_world import host_world as host_world
from tests.test_live_conversation_driver import Batch, Host, Session, started, until

POINTER = 'The requested tool batch returned'


class Held:
    """One tool name held inside its effect on whichever registry executes it; the rest run normally."""

    def __init__(self, monkeypatch, name='list_available_tools', text='Slow work finished exactly once.', *, block=True):
        from ouroboros.tools.registry_core import ToolRegistry
        from ouroboros.tools.tool_result import ToolResult
        self.entered, self.release, self.effects = threading.Event(), threading.Event(), []
        if not block:
            self.release.set()
        original = ToolRegistry.execute_result

        def execute_result(registry, tool, args):
            if tool != name:
                return original(registry, tool, args)
            self.effects.append(tool)
            self.entered.set()
            assert self.release.wait(60), 'the scenario never released the effect'
            return ToolResult(status='ok', code='OK', text=text)
        monkeypatch.setattr(ToolRegistry, 'execute_result', execute_result)

    async def started(self, timeout=20):
        deadline = time.monotonic() + timeout
        while not self.entered.is_set():
            assert time.monotonic() < deadline, 'the effect never started'
            await asyncio.sleep(.01)


async def reached(predicate):
    """A gate inside the call's own loop, for forcing one real order of its steps."""
    deadline = time.monotonic() + 30
    while not predicate():
        assert time.monotonic() < deadline, 'the forced order never arrived'
        await asyncio.sleep(.001)


def invocation(call_id, name, **arguments):
    return {'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments)}}


def is_request(row, call_id):
    """A request's host fact, canonical or as fitted (the wire drops private row keys)."""
    content = str(row.get('content'))
    return 'you requested' in content and f'(call {call_id})' in content and not row.get('tool_calls')


def unit_at(messages, call_id):
    return next(i for i, row in enumerate(messages)
                if any(call.get('id') == call_id for call in row.get('tool_calls') or ()))


def saved(host):
    from ouroboros.working_checkpoint import read_working  # the whole state, also from the store-once wrapper
    return read_working(host.ctx.drive_root, host.ctx.task_id, 1)[1]


# --- the real host: history stays writable while an effect runs ------------------------------------


def test_a_running_effect_keeps_history_writable_and_lands_as_one_unit(host_world, monkeypatch):
    import copy
    from ouroboros.context_compaction import _atomic_units
    from ouroboros.working_checkpoint import close_unanswered_calls
    host, slow = host_world.host, Held(monkeypatch)

    async def run():
        batch = asyncio.create_task(host.tool_batch([invocation('slow-1', 'list_available_tools')]))
        await slow.started()
        assert is_request(host.messages[-1], 'slow-1'), 'the request is a host fact before its effect'
        assert not any(row.get('tool_calls') for row in host.messages), 'no call dangles while it runs'
        pre = saved(host)
        assert pre['working']['boundary'] == 'pre_effect' and is_request(pre['messages'][-1], 'slow-1')
        recovered = copy.deepcopy(pre['messages'])
        assert close_unanswered_calls(recovered, 'worker_crash') == [] and recovered == pre['messages'], (
            'recovery finds a request whose outcome is unknown, never a call to close or repeat')
        assert 'outcome is unknown' in pre['messages'][-1]['content']
        await asyncio.wait_for(host.caption('owner', 'Owner words while it runs.', 'owner-1'), 10)
        await asyncio.wait_for(host.caption('ouroboros', 'Spoken meanwhile.', 'spoken-1', completed=True), 10)
        await asyncio.wait_for(host.caption('ouroboros', 'Spoken meanwhile, revised.', 'spoken-1',
                                            completed=True, replace=True), 10)
        assert not batch.done() and slow.effects == ['list_available_tools']
        slow.release.set()
        return await asyncio.wait_for(batch, 30)
    rows = asyncio.run(run())
    messages = host.messages
    at = unit_at(messages, 'slow-1')
    request = next(i for i, row in enumerate(messages) if is_request(row, 'slow-1'))
    words = next(i for i, row in enumerate(messages) if row.get('content') == 'Owner words while it runs.')
    spoken = next(i for i, row in enumerate(messages) if str(row.get('content')).endswith('Spoken meanwhile, revised.'))
    assert request < words < spoken < at and messages[at + 1].get('tool_call_id') == 'slow-1'
    assert any(unit.start == at and unit.end == at + 1 for unit in _atomic_units(messages)), 'a complete unit'
    assert [row.get('tool_call_id') for row in rows] == ['slow-1'], 'captions written meanwhile are not changes'
    assert slow.effects == ['list_available_tools'] and 'slow-1' in host.executed
    post = saved(host)
    assert post['working']['boundary'] == 'post_batch'
    assert post['messages'][unit_at(post['messages'], 'slow-1') + 1].get('tool_call_id') == 'slow-1'


def test_call_ids_decide_replay_and_each_request_is_recorded_once(host_world, monkeypatch):
    host, work = host_world.host, Held(monkeypatch, block=False)
    first = invocation('same-1', 'list_available_tools')

    async def run():
        await host.tool_requested(first)  # recorded on arrival, before its batch
        await host.tool_requested(first)  # a repeated arrival records nothing
        one = await host.tool_batch([first])
        replay = await host.tool_batch([first])
        with pytest.raises(RealtimeSetupError, match='tool_identity'):
            await host.tool_batch([invocation('same-1', 'list_available_tools', namespace='builtin')])
        again = await host.tool_batch([invocation('same-2', 'list_available_tools')])
        return one, replay, again
    one, replay, again = asyncio.run(run())
    assert work.effects == ['list_available_tools'] * 2, 'the same id replays; a new id with equal arguments runs'
    assert list(replay) == list(one) and [row['tool_call_id'] for row in again] == ['same-2']
    requests = [row for row in host.messages if row.get('_host_context_kind') == 'call_tool_request']
    assert [(is_request(row, 'same-1'), is_request(row, 'same-2')) for row in requests] == [(True, False), (False, True)]


def test_each_tool_round_is_the_step_a_reissue_is_read_against(host_world):
    """A live call has no model call per tool round, so each round names its own step on the context
    (``_live_tool_step``, what Safety reads a returned call's re-issue against); the call meta stays as it was."""
    host, seen = host_world.host, []

    def handler(ctx, **_):
        seen.append((dict(ctx._current_llm_call_meta), ctx._live_tool_step))
        return 'read.'
    host.tools.override_handler('get_task_result', handler)
    host.ctx._current_llm_call_meta, start = {'origin': 'kept'}, host.rounds

    async def run():
        for call_id in ('round-a', 'round-b'):
            await host.tool_batch([invocation(call_id, 'get_task_result', task_id='none')])
    asyncio.run(run())
    rounds = [f'{host.ctx.task_id}:call-tools:{start + n}' for n in (1, 2)]
    assert seen == [({'origin': 'kept'}, step) for step in rounds]


def test_an_effect_s_loaded_tools_and_image_follow_its_unit_while_history_changes(host_world, monkeypatch, tmp_path):
    import ouroboros.safety as safety
    import ouroboros.tools.vision as vision
    from tests.test_live_image_delivery import pixels
    from ouroboros.context_compaction import _atomic_units
    host, slow = host_world.host, Held(monkeypatch)
    uploads = tmp_path / 'isolated_uploads'
    uploads.mkdir()
    (uploads / 'seen.png').write_bytes(pixels())
    monkeypatch.setattr(safety, 'check_safety', lambda *_a, **_k: (True, ''))
    monkeypatch.setattr(vision, '_allowed_file_roots', lambda *_a, **_k: [uploads])
    assert 'read_file' not in {schema['function']['name'] for schema in host.schemas}
    calls = [invocation('load', 'enable_tools', tools='read_file'),
             invocation('look', 'view_image', path=str(uploads / 'seen.png')), invocation('slow', 'list_available_tools')]

    async def run():
        batch = asyncio.create_task(host.tool_batch(calls))
        await slow.started()
        assert 'read_file' not in {schema['function']['name'] for schema in host.schemas}, 'staged until its unit'
        assert not any(isinstance(row.get('content'), list) and 'seen.png' in json.dumps(row) for row in host.messages)
        await asyncio.wait_for(host.caption('owner', 'Words during the load.', 'owner-2'), 10)
        host.schemas[:] = [*host.schemas]  # an adopted connection publishes its own (equal) list
        host.schemas.append({'type': 'function', 'function': {'name': 'adopted_extra', 'parameters': {}}})
        slow.release.set()
        await asyncio.wait_for(batch, 30)
        loaded = await host.tool_batch([invocation('later', 'enable_tools', tools='search_code')])
        return loaded
    later = asyncio.run(run())
    names = [schema['function']['name'] for schema in host.schemas]
    assert {'read_file', 'adopted_extra', 'search_code'} <= set(names) and len(names) == len(set(names))
    assert 'read_file' in host.ctx._actor_loaded_tool_names
    assert 'loaded' in next(row['content'] for row in later if row.get('tool_call_id') == 'later').lower()
    messages = host.messages
    at = unit_at(messages, 'slow')
    assert [messages[at + n].get('tool_call_id') for n in (1, 2, 3)] == ['load', 'look', 'slow']
    assert any(unit.start == at and unit.end == at + 3 for unit in _atomic_units(messages))
    image = next(i for i, row in enumerate(messages) if isinstance(row.get('content'), list) and 'seen.png' in json.dumps(row))
    words = next(i for i, row in enumerate(messages) if row.get('content') == 'Words during the load.')
    assert words < at < image, 'the image the effect attached follows its unit'


def test_an_authored_view_waits_for_the_running_effect(host_world, monkeypatch):
    host, slow = host_world.host, Held(monkeypatch)
    host.ctx._pending_compaction = {'synthetic': 'pending authored view'}
    applied = []
    monkeypatch.setattr('ouroboros.live_context_view.apply_context_view', lambda h: applied.append(True) or False)

    async def run():
        assert host.context_view_pending()
        batch = asyncio.create_task(host.tool_batch([invocation('slow-view', 'list_available_tools')]))
        await slow.started()
        assert not host.context_view_pending() and not await host.apply_context_view()
        assert not applied, 'an authored view never rewrites history under a running effect'
        slow.release.set()
        await asyncio.wait_for(batch, 30)
        assert host.context_view_pending()
        await host.apply_context_view()
    asyncio.run(run())
    assert applied == [True]
    host.ctx._pending_compaction = None


def test_a_requested_call_that_never_started_is_recorded_as_not_run(host_world):
    host = host_world.host

    async def run():
        await host.tool_requested(invocation('queued-1', 'list_available_tools'))
        await host.finish()
    asyncio.run(run())
    request = next(i for i, row in enumerate(host.messages) if is_request(row, 'queued-1'))
    assert 'did not run' in host.messages[-1]['content'] and request < len(host.messages) - 1
    assert not any(row.get('tool_calls') for row in host.messages)
    assert any('did not run' in str(row.get('content')) for row in saved(host)['messages'])


def test_a_copied_catalog_keeps_its_own_discovery_binding(host_world):
    from ouroboros.tools.tool_discovery import bind_resident_schemas
    host = host_world.host
    lane, staged = host.tools.copy_for(host.ctx), []
    bind_resident_schemas(lane, staged)
    assert lane._entries['enable_tools'].handler is not host.tools._entries['enable_tools'].handler
    lane.execute_result('enable_tools', {'tools': 'read_file'})
    assert [schema['function']['name'] for schema in staged] == ['read_file']
    assert 'read_file' not in {schema['function']['name'] for schema in host.schemas}


# --- the driver: routing, native resume, expiry and hearing gaps ------------------------------------


def held_batch(host, *, silent=False):
    gate, running, batches = asyncio.Event(), asyncio.Event(), []

    async def batch(calls):
        batches.append([call['id'] for call in calls])
        running.set()
        await gate.wait()
        return Batch([{'role': 'tool', 'tool_call_id': call['id'], 'content': 'Done.'} for call in calls],
                     silent=[call['id'] for call in calls] if silent else ())
    host.tool_batch = batch
    return gate, running, batches


@pytest.mark.parametrize('handle_after_call', [False, True])
def test_native_resume_needs_a_handle_published_after_the_open_call(handle_after_call):
    async def scenario():
        host = Host()
        gate, running, _ = held_batch(host)
        task, first = await started(host)
        try:
            first.emit('ready', input_rate=24000, output_rate=24000)
            if not handle_after_call:
                first.emit('resumable', handle='before-call')
            first.emit('tool_call', call_id='work', name='slow_tool', arguments='{}')
            await running.wait()
            if handle_after_call:
                first.emit('resumable', handle='after-call')
            first.emit('closed', resumable=True, reason='transport_lost')
            await until(lambda: len(host.sessions) == 2)
            assert host.options[1].resume_handle == ('after-call' if handle_after_call else '')
            second = host.sessions[1]
            second.emit('ready', input_rate=24000, output_rate=24000)
            gate.set()
            if handle_after_call:  # the resumed conversation holds the call and receives its output
                await until(lambda: any(isinstance(x, ToolOutputs) for x in second.sent))
            else:  # a conversation that never issued the call gets the unit through a fresh setup
                await until(lambda: len(host.sessions) == 3)
                assert not any(isinstance(x, ToolOutputs) for session in host.sessions for x in session.sent)
                host.sessions[2].emit('ready', input_rate=24000, output_rate=24000)
                await until(lambda: any(POINTER in str(x) for x in host.sessions[2].sent))
        finally:
            gate.set()
            host.hub.ended_by_owner = True
            if not task.done(): await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


def test_a_renewal_does_not_wait_for_a_running_batch_and_queued_calls_keep_order():
    async def scenario():
        host = Host()
        gate, running, batches = held_batch(host)
        task, first = await started(host)
        try:
            first.emit('ready', input_rate=24000, output_rate=24000)
            first.emit('tool_call', call_id='one', name='work', arguments='{}')
            await running.wait()
            first.emit('tool_call', call_id='two', name='work', arguments='{}')
            first.emit('renew', within_s=2, resume=False, reason='audio_span')
            await until(lambda: len(host.sessions) == 2)
            assert host.options[1].resume_handle == '' and batches == [['one']]
            host.sessions[1].emit('ready', input_rate=24000, output_rate=24000)
            gate.set()
            await until(lambda: batches == [['one'], ['two']])
        finally:
            gate.set()
            host.hub.ended_by_owner = True
            if not task.done(): await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


def test_a_batch_completing_during_the_reconnection_reaches_the_fresh_session_once():
    async def scenario():
        host = Host()
        gate, running, _ = held_batch(host)
        fitting, session = asyncio.Event(), host.session

        async def fitted_after_completion(consume, options):
            if host.sessions:  # the replacement's fit is taken only once the batch has completed
                await fitting.wait()
            return await session(consume, options)
        host.session = fitted_after_completion
        task, first = await started(host)
        try:
            first.emit('ready', input_rate=24000, output_rate=24000)
            first.emit('tool_call', call_id='work', name='slow_tool', arguments='{}')
            await running.wait()
            first.emit('closed', resumable=False, transport_loss=True, reason='transport_lost')
            await asyncio.sleep(.05)
            gate.set()
            await asyncio.sleep(.05)
            fitting.set()
            await until(lambda: len(host.sessions) == 2)
            second = host.sessions[1]
            second.emit('ready', input_rate=24000, output_rate=24000)
            await until(lambda: any(POINTER in str(x) for x in second.sent))
            await asyncio.sleep(.05)
            assert len(host.sessions) == 2, 'no second refit for a unit the fresh fit already holds'
            assert not any(isinstance(x, ToolOutputs) for session in host.sessions for x in session.sent)
            assert sum(POINTER in str(x) for x in second.sent) == 1
        finally:
            gate.set()
            host.hub.ended_by_owner = True
            if not task.done(): await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


def test_an_ordinary_result_written_before_a_drop_is_offered_on_the_fresh_session():
    async def scenario():
        host = Host()
        host.tool_batch = lambda calls: asyncio.sleep(0, Batch([{'role': 'tool', 'tool_call_id': 'done', 'content': 'Done.'}]))
        task, first = await started(host)
        try:
            first.emit('ready', input_rate=24000, output_rate=24000)
            first.emit('tool_call', call_id='done', name='quick_tool', arguments='{}')
            await until(lambda: any(isinstance(x, ToolOutputs) for x in first.sent))
            first.emit('closed', resumable=False, transport_loss=True, reason='transport_lost')  # before any answer
            await until(lambda: len(host.sessions) == 2)
            host.sessions[1].emit('ready', input_rate=24000, output_rate=24000)
            await until(lambda: any(POINTER in str(x) for x in host.sessions[1].sent))
            assert not any(isinstance(x, ToolOutputs) for x in host.sessions[1].sent)
        finally:
            host.hub.ended_by_owner = True
            if not task.done(): await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


@pytest.mark.parametrize('past_deadline', [False, True])
def test_announced_expiries_are_not_unproven_drops(past_deadline):
    from skills.live_conversation.driver import UNPROVEN_RESUMES

    async def scenario():
        host = Host()
        task, session = await started(host)
        try:
            for n in range(1, UNPROVEN_RESUMES + 3):
                session.emit('ready', input_rate=24000, output_rate=24000)
                session.renewal_ready = False  # an open native call holds quiet; the engine closes at its deadline
                session.emit('renew', within_s=.01 if past_deadline else 5, resume=True, reason='go_away')
                await asyncio.sleep(.03)
                session.emit('closed', resumable=False, transport_loss=True, reason='transport_lost')
                await until(lambda: len(host.sessions) == n + 1 or task.done())
                assert not task.done(), 'an announced close is an expiry, never renewal_pending_work'
                session = host.sessions[-1]
                assert host.options[-1].resume_handle == ''
        finally:
            host.hub.ended_by_owner = True
            if not task.done(): await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


def test_a_loss_while_someone_speaks_is_a_hearing_gap_fact_not_owner_words():
    async def scenario():
        host = Host()
        facts = []

        async def record_fact(text):
            facts.append(text)
        host.record_fact = record_fact
        task, first = await started(host)
        try:
            first.emit('ready', input_rate=24000, output_rate=24000)
            first.emit('input_started')
            await asyncio.sleep(.02)
            first.emit('closed', resumable=False, transport_loss=True, reason='transport_lost')
            await until(lambda: len(host.sessions) == 2)
            host.sessions[1].emit('ready', input_rate=24000, output_rate=24000)
            await until(lambda: facts)
            assert len(facts) == 1 and 'may not have been heard' in facts[0]
            assert not host.captions and not host.context, 'neither owner words nor a peer observation'
            await until(lambda: any('may not have been heard' in str(x) for x in host.sessions[1].sent))
        finally:
            host.hub.ended_by_owner = True
            if not task.done(): await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


def test_a_silent_batch_finishing_on_another_conversation_needs_neither_refit_nor_pointer():
    async def scenario():
        host = Host()
        gate, running, _ = held_batch(host, silent=True)
        task, first = await started(host)
        try:
            first.emit('ready', input_rate=24000, output_rate=24000)
            first.emit('tool_call', call_id='quiet', name='attend', arguments='{}')
            await running.wait()
            first.emit('closed', resumable=False, transport_loss=True, reason='transport_lost')
            await until(lambda: len(host.sessions) == 2)
            host.sessions[1].emit('ready', input_rate=24000, output_rate=24000)
            gate.set()
            await asyncio.sleep(.1)
            assert len(host.sessions) == 2, 'its history waits for the next fit; nobody is owed an answer'
            assert not any(isinstance(x, ToolOutputs) or POINTER in str(x) for s in host.sessions for x in s.sent)
        finally:
            gate.set()
            host.hub.ended_by_owner = True
            if not task.done(): await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


def test_a_call_tool_handler_runs_outside_the_history_lock_and_may_record_a_fact(host_world):
    from ouroboros.contracts.live_conversation import CallTool
    host = host_world.host
    proceed = []

    async def handler(target, arguments):
        await asyncio.wait_for(target.record_fact('The handler saw this.'), 5)  # a held lock is a typed failure here
        await reached(lambda: proceed)
        return 'Recorded.'
    tool = host.call_tools['recording'] = CallTool('recording', 'Records one fact.', {'type': 'object', 'properties': {}}, handler)

    async def run():
        batch = asyncio.create_task(host.tool_batch([invocation('fact-1', tool.name)]))
        await reached(lambda: any('The handler saw this.' in str(row.get('content')) for row in host.messages))
        await asyncio.wait_for(host.caption('owner', 'Words during the handler.', 'owner-h'), 10)
        assert not batch.done() and not any(row.get('tool_calls') for row in host.messages)
        proceed.append(True)
        return await asyncio.wait_for(batch, 30)
    rows = asyncio.run(run())
    messages = host.messages
    fact = next(i for i, row in enumerate(messages) if 'The handler saw this.' in str(row.get('content')))
    words = next(i for i, row in enumerate(messages) if row.get('content') == 'Words during the handler.')
    at = unit_at(messages, 'fact-1')
    assert fact < words < at and messages[at + 1] == {'role': 'tool', 'tool_call_id': 'fact-1', 'content': 'Recorded.'}
    assert [row.get('tool_call_id') for row in rows] == ['fact-1'], 'a recorded fact is delivered by its offer or the next fit'


def test_a_handler_s_own_rows_are_its_changes_only_when_no_other_writer_moved_history(host_world):
    from ouroboros.contracts.live_conversation import CallTool
    host = host_world.host
    captioned = []

    async def attaches(target, arguments):
        target.messages.append({'role': 'user', 'content': 'Attached by the handler.'})
        if arguments.get('with_caption'):
            await target.caption('owner', 'Words while it attaches.', 'owner-a')
            captioned.append(True)
        return 'Attached.'
    host.call_tools['attaches'] = CallTool('attaches', 'Attaches.', {'type': 'object', 'properties': {}}, attaches)

    async def run():
        alone = await host.tool_batch([invocation('alone-1', 'attaches')])
        assert [row['content'] for row in alone] == ['Attached.', 'Attached by the handler.']
        at = unit_at(host.messages, 'alone-1')
        assert host.messages[at + 2]['content'] == 'Attached by the handler.', 'it follows its unit'
        overlapped = await host.tool_batch([invocation('overlap-1', 'attaches', with_caption=True)])
        assert [row['content'] for row in overlapped] == ['Attached.'], 'owner words are never replayed as changes'
        assert captioned
    asyncio.run(run())


def test_a_failing_call_tool_leaves_its_batch_whole_and_ends_the_call_typed(host_world, monkeypatch):
    from ouroboros.context_compaction import _atomic_units
    from ouroboros.contracts.live_conversation import CallTool
    host, slow = host_world.host, Held(monkeypatch, block=False)
    later = []

    async def failing(target, arguments):
        raise RuntimeError('a terminal failure inside a handler')

    async def after(target, arguments):
        later.append(True)
        return 'Never.'
    schema = {'type': 'object', 'properties': {}}
    host.call_tools['failing'] = CallTool('failing', 'Fails.', schema, failing)
    host.call_tools['after'] = CallTool('after', 'Follows.', schema, after)
    with pytest.raises(RuntimeError):
        asyncio.run(host.tool_batch([invocation('ordinary-1', 'list_available_tools'), invocation('failing-1', 'failing'),
                                     invocation('after-1', 'after')]))
    messages = host.messages
    at = unit_at(messages, 'ordinary-1')
    assert [row.get('tool_call_id') for row in messages[at + 1:at + 4]] == ['ordinary-1', 'failing-1', 'after-1']
    assert any(unit.start == at and unit.end == at + 3 for unit in _atomic_units(messages)), 'its completed effect is recorded'
    assert 'failed' in messages[at + 2]['content'] and 'did not run' in messages[at + 3]['content'] and not later
    assert slow.effects == ['list_available_tools']


def test_a_pending_request_keeps_its_complete_arguments(host_world, monkeypatch):
    host, slow = host_world.host, Held(monkeypatch)
    arguments = 'exact-argument-' * 400  # the only copy while the call runs: a reconnection fits this row

    async def run():
        batch = asyncio.create_task(host.tool_batch([invocation('long-1', 'list_available_tools', source=arguments)]))
        await slow.started()
        assert arguments in host.messages[-1]['content'] and is_request(host.messages[-1], 'long-1')
        assert arguments in saved(host)['messages'][-1]['content'], 'and so does recovery'
        slow.release.set()
        await asyncio.wait_for(batch, 30)
    asyncio.run(run())


def test_two_finished_batches_of_an_old_conversation_offer_one_pointer():
    async def scenario():
        host = Host()
        gate, running, batches = held_batch(host)
        task, first = await started(host)
        try:
            first.emit('ready', input_rate=24000, output_rate=24000)
            first.emit('tool_call', call_id='one', name='work', arguments='{}')
            await running.wait()
            first.emit('tool_call', call_id='two', name='work', arguments='{}')
            first.emit('closed', resumable=False, reason='transport_lost', transport_loss=True)
            await until(lambda: len(host.sessions) == 2)
            host.sessions[1].emit('ready', input_rate=24000, output_rate=24000)
            gate.set()
            await until(lambda: batches == [['one'], ['two']] and len(host.sessions) >= 3)
            last = host.sessions[-1]
            last.emit('ready', input_rate=24000, output_rate=24000)
            await until(lambda: any(POINTER in str(x) for x in last.sent))
            rows = [row for x in last.sent if isinstance(x, ContextUpdate) for row in x.messages]
            assert sum(POINTER in str(row) for row in rows) == 1
        finally:
            gate.set()
            host.hub.ended_by_owner = True
            if not task.done(): await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


@pytest.mark.parametrize('answered', [False, True])
def test_a_written_result_owes_its_answer_across_any_fresh_switch(answered):
    async def scenario():
        host = Host()
        host.tool_batch = lambda calls: asyncio.sleep(0, Batch([{'role': 'tool', 'tool_call_id': 'done', 'content': 'Done.'}]))
        task, first = await started(host)
        receipt = dict(usage={}, raw_usage={}, status='completed')
        try:
            first.emit('ready', input_rate=24000, output_rate=24000)
            first.emit('response_started', response_id='carried-the-call')
            first.emit('tool_call', call_id='done', name='quick_tool', arguments='{}')
            await until(lambda: any(isinstance(x, ToolOutputs) for x in first.sent))
            # This response began before the write: its receipt ends the tool turn, not the answer.
            first.emit('response_done', response_id='carried-the-call', **receipt)
            if answered:
                first.emit('response_started', response_id='the-answer')
                first.emit('response_done', response_id='the-answer', **receipt)
            first.emit('renew', within_s=5, resume=False, reason='tools')  # a plain refit at the next quiet moment
            await until(lambda: len(host.sessions) == 2)
            second = host.sessions[1]
            second.emit('ready', input_rate=24000, output_rate=24000)
            if answered:
                await asyncio.sleep(.05)
                assert not second.sent, 'an answered result offers no second turn'
            else:
                await until(lambda: any(POINTER in str(x) for x in second.sent))
                assert sum(POINTER in str(x) for x in second.sent) == 1
        finally:
            host.hub.ended_by_owner = True
            if not task.done(): await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


RECEIPT = dict(usage={}, raw_usage={}, status='completed')


class Adapter(Session):
    """The fake port's session with an adapter's delivery record: rows held only after a successful native write
    (``JsonSession.hold_context``), apart from a send that may queue them. ``records=False``: an older session."""

    def __init__(self, during=None, *, records=True):
        super().__init__()
        self.waiting, self.during, self.queued, self.finished = False, during, [], 0
        if records:
            self.held_context_rows = []

    async def receive(self):
        self.waiting = True
        try:
            return await self.events.get()
        finally:
            self.waiting = False

    async def handled(self):
        """Every emitted event is handled: the driver reads again with nothing left."""
        await until(lambda: self.waiting and self.events.empty())

    def deliver(self):
        for command in self.queued:
            if hasattr(self, 'held_context_rows'):
                self.held_context_rows.extend(command.messages)
        self.queued.clear()

    async def send(self, command):
        self.sent.append(command)
        if isinstance(command, ContextUpdate) and hasattr(self, 'held_context_rows'):
            self.held_context_rows.extend(command.messages)  # an update is written and held at once
        if isinstance(command, ToolOutputs):
            self.queued.append(command)
            if self.during is None:
                self.deliver()
            else:
                await self.during(self)
            self.finished += 1  # the driver's send returns from here


async def speak(host, session, response, *, start=True, continues=False):
    """One response's native events: its words, heard on the page, then its receipt."""
    item = response + '-item'
    if start:
        session.emit('response_started', response_id=response)
    session.emit('audio', item_id=item, rate=24000, pcm=b'\x00\x00' * 480)
    session.emit('caption', speaker='ouroboros', item_id=item, text='Spoken words.', final=True)
    await session.handled()
    host.hub.events.append(cp.PeerControl('peer', cp.PEER_OWNER, {'type': 'played', 'item_id': item, 'played_ms': 10_000}))
    await until(lambda: not host.hub.events)
    session.emit('response_done', response_id=response, continues=continues, **RECEIPT)
    await session.handled()


@pytest.mark.parametrize('case,owed', [
    ('answer', 0), ('answer_while_sending', 0), ('queued_then_answer', 0), ('thinking_then_answer', 0),
    ('thinking_then_idle', 0), ('call_turn_speaks_after', 1), ('answer_begun_before_a_second_result', 1)])
def test_only_a_response_begun_after_delivery_settles_its_result(case, owed):
    """A written ordinary result is answered by a response the driver saw begin after the adapter held that
    result, once it finishes without continuing. Speech, transcript tails and receipts carry no such identity:
    an earlier response that speaks or ends afterwards leaves the answer owed, and a later fresh session one pointer."""

    async def scenario():
        host = Host()
        host.tool_batch = lambda calls: asyncio.sleep(0, Batch([{'role': 'tool', 'tool_call_id': call['id'], 'content': 'Done.'}
                                                                 for call in calls]))

        async def while_writing(session):
            if case == 'answer_while_sending':  # delivered first; the reader keeps running while send awaits
                session.deliver()
                await speak(host, session, 'answer')
            elif case in ('begun_while_writing', 'older_session_begun_while_sending'):
                await speak(host, session, 'early')  # the write is still in flight: nothing delivered yet
                session.deliver()
            # A queued case holds nothing yet: an OpenAI/Grok adapter queues behind another response.
        during = while_writing if case in ('answer_while_sending', 'begun_while_writing', 'older_session_begun_while_sending',
                                           'queued_then_answer', 'queued_behind_an_answer') else None

        async def session(consume, options):
            made = Adapter(None if host.sessions else during, records=case != 'older_session_begun_while_sending')
            host.sessions.append(made)
            host.options.append(options)
            return await consume(made)
        host.session = session
        task, first = await started(host)
        try:
            first.emit('ready', input_rate=24000, output_rate=24000)
            if case == 'earlier_answer_finishes_after':
                first.emit('response_started', response_id='earlier')  # still answering when the call arrives
            first.emit('response_started', response_id='call-turn')
            first.emit('tool_call', call_id='done', name='quick_tool', arguments='{}')
            if case == 'call_turn_tail_redone':
                first.emit('response_done', response_id='call-turn', continues=True, **RECEIPT)  # thinking goes on
            elif case != 'call_turn_speaks_after':
                first.emit('response_done', response_id='call-turn', **RECEIPT)
            await until(lambda: first.finished)  # the result's send returned, whatever ran inside it
            await first.handled()
            if case == 'answer':
                await speak(host, first, 'answer')
            elif case in ('queued_then_answer', 'queued_behind_an_answer'):
                await speak(host, first, 'behind')  # requested before the queued result was written
                if case == 'queued_then_answer':
                    first.deliver()
                    await speak(host, first, 'answer')
            elif case in ('thinking_then_answer', 'thinking_then_idle', 'thinking_only'):
                first.emit('response_started', response_id='thinking')  # a usage-only reasoning phase
                first.emit('response_done', response_id='thinking', continues=True, **RECEIPT)
                if case == 'thinking_then_answer':
                    await speak(host, first, 'answer')
                elif case == 'thinking_then_idle':  # it ends the interaction with its own receipt, wordless
                    first.emit('response_done', response_id='thinking', **RECEIPT)
            elif case == 'call_turn_speaks_after':
                await speak(host, first, 'call-turn', start=False)
            elif case == 'call_turn_tail_redone':
                first.emit('caption', speaker='ouroboros', item_id='call-turn-item', text='A late tail.', final=True)
                first.emit('response_done', response_id='call-turn', **RECEIPT)  # the same receipt ends the interaction
            elif case == 'earlier_answer_finishes_after':
                await speak(host, first, 'earlier', start=False)
            elif case == 'answer_begun_before_a_second_result':  # it answers the first result, never one written later
                first.emit('response_started', response_id='answer')
                first.emit('tool_call', call_id='second', name='quick_tool', arguments='{}')
                await until(lambda: first.finished == 2)
                first.emit('response_done', response_id='answer', **RECEIPT)
            await first.handled()
            first.emit('closed', resumable=False, transport_loss=True, reason='transport_lost')
            await until(lambda: len(host.sessions) == 2)
            second = host.sessions[1]
            second.emit('ready', input_rate=24000, output_rate=24000)
            deadline = time.monotonic() + 1  # an owed pointer is offered at the first quiet moment
            while time.monotonic() < deadline and not any(POINTER in str(x) for x in second.sent):
                await asyncio.sleep(.01)
            await asyncio.sleep(.05)
            assert sum(POINTER in str(x) for x in second.sent) == owed, (case, second.sent)
        finally:
            host.hub.ended_by_owner = True
            if not task.done(): await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


def test_a_failed_result_write_leaves_one_pointer_on_the_fresh_session():
    from websockets.exceptions import ConnectionClosedError
    from websockets.frames import Close

    async def scenario():
        host = Host()
        host.tool_batch = lambda calls: asyncio.sleep(0, Batch([{'role': 'tool', 'tool_call_id': 'done', 'content': 'Done.'}]))
        task, first = await started(host)
        send = first.send

        async def refused(command):
            if isinstance(command, ToolOutputs):  # the writer meets the closed socket before the reader
                first.emit('closed', resumable=False, transport_loss=True, reason='transport_lost')
                raise ConnectionClosedError(Close(1012, 'synthetic restart'), Close(1012, 'synthetic restart'), True)
            await send(command)
        first.send = refused
        try:
            first.emit('ready', input_rate=24000, output_rate=24000)
            first.emit('tool_call', call_id='done', name='quick_tool', arguments='{}')
            await until(lambda: len(host.sessions) == 2)
            second = host.sessions[1]
            second.emit('ready', input_rate=24000, output_rate=24000)
            await until(lambda: any(isinstance(x, ContextUpdate) for x in second.sent))
            rows = [row for x in second.sent if isinstance(x, ContextUpdate) for row in x.messages]
            assert sum('fitted history' in str(row) for row in rows) == 1
            # That pointer's answer stays owed: lost before answering, the next fresh session is asked again.
            second.emit('closed', resumable=False, transport_loss=True, reason='transport_lost')
            await until(lambda: len(host.sessions) == 3)
            third = host.sessions[2]
            third.emit('ready', input_rate=24000, output_rate=24000)
            await until(lambda: any(POINTER in str(x) for x in third.sent))
            await asyncio.sleep(.05)
            assert sum(POINTER in str(x) for x in third.sent) == 1 and not any(
                isinstance(x, ToolOutputs) for session in host.sessions[1:] for x in session.sent)
        finally:
            host.hub.ended_by_owner = True
            if not task.done(): await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


@pytest.mark.parametrize('records,second,owed', [
    (False, 'unanswered', 1), (False, 'answered', 0), (True, 'answered', 0)])
def test_an_unanswered_pointer_stays_owed_through_another_fresh_loss(records, second, owed):
    """A fresh session's pointer carries the result's call-owned answer: only a response begun after that session
    took the pointer settles it. Dropped before then, the next fresh session is owed the same one pointer."""
    from ouroboros.contracts.live_conversation import LiveEvent

    async def scenario():
        host = Host()
        host.tool_batch = lambda calls: asyncio.sleep(0, Batch([{'role': 'tool', 'tool_call_id': call['id'], 'content': 'Done.'}
                                                                 for call in calls]))

        async def session(consume, options):
            made = Adapter(records=records)
            host.sessions.append(made)
            host.options.append(options)
            return await consume(made)
        host.session = session
        task, first = await started(host)
        try:
            first.emit('ready', input_rate=24000, output_rate=24000)
            first.emit('tool_call', call_id='done', name='quick_tool', arguments='{}')
            await until(lambda: first.finished)
            await first.handled()
            first.emit('closed', resumable=False, transport_loss=True, reason='transport_lost')
            await until(lambda: len(host.sessions) == 2)
            replacement = host.sessions[1]
            ready = LiveEvent('ready', {'input_rate': 24000, 'output_rate': 24000})
            if second == 'answer_begun_before_its_pointer':  # an answer already under way holds the pointer back
                replacement.events.put_nowait([ready, LiveEvent('response_started', {'response_id': 'earlier'})])
                await speak(host, replacement, 'earlier', start=False)
            else:
                replacement.events.put_nowait([ready])
            await until(lambda: any(POINTER in str(x) for x in replacement.sent))
            if second == 'answered':
                await speak(host, replacement, 'answer')
            await replacement.handled()
            replacement.emit('closed', resumable=False, transport_loss=True, reason='transport_lost')
            await until(lambda: len(host.sessions) == 3)
            third = host.sessions[2]
            third.emit('ready', input_rate=24000, output_rate=24000)
            deadline = time.monotonic() + 1  # an owed pointer is offered at the first quiet moment
            while time.monotonic() < deadline and not any(POINTER in str(x) for x in third.sent):
                await asyncio.sleep(.01)
            await asyncio.sleep(.05)
            assert sum(POINTER in str(x) for x in third.sent) == owed, (second, third.sent)
        finally:
            host.hub.ended_by_owner = True
            if not task.done(): await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


def look_call(host):
    """A driver over the fake port whose next batch is a requested look with one image, measured before delivery."""
    from skills.live_conversation.driver import _Call
    host.tool_batch = lambda calls: asyncio.sleep(0, Batch([{'role': 'tool', 'tool_call_id': 'look', 'content': '{"frames": []}'}],
                                                           silent=['look']))
    call = _Call(host)
    call.serving_catalog = lambda: ()
    call.attention.looks['look'] = [{'role': 'user', '_call_frame_source': 'screen', '_call_frame_digest': 'image',
        'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,YQ=='}}]}]
    return call


@pytest.mark.parametrize('switch,fits', [(True, True), (True, False), (False, True)])
def test_a_fresh_switch_while_a_delivery_is_measured_keeps_its_output_native_only(switch, fits):
    """Ownership is read after the delivery measure's await: a session adopted meanwhile holds the unit in its fit
    and is owed its answer, never the old native output, whether or not the measure admitted it (an unfit one is
    withheld, then released by this connection while renewal is suppressed). With no switch it stays native, once."""

    async def scenario():
        host = Host()
        fitting, release, verdicts = asyncio.Event(), asyncio.Event(), [fits]

        async def delivery_fits(rows, schemas):
            fitting.set()
            await release.wait()
            return verdicts.pop(0) if verdicts else True  # later serving guards admit it
        host.delivery_fits = delivery_fits
        call = look_call(host)
        call.failed_preparation_revision = 0  # a speculative fit already failed for this history: no renewal now
        task = asyncio.create_task(call.run())
        try:
            await until(lambda: host.sessions)
            first = host.sessions[0]
            first.emit('ready', input_rate=24000, output_rate=24000)
            first.emit('tool_call', call_id='look', name='attend', arguments='{"look": true}')
            await asyncio.wait_for(fitting.wait(), 2)
            if switch:
                first.emit('closed', resumable=False, transport_loss=True, reason='transport_lost')
                await until(lambda: len(host.sessions) == 2)
                host.sessions[1].emit('ready', input_rate=24000, output_rate=24000)
                await until(lambda: call.connected.is_set() and call.conversation == 2)
            release.set()
            serving = host.sessions[-1]
            await until(lambda: not call.tools_running and (any(POINTER in str(x) for x in serving.sent) if switch else
                                                            any(isinstance(x, ToolOutputs) for x in serving.sent)))
            outputs = [x for session in host.sessions for x in session.sent if isinstance(x, ToolOutputs)]
            if switch:
                assert not outputs and 'look' not in call.issued, 'never an old native output on a fresh conversation'
                assert sum(POINTER in str(x) for x in serving.sent) == 1 and call.answer_due is not None
            else:
                assert len(outputs) == 1 and outputs[0] in first.sent, 'the issuing conversation keeps its native output'
        finally:
            release.set()
            host.hub.ended_by_owner = True
            if not task.done(): await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


def test_a_withheld_result_stays_owed_through_another_fresh_loss():
    """A result withheld for a refit reaches the fresh session through its fit and one pointer; that pointer's
    obligation survives the fresh session being lost before it answers."""

    async def scenario():
        host = Host()
        host.tool_batch = lambda calls: asyncio.sleep(0, Batch([{'role': 'tool', 'tool_call_id': 'held', 'content': 'Done.'}]))
        task, first = await started(host)
        try:
            first.emit('ready', input_rate=24000, output_rate=24000)
            host.due = True  # the window is due at the result: it is withheld and the connection refitted
            first.emit('tool_call', call_id='held', name='quick_tool', arguments='{}')
            await until(lambda: len(host.sessions) == 2)
            second = host.sessions[1]
            second.emit('ready', input_rate=24000, output_rate=24000)
            await until(lambda: any(POINTER in str(x) for x in second.sent))
            assert not any(isinstance(x, ToolOutputs) for session in host.sessions for x in session.sent)
            second.emit('closed', resumable=False, transport_loss=True, reason='transport_lost')
            await until(lambda: len(host.sessions) == 3)
            third = host.sessions[2]
            third.emit('ready', input_rate=24000, output_rate=24000)
            await until(lambda: any(POINTER in str(x) for x in third.sent))
            await asyncio.sleep(.05)
            assert sum(POINTER in str(x) for x in third.sent) == 1
        finally:
            host.hub.ended_by_owner = True
            if not task.done(): await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


@pytest.mark.parametrize('silent', [('kept',), ()])
def test_a_deferred_output_released_after_a_fresh_switch_reaches_only_the_fit(silent):
    """The send seam reads ownership after its own waits: an output withheld for its serving measure, whose
    conversation is replaced meanwhile, is never written to the fresh session, and custody ends cleanly."""
    from skills.live_conversation.driver import _Call

    async def scenario():
        host = Host()
        fitting, release = asyncio.Event(), asyncio.Event()

        async def delivery_fits(rows, schemas):
            fitting.set()
            await release.wait()
            return True
        host.delivery_fits = delivery_fits
        host.tool_batch = lambda calls: asyncio.sleep(0, Batch([{'role': 'tool', 'tool_call_id': 'kept', 'content': 'Done.'}],
                                                               silent=silent))
        call = _Call(host)
        call.serving_catalog = lambda: ()
        call.failed_preparation_revision = 0  # a speculative fit already failed for this history: no renewal now
        task = asyncio.create_task(call.run())
        try:
            await until(lambda: host.sessions)
            first = host.sessions[0]
            first.emit('ready', input_rate=24000, output_rate=24000)
            host.due = True  # the window is due: the result is withheld, then measured before its release
            first.emit('tool_call', call_id='kept', name='quick_tool', arguments='{}')
            await asyncio.wait_for(fitting.wait(), 2)
            host.due = False
            first.emit('closed', resumable=False, transport_loss=True, reason='transport_lost')
            await until(lambda: len(host.sessions) == 2)
            second = host.sessions[1]
            second.emit('ready', input_rate=24000, output_rate=24000)
            await until(lambda: call.connected.is_set() and call.conversation == 2)
            release.set()
            await until(lambda: not call.tools_running or task.done())
            await asyncio.sleep(.05)
            assert not task.done(), task.exception() if task.done() and not task.cancelled() else 'the call ended'
            assert not any(isinstance(x, ToolOutputs) for x in second.sent), 'never an old native output on a fresh conversation'
            assert not call.deferred_tool_outputs and 'kept' not in call.issued
            assert sum(POINTER in str(x) for x in second.sent) == (0 if silent else 1), 'its fit owes one answer at most'
        finally:
            release.set()
            host.hub.ended_by_owner = True
            if not task.done(): await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


def test_end_starts_nothing_queued_while_a_refit_is_pending():
    async def scenario():
        host = Host()
        gate, running, batches = held_batch(host)
        task, first = await started(host)
        first.emit('ready', input_rate=24000, output_rate=24000)
        first.emit('tool_call', call_id='running', name='work', arguments='{}')
        await running.wait()
        first.emit('tool_call', call_id='queued', name='work', arguments='{}')
        await asyncio.sleep(.03)
        host.due = True  # this batch's completion requests a refit while the old socket closes
        host.hub.ended_by_owner = True
        gate.set()
        await asyncio.wait_for(task, 2)
        assert batches == [['running']], 'the requested call stays a host fact; it never starts after End'
    asyncio.run(scenario())
