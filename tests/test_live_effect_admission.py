"""An effect begins only if the call has not ended by the moment it begins.

A batch is admitted, then waits for history and its checkpoint, and its handlers run one after another; Stop and End can
arrive in any of those waits. The call's own drain latches a consumed Stop where the task's control reader no longer sees
it, and End is the page's. So each handler's start decides: ordinary handlers at their existing gate, call tools before
each one. What began finishes and is recorded, what did not begin says so, and then Stop raises its typed reason while End
lets the call finish. The real host, mailbox drain, executor and gate, then the real driver, gateway and Gemini adapter.
"""
import asyncio
import queue
import threading
from dataclasses import replace

import pytest

from ouroboros.contracts.live_conversation import CallTool, ToolOutputs
from ouroboros.contracts.realtime import RealtimeSetupError
from ouroboros.live_conversation import end_feature_calls
from ouroboros.llm_attempt import PhysicalDispatchInterrupted
from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, write_owner_message
from tests._extension_loader_shared import _clear_loader_state  # noqa: F401
from tests._live_world import host_world as host_world
from tests.test_live_conversation_driver import Batch, Host, until
from tests.test_live_tool_continuity import invocation, is_request, reached, unit_at

STOP = 'finalize-for-this-call'


def tool_row(host, call_id):
    rows = [row for row in host.messages if row.get('role') == 'tool' and row.get('tool_call_id') == call_id]
    assert len(rows) == 1, (call_id, rows)
    return str(rows[0]['content'])


def stop_written(host, msg_id=STOP):
    assert write_owner_message(host.ctx.drive_root, 'synthetic finalize', host.ctx.task_id, msg_id=msg_id, kind=KIND_FINALIZE_NOW)


def wrap_up_written(host):
    """The owner's real Wrap up: a durable graceful intent and its current mailbox control."""
    from ouroboros import cancel_intents
    from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION
    from supervisor.owner_stop import owner_stop_control_id
    intent = cancel_intents.request_cancel(host.ctx.drive_root, host.ctx.task_id,
                                           requested_stop_policy=cancel_intents.STOP_POLICY_FINALIZE)
    assert write_owner_message(host.ctx.drive_root, REASON_OWNER_REQUESTED_FINALIZATION, host.ctx.task_id,
                               msg_id=owner_stop_control_id(intent), kind=KIND_FINALIZE_NOW)


def hard_stop_written(host):
    from ouroboros import cancel_intents
    cancel_intents.request_cancel(host.ctx.drive_root, host.ctx.task_id, requested_stop_policy=cancel_intents.STOP_POLICY_IMMEDIATE)


def page_ended(host):
    with host.hub._condition:
        host.hub._owner_bye = True


def call_tool(host, name, entered, *, gate=None):
    """A call tool that records its entry, optionally holding there until the scenario lets it finish."""
    async def handler(target, arguments):
        entered.append(name)
        if gate is not None:
            await gate.wait()
        return f'{name} finished.'
    host.call_tools[name] = CallTool(name, 'Synthetic effect.', {'type': 'object', 'properties': {}}, handler)


def ordinary_tool(host, name, entered, *, held=None, release=None, bound=None):
    """The real executor and gate run; only this handler's body is replaced. ``bound`` receives the call reader its
    worker's context holds."""
    def handler(ctx, **_):
        entered.append(name)
        if bound is not None:
            from ouroboros.owner_pause import _CALL_END
            bound.append(_CALL_END.get())
        if held is not None:
            held.set()
            assert release.wait(30), 'the scenario never released the handler'
        return f'{name} finished.'
    host.tools.override_handler(name, handler)


def dynamic_tool(monkeypatch, kind, entered):
    """Real registry dispatch; only the catalog, liveness and external body are synthetic."""
    from ouroboros import extension_loader, extension_process_runner, mcp_client, safety
    from ouroboros.tools import extension_dispatch
    from ouroboros.tools.registry_core import ToolRegistry
    from ouroboros.tools.tool_result import ToolResult

    name = 'mcp_probe__remote_effect' if kind == 'mcp' else 'ext_9_probe_effect'
    real_check = safety.check_safety
    monkeypatch.setattr(safety, 'check_safety', lambda tool_name, *a, **k:
                        (True, '') if tool_name == name else real_check(tool_name, *a, **k))

    def body(*_a, **_kw):
        entered.append(name)
        return f'{name} finished.'

    if kind == 'mcp':
        real_miss = ToolRegistry._mcp_name_miss
        monkeypatch.setattr(ToolRegistry, '_mcp_name_miss',
                            lambda registry, tool_name: None if tool_name == name else real_miss(registry, tool_name))
        monkeypatch.setattr(mcp_client, '_call_mcp_tool_result',
                            lambda *a, **k: ToolResult(status='ok', code='OK', text=body()))
    else:
        descriptor = {'name': name, 'skill': 'probe', 'out_of_process': kind == 'isolated',
                      'timeout_sec': 30, 'handler': body, 'wants_ctx': False, 'skills_repo_path': ''}
        real_candidate = extension_dispatch._extension_dispatch_candidate
        monkeypatch.setattr(extension_dispatch, '_extension_dispatch_candidate', lambda ctx, tool_name:
                            (dict(descriptor), False) if tool_name == name else real_candidate(ctx, tool_name))
        monkeypatch.setattr(extension_loader, 'is_extension_live', lambda *_a, **_k: True)
        monkeypatch.setattr(extension_process_runner, 'dispatch_extension_tool_subprocess', body)
        monkeypatch.setattr(extension_process_runner, 'disclose_inprocess_extension_dispatch', lambda *_a, **_k: None)
    return name


async def outcome_of(batch):
    return (await asyncio.gather(batch, return_exceptions=True))[0]


def assert_stopped(outcome, code):
    assert isinstance(outcome, RealtimeSetupError) and outcome.code == code, repr(outcome)


# --- waiting for history ------------------------------------------------------------------------------------------------


def holding_drain(monkeypatch):
    """The real typed drain holds history here, before it reads the mailbox: the batch behind it is already admitted."""
    from ouroboros import loop_round_limits
    entered, release = threading.Event(), threading.Event()
    original = loop_round_limits._drain_incoming_messages

    def drain(*args, **kwargs):
        entered.set()
        assert release.wait(30)
        return original(*args, **kwargs)
    monkeypatch.setattr(loop_round_limits, '_drain_incoming_messages', drain)
    return entered, release


@pytest.mark.parametrize('control', ['none', 'stop-drained', 'end'])
def test_a_control_taken_while_a_batch_waits_for_history_starts_no_call_tool(host_world, monkeypatch, control):
    host = host_world.host
    host.incoming = queue.Queue()
    entered = []
    call_tool(host, 'probe_effect', entered)
    drain_entered, release = holding_drain(monkeypatch)

    async def scenario():
        drain = asyncio.create_task(host.mailbox())
        assert await asyncio.to_thread(drain_entered.wait, 30)
        batch = asyncio.create_task(host.tool_batch([invocation('waiting', 'probe_effect')]))
        await reached(lambda: bool(host.tool_jobs))
        assert not entered, 'admitted, and waiting for history'
        if control == 'stop-drained':
            stop_written(host)
        elif control == 'end':
            page_ended(host)
        release.set()
        outcome = await outcome_of(batch)
        await asyncio.gather(drain, return_exceptions=True)
        if control == 'none':
            assert entered == ['probe_effect'] and tool_row(host, 'waiting') == 'probe_effect finished.'
            return
        assert not entered, 'the handler never began'
        assert 'did not run: the call ended first' in tool_row(host, 'waiting')
        if control == 'stop-drained':
            assert_stopped(outcome, 'owner_stop')
        else:
            assert list(outcome) and 'did not run' in str(outcome), 'End is no failure: the batch comes back'
        assert any(is_request(row, 'waiting') for row in host.messages) and unit_at(host.messages, 'waiting') is not None
    asyncio.run(scenario())


# --- the pre-effect checkpoint ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("control", "code", "kind"), [
    ('none', '', 'call-tool'),
    ('stop-durable', 'finalize_requested', 'ordinary'),
    ('calls-disabled', 'calls_disabled', 'inprocess'),
    ('end', '', 'mcp'),
    ('end', '', 'isolated'),
])
def test_a_control_taken_during_the_pre_effect_checkpoint_starts_nothing(host_world, monkeypatch, kind, control, code):
    host = host_world.host
    entered = []
    if kind == 'call-tool':
        call_tool(host, 'probe_effect', entered)
        name, arguments = 'probe_effect', {}
    elif kind == 'ordinary':
        ordinary_tool(host, 'get_task_result', entered)
        name, arguments = 'get_task_result', {'task_id': 'none'}
    else:
        name, arguments = dynamic_tool(monkeypatch, kind, entered), {}
    checkpoint, reached_checkpoint, release = host.checkpoint, threading.Event(), threading.Event()

    def held(boundary):
        if boundary == 'pre_effect':
            reached_checkpoint.set()
            assert release.wait(30)
        return checkpoint(boundary)
    monkeypatch.setattr(host, 'checkpoint', held)
    async def scenario():
        batch = asyncio.create_task(host.tool_batch([invocation('checkpointed', name, **arguments)]))
        assert await asyncio.to_thread(reached_checkpoint.wait, 30)
        if control == 'stop-durable':
            stop_written(host)
        elif control == 'calls-disabled':
            end_feature_calls(host.turn.feature['key'])
        elif control == 'end':
            page_ended(host)
        release.set()
        outcome = await outcome_of(batch)
        if control == 'none':
            assert entered == [name] and f'{name} finished.' in tool_row(host, 'checkpointed')
            return
        assert not entered
        row = tool_row(host, 'checkpointed')
        assert 'did not run' in row if kind == 'call-tool' else 'NOT STARTED' in row, row
        if code:
            assert_stopped(outcome, code)
        else:
            assert not isinstance(outcome, BaseException), repr(outcome)
    asyncio.run(scenario())


# --- a later handler of the same batch ----------------------------------------------------------------------------------


@pytest.mark.parametrize(("second", "control"), [
    ('ordinary', 'none'),
    ('ordinary', 'hard-stop'),
    ('mcp', 'stop-undrained'),
    ('mcp', 'end'),
    ('isolated', 'stop-drained'),
    ('inprocess', 'wrap-up-drained'),
])
def test_a_control_taken_while_an_ordinary_handler_runs_starts_no_later_one(host_world, monkeypatch, control, second):
    """The critic's ordering: the running handler finishes with its exact result, the next one never begins."""
    host = host_world.host
    host.incoming = queue.Queue()
    entered, held, release = [], threading.Event(), threading.Event()
    ordinary_tool(host, 'get_task_result', entered, held=held, release=release)
    if second == 'ordinary':
        ordinary_tool(host, 'send_user_message', entered)
        name, arguments = 'send_user_message', {'text': 'Sent after Stop?'}
    else:
        name, arguments = dynamic_tool(monkeypatch, second, entered), {}

    async def scenario():
        batch = asyncio.create_task(host.tool_batch([invocation('first', 'get_task_result', task_id='none'),
                                                     invocation('second', name, **arguments)]))
        assert await asyncio.to_thread(held.wait, 30)
        drain = None
        if control.startswith('stop'):
            stop_written(host)
        elif control == 'wrap-up-drained':
            wrap_up_written(host)
        elif control == 'hard-stop':
            hard_stop_written(host)
        if control.endswith('-drained'):
            drain = asyncio.create_task(host.mailbox())
            await asyncio.wait({drain}, timeout=10)
            assert drain.done() and host._end_reason[0] == 'owner_stop', 'history is free while an effect runs'
        elif control == 'end':
            page_ended(host)
        release.set()
        outcome = await outcome_of(batch)
        if drain is not None:
            await asyncio.gather(drain, return_exceptions=True)
        assert 'get_task_result finished.' in tool_row(host, 'first'), 'an effect already begun finishes and is recorded'
        if control == 'none':
            assert entered == ['get_task_result', name]
            return
        assert entered == ['get_task_result'], 'the next handler never begins'
        assert 'NOT STARTED' in tool_row(host, 'second') and 'may have happened' not in tool_row(host, 'second')
        if control == 'end':
            assert not isinstance(outcome, BaseException), repr(outcome)
        else:
            assert_stopped(outcome, {'stop-undrained': 'finalize_requested', 'hard-stop': 'cancelled'}.get(control, 'owner_stop'))
    asyncio.run(scenario())


@pytest.mark.parametrize('control,code', [('none', ''), ('stop-durable', 'finalize_requested'), ('stop-drained', 'owner_stop'),
                                          ('end', '')])
def test_a_control_taken_while_a_call_tool_runs_starts_no_later_one(host_world, control, code):
    host = host_world.host
    host.incoming = queue.Queue()
    entered = []

    async def scenario():
        gate = asyncio.Event()
        call_tool(host, 'first_effect', entered, gate=gate)
        call_tool(host, 'second_effect', entered)
        batch = asyncio.create_task(host.tool_batch([invocation('first', 'first_effect'), invocation('second', 'second_effect')]))
        await reached(lambda: entered == ['first_effect'])
        if control.startswith('stop'):
            stop_written(host)
        elif control == 'wrap-up-drained':
            wrap_up_written(host)
        elif control == 'hard-stop':
            hard_stop_written(host)
        if control.endswith('-drained'):
            await asyncio.gather(host.mailbox(), return_exceptions=True)
            assert host._end_reason[0] == 'owner_stop'
        elif control == 'end':
            page_ended(host)
        gate.set()
        outcome = await outcome_of(batch)
        assert tool_row(host, 'first') == 'first_effect finished.'
        if control == 'none':
            assert entered == ['first_effect', 'second_effect']
            return
        assert entered == ['first_effect'] and 'did not run: the call ended first' in tool_row(host, 'second')
        if code:
            assert_stopped(outcome, code)
        else:
            assert not isinstance(outcome, BaseException), repr(outcome)
        assert set(host.executed) >= {'first', 'second'}, 'one whole unit: a repeated id is not run again'
    asyncio.run(scenario())


@pytest.mark.parametrize('control', ['stop-drained', 'end'])
def test_a_control_taken_during_an_ordinary_effect_starts_no_call_tool_after_it(host_world, control):
    host = host_world.host
    host.incoming = queue.Queue()
    entered, held, release = [], threading.Event(), threading.Event()
    ordinary_tool(host, 'get_task_result', entered, held=held, release=release)
    call_tool(host, 'probe_effect', entered)

    async def scenario():
        batch = asyncio.create_task(host.tool_batch([invocation('ordinary', 'get_task_result', task_id='none'),
                                                     invocation('side', 'probe_effect')]))
        assert await asyncio.to_thread(held.wait, 30)
        if control == 'end':
            page_ended(host)
        else:
            stop_written(host)
            await asyncio.gather(host.mailbox(), return_exceptions=True)
        release.set()
        outcome = await outcome_of(batch)
        assert entered == ['get_task_result'] and 'did not run: the call ended first' in tool_row(host, 'side')
        assert 'get_task_result finished.' in tool_row(host, 'ordinary')
        if control == 'end':
            assert not isinstance(outcome, BaseException), repr(outcome)
        else:
            assert_stopped(outcome, 'owner_stop')
    asyncio.run(scenario())


def test_a_parallel_batch_refused_before_it_begins_says_so_for_every_call(host_world, monkeypatch):
    from ouroboros.loop_tool_execution import _PARALLEL_SAFE_TOOLS, tool_calls_can_run_parallel
    host = host_world.host
    names = sorted(_PARALLEL_SAFE_TOOLS & set(host.tools._entries))[:2]
    calls = [invocation(f'parallel-{name}', name) for name in names]
    assert tool_calls_can_run_parallel(calls)
    entered = []
    for name in names:
        ordinary_tool(host, name, entered)
    checkpoint = host.checkpoint

    def ended_during_checkpoint(boundary):
        if boundary == 'pre_effect':
            page_ended(host)
        return checkpoint(boundary)
    monkeypatch.setattr(host, 'checkpoint', ended_during_checkpoint)

    async def scenario():
        await host.tool_batch(calls)
        assert not entered
        assert all('NOT STARTED' in tool_row(host, call['id']) for call in calls)
    asyncio.run(scenario())


@pytest.mark.parametrize('ending', ['finished', 'control', 'job-cancelled'])
def test_the_gate_reads_a_calls_end_only_during_its_own_batch(host_world, ending):
    """The task context outlives the batch, and its later work is no effect of this call. The reader lives only in the
    contexts the batch started its work from, never on the shared context, so later work, started elsewhere, cannot
    read it however the batch ended. ``cancelled-waiter`` is the call's teardown, which joins the batch;
    ``job-cancelled`` cancels the batch's own job, though nothing cancels a shielded batch today."""
    from ouroboros.owner_pause import run_tool_handler
    host = host_world.host
    host.incoming = queue.Queue()

    async def scenario():
        entered, held, release, bound = [], threading.Event(), threading.Event(), []
        ordinary_tool(host, 'get_task_result', entered, held=held, release=release, bound=bound)
        ordinary_tool(host, 'send_user_message', entered)
        batch = asyncio.create_task(host.tool_batch([invocation('first', 'get_task_result', task_id='none'),
                                                     invocation('second', 'send_user_message', text='later')]))
        assert await asyncio.to_thread(held.wait, 30)
        assert getattr(host.ctx, '_recorded_call_end', None) is None, 'the shared context never holds the reader'
        (source, reader), = bound
        assert source is host.ctx and reader() is None, "the running handler's own context reads this open call"
        job = next(iter(host.tool_jobs))
        if ending == 'control':
            stop_written(host)
            await asyncio.gather(host.mailbox(), return_exceptions=True)
        elif ending == 'cancelled-waiter':
            batch.cancel()
        elif ending == 'job-cancelled':
            job.cancel()
        await asyncio.sleep(.05)
        release.set()
        outcome = await outcome_of(batch)
        await asyncio.gather(job, return_exceptions=True)
        assert getattr(host.ctx, '_recorded_call_end', None) is None, 'nor once the batch is over'
        if ending == 'finished':
            assert entered == ['get_task_result', 'send_user_message'] and not isinstance(outcome, BaseException)
        elif ending == 'control':
            assert entered == ['get_task_result'] and isinstance(outcome, RealtimeSetupError)
        else:
            assert isinstance(outcome, asyncio.CancelledError)
        host._end_reason = ('owner_stop', 'ordinary owner control ended the call')
        assert await asyncio.to_thread(run_tool_handler, host.ctx, lambda: 'later work ran') == 'later work ran'
    asyncio.run(scenario())


# --- a worker the batch stopped waiting for ---------------------------------------------------------------------------


def outlived(host, monkeypatch, kind, effects, refusals):
    """A worker that outlives its batch: the tool wait is cut to 0.1 s for this one tool, which then holds until
    released, a begun builtin body before its nested start or a dynamic MCP entry in its preparation. Returns the
    invocation, the event set on entering that hold, its release, and the event set once the worker's gate decided."""
    from ouroboros import loop_tool_execution, owner_pause, safety
    entered, release, gated = threading.Event(), threading.Event(), threading.Event()
    if kind == 'builtin-nested':
        name, args = 'knowledge_read', {'topic': 'fixture'}

        def body(ctx, **_):
            entered.set()
            try:
                assert release.wait(30), 'the scenario never released the worker'
                with owner_pause.operation_start():
                    effects.append('nested-start')
                return 'nested start finished'
            except PhysicalDispatchInterrupted as exc:
                refusals.append((exc.control_reason, exc.invocation_not_started))
                raise
            finally:
                gated.set()
        host.tools.override_handler(name, body)
    else:
        name, args = dynamic_tool(monkeypatch, 'mcp', effects), {}
        prepared, admit = safety.check_safety, owner_pause.admit_tool_dispatch

        def preparation(tool, *a, **kw):
            if tool == name:
                entered.set()
                assert release.wait(30), 'the scenario never released the preparation'
            return prepared(tool, *a, **kw)

        def entry(source):
            try:
                return admit(source)
            except PhysicalDispatchInterrupted as exc:
                refusals.append((exc.control_reason, exc.invocation_not_started))
                raise
            finally:
                gated.set()
        monkeypatch.setattr(safety, 'check_safety', preparation)
        monkeypatch.setattr(owner_pause, 'admit_tool_dispatch', entry)
    timeout = loop_tool_execution._get_tool_timeout
    monkeypatch.setattr(loop_tool_execution, '_get_tool_timeout',
                        lambda tools, tool, arguments: .1 if tool == name else timeout(tools, tool, arguments))
    return invocation(f'late-{kind}', name, **args), entered, release, gated


def call_control(host, control):
    """The page's End, or a Stop the call's own drain consumed: neither is in the task's control reader."""
    if control == 'end':
        page_ended(host)
    elif control == 'stop-drained':
        stop_written(host)
        return asyncio.gather(host.mailbox(), return_exceptions=True)
    return asyncio.sleep(0)


@pytest.mark.parametrize(("kind", "control"), [
    ('builtin-nested', 'open'),
    ('builtin-nested', 'stop-drained'),
    ('mcp-entry', 'end'),
])
def test_a_worker_the_batch_stopped_waiting_for_keeps_its_calls_end(host_world, monkeypatch, kind, control):
    """Astra's ordering: a non-foreground tool outlives its wait, the batch returns its timeout result and the
    conversation goes on, then End or a drained Stop arrives before the worker's next start. That start is new work of
    an ended call: refused with the call's own reason, and a begun body says it had begun. The open call's start
    runs."""
    from ouroboros import budget_pause
    from ouroboros.live_tool_lane import CALL_END
    host = host_world.host
    host.incoming = queue.Queue()
    effects, refusals = [], []
    call, entered, release, gated = outlived(host, monkeypatch, kind, effects, refusals)

    async def scenario():
        batch = asyncio.create_task(host.tool_batch([call]))
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            assert 'TOOL_TIMEOUT' in str(await asyncio.wait_for(batch, 10)), 'the batch stopped waiting'
            assert getattr(host.ctx, '_recorded_call_end', None) is None and not effects
            await call_control(host, control)
            release.set()
            assert await asyncio.to_thread(gated.wait, 10)
            drained = await asyncio.to_thread(budget_pause.drain_local_tool_futures, host.ctx, timeout_sec=10)
            assert drained['drained'], drained
        finally:
            release.set()
            await asyncio.gather(batch, return_exceptions=True)
    asyncio.run(scenario())
    if control == 'open':
        assert effects == ['nested-start' if kind == 'builtin-nested' else 'mcp_probe__remote_effect'] and not refusals
        return
    assert effects == [], 'no new effect of an ended call'
    assert refusals == [(CALL_END if control == 'end' else 'owner_stop', kind == 'mcp-entry')], refusals


@pytest.mark.parametrize(("phase", "control"), [
    ('entry', 'open'),
    ('after-initialize', 'end'),
    ('after-initialize', 'stop-drained'),
])
def test_a_real_mcp_send_the_batch_stopped_waiting_for_keeps_its_calls_end(host_world, monkeypatch, phase, control):
    """The same ordering through the real MCPManager, held in its preparation before the entry gate or in the session's
    initialize before the final send gate; only the SDK transport, session and catalog are synthetic. After End or a
    drained Stop nothing is sent, and a session already opened still closes."""
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from ouroboros import budget_pause, loop_tool_execution, mcp_client, safety
    from tests.test_mcp_client import _settings
    host, name = host_world.host, 'mcp_demo__effect'
    host.incoming = queue.Queue()
    entered, release, observed = threading.Event(), threading.Event(), []
    timeout = loop_tool_execution._get_tool_timeout
    monkeypatch.setattr(loop_tool_execution, '_get_tool_timeout',
                        lambda tools, tool, arguments: .1 if tool == name else timeout(tools, tool, arguments))
    monkeypatch.setattr(host.tools, '_mcp_name_miss', lambda _name: None)

    @asynccontextmanager
    async def transport(_cfg):
        observed.append('transport_open')
        try:
            yield None, None
        finally:
            observed.append('transport_close')

    class Session:
        def __init__(self, *_args):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            observed.append('session_close')

        async def initialize(self):
            observed.append('initialized')
            if phase == 'after-initialize':
                entered.set()
                assert await asyncio.to_thread(release.wait, 30)

        async def call_tool(self, *_args):
            observed.append('call')
            return SimpleNamespace(content=[SimpleNamespace(type='text', text='remote answer')], isError=False)
    monkeypatch.setattr(mcp_client, '_MCP_SDK_AVAILABLE', True)
    monkeypatch.setattr(mcp_client, '_transport_factory', transport)
    monkeypatch.setattr(mcp_client, 'ClientSession', Session)
    manager = mcp_client.MCPManager()

    async def catalog(*_args):
        return [{'name': 'effect', 'description': 'fixture', 'input_schema': {'type': 'object'}}]
    manager._async_list_tools = catalog
    manager.reconfigure(_settings({'id': 'demo', 'enabled': True, 'transport': 'streamable_http',
                                   'url': 'https://unused.invalid/mcp'}))
    assert manager.refresh_server('demo')['ok']
    monkeypatch.setattr(mcp_client, '_call_mcp_tool_result', manager._call_tool_result)
    checked = safety.check_safety

    def preparation(tool, *args, **kwargs):
        if tool != name:
            return checked(tool, *args, **kwargs)
        if phase == 'entry':
            entered.set()
            assert release.wait(30)
        return True, ''
    monkeypatch.setattr(safety, 'check_safety', preparation)

    async def scenario():
        batch = asyncio.create_task(host.tool_batch([invocation('real-mcp-late', name)]))
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            assert 'TOOL_TIMEOUT' in str(await asyncio.wait_for(batch, 10)), 'the batch stopped waiting'
            assert 'call' not in observed
            await call_control(host, control)
            release.set()
            drained = await asyncio.to_thread(budget_pause.drain_local_tool_futures, host.ctx, timeout_sec=10)
            assert drained['drained'], drained
        finally:
            release.set()
            await asyncio.gather(batch, return_exceptions=True)
    asyncio.run(scenario())
    opened = ['transport_open', 'initialized']
    if control == 'open':
        assert observed == [*opened, 'call', 'session_close', 'transport_close'], observed
    else:
        assert observed == ([] if phase == 'entry' else [*opened, 'session_close', 'transport_close']), observed


@pytest.mark.parametrize('kind', ['builtin-nested', 'mcp-entry'])
def test_a_later_call_on_the_shared_context_neither_lends_nor_inherits_an_ended_calls_end(host_world, monkeypatch,
                                                                                          kind):
    """The task context outlives a call. An ended call's worker, released while a later call's batch runs on that same
    context, keeps its own call's End rather than the later call's open state, and starts nothing. The later call's
    effect runs, and ordinary work after both calls reads no call's reader."""
    from ouroboros import budget_pause
    from ouroboros.call_peers import LiveCallHub
    from ouroboros.live_conversation import ConversationHost
    from ouroboros.live_tool_lane import CALL_END
    from ouroboros.owner_pause import run_tool_handler
    first = host_world.host
    first.incoming = queue.Queue()
    effects, refusals, later_entered = [], [], []
    call, entered, release, gated = outlived(first, monkeypatch, kind, effects, refusals)
    held, proceed = threading.Event(), threading.Event()
    ordinary_tool(first, 'get_task_result', later_entered, held=held, release=proceed)

    async def scenario():
        batch, later, second = asyncio.create_task(first.tool_batch([call])), None, None
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            assert 'TOOL_TIMEOUT' in str(await asyncio.wait_for(batch, 10))
            page_ended(first)  # the first call is over; its worker still waits
            second = ConversationHost(first.turn, first.ctx, list(first.messages), first.tools, host_world.task, '',
                                      LiveCallHub(first.turn.spec))
            second.incoming = queue.Queue()
            later = asyncio.create_task(second.tool_batch([invocation('later', 'get_task_result', task_id='none')]))
            assert await asyncio.to_thread(held.wait, 10), "the later call's batch runs on the same context"
            release.set()
            assert await asyncio.to_thread(gated.wait, 10), "the first call's worker decided meanwhile"
            proceed.set()
            assert 'get_task_result finished.' in str(await asyncio.wait_for(later, 10))
            drained = await asyncio.to_thread(budget_pause.drain_local_tool_futures, first.ctx, timeout_sec=10)
            assert drained['drained'], drained
            page_ended(second)
            assert await asyncio.to_thread(run_tool_handler, first.ctx, lambda: 'later work ran') == 'later work ran'
        finally:
            release.set()
            proceed.set()
            await asyncio.gather(*(task for task in (batch, later) if task is not None), return_exceptions=True)
            if second is not None:
                await second.finish()
                second.hub.close('test_done')
    asyncio.run(scenario())
    assert later_entered == ['get_task_result'], "the later call's own effect ran"
    assert effects == [], "the ended call's worker started nothing"
    assert refusals == [(CALL_END, kind == 'mcp-entry')], refusals


def test_a_batch_that_comes_back_after_end_delivers_nothing_more():
    """The driver's own seam: End while the batch ran, and the worker has not been cancelled yet."""
    from skills.live_conversation.driver import _Call
    from tests.test_live_deferred_custody import attach
    from tests.test_live_tool_continuity import Adapter

    async def scenario():
        host = Host()
        release = asyncio.Event()

        async def tool_batch(calls):
            await release.wait()
            host.hub.ended_by_owner = True
            return Batch([{'role': 'tool', 'tool_call_id': 'running', 'content': 'Done.'}])
        host.tool_batch = tool_batch
        call = _Call(host)
        world = attach(host, call, make=Adapter)
        session = world.make(None)
        call.session = session
        call.connected.set()
        call.writer = asyncio.create_task(call.save_rows())
        worker = asyncio.create_task(call.tools())
        try:
            await call.tool_queue.put(invocation('running', 'quick_tool'))
            await until(lambda: call.tools_running)
            release.set()
            await asyncio.wait_for(call.tool_queue.join(), 5)
            assert not worker.done(), 'the worker carries on: only this batch had nothing more to deliver'
            assert not [command for command in session.sent if isinstance(command, ToolOutputs)]
        finally:
            for task in (worker, call.writer):
                task.cancel()
            await asyncio.gather(worker, call.writer, return_exceptions=True)
    asyncio.run(scenario())


# --- the whole real call: driver, gateway, Gemini adapter, loopback ----------------------------------------------------------


def checked_then_held(monkeypatch, host):
    """The tools worker passes its host.check, and holds there until the scenario has the real drain inside history."""
    checked, proceed = threading.Event(), threading.Event()
    real = host.check

    async def check():
        await real()
        if asyncio.current_task().get_coro().__qualname__.endswith('_Call.tools') and not checked.is_set():
            checked.set()
            while not proceed.is_set():
                await asyncio.sleep(.001)
    monkeypatch.setattr(host, 'check', check)
    return checked, proceed


@pytest.mark.parametrize('control', ['stop-drained', 'end'])
def test_real_call_tool_admitted_before_a_control_never_begins(host_world, monkeypatch, control):
    """Astra's orderings over the real stack, with the real set_call_languages handler."""
    from tests._live_stack import stack
    drain_entered, release_drain = threading.Event(), threading.Event()
    armed = threading.Event()
    from ouroboros import loop_round_limits
    original = loop_round_limits._drain_incoming_messages

    def drain(*args, **kwargs):
        if armed.is_set() and not drain_entered.is_set():
            drain_entered.set()
            assert release_drain.wait(30)
        return original(*args, **kwargs)
    monkeypatch.setattr(loop_round_limits, '_drain_incoming_messages', drain)
    entered = []

    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini') as live:
            host = live.host
            checked, proceed = checked_then_held(monkeypatch, host)
            tool = host.call_tools['set_call_languages']

            async def handler(target, arguments):
                entered.append(arguments)
                return await tool.handler(target, arguments)
            host.call_tools['set_call_languages'] = replace(tool, handler=handler)
            before = host.state.get('languages')
            live.wire.tool('waiting-language', 'set_call_languages', {'languages': ['en-US']})
            try:
                await live.wait(checked.is_set, 30)
                armed.set()
                await live.wait(drain_entered.is_set, 30)
                proceed.set()
                await live.wait(lambda: bool(host.tool_jobs), 30)
                assert not entered
                if control == 'end':
                    live.page.end()
                    await live.wait(lambda: host.hub.ended_by_owner, 30)
                else:
                    stop_written(host, 'driver-stop')
                release_drain.set()
                await asyncio.wait_for(asyncio.shield(live.running), 60)
            finally:
                proceed.set()
                release_drain.set()
        assert not entered, 'the handler never began'
        assert host.state.get('languages') == before, 'its effect never happened'
        reason = live.result[1]['reason_code']
        assert reason == 'finished' if control == 'end' else reason in {'owner_stop', 'finalize_requested'}, live.result
        assert 'did not run' in tool_row(host, 'waiting-language')
    asyncio.run(scenario())


@pytest.mark.parametrize('control', ['stop', 'wrap-up', 'end'])
def test_real_second_ordinary_handler_never_begins_after_a_control(host_world, monkeypatch, control):
    """The original case end to end: one native batch of two ordinary calls, the first running when the owner's control lands.
    The call's teardown joins the running effect; the second handler's own gate refuses it."""
    from tests._live_stack import stack
    entered, held, release = [], threading.Event(), threading.Event()

    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini') as live:
            host = live.host
            ordinary_tool(host, 'get_task_result', entered, held=held, release=release)
            ordinary_tool(host, 'send_user_message', entered)
            live.wire.put({'toolCall': {'functionCalls': [
                {'id': 'first', 'name': 'get_task_result', 'args': {'task_id': 'none'}},
                {'id': 'second', 'name': 'send_user_message', 'args': {'text': 'Sent after the control?'}}]}})
            try:
                await live.wait(held.is_set, 30)
                if control == 'end':
                    live.page.end()
                    await live.wait(lambda: host.hub.ended_by_owner, 30)
                else:
                    {'stop': stop_written, 'wrap-up': wrap_up_written, 'hard-stop': hard_stop_written}[control](host)
                    await live.wait(lambda: host._end_reason is not None or live.running.done()
                                    or not live.call.connected.is_set(), 30)
            finally:
                release.set()
            await asyncio.wait_for(asyncio.shield(live.running), 60)
        assert entered == ['get_task_result'], 'the second handler never began'
        assert 'get_task_result finished.' in tool_row(host, 'first')
        assert 'NOT STARTED' in tool_row(host, 'second')
        reason = live.result[1]['reason_code']
        assert reason == 'finished' if control == 'end' else reason in {'owner_stop', 'finalize_requested', 'cancelled'}, live.result
    asyncio.run(scenario())
