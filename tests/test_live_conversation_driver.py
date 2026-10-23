"""Feature driver contract tests: fake admitted ports, no provider or credentials."""
import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros import call_protocol as cp
from ouroboros.contracts.live_conversation import LiveConfig, LiveEvent, LiveOptions, Respond, Video
from ouroboros.contracts.realtime import CatalogModel, RealtimeSetupError


class Hub:
    def __init__(self):
        self.closed = self.ended_by_owner = self.ended_by_ouroboros = False
        self.spec = SimpleNamespace(opened_by=cp.OPENED_BY_OWNER, presence_binding_id='')
        self.events, self.states, self.audio = [], [], []

    def peers(self): return [SimpleNamespace(role=cp.PEER_OWNER, mic=True)]
    def next_event(self, _timeout): return self.events.pop(0) if self.events else None
    def send_state(self, *args): self.states.append(args)
    def set_engine_rates(self, *_args): pass
    def pause_input(self): pass
    def send_caption(self, *_args, **_kwargs): pass
    def send_turn(self, *_args): pass
    def send_activity(self, *_args): pass
    def flush(self): pass
    def play(self, pcm, item): self.audio.append((pcm, item))
    def end_by_ouroboros(self): self.ended_by_ouroboros = True


class Session:
    renewal_ready = True

    def __init__(self):
        self.events = asyncio.Queue()
        self.sent = []

    async def receive(self): return await self.events.get()
    async def send(self, command): self.sent.append(command)
    def final_captions(self): return []
    def emit(self, kind, **data): self.events.put_nowait([LiveEvent(kind, data)])


class Batch(list):
    """The additive ToolBatch port shape; trunk owns its contract class."""
    def __init__(self, rows=(), *, silent=()):
        super().__init__(rows)
        self.silent = tuple(silent)


class Host:
    """No old private host fields/methods exist on this port."""
    def __init__(self):
        self.hub, self.state, self.schemas = Hub(), {}, []
        self.config = LiveConfig(enabled=True, provider='test:engine', voice='test-voice',
                                 instructions='Owner call instructions.', pause='long', interruptions=False,
                                 thinking='high', search=True)
        self.model = CatalogModel('test', 'Test', vision=True, honours=frozenset({'pause', 'interruptions'}))
        self.opening = 'The owner started a call in this room.'
        self.sessions, self.options, self.context, self.usage, self.captions = [], [], [], [], []
        self.due = False

    async def check(self): pass
    async def session(self, consume, options):
        self.due = False
        session = Session()
        self.sessions.append(session)
        self.options.append(options)
        return await consume(session)
    async def mailbox(self): return []
    async def tool_batch(self, calls): return Batch()
    async def caption(self, **row): self.captions.append(row)
    async def observe_context(self, row): self.context.append(row)
    def response_started(self, response_id): self.usage.append(('start', response_id))
    def response_done(self, response_id, **values): self.usage.append(('done', response_id, values))
    def renewal_due(self): return self.due


async def until(predicate):
    async def wait():
        while not predicate(): await asyncio.sleep(.005)
    await asyncio.wait_for(wait(), 2)


async def started(host, **kwargs):
    from skills.live_conversation.driver import run
    task = asyncio.create_task(run(host, **kwargs))
    await until(lambda: host.sessions or task.done())
    if task.done(): await task
    return task, host.sessions[0]


def test_port_greeting_options_usage_and_context():
    async def scenario():
        host = Host()
        task, session = await started(host)
        try:
            options = host.options[0]
            assert isinstance(options, LiveOptions)
            assert (options.voice, options.pause, options.interruptions) == ('test-voice', 'long', False)
            assert (options.thinking, options.search) == ('', False)
            assert options.instructions.startswith('## Live calls')
            assert options.instructions.endswith(host.config.instructions)
            assert 'no languages recorded' in options.instructions
            session.emit('ready', input_rate=24000, output_rate=24000, vision=False)
            session.emit('ready', input_rate=24000, output_rate=24000)
            session.emit('response_started', response_id='r1')
            session.emit('response_done', response_id='r1', usage={'audio_output_tokens': 42},
                         raw_usage={'vendor': 42}, status='interrupted', continues=False)
            host.hub.events.append(cp.PeerControl('peer', cp.PEER_OWNER, {'type':'context', 'text':'A peer observation'}))
            host.hub.events.append(cp.PeerVideo('peer', cp.PEER_OWNER, 'image/jpeg', b'fake-image'))
            await until(lambda: len(host.usage) == 2 and host.context and any(isinstance(x, Video) for x in session.sent))
            assert [x for x in session.sent if isinstance(x, Respond)] == [Respond(host.opening)]
            assert host.usage[-1] == ('done', 'r1', {'usage': {'audio_output_tokens':42}, 'raw_usage':{'vendor':42}, 'status':'interrupted'})
            assert 'not an owner instruction' in host.context[0]['content']
        finally:
            host.hub.ended_by_owner = True
            await asyncio.wait_for(task, 2)
        assert not any(state[0] == cp.STATE_CLOSED for state in host.hub.states)
    asyncio.run(scenario())


@pytest.mark.parametrize('reason', ['host', 'refit', 'resume'])
def test_renewal_waits_for_a_quiet_boundary_and_greets_only_once(reason):
    async def scenario():
        host = Host()
        task, first = await started(host)
        try:
            first.emit('ready', input_rate=24000, output_rate=24000)
            first.emit('resumable', handle='old-handle')
            first.emit('resumable', handle='latest-handle')
            first.renewal_ready = False
            if reason == 'host': host.due = True
            else: first.emit('renew', within_s=2, reason='rotate', resume=reason == 'resume')
            await asyncio.sleep(.05)
            assert len(host.sessions) == 1
            first.renewal_ready = True
            await until(lambda: len(host.sessions) == 2)
            assert host.options[1].resume_handle == ('latest-handle' if reason == 'resume' else '')
            assert any(state[0] == cp.STATE_RECONNECTING for state in host.hub.states)
            second = host.sessions[1]
            second.emit('ready', input_rate=24000, output_rate=24000)
            await asyncio.sleep(.03)
            assert [x for x in first.sent if isinstance(x, Respond)] == [Respond(host.opening)]
            assert not any(isinstance(x, Respond) for x in second.sent)
        finally:
            host.hub.ended_by_owner = True
            await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


def test_resumable_close_ends_native_work_and_refit_discards_the_old_handle():
    async def scenario():
        host = Host()
        task, first = await started(host)
        first.emit('ready', input_rate=24000, output_rate=24000)
        first.emit('resumable', handle='old-conversation')
        first.emit('input_started')
        first.renewal_ready = False
        first.emit('closed', resumable=True, reason='rotate')
        try:
            await until(lambda: len(host.sessions) == 2)
            second = host.sessions[1]
            second.emit('ready', input_rate=24000, output_rate=24000)
            second.emit('renew', within_s=2, resume=False, reason='refit')
            await until(lambda: len(host.sessions) == 3)
            assert host.options[2].resume_handle == ''
            host.sessions[2].emit('closed', resumable=True, reason='lost-before-new-handle')
            with pytest.raises(RealtimeSetupError, match='provider_closed'):
                await asyncio.wait_for(task, 2)
            assert len(host.sessions) == 3
        finally:
            host.hub.ended_by_owner = True
            if not task.done(): await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


def test_unsupported_vision_and_nonresumable_close_refuse_without_retry():
    async def scenario(kind):
        host = Host()
        host.model = replace(host.model, vision=False)
        task, session = await started(host)
        session.emit('ready', input_rate=24000, output_rate=24000, vision=True)
        if kind == 'vision':
            host.hub.events.append(cp.PeerVideo('peer', cp.PEER_OWNER, 'image/jpeg', b'fake-image'))
        else: session.emit('closed', reason='lost', resumable=False)
        with pytest.raises(RealtimeSetupError, match='vision_unsupported|provider_closed'):
            await asyncio.wait_for(task, 2)
        assert len(host.sessions) == 1
    for kind in ('vision', 'closed'): asyncio.run(scenario(kind))


def test_languages_are_validated_persisted_and_separate_for_each_binding(tmp_path):
    from skills.live_conversation.driver import record_call_languages, recorded_call_languages
    assert recorded_call_languages(tmp_path) == ()
    assert record_call_languages(tmp_path, ['ru-RU', 'en-us', 'ru-RU']) == ('ru-RU', 'en-US')
    assert record_call_languages(tmp_path, ['de-DE'], binding_id='binding-one') == ('de-DE',)
    assert recorded_call_languages(tmp_path) == ('ru-RU', 'en-US')
    assert recorded_call_languages(tmp_path, 'binding-one') == ('de-DE',)
    assert recorded_call_languages(tmp_path, 'binding-two') == ()
    for invalid in (['Russian'], ['en-US', 42], 'ru-RU'):
        with pytest.raises(ValueError): record_call_languages(tmp_path, invalid)
        assert recorded_call_languages(tmp_path) == ('ru-RU', 'en-US')
    assert record_call_languages(tmp_path, []) == ()
    assert recorded_call_languages(tmp_path) == ()


def install_tools(host, state_dir):
    import json
    from skills.live_conversation.driver import call_tools
    tools = {tool.name: tool for tool in call_tools(state_dir)}
    async def batch(calls):
        return Batch([{'role':'tool', 'tool_call_id':call['id'],
                 'content':await tools[call['function']['name']].handler(host, json.loads(call['function']['arguments']))}
                for call in calls], silent=[call['id'] for call in calls if tools[call['function']['name']].silent])
    host.tool_batch = batch
    return tools


def test_language_tool_refits_at_next_boundary_and_later_calls_start_with_record(tmp_path):
    from ouroboros.contracts.live_conversation import ToolOutputs
    async def scenario():
        host = Host()
        host.model = replace(host.model, honours=frozenset({'languages'}))
        tools = install_tools(host, tmp_path)
        assert tools['set_call_languages'].silent
        task, session = await started(host, state_dir=tmp_path)
        try:
            assert host.options[0].languages == ()
            session.emit('ready', input_rate=24000, output_rate=24000)
            session.renewal_ready = False
            session.emit('response_done', response_id='tool-response', usage={}, raw_usage={}, status='completed', continues=True)
            session.emit('tool_call', call_id='lang', name='set_call_languages', arguments='{"languages":["ru-RU"]}')
            await until(lambda: any(isinstance(x, ToolOutputs) for x in session.sent))
            outputs = next(x for x in session.sent if isinstance(x, ToolOutputs))
            assert outputs.silent == ('lang',)
            assert 'expects ru-RU' in outputs.messages[0]['content']
            assert len(host.sessions) == 1
            session.renewal_ready = True
            await until(lambda: len(host.sessions) == 2)
            assert host.options[1].languages == ('ru-RU',)
            assert host.options[1].resume_handle == ''
            assert 'Speech recognition expects ru-RU' in host.options[1].instructions
        finally:
            host.hub.ended_by_owner = True
            await asyncio.wait_for(task, 2)
        later = Host()
        later.model = host.model
        task, _ = await started(later, state_dir=tmp_path)
        assert later.options[0].languages == ('ru-RU',)
        later.hub.ended_by_owner = True
        await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


def test_end_call_is_silent_and_waits_until_goodbye_audio_was_played(tmp_path):
    from ouroboros.contracts.live_conversation import ToolOutputs
    async def scenario():
        host = Host()
        tools = install_tools(host, tmp_path)
        assert tools['end_call'].silent
        task, session = await started(host, state_dir=tmp_path)
        try:
            session.emit('ready', input_rate=24000, output_rate=24000)
            session.emit('audio', item_id='goodbye', rate=24000, pcm=b'\0' * 48000)
            session.emit('response_done', response_id='goodbye', usage={}, raw_usage={}, status='completed', continues=True)
            session.emit('tool_call', call_id='hangup', name='end_call', arguments='{}')
            await until(lambda: any(isinstance(x, ToolOutputs) for x in session.sent))
            assert next(x for x in session.sent if isinstance(x, ToolOutputs)).silent == ('hangup',)
            assert host.hub.ended_by_ouroboros and not task.done()
            assert not any(state[0] == cp.STATE_CLOSED for state in host.hub.states)
            host.hub.events.append(cp.PeerControl('peer', cp.PEER_OWNER, {'type':'played', 'item_id':'goodbye', 'played_ms':300}))
            await asyncio.sleep(.03)
            assert not task.done(), 'partial playback is not a completed goodbye'
            host.hub.events.append(cp.PeerControl('peer', cp.PEER_OWNER, {'type':'played', 'item_id':'goodbye', 'played_ms':1000}))
            await asyncio.wait_for(task, 2)
            assert host.hub.states[-1] == (cp.STATE_CLOSED, 'ouroboros_bye')
        finally:
            host.hub.ended_by_owner = True
            if not task.done(): await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


@pytest.mark.parametrize('honoured', [False, True])
def test_options_only_forward_honoured_knobs(honoured):
    from skills.live_conversation.driver import session_options
    host = Host()
    host.state['languages'] = ('ru-RU',)
    host.model = CatalogModel('fixture','Fixture', thinking_levels=('low','high') if honoured else (),
        honours=frozenset({'pause','interruptions','thinking','search','languages'}) if honoured else frozenset())
    options = session_options(host)
    assert (options.pause,options.interruptions,options.thinking,options.search,options.languages) == (
        ('long',False,'high',True,('ru-RU',)) if honoured else ('normal',True,'',False,()))


@pytest.mark.parametrize('levels', [('none', 'high'), ('minimal', 'low', 'medium', 'high', 'xhigh'), ('low', 'medium', 'high'), ()])
@pytest.mark.parametrize('configured', ['', 'none', 'low', 'high', 'xhigh', 'obsolete'])
def test_thinking_requires_a_catalog_level(levels, configured):
    from skills.live_conversation.driver import session_options
    host = Host()
    host.model = replace(host.model, honours=frozenset({'thinking'}) if levels else frozenset(), thinking_levels=levels)
    host.config = replace(host.config, thinking=configured)
    assert session_options(host).thinking == (configured if configured in levels else levels[0] if levels else '')


@pytest.mark.parametrize('silent', [(), ('call-id',)])
def test_tool_outputs_use_host_silent_ids_without_opening_a_response(silent):
    from ouroboros.contracts.live_conversation import ToolOutputs
    async def scenario():
        host = Host()
        changes = Batch([{'role': 'tool', 'tool_call_id': 'call-id', 'content': 'result'}], silent=silent)
        async def batch(calls): return changes
        host.tool_batch = batch
        task, session = await started(host)
        try:
            session.emit('ready', input_rate=24000, output_rate=24000)
            # Deliberately disagree with a feature tool name: the port owns silence.
            session.emit('tool_call', call_id='call-id', name='end_call' if not silent else 'another_silent_tool', arguments='{}')
            await until(lambda: any(isinstance(x, ToolOutputs) for x in session.sent))
            outputs = next(x for x in session.sent if isinstance(x, ToolOutputs))
            assert outputs == ToolOutputs(tuple(changes), tuple(host.schemas), silent=silent)
            assert [x for x in session.sent if isinstance(x, Respond)] == [Respond(host.opening)]
        finally:
            host.hub.ended_by_owner = True
            await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


def test_resume_keeps_playback_captions_and_response_identity_until_completion():
    async def scenario():
        host = Host()
        task, first = await started(host)
        finalized = []
        first.final_captions = lambda: finalized.append('first') or []
        try:
            first.emit('ready', input_rate=24000, output_rate=24000)
            first.emit('resumable', handle='continuation')
            first.emit('response_started', response_id='r1')
            first.emit('audio', item_id='spoken', rate=24000, pcm=b'\0' * 48000)
            first.emit('caption', speaker='ouroboros', item_id='spoken', text='A continuing thought.', final=True)
            await until(lambda: host.hub.audio)
            host.hub.events.append(cp.PeerControl('peer', cp.PEER_OWNER, {'type':'played', 'item_id':'spoken', 'played_ms':400}))
            first.emit('closed', resumable=True, reason='transport_lost')
            await until(lambda: len(host.sessions) == 2)
            assert not finalized, 'a resumed connection must retain its parser captions'
            assert not host.captions, 'the unfinished spoken row still belongs to the call'
            assert host.usage == [('start', 'r1')], 'do not settle the response at connection close'
            second = host.sessions[1]
            second.emit('ready', input_rate=24000, output_rate=24000)
            second.emit('response_started', response_id='r1')
            second.emit('audio', item_id='spoken', rate=24000, pcm=b'\0' * 48000)
            second.emit('response_done', response_id='r1', usage={'audio_output_tokens':42}, raw_usage={'vendor':42}, status='completed', continues=True)
            await until(lambda: any(x[0] == 'done' for x in host.usage))
            assert [x for x in host.usage if x[0] == 'start'] == [('start', 'r1')]
            host.due = True
            host.hub.events.append(cp.PeerControl('peer', cp.PEER_OWNER, {'type':'played', 'item_id':'spoken', 'played_ms':1500}))
            await asyncio.sleep(.04)
            assert not host.captions and len(host.sessions) == 2
            host.hub.events.append(cp.PeerControl('peer', cp.PEER_OWNER, {'type':'played', 'item_id':'spoken', 'played_ms':2000}))
            await until(lambda: host.captions)
            assert host.captions[0]['played_ms'] == 2000
            assert host.captions[0]['completed'] is True
            assert len(host.sessions) == 2, 'continues holds the interaction open after playback'
            second.emit('response_done', response_id='r1', usage={'audio_output_tokens':42}, raw_usage={'vendor':42}, status='completed', continues=False)
            await until(lambda: len(host.sessions) == 3)
            assert len([x for x in host.usage if x[0] == 'done']) == 1, 'liveness alone is not another receipt'
            assert host.options[2].resume_handle == ''
        finally:
            host.hub.ended_by_owner = True
            if not task.done(): await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


def test_tool_finishing_during_resume_goes_to_the_new_session():
    from ouroboros.contracts.live_conversation import ToolOutputs
    async def scenario():
        host = Host()
        running, release = asyncio.Event(), asyncio.Event()
        cancelled = []
        async def batch(calls):
            running.set()
            try: await release.wait()
            except asyncio.CancelledError:
                cancelled.append(True)
                raise
            return Batch([{'role':'tool', 'tool_call_id':'work', 'content':'Finished'}], silent=('work',))
        host.tool_batch = batch
        task, first = await started(host)
        try:
            first.emit('ready', input_rate=24000, output_rate=24000)
            first.emit('resumable', handle='tool-continuation')
            first.emit('tool_call', call_id='work', name='slow_tool', arguments='{}')
            await running.wait()
            first.emit('closed', resumable=True, reason='transport_lost')
            await until(lambda: len(host.sessions) == 2)
            release.set()
            await asyncio.sleep(.03)
            assert not cancelled
            assert not any(isinstance(x, ToolOutputs) for x in first.sent)
            second = host.sessions[1]
            assert not second.sent, 'outputs wait for the new ready event'
            second.emit('ready', input_rate=24000, output_rate=24000)
            await until(lambda: any(isinstance(x, ToolOutputs) for x in second.sent))
            assert next(x for x in second.sent if isinstance(x, ToolOutputs)).silent == ('work',)
            assert not any(isinstance(x, Respond) for x in second.sent)
        finally:
            release.set()
            host.hub.ended_by_owner = True
            if not task.done(): await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


def test_failed_resume_finalizes_only_on_failure_and_keeps_the_unfinished_reservation():
    async def scenario():
        host = Host()
        original_session = host.session
        finalized, at_reopen = [], []
        async def session(consume, options):
            if host.sessions:
                at_reopen.append((list(finalized), list(host.usage)))
                raise RealtimeSetupError('reopen_failed', 'fake failure')
            return await original_session(consume, options)
        host.session = session
        task, first = await started(host)
        def final_captions():
            finalized.append(True)
            return [LiveEvent('caption', {'speaker':'ouroboros', 'item_id':'partial', 'text':'Unfinished', 'final':True, 'incomplete':True})]
        first.final_captions = final_captions
        first.emit('ready', input_rate=24000, output_rate=24000)
        first.emit('resumable', handle='handle')
        first.emit('response_started', response_id='unfinished')
        first.emit('closed', resumable=True, reason='transport_lost')
        with pytest.raises(RealtimeSetupError, match='reopen_failed'):
            await asyncio.wait_for(task, 2)
        assert at_reopen == [([], [('start', 'unfinished')])]
        assert finalized == [True]
        assert host.captions[0]['text'] == '[Partial transcript] Unfinished'
        assert host.captions[0]['completed'] is False
    asyncio.run(scenario())


def test_interrupted_audio_stays_silent_after_resume_while_cancelled_usage_drains():
    async def scenario():
        host = Host()
        task, first = await started(host)
        try:
            first.emit('ready', input_rate=24000, output_rate=24000)
            first.emit('resumable', handle='handle')
            first.emit('response_started', response_id='interrupted')
            first.emit('audio', item_id='cut-off', rate=24000, pcm=b'\0' * 48000)
            first.emit('speech_started')
            first.emit('closed', resumable=True, reason='transport_lost')
            await until(lambda: len(host.sessions) == 2)
            second = host.sessions[1]
            second.emit('ready', input_rate=24000, output_rate=24000)
            second.emit('audio', item_id='cut-off', rate=24000, pcm=b'\0' * 48000)
            second.emit('response_done', response_id='interrupted', usage={'audio_output_tokens':12}, raw_usage={'cancelled':12}, status='cancelled', continues=False)
            await until(lambda: len(host.usage) == 2)
            assert len(host.hub.audio) == 1
            assert host.usage[-1] == ('done', 'interrupted', {'usage':{'audio_output_tokens':12}, 'raw_usage':{'cancelled':12}, 'status':'cancelled'})
        finally:
            host.hub.ended_by_owner = True
            if not task.done(): await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


def test_end_call_stops_extended_thinking_followup_after_goodbye(tmp_path):
    from ouroboros.contracts.live_conversation import ToolOutputs
    async def scenario():
        host = Host()
        install_tools(host, tmp_path)
        task, session = await started(host)
        try:
            session.emit('ready', input_rate=24000, output_rate=24000)
            session.emit('audio', item_id='goodbye', rate=24000, pcm=b'\0' * 48000)
            session.emit('tool_call', call_id='hangup', name='end_call', arguments='{}')
            await until(lambda: any(isinstance(x, ToolOutputs) for x in session.sent))
            session.renewal_ready = False
            session.emit('response_started', response_id='unwanted-followup')
            session.emit('audio', item_id='followup', rate=24000, pcm=b'\0' * 48000)
            await asyncio.sleep(.03)
            assert [item for _, item in host.hub.audio] == ['goodbye']
            host.hub.events.append(cp.PeerControl('peer', cp.PEER_OWNER, {'type':'played', 'item_id':'goodbye', 'played_ms':300}))
            await asyncio.sleep(.03)
            assert not task.done()
            host.hub.events.append(cp.PeerControl('peer', cp.PEER_OWNER, {'type':'played', 'item_id':'goodbye', 'played_ms':1000}))
            await asyncio.wait_for(task, 2)
            assert len(host.sessions) == 1
        finally:
            host.hub.ended_by_owner = True
            if not task.done(): await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


def test_pending_refit_survives_a_transport_resume_until_the_response_finishes():
    async def scenario():
        host = Host()
        task, first = await started(host)
        try:
            first.emit('ready', input_rate=24000, output_rate=24000)
            first.emit('resumable', handle='handle')
            first.emit('response_started', response_id='r1')
            host.state['refit'] = True
            first.emit('closed', resumable=True, reason='transport_lost')
            await until(lambda: len(host.sessions) == 2)
            second = host.sessions[1]
            second.emit('ready', input_rate=24000, output_rate=24000)
            await asyncio.sleep(.03)
            assert host.options[1].resume_handle == 'handle'
            assert len(host.sessions) == 2
            second.emit('response_done', response_id='r1', usage={}, raw_usage={}, status='completed', continues=False)
            await until(lambda: len(host.sessions) == 3)
            assert host.options[2].resume_handle == ''
        finally:
            host.hub.ended_by_owner = True
            if not task.done(): await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


def test_final_caption_writer_failure_does_not_leave_teardown_waiting():
    from skills.live_conversation.driver import _Call
    async def scenario():
        host = Host()
        async def failed_caption(**row): raise RuntimeError('fake caption storage failure')
        host.caption = failed_caption
        call = _Call(host)
        session = Session()
        session.final_captions = lambda: [LiveEvent('caption', {'speaker':'owner', 'item_id':str(i), 'text':'partial', 'final':True}) for i in range(2)]
        call.last_session = session
        call.writer = asyncio.create_task(call.save_rows())
        flushing = asyncio.create_task(call.finalize())
        try:
            done, _ = await asyncio.wait([flushing], timeout=.2)
            assert flushing in done, 'writer failure must not strand queued final captions'
            with pytest.raises(RuntimeError, match='caption storage failure'): await flushing
        finally:
            flushing.cancel()
            call.writer.cancel()
            await asyncio.gather(flushing, call.writer, return_exceptions=True)
    asyncio.run(scenario())


@pytest.mark.parametrize('source', ['host', 'provider'])
@pytest.mark.parametrize('barrier', ['response', 'input', 'tools', 'playback', 'caption', 'native'])
def test_fresh_renewal_waits_for_every_quiet_barrier(source, barrier):
    async def scenario():
        host = Host()
        released, running = asyncio.Event(), asyncio.Event()
        batches = []
        async def batch(calls):
            batches.append(calls)
            running.set()
            await released.wait()
            return Batch(silent=[call['id'] for call in calls])
        host.tool_batch = batch
        task, first = await started(host)
        try:
            first.emit('ready', input_rate=24000, output_rate=24000)
            first.emit('resumable', handle='must-not-refit-with-this')
            if barrier == 'response': first.emit('response_started', response_id='busy')
            if barrier == 'input': first.emit('input_started')
            if barrier == 'tools':
                first.emit('tool_call', call_id='one', name='work', arguments='{}')
                await running.wait()
                first.emit('tool_call', call_id='two', name='work', arguments='{}')
            if barrier == 'playback': first.emit('audio', item_id='spoken', rate=24000, pcm=b'\0' * 48000)
            if barrier == 'caption': first.emit('caption', speaker='ouroboros', item_id='spoken', text='Waiting for audio', final=True)
            if barrier == 'native': first.renewal_ready = False
            await asyncio.sleep(.03)
            if source == 'host': host.due = True
            else: first.emit('renew', within_s=2, resume=False, reason='audio_span')
            await asyncio.sleep(.05)
            assert len(host.sessions) == 1
            if barrier == 'response': first.emit('response_done', response_id='busy', usage={}, raw_usage={}, status='completed', continues=False)
            if barrier == 'input': first.emit('input_done')
            if barrier == 'tools': released.set()
            if barrier == 'caption': first.emit('audio', item_id='spoken', rate=24000, pcm=b'\0' * 48000)
            if barrier in {'caption', 'playback'}:
                # A peer can acknowledge only PCM the driver has delivered.
                # emit() queues provider input; the peer task may run first.
                await until(lambda: any(item == 'spoken' for _, item in host.hub.audio))
                host.hub.events.append(cp.PeerControl('peer', cp.PEER_OWNER, {'type':'played', 'item_id':'spoken', 'played_ms':1000}))
            if barrier == 'native': first.renewal_ready = True
            await until(lambda: len(host.sessions) == 2)
            assert host.options[1].resume_handle == ''
            if barrier == 'tools': assert [c['id'] for batch in batches for c in batch] == ['one', 'two']
        finally:
            released.set()
            host.hub.ended_by_owner = True
            if not task.done(): await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


@pytest.mark.parametrize('reason', ['audio_span', 'audio_drop', 'tools', 'languages'])
def test_refit_reasons_never_reuse_a_resumption_handle(reason):
    async def scenario():
        host = Host()
        task, first = await started(host)
        try:
            first.emit('ready', input_rate=24000, output_rate=24000)
            first.emit('resumable', handle='old')
            first.emit('resumable', handle='latest')
            first.emit('renew', within_s=2, resume=False, reason=reason)
            await until(lambda: len(host.sessions) == 2)
            assert host.options[1].resume_handle == ''
        finally:
            host.hub.ended_by_owner = True
            await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


@pytest.mark.parametrize('revoked', [False, True])
def test_goaway_without_a_current_handle_does_not_claim_a_resume(revoked):
    async def scenario():
        host = Host()
        task, session = await started(host)
        session.emit('ready', input_rate=24000, output_rate=24000)
        if revoked:
            session.emit('resumable', handle='stale')
            session.emit('resumable', handle='')
        session.emit('renew', within_s=2, resume=True, reason='go_away')
        with pytest.raises(RealtimeSetupError, match='resume_unavailable'):
            await asyncio.wait_for(task, 2)
        assert len(host.sessions) == 1
    asyncio.run(scenario())


@pytest.mark.parametrize('provider_deadline', [False, True])
def test_only_provider_renewal_has_a_deadline_and_final_captions_are_saved(provider_deadline):
    async def scenario():
        host = Host()
        task, first = await started(host)
        final_calls = []
        def final_captions():
            final_calls.append(True)
            return [LiveEvent('caption', {'speaker':'owner', 'item_id':'pending', 'text':'tail', 'final':True, 'incomplete':True})]
        first.final_captions = final_captions
        first.emit('ready', input_rate=24000, output_rate=24000)
        first.renewal_ready = False
        if provider_deadline: first.emit('renew', within_s=.02, resume=False, reason='audio_drop')
        else: host.due = True
        try:
            await asyncio.sleep(.06)
            if provider_deadline:
                with pytest.raises(RealtimeSetupError, match='renewal_pending_work'): await asyncio.wait_for(task, 2)
                assert len(host.sessions) == 1
            else:
                assert not task.done() and len(host.sessions) == 1
                first.renewal_ready = True
                await until(lambda: len(host.sessions) == 2)
            assert final_calls == [True]
            assert host.captions[0]['text'] == '[Partial transcript] tail'
        finally:
            host.hub.ended_by_owner = True
            if not task.done(): await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


@pytest.mark.parametrize('stop', ['end', 'cancel'])
def test_final_teardown_saves_partial_captions_once(stop):
    async def scenario():
        host = Host()
        task, session = await started(host)
        final_calls = []
        def final_captions():
            final_calls.append(True)
            return [LiveEvent('caption', {'speaker':'ouroboros', 'item_id':'partial', 'text':'Still talking', 'final':True, 'incomplete':True})]
        session.final_captions = final_captions
        session.emit('ready', input_rate=24000, output_rate=24000)
        session.emit('response_started', response_id='pending')
        await until(lambda: host.usage)
        if stop == 'cancel':
            task.cancel()
            with pytest.raises(asyncio.CancelledError): await asyncio.wait_for(task, 2)
        else:
            host.hub.ended_by_owner = True
            await asyncio.wait_for(task, 2)
        assert final_calls == [True]
        assert host.captions == [{'speaker':'ouroboros', 'text':'[Partial transcript] Still talking', 'item_id':'partial',
                                  'heard':False, 'played_ms':0, 'interrupted':False, 'completed':False}]
        assert host.usage == [('start', 'pending')], 'host teardown owns the unreceipted reservation'
    asyncio.run(scenario())


@pytest.mark.parametrize('source', ['mailbox', 'peer'])
@pytest.mark.parametrize('crosses_window', [False, True])
def test_context_additions_refit_before_delivery_when_the_window_is_full(source, crosses_window):
    from ouroboros.contracts.live_conversation import ContextUpdate
    async def scenario():
        host = Host()
        queued = []
        async def mailbox():
            rows, queued[:] = list(queued), []
            if rows:
                host.context.extend(rows)
                host.due = crosses_window
            return rows
        async def observe(row):
            host.context.append(row)
            host.due = crosses_window
        host.mailbox, host.observe_context = mailbox, observe
        task, first = await started(host)
        try:
            first.emit('ready', input_rate=24000, output_rate=24000)
            first.emit('resumable', handle='must-not-resume-a-refit')
            first.renewal_ready = False
            await until(lambda: bool(first.sent))
            if source == 'mailbox': queued.append({'role':'user', 'content':'New room message'})
            else: host.hub.events.append(cp.PeerControl('peer', cp.PEER_OWNER, {'type':'context', 'text':'Observed context'}))
            await until(lambda: bool(host.context))
            await asyncio.sleep(.04)
            additions = [x for x in first.sent if isinstance(x, ContextUpdate)]
            if crosses_window:
                assert additions == [], 'an over-window addition must be kept for the fresh fit'
            else:
                assert additions == [ContextUpdate(tuple(host.context))]
            assert len(host.sessions) == 1, 'the native quiet boundary still owns replacement'
            first.renewal_ready = True
            if crosses_window:
                await until(lambda: len(host.sessions) == 2)
                assert host.options[1].resume_handle == ''
                assert not any(isinstance(x, ContextUpdate) for x in first.sent)
            else:
                await asyncio.sleep(.05)
                assert len(host.sessions) == 1
        finally:
            host.hub.ended_by_owner = True
            await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


@pytest.mark.parametrize('refused', [False, True])
def test_response_refusal_drains_received_accounting_and_final_captions(refused):
    async def scenario():
        host = Host()
        batches = []
        def start(response):
            host.usage.append(('start', response))
            if refused:
                raise RealtimeSetupError('budget_exhausted', 'The shared spending limit has been reached.')
        async def tools(calls):
            batches.extend(calls)
            return Batch()
        host.response_started, host.tool_batch = start, tools
        task, session = await started(host)
        try:
            session.final_captions = lambda: [LiveEvent('caption', dict(speaker='owner', item_id='tail', text='Final tail', final=True))]
            session.emit('ready', input_rate=24000, output_rate=24000)
            await until(lambda: bool(session.sent))
            receipt = dict(usage={'audio_output_tokens':17}, raw_usage={'test':17}, status='completed')
            session.events.put_nowait([
                LiveEvent('response_started', {'response_id':'r1'}),
                LiveEvent('caption', dict(speaker='owner', item_id='heard', text='Already received', final=True)),
                LiveEvent('audio', dict(item_id='answer', rate=24000, pcm=b'\0' * 480)),
                LiveEvent('tool_call', dict(call_id='work', name='work', arguments='{}')),
                LiveEvent('response_done', dict(response_id='r1', continues=False, **receipt)),
                LiveEvent('caption', dict(speaker='ouroboros', item_id='answer', text='Final words', final=True)),
            ])
            if refused:
                with pytest.raises(RealtimeSetupError, match='budget_exhausted'):
                    await asyncio.wait_for(task, 2)
            else:
                await until(lambda: bool(batches))
                host.hub.ended_by_owner = True
                await asyncio.wait_for(task, 2)
            assert host.usage == [('start', 'r1'), ('done', 'r1', receipt)]
            assert {row['text'] for row in host.captions} == {'Already received', 'Final words', 'Final tail'}
            if refused:
                assert host.hub.audio == []
                assert batches == []
            else:
                assert [item for _, item in host.hub.audio] == ['answer']
                assert len(batches) == 1
            assert not any(state == (cp.STATE_CLOSED, 'ouroboros_bye') for state in host.hub.states)
        finally:
            host.hub.ended_by_owner = True
            if not task.done(): await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


@pytest.mark.parametrize('configured', ['', 'low', 'obsolete', 'none', 'high'])
def test_grok_feature_options_put_the_lowest_default_on_the_wire(configured):
    from ouroboros.contracts.realtime import RealtimeInput
    from skills.grok_realtime.plugin import CATALOG, GrokSession
    from skills.live_conversation.driver import session_options
    host = Host()
    host.model = CATALOG.models[0]
    host.config = replace(host.config, thinking=configured, voice='')
    selected = RealtimeInput(host.model.id, '[]', '[]', 256, 'nano', 'core', 'view')
    def never_connect(*args, **kwargs):
        raise AssertionError('This test only inspects the physical setup; no network is allowed')
    session = GrokSession(selected, session_options(host), connect=never_connect, observe=lambda *args: None)
    assert session.events[0]['session']['reasoning'] == {'effort': 'high' if configured == 'high' else 'none'}


@pytest.mark.parametrize('transport_closed', [False, True])
def test_peer_poll_defers_a_pending_refit_after_the_transport_requests_resume(transport_closed):
    from skills.live_conversation.driver import _Call
    async def scenario():
        host = Host()
        async def check(): raise AssertionError('the peer loop must not repeat the host control watcher')
        host.check = check
        host.state.update(refit=True, resume_handle='latest-handle')
        call = _Call(host)
        call.session = Session()
        call.response_open = True
        call.renewing = call.resume = True
        call.provider_closed = transport_closed
        peer = asyncio.create_task(call.peers())
        try:
            await until(lambda: peer.done() or not call.resume)
            if peer.done(): await peer
            assert call.resume is transport_closed, 'a dead transport must resume its active response before refitting'
            assert host.state['refit'] is True, 'resumption must retain the later refit obligation'
        finally:
            peer.cancel()
            await asyncio.gather(peer, return_exceptions=True)
    asyncio.run(scenario())


@pytest.mark.parametrize('check_delay', [0, .08])
def test_owner_audio_at_fifty_frames_per_second_survives_slow_control_reads(check_delay):
    from ouroboros.call_peers import LiveCallHub
    from ouroboros.contracts.live_conversation import Audio
    async def scenario():
        host = Host()
        host.hub = LiveCallHub(cp.CallSpec('audio-burst', 1, 'test', 'test'))
        host.hub.join(cp.PeerInfo('owner', cp.PEER_OWNER, True, True), 24000)
        async def check(): await asyncio.sleep(check_delay)
        host.check = check
        task, session = await started(host)
        try:
            session.emit('ready', input_rate=24000, output_rate=24000)
            await until(lambda: bool(session.sent))
            # Each frame has a distinct PCM value, so count, order and content
            # catch both dropped audio and duplicated/replaced frames.
            expected = [index.to_bytes(2, 'little') * 480 for index in range(1, 121)]
            clock = asyncio.get_running_loop()
            start = clock.time()
            for index, pcm in enumerate(expected):
                await asyncio.sleep(max(0, start + index / 50 - clock.time()))
                assert host.hub.receive_audio('owner', cp.AudioFrame(cp.FRAME_AUDIO_IN, 0, index, 24000, 0, pcm))
            deadline = clock.time() + 1.5
            while len([x for x in session.sent if isinstance(x, Audio)]) < len(expected) and clock.time() < deadline:
                await asyncio.sleep(.01)
            delivered = [x.pcm for x in session.sent if isinstance(x, Audio)]
            assert delivered == expected, f'{len(delivered)}/{len(expected)} owner frames reached the provider'
        finally:
            host.hub.receive_control('owner', {'type': 'bye'})
            await asyncio.wait_for(task, 2)
            host.hub.close('finished')
    asyncio.run(scenario())


def test_due_audio_is_drained_before_another_control_or_renewal_poll():
    from ouroboros.contracts.live_conversation import Audio
    from skills.live_conversation.driver import _Call
    async def scenario():
        host = Host()
        call = _Call(host)
        call.session, call.input_rate = Session(), 24000
        history = []
        frames = [index.to_bytes(2, 'little') * 480 for index in range(1, 9)]
        host.hub.events = [cp.PeerAudio('owner', cp.PEER_OWNER, pcm) for pcm in frames]
        async def check(): history.append('check')
        def renewal_due():
            history.append('renewal')
            return False
        def next_event(_timeout):
            if host.hub.events:
                history.append('audio')
                return host.hub.events.pop(0)
            host.hub.ended_by_owner = True
            return None
        host.check, host.renewal_due, host.hub.next_event = check, renewal_due, next_event
        await asyncio.wait_for(call.peers(), 2)
        assert [x.pcm for x in call.session.sent if isinstance(x, Audio)] == frames
        first, last = history.index('audio'), len(history) - 1 - history[::-1].index('audio')
        assert history[first:last + 1] == ['audio'] * len(frames), history
        assert 'check' not in history, 'the host owns continuous control monitoring'
    asyncio.run(scenario())


@pytest.mark.parametrize('reason', ['owner_stop', 'panic', 'budget_exhausted'])
def test_host_control_watcher_ends_busy_audio_promptly_and_joins_the_driver(reason):
    from ouroboros.contracts.live_conversation import Audio
    from ouroboros.live_conversation import ConversationHost
    async def scenario():
        host = Host()
        stop, checked = asyncio.Event(), asyncio.Event()
        async def check():
            await asyncio.sleep(.08)  # a strict store/budget read may be slow
            if stop.is_set(): raise RealtimeSetupError(reason, 'Test control ended the call.')
            checked.set()
        host.check = check
        task, session = await started(host)
        controls = asyncio.create_task(ConversationHost.watch_controls(host))
        try:
            session.emit('ready', input_rate=24000, output_rate=24000)
            await until(lambda: bool(session.sent))
            # All frames are already due. Even a send that completes without
            # suspending must give the independent control watcher a turn.
            frame = cp.PeerAudio('owner', cp.PEER_OWNER, b'\x01\x00' * 480)
            host.hub.events = [frame] * 100_000
            await checked.wait()
            await until(lambda: any(isinstance(x, Audio) for x in session.sent))
            assert not task.done(), 'ordinary control reads must keep the call alive'
            checked.clear()
            await asyncio.wait_for(checked.wait(), .4)
            stop.set()
            start = asyncio.get_running_loop().time()
            with pytest.raises(RealtimeSetupError, match=reason):
                await asyncio.wait_for(controls, .4)
            # The real host cancels and joins the feature when its watcher ends.
            task.cancel()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), .4)
            assert asyncio.get_running_loop().time() - start < .4
            assert host.hub.events, 'draining media must not starve the control watcher'
            sent = len(session.sent)
            await asyncio.sleep(.03)
            assert len(session.sent) == sent, 'no audio can escape after the driver is joined'
        finally:
            for child in (task, controls): child.cancel()
            await asyncio.gather(task, controls, return_exceptions=True)
    asyncio.run(scenario())


@pytest.mark.parametrize('source', ['instructions', 'tool'])
def test_call_guidance_distinguishes_interrupting_an_answer_from_ending_the_call(tmp_path, source):
    from skills.live_conversation.driver import call_tools, session_options
    if source == 'instructions':
        text = session_options(Host()).instructions
    else:
        text = next(tool.description for tool in call_tools(tmp_path) if tool.name == 'end_call')
    text = ' '.join(text.split())
    for distinction in (
        'Use end_call only when the owner wants the whole call to end',
        'says goodbye or asks to hang up',
        "An interruption such as 'stop', 'enough' or 'wait' ends the current answer, not the call",
        'acknowledge briefly and listen for the next request',
    ):
        assert distinction in text, (source, distinction)


def test_call_text_speaks_before_slow_tools_offers_sight_and_keeps_the_language_caveat():
    from skills.live_conversation.driver import CALL_INSTRUCTIONS, languages_opening_sentence

    # The owner's 18.10 remarks: silence while a tool runs, and a voice unaware it could ask to see.
    text = " ".join(CALL_INSTRUCTIONS.split())  # the block is wrapped by hand
    assert "first say in a few words what I am doing, then call it" in text
    assert "show the screen or camera (Show) or take a screenshot" in text
    recorded = languages_opening_sentence(["ru-RU"])
    assert "expects ru-RU" in recorded and "ask rather than switch" in recorded
    assert "ask when unsure" in languages_opening_sentence([])


def test_repeated_drops_without_an_answer_end_the_call_instead_of_resuming_forever():
    from skills.live_conversation.driver import UNPROVEN_RESUMES

    async def scenario():
        host = Host()
        task, session = await started(host)
        try:
            for n in range(1, UNPROVEN_RESUMES + 2):
                session.emit('ready', input_rate=24000, output_rate=24000)
                session.emit('resumable', handle=f'handle-{n}')
                session.emit('closed', resumable=True, reason='lost')
                if n <= UNPROVEN_RESUMES:
                    await until(lambda: len(host.sessions) == n + 1)  # resumed on a fresh session
                    session = host.sessions[n]
            with pytest.raises(RealtimeSetupError, match='reconnect_failed'):
                await asyncio.wait_for(task, 2)
            assert len(host.sessions) == UNPROVEN_RESUMES + 1
        finally:
            host.hub.ended_by_owner = True
            if not task.done(): await asyncio.wait_for(task, 2)
    asyncio.run(scenario())
