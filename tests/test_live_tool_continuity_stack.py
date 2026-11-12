"""The real stack holds a call together while one of its ordinary tools runs.

Native adapters over loopback wires with a held effect: the replacement connection hears
and answers before the effect ends, then receives its one whole result. The host and
fake-port layers of the same contract live in ``test_live_tool_continuity.py``.
"""
import asyncio
import json
import time

import pytest

from ouroboros import call_protocol as cp
from tests._extension_loader_shared import _clear_loader_state  # noqa: F401
from tests._live_world import host_world as host_world
from tests.test_live_tool_continuity import Held, POINTER, is_request, reached, unit_at


def native_output(wire, call_id):
    """A function result written live; a fitted setup re-creating a completed pair is not one."""
    for event in wire.sent:
        if any(row.get('id') == call_id for row in event.get('toolResponse', {}).get('functionResponses', ())):
            return True
        item = event.get('item', {}) if event.get('type') == 'conversation.item.create' else {}
        if (item.get('type') == 'function_call_output' and item.get('call_id') == call_id
                and not str(item.get('id', '')).startswith('setup_item')):
            return True
    return False


def page_state(live):
    return next((v['state'] for v in reversed(live.page.controls) if 'state' in v), '')


async def answered_while_held(live, slow, words, answer):
    """With the effect still held, the native wire takes the owner's audio and words and the adapter stack speaks
    back: the native answer, its final caption and played audio on the page, and both rows in history."""
    wire, heard, frames = live.wire, len(live.wire.audio), len(live.page.frames)
    live.provider.script.append(('say', answer))
    await live.owner_says(words)  # page microphone -> native audio -> native transcript -> page caption
    assert len(wire.audio) > heard
    await live.wait(lambda: answer in wire.spoken, 30)

    def caption():
        return next((v for v in live.page.controls if v.get('text') == answer and v.get('final')), None)
    await live.wait(caption, 30)
    await live.wait(lambda: len(live.page.frames) > frames and live.page.played.get(caption()['item_id'], 0) > 0, 30)
    await live.wait(lambda: any(row.get('content') == words for row in live.host.messages)
                    and any(str(row.get('content')).endswith(answer) for row in live.host.messages), 30)
    assert not slow.release.is_set() and live.call.tools_running and len(slow.effects) == 1, 'the effect is still held'
    assert not any('slow' in str(row.get('tool_calls')) for row in live.host.messages), 'its unit has not landed'


def fitted(live):
    session = live.call.session
    return json.loads(session.selected.messages_json) if session is not None else []


@pytest.mark.parametrize(("engine", "successor"), [
    ('openai', 'sequential'),
    ('grok', 'prepared'),
    ('gemini', 'sequential'),
])
def test_a_dropped_connection_returns_live_while_its_tool_runs(host_world, monkeypatch, engine, successor):
    from ouroboros.context_compaction import _atomic_units
    from tests._live_stack import stack
    slow = Held(monkeypatch)
    words, answer = 'Owner words while the tool runs.', 'Answered while the tool runs.'

    async def scenario():
        async with stack(host_world, monkeypatch, engine) as live:
            first = live.wire
            first.tool('slow-call', 'list_available_tools', {})
            try:
                await slow.started()
                if successor == 'prepared':  # a renewal already holds a silent successor when the socket dies
                    live.host.state['refit'] = True
                    await live.wait(lambda: live.call.preparing is not None and live.call.preparing.done(), 30)
                await first.websocket.close(code=1012, reason='synthetic service restart')
                await live.wait(lambda: live.host.connection_number == 2 and live.call.connected.is_set(), 30)
                # The page hears of LIVE through its own socket, after the call already serves.
                await live.wait(lambda: page_state(live) == cp.STATE_LIVE, 30)
                assert not slow.release.is_set(), 'LIVE again while the effect runs'
                assert live.wire is not first and len(live.provider.connections) == 2, 'its successor itself serves'
                rows = fitted(live)
                assert any(is_request(row, 'slow-call') for row in rows), 'the fit carries the pending request'
                assert not any(call.get('id') == 'slow-call' for row in rows for call in row.get('tool_calls') or ())
                await answered_while_held(live, slow, words, answer)
                assert sum(row.get('text') == words for row in live.transcript()) == 1
                assert not live.host.executed.get('slow-call') and not native_output(live.wire, 'slow-call')
            finally:
                slow.release.set()
            await live.wait(lambda: any(POINTER in json.dumps(v) for v in live.wire.sent), 60)
            await live.quiet()
            messages = live.host.messages
            at = unit_at(messages, 'slow-call')
            said = next(i for i, row in enumerate(messages) if row.get('content') == words)
            spoken = [i for i, row in enumerate(messages) if str(row.get('content')).endswith(answer)]
            assert len(spoken) == 1 and said < spoken[0] < at and messages[at + 1].get('tool_call_id') == 'slow-call'
            assert any(unit.start == at and unit.end == at + 1 for unit in _atomic_units(messages))
            assert sum('Synthetic greeting heard.' in str(row.get('content')) for row in messages) == 1, (
                'an adoption during the effect resurrects no earlier history')
            assert sum(is_request(row, 'slow-call') for row in messages) == 1
            assert any(row.get('tool_call_id') == 'slow-call' for row in fitted(live)), 'the unit arrives in setup'
            assert not any(native_output(wire, 'slow-call') for wire in live.provider.connections)
            if engine == 'grok':
                assert not any(v.get('item', {}).get('type') == 'function_call' for v in live.wire.sent)
            assert sum(POINTER in json.dumps(v) for wire in live.provider.connections for v in wire.sent) == 1
            assert sum(answer in wire.spoken for wire in live.provider.connections) == 1
            assert slow.effects == ['list_available_tools']
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())


@pytest.mark.parametrize('engine', ['openai', 'grok', 'gemini'])
def test_typed_room_mail_during_a_running_tool_is_answered_before_its_result(host_world, monkeypatch, engine):
    from tests._live_stack import stack
    slow = Held(monkeypatch)
    text, answer = 'Typed while the tool runs.', 'Answered the typed words.'

    async def scenario():
        async with stack(host_world, monkeypatch, engine) as live:
            first = live.wire
            first.tool('slow-mail', 'list_available_tools', {})
            try:
                await slow.started()
                await first.websocket.close(code=1012, reason='synthetic service restart')
                await live.wait(lambda: live.host.connection_number == 2 and live.call.connected.is_set(), 30)
                await live.wait(lambda: page_state(live) == cp.STATE_LIVE, 30)
                live.provider.script.append(('say', answer))
                live.mail.put({'text': text, 'msg_id': 'typed-while-held'})
                await live.wait(lambda: any(v.get('text') == answer and v.get('final') for v in live.page.controls), 30)
                assert any(text in str(row.get('content')) for row in live.host.messages)
                assert not slow.release.is_set() and len(slow.effects) == 1, 'the effect is still held'
            finally:
                slow.release.set()
            await live.wait(lambda: any(POINTER in json.dumps(v) for v in live.wire.sent), 60)
            await live.quiet()
            messages = live.host.messages
            typed = next(i for i, row in enumerate(messages) if text in str(row.get('content')))
            at = unit_at(messages, 'slow-mail')
            assert typed < at and messages[at + 1].get('tool_call_id') == 'slow-mail'
            assert sum(POINTER in json.dumps(v) for wire in live.provider.connections for v in wire.sent) == 1
            assert not any(native_output(wire, 'slow-mail') for wire in live.provider.connections)
            assert slow.effects == ['list_available_tools']
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())


@pytest.mark.parametrize('engine,queued', [('openai', 'ordinary'), ('openai', 'silent'), ('gemini', 'silent')])
def test_a_queued_second_batch_stays_visible_and_ordered(host_world, monkeypatch, engine, queued):
    from tests._live_stack import stack
    slow = Held(monkeypatch)
    order = []
    second = ('get_task_result', {'task_id': 'none'}) if queued == 'ordinary' else (
        'attend', {'see': None, 'look': None, 'revisit_after_s': None})  # silent: no answer is owed for it

    async def scenario():
        async with stack(host_world, monkeypatch, engine) as live:
            batch = live.host.tool_batch

            async def observed(calls):
                order.append([call['id'] for call in calls])
                return await batch(calls)
            monkeypatch.setattr(live.host, 'tool_batch', observed)
            first = live.wire
            first.tool('slow-first', 'list_available_tools', {})
            try:
                await slow.started()
                first.tool('queued-second', *second)
                await live.wait(lambda: any(is_request(row, 'queued-second') for row in live.host.messages), 30)
                await first.websocket.close(code=1012, reason='synthetic service restart')
                await live.wait(lambda: live.host.connection_number == 2 and live.call.connected.is_set(), 30)
                rows = fitted(live)
                first_at = next(i for i, row in enumerate(rows) if is_request(row, 'slow-first'))
                second_at = next(i for i, row in enumerate(rows) if is_request(row, 'queued-second'))
                assert first_at < second_at and order == [['slow-first']]
                await live.wait(lambda: page_state(live) == cp.STATE_LIVE, 30)
                await answered_while_held(live, slow, 'Words before the queue moves.', 'Answered before the queue moves.')
                assert order == [['slow-first']], 'the queued batch waits its turn while the conversation goes on'
            finally:
                slow.release.set()
            await live.wait(lambda: 'queued-second' in live.host.executed and live.call.tool_queue.empty()
                            and not live.call.tools_running, 60)
            await live.quiet()
            await live.wait(lambda: any(POINTER in json.dumps(v) for v in live.wire.sent), 60)
            await live.quiet()
            assert order == [['slow-first'], ['queued-second']] and slow.effects == ['list_available_tools']
            assert unit_at(live.host.messages, 'slow-first') < unit_at(live.host.messages, 'queued-second')
            assert not any(native_output(wire, 'queued-second') for wire in live.provider.connections)
            assert all(json.dumps(v).count(POINTER) <= 1 for wire in live.provider.connections for v in wire.sent)
            if queued == 'silent':
                assert sum(POINTER in json.dumps(v) for wire in live.provider.connections for v in wire.sent) == 1, (
                    'only the ordinary batch owes an answer')
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())


@pytest.mark.parametrize('engine', ['grok', 'openai'])
def test_completion_while_a_successor_prepares_refits_instead_of_replaying_its_call(host_world, monkeypatch, engine):
    from tests._live_stack import stack
    slow = Held(monkeypatch)

    async def scenario():
        async with stack(host_world, monkeypatch, engine) as live:
            first = live.wire
            first.tool('slow-prepared', 'list_available_tools', {})
            try:
                await slow.started()
                live.host.state['refit'] = True  # a renewal prepares a silent successor while the effect runs
                await live.wait(lambda: len(live.provider.connections) == 2, 30)
                prepared = live.provider.connections[1]
                await live.wait(lambda: prepared.sent, 30)
            finally:
                slow.release.set()
            await live.wait(lambda: 'slow-prepared' in live.host.executed, 60)
            await live.wait(lambda: any(row.get('tool_call_id') == 'slow-prepared' for row in fitted(live)), 60)
            await live.wait(lambda: any(POINTER in json.dumps(v) for v in live.wire.sent), 60)
            await live.quiet()
            assert not any(native_output(wire, 'slow-prepared') for wire in live.provider.connections)
            assert not any(v.get('item', {}).get('call_id') == 'slow-prepared' for v in prepared.sent), (
                'the snapshot taken before completion is refitted, never caught up with live items')
            serving = live.wire
            assert serving is not prepared
            items = [v['item'] for v in serving.sent if v.get('type') == 'conversation.item.create']
            if engine == 'grok':
                assert not any(item.get('type') == 'function_call' for item in items), 'xAI would empty its arguments'
            else:
                assert [item['type'] for item in items if item.get('call_id') == 'slow-prepared'] == [
                    'function_call', 'function_call_output'], 'one fitted pair, never a live replay'
            assert slow.effects == ['list_available_tools']
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())


@pytest.mark.parametrize('engine,control,dropped', [
    ('openai', 'end', False), ('openai', 'stop', False), ('grok', 'end', True),
    ('gemini', 'stop', False)])
def test_end_or_stop_during_a_running_tool_keeps_custody_and_starts_nothing_queued(host_world, monkeypatch, engine,
                                                                                  control, dropped):
    from tests._live_stack import stack
    slow = Held(monkeypatch)
    observed = {}

    async def scenario():
        async with stack(host_world, monkeypatch, engine) as live:
            first = live.wire
            first.tool('slow-end', 'list_available_tools', {})
            try:
                await slow.started()
                first.tool('never-started', 'get_task_result', {'task_id': 'none'})
                await live.wait(lambda: any(is_request(row, 'never-started') for row in live.host.messages), 30)
                if dropped:  # the fresh connection serves while the effect runs; its end has the same custody
                    await first.websocket.close(code=1012, reason='synthetic service restart')
                    await live.wait(lambda: live.host.connection_number == 2 and live.call.connected.is_set(), 30)
                    await live.wait(lambda: page_state(live) == cp.STATE_LIVE, 30)
                    await answered_while_held(live, slow, 'Words before the end.', 'Answered before the end.')
                serving = live.wire
                if control == 'end':
                    live.page.end()
                else:
                    live.host._end_reason = ('owner_stop', 'ordinary task control ended the call')
                await live.wait(lambda: serving.closed, 30)
                observed['media_closed_before_release'] = not slow.release.is_set()
                await asyncio.sleep(.1)
                observed['settled_before_release'] = live.running.done()
            finally:
                slow.release.set()
        observed['messages'] = live.host.messages
        observed['result'] = live.result
    asyncio.run(scenario())
    assert observed['media_closed_before_release'] and not observed['settled_before_release']
    assert slow.effects == ['list_available_tools'], 'the started effect finished once; the queued one never ran'
    messages = observed['messages']
    assert messages[unit_at(messages, 'slow-end') + 1].get('tool_call_id') == 'slow-end'
    assert any('never-started' in str(row.get('content')) and 'did not run' in str(row.get('content')) for row in messages)
    assert not any(call.get('id') == 'never-started' for row in messages for call in row.get('tool_calls') or ())


@pytest.mark.parametrize('engine,failure,reason', [
    ('openai', 'event', 'provider_error'), ('grok', 'close', 'provider_closed'), ('gemini', 'replacement', 'provider_error')])
def test_a_true_refusal_while_a_tool_runs_stays_terminal_and_keeps_custody(host_world, monkeypatch, engine, failure, reason):
    from tests._live_stack import stack
    from tests._live_stack_wire import NativeWire
    slow = Held(monkeypatch)
    send = NativeWire.send
    refusal = {'type': 'error', 'error': {'code': 'invalid_request', 'message': 'Synthetic refusal'}}

    async def refuses_replacement(wire, raw):
        if failure == 'replacement' and wire.number == 2:
            wire.put(refusal)
            return
        return await send(wire, raw)
    monkeypatch.setattr(NativeWire, 'send', refuses_replacement)

    async def scenario():
        async with stack(host_world, monkeypatch, engine) as live:
            first = live.wire
            first.tool('slow-refused', 'list_available_tools', {})
            try:
                await slow.started()
                if failure == 'event':
                    first.put(refusal)
                else:
                    await first.websocket.close(code=1008 if failure == 'close' else 1012, reason='synthetic close')
                expected = 2 if failure == 'replacement' else 1
                await live.wait(lambda: len(live.provider.connections) == expected and live.wire.closed and first.closed, 30)
                await asyncio.sleep(.1)
                assert not live.running.done(), 'the running effect stays in custody until it finishes'
                assert not slow.release.is_set() and page_state(live) != cp.STATE_LIVE
            finally:
                slow.release.set()
            await asyncio.wait_for(asyncio.shield(live.running), 60)
        assert live.result[1]['reason_code'] == reason, live.result
        assert len(live.provider.connections) == (2 if failure == 'replacement' else 1), 'no reconnection after a refusal'
        messages = live.host.messages
        assert messages[unit_at(messages, 'slow-refused') + 1].get('tool_call_id') == 'slow-refused'
        assert slow.effects == ['list_available_tools']
    asyncio.run(scenario())


@pytest.mark.parametrize('thinking,handle', [(False, ''), (True, ''), (False, 'before-call'), (True, 'before-call')])
def test_gemini_expiry_during_a_long_tool_continues_fresh_without_a_covering_handle(host_world, monkeypatch, thinking,
                                                                                    handle):
    from tests._live_stack import stack
    slow = Held(monkeypatch)

    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini', thinking=thinking) as live:
            first = live.wire
            if handle:  # issued before the call, it cannot prove the resumed conversation holds the call
                first.put({'sessionResumptionUpdate': {'resumable': True, 'newHandle': handle}})
                await live.wait(lambda: live.host.state.get('resume_handle') == handle, 30)
            first.tool('slow-expiry', 'list_available_tools', {})
            try:
                await slow.started()
                first.put({'goAway': {'timeLeft': '0.05s'}})
                await asyncio.sleep(.3)  # past the deadline: no renewal_pending_work; the open call holds quiet
                assert not live.running.done()
                await first.websocket.close(code=1001, reason='synthetic session expiry')
                await live.wait(lambda: live.host.connection_number >= 2 and live.call.connected.is_set(), 30)
                assert live.wire.sent[0]['setup']['sessionResumption'] == {}
                assert any(is_request(row, 'slow-expiry') for row in fitted(live))
            finally:
                slow.release.set()
            await live.wait(lambda: any(POINTER in json.dumps(v) for v in live.wire.sent), 60)
            await live.quiet()
            assert not any(native_output(wire, 'slow-expiry') for wire in live.provider.connections)
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())


def force_look_order(order, live, monkeypatch):
    """One real order of the look's unit, its pixels and the fresh connection's snapshot and adoption."""
    host, call = live.host, live.call
    if order == 'snapshot_between_unit_and_pixels':
        from ouroboros.live_connections import PreparedConnection
        fit, observe, sources = PreparedConnection._fit, host.observe_context, []

        async def noting_fit(lease, credentials, *, rebound=False):
            sources.append(lease.source)
            return await fit(lease, credentials, rebound=rebound)

        async def held_pixels(row):  # a successor's snapshot already holds the unit, not yet these pixels
            if isinstance(row.get('content'), list):
                await reached(lambda: any(item.get('id') == 'queued-look' for source in sources
                                          for rows in source for item in rows.get('tool_calls') or ()))
            return await observe(row)
        monkeypatch.setattr(PreparedConnection, '_fit', noting_fit)
        monkeypatch.setattr(host, 'observe_context', held_pixels)
    elif order == 'pixels_after_adoption':
        observe = host.observe_context

        async def held_pixels(row):
            if isinstance(row.get('content'), list):
                await reached(lambda: call.conversation >= 3)
            return await observe(row)
        monkeypatch.setattr(host, 'observe_context', held_pixels)
    elif order == 'adoption_before_unit':
        batch = host.tool_batch

        async def held_look(calls):
            if any(item['id'] == 'queued-look' for item in calls):  # the frame already went live with the first pointer
                await reached(lambda: call.conversation >= 3 and any(POINTER in json.dumps(v) for v in live.wire.sent))
            return await batch(calls)
        monkeypatch.setattr(host, 'tool_batch', held_look)


@pytest.mark.parametrize('order', ['natural', 'snapshot_between_unit_and_pixels', 'adoption_before_unit'])
def test_a_queued_requested_look_reaches_the_fresh_connection_once_before_its_pointer(host_world, monkeypatch, order):
    import base64
    import hashlib
    from tests._live_stack import stack
    from tests._live_stack_page import jpeg as synthetic_frame
    slow = Held(monkeypatch)
    frame = synthetic_frame('queued-look')

    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini') as live:
            live.page.video(frame)
            live.page.share(True, 'screen')
            await live.wait(lambda: any('started showing the screen' in str(row) for row in live.host.messages), 30)
            await live.wait(lambda: any(f.data == frame for f in live.call.attention.frames.values()), 30)
            await live.wait(lambda: not live.call.pending_offers and live.call.quiet(), 30)
            force_look_order(order, live, monkeypatch)
            first = live.wire
            first.tool('slow-look', 'list_available_tools', {})
            try:
                await slow.started()
                first.tool('queued-look', 'attend', {'see': 'asked', 'look': True, 'revisit_after_s': None})
                await live.wait(lambda: any(is_request(row, 'queued-look') for row in live.host.messages), 30)
                await first.websocket.close(code=1012, reason='synthetic service restart')
                await live.wait(lambda: live.host.connection_number == 2 and live.call.connected.is_set(), 30)
            finally:
                slow.release.set()
            # Executed is not routed: a refit the look still owes may begin only after it returns.
            await live.wait(lambda: 'queued-look' in live.host.executed and live.call.tool_queue.empty()
                            and not live.call.tools_running, 60)
            await live.quiet()
            await live.wait(lambda: any(POINTER in json.dumps(v) for v in live.wire.sent), 60)
            await live.quiet()
            encoded = base64.b64encode(frame).decode()
            events = [json.dumps(v) for v in live.wire.sent]
            arrivals = [i for i, event in enumerate(events) if encoded in event]
            pointer = next(i for i, event in enumerate(events) if POINTER in event)
            assert len(arrivals) == 1 and arrivals[0] < pointer, "the look's exact pixels arrive once, before its pointer"
            assert '_requested_look' not in events[pointer], 'a private row key never reaches the wire'
            assert not any(native_output(wire, 'queued-look') for wire in live.provider.connections)
            fit = encoded in live.call.session.selected.messages_json
            frame_lane = 'video' in live.wire.sent[arrivals[0]].get('realtimeInput', {})
            if order == 'snapshot_between_unit_and_pixels':  # adoption's catch-up, after setup, before consume
                assert not fit and not frame_lane and 'clientContent' in live.wire.sent[arrivals[0]]
            elif order == 'pixels_after_adoption':
                assert fit, 'the session adopted before the pixels were recorded is refitted'
            elif order == 'adoption_before_unit':  # visible before the look, so it has no row: delivered as requested
                assert not fit and frame_lane
            assert hashlib.sha256(frame).hexdigest() in live.call.attention.seen.values(), 'never offered live again'
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())


@pytest.mark.parametrize('engine', ['openai', 'grok', 'gemini'])
def test_a_frame_shared_while_a_tool_runs_reaches_the_fresh_connection(host_world, monkeypatch, engine):
    import base64
    from tests._live_stack import stack
    from tests._live_stack_page import jpeg as synthetic_frame
    slow = Held(monkeypatch)
    frame = synthetic_frame('shared-while-held')

    async def scenario():
        async with stack(host_world, monkeypatch, engine) as live:
            first = live.wire
            first.tool('slow-frame', 'list_available_tools', {})
            try:
                await slow.started()
                await first.websocket.close(code=1012, reason='synthetic service restart')
                await live.wait(lambda: live.host.connection_number == 2 and live.call.connected.is_set(), 30)
                await live.wait(lambda: page_state(live) == cp.STATE_LIVE, 30)
                live.page.video(frame)
                live.page.share(True, 'screen')
                await live.wait(lambda: any('started showing the screen' in str(row) for row in live.host.messages), 30)
                await live.wait(lambda: any(f.data == frame for f in live.call.attention.frames.values()), 30)
                await live.wait(lambda: not live.call.pending_offers and live.call.quiet(), 30)
                await answered_while_held(live, slow, 'What do you see?', 'Answered while seeing the frame.')
                encoded = base64.b64encode(frame).decode()
                arrivals = sum(encoded in json.dumps(v) for v in live.wire.sent)
                if live.host.model.vision:
                    assert arrivals == 1, 'the seeing engine has the frame before the effect ends, once'
                else:
                    assert not arrivals and any('blind' in str(row.get('content')) for row in live.host.messages)
            finally:
                slow.release.set()
            await live.wait(lambda: any(POINTER in json.dumps(v) for v in live.wire.sent), 60)
            await live.quiet()
            messages = live.host.messages
            assert messages[unit_at(messages, 'slow-frame') + 1].get('tool_call_id') == 'slow-frame'
            assert not any(native_output(wire, 'slow-frame') for wire in live.provider.connections)
            assert slow.effects == ['list_available_tools']
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())


@pytest.mark.parametrize('engine,handle', [('openai', ''), ('gemini', ''), ('openai', 'cut'), ('gemini', 'before-write')])
def test_a_result_written_just_before_a_drop_is_answered_once(host_world, monkeypatch, engine, handle):
    from tests._live_stack import stack

    def update():
        return {'sessionResumptionUpdate': {'resumable': True, 'newHandle': handle}}

    async def scenario():
        async with stack(host_world, monkeypatch, engine) as live:
            first = live.wire
            # The engine took the result and has not begun its answer, or the drop cuts it off.
            first.answer = (lambda: first.say('Cut off by the drop.', complete=False)) if handle == 'cut' else (lambda: None)
            if handle == 'before-write':  # a real handle, read in order, that predates the native result
                first.put(update())
                await live.wait(lambda: live.host.state.get('resume_handle') == handle, 30)
            first.tool('written-call', 'get_task_result', {'task_id': 'none'})
            await live.wait(lambda: native_output(first, 'written-call'), 30)
            if handle == 'cut':
                await live.wait(lambda: 'Cut off by the drop.' in first.spoken and live.call.audio_ms, 30)
            if handle.startswith('after-write'):
                first.put(update())
                await live.wait(lambda: live.call.session.resume_after_tool_response, 30)
            await first.websocket.close(code=1012, reason='synthetic service restart')
            await live.wait(lambda: live.host.connection_number == 2 and live.call.connected.is_set(), 30)
            pointers = lambda: sum(POINTER in json.dumps(v) for wire in live.provider.connections for v in wire.sent)
            if handle.startswith('after-write'):  # the server's own later handle holds the result: same conversation
                assert live.wire.sent[0]['setup']['sessionResumption'] == {'handle': handle}
                if handle == 'after-write':
                    live.wire.say('Answered on the resumed conversation.')
                    await live.quiet()
                # Unanswered, the resumed turn stays open for its native answer: quiet waits for it, by design.
                assert not pointers() and live.call.conversation == 1
                # The answer is owed across the resume until it is given there; a later fresh switch knows which.
                live.wire.put({'sessionResumptionUpdate': {'resumable': False}})
                await live.wait(lambda: not live.host.state.get('resume_handle'), 30)
                await live.wire.websocket.close(code=1012, reason='synthetic service restart')
                await live.wait(lambda: live.host.connection_number == 3 and live.call.connected.is_set(), 30)
                if handle != 'after-write':
                    await live.wait(lambda: any(POINTER in json.dumps(v) for v in live.wire.sent), 30)
                await live.quiet()
                await asyncio.sleep(.2)  # an owed pointer is offered at the first quiet moment
                assert pointers() == int(handle != 'after-write') and live.call.conversation == 2
            else:
                await live.wait(lambda: any(POINTER in json.dumps(v) for v in live.wire.sent), 30)
                await live.quiet()
                assert pointers() == 1 and live.call.conversation == 2
                if engine == 'gemini':
                    assert live.wire.sent[0]['setup']['sessionResumption'] == {}
            assert not native_output(live.wire, 'written-call'), 'never written a second time'
            assert sum(row.get('tool_call_id') == 'written-call' for row in live.host.trace['tool_calls']) == 1
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())


def test_a_handle_withdrawn_at_the_resume_boundary_starts_a_fresh_conversation(host_world, monkeypatch):
    from tests._live_stack import stack

    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini') as live:
            call, first = live.call, live.wire
            first.put({'sessionResumptionUpdate': {'resumable': True, 'newHandle': 'soon-withdrawn'}})
            await live.wait(lambda: live.host.state.get('resume_handle') == 'soon-withdrawn', 30)
            join, withdrawn = call.rows.join, []

            async def read_during_join():
                # Quiet was observed and the native reader still runs while the exit joins canonical rows.
                if call.renewing and call.resume and not withdrawn:
                    withdrawn.append(True)
                    first.put({'sessionResumptionUpdate': {'resumable': False}})
                    await reached(lambda: live.host.state.get('resume_handle') == '')
                await join()
            monkeypatch.setattr(call.rows, 'join', read_during_join)
            first.put({'goAway': {'timeLeft': '30s'}})
            await live.wait(lambda: len(live.provider.connections) == 2 and call.connected.is_set(), 30)
            await live.quiet()
            assert withdrawn and live.wire.sent[0]['setup']['sessionResumption'] == {}
            assert call.conversation == 2 and not call.connection_resumed, 'a fresh fit is a new provider conversation'
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())


@pytest.mark.parametrize('engine,held,handle', [('openai', False, ''), ('grok', True, ''), ('gemini', True, ''),
                                                ('gemini', False, 'covering')])
def test_a_drop_while_the_owner_speaks_leaves_a_hearing_gap_fact_never_owner_words(host_world, monkeypatch, engine, held,
                                                                                    handle):
    from tests._live_stack import stack
    slow = Held(monkeypatch)

    def gaps(messages):
        return [row for row in messages if row.get('_host_context_kind') == 'call_fact'
                and 'may not have been heard' in row['content']]

    async def scenario():
        async with stack(host_world, monkeypatch, engine) as live:
            directives = list(getattr(live.host.ctx, '_owner_directives', None) or ())
            owner_rows = sum(row.get('_host_context_kind') == 'owner_words' for row in live.host.messages)
            first = live.wire
            if handle:
                first.put({'sessionResumptionUpdate': {'resumable': True, 'newHandle': handle}})
                await live.wait(lambda: live.host.state.get('resume_handle') == handle, 30)
            try:
                if held:
                    first.tool('slow-gap', 'list_available_tools', {})
                    await slow.started()
                heard = len(first.audio)
                first.input_start('owner-cut-off')
                await live.wait(lambda: len(first.audio) >= heard + 960 and live.call.input_open, 30)
                await first.websocket.close(code=1012, reason='synthetic service restart')
                await live.wait(lambda: live.host.connection_number == 2 and live.call.connected.is_set(), 30)
                if handle:  # the owner's gap fact is for fresh recovery; a resume's retention of partial speech is unproven
                    assert live.wire.sent[0]['setup']['sessionResumption'] == {'handle': handle}
                    live.provider.script.append(('say', 'Answered on the resumed conversation.'))
                    live.wire.input_end('owner-cut-off', 'Finished on the resumed connection.')
                    await live.quiet()
                    assert not gaps(live.host.messages)
                    return
                await live.wait(lambda: gaps(live.host.messages), 30)
                if held:
                    await live.wait(lambda: page_state(live) == cp.STATE_LIVE, 30)
                    assert any(is_request(row, 'slow-gap') for row in fitted(live))
                    await answered_while_held(live, slow, 'Said again after the gap.', 'Answered after the gap.')
            finally:
                slow.release.set()
            if held:
                await live.wait(lambda: any(POINTER in json.dumps(v)
                                           for wire in live.provider.connections for v in wire.sent), 60)
            await live.quiet()
            assert len(gaps(live.host.messages)) == 1
            assert sum(row.get('_host_context_kind') == 'owner_words' for row in live.host.messages) == owner_rows + int(held)
            assert not any('may not have been heard' in str(row.get('text')) for row in live.transcript())
            kept = list(getattr(live.host.ctx, '_owner_directives', None) or ())
            assert kept[:len(directives)] == directives and [row['content'] for row in kept[len(directives):]] == (
                ['Said again after the gap.'] if held else []), 'only words actually recognised become directives'
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())



@pytest.mark.parametrize('crossing', [True, False])
def test_a_queued_view_request_answers_to_the_view_that_issued_it(host_world, monkeypatch, crossing):
    """compact_context's public promise: an omitted revision binds the send that produced this call.

    A fresh fit rebuilds its facts row, so that view no longer prefixes history: the request is refused
    with its receipt, never applied to a successor's view holding facts its author never saw.
    """
    from tests._live_stack import stack
    slow = Held(monkeypatch)
    note, arrival = 'Folded from the issuing view.', 'may not have been heard' if crossing else 'started showing the screen'

    async def scenario():
        async with stack(host_world, monkeypatch, 'openai') as live:
            first = live.wire
            first.tool('slow-view', 'list_available_tools', {})
            try:
                await slow.started()
                first.tool('queued-fold', 'compact_context', {'working_note': note, 'keep_unit_ids': []})
                await live.wait(lambda: any(is_request(row, 'queued-fold') for row in live.host.messages), 30)
                if crossing:
                    await first.websocket.close(code=1012, reason='synthetic service restart')
                    await live.wait(lambda: live.host.connection_number == 2 and live.call.connected.is_set(), 30)
                    second, heard = live.wire, len(live.wire.audio)
                    second.input_start('owner-cut-off')
                    await live.wait(lambda: len(second.audio) >= heard + 960 and live.call.input_open, 30)
                    await second.websocket.close(code=1012, reason='synthetic service restart')
                    await live.wait(lambda: live.host.connection_number == 3 and live.call.connected.is_set(), 30)
                    batch = live.host.tool_batch

                    async def after_the_refit(calls):
                        if any(item['id'] == 'queued-fold' for item in calls):
                            # This successor's fit holds the fact and still prefixes history when the view applies.
                            await reached(lambda: live.call.conversation >= 4
                                          and arrival in json.dumps(live.host.ctx._last_context_observation['messages']))
                        return await batch(calls)
                    monkeypatch.setattr(live.host, 'tool_batch', after_the_refit)
                else:
                    live.page.share(True, 'screen')
                await live.wait(lambda: arrival in json.dumps(live.host.messages), 30)
            finally:
                slow.release.set()
            # The view is published under the history lock after its receipt: the model is told either way.
            await live.wait(lambda: any('[Context view receipt]' in str(row.get('content')) for row in live.host.messages), 60)
            await live.quiet()
            receipt = live.host.ctx._context_view_receipt
            assert arrival in json.dumps(live.host.messages), 'a fact newer than the issuing view is an arrival, never folded'
            assert receipt['status'] == ('binding_mismatch' if crossing else 'applied'), receipt
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())


@pytest.mark.parametrize('engine,thinking', [('openai', False), ('grok', False), ('gemini', False), ('gemini', True)])
def test_an_answered_result_owes_nothing_after_a_later_drop(host_world, monkeypatch, engine, thinking):
    from tests._live_stack import stack
    answer = 'Answered the result once.'

    async def scenario():
        async with stack(host_world, monkeypatch, engine, thinking=thinking) as live:
            first = live.wire
            live.provider.script.append(('say', answer))
            first.tool('answered-call', 'get_task_result', {'task_id': 'none'})
            await live.wait(lambda: answer in first.spoken and any(
                str(row.get('content')).endswith(answer) for row in live.host.messages), 30)
            await live.quiet()
            spoken = len(first.spoken)
            await first.websocket.close(code=1012, reason='synthetic service restart')
            await live.wait(lambda: live.host.connection_number == 2 and live.call.connected.is_set(), 30)
            await live.quiet()
            await asyncio.sleep(.2)  # an owed pointer would have been offered at this quiet boundary
            assert not live.call.pending_offers and not live.wire.spoken and len(first.spoken) == spoken, (
                'a drop later in the call is no reason to speak unprompted')
            assert not any(POINTER in json.dumps(v) for wire in live.provider.connections for v in wire.sent)
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())


@pytest.mark.parametrize('thinking,owed', [(False, False), (True, False), (False, True)])
def test_a_batch_landing_while_a_native_resume_opens_continues_fresh(host_world, monkeypatch, thinking, owed):
    from ouroboros.live_connections import PreparedConnection
    from tests._live_stack import stack
    slow = Held(monkeypatch)
    adopt = PreparedConnection._adopt

    async def adopt_after_the_unit(lease):
        if lease.options.resume_handle:  # the resumed socket is open; history moves before its adoption
            slow.release.set()
            await reached(lambda: 'late-call' in lease.host.executed)
        return await adopt(lease)
    monkeypatch.setattr(PreparedConnection, '_adopt', adopt_after_the_unit)

    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini', thinking=thinking) as live:
            first = live.wire
            if owed:  # a written result still awaits its answer when the old connection dies
                first.answer = lambda: None
                first.tool('written-first', 'get_task_result', {'task_id': 'none'})
                await live.wait(lambda: native_output(first, 'written-first') and not live.call.tools_running, 30)
                complete, first.complete = first.complete, lambda: None
            first.tool('late-call', 'list_available_tools', {})
            if owed:
                first.complete = complete
            try:
                await slow.started()
                first.put({'sessionResumptionUpdate': {'resumable': True, 'newHandle': 'after-call'}})
                await live.wait(lambda: live.host.state.get('resume_handle') == 'after-call', 30)
                await first.websocket.close(code=1012, reason='synthetic service restart')
                await live.wait(lambda: any(POINTER in json.dumps(v) for v in live.wire.sent), 60)
            finally:
                slow.release.set()
            await live.quiet()
            assert [wire.sent[0]['setup']['sessionResumption'] for wire in live.provider.connections] == [
                {}, {'handle': 'after-call'}, {}], 'the spent handle is never opened twice'
            assert live.call.conversation == 2 and not live.call.connection_resumed
            assert any(row.get('tool_call_id') == 'late-call' for row in fitted(live)), 'the unit arrives in setup'
            assert not any(native_output(wire, 'late-call') for wire in live.provider.connections)
            assert sum(POINTER in json.dumps(v) for wire in live.provider.connections for v in wire.sent) == 1
            assert slow.effects == ['list_available_tools']
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())


@pytest.mark.parametrize('engine,further', [('openai', 'issuing'), ('openai', 'ordinary'), ('openai', 'silent')])
def test_a_further_call_behind_the_running_effect_delays_only_openais_answer(host_world, monkeypatch, engine, further):
    """The disclosed limit: OpenAI's adapter holds response.create while its own call awaits an output,
    on the connection that issued the running call or after a further call queued behind it.

    The words are heard and recorded on every engine; nothing is lost. Whether that wire accepts a
    response with an unanswered call is unverified, so this pins current behaviour, never a claim.
    """
    from tests._live_stack import stack
    slow = Held(monkeypatch)
    words, answer = 'Words after a further call.', 'Answered after a further call.'
    call = ('get_task_result', {'task_id': 'none'}) if further == 'ordinary' else (
        'attend', {'see': None, 'look': None, 'revisit_after_s': None})  # the instructed way to stay silent

    async def scenario():
        async with stack(host_world, monkeypatch, engine) as live:
            first = live.wire
            first.tool('slow-first', 'list_available_tools', {})
            waiting = 'slow-first' if further == 'issuing' else 'queued-further'
            try:
                await slow.started()
                fresh = first
                if further != 'issuing':
                    await first.websocket.close(code=1012, reason='synthetic service restart')
                    await live.wait(lambda: live.host.connection_number == 2 and live.call.connected.is_set(), 30)
                    await live.wait(lambda: page_state(live) == cp.STATE_LIVE, 30)
                    fresh = live.wire
                    fresh.tool('queued-further', *call)
                await live.wait(lambda: any(is_request(row, waiting) for row in live.host.messages), 30)
                live.provider.script.append(('say', answer))
                await live.owner_says(words)
                await live.wait(lambda: any(row.get('content') == words for row in live.host.messages), 30)
                if engine == 'openai':
                    await live.wait(lambda: live.call.session.want_response, 30)
                    assert waiting in live.call.session.waiting_tools and answer not in fresh.spoken
                else:
                    await live.wait(lambda: answer in fresh.spoken, 30)
                assert not slow.release.is_set() and len(slow.effects) == 1
            finally:
                slow.release.set()
            await live.wait(lambda: any(answer in wire.spoken for wire in live.provider.connections), 60)
            await live.wait(lambda: waiting in live.host.executed and not live.call.tools_running, 60)
            await live.quiet()
            assert sum(row.get('text') == words for row in live.transcript()) == 1
            assert sum(answer in wire.spoken for wire in live.provider.connections) == 1
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())


@pytest.mark.parametrize('engine,thinking,case,owed', [
    ('grok', False, 'owner_answer_finishes_after_queueing', 1),
    ('gemini', False, 'call_turn_keeps_talking', 1),
    ('gemini', True, 'thinking_then_answer', 0)])
def test_only_a_turn_begun_after_delivery_settles_a_written_result(host_world, monkeypatch, engine, thinking, case, owed):
    """Through real adapters: speech of a response begun before the result was delivered, its transcript tail
    or its closing receipt never answers that result; a later fresh session gets one pointer. Extended thinking
    may continue the call's own turn after the result (``call_turn_continues``): no native signal tells that from
    talk-over, so the answer stays owed and the neutral pointer leaves the choice to the mind and its history."""
    from tests._live_stack import owed_state, stack
    slow = Held(monkeypatch)
    in_progress = {'serverContent': {'turnComplete': True, 'interactionStatus': 'IN_PROGRESS'}}

    async def scenario():
        async with stack(host_world, monkeypatch, engine, thinking=thinking) as live:
            first, call = live.wire, live.call
            known = set(call.receipts)
            try:
                if case == 'owner_answer_finishes_after_queueing':
                    first.tool('held-call', 'list_available_tools', {})
                    await slow.started()
                    live.provider.script.append(('partial', 'Answering the owner meanwhile.'))
                    await live.owner_says('Words while the tool runs.')  # xAI VAD begins its answer to them
                    await live.wait(lambda: 'Answering the owner meanwhile.' in first.spoken and call.active_responses, 30)
                    first.answer = lambda: None  # the drop comes before any answer to the result
                    slow.release.set()
                    await live.wait(lambda: 'held-call' in live.host.executed and not call.tools_running, 60)
                    assert not native_output(first, 'held-call'), 'queued behind the answer to the owner'
                    item = f'item-{first.number}-{first.responses}'
                    first.put({'type': 'response.output_audio_transcript.done', 'item_id': item,
                               'transcript': 'Answering the owner meanwhile.'})
                    first.complete()  # that answer's receipt releases the queued result
                    await live.wait(lambda: native_output(first, 'held-call') and not call.active_responses, 30)
                elif case in ('call_turn_keeps_talking', 'call_turn_continues'):
                    first.answer = lambda: None
                    complete, first.complete = first.complete, lambda: None
                    first.tool('turn-call', 'get_task_result', {'task_id': 'none'})  # NON_BLOCKING: its turn stays open
                    first.complete = complete
                    await live.wait(lambda: native_output(first, 'turn-call'), 30)
                    words = 'Still talking in the call turn.'
                    first.say(words)  # the same local response speaks after delivery, then completes
                    await live.wait(lambda: not call.response_open and any(
                        str(row.get('content')).endswith(words) for row in live.host.messages), 30)
                elif case == 'call_turn_tail_then_idle':
                    first.answer = lambda: None
                    first.put({'toolCall': {'functionCalls': [{'id': 'turn-call', 'name': 'get_task_result',
                                                               'args': {'task_id': 'none'}}]}})
                    first.put({**in_progress, 'usageMetadata': first.usage()})  # the interaction goes on
                    await live.wait(lambda: native_output(first, 'turn-call'), 30)
                    first.put({'serverContent': {'outputTranscription': {'text': 'A late tail of that turn.'}}})
                    first.put({'serverContent': {'turnComplete': True, 'interactionStatus': 'IDLE'}})  # its receipt again
                    await live.wait(lambda: not call.response_open, 30)
                else:  # the call's turn ends IDLE before its result; the result begins a reasoning phase
                    words = 'Answered after thinking.'
                    first.answer = (lambda: (first.put({**in_progress, 'usageMetadata': first.usage()}), first.say(words))
                                    if case == 'thinking_then_answer' else
                                    first.put({**in_progress, 'usageMetadata': first.usage()}))
                    first.tool('think-call', 'get_task_result', {'task_id': 'none'})
                    await live.wait(lambda: native_output(first, 'think-call') and len(set(call.receipts) - known) >= 2, 30)
                    if case == 'thinking_then_answer':
                        await live.wait(lambda: words in first.spoken and any(
                            str(row.get('content')).endswith(words) for row in live.host.messages), 30)
                        await live.quiet()
            finally:
                slow.release.set()
            written = [cid for cid in ('held-call', 'turn-call', 'think-call') if cid in first.outputs]
            await first.websocket.close(code=1012, reason='synthetic service restart')
            await live.wait(lambda: live.host.connection_number == 2 and call.connected.is_set(), 30)
            pointers = lambda: sum(POINTER in json.dumps(v) for wire in live.provider.connections for v in wire.sent)
            await live.quiet()
            deadline = time.monotonic() + 1  # an owed pointer is offered at the first quiet moment
            while time.monotonic() < deadline and not pointers():
                await asyncio.sleep(.01)
            await live.quiet()
            assert pointers() == owed, (case, pointers(), owed_state(call))
            assert len(written) == 1 and not any(native_output(wire, cid) for wire in live.provider.connections[1:]
                                                 for cid in written), 'its one native output, never repeated'
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())


@pytest.mark.parametrize('engine,thinking,answered', [
    ('openai', False, False), ('grok', False, False), ('gemini', False, False), ('gemini', True, False),
    ('openai', False, True), ('gemini', True, True)])
def test_an_unanswered_result_stays_owed_across_successive_fresh_losses(host_world, monkeypatch, engine, thinking, answered):
    """Through real adapters: a fresh session's pointer carries the result's answer obligation. Unanswered when
    that session drops, the next fresh session is owed the same one pointer; answered there, a later drop is silent."""
    from tests._live_stack import stack
    from tests._live_stack_wire import NativeWire
    answer = NativeWire.answer

    async def scenario():
        async with stack(host_world, monkeypatch, engine, thinking=thinking) as live:
            monkeypatch.setattr(NativeWire, 'answer', lambda wire: answer(wire) if answered and wire.number == 2 else None)
            first = live.wire
            first.tool('owed-call', 'get_task_result', {'task_id': 'none'})
            await live.wait(lambda: native_output(first, 'owed-call') and not live.call.tools_running, 30)
            await first.websocket.close(code=1012, reason='synthetic service restart')
            await live.wait(lambda: live.host.connection_number == 2 and live.call.connected.is_set(), 30)
            second = live.wire
            await live.wait(lambda: any(POINTER in json.dumps(v) for v in second.sent), 30)
            if answered:
                await live.wait(lambda: second.spoken, 30)
                await live.quiet()
            await second.websocket.close(code=1012, reason='synthetic service restart')
            await live.wait(lambda: live.host.connection_number == 3 and live.call.connected.is_set(), 30)
            third = live.wire
            pointers = lambda: sum(POINTER in json.dumps(v) for v in third.sent)
            if answered:  # nothing is owed: a bounded look at the fresh session finds no pointer
                deadline = time.monotonic() + 1
                while time.monotonic() < deadline and not pointers():
                    await asyncio.sleep(.01)
            else:  # the owed pointer waits for the first quiet moment, however late a loaded host reaches it
                await live.wait(pointers, 30)
            await asyncio.sleep(.05)
            assert pointers() == int(not answered), (engine, thinking, answered, pointers())
            assert not third.spoken, 'a settled answer is never asked for again'
            assert not any(native_output(wire, 'owed-call') for wire in live.provider.connections[1:])
            assert not any('_tool_answer' in json.dumps(frame) for wire in live.provider.connections for frame in wire.sent)
            if not answered:  # the private marker stays on the adapter's record, where its delivery is read
                assert any('owed-call' in (row.get('_tool_answer') or ()) for row in live.call.session.held_context_rows)
                assert live.call.answer_delivered()
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())


@pytest.mark.parametrize('engine', ['openai', 'gemini'])
def test_a_fresh_switch_during_a_delivery_measure_routes_the_look_through_its_fit(host_world, monkeypatch, engine):
    """Through real adapters: the issuing socket drops while the look's serving delivery is measured. The fresh
    session already fits the unit; it is owed the look's answer, never sent the old function output."""
    import threading
    from tests._live_stack import stack
    from tests._live_stack_page import jpeg as synthetic_frame
    fitting, release = threading.Event(), threading.Event()

    async def scenario():
        async with stack(host_world, monkeypatch, engine) as live:
            live.call.attention.mode = 'asked'
            live.page.video(synthetic_frame('delivery-measure-race'))
            await live.wait(lambda: live.call.attention.frames, 30)
            measure = live.host.delivery_fits

            async def gated(rows, schemas):
                result = await measure(rows, schemas)
                if any(row.get('tool_call_id') == 'look-race' for row in rows):
                    assert result, 'the real measure admits this delivery before the gate'
                    fitting.set()
                    while not release.is_set():
                        await asyncio.sleep(.001)
                return result
            monkeypatch.setattr(live.host, 'delivery_fits', gated)
            first = live.wire
            first.tool('look-race', 'attend', {'see': None, 'look': True, 'revisit_after_s': None})
            try:
                await live.wait(fitting.is_set, 30)
                await first.websocket.close(code=1012, reason='synthetic service restart')
                await live.wait(lambda: live.host.connection_number == 2 and live.call.connected.is_set(), 30)
                second = live.wire
            finally:
                release.set()
            await live.wait(lambda: not live.call.tools_running and any(POINTER in json.dumps(v) for v in second.sent), 30)
            await live.quiet()
            assert 'look-race' in live.call.session.selected.messages_json, 'the fresh fit holds the unit'
            assert not any(native_output(wire, 'look-race') for wire in live.provider.connections), 'no orphan output'
            assert sum(POINTER in json.dumps(v) for wire in live.provider.connections for v in wire.sent) == 1
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())
