"""What the owner said, at a connection boundary: being said, awaiting recognition, recognised, answered.

The adapter knows which words are still being said or await their transcript (``input_pending``). A planned forced switch
waits for those, though never for an open tool; a real loss leaves the hearing-gap fact, with no invented words. Recognised
words a response has not answered stay owed: a fresh session gets one neutral pointer, every fresh session until a response
answers, and the mind may still stay silent. Real adapters over loopback, then the fake port for orderings a scenario
cannot make a provider produce on demand.
"""
import asyncio
import json

import pytest

from ouroboros import call_protocol as cp
from tests._extension_loader_shared import _clear_loader_state  # noqa: F401
from tests._live_world import host_world as host_world
from tests.test_live_deferred_custody import answer, mail_updates, successor
from tests.test_live_delivery_owner import never, opened, room_pointers
from tests.test_live_delivery_owner_stack import RESTART, handle, live_frames, spoken
from tests.test_live_room_obligation import (answers_only_on, asked, canonical, fresh_after, mailbox_of,
                                             no_private_marker_on_any_wire, respond_ended, resumed, room_frames, room_mail,
                                             spoken_out)
from tests.test_live_tool_continuity import POINTER, Held
from tests.test_live_tool_continuity_stack import native_output, page_state
from tests._cases import chosen_rows

WORDS = 'Please also cancel my dentist appointment on Friday.'
OLDER = 'First phrase, recognised.'
QUESTION = 'What time is my flight tomorrow?'
GAP = 'may not have been heard'
ROUTES = [('openai', False), ('grok', False), ('gemini', True)]
ROUTE_IDS = ['openai', 'grok', 'gemini-thinking']


def gaps(live):
    return sum(GAP in str(row.get('content')) for row in live.host.messages)


def stop_without_transcript(wire, item):
    """What the provider sends at end of speech; the transcript comes later."""
    wire.put({'type': 'input_audio_buffer.speech_stopped', 'item_id': item})
    wire.put({'type': 'input_audio_buffer.committed', 'item_id': item})


def transcript(wire, item, text):
    wire.put({'type': 'conversation.item.input_audio_transcription.completed', 'item_id': item, 'transcript': text,
              'status': 'completed', 'usage': {'type': 'tokens', 'input_tokens': 10, 'output_tokens': 3,
                                               'input_token_details': {'text_tokens': 2, 'audio_tokens': 8}}})


def in_room(live, text=WORDS):
    return sum(row.get('text') == text for row in live.transcript())


async def fresh_during_an_effect(live, slow):
    """A batch runs on the first connection, which drops: its result will finish on the fresh one the owner now talks on."""
    first = live.wire
    first.tool('slow', 'list_available_tools', {})
    await slow.started()
    await first.websocket.close(**RESTART)
    await live.wait(lambda: live.host.connection_number == 2 and live.call.connected.is_set(), 30)
    await live.wait(lambda: page_state(live) == cp.STATE_LIVE, 30)
    return live.wire


async def ready_to_switch(live):
    await live.wait(lambda: live.call.tool_refit.is_set() and live.call.preparation_ready(), 60)


async def switched(live, wire):
    await live.wait(lambda: wire.closed and live.call.connected.is_set() and live.call.session is not None
                    and not live.call.tool_refit.is_set(), 60)
    return live.wire


# --- a planned forced switch ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize('engine', ['openai', 'grok'])
def test_real_forced_switch_waits_for_words_awaiting_their_transcript(host_world, monkeypatch, engine):
    """The owner's chosen flow, talking on while a tool finishes. The words reach history and the fit, and stay owed."""
    from tests._live_stack import stack
    slow = Held(monkeypatch)

    async def scenario():
        async with stack(host_world, monkeypatch, engine) as live:
            answers_only_on(monkeypatch, 3)
            try:
                second = await fresh_during_an_effect(live, slow)
                second.input_start('owner-next')
                await live.wait(lambda: live.call.input_open, 30)
                assert live.call.session.input_pending
            finally:
                slow.release.set()
            await ready_to_switch(live)
            stop_without_transcript(second, 'owner-next')
            await live.wait(lambda: not live.call.session.speech_active, 30)
            assert live.call.session.input_pending and not live.call.input_open, 'VAD ended; recognition has not'
            await asyncio.sleep(.3)
            assert not second.closed, 'held for words the adapter knows are coming'
            transcript(second, 'owner-next', WORDS)
            third = await switched(live, second)
            await live.wait(lambda: room_frames(third) and third.spoken, 30)
            assert canonical(live, WORDS) == 1 and in_room(live) == 1 and not gaps(live)
            assert WORDS in json.dumps(third.sent), 'the successor holds them'
            assert len(room_frames(third)) == 1 and sum(POINTER in json.dumps(frame) for frame in third.sent) == 1
            assert len(third.spoken) == 1, 'one response to the result and the words'
            assert slow.effects == ['list_available_tools'] and no_private_marker_on_any_wire(live)
    asyncio.run(scenario())


@pytest.mark.parametrize('engine', ['openai', 'grok'])
def test_real_forced_switch_is_not_fooled_by_an_older_transcript_during_new_speech(host_world, monkeypatch, engine):
    from tests._live_stack import stack
    slow = Held(monkeypatch)

    async def scenario():
        async with stack(host_world, monkeypatch, engine) as live:
            answers_only_on(monkeypatch)
            try:
                second = await fresh_during_an_effect(live, slow)
                second.input_start('phrase-one')
                await live.wait(lambda: live.call.input_open, 30)
                stop_without_transcript(second, 'phrase-one')
                await live.wait(lambda: not live.call.input_open, 30)
                second.input_start('phrase-two')
                await live.wait(lambda: live.call.input_open, 30)
                transcript(second, 'phrase-one', OLDER)
                await live.wait(lambda: canonical(live, OLDER) == 1, 30)
                assert live.call.input_open, 'a transcript closes its own words, never the words being said now'
            finally:
                slow.release.set()
            await ready_to_switch(live)
            await asyncio.sleep(.3)
            assert not second.closed, 'the owner is mid-sentence'
            stop_without_transcript(second, 'phrase-two')
            transcript(second, 'phrase-two', WORDS)
            third = await switched(live, second)
            await live.wait(lambda: room_frames(third), 30)
            assert canonical(live, OLDER) == 1 and canonical(live, WORDS) == 1 and not gaps(live)
    asyncio.run(scenario())


@pytest.mark.parametrize('engine', ['openai', 'grok'])
def test_real_failed_recognition_releases_a_forced_switch_with_nothing_invented(host_world, monkeypatch, engine):
    from tests._live_stack import stack
    slow = Held(monkeypatch)

    async def scenario():
        async with stack(host_world, monkeypatch, engine) as live:
            answers_only_on(monkeypatch)
            try:
                second = await fresh_during_an_effect(live, slow)
                second.input_start('noise')
                await live.wait(lambda: live.call.input_open, 30)
                stop_without_transcript(second, 'noise')
            finally:
                slow.release.set()
            await ready_to_switch(live)
            second.put({'type': 'conversation.item.input_audio_transcription.failed', 'item_id': 'noise'})
            third = await switched(live, second)
            await live.wait(lambda: live.call.session is not None and not live.call.session.input_pending, 30)
            await live.wait(lambda: any(POINTER in json.dumps(frame) for frame in third.sent), 30)
            assert not room_frames(third) and not gaps(live), 'no words were recognised, and nothing was lost at a loss'
    asyncio.run(scenario())


@pytest.mark.parametrize('engine', ['openai', 'grok'])
def test_real_ordinary_renewal_still_holds_for_the_transcript(host_world, monkeypatch, engine):
    from tests._live_stack import stack

    async def scenario():
        async with stack(host_world, monkeypatch, engine) as live:
            first = live.wire
            first.input_start('owner-next')
            await live.wait(lambda: live.call.input_open, 30)
            live.host.state['refit'] = True
            await live.wait(lambda: live.call.renewing and live.call.preparation_ready(), 60)
            stop_without_transcript(first, 'owner-next')
            await asyncio.sleep(.3)
            assert not first.closed
            transcript(first, 'owner-next', WORDS)
            if engine == 'grok':
                first.answer()  # xAI VAD owns this response
            await live.wait(lambda: first.closed and live.call.connected.is_set(), 60)
            assert canonical(live, WORDS) == 1 and in_room(live) == 1 and not gaps(live)
    asyncio.run(scenario())


# --- a real loss ---------------------------------------------------------------------------------------------------------


async def lost(live, wire):
    count = len(live.provider.connections)
    await wire.websocket.close(**RESTART)
    await live.wait(lambda: len(live.provider.connections) == count + 1 and live.call.connected.is_set(), 30)
    return live.wire


@pytest.mark.parametrize('engine', ['openai', 'grok'])
def test_real_loss_before_the_transcript_leaves_the_gap_fact_and_no_words(host_world, monkeypatch, engine):
    from tests._live_stack import stack

    async def scenario():
        async with stack(host_world, monkeypatch, engine) as live:
            answers_only_on(monkeypatch)
            first = live.wire
            first.input_start('owner-next')
            await live.wait(lambda: live.call.input_open, 30)
            stop_without_transcript(first, 'owner-next')
            await live.wait(lambda: not live.call.input_open and live.call.session.input_pending, 30)
            second = await lost(live, first)
            await live.wait(lambda: live_frames(second, GAP) or GAP in json.dumps(second.sent), 30)
            assert gaps(live) == 1 and not canonical(live, WORDS) and not in_room(live)
            assert not room_frames(second), 'no recognised words: the gap fact is what is known'
    asyncio.run(scenario())


@pytest.mark.parametrize('engine', ['openai', 'grok'])
def test_real_loss_during_new_speech_after_an_older_transcript_leaves_the_gap_fact(host_world, monkeypatch, engine):
    from tests._live_stack import stack

    async def scenario():
        async with stack(host_world, monkeypatch, engine) as live:
            answers_only_on(monkeypatch)
            first = live.wire
            first.input_start('first-utterance')
            await live.wait(lambda: live.call.input_open, 30)
            stop_without_transcript(first, 'first-utterance')
            await live.wait(lambda: not live.call.input_open, 30)
            first.input_start('second-utterance')
            await live.wait(lambda: live.call.input_open, 30)
            transcript(first, 'first-utterance', OLDER)
            await live.wait(lambda: canonical(live, OLDER) == 1, 30)
            second = await lost(live, first)
            await live.wait(lambda: GAP in json.dumps(second.sent), 30)
            assert gaps(live) == 1 and len(room_frames(second)) == 1, 'what was recognised is owed; the rest may be unheard'
    asyncio.run(scenario())


@pytest.mark.parametrize('engine', ['openai', 'grok'])
def test_real_loss_with_nothing_pending_owes_no_gap(host_world, monkeypatch, engine):
    """The opposite order: the transcript landed and its answer finished before the socket died."""
    from tests._live_stack import stack

    async def scenario():
        async with stack(host_world, monkeypatch, engine) as live:
            first, said = live.wire, spoken(live)
            await live.owner_says(WORDS)
            await live.wait(lambda: spoken(live) == said + 1, 30)
            await spoken_out(live)
            assert not live.call.session.input_pending
            second = await lost(live, first)
            await asyncio.sleep(.3)
            assert not gaps(live) and not room_frames(second) and spoken(live) == said + 1
    asyncio.run(scenario())


@pytest.mark.parametrize('pending,flag,ready', [(None, False, True), (None, True, False), (True, False, False)],
                         ids=['old-provider-idle', 'old-provider-speaking', 'adapter-says-pending'])
def test_a_forced_switch_reads_input_apart_from_renewal_ready(pending, flag, ready):
    """Neither an open call nor any other native work holds it: only the words, and old providers keep their flag."""
    from types import SimpleNamespace
    from skills.live_conversation.driver import _Call
    from tests.test_live_conversation_driver import Host
    call = _Call(Host())
    fields = dict(renewal_ready=False, waiting_tools={'still-open'})
    call.session = SimpleNamespace(**fields, **({} if pending is None else {'input_pending': pending}))
    call.input_open = flag
    assert call.refit_ready() is ready


# --- spoken words a response has not answered ---------------------------------------------------------------------------


@pytest.mark.parametrize('stage', ['before_answer', 'mid_answer'])
@pytest.mark.parametrize('engine,thinking', chosen_rows(ROUTES, 0, 2), ids=chosen_rows(ROUTE_IDS, 0, 2))
def test_real_unanswered_words_are_offered_on_each_fresh_session_until_answered(host_world, monkeypatch, engine, thinking, stage):
    from tests._live_stack import stack

    async def scenario():
        async with stack(host_world, monkeypatch, engine, thinking=thinking) as live:
            answers_only_on(monkeypatch, *((1, 3) if stage == 'mid_answer' else (3,)))
            first = live.wire
            if stage == 'mid_answer':
                live.provider.script.append(('partial', 'Your flight leaves at'))
            first.input_start('q')
            await live.wait(lambda: live.call.input_open, 30)
            first.input_end('q', QUESTION)
            await live.wait(lambda: canonical(live, QUESTION) == 1, 30)
            if stage == 'mid_answer':
                await live.wait(lambda: first.spoken and live.call.active_responses, 30)
            second = await lost(live, first)
            await asked(live, second, True)
            third = await lost(live, second)
            await live.wait(lambda: room_frames(third) and third.spoken, 30)
            assert len(third.spoken) == 1 and len(room_frames(third)) == 1
            await spoken_out(live)
            answers_only_on(monkeypatch)
            fourth = await lost(live, third)
            await asked(live, fourth, False)
            assert canonical(live, QUESTION) == 1 and not any(live_frames(wire, QUESTION) for wire in live.provider.connections)
            assert not fourth.spoken and no_private_marker_on_any_wire(live)
    asyncio.run(scenario())


@pytest.mark.parametrize('engine,thinking', chosen_rows([*ROUTES, ('gemini', False)], 0, 1, 3), ids=chosen_rows([*ROUTE_IDS, 'gemini-ordinary'], 0, 1, 3))
def test_real_words_answered_before_a_drop_stay_silent(host_world, monkeypatch, engine, thinking):
    from tests._live_stack import stack

    async def scenario():
        async with stack(host_world, monkeypatch, engine, thinking=thinking) as live:
            first, said = live.wire, spoken(live)
            await live.owner_says(QUESTION)
            await live.wait(lambda: spoken(live) == said + 1, 30)
            await spoken_out(live)
            second = await lost(live, first)
            await asked(live, second, False)
            assert spoken(live) == said + 1 and not gaps(live)
    asyncio.run(scenario())


@pytest.mark.parametrize('engine', ['openai', 'grok'])
def test_real_transcript_landing_after_the_answer_finds_nothing_owed(host_world, monkeypatch, engine):
    """Recognition is asynchronous on these engines: the answer may finish before its words are known."""
    from tests._live_stack import stack

    async def scenario():
        async with stack(host_world, monkeypatch, engine) as live:
            first, said = live.wire, spoken(live)
            first.input_start('late')
            await live.wait(lambda: live.call.input_open, 30)
            stop_without_transcript(first, 'late')
            if engine == 'grok':
                first.answer()  # xAI VAD owns this response
            await live.wait(lambda: spoken(live) == said + 1 and not live.call.active_responses, 30)
            transcript(first, 'late', QUESTION)
            await live.wait(lambda: canonical(live, QUESTION) == 1, 30)
            await spoken_out(live)
            second = await lost(live, first)
            await asked(live, second, False)
    asyncio.run(scenario())


def test_real_gemini_response_begun_before_a_lagging_transcript_answers_it(host_world, monkeypatch):
    """Transcription order is not guaranteed and no native VAD start came first: the answer is under way when its words appear."""
    from tests._live_stack import stack

    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini') as live:
            answers_only_on(monkeypatch)
            first = live.wire
            first.say('The answer, begun first.', complete=False)
            await live.wait(lambda: live.call.active_responses, 30)
            first.put({'serverContent': {'inputTranscription': {'text': QUESTION}}})
            await live.wait(lambda: live.call.input_open, 30)
            first.complete()
            await live.wait(lambda: canonical(live, QUESTION) == 1 and not live.call.active_responses, 30)
            await spoken_out(live)
            second = await lost(live, first)
            await asked(live, second, False)
    asyncio.run(scenario())


def test_real_gemini_silent_turn_after_the_words_is_a_real_choice(host_world, monkeypatch):
    from tests._live_stack import stack

    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini') as live:
            answers_only_on(monkeypatch)
            first = live.wire
            first.input_start('q')
            await live.wait(lambda: live.call.input_open, 30)
            first.put({'voiceActivity': {'type': 'ACTIVITY_END'}})
            first.put({'serverContent': {'inputTranscription': {'text': QUESTION}}})
            first.complete()  # a usage-only turn end: no speech, and a finished response
            await live.wait(lambda: canonical(live, QUESTION) == 1 and not live.call.response_open, 30)
            second = await lost(live, first)
            await asked(live, second, False)
    asyncio.run(scenario())


@pytest.mark.parametrize('engine,thinking', ROUTES, ids=ROUTE_IDS)
def test_real_barge_in_owes_the_new_words_not_an_answer_to_them_by_the_old_response(host_world, monkeypatch, engine, thinking):
    from tests._live_stack import stack

    async def scenario():
        async with stack(host_world, monkeypatch, engine, thinking=thinking) as live:
            answers_only_on(monkeypatch, 1)
            first = live.wire
            live.provider.script.append(('partial', 'A long answer being spoken'))
            first.input_start('q')
            await live.wait(lambda: live.call.input_open, 30)
            first.input_end('q', QUESTION)
            await live.wait(lambda: first.spoken and live.call.active_responses and canonical(live, QUESTION) == 1, 30)
            answers_only_on(monkeypatch)
            first.input_start('barge')
            await live.wait(lambda: live.call.input_open and not live.call.active_responses, 30)
            first.input_end('barge', WORDS)
            await live.wait(lambda: canonical(live, WORDS) == 1, 30)
            second = await lost(live, first)
            await asked(live, second, True)
    asyncio.run(scenario())


def test_real_partial_words_at_a_gemini_loss_are_labelled_owed_and_answered_once(host_world, monkeypatch):
    """Ordinary Gemini closes words only at turn completion, so a loss before it keeps what was recognised, marked partial,
    with the gap fact beside it: one offer, one response."""
    from tests._live_stack import stack

    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini') as live:
            answers_only_on(monkeypatch, 2)
            first = live.wire
            first.input_start('q')
            first.put({'serverContent': {'inputTranscription': {'text': QUESTION}}})
            await live.wait(lambda: live.call.session.text.get('owner'), 30)
            second = await lost(live, first)
            await live.wait(lambda: room_frames(second) and second.spoken, 30)
            assert gaps(live) == 1 and len(second.spoken) == 1
            assert any(row.get('content') == f'[Partial transcript] {QUESTION}' or QUESTION in str(row.get('content'))
                       for row in live.host.messages)
    asyncio.run(scenario())


@pytest.mark.parametrize('engine', ['openai', 'gemini'])
def test_real_unanswered_words_beside_a_written_result_get_one_answer_and_no_repeat(host_world, monkeypatch, engine):
    from tests._live_stack import stack

    async def scenario():
        async with stack(host_world, monkeypatch, engine, thinking=engine == 'gemini') as live:
            answers_only_on(monkeypatch, 2)
            first = live.wire
            first.tool('kept-call', 'get_task_result', {'task_id': 'none'})
            await live.wait(lambda: native_output(first, 'kept-call') and not live.call.tools_running, 30)
            first.input_start('q')
            await live.wait(lambda: live.call.input_open, 30)
            first.input_end('q', QUESTION)
            await live.wait(lambda: canonical(live, QUESTION) == 1, 30)
            second = await lost(live, first)
            await live.wait(lambda: room_frames(second) and second.spoken, 30)
            await spoken_out(live)
            assert len(second.spoken) == 1 and len(room_frames(second)) == 1
            assert sum(POINTER in json.dumps(frame) for frame in second.sent) == 1 and not native_output(second, 'kept-call')
            answers_only_on(monkeypatch)
            third = await lost(live, second)
            await asked(live, third, False)
            assert not any(POINTER in json.dumps(frame) for frame in third.sent)
            assert sum(row.get('tool_call_id') == 'kept-call' for row in live.host.messages if row.get('role') == 'tool') == 1
    asyncio.run(scenario())


def test_real_native_resume_keeps_recognised_words_owed_with_no_pointer(host_world, monkeypatch):
    from tests._live_stack import stack

    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini', thinking=True) as live:
            answers_only_on(monkeypatch)
            first = live.wire
            first.input_start('q')
            await live.wait(lambda: live.call.input_open, 30)
            first.input_end('q', QUESTION)
            await live.wait(lambda: canonical(live, QUESTION) == 1, 30)
            handle(first, 'after-the-words')
            await live.wait(lambda: live.host.state.get('resume_handle') == 'after-the-words', 30)
            second = await lost(live, first)
            assert live.call.conversation == 1 and second.sent[0]['setup']['sessionResumption'] == {'handle': 'after-the-words'}
            await asyncio.sleep(.3)
            assert not room_frames(second)
            second.put({'sessionResumptionUpdate': {'resumable': False}})
            await live.wait(lambda: not live.host.state.get('resume_handle'), 30)
            third = await fresh_after(live, second)
            assert live.call.conversation == 2
            await asked(live, third, True)
    asyncio.run(scenario())


# --- the fake port: orderings a scenario cannot make a provider produce on demand ---------------------------------------


def said(session, text=QUESTION, **started):
    session.emit('input_started', **started)
    session.emit('caption', speaker='owner', item_id='words', text=text, final=True)
    session.emit('input_done')


@pytest.mark.parametrize('case,owed', [
    ('no-response', True),
    ('begun-after-them-completes', False),
    ('begun-after-them-continues-then-completes', False),
    ('adapter-names-the-response-already-begun', False),
    ('older-words-join-newer-ones-a-later-response-answers', False),
])
def test_what_answers_spoken_words(case, owed):
    async def scenario():
        async with opened(holds=never, fit_failed=False) as world:
            first = world.first
            if case == 'talk-over-completes-during-them':
                first.emit('response_started', response_id='earlier')
                said(first)
                respond_ended(first, 'earlier')
            elif case == 'crossed-the-words-and-was-cut':
                first.emit('input_started')
                first.emit('response_started', response_id='crossed')
                respond_ended(first, 'crossed', status='interrupted')
                first.emit('caption', speaker='owner', item_id='words', text=QUESTION, final=True)
                first.emit('input_done')
            elif case == 'adapter-names-the-response-already-begun':
                first.emit('response_started', response_id='lagged')
                said(first, response_id='lagged')
                respond_ended(first, 'lagged')
            elif case == 'cut-short-with-no-new-speech':
                said(first)
                first.emit('response_started', response_id='cut')
                respond_ended(first, 'cut', status='interrupted')
            elif case == 'barged-in-by-a-sound-never-recognised':
                said(first)
                first.emit('response_started', response_id='cut')
                first.emit('input_started')
                respond_ended(first, 'cut', status='interrupted')
                first.emit('input_done')
            else:
                said(first)
                if case == 'begun-after-them-failed':
                    first.emit('response_started', response_id='answer')
                    respond_ended(first, 'answer', status='failed')
                elif case == 'begun-after-them-still-continuing':
                    first.emit('response_started', response_id='answer')
                    respond_ended(first, 'answer', continues=True)
                elif case == 'begun-after-them-completes':
                    await answer(first, 'answer')
                elif case == 'begun-after-them-continues-then-completes':
                    first.emit('response_started', response_id='answer')
                    respond_ended(first, 'answer', continues=True)
                    respond_ended(first, 'answer')
                elif case == 'older-words-join-newer-ones-a-later-response-answers':
                    said(first, 'Newer words.')
                    await answer(first, 'answer')
            await first.handled()
            second = await successor(world, first)
            await asyncio.sleep(.05)
            assert room_pointers(second) == int(owed)
    asyncio.run(scenario())


def test_an_old_provider_caption_without_an_input_event_owes_nothing():
    async def scenario():
        async with opened(holds=never, fit_failed=False) as world:
            first = world.first
            first.emit('caption', speaker='owner', item_id='words', text=QUESTION, final=True)
            await first.handled()
            second = await successor(world, first)
            await asyncio.sleep(.05)
            assert not room_pointers(second)
    asyncio.run(scenario())


def test_unrecognised_words_stay_the_gap_facts():
    async def scenario():
        async with opened(holds=never, fit_failed=False) as world:
            first = world.first
            first.emit('input_started')
            await first.handled()
            second = await successor(world, first)
            await asyncio.sleep(.05)
            assert not room_pointers(second) and GAP in str(second.sent)
    asyncio.run(scenario())


def test_spoken_and_typed_words_share_one_pointer_and_one_answer():
    async def scenario():
        async with opened(holds=never, fit_failed=False) as world:
            box = mailbox_of(world)
            first = world.first
            said(first)
            box.append(room_mail())
            await asyncio.wait_for(_written(first), 2)
            second = await successor(world, first)
            await _pointed(second)
            await answer(second, 'answers-both')
            third = await successor(world, second)
            await asyncio.sleep(.05)
            assert room_pointers(second) == 1 and not room_pointers(third) and not mail_updates(third)
    asyncio.run(scenario())


async def _written(session):
    while not mail_updates(session):
        await asyncio.sleep(.005)


async def _pointed(session):
    while not room_pointers(session):
        await asyncio.sleep(.005)


@pytest.mark.parametrize('answered', [False, True], ids=['unanswered', 'answered-there'])
def test_a_native_resume_carries_spoken_words_owed(answered):
    async def scenario():
        async with opened(holds=never, fit_failed=False) as world:
            first = world.first
            said(first)
            await first.handled()
            second = await resumed(world, first)
            await asyncio.sleep(.05)
            assert not room_pointers(second) and world.call.conversation == 1
            if answered:
                await answer(second, 'answer')
            third = await successor(world, second)
            await asyncio.sleep(.05)
            assert room_pointers(third) == int(not answered)
    asyncio.run(scenario())
