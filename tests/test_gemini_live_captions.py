"""Replay the R3 thinking greeting on fake sockets, including its native boundary."""
from __future__ import annotations

import asyncio
import json

import pytest

from ouroboros.contracts.live_conversation import LiveOptions
from tests.test_gemini_live_boundaries import deny_network  # noqa: F401
from tests.test_gemini_live_boundaries import (
    gemini as gemini_fixture, audio_content, make_session, receive,
)

gemini = gemini_fixture
THINKING = "gemini-3.8-live-extended-thinking"
ORDINARY = "gemini-3.8-live"
GREETING = "Привет! Слышу тебя отлично. Это проверочный звонок — чем могу помочь?"

# Recorded received serverContent/usage frames through native IDLE, plus the
# next response's first audio/caption. Source: probe/r3/evidence/thinking/
# wire-90d3d7bd.jsonl (line numbers retained). Only PCM is replaced with synthetic
# silence; no setup, credentials, outbound context or captured audio is retained.
REPLAY = json.loads(r'''
[
[56,{"serverContent":{}}],
[65,{"serverContent":{"generationComplete":true}}],
[66,{"serverContent":{"turnComplete":true,"interactionStatus":"IN_PROGRESS"},"usageMetadata":{"promptTokenCount":39001,"totalTokenCount":39115,"promptTokensDetails":[{"modality":"TEXT","tokenCount":38668},{"modality":"AUDIO","tokenCount":222}],"thoughtsTokenCount":114}}],
[363,{"serverContent":{}}],
[377,{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AAA="}}],"role":"model"},"outputTranscription":{"text":"Привет! Слышу "}}}],
[391,{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AAA="}}],"role":"model"}}}],
[404,{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AAA="}}],"role":"model"}}}],
[416,{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AAA="}}],"role":"model"}}}],
[429,{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AAA="}}],"role":"model"}}}],
[442,{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AAA="}}],"role":"model"}}}],
[456,{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AAA="}}],"role":"model"},"outputTranscription":{"text":"тебя отлично."}}}],
[468,{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AAA="}}],"role":"model"}}}],
[479,{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AAA="}}],"role":"model"}}}],
[492,{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AAA="}}],"role":"model"}}}],
[503,{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AAA="}}],"role":"model"}}}],
[514,{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AAA="}}],"role":"model"}}}],
[527,{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AAA="}}],"role":"model"},"outputTranscription":{"text":" Это проверочный"}}}],
[540,{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AAA="}}],"role":"model"}}}],
[554,{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AAA="}}],"role":"model"}}}],
[563,{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AAA="}}],"role":"model"}}}],
[574,{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AAA="}}],"role":"model"},"outputTranscription":{"text":" звонок — чем"}}}],
[586,{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AAA="}}],"role":"model"}}}],
[599,{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AAA="}}],"role":"model"},"outputTranscription":{"text":" могу помочь?"}}}],
[609,{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AAA="}}],"role":"model"}}}],
[620,{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AAA="}}],"role":"model"}}}],
[626,{"serverContent":{"generationComplete":true}}],
[631,{"serverContent":{"turnComplete":true,"interactionStatus":"IDLE"},"usageMetadata":{"promptTokenCount":113188,"responseTokenCount":171,"totalTokenCount":125729,"cachedContentTokenCount":70662,"promptTokensDetails":[{"modality":"TEXT","tokenCount":112837},{"modality":"AUDIO","tokenCount":222}],"responseTokensDetails":[{"modality":"TEXT","tokenCount":43},{"modality":"AUDIO","tokenCount":128}],"thoughtsTokenCount":12370}}],
[5767,{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AAA="}}],"role":"model"},"outputTranscription":{"text":"Кодовое слово "}}}]
]
''')


def finals(events):
    return [e.data for e in events if e.kind == "caption" and e.data["final"]]


async def replay_greeting(session, wire):
    events = []
    for line, event in REPLAY:
        if line <= 631:
            events.extend(await receive(session, wire, event))
    return events


@pytest.mark.parametrize("model", [THINKING, ORDINARY])
def test_recorded_greeting_finalizes_at_native_idle_on_both_routes(gemini, model):
    async def run():
        session, wire = make_session(gemini, model=model)
        try:
            await session.open()
            await session.receive()
            emitted, audio_item = [], None
            for line, event in REPLAY:
                if line > 631:
                    break
                events = await receive(session, wire, event)
                emitted.extend(events)
                for item in events:
                    if item.kind == "audio":
                        audio_item = audio_item or item.data["item_id"]
                        assert item.data["item_id"] == audio_item
                if line < 631:
                    assert not finals(emitted), f"Premature final at wire line {line}"
                if line == 626:
                    assert not session.renewal_ready  # Generation alone has no final transcript.
            expected = dict(speaker="ouroboros", item_id=audio_item, text=GREETING, final=True)
            assert finals(emitted) == [expected]
            assert session.renewal_ready
            repeated = await receive(session, wire, {"serverContent": {
                "turnComplete": True, "interactionStatus": "IDLE"}})
            assert not finals(repeated)
            assert session.final_captions() == []  # Stop must not repeat a completed row.
            next_events = await receive(session, wire, REPLAY[-1][1])
            assert finals(next_events) == []
            next_item = next(e.data["item_id"] for e in next_events if e.kind == "audio")
            assert next_item != audio_item
            assert not session.renewal_ready
        finally:
            await session.close()
    asyncio.run(run())


@pytest.mark.parametrize("resumed", [False, True])
def test_late_transcription_corrects_same_completed_caption_once(gemini, resumed):
    async def run():
        session, wire = make_session(gemini, model=THINKING)
        try:
            await session.open()
            await session.receive()
            events = await replay_greeting(session, wire)
            original = finals(events)
            assert len(original) == 1, "The recorded IDLE must finish the greeting"
            item = original[0]["item_id"]
            if resumed:
                await receive(session, wire, {"sessionResumptionUpdate": {
                    "newHandle": "test-resume-handle", "resumable": True}})
                await receive(session, wire, {"goAway": {"timeLeft": "1s"}})
                await session.close()
                session, wire = make_session(gemini, LiveOptions(resume_handle="test-resume-handle"), model=THINKING)
                await session.open()
                await session.receive()
            correction = {"serverContent": {"outputTranscription": {
                "text": " До связи.", "finished": True, "startOffset": "8s", "endOffset": "9s"}}}
            events = await receive(session, wire, correction)
            assert [e.data for e in events if e.kind == "caption"] == [dict(
                speaker="ouroboros", item_id=item, text=GREETING + " До связи.", final=True, replace=True)]
            assert session.renewal_ready
            # A repeated identified correction, empty finished marker, IDLE and
            # Stop must not duplicate the revised final or lose its prefix.
            for message in (correction, {"serverContent": {"outputTranscription": {"finished": True}}},
                            {"serverContent": {"turnComplete": True, "interactionStatus": "IDLE"}}):
                assert not finals(await receive(session, wire, message))
            assert session.final_captions() == []
            next_events = await receive(session, wire, REPLAY[-1][1])
            next_item = next(e.data["item_id"] for e in next_events if e.kind == "audio")
            assert next_item != item and not finals(next_events)
            next_end = await receive(session, wire, {"serverContent": {
                "turnComplete": True, "interactionStatus": "IDLE"}})
            assert finals(next_end) == [dict(speaker="ouroboros", item_id=next_item,
                                             text="Кодовое слово ", final=True)]
        finally:
            await session.close()
    asyncio.run(run())


@pytest.mark.parametrize("model", [THINKING, ORDINARY])
def test_generation_requires_finished_transcription_and_in_progress_holds_renewal(gemini, model):
    async def run():
        session, wire = make_session(gemini, model=model)
        try:
            await session.open()
            await session.receive()
            events = await receive(session, wire, audio_content("Still ", finished=False))
            item = next(e.data["item_id"] for e in events if e.kind == "audio")
            for content in ({"generationComplete": True}, {"turnComplete": True, "interactionStatus": "IN_PROGRESS"}):
                assert not finals(await receive(session, wire, {"serverContent": content}))
                assert not session.renewal_ready
            events = await receive(session, wire, {"serverContent": {
                "generationComplete": True, "outputTranscription": {"text": "speaking.", "finished": True}}})
            assert finals(events) == [dict(speaker="ouroboros", item_id=item, text="Still speaking.", final=True)]
            assert not session.renewal_ready  # Caption completion cannot end ongoing reasoning.
            events = await receive(session, wire, {"serverContent": {"turnComplete": True, "interactionStatus": "IDLE"}})
            assert not finals(events)
            assert session.renewal_ready
            assert session.final_captions() == []
        finally:
            await session.close()
    asyncio.run(run())


def test_idle_before_transcription_keeps_missing_caption_pending(gemini):
    async def run():
        session, wire = make_session(gemini, model=THINKING)
        try:
            await session.open()
            await session.receive()
            event = audio_content()
            del event["serverContent"]["outputTranscription"]
            events = await receive(session, wire, event)
            item = next(e.data["item_id"] for e in events if e.kind == "audio")
            assert not finals(await receive(session, wire, {"serverContent": {
                "turnComplete": True, "interactionStatus": "IDLE"}}))
            assert not session.renewal_ready  # IDLE cannot invent the missing words.
            events = await receive(session, wire, {"serverContent": {"outputTranscription": {"text": "Late words"}}})
            assert finals(events) == [dict(speaker="ouroboros", item_id=item, text="Late words", final=True)]
            assert session.renewal_ready
            assert session.final_captions() == []
        finally:
            await session.close()
    asyncio.run(run())
