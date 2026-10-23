"""Resumption handles, replay identities and teardown on synthetic sockets."""
from __future__ import annotations

import asyncio
import pytest

from ouroboros.contracts.live_conversation import ContextUpdate, LiveOptions, ToolOutputs
from ouroboros.contracts.realtime import RealtimeSetupError
from tests.test_gemini_live_boundaries import deny_network  # noqa: F401
from tests.test_gemini_live_boundaries import gemini as gemini_fixture, audio_content, make_session, receive

gemini = gemini_fixture


def checkpoint(handle="test-resume-handle", resumable=True):
    return {"sessionResumptionUpdate": {"newHandle": handle, "resumable": resumable}}


def test_handle_updates_goaway_and_resumed_setup(gemini):
    async def run():
        session, wire = make_session(gemini)
        try:
            await session.open()
            await session.receive()
            assert wire.sent[0]["setup"]["sessionResumption"] == {}
            for handle, valid in [("first", True), ("", False), ("test-resume-handle", True)]:
                events = await receive(session, wire, checkpoint(handle, valid))
                assert next(e.data for e in events if e.kind == "resumable") == {"handle": handle}
            events = await receive(session, wire, {"goAway": {"timeLeft": {"seconds": 2, "nanos": 500000000}}})
            assert next(e.data for e in events if e.kind == "renew") == {
                "within_s": 2.5, "reason": "go_away", "resume": True}
        finally:
            await session.close()
        resumed, next_wire = make_session(gemini, LiveOptions(resume_handle="test-resume-handle"))
        try:
            await resumed.open()
            assert next_wire.sent[0]["setup"]["sessionResumption"] == {"handle": "test-resume-handle"}
        finally:
            await resumed.close()
    asyncio.run(run())


def test_transport_loss_reports_resumability_and_keeps_unsettled_usage(gemini):
    async def run():
        for valid in (True, False):
            session, wire = make_session(gemini)
            try:
                await session.open()
                await session.receive()
                await receive(session, wire, checkpoint("resume" if valid else "", valid))
                await receive(session, wire, audio_content("Partial", finished=False))
                await receive(session, wire, {"usageMetadata": {"responseTokensDetails": [
                    {"modality": "AUDIO", "tokenCount": 12}]}})
                events = await receive(session, wire, ConnectionError("socket went away"))
                assert next(e.data for e in events if e.kind == "closed") == {
                    "reason": "transport_lost", "resumable": valid}
                if valid:
                    assert not any(e.kind == "response_done" for e in events)  # One response spans the reconnect.
                    assert session.usage["responseTokensDetails"][0]["tokenCount"] == 12
                else:
                    done = next(e.data for e in events if e.kind == "response_done")
                    assert done["status"] == "incomplete"
                    assert done["usage"] == {"audio_output_tokens": 12}
            finally:
                await session.close()
            assert wire.closed
            assert session.read_task is None
    asyncio.run(run())


def test_resumed_replay_keeps_captions_tools_and_usage_once(gemini):
    async def run():
        session, wire = make_session(gemini)
        call = {"toolCall": {"functionCalls": [{"id": "tool-1", "name": "read_file", "args": {"path": "fake"}}]}}
        spoken = {"responseId": "response-1", **audio_content("Remember this.")}
        caption = {"serverContent": {"inputTranscription": {
            "text": "My fact.", "finished": True, "startOffset": "1s", "endOffset": "2s"}}}
        ended = {"responseId": "response-1", "serverContent": {"turnComplete": True},
                 "usageMetadata": {"responseTokensDetails": [{"modality": "AUDIO", "tokenCount": 4}]}}
        try:
            await session.open()
            await session.receive()
            await receive(session, wire, caption)
            await receive(session, wire, spoken)
            events = await receive(session, wire, call)
            assert any(e.kind == "tool_call" for e in events)
            await session.send(ToolOutputs(({"role": "tool", "tool_call_id": "tool-1", "content": "Done"},), ()))
            events = await receive(session, wire, ended)
            response_id = next(e.data["response_id"] for e in events if e.kind == "response_done")
            await receive(session, wire, checkpoint())
            await receive(session, wire, {"goAway": {"timeLeft": "1s"}})
        finally:
            await session.close()
        resumed, next_wire = make_session(gemini, LiveOptions(resume_handle="test-resume-handle"))
        try:
            await resumed.open()
            await resumed.receive()
            for event in (caption, spoken, call, ended):
                events = await receive(resumed, next_wire, event)
                assert not any(e.kind in {"caption", "audio", "tool_call", "response_started", "response_done"} for e in events)
            # A repeated call may still owe a native response; the host effect is not run again.
            assert next_wire.sent[-1]["toolResponse"]["functionResponses"][0]["id"] == "tool-1"
            # Identical words with a new native identity are a real new utterance.
            fresh_caption = {"serverContent": {"inputTranscription": {
                **caption["serverContent"]["inputTranscription"], "startOffset": "3s", "endOffset": "4s"}}}
            assert any(e.kind == "caption" for e in await receive(resumed, next_wire, fresh_caption))
            assert any(e.kind == "audio" for e in await receive(resumed, next_wire, {**spoken, "responseId": "response-2"}))
            events = await receive(resumed, next_wire, {**ended, "responseId": "response-2"})
            assert next(e.data["response_id"] for e in events if e.kind == "response_done") != response_id
        finally:
            await resumed.close()
    asyncio.run(run())


def test_held_context_survives_a_resumed_disconnect_once(gemini):
    async def run():
        session, wire = make_session(gemini)
        try:
            await session.open()
            await session.receive()
            await receive(session, wire, checkpoint())
            await receive(session, wire, audio_content("First ", finished=False))
            await session.send(ContextUpdate(({"role": "user", "content": "Held context"},)))
            assert len(wire.sent) == 1
            await receive(session, wire, ConnectionError("lost"))
        finally:
            await session.close()
        resumed, next_wire = make_session(gemini, LiveOptions(resume_handle="test-resume-handle"))
        try:
            await resumed.open()
            await resumed.receive()
            events = await receive(resumed, next_wire, {"serverContent": {
                "outputTranscription": {"text": "ending", "finished": True}, "turnComplete": True}})
            caption = next(e.data for e in events if e.kind == "caption" and e.data["final"])
            assert caption["text"] == "First ending"
            assert "Held context" in next_wire.sent[-1]["clientContent"]["turns"][0]["parts"][0]["text"]
            await receive(resumed, next_wire, {"serverContent": {"turnComplete": True}})
            assert len([e for e in next_wire.sent if "clientContent" in e]) == 1
        finally:
            await resumed.close()
    asyncio.run(run())


def test_provider_errors_are_not_transport_retries(gemini):
    async def run():
        session, wire = make_session(gemini)
        try:
            await session.open()
            await session.receive()
            await receive(session, wire, checkpoint())
            with pytest.raises(RealtimeSetupError, match="provider_error"):
                await receive(session, wire, {"error": {"message": "fake rejection"}})
        finally:
            await session.close()
    asyncio.run(run())


def test_tool_identity_conflicts_refuse_but_distinct_calls_still_work(gemini):
    async def run():
        session, wire = make_session(gemini)
        try:
            await session.open()
            await session.receive()
            call = {"id": "same-id", "name": "read_file", "args": {"path": "first"}}
            events = await receive(session, wire, {"toolCall": {"functionCalls": [call, call]}})
            assert len([e for e in events if e.kind == 'tool_call']) == 1
            assert await receive(session, wire, {"toolCall": {"functionCalls": [call]}}) == []
            with pytest.raises(RealtimeSetupError, match="tool_identity"):
                await receive(session, wire, {"toolCall": {"functionCalls": [{**call, "args": {"path": "different"}}]}})
            events = await receive(session, wire, {"toolCall": {"functionCalls": [{**call, "id": "new-id"}]}})
            assert any(e.kind == "tool_call" for e in events)
        finally:
            await session.close()
    asyncio.run(run())


def test_handle_before_setup_ack_is_not_lost(gemini):
    async def run():
        session, wire = make_session(gemini)
        try:
            wire.put(checkpoint())
            await session.open()
            events = await session.receive()
            assert events[0].kind == "ready"
            assert next(e.data["handle"] for e in events if e.kind == "resumable") == "test-resume-handle"
        finally:
            await session.close()
    asyncio.run(run())


def test_failed_reopen_keeps_state_for_another_attempt(gemini):
    async def run():
        session, wire = make_session(gemini)
        try:
            await session.open()
            await session.receive()
            await receive(session, wire, checkpoint())
            await receive(session, wire, audio_content("Keep ", finished=False))
            await receive(session, wire, ConnectionError("lost"))
        finally:
            await session.close()
        failed, broken_wire = make_session(gemini, LiveOptions(resume_handle="test-resume-handle"))
        try:
            broken_wire.put(ConnectionError("open failed"))
            with pytest.raises(ConnectionError):
                await failed.open()
        finally:
            await failed.close()
        resumed, next_wire = make_session(gemini, LiveOptions(resume_handle="test-resume-handle"))
        try:
            await resumed.open()
            await resumed.receive()
            events = await receive(resumed, next_wire, {"serverContent": {
                "outputTranscription": {"text": "me", "finished": True}, "turnComplete": True}})
            assert next(e.data["text"] for e in events if e.kind == "caption" and e.data["final"]) == "Keep me"
        finally:
            await resumed.close()
    asyncio.run(run())
