"""A Gemini start the server accepts and then refuses ends as a refusal, not as a lost connection.

Recorded 27.11.2026 (spoken probe m08, a setup above the model's input limit): setupComplete, a resumption handle,
the owner's first audio, then the server closed the socket with 1007 "Request contains an invalid argument." at
the first generation, with no error event and no usage. A 1007 before setupComplete or after a generation began keeps
its earlier meaning. 1007 is Google's generic invalid-argument code: nothing here is a window measurement.
"""
from __future__ import annotations

import asyncio
import json

import pytest
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close

from ouroboros.contracts.live_conversation import Audio
from ouroboros.contracts.realtime import RealtimeSetupError
from tests._live_world import host_world as host_world
from tests.test_gemini_live_boundaries import deny_network  # noqa: F401
from tests.test_gemini_live_boundaries import audio_content, gemini as gemini_fixture, make_session, receive

gemini = gemini_fixture
REFUSED = "Request contains an invalid argument."


def refused():
    return ConnectionClosedError(Close(1007, REFUSED), None)


@pytest.mark.parametrize("model", ["gemini-3.8-live", "gemini-3.8-live-extended-thinking"])
def test_a_close_1007_after_setup_and_before_any_generation_is_the_engines_refusal(gemini, model):
    async def run():
        session, wire = make_session(gemini, model=model)
        observed = []
        session.observe = lambda direction, raw: observed.append((direction, raw))
        try:
            await session.open()
            await session.receive()
            await receive(session, wire, {"sessionResumptionUpdate": {"newHandle": "m08", "resumable": True}})
            await session.send(Audio(b"\x00\x00" * 160, 16000))  # the owner's first words were streamed
            with pytest.raises(RealtimeSetupError) as raised:
                await receive(session, wire, refused())
            assert raised.value.code == "input_refused"
            assert "1007 " + REFUSED in str(raised.value)  # the exact code and reason, for Details
            assert not session.resume_requested  # nothing to resume: no retry
            assert not [raw for direction, raw in observed  # no usage sample, no provider-window fact
                        if direction == 'usage' or 'call.provider_window' in raw]
        finally:
            await session.close()
    asyncio.run(run())


def test_a_close_1007_after_a_generation_began_keeps_its_meaning(gemini):
    async def run():
        session, wire = make_session(gemini)
        try:
            await session.open()
            await session.receive()
            await receive(session, wire, audio_content("Hello", finished=False))
            events = await receive(session, wire, refused())
            closed = next(event.data for event in events if event.kind == "closed")
            assert closed == {"reason": "transport_lost", "resumable": False, "transport_loss": False,
                              "detail": "1007 " + REFUSED}
        finally:
            await session.close()
    asyncio.run(run())


def test_a_close_1007_before_setup_complete_keeps_its_meaning(gemini):
    async def run():
        session, wire = make_session(gemini)
        wire.put(refused())  # read before the setupComplete the fake wire answers to the setup
        try:
            with pytest.raises(ConnectionClosedError):
                await session.open()
        finally:
            await session.close()
    asyncio.run(run())


def test_the_call_ends_on_the_refusal_without_a_reconnection(host_world, gemini):
    from skills.live_conversation.driver import _Call

    async def run():
        session, wire = make_session(gemini)
        try:
            await session.open()
            await session.receive()  # the ready event
            call = _Call(host_world.host)
            call.session = session
            call.connected.set()
            wire.put(refused())
            with pytest.raises(RealtimeSetupError) as raised:
                await asyncio.wait_for(call.provider(), 5)
            assert raised.value.code == "input_refused"
            assert not call.renewing and not call.provider_closed  # ended, not reconnecting
        finally:
            await session.close()
    asyncio.run(run())


@pytest.mark.parametrize("answered", [False, True])
def test_the_refused_input_ends_as_one_calm_notice_with_the_close_behind_details(host_world, monkeypatch, answered):
    from ouroboros import call_peers
    host = host_world.host
    endings = []
    monkeypatch.setattr(call_peers, "get_hub", lambda _: host.hub)
    monkeypatch.setattr(call_peers, "close_hub", lambda *a, **kw: endings.append((a, kw)))

    async def refusing_engine(port):
        if answered:
            port.completed.add("an-earlier-answer")  # the running call's host settled one answer before
        raise RealtimeSetupError("input_refused", "1007 " + REFUSED)
    monkeypatch.setitem(host.turn.feature, "handler", refusing_engine)
    sentence, metadata, trace = host.turn.run(host.ctx, host.messages, host.tools, task=host.task, initial_effort="")
    notice = metadata["host_notice_row"]["notice"]
    assert metadata["reason_code"] == "input_refused" and metadata["execution_status"] == "infra_failed"
    assert sentence.startswith("The call ended on" if answered else "The call could not start on")
    assert "refused its input" in sentence and "1007" not in sentence and REFUSED not in sentence
    assert ("Its transcript is in this room." in sentence) is answered
    assert "1007 Request contains an invalid argument" in notice["details"] and "no window" in notice["details"]
    assert notice["code"] == "input_refused" and notice["step"] == "open_calls"
    assert endings[0][1]["detail"] == metadata["host_notice_row"]["text"] and endings[0][1]["action"] == "settings"
    assert "fit_refusal" not in trace and "failure" not in metadata
    assert not any(json.loads(json.loads(line)["raw"]).get("type") == "call.fit_refused"
                   for line in host.wire_path.read_text(encoding="utf-8").splitlines())
